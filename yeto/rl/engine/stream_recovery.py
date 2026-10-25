"""Peer recovery of a full-parameter role stream over a sharded trainer (S27 G-ED).

Engine-neutral. :mod:`yeto.rl.peer_recovery` is the exchange with the syncer
and the snapshot store; :class:`~yeto.rl.engine.full_parameter_stream.
FullParameterRoleStream` drives it (``_initialize_recovering`` on a fresh
trainer, ``serve_peer_snapshots`` at the donor's safe point). This module is
the part between the two that the Miles and the verl islands share:

* :class:`PeerRecoveryOptions` and :func:`peer_recovery_options`: the island
  options (``--rl-peer-recovery-*``, and :data:`PEER_RECOVER_ENV`, which the
  launcher sets on a relaunched island, ``yeto.launch.peer_recovery``);
* :class:`RankShardSnapshot`: a trainer whose ranks each write one shard
  file. It writes the rank list (:data:`RANKS`: every shard's file, sha256,
  size, rank coordinate and the rank's state / RNG / LR-scheduler digests),
  names the trainer layout (the layout tag: only a donor with the same tag can
  serve), and after a restore checks that every rank re-exports exactly the
  donor's digests (else the restore fails and the island must not train). A
  backend gives three verbs: the rank layout, save every rank's shard, restore
  every rank's shard (Miles: ``cut_plugin``; verl: the FSDP2 rank files of
  :mod:`yeto.rl.adapters.verl.cut`);
* :class:`StreamRecoveryIsland`: the
  :class:`~yeto.rl.peer_recovery.FileRecoverableIsland` of one role stream
  over such a trainer.

Replay. After the restore the island reports "nothing held", so the syncer
resends the current value of every fragment, and the stream applies the
resent fragments as its anchors in one apply. The island therefore takes the
donor's inner optimizer state (moments, step, LR schedule) and RNG; its
parameters are the syncer's current fragments. The donor's local progress
since its last push is not copied: the donor still pushes it itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.protocol import BcastFragment
from yeto.rl.peer_recovery import (
    SnapshotIntegrityError,
    SnapshotRefused,
    check_restored_digests,
    digest_fingerprint,
)

RANKS = "ranks.json"
# "1": the launcher relaunched this island (yeto.launch.peer_recovery.relaunch_envs)
PEER_RECOVER_ENV = "YETO_RL_PEER_RECOVER"
# Per-rank digests of a shard save and of its restore: the restored trainer
# must re-export the donor's state, RNG and LR scheduler position exactly.
DIGEST_KEYS = ("state_digest", "rng_digest", "scheduler_samples")
DEFAULT_TTL_S = 3600.0
LOG_PREFIX = "[yeto-peer-recovery] "


@dataclass(frozen=True)
class PeerRecoveryOptions:
    """Island options of peer recovery (from ``--rl-peer-recovery-*``).

    ``store``: checkpoint store path of the snapshots (resolved with
    :func:`yeto.rl.peer_recovery.snapshot_store_for`). ``recover``: this
    process lost its trainer state and asks the syncer for a donor.
    ``staging``: local directory for object-store transfers."""

    store: str
    recover: bool = False
    ttl_s: float = DEFAULT_TTL_S
    timeout_s: float = DEFAULT_TTL_S
    staging: str | None = None

    def __post_init__(self) -> None:
        if not self.store:
            raise ValueError("peer recovery needs a snapshot store")
        if not self.ttl_s > 0 or not self.timeout_s > 0:
            raise ValueError("peer recovery ttl and timeout must be positive")

    def to_plan(self) -> dict[str, Any]:
        """JSON form (an island plan written before the trainer process starts)."""
        return {"store": self.store, "recover": self.recover, "ttl_s": self.ttl_s,
                "timeout_s": self.timeout_s, "staging": self.staging}

    @classmethod
    def from_plan(cls, raw: Mapping[str, Any] | None) -> "PeerRecoveryOptions | None":
        if not raw:
            return None
        return cls(store=str(raw["store"]), recover=bool(raw.get("recover", False)),
                   ttl_s=float(raw.get("ttl_s", DEFAULT_TTL_S)),
                   timeout_s=float(raw.get("timeout_s", raw.get("ttl_s", DEFAULT_TTL_S))),
                   staging=raw.get("staging"))


def peer_recovery_options(args: Any, environ: Mapping[str, str] | None = None) -> PeerRecoveryOptions | None:
    """Peer recovery of a full-parameter island, or None without
    ``--rl-peer-recovery-store``. The island recovers when ``--rl-peer-recover``
    is given or the launcher relaunched it (:data:`PEER_RECOVER_ENV`)."""
    store = getattr(args, "rl_peer_recovery_store", None)
    env = os.environ if environ is None else environ
    recover = bool(getattr(args, "rl_peer_recover", False)) or env.get(PEER_RECOVER_ENV) == "1"
    if store is None:
        if recover or getattr(args, "rl_peer_recovery_staging", None):
            raise ValueError("--rl-peer-recover / --rl-peer-recovery-staging need --rl-peer-recovery-store")
        return None
    ttl = getattr(args, "rl_peer_recovery_ttl_s", None)
    ttl_s = DEFAULT_TTL_S if ttl is None else float(ttl)
    return PeerRecoveryOptions(
        store=str(store),
        recover=recover,
        ttl_s=ttl_s,
        # A recovery may wait as long as its snapshot lives (a large trainer's
        # snapshot write, commit, read and the full resend take tens of minutes).
        timeout_s=ttl_s,
        staging=getattr(args, "rl_peer_recovery_staging", None),
    )


def snapshot_layout_tag(prefix: str, schema: str, parameter_layout_hash: str,
                        layout: Mapping[str, int]) -> str:
    """``<prefix>sha256`` of the fragment layout hash and the trainer layout."""
    body = {"schema": schema, "parameter_layout_hash": parameter_layout_hash,
            "layout": {str(key): int(value) for key, value in layout.items()}}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return prefix + hashlib.sha256(encoded).hexdigest()


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


class RankShardSnapshot:
    """Writes and restores the per-rank shards of one trainer (module doc).

    Subclasses set :attr:`schema` and :attr:`tag_prefix` and give the three
    backend verbs (:meth:`rank_layout`, :meth:`save_ranks`,
    :meth:`restore_ranks`). A rank result is a mapping with ``path`` (relative
    to the snapshot directory), ``sha256``, ``bytes``, ``coord`` and, when the
    backend reports them, the :data:`DIGEST_KEYS`."""

    schema: str = ""
    tag_prefix: str = ""
    digest_keys: tuple[str, ...] = DIGEST_KEYS

    def __init__(self, parameter_layout_hash: str, *, log: Callable[[str], Any] = _stderr) -> None:
        self.parameter_layout_hash = parameter_layout_hash
        self.log = log
        self._tag: str | None = None
        # The LR scheduler position every rank reports after a restore (the
        # donor's, in the backend's ``scheduler_samples`` unit); None before.
        self.restored_scheduler_samples: int | None = None

    # -- backend verbs ----------------------------------------------------------
    async def rank_layout(self) -> Mapping[str, int]:
        """The trainer's parallel layout (world size and parallel dimensions)."""
        raise NotImplementedError

    async def save_ranks(self, directory: Path) -> list[Mapping[str, Any]]:
        """Every rank writes its shard into ``directory``; raise
        :class:`~yeto.rl.peer_recovery.SnapshotRefused` when one cannot."""
        raise NotImplementedError

    async def restore_ranks(self, directory: Path, files: list[dict[str, Any]]) -> list[Mapping[str, Any]]:
        """Every rank loads its shard into the fresh trainer; raise when one
        cannot (the trainer may be partly written: it must not train)."""
        raise NotImplementedError

    # -- neutral part ----------------------------------------------------------
    async def layout_tag(self) -> str:
        if self._tag is None:
            self._tag = snapshot_layout_tag(self.tag_prefix, self.schema, self.parameter_layout_hash,
                                            await self.rank_layout())
        return self._tag

    async def save(self, directory: Path) -> None:
        """Every rank writes its shard; refuses unless every shard is here."""
        directory = Path(directory)
        results = [dict(r) for r in await self.save_ranks(directory)]
        files = [{"path": r["path"], "sha256": r["sha256"], "bytes": int(r["bytes"]), "coord": r["coord"],
                  **{k: r[k] for k in self.digest_keys if k in r}} for r in results]
        missing = [f["path"] for f in files if not (directory / f["path"]).is_file()]
        if missing:
            raise SnapshotRefused(
                f"rank shards {missing} are not in the snapshot directory {directory} (a rank on "
                "another node needs a node-shared --rl-peer-recovery-staging or snapshot store)")
        ranks = {"schema": self.schema, "layout_tag": await self.layout_tag(), "files": files}
        (directory / RANKS).write_text(json.dumps(ranks, sort_keys=True), encoding="utf-8")
        if all(k in f for f in files for k in self.digest_keys):
            self.log(f"{LOG_PREFIX}donor snapshot: {len(files)} rank shard(s), "
                     f"{sum(f['bytes'] for f in files)} bytes, "
                     f"digests {digest_fingerprint(files, self.digest_keys)}")

    async def restore(self, directory: Path) -> str | None:
        """Load every rank's shard on this fresh trainer; returns the digest
        fingerprint (None when the donor wrote no digests)."""
        directory = Path(directory)
        try:
            ranks = json.loads((directory / RANKS).read_text(encoding="utf-8"))
            files = list(ranks["files"])
            tag = ranks["layout_tag"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SnapshotIntegrityError(f"peer snapshot has no valid {RANKS}: {exc}") from exc
        if ranks.get("schema") != self.schema or tag != await self.layout_tag():
            raise SnapshotIntegrityError("peer snapshot rank list is from another trainer layout")
        results = [dict(r) for r in await self.restore_ranks(directory, files)]
        # Every rank must re-export the donor's digests (as a cut restore
        # checks); else the trainer is not the donor's and must not train.
        fingerprint = check_restored_digests(files, results, self.digest_keys)
        positions = {int(r["scheduler_samples"]) for r in results if r.get("scheduler_samples") is not None}
        if len(positions) == 1:
            self.restored_scheduler_samples = positions.pop()
        if fingerprint is not None:
            checked = sum(1 for f in files if all(k in f for k in self.digest_keys))
            self.log(f"{LOG_PREFIX}restored {checked}/{len(files)} rank shard(s): state, RNG and LR "
                     f"scheduler digests equal the donor's, digests {fingerprint}")
        return fingerprint


class StreamRecoveryIsland:
    """:class:`~yeto.rl.peer_recovery.FileRecoverableIsland` of one role stream.

    ``shards``: a :class:`RankShardSnapshot` (``layout_tag``, ``save``,
    ``restore`` coroutines). Runs on a worker thread (``asyncio.to_thread``)
    while ``loop`` (the stream's event loop) is free; trainer calls are sent
    to that loop.

    Donor: :meth:`fragment_versions` are the stream's applied versions.
    Recovering: broadcasts are collected (the newest per fragment) until the
    stream applies them as its anchors (:attr:`resent`)."""

    def __init__(self, stream: Any, shards: Any, loop: asyncio.AbstractEventLoop, *,
                 recovering: bool) -> None:
        self.stream = stream
        self.shards = shards
        self.loop = loop
        self.recovering = recovering
        self.resent: dict[int, Any] = {}
        self.restored = False
        self._tag: str | None = None

    @property
    def scheduler_samples(self) -> int | None:
        """The restored trainer's LR scheduler position (None: not restored, or unknown)."""
        return getattr(self.shards, "restored_scheduler_samples", None)

    def _call(self, coro: Awaitable[Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def fragment_versions(self) -> tuple[int, ...]:
        if not self.recovering:
            return self.stream.fragment_versions
        count = self.stream.config.expected_fragments
        return tuple(self.resent[p].version if p in self.resent else 0 for p in range(count))

    def layout_tag(self) -> str:
        if self._tag is None:
            self._tag = self._call(self.shards.layout_tag())
        return self._tag

    def write_recovery_snapshot(self, directory: Path) -> None:
        self._call(self.shards.save(Path(directory)))

    def read_recovery_snapshot(self, directory: Path, versions: Sequence[int]) -> None:
        if self.restored:
            # A second source after a redial: the trainer is no longer fresh.
            raise RuntimeError("peer snapshot restored twice; restart the island")
        self.restored = True
        self._call(self.shards.restore(Path(directory)))
        return None  # anchors come from the resent fragments (module doc)

    def apply_broadcast(self, update: BcastFragment) -> None:
        stored = self.stream._stored_update(update)
        self.stream._validate_fragment_version(stored.fragment_id, stored.version)
        previous = self.resent.get(stored.fragment_id)
        if previous is not None and previous.version >= stored.version:
            stored.release()
            return
        if previous is not None:
            previous.release()
        self.resent[stored.fragment_id] = stored

    def restore_degraded(self) -> None:
        # No donor: the fresh trainer keeps fresh optimizer state.
        return None

    def release(self) -> None:
        for stored in self.resent.values():
            stored.release()
        self.resent.clear()
