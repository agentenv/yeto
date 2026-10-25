"""Peer recovery of a Miles full-parameter island (S27 E-3b).

The engine-neutral exchange is :mod:`yeto.rl.peer_recovery`; this module is
the Miles trainer behind it.

Snapshot content. Every Megatron rank writes one cut shard through
:mod:`yeto.rl.adapters.miles.cut_plugin` (its model copy, its inner optimizer
state with the FP32 main copy and moments, the LR scheduler, RNG) into the
snapshot directory, and the driver adds :data:`RANKS` (every shard's file,
sha256, size and rank coordinate). The rank shards are written in place on a
shared snapshot directory, or into a staging directory that every rank of the
island sees (one node, or a node-shared path); a shard that did not land
there refuses the snapshot (the syncer then asks another donor). Each rank's
entry also keeps the cut summary digests (state, RNG, LR scheduler position);
after the restore every rank must re-export exactly those, as the
single-island cut restore checks, else the restore fails.

Layout tag. The parameter layout hash plus the trainer parallel layout read
back from the ranks (world, TP, PP, CP, EP, DP). Only a donor with the same
tag can serve; the shard restore checks every rank coordinate again.

Replay. After the restore the island reports "nothing held", so the syncer
resends the current value of every fragment. Broadcasts overwrite (no update
log; see the E-3 design), and the resent fragments become the island's
anchors in one apply. The donor's local progress since its last broadcast
is therefore not copied: the donor still pushes it itself, and the
recovering island never counts it a second time. What the island takes from
the donor is the inner optimizer state (moments, step, LR schedule) and RNG.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.protocol import BcastFragment
from yeto.rl.peer_recovery import SnapshotIntegrityError, SnapshotRefused

RANKS = "ranks.json"
RANKS_SCHEMA = "yeto.miles_peer_snapshot/v1"
SNAPSHOT_CUT_ID = "peer-snapshot"
_LAYOUT_KEYS = ("world", "tp", "pp", "cp", "ep", "dp")
# Per-rank digests of the cut summary (cut_plugin): the restored trainer must
# re-export the donor's state, RNG and LR scheduler position exactly.
DIGEST_KEYS = ("state_digest", "rng_digest", "scheduler_samples")


def digest_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """One short digest over every rank's (path, DIGEST_KEYS): the donor logs it
    when it writes a snapshot and the recovering island when it restored one,
    so the two log lines show bitwise-equal trainer state."""
    body = sorted([str(r.get("path")), *(str(r.get(k)) for k in DIGEST_KEYS)] for r in rows)
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PeerRecoveryOptions:
    """Island options of peer recovery (from ``--rl-peer-recovery-*``).

    ``store``: checkpoint store path of the snapshots (resolved with
    :func:`yeto.rl.peer_recovery.snapshot_store_for`). ``recover``: this
    process lost its trainer state and asks the syncer for a donor.
    ``staging``: local directory for object-store transfers."""

    store: str
    recover: bool = False
    ttl_s: float = 3600.0
    timeout_s: float = 3600.0
    staging: str | None = None

    def __post_init__(self) -> None:
        if not self.store:
            raise ValueError("peer recovery needs a snapshot store")
        if not self.ttl_s > 0 or not self.timeout_s > 0:
            raise ValueError("peer recovery ttl and timeout must be positive")


def parallel_layout(coords: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The trainer layout from every rank's ``cut_plugin.rank_coords``."""
    if not coords:
        raise RuntimeError("trainer reported no ranks")
    first = coords[0]
    return {
        "world": len(coords),
        "tp": int(first.get("tp_size", 1)),
        "pp": int(first.get("pp_size", 1)),
        "cp": int(first.get("cp_size", 1)),
        "ep": int(first.get("ep_size", 1)),
        "dp": int(first.get("dp_size", 1)),
    }


def layout_tag(parameter_layout_hash: str, layout: Mapping[str, int]) -> str:
    body = {"schema": RANKS_SCHEMA, "parameter_layout_hash": parameter_layout_hash,
            "layout": {key: int(layout[key]) for key in _LAYOUT_KEYS}}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return "miles-" + hashlib.sha256(encoded).hexdigest()


def _refusals(results: Sequence[Mapping[str, Any]]) -> list[str]:
    from yeto.rl.adapters.miles.cut_plugin import REFUSAL_KIND, REFUSED

    return sorted({f"[{r.get(REFUSAL_KIND, 'refused')}] {r[REFUSED]}" for r in results if REFUSED in r})


class MilesRankShards:
    """Writes and restores the per-rank shards of one trainer group.

    ``group``: the Miles train group (``run_plugin`` runs a plugin function
    on every rank and returns their results). ``parameter_layout_hash``: the
    role's fragment layout hash."""

    def __init__(self, group: Any, parameter_layout_hash: str) -> None:
        self.group = group
        self.parameter_layout_hash = parameter_layout_hash
        self._tag: str | None = None

    async def layout_tag(self) -> str:
        if self._tag is None:
            from yeto.rl.adapters.miles.cut_plugin import RANK_COORDS

            coords = [dict(c) for c in await self.group.run_plugin(RANK_COORDS, {})]
            self._tag = layout_tag(self.parameter_layout_hash, parallel_layout(coords))
        return self._tag

    async def save(self, directory: Path) -> None:
        """Every rank writes its shard; refuses unless every shard is here."""
        from yeto.rl.adapters.miles.cut_plugin import SAVE_CUT_SHARD

        results = [dict(r) for r in await self.group.run_plugin(
            SAVE_CUT_SHARD, {"directory": str(directory), "cut_id": SNAPSHOT_CUT_ID})]
        refused = _refusals(results)
        if refused:
            raise SnapshotRefused("rank refused the peer snapshot: " + "; ".join(refused))
        files = [{"path": r["path"], "sha256": r["sha256"], "bytes": int(r["bytes"]), "coord": r["coord"],
                  **{k: r[k] for k in DIGEST_KEYS if k in r}} for r in results]
        missing = [f["path"] for f in files if not (directory / f["path"]).is_file()]
        if missing:
            raise SnapshotRefused(
                f"rank shards {missing} are not in the snapshot directory {directory} (a rank on "
                "another node needs a node-shared --rl-peer-recovery-staging or snapshot store)")
        ranks = {"schema": RANKS_SCHEMA, "layout_tag": await self.layout_tag(), "files": files}
        (directory / RANKS).write_text(json.dumps(ranks, sort_keys=True), encoding="utf-8")
        if all(k in f for f in files for k in DIGEST_KEYS):
            print(f"[yeto-peer-recovery] donor snapshot: {len(files)} rank shard(s), "
                  f"{sum(f['bytes'] for f in files)} bytes, digests {digest_fingerprint(files)}",
                  file=sys.stderr, flush=True)

    async def restore(self, directory: Path) -> None:
        """Load every rank's shard on this fresh trainer."""
        from yeto.rl.adapters.miles.cut_plugin import RESTORE_CUT_SHARD

        try:
            ranks = json.loads((directory / RANKS).read_text(encoding="utf-8"))
            files = list(ranks["files"])
            tag = ranks["layout_tag"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SnapshotIntegrityError(f"peer snapshot has no valid {RANKS}: {exc}") from exc
        if ranks.get("schema") != RANKS_SCHEMA or tag != await self.layout_tag():
            raise SnapshotIntegrityError("peer snapshot rank list is from another trainer layout")
        results = [dict(r) for r in await self.group.run_plugin(
            RESTORE_CUT_SHARD, {"directory": str(directory), "files": files, "cut_id": SNAPSHOT_CUT_ID})]
        refused = _refusals(results)
        if refused:
            # The trainer may be partly written: the caller must not train on it.
            raise RuntimeError("rank refused the peer snapshot restore: " + "; ".join(refused))
        self._check_restored(files, results)

    @staticmethod
    def _check_restored(files: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> None:
        """Every rank whose shard carries the donor's digests re-exports exactly
        them after the restore (as the single-island cut restore checks). A
        mismatch leaves a trainer that is not the donor's: the caller must not
        train on it."""
        saved = [f for f in files if all(k in f for k in DIGEST_KEYS)]
        if not saved:
            return
        by_path = {r.get("path"): r for r in results}
        problems = []
        for f in saved:
            r = by_path.get(f["path"])
            if r is None:
                problems.append(f"{f['path']}: no restore summary")
                continue
            problems += [f"{f['path']}: restored {k} differs from the donor's (rank diff {r.get('diff')})"
                         for k in DIGEST_KEYS if r.get(k) != f[k]]
        if problems:
            raise RuntimeError("peer snapshot restore is not the donor's state: " + "; ".join(problems[:8]))
        print(f"[yeto-peer-recovery] restored {len(saved)}/{len(files)} rank shard(s): state, RNG and LR "
              f"scheduler digests equal the donor's, digests {digest_fingerprint([by_path[f['path']] for f in saved])}",
              file=sys.stderr, flush=True)


class MilesPeerRecoveryIsland:
    """:class:`~yeto.rl.peer_recovery.FileRecoverableIsland` of one role stream.

    Runs on a worker thread (``asyncio.to_thread``) while ``loop`` (the
    engine's event loop) is free; trainer calls are sent to that loop.

    Donor: :attr:`fragment_versions` are the stream's applied versions.
    Recovering: broadcasts are collected (the newest per fragment) until the
    stream applies them as its anchors (:attr:`resent`)."""

    def __init__(self, stream: Any, shards: MilesRankShards, loop: asyncio.AbstractEventLoop, *,
                 recovering: bool) -> None:
        self.stream = stream
        self.shards = shards
        self.loop = loop
        self.recovering = recovering
        self.resent: dict[int, Any] = {}
        self.restored = False
        self._tag: str | None = None

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
