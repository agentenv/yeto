"""Island side of peer recovery (S27 E-3).

An island that lost its trainer state (process or node gone) starts a fresh
trainer and connects with ``SyncerClient(..., recover=True)``. The syncer
keeps it off the round roster (merge weight 0) and drives this exchange
(syncer side: ``syncer/src/recovery.rs``):

1. ``RECOVERY_CLOCK``: the syncer clock ``(t_s, V_s)``.
2. A donor island gets ``PEER_SNAPSHOT_REQ``. :class:`DonorService` keeps
   training and, at the first poll where every fragment version it holds is
   at least ``V_s[p]``, writes its parameters and inner optimizer state to the
   shared :class:`DirectorySnapshotStore` and answers with its versions, the
   snapshot sha256 and the store locator.
3. ``RECOVERY_SOURCE`` names that snapshot (or the degraded source).
   :func:`recover` checks the sha256 and the trainer layout tag, restores the
   snapshot, and reports the versions it now holds. A snapshot it cannot use
   is reported as failed and the syncer asks the next donor.
4. The syncer resends every fragment newer than those versions and sends
   ``RECOVERY_DONE``. :func:`recover` applies the resent fragments and returns.

Degraded source (no donor): the island restores on its own
(:meth:`RecoverableIsland.restore_degraded`: its latest verified cut if the
layout matches, else fresh optimizer state) and receives every fragment.

The snapshot never goes through the syncer. The store is a directory that
donor and recovering island both see; locators are bare file names inside it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from yeto import recovery_protocol as rp
from yeto.protocol import BcastFragment, SyncerClient

_META_KEY = "__yeto_recovery_meta__"


class RecoverableIsland(Protocol):
    """What an island trainer exposes to peer recovery."""

    def fragment_versions(self) -> Sequence[int]:
        """Syncer version of every fragment the island holds now."""

    def layout_tag(self) -> str:
        """Trainer layout identity (parallelism, shapes); a snapshot from a
        donor with another tag cannot be restored."""

    def export_recovery_state(self) -> Mapping[str, np.ndarray]:
        """Parameters and inner optimizer state (moments, step counters,
        LR schedule position) as named arrays."""

    def import_recovery_state(self, state: Mapping[str, np.ndarray], versions: Sequence[int]) -> None:
        """Replace parameters and inner optimizer state; the island now holds
        fragment ``versions``."""

    def apply_broadcast(self, update: BcastFragment) -> None:
        """Apply one syncer broadcast (the normal training-loop path)."""

    def restore_degraded(self) -> Sequence[int] | None:
        """No donor: restore the latest verified own cut if its layout matches,
        else fresh optimizer state. Return the fragment versions held, or None
        when nothing reliable is held (the syncer then resends every fragment)."""


class SnapshotIntegrityError(RuntimeError):
    """A snapshot is missing, corrupt, or from another layout."""


@dataclass(frozen=True)
class Snapshot:
    versions: tuple[int, ...]
    layout_tag: str
    state: dict[str, np.ndarray]


class DirectorySnapshotStore:
    """Snapshots as ``.npz`` files (no pickled objects) in a shared directory.

    Written to a temporary name and renamed, so a reader never sees a partial
    file. The sha256 covers the whole file.
    """

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, snapshot: Snapshot, *, name: str | None = None) -> tuple[str, bytes]:
        meta = {"versions": list(snapshot.versions), "layout_tag": snapshot.layout_tag}
        arrays = {k: np.asarray(v) for k, v in snapshot.state.items()}
        if _META_KEY in arrays:
            raise ValueError(f"state key {_META_KEY!r} is reserved")
        arrays[_META_KEY] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        data = buf.getvalue()
        locator = name or f"peer-snapshot-{uuid.uuid4().hex}.npz"
        path = self._path(locator)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return locator, hashlib.sha256(data).digest()

    def get(self, locator: str, sha256: bytes) -> Snapshot:
        try:
            data = self._path(locator).read_bytes()
        except OSError as exc:
            raise SnapshotIntegrityError(f"snapshot {locator!r} unreadable: {exc}") from exc
        if hashlib.sha256(data).digest() != sha256:
            raise SnapshotIntegrityError(f"snapshot {locator!r} sha256 mismatch")
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        meta = json.loads(arrays.pop(_META_KEY).tobytes().decode("utf-8"))
        return Snapshot(tuple(int(v) for v in meta["versions"]), str(meta["layout_tag"]), arrays)

    def remove(self, locator: str) -> None:
        self._path(locator).unlink(missing_ok=True)

    def _path(self, locator: str) -> Path:
        if not locator or "/" in locator or "\\" in locator or locator.startswith("."):
            raise SnapshotIntegrityError(f"invalid snapshot locator {locator!r}")
        return self.root / locator


class DonorService:
    """Answers snapshot requests while the donor keeps training.

    Call :meth:`poll` from the training loop after broadcasts are applied
    (a safe point: no optimizer step in progress).
    """

    def __init__(self, client: SyncerClient, island: RecoverableIsland, store: DirectorySnapshotStore) -> None:
        self.client = client
        self.island = island
        self.store = store
        self._waiting: dict[tuple[int, int], rp.SnapshotRequest] = {}
        self.served: list[tuple[rp.SnapshotRequest, str]] = []

    def poll(self) -> int:
        """Serve every request the island is caught up for; returns how many."""
        for request in self.client.drain_snapshot_requests():
            # A newer request for the same recovering generation replaces the old one.
            self._waiting[(request.recovering_id, request.recovering_generation)] = request
        if not self._waiting:
            return 0
        held = tuple(self.island.fragment_versions())
        served = 0
        for key, request in list(self._waiting.items()):
            if any(h < v for h, v in zip(held, request.versions, strict=True)):
                continue  # not yet at the clock; keep training
            snapshot = Snapshot(held, self.island.layout_tag(), dict(self.island.export_recovery_state()))
            locator, sha256 = self.store.put(snapshot)
            if not self.client.send_snapshot_ready(request, held, sha256, locator):
                continue  # disconnected; the syncer re-asks or picks another donor
            del self._waiting[key]
            self.served.append((request, locator))
            served += 1
        return served


@dataclass(frozen=True)
class RecoveryOutcome:
    path: str  # "peer" or "degraded"
    donor: int | None
    t_s: int
    done_step: int
    steps: int
    replayed: int
    refused_sources: int
    seconds: float
    #: Versions held after the restore, before the replay (None: degraded
    #: restore that held nothing reliable).
    restored_versions: tuple[int, ...] | None
    #: Versions held on return (the syncer versions at RECOVERY_DONE).
    versions: tuple[int, ...]


def recover(
    client: SyncerClient,
    island: RecoverableIsland,
    store: DirectorySnapshotStore,
    *,
    timeout: float = 3600.0,
    poll_s: float = 0.05,
    heartbeat_s: float = 1.0,
) -> RecoveryOutcome:
    """Run the recovering island's side until the syncer sends RECOVERY_DONE
    and every resent fragment is applied. Raises TimeoutError past ``timeout``.

    Heartbeats every ``heartbeat_s`` keep the island's lease while it waits
    for a donor (the training loop is not running yet)."""
    t0 = time.monotonic()
    next_heartbeat = t0
    deadline = t0 + timeout
    source: rp.RecoverySource | None = None
    held: Sequence[int] | None = None
    refused = 0
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"peer recovery did not finish within {timeout:.0f}s")
        client.check_health()
        if time.monotonic() >= next_heartbeat:
            client.heartbeat(0)
            next_heartbeat = time.monotonic() + heartbeat_s
        event = client.next_recovery_event(timeout=poll_s)
        if event is None:
            continue
        if isinstance(event, rp.RecoveryClock):
            source, held = None, None  # (re)started, possibly after a redial
        elif isinstance(event, rp.RecoverySource):
            if event.is_peer:
                try:
                    _restore_peer(island, store, event)
                except SnapshotIntegrityError as exc:
                    refused += 1
                    client.send_recovery_failed(str(exc))
                    continue
                held = event.versions
            else:
                held = island.restore_degraded()
            source = event
            client.send_recovery_restored(held)
        elif isinstance(event, rp.RecoveryDone):
            _apply_until(client, island, event.versions, deadline, poll_s)
            peer = source is not None and source.is_peer
            return RecoveryOutcome(
                path="peer" if peer else "degraded",
                donor=source.donor if peer else None,
                t_s=event.t_s,
                done_step=event.done_step,
                steps=event.steps,
                replayed=event.replayed,
                refused_sources=refused,
                seconds=time.monotonic() - t0,
                restored_versions=None if held is None else tuple(held),
                versions=tuple(island.fragment_versions()),
            )


def _restore_peer(island: RecoverableIsland, store: DirectorySnapshotStore, source: rp.RecoverySource) -> None:
    """Restore the donor snapshot; it is removed once restored (each snapshot
    serves exactly one recovering generation)."""
    snapshot = store.get(source.text, source.sha256)
    if snapshot.versions != source.versions:
        raise SnapshotIntegrityError("snapshot versions differ from the donor's report")
    if snapshot.layout_tag != island.layout_tag():
        raise SnapshotIntegrityError(
            f"snapshot layout {snapshot.layout_tag!r} differs from this island's {island.layout_tag()!r}"
        )
    island.import_recovery_state(snapshot.state, snapshot.versions)
    store.remove(source.text)


def _apply_until(client: SyncerClient, island: RecoverableIsland, target, deadline: float, poll_s: float) -> None:
    """Apply broadcasts until the island holds at least ``target``."""
    while True:
        for update in client.drain_updates():
            island.apply_broadcast(update)
        held = island.fragment_versions()
        if all(h >= t for h, t in zip(held, target, strict=True)):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"recovery replay incomplete: hold {list(held)}, need {list(target)}")
        client.check_health()
        time.sleep(poll_s)
