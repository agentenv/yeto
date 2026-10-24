"""Island side of peer recovery (S27 E-3).

An island that lost its trainer state (process or node gone) starts a fresh
trainer and connects with ``SyncerClient(..., recover=True)``. The syncer
keeps it off the round roster (merge weight 0) and drives this exchange
(syncer side: ``syncer/src/recovery.rs``):

1. ``RECOVERY_CLOCK``: the syncer clock ``(t_s, V_s)``.
2. A donor island gets ``PEER_SNAPSHOT_REQ``. :class:`DonorService` keeps
   training and, at the first poll where every fragment version it holds is
   at least ``V_s[p]``, writes its parameters and inner optimizer state into
   the :class:`SnapshotStore` and answers with its versions, the snapshot
   sha256 and the snapshot locator.
3. ``RECOVERY_SOURCE`` names that snapshot (or the degraded source).
   :func:`recover` checks the sha256 and the trainer layout tag, restores the
   snapshot, and reports the versions it now holds. A snapshot it cannot use
   is reported as failed and the syncer asks the next donor.
4. The syncer resends every fragment newer than those versions and sends
   ``RECOVERY_DONE``. :func:`recover` applies the resent fragments and returns.

Degraded source (no donor): the island restores on its own
(:meth:`RecoverableIsland.restore_degraded`: its latest verified cut if the
layout matches, else fresh optimizer state) and receives every fragment.

Snapshot transport. The snapshot never goes through the syncer, and islands
never stream it to each other. It is a set of files plus a JSON manifest in a
store that donor and recovering island both reach:

* a shared directory (one host, NFS, a bucket mount, a Modal Volume mounted on
  every island: :meth:`SnapshotStore.for_checkpoint_store` commits the Volume
  after a write and reloads it before a read);
* an object store (S3, S3-compatible, OSS) through the run's checkpoint store
  object API (``CheckpointStore.objects``).

The manifest lists every file with its sha256 and size; the protocol sha256 is
the sha256 of the manifest bytes, so one digest covers every file. Keys are
``<prefix>/<locator>/MANIFEST.json`` and ``<prefix>/<locator>/files/<path>``.
The locator starts with the creation time, so :meth:`SnapshotStore.sweep`
removes snapshots older than a TTL without reading them: snapshots of a donor
or recovering island that died, and partial uploads. The recovering island
removes a snapshot once it restored it.

Islands come in two forms. A :class:`RecoverableIsland` hands its state over
as named arrays (small trainers, tests; stored as one ``.npz`` without pickled
objects). A :class:`FileRecoverableIsland` writes and reads the snapshot files
itself (a sharded trainer: one file per rank).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from yeto import recovery_protocol as rp
from yeto.protocol import BcastFragment, SyncerClient
from yeto.rl.engine.object_store import (
    DirectoryObjectStore,
    ObjectStore,
    download_file,
    upload_file,
)

SNAPSHOT_PREFIX = "peer-snapshots"
MANIFEST = "MANIFEST.json"
MANIFEST_SCHEMA = "yeto.peer_snapshot/v1"
ARRAY_STATE = "state.npz"
DEFAULT_TTL_S = 3600.0
DEFAULT_SWEEP_EVERY_S = 300.0
_LOCATOR = re.compile(r"^s(\d{13})-([0-9a-f]{32})$")
_CHUNK = 1 << 20


class RecoverableIsland(Protocol):
    """What an island trainer with array state exposes to peer recovery."""

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


@runtime_checkable
class FileRecoverableIsland(Protocol):
    """An island that writes and reads its snapshot as files (one per rank)."""

    def fragment_versions(self) -> Sequence[int]: ...
    def layout_tag(self) -> str: ...
    def apply_broadcast(self, update: BcastFragment) -> None: ...
    def restore_degraded(self) -> Sequence[int] | None: ...

    def write_recovery_snapshot(self, directory: Path) -> None:
        """Write parameters and inner optimizer state into the empty
        ``directory`` (relative file names are kept)."""

    def read_recovery_snapshot(self, directory: Path, versions: Sequence[int]) -> Sequence[int] | None:
        """Restore the snapshot in ``directory`` (taken at fragment
        ``versions``). Return the versions to report as held: ``versions``, or
        None to have the syncer resend every fragment."""


class SnapshotIntegrityError(RuntimeError):
    """A snapshot is missing, corrupt, or from another layout."""


class SnapshotRefused(RuntimeError):
    """The donor cannot write a snapshot now (e.g. a rank refused its shard).
    The request is dropped: the syncer times out and asks the next donor."""


@dataclass(frozen=True)
class Snapshot:
    """Array state of a :class:`RecoverableIsland`."""

    versions: tuple[int, ...]
    layout_tag: str
    state: dict[str, np.ndarray]


@dataclass(frozen=True)
class SnapshotManifest:
    versions: tuple[int, ...]
    layout_tag: str
    created_unix: float
    files: dict[str, dict[str, Any]]  # relative path -> {"sha256": hex, "bytes": n}

    def to_bytes(self) -> bytes:
        body = {"schema": MANIFEST_SCHEMA, "versions": list(self.versions), "layout_tag": self.layout_tag,
                "created_unix": self.created_unix, "files": self.files}
        return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "SnapshotManifest":
        try:
            raw = json.loads(data.decode("utf-8"))
            if raw.get("schema") != MANIFEST_SCHEMA:
                raise ValueError(f"schema {raw.get('schema')!r}")
            files = {str(k): {"sha256": str(v["sha256"]), "bytes": int(v["bytes"])}
                     for k, v in dict(raw["files"]).items()}
            for rel in files:
                _check_rel(rel)
            return cls(tuple(int(v) for v in raw["versions"]), str(raw["layout_tag"]),
                       float(raw["created_unix"]), files)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SnapshotIntegrityError(f"snapshot manifest is invalid: {exc}") from exc


def _check_rel(rel: str) -> str:
    parts = rel.split("/")
    if not rel or rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
        raise SnapshotIntegrityError(f"invalid snapshot file path {rel!r}")
    return rel


def _check_locator(locator: str) -> str:
    if not isinstance(locator, str) or not _LOCATOR.match(locator):
        raise SnapshotIntegrityError(f"invalid snapshot locator {locator!r}")
    return locator


def locator_created_unix(locator: str) -> float:
    """Creation time encoded in a locator (milliseconds since the epoch)."""
    return int(_LOCATOR.match(_check_locator(locator)).group(1)) / 1000.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(directory: Path) -> list[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


class SnapshotWriter:
    """One snapshot being written: fill :attr:`directory`, then :meth:`commit`.

    On a directory backend :attr:`directory` is inside the store (no copy); on
    an object store it is a local staging directory, uploaded by
    :meth:`commit` and removed afterwards. :meth:`abort` drops what was
    written."""

    def __init__(self, store: "SnapshotStore", locator: str, directory: Path, *, staged: bool) -> None:
        self.store = store
        self.locator = locator
        self.directory = directory
        self._staged = staged
        self._done = False

    def commit(self, versions: Sequence[int], layout_tag: str) -> tuple[str, bytes]:
        """Hash every file, upload (object store), write the manifest last and
        commit the store. Returns ``(locator, manifest sha256)``."""
        if self._done:
            raise RuntimeError("snapshot already committed or aborted")
        store = self.store
        files = {}
        for rel in _tree(self.directory):
            path = self.directory / rel
            files[_check_rel(rel)] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            if self._staged:
                upload_file(store.objects, store.file_key(self.locator, rel), path)
        if not files:
            raise SnapshotIntegrityError("refusing an empty snapshot")
        manifest = SnapshotManifest(tuple(int(v) for v in versions), str(layout_tag),
                                    locator_created_unix(self.locator), files)
        data = manifest.to_bytes()
        store.objects.write(store.manifest_key(self.locator), data)
        store.commit()
        self._finish()
        return self.locator, hashlib.sha256(data).digest()

    def abort(self) -> None:
        if self._done:
            return
        self._finish()
        self.store.remove(self.locator)

    def _finish(self) -> None:
        self._done = True
        if self._staged:
            shutil.rmtree(self.directory, ignore_errors=True)

    def __enter__(self) -> "SnapshotWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()


class SnapshotStore:
    """Peer snapshots on an :class:`~yeto.rl.engine.object_store.ObjectStore`.

    ``local_root``: the directory the object store maps to when it is a
    directory (files are then written and read in place); None for a remote
    object store (files go through a local staging directory). ``commit`` /
    ``reload``: the medium's visibility calls (a Modal Volume needs both)."""

    def __init__(self, objects: ObjectStore, *, prefix: str = SNAPSHOT_PREFIX,
                 local_root: str | os.PathLike | None = None,
                 commit: Callable[[], Any] | None = None, reload: Callable[[], Any] | None = None,
                 staging: str | os.PathLike | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.objects = objects
        self.prefix = prefix.strip("/")
        self.local_root = None if local_root is None else Path(local_root).expanduser()
        self._commit = commit
        self._reload = reload
        self.staging = Path(staging).expanduser() if staging is not None else None
        if self.staging is not None:
            self.staging.mkdir(parents=True, exist_ok=True)
        self.clock = clock

    @classmethod
    def for_directory(cls, root: str | os.PathLike, **kwargs: Any) -> "SnapshotStore":
        objects = DirectoryObjectStore(root)
        return cls(objects, local_root=objects.root, **kwargs)

    @classmethod
    def for_checkpoint_store(cls, store: Any, **kwargs: Any) -> "SnapshotStore":
        """Reuse the run's checkpoint store (:func:`yeto.rl.engine.resume.store_for`):
        its object API when it has one (S3, OSS), else its directory (local,
        shared, bucket mount, Modal Volume) with its commit and reload."""
        objects = getattr(store, "objects", None)
        if objects is not None:
            return cls(objects, **kwargs)
        objects = DirectoryObjectStore(store.root)
        return cls(objects, local_root=objects.root, commit=store.commit, reload=store.reload, **kwargs)

    # -- keys -------------------------------------------------------------------
    def snapshot_prefix(self, locator: str) -> str:
        return f"{self.prefix}/{_check_locator(locator)}/"

    def manifest_key(self, locator: str) -> str:
        return self.snapshot_prefix(locator) + MANIFEST

    def file_key(self, locator: str, rel: str) -> str:
        return self.snapshot_prefix(locator) + "files/" + _check_rel(rel)

    def _local_files(self, locator: str) -> Path | None:
        if self.local_root is None:
            return None
        return self.local_root / self.snapshot_prefix(locator) / "files"

    # -- visibility -------------------------------------------------------------
    def commit(self) -> None:
        if self._commit is not None:
            self._commit()

    def reload(self) -> None:
        if self._reload is not None:
            self._reload()

    # -- write ------------------------------------------------------------------
    def new_locator(self) -> str:
        return f"s{int(self.clock() * 1000):013d}-{uuid.uuid4().hex}"

    def begin(self) -> SnapshotWriter:
        locator = self.new_locator()
        in_place = self._local_files(locator)
        if in_place is not None:
            in_place.mkdir(parents=True)
            return SnapshotWriter(self, locator, in_place, staged=False)
        directory = Path(tempfile.mkdtemp(prefix=f"{locator}-", dir=self.staging))
        return SnapshotWriter(self, locator, directory, staged=True)

    def put(self, snapshot: Snapshot) -> tuple[str, bytes]:
        """Array state as one ``.npz`` file (no pickled objects)."""
        with self.begin() as writer:
            write_array_state(writer.directory, snapshot.state)
            return writer.commit(snapshot.versions, snapshot.layout_tag)

    # -- read -------------------------------------------------------------------
    def manifest(self, locator: str, sha256: bytes) -> SnapshotManifest:
        self.reload()
        data = self.objects.read(self.manifest_key(locator))
        if data is None:
            raise SnapshotIntegrityError(f"snapshot {locator!r} not found")
        if hashlib.sha256(data).digest() != sha256:
            raise SnapshotIntegrityError(f"snapshot {locator!r} sha256 mismatch")
        return SnapshotManifest.from_bytes(data)

    def fetch(self, locator: str, sha256: bytes, into: str | os.PathLike) -> tuple[SnapshotManifest, Path]:
        """Check the manifest against ``sha256`` and every file against the
        manifest. Returns the manifest and the directory holding the files: in
        place on a directory backend, else ``into`` (downloaded)."""
        manifest = self.manifest(locator, sha256)
        in_place = self._local_files(locator)
        directory = in_place if in_place is not None else Path(into)
        for rel, meta in manifest.files.items():
            path = directory / rel
            if in_place is None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".part")
                if not download_file(self.objects, self.file_key(locator, rel), tmp):
                    raise SnapshotIntegrityError(f"snapshot {locator!r} misses {rel!r}")
                os.replace(tmp, path)
            if not path.is_file() or path.stat().st_size != meta["bytes"]:
                raise SnapshotIntegrityError(f"snapshot {locator!r}: {rel!r} is missing or has the wrong size")
            if _sha256_file(path) != meta["sha256"]:
                raise SnapshotIntegrityError(f"snapshot {locator!r}: {rel!r} sha256 mismatch")
        return manifest, directory

    def get(self, locator: str, sha256: bytes) -> Snapshot:
        """Array state written by :meth:`put`."""
        with tempfile.TemporaryDirectory(dir=self.staging) as tmp:
            manifest, directory = self.fetch(locator, sha256, tmp)
            return Snapshot(manifest.versions, manifest.layout_tag, read_array_state(directory))

    # -- cleanup ----------------------------------------------------------------
    def locators(self) -> list[str]:
        """Every snapshot in the store, committed or partial."""
        seen = set()
        for key in self.objects.list(self.prefix + "/"):
            head = key[len(self.prefix) + 1:].split("/", 1)[0]
            if _LOCATOR.match(head):
                seen.add(head)
        if self.local_root is not None and (self.local_root / self.prefix).is_dir():
            # an in-place snapshot whose writer died before its first file
            seen.update(p.name for p in (self.local_root / self.prefix).iterdir() if _LOCATOR.match(p.name))
        return sorted(seen)

    def remove(self, locator: str) -> None:
        """Delete a snapshot: the manifest first, so a reader never sees a
        manifest whose files are gone."""
        self.objects.delete(self.manifest_key(locator))
        for key in self.objects.list(self.snapshot_prefix(locator)):
            self.objects.delete(key)
        in_place = self._local_files(locator)
        if in_place is not None:
            shutil.rmtree(in_place.parent, ignore_errors=True)
        self.commit()

    def sweep(self, ttl_s: float, *, keep: Sequence[str] = ()) -> list[str]:
        """Remove every snapshot created more than ``ttl_s`` ago (abandoned by
        a dead donor or recovering island, or timed out); returns their
        locators. ``keep``: locators still in use here."""
        now = self.clock()
        removed = []
        for locator in self.locators():
            if locator in keep or now - locator_created_unix(locator) <= ttl_s:
                continue
            self.remove(locator)
            removed.append(locator)
        return removed


class DirectorySnapshotStore(SnapshotStore):
    """A :class:`SnapshotStore` in a directory that every island sees."""

    def __init__(self, root: str | os.PathLike, **kwargs: Any) -> None:
        objects = DirectoryObjectStore(root)
        objects.root.mkdir(parents=True, exist_ok=True)
        super().__init__(objects, local_root=objects.root, **kwargs)
        self.root = objects.root


def snapshot_store_for(path: str | os.PathLike, *, environ: Mapping[str, str] | None = None,
                       staging: str | os.PathLike | None = None) -> SnapshotStore:
    """The snapshot store of an island: ``path`` is a checkpoint store path, so
    the launch layer's store selection (Modal Volume, S3, OSS, bucket mount,
    plain directory: :func:`yeto.rl.engine.resume.store_for`) applies."""
    from yeto.rl.engine.resume import store_for

    return SnapshotStore.for_checkpoint_store(store_for(path, environ=environ), staging=staging)


def write_array_state(directory: Path, state: Mapping[str, np.ndarray]) -> None:
    arrays = {k: np.asarray(v) for k, v in state.items()}
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    (Path(directory) / ARRAY_STATE).write_bytes(buf.getvalue())


def read_array_state(directory: Path) -> dict[str, np.ndarray]:
    path = Path(directory) / ARRAY_STATE
    if not path.is_file():
        raise SnapshotIntegrityError(f"snapshot has no {ARRAY_STATE}")
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


class _ArrayFiles:
    """A :class:`RecoverableIsland` seen as a :class:`FileRecoverableIsland`."""

    def __init__(self, island: RecoverableIsland) -> None:
        self.island = island

    def __getattr__(self, name: str) -> Any:
        return getattr(self.island, name)

    def write_recovery_snapshot(self, directory: Path) -> None:
        write_array_state(directory, self.island.export_recovery_state())

    def read_recovery_snapshot(self, directory: Path, versions: Sequence[int]) -> Sequence[int]:
        self.island.import_recovery_state(read_array_state(directory), versions)
        return versions


def as_file_island(island: Any) -> FileRecoverableIsland:
    if callable(getattr(island, "write_recovery_snapshot", None)):
        return island
    return _ArrayFiles(island)


class DonorService:
    """Answers snapshot requests while the donor keeps training.

    Call :meth:`poll` from the training loop after broadcasts are applied
    (a safe point: no optimizer step in progress). Every ``sweep_every_s`` it
    also removes snapshots older than ``ttl_s`` from the store (any island's)."""

    def __init__(self, client: SyncerClient, island: Any, store: SnapshotStore, *,
                 ttl_s: float = DEFAULT_TTL_S, sweep_every_s: float = DEFAULT_SWEEP_EVERY_S) -> None:
        if not ttl_s > 0 or not sweep_every_s > 0:
            raise ValueError("snapshot ttl and sweep interval must be positive")
        self.client = client
        self.island = as_file_island(island)
        self.store = store
        self.ttl_s = ttl_s
        self.sweep_every_s = sweep_every_s
        self._next_sweep = 0.0
        self._waiting: dict[tuple[int, int], rp.SnapshotRequest] = {}
        self.served: list[tuple[rp.SnapshotRequest, str]] = []
        self.refused: list[tuple[rp.SnapshotRequest, str]] = []
        self.swept: list[str] = []

    def poll(self) -> int:
        """Serve every request the island is caught up for; returns how many."""
        self._maybe_sweep()
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
            try:
                with self.store.begin() as writer:
                    self.island.write_recovery_snapshot(writer.directory)
                    locator, sha256 = writer.commit(held, self.island.layout_tag())
            except SnapshotRefused as exc:
                del self._waiting[key]
                self.refused.append((request, str(exc)))
                continue
            if not self.client.send_snapshot_ready(request, held, sha256, locator):
                self.store.remove(locator)
                continue  # disconnected; the syncer re-asks or picks another donor
            del self._waiting[key]
            self.served.append((request, locator))
            served += 1
        return served

    def _maybe_sweep(self) -> None:
        now = time.monotonic()
        if now < self._next_sweep:
            return
        self._next_sweep = now + self.sweep_every_s
        self.swept += self.store.sweep(self.ttl_s)


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
    #: Versions reported as held after the restore, before the replay (None:
    #: the island asked for every fragment).
    restored_versions: tuple[int, ...] | None
    #: Versions held on return (the syncer versions at RECOVERY_DONE).
    versions: tuple[int, ...]


def recover(
    client: SyncerClient,
    island: Any,
    store: SnapshotStore,
    *,
    timeout: float = 3600.0,
    poll_s: float = 0.05,
    heartbeat_s: float = 1.0,
) -> RecoveryOutcome:
    """Run the recovering island's side until the syncer sends RECOVERY_DONE
    and every resent fragment is applied. Raises TimeoutError past ``timeout``.

    ``island``: a :class:`RecoverableIsland` or :class:`FileRecoverableIsland`.
    Heartbeats every ``heartbeat_s`` keep the island's lease while it waits
    for a donor (the training loop is not running yet)."""
    island = as_file_island(island)
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
                    held = _restore_peer(island, store, event)
                except SnapshotIntegrityError as exc:
                    refused += 1
                    client.send_recovery_failed(str(exc))
                    continue
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


def _restore_peer(island: FileRecoverableIsland, store: SnapshotStore,
                  source: rp.RecoverySource) -> Sequence[int] | None:
    """Restore the donor snapshot; it is removed once restored (each snapshot
    serves exactly one recovering generation). Returns the versions to report."""
    locator = _check_locator(source.text)
    with tempfile.TemporaryDirectory(dir=store.staging) as tmp:
        manifest, directory = store.fetch(locator, source.sha256, tmp)
        if manifest.versions != tuple(source.versions):
            raise SnapshotIntegrityError("snapshot versions differ from the donor's report")
        if manifest.layout_tag != island.layout_tag():
            raise SnapshotIntegrityError(
                f"snapshot layout {manifest.layout_tag!r} differs from this island's {island.layout_tag()!r}"
            )
        held = island.read_recovery_snapshot(directory, manifest.versions)
    store.remove(locator)
    return held


def _apply_until(client: SyncerClient, island: Any, target, deadline: float, poll_s: float) -> None:
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
