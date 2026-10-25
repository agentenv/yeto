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
the sha256 of the manifest bytes, so one digest covers every file. Files are
hashed on several threads (``hash_workers``). A Modal Volume commit gets a
timeout scaled to the snapshot bytes (``commit_budget``), and an in-place
writer holds the store's writer guard (``hold_writer``) until its commit, so a
Volume reload in the same container cannot drop the uncommitted files. Keys are
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
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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
DEFAULT_HASH_WORKERS = 8  # one Megatron rank shard per thread on an 8-GPU trainer
# S27 G-EL: a manifest that is not visible yet is read again after store reloads
# with backoff (2, 4, 8, 16, 30, 30 ... s) for up to this long in total; 0: refuse
# at once (the behavior before). Hardening only: in s27-elastic-a the manifest was
# missing because the snapshot was on the donor's local disk (a launch bug).
DEFAULT_MANIFEST_WAIT_S = 120.0
MANIFEST_RETRY_FIRST_S = 2.0
MANIFEST_RETRY_MAX_S = 30.0
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

    # Optional (S27 G-EL), all three or none: a donor that takes its state at the
    # safe point and writes the files while it trains on (DonorService ``capture``).
    #
    # start_recovery_snapshot(directory) -> handle: copy the state (host memory)
    #     and start writing it into ``directory``; return quickly.
    # recovery_snapshot_written(handle) -> bool: whether every file is written
    #     (asked at later safe points; never waits).
    # finish_recovery_snapshot(handle) -> None: collect the written files once
    #     ``recovery_snapshot_written`` is True; raises SnapshotRefused when a
    #     part was refused or failed.


def captures_snapshots(island: Any) -> bool:
    """The island implements the optional capture-then-write methods."""
    return all(callable(getattr(island, name, None)) for name in (
        "start_recovery_snapshot", "recovery_snapshot_written", "finish_recovery_snapshot"))


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


def _sha256_files(paths: Sequence[Path], workers: int) -> list[str]:
    """sha256 of every path, on up to ``workers`` threads (hashlib releases the
    GIL on large updates, so rank shards hash in parallel)."""
    if workers <= 1 or len(paths) <= 1:
        return [_sha256_file(path) for path in paths]
    with ThreadPoolExecutor(max_workers=min(workers, len(paths)), thread_name_prefix="snapshot-sha256") as pool:
        return list(pool.map(_sha256_file, paths))


def _tree(directory: Path) -> list[str]:
    return sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file())


class SnapshotWriter:
    """One snapshot being written: fill :attr:`directory`, then :meth:`commit`.

    On a directory backend :attr:`directory` is inside the store (no copy); on
    an object store it is a local staging directory, uploaded by
    :meth:`commit` and removed afterwards. :meth:`abort` drops what was
    written."""

    def __init__(self, store: "SnapshotStore", locator: str, directory: Path, *, staged: bool,
                 guard: Any = None) -> None:
        self.store = store
        self.locator = locator
        self.directory = directory
        self._staged = staged
        self._guard = guard  # the store's writer guard, released after the commit
        self._done = False
        self.bytes = 0  # snapshot bytes, known after commit()

    def commit(self, versions: Sequence[int], layout_tag: str) -> tuple[str, bytes]:
        """Hash every file, upload (object store), write the manifest last and
        commit the store. Returns ``(locator, manifest sha256)``."""
        if self._done:
            raise RuntimeError("snapshot already committed or aborted")
        store = self.store
        rels = [_check_rel(rel) for rel in _tree(self.directory)]
        if not rels:
            raise SnapshotIntegrityError("refusing an empty snapshot")
        paths = [self.directory / rel for rel in rels]
        digests = _sha256_files(paths, store.hash_workers)
        files = {rel: {"sha256": digest, "bytes": path.stat().st_size}
                 for rel, path, digest in zip(rels, paths, digests)}
        if self._staged:
            for rel, path in zip(rels, paths):
                upload_file(store.objects, store.file_key(self.locator, rel), path)
        self.bytes = sum(meta["bytes"] for meta in files.values())
        manifest = SnapshotManifest(tuple(int(v) for v in versions), str(layout_tag),
                                    locator_created_unix(self.locator), files)
        data = manifest.to_bytes()
        store.objects.write(store.manifest_key(self.locator), data)
        store.commit(pending_bytes=self.bytes)
        self._finish()
        return self.locator, hashlib.sha256(data).digest()

    def abort(self) -> None:
        if self._done:
            return
        self._finish()
        self.store.remove(self.locator)

    def _finish(self) -> None:
        self._done = True
        guard, self._guard = self._guard, None
        if guard is not None:
            guard.release()
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
    ``reload``: the medium's visibility calls (a Modal Volume needs both).
    ``commit_budget(bytes)``: a context in which a commit of that many new
    bytes may take longer (the Modal Volume's byte-scaled timeout).
    ``hold_writer()``: a guard an in-place writer holds until its commit (a
    Modal Volume reload skips while it is held). ``hash_workers``: threads
    that hash the files of one snapshot."""

    def __init__(self, objects: ObjectStore, *, prefix: str = SNAPSHOT_PREFIX,
                 local_root: str | os.PathLike | None = None,
                 commit: Callable[[], Any] | None = None, reload: Callable[[], Any] | None = None,
                 commit_budget: Callable[[int], Any] | None = None,
                 hold_writer: Callable[[], Any] | None = None,
                 hash_workers: int = DEFAULT_HASH_WORKERS,
                 staging: str | os.PathLike | None = None,
                 clock: Callable[[], float] = time.time,
                 manifest_wait_s: float = DEFAULT_MANIFEST_WAIT_S,
                 sleep: Callable[[float], Any] = time.sleep) -> None:
        if hash_workers < 1:
            raise ValueError("hash_workers must be >= 1")
        if not manifest_wait_s >= 0:
            raise ValueError("manifest_wait_s must be >= 0")
        self.manifest_wait_s = float(manifest_wait_s)
        self._sleep = sleep
        #: Reloads that found no manifest yet, over this store's life.
        self.manifest_retries = 0
        self.objects = objects
        self.prefix = prefix.strip("/")
        self.local_root = None if local_root is None else Path(local_root).expanduser()
        self._commit = commit
        self._reload = reload
        self._commit_budget = commit_budget
        self._hold_writer = hold_writer
        self.hash_workers = int(hash_workers)
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
        return cls(objects, local_root=objects.root, commit=store.commit, reload=store.reload,
                   commit_budget=getattr(store, "commit_budget", None),
                   hold_writer=getattr(store, "hold_writer", None), **kwargs)

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
    def commit(self, pending_bytes: int | None = None) -> None:
        """Make the writes visible; ``pending_bytes`` (new bytes) widens the
        commit timeout where the medium has one."""
        if self._commit is None:
            return
        budget = (self._commit_budget(int(pending_bytes))
                  if self._commit_budget is not None and pending_bytes else nullcontext())
        with budget:
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
            guard = self._hold_writer() if self._hold_writer is not None else None
            try:
                in_place.mkdir(parents=True)
            except BaseException:
                if guard is not None:
                    guard.release()
                raise
            return SnapshotWriter(self, locator, in_place, staged=False, guard=guard)
        directory = Path(tempfile.mkdtemp(prefix=f"{locator}-", dir=self.staging))
        return SnapshotWriter(self, locator, directory, staged=True)

    def put(self, snapshot: Snapshot) -> tuple[str, bytes]:
        """Array state as one ``.npz`` file (no pickled objects)."""
        with self.begin() as writer:
            write_array_state(writer.directory, snapshot.state)
            return writer.commit(snapshot.versions, snapshot.layout_tag)

    # -- read -------------------------------------------------------------------
    def manifest(self, locator: str, sha256: bytes) -> SnapshotManifest:
        """The manifest named by the syncer, checked against ``sha256``. Not found
        after a reload: reload and read again with backoff for up to
        ``manifest_wait_s`` seconds in total before refusing."""
        waited, pause = 0.0, MANIFEST_RETRY_FIRST_S
        while True:
            self.reload()
            data = self.objects.read(self.manifest_key(locator))
            if data is not None:
                break
            remaining = self.manifest_wait_s - waited
            if remaining <= 0:
                suffix = f" after {waited:.0f}s of store reloads" if waited else ""
                raise SnapshotIntegrityError(f"snapshot {locator!r} not found{suffix}")
            step = min(pause, remaining)
            self.manifest_retries += 1
            print(f"[yeto-peer-recovery] snapshot {locator!r} not visible yet; reloading the store "
                  f"again in {step:.0f}s ({waited:.0f}/{self.manifest_wait_s:.0f}s waited)",
                  file=sys.stderr, flush=True)
            self._sleep(step)
            waited += step
            pause = min(2 * pause, MANIFEST_RETRY_MAX_S)
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
        rels = list(manifest.files)
        digests = _sha256_files([directory / rel for rel in rels], self.hash_workers)
        for rel, digest in zip(rels, digests):
            if digest != manifest.files[rel]["sha256"]:
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
                       staging: str | os.PathLike | None = None,
                       manifest_wait_s: float = DEFAULT_MANIFEST_WAIT_S) -> SnapshotStore:
    """The snapshot store of an island: ``path`` is a checkpoint store path, so
    the launch layer's store selection (Modal Volume, S3, OSS, bucket mount,
    plain directory: :func:`yeto.rl.engine.resume.store_for`) applies."""
    from yeto.rl.engine.resume import store_for

    return SnapshotStore.for_checkpoint_store(store_for(path, environ=environ), staging=staging,
                                              manifest_wait_s=manifest_wait_s)


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


def digest_fingerprint(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    """One short digest over every row's ``path`` and ``keys``: the donor logs it
    when it writes a snapshot and the recovering island when it restored one, so
    the two log lines show the same trainer state."""
    body = sorted([str(r.get("path")), *(str(r.get(k)) for k in keys)] for r in rows)
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


def check_restored_digests(saved: Sequence[Mapping[str, Any]], restored: Sequence[Mapping[str, Any]],
                           keys: Sequence[str]) -> str | None:
    """A sharded trainer's restore must re-export the donor's per-shard digests.

    ``saved``: the snapshot's shard rows (``path`` + ``keys``, written by the
    donor); ``restored``: the rows each rank reports after its restore. Rows
    without every key are not checked (an older donor). Returns the digest
    fingerprint of the checked rows (None when none carries the keys); raises
    RuntimeError when a rank restored other state: the trainer is not the
    donor's and must not train."""
    checked = [row for row in saved if all(k in row for k in keys)]
    if not checked:
        return None
    by_path = {row.get("path"): row for row in restored}
    problems = []
    for row in checked:
        now = by_path.get(row["path"])
        if now is None:
            problems.append(f"{row['path']}: no restore summary")
            continue
        problems += [f"{row['path']}: restored {k} differs from the donor's (rank diff {now.get('diff')})"
                     for k in keys if now.get(k) != row[k]]
    if problems:
        raise RuntimeError("peer snapshot restore is not the donor's state: " + "; ".join(problems[:8]))
    return digest_fingerprint([by_path[row["path"]] for row in checked], keys)


def as_file_island(island: Any) -> FileRecoverableIsland:
    if callable(getattr(island, "write_recovery_snapshot", None)):
        return island
    return _ArrayFiles(island)


class DonorService:
    """Answers snapshot requests while the donor keeps training.

    Call :meth:`poll` from the training loop after broadcasts are applied
    (a safe point: no optimizer step in progress). The island writes the
    snapshot there; the files are then fixed. Hashing every file, the store
    commit and the answer to the syncer follow inline, or with ``background``
    on a worker thread, so the island trains on while a large snapshot is
    hashed and committed (Nemotron Nano: about 470 GB on a Modal Volume, whose
    commit alone can take longer than 600 s). ``background`` None (default):
    on for an island that writes its own snapshot files (a sharded trainer:
    Miles, verl), off for array state. One snapshot is in flight at a time;
    a later :meth:`poll` reports it. A snapshot that cannot be written,
    hashed or committed is refused (the syncer times out the donor and asks
    the next one); the training loop never sees that error. Every ``sweep_every_s``
    it also removes snapshots older than ``ttl_s`` from the store (any island's).

    ``capture`` (S27 G-EL; None: on when the island implements the optional
    capture-then-write methods, see :class:`FileRecoverableIsland`): at the safe
    point the island only copies its state to host memory and starts writing
    the files; the service asks at later safe points whether they are written
    (never waits), then hashes, commits and answers in the background. Training
    pauses only for the copy (Nemotron Nano: the inline write paused the donor
    for 545.6 s in s27-elastic-a). False: the island writes the files at the
    safe point (the behavior before).

    :attr:`timings` has one row per finished snapshot: ``write_s`` (training
    loop paused: the copy, or the whole write inline), ``background_write_s``
    (capture mode: from the copy to the last written file, training ran),
    ``commit_s`` (hash + commit + answer), ``bytes`` and ``result``.
    ``log`` (e.g. ``print`` to stderr) gets one line per refused or finished
    snapshot, prefixed with ``name`` (the role)."""

    def __init__(self, client: SyncerClient, island: Any, store: SnapshotStore, *,
                 ttl_s: float = DEFAULT_TTL_S, sweep_every_s: float = DEFAULT_SWEEP_EVERY_S,
                 background: bool | None = None, capture: bool | None = None,
                 log: Callable[[str], Any] | None = None, name: str = "island") -> None:
        if not ttl_s > 0 or not sweep_every_s > 0:
            raise ValueError("snapshot ttl and sweep interval must be positive")
        self.client = client
        file_island = callable(getattr(island, "write_recovery_snapshot", None))
        self.island = as_file_island(island)
        self.store = store
        self.ttl_s = ttl_s
        self.sweep_every_s = sweep_every_s
        self.background = file_island if background is None else bool(background)
        can_capture = captures_snapshots(self.island)
        if capture and not can_capture:
            raise ValueError("this island cannot capture a snapshot and write it later")
        self.capture = can_capture if capture is None else bool(capture)
        # The snapshot whose files the island is still writing (capture mode).
        self._capturing: dict[str, Any] | None = None
        self.log = log
        self.name = name
        self._next_sweep = 0.0
        self._waiting: dict[tuple[int, int], rp.SnapshotRequest] = {}
        self.served: list[tuple[rp.SnapshotRequest, str]] = []
        self.refused: list[tuple[rp.SnapshotRequest, str]] = []
        self.swept: list[str] = []
        self.timings: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._done: list[tuple[tuple[int, int], rp.SnapshotRequest, str, Any, dict[str, Any]]] = []
        self._worker: threading.Thread | None = None
        self._in_flight: str | None = None  # locator being hashed / committed

    @property
    def busy(self) -> bool:
        """A snapshot is being written (capture mode), hashed or committed."""
        return self._capturing is not None or (self._worker is not None and self._worker.is_alive())

    def poll(self) -> int:
        """Serve every request the island is caught up for; returns how many
        snapshots were handed to the syncer since the last poll."""
        self._maybe_sweep()
        for request in self.client.drain_snapshot_requests():
            # A newer request for the same recovering generation replaces the old one.
            self._waiting[(request.recovering_id, request.recovering_generation)] = request
        served = self._collect()
        if self._capturing is not None:
            self._check_captured()
            return served
        if self.busy or not self._waiting:
            return served
        held = tuple(self.island.fragment_versions())
        for key, request in list(self._waiting.items()):
            if any(h < v for h, v in zip(held, request.versions, strict=True)):
                continue  # not yet at the clock; keep training
            del self._waiting[key]
            started = time.monotonic()
            writer = self.store.begin()
            try:
                if self.capture:
                    handle = self.island.start_recovery_snapshot(writer.directory)
                else:
                    self.island.write_recovery_snapshot(writer.directory)
                tag = self.island.layout_tag()
            except SnapshotRefused as exc:
                writer.abort()
                self._refuse(request, str(exc))
                continue
            except BaseException:
                writer.abort()
                raise
            if self.capture:
                self._in_flight = writer.locator
                self._capturing = {"job": (key, request, writer, held, tag, time.monotonic() - started),
                                   "handle": handle, "captured": time.monotonic()}
                break  # one snapshot in flight; later polls check its files
            job = (key, request, writer, held, tag, time.monotonic() - started)
            if self.background:
                self._in_flight = writer.locator
                self._worker = threading.Thread(target=self._finish, args=job, daemon=True,
                                                name="peer-snapshot-commit")
                self._worker.start()
                break  # one snapshot in flight; the rest wait for a later poll
            self._finish(*job)
            served += self._collect()
        return served

    def _check_captured(self) -> None:
        """Capture mode, at a safe point: once every file of the captured snapshot
        is written, collect them and hash + commit + answer on the worker thread."""
        pending = self._capturing
        key, request, writer, held, tag, write_s = pending["job"]
        try:
            if not self.island.recovery_snapshot_written(pending["handle"]):
                return  # still writing; keep training
            self.island.finish_recovery_snapshot(pending["handle"])
        except SnapshotRefused as exc:
            self._capturing, self._in_flight = None, None
            writer.abort()
            self._refuse(request, str(exc))
            return
        except BaseException:
            self._capturing, self._in_flight = None, None
            writer.abort()
            raise
        self._capturing = None
        background_write_s = time.monotonic() - pending["captured"]
        self._worker = threading.Thread(target=self._finish, args=(key, request, writer, held, tag, write_s),
                                        kwargs={"background_write_s": background_write_s}, daemon=True,
                                        name="peer-snapshot-commit")
        self._worker.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the hash + commit in flight (tests, shutdown); True when none is
        left. A captured snapshot still being written needs more polls."""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return not self.busy

    def _finish(self, key, request: rp.SnapshotRequest, writer: SnapshotWriter, held, tag: str,
                write_s: float, background_write_s: float | None = None) -> None:
        started = time.monotonic()
        try:
            locator, sha256 = writer.commit(held, tag)
        except Exception as exc:  # noqa: BLE001 - refused, never raised into training
            try:
                writer.abort()
            except Exception:  # noqa: BLE001 - the TTL sweep removes what is left
                pass
            result, detail = "refused", f"{type(exc).__name__}: {exc}"
        else:
            if self.client.send_snapshot_ready(request, held, sha256, locator):
                result, detail = "served", locator
            else:  # disconnected; the request waits for the syncer to re-ask or move on
                try:
                    self.store.remove(locator)
                except Exception:  # noqa: BLE001 - the TTL sweep removes it
                    pass
                result, detail = "retry", None
        timing = {"recovering_id": request.recovering_id, "result": result, "bytes": writer.bytes,
                  "write_s": round(write_s, 3), "commit_s": round(time.monotonic() - started, 3)}
        if background_write_s is not None:
            timing["background_write_s"] = round(background_write_s, 3)
        with self._lock:
            self._done.append((key, request, result, detail, timing))

    def _collect(self) -> int:
        with self._lock:
            done, self._done = self._done, []
        served = 0
        for key, request, result, detail, timing in done:
            self.timings.append(timing)
            if result == "served":
                self.served.append((request, detail))
                served += 1
            elif result == "refused":
                self._refuse(request, detail)
            else:
                self._waiting.setdefault(key, request)
            if "background_write_s" in timing:
                self._say(f"donor snapshot for learner {timing['recovering_id']}: {result}, {timing['bytes']} bytes, "
                          f"capture {timing['write_s']:.1f}s (training paused), "
                          f"write {timing['background_write_s']:.1f}s + "
                          f"hash+commit {timing['commit_s']:.1f}s (training ran)")
            else:
                self._say(f"donor snapshot for learner {timing['recovering_id']}: {result}, {timing['bytes']} bytes, "
                          f"write {timing['write_s']:.1f}s (training paused), "
                          f"hash+commit {timing['commit_s']:.1f}s (training ran)")
        if not self.busy:
            self._worker, self._in_flight = None, None
        return served

    def _refuse(self, request: rp.SnapshotRequest, reason: str) -> None:
        self.refused.append((request, reason))
        self._say(f"donor refused a snapshot for learner {request.recovering_id}: {reason}")

    def _say(self, text: str) -> None:
        if self.log is not None:
            self.log(f"[yeto-peer-recovery] {self.name} {text}")

    def _maybe_sweep(self) -> None:
        now = time.monotonic()
        if now < self._next_sweep:
            return
        self._next_sweep = now + self.sweep_every_s
        keep = () if self._in_flight is None else (self._in_flight,)
        self.swept += self.store.sweep(self.ttl_s, keep=keep)


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
