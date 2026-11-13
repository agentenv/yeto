"""Object storage port: whole-object reads and writes by key (neutral core).

Some shared state of a run is written by one host and read by another while both
run: the sample buffer (samplers -> trainer), the trainer commands (head ->
trainer island) and the newest policy release (trainer -> samplers). A bucket
mount (goofys, sky Storage MOUNT) cannot carry it: it does not append to or
rewrite an object in place, and a long-lived mount may serve stale metadata of an
object another host replaced. So these paths address the store as objects:

* every write replaces a whole object and is visible to every reader when it
  returns (S3 and S3-compatible stores such as MinIO give read-after-write and
  list-after-write consistency);
* ``create`` writes only when the key is absent: two writers that race for the
  same key get one winner (a sequence number taken once);
* ``list`` returns the keys under a prefix in byte order, after ``start_after``,
  so a reader that names its records by a zero-padded sequence number reads only
  the new ones.

Keys are relative to the store root (the run's ``--rl-checkpoint-store``) and use
``/`` as separator. Backends: :class:`DirectoryObjectStore` here (a local or
shared directory, for tests and single-host runs); ``yeto.cloud.s3_store`` for
S3 and S3-compatible endpoints; ``yeto.cloud.oss_store`` for Alibaba Cloud OSS.

A backend may also move whole files without holding them in memory
(``write_file`` / ``read_file``, e.g. a multipart upload of a policy release of
hundreds of GB) and report an object's size without reading it (``size``).
:func:`upload_file`, :func:`download_file` and :func:`object_size` use these
when the backend has them and fall back to whole-object reads and writes.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

TMP_SUFFIX = ".tmp"
# a download in progress next to its target (``S3ObjectStore.read_file``,
# ``DirectoryObjectStore.read_file``): renamed onto the target once whole, so a file
# at its final name is complete and a resumed release download keeps it
DOWNLOAD_SUFFIX = ".yeto-get" + TMP_SUFFIX


class ObjectStoreError(RuntimeError):
    pass


def atomic_create(objects: object) -> bool:
    """Whether ``objects.create`` is known not to be atomic: a store that declares
    ``atomic_create = False`` (a check-then-write create, e.g. the OSS relay) or one
    that found its endpoint refuses conditional writes (``conditional_writes`` False:
    its create fell back to check-then-write). Unknown counts as atomic; a caller
    that writes a record others race for must call it after a first create
    (``trainer_lease.TrainerLease.claim`` probes)."""
    return (getattr(objects, "atomic_create", True) is not False
            and getattr(objects, "conditional_writes", True) is not False)


@runtime_checkable
class ObjectStore(Protocol):
    """Whole objects by key. ``read`` returns None for a missing key."""

    def read(self, key: str) -> bytes | None: ...
    def write(self, key: str, data: bytes) -> None: ...
    def create(self, key: str, data: bytes) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str, *, start_after: str = "") -> list[str]: ...
    def delete(self, key: str) -> None: ...


# Store answers that no retry changes: the credentials or the bucket are wrong (an
# expired *session* token is left out: a credential provider may refresh it)
FATAL_STORE_CODES = frozenset({"AccessDenied", "403", "401", "Unauthorized", "InvalidAccessKeyId",
                               "SignatureDoesNotMatch", "NoSuchBucket", "InvalidBucketName", "AllAccessDisabled",
                               "AccountProblem"})
FATAL_STORE_ERRORS = frozenset({"NoCredentialsError", "PartialCredentialsError", "ProfileNotFound"})


# the error codes of an absent key (a store that answers with a bare ClientError)
ABSENT_STORE_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def store_error_code(error: BaseException) -> str:
    """The store's error code of a botocore ``ClientError``; "" for anything else
    (a connection-level error has ``response = None``)."""
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return ""
    return str((response.get("Error") or {}).get("Code", ""))


def is_absent_store_error(error: BaseException) -> bool:
    """A bare ``ClientError`` that says the key is not there (404 / NoSuchKey / NotFound)."""
    return store_error_code(error) in ABSENT_STORE_CODES


def is_fatal_store_error(error: BaseException) -> bool:
    """Whether a failed store request is a configuration error (refused
    credentials, no such bucket, no credentials at all) rather than an outage: a
    poll that fails this way is not read again, it ends the process (it would
    fail the same way forever while looking alive)."""
    cause = error.__cause__  # wrapped (incarnation_lease.LeaseReadError): judged by what the store said
    if cause is not None and cause is not error and is_fatal_store_error(cause):
        return True
    if type(error).__name__ in FATAL_STORE_ERRORS:
        return True
    return store_error_code(error) in FATAL_STORE_CODES


def upload_file(objects: ObjectStore, key: str, path: str | os.PathLike[str]) -> None:
    """Write the file at ``path`` as object ``key``."""
    write_file = getattr(objects, "write_file", None)
    if write_file is not None:
        write_file(key, path)
    else:
        objects.write(key, Path(path).read_bytes())


def download_file(objects: ObjectStore, key: str, path: str | os.PathLike[str]) -> bool:
    """Write object ``key`` to the file at ``path``; False (nothing written) when
    the key is missing. The caller renames a temporary ``path`` into place."""
    read_file = getattr(objects, "read_file", None)
    if read_file is not None:
        return bool(read_file(key, path))
    data = objects.read(key)
    if data is None:
        return False
    Path(path).write_bytes(data)
    return True


def object_size(objects: ObjectStore, key: str) -> int | None:
    """The size of object ``key`` in bytes, None when it is missing."""
    size = getattr(objects, "size", None)
    if size is not None:
        return size(key)
    data = objects.read(key)
    return None if data is None else len(data)


def object_checksum(objects: ObjectStore, key: str) -> str | None:
    """The SHA-256 checksum the store keeps for object ``key`` (``S3ObjectStore.checksum``:
    base64, ``-<parts>`` for a multipart object), None when the store keeps none."""
    checksum = getattr(objects, "checksum", None)
    return None if checksum is None else checksum(key)


def check_key(key: str) -> str:
    parts = key.split("/")
    if not key or key.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ObjectStoreError(f"bad object key {key!r}")
    return key


class DirectoryObjectStore:
    """:class:`ObjectStore` over a directory: one file per key. Writes go to a
    temporary file first, so a reader never sees a partial object; ``create``
    links the temporary file to its key, which fails when the key exists."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser()

    def _path(self, key: str) -> Path:
        return self.root / check_key(key)

    def _staged(self, path: Path, data: bytes) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}{TMP_SUFFIX}")
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        return tmp

    def read(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        os.replace(self._staged(path, data), path)

    def create(self, key: str, data: bytes) -> bool:
        path = self._path(key)
        tmp = self._staged(path, data)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return False
        finally:
            tmp.unlink()
        return True

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int | None:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def write_file(self, key: str, source: str | os.PathLike[str]) -> None:
        path = self._path(key)
        tmp = self._staged(path, b"")
        shutil.copyfile(source, tmp)
        os.replace(tmp, path)

    def read_file(self, key: str, target: str | os.PathLike[str]) -> bool:
        path = self._path(key)
        if not path.is_file():
            return False
        part = f"{target}{DOWNLOAD_SUFFIX}"
        shutil.copyfile(path, part)
        os.replace(part, target)
        return True

    def list(self, prefix: str, *, start_after: str = "") -> list[str]:
        base = self.root / prefix.rsplit("/", 1)[0] if "/" in prefix else self.root
        if not base.is_dir():
            return []
        keys = (p.relative_to(self.root).as_posix() for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(TMP_SUFFIX))
        return sorted(k for k in keys if k.startswith(prefix) and k > start_after)

    def list_page(self, prefix: str, *, start_after: str = "", limit: int = 1000) -> tuple[list[str], str | None]:
        """One listing request: up to ``limit`` keys after ``start_after`` and the key
        to continue from (None: that was the last page)."""
        keys = self.list(prefix, start_after=start_after)
        return keys[:limit], (keys[limit - 1] if len(keys) > limit else None)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


PUSHED_STATE = ".yeto-pushed.json"  # in the mirror root: what the object store holds of it
TREE_LOCK = "tree"  # the lock of every subtree that is not independent (see TreeMirror)
DEFAULT_FILE_CONCURRENCY = 8  # files of one upload tier moved at once by a push


def _under(rel: str, paths: tuple[str, ...]) -> bool:
    return any(rel == p or rel.startswith(p + "/") for p in paths)


class TreeMirror:
    """A local directory that stands for some subtrees of an object store.

    Code that writes files under ``root`` and then commits (round cuts, snapshots,
    ``LATEST``, policy releases) keeps working on a local disk; :meth:`push` makes
    the commit visible as objects and :meth:`pull` makes another host's commit
    visible here. Files are whole objects, so nothing is appended or rewritten in
    place on the object store.

    * ``push`` uploads every file under ``root`` that is new or changed since the
      last push (size and mtime), in three tiers: first the data files, then the
      record files (``records``, by file name: a record lists the data files next
      to it), then the pointer files (``pointers``, by file name), so a reader that
      follows a pointer finds what it names. The files of one tier move
      ``file_concurrency`` at a time; a tier starts when the previous one is
      complete. Then it deletes the objects of files removed here (a prune). It
      returns the seconds it took.
    * ``push(paths)`` does the same for the files under ``paths`` only (relative
      files or directories). ``independent`` names top-level directories (by
      prefix, e.g. ``policy-releases`` also covers ``policy-releases-w4a8``) whose
      writer commits them with their own paths: a push without ``paths`` leaves
      them alone, so a policy release never waits for a round cut of the same
      store, nor uploads it (GPU-A it10: the record and ``LATEST.json`` of a
      6 GB release waited 40 minutes behind a 178 GB cut).
    * ``pull`` downloads the keys under ``subtrees`` that are missing here, and
      every pointer file again (pointers are the only rewritten objects).

    Locks (``flock``, so several processes of one host share one mirror): each
    independent directory has its own lock and the rest of the tree one more
    (:data:`TREE_LOCK`), held while that part is pushed, so one part is never pushed
    twice at once while the other parts move freely. What was pushed or pulled is
    kept in :data:`PUSHED_STATE` under ``root``, read and rewritten under a lock of
    its own that is held only for that read-modify-write, never during a transfer.
    Every writer of the subtrees must be on this host, unless it uploads its files
    itself and records them with :meth:`record_pushed` (a rank on another node of a
    multi-node island, ``cut_plugin``). Files move with :func:`upload_file` /
    :func:`download_file`, so a large file is never held in memory when the backend
    streams it."""

    def __init__(self, root: str | os.PathLike[str], objects: ObjectStore, *,
                 subtrees: tuple[str, ...], pointers: frozenset[str],
                 records: frozenset[str] = frozenset(), independent: tuple[str, ...] = (),
                 file_concurrency: int = DEFAULT_FILE_CONCURRENCY) -> None:
        import threading

        if int(file_concurrency) < 1:
            raise ValueError("file concurrency must be >= 1")
        self.root = Path(root).expanduser()
        self.objects = objects
        self.subtrees = tuple(subtrees)
        self.pointers = frozenset(pointers)
        self.records = frozenset(records)
        self.independent = tuple(d.strip("/") for d in independent)
        self.file_concurrency = int(file_concurrency)
        self._state_mutex = threading.Lock()  # threads of this process; flock covers the others
        self.root.mkdir(parents=True, exist_ok=True)

    def _upload_rank(self, rel: str) -> int:
        name = rel.rsplit("/", 1)[-1]
        return 2 if name in self.pointers else 1 if name in self.records else 0

    def _is_independent(self, rel: str) -> bool:
        top = rel.split("/", 1)[0]
        return any(top.startswith(d) for d in self.independent)

    def _lock_key(self, rel: str) -> str:
        return rel.split("/", 1)[0] if self._is_independent(rel) else TREE_LOCK

    def _flock(self, name: str):
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def held():
            with open(self.root / name, "a+b") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

        return held()

    def _locked(self):
        """The state lock: only around a read-modify-write of :data:`PUSHED_STATE`."""
        import contextlib

        @contextlib.contextmanager
        def held():
            with self._state_mutex, self._flock(PUSHED_STATE + ".lock"):
                yield

        return held()

    def _part_locks(self, keys: list[str]):
        import contextlib

        stack = contextlib.ExitStack()
        for key in sorted(set(keys)):  # one order everywhere: no two pushes wait on each other
            stack.enter_context(self._flock(f"{PUSHED_STATE}.{key}.lock"))
        return stack

    def _state(self) -> dict[str, list[int]]:
        import json

        path = self.root / PUSHED_STATE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _save_state(self, state: dict[str, list[int]]) -> None:
        import json

        tmp = self.root / (PUSHED_STATE + TMP_SUFFIX)
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.root / PUSHED_STATE)

    def _update_state(self, done: dict[str, list[int]] | None = None, removed: list[str] = ()) -> None:
        with self._locked():
            state = self._state()
            state.update(done or {})
            for rel in removed:
                state.pop(rel, None)
            self._save_state(state)

    def record_pushed(self, rel: str) -> None:
        """Mark the local file ``rel`` as already in the object store (its writer
        uploaded it itself), so no push of this host uploads it again."""
        st = (self.root / rel).stat()
        self._update_state({rel: [st.st_size, st.st_mtime_ns]})

    def push_file(self, rel: str) -> int:
        """Upload the local file ``rel`` now (its writer, e.g. a trainer rank on any
        node, moves its own file instead of waiting for the driver's push), mark it
        pushed and return the size the object store reports for it (a HEAD), so the
        caller can hand a checked size to whoever commits the pointer. A large file
        moves as a resumable multipart upload with a SHA-256 per part (``write_file``):
        a call after a failed one continues it. :meth:`object_checksum` then gives the
        checksum the store keeps (one more HEAD)."""
        rel = check_key(rel)
        upload_file(self.objects, rel, self.root / rel)
        self.record_pushed(rel)
        size = object_size(self.objects, rel)
        if size is None:
            raise ObjectStoreError(f"{rel} is not in the object store after its upload")
        return int(size)

    def object_checksum(self, rel: str) -> str | None:
        return object_checksum(self.objects, check_key(rel))

    def evict_file(self, rel: str) -> int:
        """Delete the local copy of the pushed file ``rel`` and forget it in the pushed
        state, leaving its object in the store (a later push must not take the missing
        file for a prune). Returns the bytes freed."""
        rel = check_key(rel)
        path = self.root / rel
        with self._part_locks([self._lock_key(rel)]):
            freed = path.stat().st_size if path.is_file() else 0
            path.unlink(missing_ok=True)
            self._update_state(removed=[rel])
        return freed

    def delete_tree(self, rel: str) -> list[str]:
        """Delete every object under the directory ``rel`` (also those other hosts
        uploaded: the shards of a multi-node cut) and forget them in the pushed state."""
        rel = check_key(rel.strip("/"))
        keys = list(self.objects.list(rel + "/"))
        for key in keys:
            self.objects.delete(key)
        with self._locked():
            state = self._state()
            for key in [k for k in state if _under(k, (rel,))]:
                del state[key]
            self._save_state(state)
        return keys

    def evict_tree(self, rel: str) -> int:
        """Delete the local files under the directory ``rel`` and forget them in the
        pushed state, leaving their objects in the store: no later push takes the
        missing files for a prune. Returns the bytes freed."""
        rel = check_key(rel.strip("/"))
        freed = 0
        with self._part_locks([self._lock_key(rel)]):
            path = self.root / rel
            if path.is_dir():
                freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path, ignore_errors=True)
            self._update_state(removed=[k for k in self._state() if _under(k, (rel,))])
        return freed

    def _local_files(self, paths: tuple[str, ...] | None = None) -> dict[str, list[int]]:
        if paths is None:
            candidates = self.root.rglob("*")
        else:
            candidates = []
            for p in paths:
                path = self.root / p
                candidates += [path] if path.is_file() else list(path.rglob("*")) if path.is_dir() else []
        out = {}
        for path in candidates:
            rel = path.relative_to(self.root).as_posix()
            if (path.is_file() and not path.name.endswith(TMP_SUFFIX)
                    and not rel.startswith(PUSHED_STATE)
                    and (paths is not None or not self._is_independent(rel))):
                st = path.stat()
                out[rel] = [st.st_size, st.st_mtime_ns]
        return out

    def push(self, paths: "list[str] | tuple[str, ...] | None" = None) -> float:
        import time
        from concurrent.futures import ThreadPoolExecutor

        started = time.monotonic()
        scope = None if paths is None else tuple(sorted({check_key(str(p).strip("/")) for p in paths}))
        keys = [TREE_LOCK] if scope is None else [self._lock_key(p) for p in scope]
        with self._part_locks(keys):
            with self._locked():
                state = self._state()
            local = self._local_files(scope)
            changed = [rel for rel, stamp in local.items() if state.get(rel) != stamp]

            def upload(rel: str) -> None:
                upload_file(self.objects, rel, self.root / rel)
                self._update_state({rel: local[rel]})

            for tier in (0, 1, 2):
                files = sorted(rel for rel in changed if self._upload_rank(rel) == tier)
                if not files:
                    continue
                with ThreadPoolExecutor(min(self.file_concurrency, len(files))) as pool:
                    for future in [pool.submit(upload, rel) for rel in files]:
                        future.result()  # the first error stops before the next tier
            if scope is None:
                gone = [rel for rel in state if rel not in local and not self._is_independent(rel)]
            else:
                gone = [rel for rel in state if rel not in local and _under(rel, scope)]
            for rel in sorted(gone):
                self.objects.delete(rel)
            if gone:
                self._update_state(removed=gone)
        return time.monotonic() - started

    def pull(self) -> None:
        with self._part_locks([TREE_LOCK]):
            done: dict[str, list[int]] = {}
            for subtree in self.subtrees:
                for key in self.objects.list(subtree):
                    path = self.root / key
                    pointer = key.rsplit("/", 1)[-1] in self.pointers
                    if path.is_file() and not pointer:
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(path.name + TMP_SUFFIX)
                    if not download_file(self.objects, key, tmp):  # deleted since the listing (a prune)
                        tmp.unlink(missing_ok=True)
                        continue
                    os.replace(tmp, path)
                    st = path.stat()
                    done[key] = [st.st_size, st.st_mtime_ns]
            self._update_state(done)
