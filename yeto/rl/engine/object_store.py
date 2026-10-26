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


class ObjectStoreError(RuntimeError):
    pass


@runtime_checkable
class ObjectStore(Protocol):
    """Whole objects by key. ``read`` returns None for a missing key."""

    def read(self, key: str) -> bytes | None: ...
    def write(self, key: str, data: bytes) -> None: ...
    def create(self, key: str, data: bytes) -> bool: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str, *, start_after: str = "") -> list[str]: ...
    def delete(self, key: str) -> None: ...


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
        shutil.copyfile(path, target)
        return True

    def list(self, prefix: str, *, start_after: str = "") -> list[str]:
        base = self.root / prefix.rsplit("/", 1)[0] if "/" in prefix else self.root
        if not base.is_dir():
            return []
        keys = (p.relative_to(self.root).as_posix() for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(TMP_SUFFIX))
        return sorted(k for k in keys if k.startswith(prefix) and k > start_after)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


PUSHED_STATE = ".yeto-pushed.json"  # in the mirror root: what the object store holds of it


class TreeMirror:
    """A local directory that stands for some subtrees of an object store.

    Code that writes files under ``root`` and then commits (round cuts, snapshots,
    ``LATEST``, policy releases) keeps working on a local disk; :meth:`push` makes
    the commit visible as objects and :meth:`pull` makes another host's commit
    visible here. Files are whole objects, so nothing is appended or rewritten in
    place on the object store.

    * ``push`` uploads every file under ``root`` that is new or changed since the
      last push (size and mtime): first the data files, then the record files
      (``records``, by file name: a record lists the data files next to it), then
      the pointer files (``pointers``, by file name), so a reader that follows a
      pointer finds what it names; then it deletes the objects of files removed here (a prune). It
      returns the seconds it took.
    * ``pull`` downloads the keys under ``subtrees`` that are missing here, and
      every pointer file again (pointers are the only rewritten objects).

    What was pushed or pulled is kept in :data:`PUSHED_STATE` under ``root`` (under
    ``flock``), so several processes of one host share one mirror without
    uploading each other's files again. Every writer of the subtrees must be on
    this host (one-node islands: a rank on another node writes to its own disk).
    Files move with :func:`upload_file` / :func:`download_file`, so a large file
    is never held in memory when the backend streams it."""

    def __init__(self, root: str | os.PathLike[str], objects: ObjectStore, *,
                 subtrees: tuple[str, ...], pointers: frozenset[str],
                 records: frozenset[str] = frozenset()) -> None:
        self.root = Path(root).expanduser()
        self.objects = objects
        self.subtrees = tuple(subtrees)
        self.pointers = frozenset(pointers)
        self.records = frozenset(records)
        self.root.mkdir(parents=True, exist_ok=True)

    def _upload_rank(self, rel: str) -> int:
        name = rel.rsplit("/", 1)[-1]
        return 2 if name in self.pointers else 1 if name in self.records else 0

    def _locked(self):
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def held():
            with open(self.root / (PUSHED_STATE + ".lock"), "a+b") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

        return held()

    def _state(self) -> dict[str, list[int]]:
        import json

        path = self.root / PUSHED_STATE
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def _save_state(self, state: dict[str, list[int]]) -> None:
        import json

        tmp = self.root / (PUSHED_STATE + TMP_SUFFIX)
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.root / PUSHED_STATE)

    def _local_files(self) -> dict[str, list[int]]:
        out = {}
        for path in self.root.rglob("*"):
            rel = path.relative_to(self.root).as_posix()
            if (path.is_file() and not path.name.endswith(TMP_SUFFIX)
                    and not rel.startswith(PUSHED_STATE)):
                st = path.stat()
                out[rel] = [st.st_size, st.st_mtime_ns]
        return out

    def push(self) -> float:
        import time

        started = time.monotonic()
        with self._locked():
            state = self._state()
            local = self._local_files()
            changed = [rel for rel, stamp in local.items() if state.get(rel) != stamp]
            changed.sort(key=lambda rel: (self._upload_rank(rel), rel))
            for rel in changed:
                upload_file(self.objects, rel, self.root / rel)
                state[rel] = local[rel]
                self._save_state(state)
            for rel in sorted(set(state) - set(local)):
                self.objects.delete(rel)
                del state[rel]
            self._save_state(state)
        return time.monotonic() - started

    def pull(self) -> None:
        with self._locked():
            state = self._state()
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
                    state[key] = [st.st_size, st.st_mtime_ns]
            self._save_state(state)
