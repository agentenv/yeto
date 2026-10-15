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
S3 and S3-compatible endpoints.
"""

from __future__ import annotations

import os
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

    def list(self, prefix: str, *, start_after: str = "") -> list[str]:
        base = self.root / prefix.rsplit("/", 1)[0] if "/" in prefix else self.root
        if not base.is_dir():
            return []
        keys = (p.relative_to(self.root).as_posix() for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(TMP_SUFFIX))
        return sorted(k for k in keys if k.startswith(prefix) and k > start_after)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)
