"""Exact, bounded-residency BF16 anchor storage for large DiLoCo islands.

The store is deliberately narrow: one process owns one fresh run directory,
and every fragment is an independently replaceable file.  It is not a
checkpoint format.  A crashed run's directory is retained for diagnosis and
must never be reused by a later run.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

import torch

_FORMAT = "yeto-bf16-anchor-v1"
_HEADER_BYTES = 4096
_IO_CHUNK_BYTES = 8 * 1024 * 1024
_T = TypeVar("_T")


def _evict_file_pages(fd: int, length: int) -> None:
    """Best-effort eviction; correctness never depends on kernel support."""

    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    fadvise = getattr(os, "posix_fadvise", None)
    if advice is None or fadvise is None:
        return
    try:
        fadvise(fd, 0, length, advice)
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes | memoryview) -> None:
    """Write one buffer completely, including under injected short writes."""

    view = memoryview(data)
    try:
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while spilling anchor")
            view = view[written:]
    finally:
        view.release()


class BF16AnchorSpill:
    """Own exact BF16 anchors while mapping at most one fragment per read."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        layout_fingerprint: bytes,
        fragment_numels: list[int] | tuple[int, ...],
    ) -> None:
        self.root = Path(root)
        if not self.root.is_absolute() or self.root == Path("/"):
            raise ValueError("anchor spill root must be an absolute path other than /")
        if len(layout_fingerprint) != 32:
            raise ValueError("anchor spill layout fingerprint must contain 32 bytes")
        if not fragment_numels or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in fragment_numels
        ):
            raise ValueError("anchor spill fragment sizes must be positive integers")

        self.layout_hex = layout_fingerprint.hex()
        self.fragment_numels = tuple(fragment_numels)
        self._written = [False] * len(self.fragment_numels)
        self._versions: list[int | None] = [None] * len(self.fragment_numels)
        self._closed = False
        self._created_files: set[Path] = set()
        self._lock_fd: int | None = None

        # Exclusive creation is intentional: an existing path may contain a
        # crashed run, and silently deleting or reusing it could change the
        # subtraction base.  Retries must use a fresh OUTPUT_DIR.
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        try:
            lock_path = self.root / ".owner.lock"
            self._lock_fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
            self._created_files.add(lock_path)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _write_all(self._lock_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(self._lock_fd)
            _fsync_directory(self.root)
            _fsync_directory(self.root.parent)
        except BaseException:
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None
            try:
                lock_path.unlink()
            except (FileNotFoundError, UnboundLocalError):
                pass
            try:
                self.root.rmdir()
            except OSError:
                pass
            raise

    @property
    def num_fragments(self) -> int:
        return len(self.fragment_numels)

    @property
    def payload_bytes(self) -> int:
        return sum(self.fragment_numels) * 2

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("anchor spill store is closed")

    def _path(self, fragment_id: int) -> Path:
        if (
            isinstance(fragment_id, bool)
            or not isinstance(fragment_id, int)
            or not 0 <= fragment_id < self.num_fragments
        ):
            raise IndexError(f"invalid anchor fragment {fragment_id!r}")
        return self.root / f"fragment-{fragment_id:03d}.anchor"

    def has(self, fragment_id: int, version: int) -> bool:
        self._check_open()
        self._path(fragment_id)
        return self._written[fragment_id] and self._versions[fragment_id] == version

    def write(self, fragment_id: int, version: int, tensor: torch.Tensor) -> None:
        """Atomically replace one anchor without making another full copy."""

        self._check_open()
        path = self._path(fragment_id)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("anchor version must be a nonnegative integer")
        if tensor.device.type != "cpu":
            raise ValueError("anchor spill accepts only CPU tensors")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"anchor spill requires torch.bfloat16, got {tensor.dtype}")
        if tensor.ndim != 1 or tensor.numel() != self.fragment_numels[fragment_id]:
            raise ValueError(
                f"fragment {fragment_id} shape/size mismatch: got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError("anchor spill tensor must be contiguous")

        tmp = self.root / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        payload = memoryview(tensor.view(torch.uint8).numpy())
        digest = hashlib.sha256()
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            _write_all(fd, b"\0" * _HEADER_BYTES)
            for start in range(0, payload.nbytes, _IO_CHUNK_BYTES):
                chunk = payload[start : start + _IO_CHUNK_BYTES]
                digest.update(chunk)
                _write_all(fd, chunk)
            metadata = {
                "dtype": "torch.bfloat16",
                "format": _FORMAT,
                "fragment_id": fragment_id,
                "layout_fingerprint": self.layout_hex,
                "numel": tensor.numel(),
                "payload_sha256": digest.hexdigest(),
                "version": version,
            }
            header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            if len(header) >= _HEADER_BYTES:
                raise RuntimeError("anchor spill metadata exceeds fixed header")
            os.lseek(fd, 0, os.SEEK_SET)
            padded = header + b"\0" * (_HEADER_BYTES - len(header))
            _write_all(fd, padded)
            os.fsync(fd)
            os.replace(tmp, path)
            self._created_files.add(path)
            _fsync_directory(self.root)
            _evict_file_pages(fd, _HEADER_BYTES + payload.nbytes)
            self._written[fragment_id] = True
            self._versions[fragment_id] = version
        finally:
            payload.release()
            os.close(fd)
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def read(
        self,
        fragment_id: int,
        version: int,
        consumer: Callable[[torch.Tensor], _T],
    ) -> _T:
        """Validate/read one anchor, consume it, then evict its file pages."""

        self._check_open()
        path = self._path(fragment_id)
        if not self._written[fragment_id]:
            raise FileNotFoundError(
                f"anchor fragment {fragment_id} has not been written"
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise RuntimeError(
                f"cannot open anchor fragment {fragment_id}: {exc}"
            ) from exc
        tensor = None
        payload = None
        try:
            expected_payload_bytes = self.fragment_numels[fragment_id] * 2
            expected_size = _HEADER_BYTES + expected_payload_bytes
            file_stat = os.fstat(fd)
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_size != expected_size
            ):
                raise RuntimeError(
                    f"anchor fragment {fragment_id} size {file_stat.st_size} != {expected_size}"
                )
            raw_header = os.pread(fd, _HEADER_BYTES, 0).rstrip(b"\0")
            try:
                metadata = json.loads(raw_header.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"anchor fragment {fragment_id} has an invalid header"
                ) from exc
            expected = {
                "dtype": "torch.bfloat16",
                "format": _FORMAT,
                "fragment_id": fragment_id,
                "layout_fingerprint": self.layout_hex,
                "numel": self.fragment_numels[fragment_id],
                "version": version,
            }
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise RuntimeError(
                        f"anchor fragment {fragment_id} {key} mismatch: "
                        f"{metadata.get(key)!r} != {value!r}"
                    )
            expected_digest = metadata.get("payload_sha256")
            if not isinstance(expected_digest, str) or len(expected_digest) != 64:
                raise RuntimeError(
                    f"anchor fragment {fragment_id} has an invalid payload digest"
                )
            # Allocate exactly one anonymous BF16 fragment. Avoid mmap here:
            # explicit ownership makes it impossible for a tensor view to
            # keep file-backed pages resident after this call returns.
            tensor = torch.empty(
                self.fragment_numels[fragment_id], dtype=torch.bfloat16
            )
            payload = memoryview(tensor.view(torch.uint8).numpy())
            digest = hashlib.sha256()
            for start in range(0, expected_payload_bytes, _IO_CHUNK_BYTES):
                stop = min(start + _IO_CHUNK_BYTES, expected_payload_bytes)
                chunk = payload[start:stop]
                position = 0
                while position < len(chunk):
                    count = os.preadv(
                        fd,
                        [chunk[position:]],
                        _HEADER_BYTES + start + position,
                    )
                    if count <= 0:
                        raise RuntimeError(
                            f"anchor fragment {fragment_id} payload is truncated"
                        )
                    position += count
                digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise RuntimeError(f"anchor fragment {fragment_id} payload is corrupt")
            if tensor.numel() != self.fragment_numels[fragment_id]:
                raise RuntimeError(
                    f"anchor fragment {fragment_id} decoded size mismatch"
                )
            result = consumer(tensor)
            if (
                isinstance(result, torch.Tensor)
                and result.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
            ):
                raise RuntimeError("anchor consumer retained the spill read buffer")
            return result
        finally:
            if payload is not None:
                payload.release()
            tensor = None
            _evict_file_pages(fd, os.fstat(fd).st_size)
            os.close(fd)

    def close(self, *, successful: bool) -> None:
        """Release ownership; delete only this process's files on success."""

        if self._closed:
            return
        if successful:
            for fragment_id, written in enumerate(self._written):
                if not written:
                    continue
                path = self._path(fragment_id)
                if path in self._created_files:
                    path.unlink(missing_ok=True)
            lock_path = self.root / ".owner.lock"
            if lock_path in self._created_files:
                lock_path.unlink(missing_ok=True)
            _fsync_directory(self.root)
        if self._lock_fd is None:
            raise RuntimeError("anchor spill ownership lock is unavailable")
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_fd = None
        self._closed = True
        if successful:
            self.root.rmdir()
            _fsync_directory(self.root.parent)


__all__ = ["BF16AnchorSpill"]
