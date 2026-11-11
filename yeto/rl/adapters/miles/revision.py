"""Identity checks of the Miles checkout an island runs (moved from the legacy engine).

``verify_miles_revision`` refuses an island whose Miles tree is not the pinned
ports commit (or an exact staged source tree); ``miles_execution_source_sha256``
is the content hash both checks and the full-parameter manifest probe use.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from pathlib import Path

from yeto.rl.adapters.miles.pins import MILES_NEXT_PINS, MilesRevisionPins

_LOWER_HEX = frozenset("0123456789abcdef")


def miles_execution_source_sha256(root: str | Path) -> str:
    """Hash the complete Miles Python/runtime-data source used by an RL job."""

    unresolved = Path(root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise RuntimeError("Miles source root is not a real directory")
    root = unresolved.resolve()
    sources = [root / "train.py"]
    for package_name in ("miles", "miles_plugins"):
        package = root / package_name
        if package.is_symlink() or not package.is_dir():
            raise RuntimeError(f"Miles source package is not a real directory: {package}")
        for path in package.rglob("*"):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise RuntimeError(f"Miles source contains a symlink: {relative}")
            if path.is_file():
                sources.append(path)
            elif not path.is_dir():
                raise RuntimeError(f"Miles source contains a special file: {relative}")
    if not sources[0].is_file() or sources[0].is_symlink():
        raise RuntimeError("Miles source has no regular train.py")

    digest = hashlib.sha256(b"yeto-miles-execution-source-v1\0")
    for path in sorted(sources, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or len(payload) != after.st_size:
            raise RuntimeError(f"Miles source changed while hashing: {relative!r}")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def verify_miles_revision(
    root: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected: MilesRevisionPins = MILES_NEXT_PINS,
) -> Path:
    """Verify either the pinned clean commit or an exact staged source tree."""

    root = Path(root).expanduser().resolve()

    if expected_source_sha256 is not None:
        source_sha256 = expected_source_sha256.lower()
        if len(source_sha256) != 64 or any(value not in _LOWER_HEX for value in source_sha256):
            raise ValueError("Miles source SHA256 is malformed")
        actual = miles_execution_source_sha256(root)
        if actual != source_sha256:
            raise RuntimeError(
                f"Miles source mismatch: expected {source_sha256}, got {actual}"
            )
        return _require_imported_from(root)

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot verify Miles checkout at {root}") from exc

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    origin = git("config", "--get", "remote.origin.url")
    if commit != expected.commit:
        raise RuntimeError(
            f"Miles revision mismatch: expected {expected.commit}, got {commit}"
        )
    if not expected.accepts_origin(origin):
        raise RuntimeError(
            f"Miles origin mismatch: expected {expected.repository}, got {origin}"
        )
    if branch != "HEAD":
        raise RuntimeError("Miles checkout is not detached")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Miles checkout is not clean")

    return _require_imported_from(root)


def _require_imported_from(root: Path) -> Path:
    miles = importlib.import_module("miles")
    package_path = Path(miles.__file__).resolve()
    if not package_path.is_relative_to(root):
        raise RuntimeError(f"imported Miles package {package_path} is outside {root}")
    return root
