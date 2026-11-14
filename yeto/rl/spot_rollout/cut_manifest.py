"""Hand a policy release to a far spot island through a nearby object store.

The npu1 uplink is ~0.2 MB/s to Nebius, so a 120B release (a 13 GB delta,
``policy_release`` ``.ydelta`` files) cannot go over the syncer link. Instead:

1. the trainer writes the release to its store as before (``r<seq>-v<version>``
   plus ``LATEST.json``);
2. a mirror copies the release directory to an object store near the spot VM
   (a Modal volume or bucket; any byte-for-byte copier) and makes a
   **manifest**: every transported file with its size and sha256, the release
   record (the store's ``LATEST.json`` for that release) and its own digest;
3. only the manifest (a few hundred KB) and its sha256 travel over the control
   plane (the syncer link / the npu1 SSH tunnel);
4. the spot VM pulls the files from the near store, :func:`place` checks every
   size and sha256 against the manifest (and the manifest against the digest it
   was told), moves the directory into its local store's ``policy-releases`` and
   writes ``LATEST.json`` last, so its engine sampler loads the release exactly
   as it loads a trainer-written one.

    python -m yeto.rl.spot_rollout.cut_manifest make --store S --release r2-v2 --out M.json
    python -m yeto.rl.spot_rollout.cut_manifest place --manifest M.json --digest SHA \\
        --staged DIR --store LOCAL_STORE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "yeto.s29.cut-manifest/v1"
RELEASES = "policy-releases"
LATEST = "LATEST.json"
CHUNK = 8 << 20


class ManifestError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def encode(manifest: Mapping[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


def digest(manifest: Mapping[str, Any]) -> str:
    """The sha256 the control plane carries (of the canonical encoding)."""
    return hashlib.sha256(encode(manifest)).hexdigest()


def make(store: Path, release: str, *, record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The manifest of ``store/policy-releases/<release>``. The release record
    is ``record`` or else the store's LATEST.json, which must name this release
    (the mirror runs after the trainer published it)."""
    root = Path(store) / RELEASES
    directory = root / release
    if not directory.is_dir():
        raise ManifestError(f"no release directory {directory}")
    record = dict(record) if record is not None else json.loads((root / LATEST).read_text())
    if record.get("path") != release:
        raise ManifestError(f"{root / LATEST} names {record.get('path')!r}, not {release!r}")
    files = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        name = path.relative_to(directory).as_posix()
        files[name] = {"size": path.stat().st_size, "sha256": file_sha256(path)}
    missing = sorted(set(record.get("files") or {}) - set(files))
    if missing:
        raise ManifestError(f"release {release} lacks recorded files {missing[:5]}")
    return {"schema": SCHEMA, "release": release, "version": record.get("version"),
            "seq": record.get("seq"), "base": (record.get("delta") or {}).get("base_path"),
            "bytes": sum(f["size"] for f in files.values()), "files": files, "record": record}


def check(manifest: Mapping[str, Any], expected_digest: str | None) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ManifestError(f"not a cut manifest: {manifest.get('schema')!r}")
    if expected_digest is not None and digest(manifest) != expected_digest:
        raise ManifestError("manifest digest does not match the control-plane digest")


def verify(manifest: Mapping[str, Any], staged: Path) -> None:
    """Every manifest file is in ``staged`` with its size and sha256; no extra files."""
    staged = Path(staged)
    present = {p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()}
    want = manifest["files"]
    if present != set(want):
        raise ManifestError(f"staged files differ: missing {sorted(set(want) - present)[:5]}, "
                            f"extra {sorted(present - set(want))[:5]}")
    for name, meta in want.items():
        path = staged / name
        if path.stat().st_size != int(meta["size"]):
            raise ManifestError(f"{name}: size {path.stat().st_size} != {meta['size']}")
        if file_sha256(path) != meta["sha256"]:
            raise ManifestError(f"{name}: sha256 mismatch")


def check_release_name(name: Any, what: str = "release") -> str:
    """A release directory name: one path component, never ``.``/``..`` or a separator."""
    if (not isinstance(name, str) or not name or name in (".", "..") or ".." in name
            or "/" in name or "\\" in name or os.sep in name or (os.altsep and os.altsep in name)
            or "\0" in name):
        raise ManifestError(f"manifest {what} {name!r} is not a plain release directory name")
    return name


def _order(record: Mapping[str, Any]) -> tuple[int, int] | None:
    """(seq, version) of a release record; None when neither is a number."""
    def num(value: Any) -> int | None:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    seq, version = num(record.get("seq")), num(record.get("version"))
    if seq is None and version is None:
        return None
    return (-1 if seq is None else seq, -1 if version is None else version)


def _newer_than_latest(record: Mapping[str, Any], latest: Path) -> bool:
    """``record`` is newer (by seq, then version) than the store's current LATEST."""
    try:
        current = json.loads(latest.read_text())
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return True  # an unreadable LATEST is replaced
    if not isinstance(current, dict):
        return True
    mine, theirs = _order(record), _order(current)
    if theirs is None:
        return True
    if mine is None:
        return False
    return mine > theirs


def _unique() -> str:
    """A temporary-name suffix no other writer (process or thread) shares."""
    return f"{os.getpid()}.{uuid.uuid4().hex[:12]}"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(top: Path) -> None:
    """fsync every file, then every directory bottom-up (so each entry is durable)."""
    for dirpath, _dirs, files in os.walk(top, topdown=False):
        for name in files:
            fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(dirpath))


def _sweep_dead_staging(root: Path, release: str) -> None:
    """Remove ``.<release>.placing.<pid>.<id>`` left by a writer that is gone."""
    prefix = f".{release}.placing."
    for entry in root.iterdir():
        if not entry.name.startswith(prefix):
            continue
        pid = entry.name[len(prefix):].split(".", 1)[0]
        if not pid.isdigit() or _alive(int(pid)):
            continue
        shutil.rmtree(entry, ignore_errors=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OverflowError):
        return True
    return True


def place(manifest: Mapping[str, Any], staged: Path, store: Path, *,
          expected_digest: str | None) -> Path:
    """Verify, move ``staged`` to ``store/policy-releases/<release>`` and write
    LATEST.json last. The release's delta base must already be in the store.
    Idempotent: a release already placed with the same files is kept. LATEST.json
    is replaced only by a release newer (seq, then version) than the one it names,
    so placing an older manifest again never rolls the store back."""
    check(manifest, expected_digest)
    check_release_name(manifest.get("release"))
    root = Path(store) / RELEASES
    root.mkdir(parents=True, exist_ok=True)
    base = manifest.get("base")
    if base:
        check_release_name(base, "delta base")
    if base and not (root / base).is_dir():
        raise ManifestError(f"delta base {base} is not in {root}")
    target = root / manifest["release"]
    if target.is_dir():
        verify(manifest, target)
    else:
        verify(manifest, Path(staged))
        _sweep_dead_staging(root, manifest["release"])
        tmp = root / f".{manifest['release']}.placing.{_unique()}"
        try:
            shutil.move(str(staged), tmp)
            _fsync_tree(tmp)  # every byte durable before the name says the release is there
            try:
                os.replace(tmp, target)
            except OSError:
                if not target.is_dir():
                    raise
                verify(manifest, target)  # another writer placed it first; keep theirs
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        _fsync_dir(root)
    if _newer_than_latest(manifest["record"], root / LATEST):
        latest = root / f".{LATEST}.{_unique()}.tmp"
        try:
            with open(latest, "w") as out:
                out.write(json.dumps(manifest["record"], sort_keys=True))
                out.flush()
                os.fsync(out.fileno())
            os.replace(latest, root / LATEST)
        finally:
            latest.unlink(missing_ok=True)
        _fsync_dir(root)
    return target


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cut_manifest")
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--store", required=True)
    m.add_argument("--release", required=True)
    m.add_argument("--out", required=True)
    q = sub.add_parser("place")
    q.add_argument("--manifest", required=True)
    q.add_argument("--digest", required=True)
    q.add_argument("--staged", required=True)
    q.add_argument("--store", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "make":
            manifest = make(Path(a.store), a.release)
            Path(a.out).write_bytes(encode(manifest))
            print(f"{digest(manifest)} {a.out} files={len(manifest['files'])} bytes={manifest['bytes']}")
        else:
            manifest = json.loads(Path(a.manifest).read_text())
            target = place(manifest, Path(a.staged), Path(a.store), expected_digest=a.digest)
            print(f"placed {target} version={manifest['version']}")
    except ManifestError as error:
        print(f"[cut-manifest] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
