"""S29 D13: hand a policy release to a far spot island through a nearby object store.

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


def make(store: Path, release: str) -> dict[str, Any]:
    """The manifest of ``store/policy-releases/<release>``. The release record
    is the store's LATEST.json, which must name this release (the mirror runs
    after the trainer published it)."""
    root = Path(store) / RELEASES
    directory = root / release
    if not directory.is_dir():
        raise ManifestError(f"no release directory {directory}")
    record = json.loads((root / LATEST).read_text())
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


def place(manifest: Mapping[str, Any], staged: Path, store: Path, *,
          expected_digest: str | None) -> Path:
    """Verify, move ``staged`` to ``store/policy-releases/<release>`` and write
    LATEST.json last. The release's delta base must already be in the store.
    Idempotent: a release already placed with the same files is kept."""
    check(manifest, expected_digest)
    root = Path(store) / RELEASES
    root.mkdir(parents=True, exist_ok=True)
    base = manifest.get("base")
    if base and not (root / base).is_dir():
        raise ManifestError(f"delta base {base} is not in {root}")
    target = root / manifest["release"]
    if target.is_dir():
        verify(manifest, target)
    else:
        verify(manifest, Path(staged))
        tmp = root / f".{manifest['release']}.placing"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.move(str(staged), tmp)
        os.replace(tmp, target)
    latest = root / f".{LATEST}.tmp"
    latest.write_text(json.dumps(manifest["record"], sort_keys=True))
    os.replace(latest, root / LATEST)
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
