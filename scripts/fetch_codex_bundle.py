#!/usr/bin/env python3
"""Fetch and attest the pinned stock Codex Linux artifact (design R-SCOPE).

Reproducible, pin-driven: every expected digest comes from ``yeto.rl``
(``CODEX_NPM_PACKAGE``, ``CODEX_NPM_TARBALL_SHA256``, ``CODEX_LINUX_BINARY_SHA256``,
``CODEX_LINUX_BINARY_SIZE_BYTES``, ``CODEX_PACKAGE_MANIFEST_SHA256``,
``CODEX_APP_SERVER_SCHEMA_SHA256``).  Steps:

1. ``npm pack <CODEX_NPM_PACKAGE>`` (or ``--tarball`` to reuse a download) and
   check the tarball sha256.
2. Extract ``package/vendor/<target>/bin/codex`` and
   ``package/vendor/<target>/codex-package.json``; check size + sha256.
3. Run ``codex --version`` and ``codex app-server generate-json-schema
   --experimental`` (the same command ``yeto.rl.adapters.miles.island_entry._preflight_codex_harness``
   runs in the container) and write the v2 schema into the bundle.  Codex
   serialises the schema with unstable key order, so the file is written in
   canonical form (``sort_keys=True, indent=2``) and compared against the pin
   by sha256; a mismatch is reported (exit 3) unless ``--allow-schema-drift``.

Output layout (``--out``, default ``~/work/codex-bundle``), the run asset dir
the launcher mounts at ``/opt/yeto/codex`` (``YETO_CODEX_BUNDLE_DIR``):

    <out>/npm/<tarball>.tgz
    <out>/codex/codex-x86_64-unknown-linux-musl
    <out>/codex/codex-package.json
    <out>/codex/codex_app_server_protocol.v2.schemas.json
    <out>/codex/BUNDLE-SHA256.json   (what was verified, with the pins)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yeto.rl import (  # noqa: E402
    CODEX_APP_SERVER_SCHEMA_SHA256,
    CODEX_CLI_VERSION,
    CODEX_LINUX_BINARY_SHA256,
    CODEX_LINUX_BINARY_SIZE_BYTES,
    CODEX_LINUX_TARGET,
    CODEX_NPM_PACKAGE,
    CODEX_NPM_TARBALL_SHA256,
    CODEX_PACKAGE_MANIFEST_SHA256,
)

BINARY_NAME = f"codex-{CODEX_LINUX_TARGET}"
MANIFEST_NAME = "codex-package.json"
SCHEMA_NAME = "codex_app_server_protocol.v2.schemas.json"
RECORD_NAME = "BUNDLE-SHA256.json"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npm_pack(package: str, into: Path, npm: str) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(
        [npm, "pack", package, "--pack-destination", str(into)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()
    name = out[-1].strip() if out else ""
    tarball = into / name
    if not name.endswith(".tgz") or not tarball.is_file():
        raise SystemExit(f"npm pack did not report a tarball (stdout: {out!r})")
    return tarball


def extract_member(tarball: Path, member: str, destination: Path) -> None:
    with tarfile.open(tarball, "r:gz") as archive:
        try:
            info = archive.getmember(member)
        except KeyError as exc:
            raise SystemExit(f"{member} missing from {tarball}") from exc
        if not info.isfile():
            raise SystemExit(f"{member} is not a regular file in {tarball}")
        source = archive.extractfile(info)
        assert source is not None
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            shutil.copyfileobj(source, handle)


def generate_schema(binary: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="yeto-codex-schema-") as temporary:
        out = Path(temporary) / "schema"
        subprocess.run(
            [str(binary), "app-server", "generate-json-schema", "--experimental", "--out", str(out)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "HOME": temporary},
        )
        with (out / SCHEMA_NAME).open(encoding="utf-8") as handle:
            return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.expanduser("~/work/codex-bundle"))
    parser.add_argument("--tarball", default=None, help="reuse an already downloaded npm tarball")
    parser.add_argument("--npm", default=shutil.which("npm") or "npm")
    parser.add_argument("--allow-schema-drift", action="store_true",
                        help="write the bundle even if the canonical schema sha256 differs from the pin")
    args = parser.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    codex_dir = out / "codex"
    tarball = Path(args.tarball).expanduser().resolve() if args.tarball else npm_pack(CODEX_NPM_PACKAGE, out / "npm", args.npm)
    tarball_sha = sha256_of(tarball)
    if tarball_sha != CODEX_NPM_TARBALL_SHA256:
        raise SystemExit(f"tarball sha256 {tarball_sha} != pin {CODEX_NPM_TARBALL_SHA256} ({tarball})")

    binary = codex_dir / BINARY_NAME
    manifest = codex_dir / MANIFEST_NAME
    extract_member(tarball, f"package/vendor/{CODEX_LINUX_TARGET}/bin/codex", binary)
    extract_member(tarball, f"package/vendor/{CODEX_LINUX_TARGET}/{MANIFEST_NAME}", manifest)
    binary.chmod(0o755)
    size = binary.stat().st_size
    if size != CODEX_LINUX_BINARY_SIZE_BYTES:
        raise SystemExit(f"binary size {size} != pin {CODEX_LINUX_BINARY_SIZE_BYTES}")
    binary_sha = sha256_of(binary)
    if binary_sha != CODEX_LINUX_BINARY_SHA256:
        raise SystemExit(f"binary sha256 {binary_sha} != pin {CODEX_LINUX_BINARY_SHA256}")
    manifest_sha = sha256_of(manifest)
    if manifest_sha != CODEX_PACKAGE_MANIFEST_SHA256:
        raise SystemExit(f"manifest sha256 {manifest_sha} != pin {CODEX_PACKAGE_MANIFEST_SHA256}")

    version = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True, timeout=30).stdout.strip()
    if version != CODEX_CLI_VERSION:
        raise SystemExit(f"codex --version {version!r} != pin {CODEX_CLI_VERSION!r}")

    schema_path = codex_dir / SCHEMA_NAME
    schema_path.write_text(json.dumps(generate_schema(binary), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    schema_sha = sha256_of(schema_path)
    schema_ok = schema_sha == CODEX_APP_SERVER_SCHEMA_SHA256

    record = {
        "npm_package": CODEX_NPM_PACKAGE,
        "tarball": str(tarball),
        "tarball_sha256": tarball_sha,
        "binary": str(binary),
        "binary_sha256": binary_sha,
        "binary_size_bytes": size,
        "cli_version": version,
        "package_manifest_sha256": manifest_sha,
        "app_server_schema_sha256": schema_sha,
        "app_server_schema_sha256_pin": CODEX_APP_SERVER_SCHEMA_SHA256,
        "app_server_schema_matches_pin": schema_ok,
    }
    (codex_dir / RECORD_NAME).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    if not schema_ok:
        print(
            f"WARNING: canonical app-server schema sha256 {schema_sha} != pin "
            f"{CODEX_APP_SERVER_SCHEMA_SHA256}; the container preflight "
            "(yeto.rl.adapters.miles.island_entry._preflight_codex_harness) will reject this bundle until the pin is re-derived",
            file=sys.stderr,
        )
        if not args.allow_schema_drift:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
