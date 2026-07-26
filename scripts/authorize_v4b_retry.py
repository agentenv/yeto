#!/usr/bin/env python3
"""Create a loss-blind whole-curve v4b retry authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--retry-group", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing retry authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if len(manifest.get("cells", [])) != 18:
        raise SystemExit("not the complete v4b manifest")
    allowed = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if args.reason not in allowed:
        raise SystemExit("retry reason is not registered")
    groups = list(dict.fromkeys(args.retry_group))
    if len(groups) != len(args.retry_group):
        raise SystemExit("duplicate retry group")
    known = {cell["retry_group_id"] for cell in manifest["cells"]}
    if set(groups) - known:
        raise SystemExit("unknown retry group")
    authority = {
        "schema": "yeto_outer_mup_v4b_retry_authority_v1",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "reason": args.reason,
        "retry_group_ids": groups,
        "selection_rule": "loss-blind whole two-new-eta curve-by-seed groups",
    }
    write_json_atomic(args.output.resolve(), authority)
    digest = sha256_file(args.output.resolve())
    sidecar.write_text(f"{digest}  {args.output.name}\n")
    print(json.dumps({"authority": str(args.output.resolve()), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
