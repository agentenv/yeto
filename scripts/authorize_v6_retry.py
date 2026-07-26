#!/usr/bin/env python3
"""Issue a hash-bound loss-blind whole-curve v6 retry authority."""

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
    parser.add_argument("--retry-group-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "yeto_outer_mup_v6_launch_manifest_v1":
        raise SystemExit("manifest schema mismatch")
    allowed = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if args.reason not in allowed:
        raise SystemExit(f"retry reason {args.reason!r} is not registered")
    groups = args.retry_group_id
    if len(set(groups)) != len(groups):
        raise SystemExit("retry groups are duplicated")
    known = {cell["retry_group_id"] for cell in manifest.get("cells", [])}
    unknown = set(groups) - known
    if unknown:
        raise SystemExit(f"unknown retry groups: {sorted(unknown)}")
    grouped = {}
    for group in groups:
        selected = [cell for cell in manifest["cells"] if cell["retry_group_id"] == group]
        if len(selected) != 5 or sorted(cell["eta_index"] for cell in selected) != list(range(5)):
            raise SystemExit(f"retry group {group} is not the complete five-eta curve")
        grouped[group] = [cell["cell_id"] for cell in selected]
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing retry authority")
    authority = {
        "schema": "yeto_outer_mup_v6_retry_authority_v1",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "reason": args.reason,
        "retry_group_ids": groups,
        "cells_by_group": grouped,
        "loss_blind_attestation": "groups selected only from infrastructure status/evidence, before inspecting endpoint losses",
    }
    write_json_atomic(args.output.resolve(), authority)
    print(json.dumps(authority, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
