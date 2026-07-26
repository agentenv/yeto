#!/usr/bin/env python3
"""Authorize one loss-blind whole-curve v5b infrastructure retry wave."""

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
    if (
        manifest.get("schema") != "yeto_outer_mup_v5b_snoo_regrid_launch_manifest_v1"
        or len(manifest.get("cells", [])) != 75
    ):
        raise SystemExit("not the complete v5b launch manifest")
    allowed_reasons = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if args.reason not in allowed_reasons:
        raise SystemExit(f"reason is not registered: {args.reason}")
    groups = args.retry_group
    if len(groups) != len(set(groups)):
        raise SystemExit("duplicate retry group")
    known_groups = {cell["retry_group_id"] for cell in manifest["cells"]}
    unknown = set(groups) - known_groups
    if unknown:
        raise SystemExit(f"unknown retry groups: {sorted(unknown)}")
    group_sizes = {
        group: sum(cell["retry_group_id"] == group for cell in manifest["cells"])
        for group in groups
    }
    if set(group_sizes.values()) != {5}:
        raise SystemExit(f"retry groups are not complete five-eta curves: {group_sizes}")
    authority = {
        "schema": "yeto_outer_mup_v5b_retry_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_now(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "reason": args.reason,
        "retry_group_ids": sorted(groups),
        "group_sizes": group_sizes,
        "loss_blind_attestation": (
            "authorization is based only on enumerated infrastructure evidence; "
            "every eta in each condition-by-seed group is retried"
        ),
        "attempt_number": 2,
    }
    write_json_atomic(args.output.resolve(), authority)
    digest = sha256_file(args.output.resolve())
    sidecar.write_text(f"{digest}  {args.output.name}\n")
    print(
        json.dumps(
            {"authority": str(args.output.resolve()), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
