#!/usr/bin/env python3
"""Authorize one loss-blind whole-seed-curve v4 retry wave."""

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
    parser.add_argument("--loss-blind-review", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing retry authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    allowed_reasons = set(manifest["retry_contract"]["allowed_reasons"])
    if args.reason not in allowed_reasons:
        raise SystemExit(f"unregistered retry reason: {args.reason}")
    groups = set(args.retry_group)
    if len(groups) != len(args.retry_group):
        raise SystemExit("duplicate retry group")
    by_group = {}
    for cell in manifest["cells"]:
        by_group.setdefault(cell["retry_group_id"], []).append(cell)
    unknown = groups - set(by_group)
    if unknown:
        raise SystemExit(f"unknown retry groups: {sorted(unknown)}")
    for group in groups:
        cells = by_group[group]
        if len(cells) != 4 or len({cell["eta_index"] for cell in cells}) != 4:
            raise SystemExit(f"retry group is not one complete four-eta curve: {group}")
    authority = {
        "schema": "yeto_outer_mup_v4_retry_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_now(),
        "manifest_sha256": sha256_file(args.manifest),
        "reason": args.reason,
        "retry_group_ids": sorted(groups),
        "loss_blind_review_attested": True,
        "attempt_number": 2,
        "selection_rule": "every registered eta in each named T/mu/training-seed curve; no finite outcome triggered selection",
    }
    write_json_atomic(args.output.resolve(), authority)
    print(json.dumps({"authority": str(args.output.resolve()), "groups": sorted(groups)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
