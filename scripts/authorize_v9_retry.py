#!/usr/bin/env python3
"""Create one loss-blind, whole-curve v9 retry authority."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import read_json, sha256_file, utc_now, write_json_atomic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage_1p7b", "stage_7b"), required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--retry-group-id", action="append", required=True)
    parser.add_argument("--failure-record", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing retry authority: {args.output}")
    manifest = read_json(args.manifest)
    allowed = set(manifest.get("retry_contract", {}).get("allowed_reasons", []))
    if args.reason not in allowed:
        raise SystemExit(f"unregistered retry reason {args.reason!r}")
    groups = args.retry_group_id
    if len(groups) != len(set(groups)):
        raise SystemExit("duplicate retry group")
    cells = [cell for cell in manifest["cells"] if cell["stage"] == args.stage]
    known = {cell["retry_group_id"] for cell in cells}
    if set(groups) - known:
        raise SystemExit("retry authority contains an unknown stage group")
    required_cell_ids = sorted(
        cell["cell_id"] for cell in cells if cell["retry_group_id"] in set(groups)
    )
    failure_records = []
    observed_cell_ids = set()
    for path in args.failure_record:
        value = read_json(path)
        if value.get("status") not in ("INFRA_FAILURE", "INVALID_WORK"):
            raise SystemExit(f"failure record is not retry-eligible: {path}")
        cell_id = value.get("cell_id")
        if cell_id not in required_cell_ids:
            raise SystemExit(f"failure record is outside selected groups: {cell_id}")
        observed_cell_ids.add(cell_id)
        failure_records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "cell_id": cell_id,
            }
        )
    # A whole curve is retried after any one of its cells has a loss-blind
    # infrastructure failure; successful attempt-1 outcomes in that curve are
    # ignored and replaced by attempt 2 exactly as preregistered.
    if not observed_cell_ids:
        raise SystemExit("retry authority has no bound failure evidence")
    authority = {
        "schema": "yeto_outer_mup_v9_retry_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_now(),
        "stage": args.stage,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "reason": args.reason,
        "retry_group_ids": groups,
        "retry_cell_ids": required_cell_ids,
        "failure_records": failure_records,
        "losses_inspected": False,
        "maximum_attempt_number": 2,
    }
    write_json_atomic(args.output.resolve(), authority)
    digest = sha256_file(args.output.resolve())
    sidecar.write_text(f"{digest}  {args.output.name}\n")
    print(
        json.dumps(
            {
                "authority": str(args.output.resolve()),
                "sha256": digest,
                "cells": len(required_cell_ids),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
