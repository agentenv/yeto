#!/usr/bin/env python3
"""Create the immutable v5 launch authority after the registered v4 drain gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


CEILING_SECONDS = 43_200


def utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v4-evidence-n1", type=int, required=True)
    parser.add_argument("--v4-evidence-n2", type=int, required=True)
    parser.add_argument("--v4-slots-n1", type=int, required=True)
    parser.add_argument("--v4-slots-n2", type=int, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing launch authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("schema") != "yeto_outer_mup_v5_snoo_launch_manifest_v1"
        or manifest.get("status") != "AUTHORIZED"
        or len(manifest.get("cells", [])) != 90
    ):
        raise SystemExit("manifest is not the complete authorized v5 grid")
    evidence_total = args.v4_evidence_n1 + args.v4_evidence_n2
    if evidence_total != 48:
        raise SystemExit(f"v4 evidence total is {evidence_total}, not 48")
    if args.v4_slots_n1 != 0 or args.v4_slots_n2 != 0:
        raise SystemExit("v4 slot controllers have not drained on both nodes")
    started = time.time()
    deadline = started + CEILING_SECONDS
    authority = {
        "schema": "yeto_outer_mup_v5_launch_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_from_timestamp(started),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registration_git_commit": manifest["registration"]["git_commit"],
        "v4_drain_evidence": {
            "captured_at_utc": utc_from_timestamp(started),
            "h200-n1": {
                "evidence_count": args.v4_evidence_n1,
                "run_slot_v4_process_count": args.v4_slots_n1,
            },
            "h200-n2": {
                "evidence_count": args.v4_evidence_n2,
                "run_slot_v4_process_count": args.v4_slots_n2,
            },
            "evidence_count_total": evidence_total,
            "required_total": 48,
            "required_process_count_each_node": 0,
            "status": "PASS",
        },
        "wall_clock_start_event": "v5 launch authority creation immediately before slot launch",
        "wall_clock_start_utc": utc_from_timestamp(started),
        "wall_clock_start_unix_s": started,
        "hard_deadline_utc": utc_from_timestamp(deadline),
        "hard_deadline_unix_s": deadline,
        "ceiling_seconds": CEILING_SECONDS,
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
