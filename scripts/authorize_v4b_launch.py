#!/usr/bin/env python3
"""Create immutable v4b launch authority after verifying the disclosed G4 result."""

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
    parser.add_argument("--g4-readout", type=Path, required=True)
    parser.add_argument("--slot-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing launch authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("schema")
        != "yeto_outer_mup_v4b_extension_launch_manifest_v1"
        or manifest.get("status") != "AUTHORIZED"
        or len(manifest.get("cells", [])) != 18
    ):
        raise SystemExit("manifest is not the complete authorized v4b extension")
    g4 = json.loads(args.g4_readout.read_text())
    expected_g4_sha = manifest.get("base_v4", {}).get("g4_readout_sha256")
    if sha256_file(args.g4_readout) != expected_g4_sha:
        raise SystemExit("G4 readout hash differs from the disclosed v4 outcome")
    if (
        g4.get("schema") != "yeto_outer_mup_v4_g4_readout_v1"
        or g4.get("gate", {}).get("verdict") != "NOT_EVALUABLE"
        or g4.get("observed_completed_cells") != 48
        or g4.get("evidence_errors")
    ):
        raise SystemExit("G4 readout does not establish the registered trigger")
    snapshot = json.loads(args.slot_snapshot.read_text())
    if (
        snapshot.get("schema") != "yeto_outer_mup_v4b_v5_slot_snapshot_v1"
        or snapshot.get("status") != "PASS"
    ):
        raise SystemExit("slot snapshot does not authorize collision-free sharing")
    started = time.time()
    deadline = started + CEILING_SECONDS
    authority = {
        "schema": "yeto_outer_mup_v4b_launch_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_from_timestamp(started),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registration_git_commit": manifest["registration"]["git_commit"],
        "g4_readout_path": str(args.g4_readout.resolve()),
        "g4_readout_sha256": expected_g4_sha,
        "slot_snapshot_path": str(args.slot_snapshot.resolve()),
        "slot_snapshot_sha256": sha256_file(args.slot_snapshot),
        "wall_clock_start_event": "immutable v4b launch authority creation immediately before coordinated slot launch",
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
