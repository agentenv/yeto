#!/usr/bin/env python3
"""Create immutable stage authority after v9 registration and fresh fleet proofs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import read_json, sha256_file, write_json_atomic  # noqa: E402


MAX_PROOF_AGE_SECONDS = 300
NODES = ("h200-n1", "h200-n2")


def utc_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def pushed_commit(manifest: dict) -> tuple[bool, str | None]:
    source = manifest["source"]
    branch = source["branch"]
    commit = source["git_commit"]
    result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
    )
    remote = (
        result.stdout.split()[0]
        if result.returncode == 0 and result.stdout.strip()
        else None
    )
    return remote == commit, remote


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage_1p7b", "stage_7b"), required=True)
    parser.add_argument("--node-proof", type=Path, action="append", required=True)
    parser.add_argument("--fleet-proof", type=Path, required=True)
    parser.add_argument("--parent-authority", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing launch authority: {args.output}")
    manifest = read_json(args.manifest)
    if (
        manifest.get("schema") != "yeto_outer_mup_v9_launch_manifest_v1"
        or manifest.get("stage") != "V9_SEALED_SCALE"
        or manifest.get("status") != "REGISTERED"
        or len(manifest.get("cells", [])) != 28
    ):
        raise SystemExit("manifest is not the complete registered v9 launch")
    if manifest.get("predictions", {}).get("verification_loss_seen") is not False:
        raise SystemExit("manifest does not bind a pre-verification prediction seal")
    pushed, remote = pushed_commit(manifest)
    if not pushed:
        raise SystemExit(
            f"registration commit is not the exact pushed branch tip: remote={remote}"
        )
    now = time.time()
    proofs = {}
    for path in args.node_proof:
        proof = read_json(path)
        node = proof.get("node")
        if node in proofs:
            raise SystemExit(f"duplicate node proof: {node}")
        checked = proof.get("checked_at_unix_s")
        if (
            proof.get("schema") != "yeto_outer_mup_v9_node_preflight_v1"
            or proof.get("status") != "PASS"
            or proof.get("stage") != args.stage
            or proof.get("manifest_sha256") != sha256_file(args.manifest)
            or proof.get("errors")
            or not isinstance(checked, (int, float))
            or not 0 <= now - checked <= MAX_PROOF_AGE_SECONDS
        ):
            raise SystemExit(f"node proof is stale, mismatched, or not PASS: {path}")
        proofs[node] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "checked_at_utc": proof.get("checked_at_utc"),
        }
    if set(proofs) != set(NODES):
        raise SystemExit(f"node proofs must cover exactly {NODES}")
    fleet = read_json(args.fleet_proof)
    fleet_checked = fleet.get("checked_at_unix_s")
    if (
        fleet.get("schema") != "yeto_outer_mup_v9_fleet_gate_v1"
        or fleet.get("status") != "PASS"
        or fleet.get("stage") != args.stage
        or fleet.get("manifest_sha256") != sha256_file(args.manifest)
        or fleet.get("errors")
        or not isinstance(fleet_checked, (int, float))
        or not 0 <= now - fleet_checked <= MAX_PROOF_AGE_SECONDS
    ):
        raise SystemExit("fleet proof is stale, mismatched, or not PASS")

    if args.stage == "stage_1p7b":
        if args.parent_authority is not None:
            raise SystemExit("stage_1p7b cannot have a parent authority")
        ceiling = int(manifest["wall_clock"]["ceiling_seconds"])
        start = now
        deadline = now + ceiling
        parent = None
    else:
        if args.parent_authority is None:
            raise SystemExit("stage_7b requires the immutable stage_1p7b authority")
        parent_value = read_json(args.parent_authority)
        if (
            parent_value.get("schema") != "yeto_outer_mup_v9_launch_authority_v1"
            or parent_value.get("status") != "AUTHORIZED"
            or parent_value.get("stage") != "stage_1p7b"
            or parent_value.get("manifest_sha256") != sha256_file(args.manifest)
        ):
            raise SystemExit("parent authority is invalid")
        start = float(parent_value["program_wall_start_unix_s"])
        deadline = float(parent_value["hard_deadline_unix_s"])
        if now >= deadline:
            raise SystemExit("v9 program deadline elapsed before 7B authority")
        parent = {
            "path": str(args.parent_authority.resolve()),
            "sha256": sha256_file(args.parent_authority),
        }
    authorized_slots = sorted(
        {
            (cell["assignment"]["node"], cell["slot_id"])
            for cell in manifest["cells"]
            if cell["stage"] == args.stage
        }
    )
    authority = {
        "schema": "yeto_outer_mup_v9_launch_authority_v1",
        "status": "AUTHORIZED",
        "stage": args.stage,
        "authorized_at_utc": utc_from_timestamp(now),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registration_git_commit": manifest["registration"]["git_commit"],
        "registration_branch": manifest["source"]["branch"],
        "registration_exact_pushed_tip": True,
        "predictions_sha256": manifest["predictions"]["sha256"],
        "prediction_seal_precedes_authority": True,
        "node_preflights": proofs,
        "fleet_proof": {
            "path": str(args.fleet_proof.resolve()),
            "sha256": sha256_file(args.fleet_proof),
            "checked_at_utc": fleet.get("checked_at_utc"),
            "prerequisite": fleet.get("prerequisite"),
        },
        "parent_authority": parent,
        "authorized_slots": [
            {"node": node, "slot_id": slot_id} for node, slot_id in authorized_slots
        ],
        "program_wall_start_utc": utc_from_timestamp(start),
        "program_wall_start_unix_s": start,
        "hard_deadline_utc": utc_from_timestamp(deadline),
        "hard_deadline_unix_s": deadline,
        "ceiling_seconds": int(deadline - start),
    }
    write_json_atomic(args.output.resolve(), authority)
    digest = sha256_file(args.output.resolve())
    sidecar.write_text(f"{digest}  {args.output.name}\n")
    print(
        json.dumps(
            {
                "authority": str(args.output.resolve()),
                "sha256": digest,
                "stage": args.stage,
                "hard_deadline_utc": authority["hard_deadline_utc"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
