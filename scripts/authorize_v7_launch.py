#!/usr/bin/env python3
"""Create a hash-bound v7 stage authority after all prerequisite proofs pass."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import v7_common as common
except ModuleNotFoundError:  # package import in tests
    from scripts import v7_common as common


STAGE_SECONDS = {"SMOKE": 90 * 60, "PILOT": 12 * 3600, "MAIN": 30 * 3600}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_pass(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text())
    if value.get("schema") != schema or value.get("status") != "PASS":
        raise SystemExit(f"required proof is not PASS: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_SECONDS, required=True)
    parser.add_argument("--node-proof", action="append", type=Path, required=True)
    parser.add_argument("--v6-drain-proof", type=Path, required=True)
    parser.add_argument("--smoke-evidence", type=Path)
    parser.add_argument("--pilot-readout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    manifest_sha = common.sha256_file(args.manifest)
    sidecar = args.manifest.with_suffix(args.manifest.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text().split()[0] != manifest_sha:
        raise SystemExit("manifest sidecar is absent or mismatched")
    drain = read_pass(args.v6_drain_proof, "yeto_outer_mup_v7_v6_drain_proof_v1")
    proof_hashes = {}
    for path in args.node_proof:
        proof = read_pass(path, "yeto_outer_mup_v7_node_authority_v1")
        node = proof.get("node")
        if node in proof_hashes or node not in common.NODES:
            raise SystemExit(f"duplicate or unknown node proof: {node!r}")
        if proof.get("manifest_sha256") != manifest_sha:
            raise SystemExit(f"node proof binds another manifest: {path}")
        proof_hashes[node] = common.sha256_file(path)
    if set(proof_hashes) != set(common.NODES):
        raise SystemExit("both node proofs are required")

    smoke_sha = None
    if args.stage == "PILOT":
        if args.smoke_evidence is None:
            raise SystemExit("PILOT authority requires --smoke-evidence")
        smoke = json.loads(args.smoke_evidence.read_text())
        smoke_cells = [
            cell for cell in manifest.get("cells", []) if cell.get("stage") == "SMOKE"
        ]
        if (
            len(smoke_cells) != 1
            or smoke.get("cell_id") != smoke_cells[0]["cell_id"]
            or smoke.get("status") != "COMPLETED"
        ):
            raise SystemExit("smoke evidence is not the registered completed smoke")
        smoke_sha = common.sha256_file(args.smoke_evidence)
    elif args.smoke_evidence is not None:
        raise SystemExit("--smoke-evidence is only valid for PILOT authority")

    pilot_sha = None
    if args.stage == "MAIN":
        if args.pilot_readout is None:
            raise SystemExit("MAIN authority requires --pilot-readout")
        pilot = read_pass(args.pilot_readout, "yeto_outer_mup_v7_pilot_readout_v1")
        registered = manifest.get("pilot", {})
        pilot_sha = common.sha256_file(args.pilot_readout)
        if registered.get("sha256") != pilot_sha:
            raise SystemExit("main manifest binds another pilot readout")
        if manifest.get("grid", {}).get("variant") != pilot.get(
            "selected_grid", {}
        ).get("variant"):
            raise SystemExit("main manifest grid variant differs from pilot")
    elif args.pilot_readout is not None:
        raise SystemExit("--pilot-readout is only valid for MAIN authority")

    started = time.time()
    authority = {
        "schema": "yeto_outer_mup_v7_launch_authority_v1",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "stage": args.stage,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "source_git_commit": manifest.get("source", {}).get("git_commit"),
        "node_proof_sha256": proof_hashes,
        "v6_drain_proof_sha256": common.sha256_file(args.v6_drain_proof),
        "v6_factorial_note_sha256": drain["factorial_note"]["sha256"],
        "smoke_evidence_sha256": smoke_sha,
        "pilot_readout_sha256": pilot_sha,
        "wall_clock_start_unix_s": started,
        "hard_deadline_unix_s": started + STAGE_SECONDS[args.stage],
        "duration_seconds": STAGE_SECONDS[args.stage],
    }
    common.write_json_atomic(args.output, authority)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "manifest_sha256": manifest_sha,
                "authority_sha256": common.sha256_file(args.output),
                "hard_deadline_unix_s": authority["hard_deadline_unix_s"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
