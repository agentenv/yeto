#!/usr/bin/env python3
"""Create immutable v4c launch authority from G4B and fresh initial claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


CEILING_SECONDS = 43_200
MAX_PROOF_AGE_SECONDS = 300
G4B_READOUT_SHA256 = "d58a05c46396d94786c6bcdcffa4f9c72abcc47036652a5260ee87256446be97"


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
    parser.add_argument("--g4b-readout", type=Path, required=True)
    parser.add_argument("--node-proof", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing launch authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("schema")
        != "yeto_outer_mup_v4c_seedpower_launch_manifest_v1"
        or manifest.get("stage") != "V4C_SEED_POWER"
        or manifest.get("status") != "AUTHORIZED"
        or len(manifest.get("cells", [])) != 44
    ):
        raise SystemExit("manifest is not the complete authorized v4c stage")
    if sha256_file(args.g4b_readout) != G4B_READOUT_SHA256:
        raise SystemExit("G4B readout differs from the preregistered trigger")
    g4b = json.loads(args.g4b_readout.read_text())
    if (
        g4b.get("schema") != "yeto_outer_mup_v4b_g4b_readout_v1"
        or g4b.get("gate", {}).get("verdict") != "NOT_EVALUABLE"
        or g4b.get("bootstrap", {}).get("valid_replicates") != 7067
        or g4b.get("evidence_errors")
        or not all(fit.get("interior") for fit in g4b.get("curve_fits", []))
    ):
        raise SystemExit("G4B readout does not establish the seed-power trigger")

    now = time.time()
    proofs = {}
    for path in args.node_proof:
        proof = json.loads(path.read_text())
        node = proof.get("node")
        if node in proofs:
            raise SystemExit(f"duplicate node proof: {node}")
        if (
            proof.get("schema") != "yeto_outer_mup_v4c_node_claim_v1"
            or proof.get("status") != "PASS"
            or proof.get("errors")
        ):
            raise SystemExit(f"node proof is not a clean pass: {path}")
        if proof.get("manifest_sha256") != sha256_file(args.manifest):
            raise SystemExit(f"node proof binds another manifest: {path}")
        checked = proof.get("checked_at_unix_s")
        if not isinstance(checked, (int, float)) or not 0 <= now - checked <= MAX_PROOF_AGE_SECONDS:
            raise SystemExit(f"node proof is stale or future-dated: {path}")
        if proof.get("target_gpus") != list(range(6)):
            raise SystemExit(f"initial proof does not claim exactly GPUs 0-5: {path}")
        for gpu in map(str, range(6)):
            if proof.get("target_gpu_state", {}).get(gpu, {}).get("compute_processes"):
                raise SystemExit(f"initial proof records occupied GPU {gpu}: {path}")
        proofs[node] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "checked_at_utc": proof.get("checked_at_utc"),
            "v5b_process_count_on_deferred_slots": len(proof.get("v5b_processes", [])),
        }
    if set(proofs) != {"h200-n1", "h200-n2"}:
        raise SystemExit(f"proofs must cover exactly both nodes, got {sorted(proofs)}")

    deadline = now + CEILING_SECONDS
    authority = {
        "schema": "yeto_outer_mup_v4c_launch_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_from_timestamp(now),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registration_git_commit": manifest["registration"]["git_commit"],
        "g4b_readout_path": str(args.g4b_readout.resolve()),
        "g4b_readout_sha256": G4B_READOUT_SHA256,
        "initial_node_claims": proofs,
        "initial_authorized_slots": [
            {"node": node, "gpu": gpu}
            for gpu in range(6)
            for node in ("h200-n1", "h200-n2")
        ],
        "deferred_slots": [
            {"node": node, "gpu": gpu}
            for gpu in (6, 7)
            for node in ("h200-n1", "h200-n2")
        ],
        "deferred_condition": (
            "Each deferred controller requires a fresh PASS claim covering GPUs "
            "6-7 on its node after all v5b controllers and compute processes drain."
        ),
        "wall_clock_start_event": (
            "immutable v4c launch authority creation immediately before twelve "
            "initial longest-first queues"
        ),
        "wall_clock_start_utc": utc_from_timestamp(now),
        "wall_clock_start_unix_s": now,
        "hard_deadline_utc": utc_from_timestamp(deadline),
        "hard_deadline_unix_s": deadline,
        "ceiling_seconds": CEILING_SECONDS,
    }
    write_json_atomic(args.output.resolve(), authority)
    digest = sha256_file(args.output.resolve())
    sidecar.write_text(f"{digest}  {args.output.name}\n")
    print(json.dumps({"authority": str(args.output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
