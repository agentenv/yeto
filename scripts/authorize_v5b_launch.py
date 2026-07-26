#!/usr/bin/env python3
"""Create immutable v5b authority from two fresh free-GPU node proofs."""

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
V4B_MANIFEST_SHA256 = (
    "f2abf80d975572dde33ee2c750c1fb91598df8bfea5a78696bdc2c5d3608b55b"
)


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
    parser.add_argument("--node-proof", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing launch authority: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("schema") != "yeto_outer_mup_v5b_snoo_regrid_launch_manifest_v1"
        or manifest.get("status") != "AUTHORIZED"
        or len(manifest.get("cells", [])) != 75
    ):
        raise SystemExit("manifest is not the complete authorized v5b grid")
    now = time.time()
    proofs = {}
    proof_hashes = {}
    for path in args.node_proof:
        proof = json.loads(path.read_text())
        node = proof.get("node")
        if node in proofs:
            raise SystemExit(f"duplicate node proof: {node}")
        if proof.get("schema") != "yeto_outer_mup_v5b_node_authority_v1":
            raise SystemExit(f"invalid node proof schema: {path}")
        if proof.get("status") != "PASS" or proof.get("errors"):
            raise SystemExit(f"node proof is not a clean pass: {path}")
        if proof.get("manifest_sha256") != sha256_file(args.manifest):
            raise SystemExit(f"node proof binds another manifest: {path}")
        checked = proof.get("checked_at_unix_s")
        if not isinstance(checked, (int, float)) or not 0 <= now - checked <= MAX_PROOF_AGE_SECONDS:
            raise SystemExit(f"node proof is stale or future-dated: {path}")
        if proof.get("target_gpus") != [6, 7]:
            raise SystemExit(f"node proof does not reserve exactly GPUs 6-7: {path}")
        for gpu in ("6", "7"):
            state = proof.get("target_gpu_state", {}).get(gpu, {})
            if state.get("compute_processes"):
                raise SystemExit(f"node proof records occupied GPU {gpu}: {path}")
        v4b = proof.get("v4b_disjointness", {})
        if (
            v4b.get("observed_manifest_sha256") != V4B_MANIFEST_SHA256
            or v4b.get("target_gpu_collisions")
        ):
            raise SystemExit(f"node proof lacks v4b disjointness: {path}")
        proofs[node] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "checked_at_utc": proof.get("checked_at_utc"),
        }
        proof_hashes[node] = sha256_file(path)
    if set(proofs) != {"h200-n1", "h200-n2"}:
        raise SystemExit(f"proofs must cover exactly both nodes, got {sorted(proofs)}")

    deadline = now + CEILING_SECONDS
    authority = {
        "schema": "yeto_outer_mup_v5b_launch_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_from_timestamp(now),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "registration_git_commit": manifest["registration"]["git_commit"],
        "node_proofs": proofs,
        "node_proof_sha256": proof_hashes,
        "authorized_slots": [
            {"node": node, "gpu": gpu}
            for gpu in (6, 7)
            for node in ("h200-n1", "h200-n2")
        ],
        "v4b_disjoint_manifest_sha256": V4B_MANIFEST_SHA256,
        "wall_clock_start_event": (
            "v5b launch authority creation immediately before four free-slot queues"
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
    print(
        json.dumps(
            {"authority": str(args.output.resolve()), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
