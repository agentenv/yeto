#!/usr/bin/env python3
"""Create the immutable v13b launch authority from two passing node proofs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tonight85_common as common


EXPECTED_SLOTS = {
    ("h200-n1", 6), ("h200-n1", 7),
    *(("h200-n2", gpu) for gpu in range(8)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proof", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing authority: {args.output}")
    manifest = common.read_json(args.manifest)
    if manifest.get("schema") != "yeto_v13b_launch_manifest_v1" or len(manifest.get("cells", [])) != 72:
        raise SystemExit("not the complete v13b manifest")
    source_commit = manifest["source"]["git_commit"]
    manifest_sha = common.sha256_file(args.manifest)
    slots = {(item["node"], int(item["gpu"])) for item in manifest["randomization"]["slots"]}
    if slots != EXPECTED_SLOTS:
        raise SystemExit(f"manifest target slots changed: {sorted(slots)}")
    proofs = [common.read_json(path) for path in args.proof]
    if {proof.get("node") for proof in proofs} != {"h200-n1", "h200-n2"}:
        raise SystemExit("need exactly one proof for each node")
    if len(proofs) != 2 or any(proof.get("status") != "PASS" for proof in proofs):
        raise SystemExit("both node proofs must pass")
    for proof in proofs:
        if proof.get("source_commit") != source_commit or proof.get("manifest_sha256") != manifest_sha:
            raise SystemExit("node proof binding mismatch")
        if not proof.get("fresh_result_root_absent") or not proof.get("fresh_result_target_absent"):
            raise SystemExit("v13b result path existed before authority")
        if proof.get("target_compute_processes"):
            raise SystemExit("node proof has occupied target GPU")
    authority = {
        "schema": "yeto_v13b_launch_authority_v1",
        "status": "AUTHORIZED",
        "authorized_at_utc": common.utc_now(),
        "source_git_commit": source_commit,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "contract_sha256": manifest["registration"]["contract"]["sha256"],
        "analysis_cutoff": manifest["analysis_cutoff"],
        "slots": [{"node": node, "gpu": gpu} for node, gpu in sorted(slots)],
        "node_proofs": [
            {"node": proof["node"], "path": str(path.resolve()), "sha256": common.sha256_file(path)}
            for proof, path in zip(proofs, args.proof)
        ],
        "loss_blind_attestation": (
            "target selection and launch authorization inspected process/input/integrity state only; "
            "no G13B endpoint exists and no comparative v13 arm estimand selected a grid"
        ),
    }
    common.write_json_atomic(args.output, authority)
    digest = common.sha256_file(args.output)
    sidecar.write_text(f"{digest}  {args.output.name}\n")
    print(json.dumps({"authority": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
