#!/usr/bin/env python3
"""Emit a node-local, loss-blind v13b checkout/GPU preflight proof."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import tonight85_common as common


REPO = Path("/root/yeto-v13b")
RESULT_ROOT = Path("/root/yeto-results-v13b")
RESULT_TARGET = Path("/data/yeto-results-v13b")
TARGETS = {"h200-n1": (6, 7), "h200-n2": tuple(range(8))}


def run(*command: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-label", choices=tuple(TARGETS), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors = []
    try:
        head = run("git", "-C", str(REPO), "rev-parse", "HEAD")
        dirty = run("git", "-C", str(REPO), "status", "--porcelain=v1", "--untracked-files=all")
        if head != args.source_commit:
            errors.append("isolated checkout commit mismatch")
        if dirty:
            errors.append("isolated checkout is dirty")
    except Exception as exc:
        head = None
        errors.append(f"checkout audit failed: {exc}")
    if RESULT_ROOT.exists() or RESULT_ROOT.is_symlink() or RESULT_TARGET.exists():
        errors.append("fresh v13b result root/target already exists before authority")

    inventory = {}
    uuid_to_index = {}
    try:
        for line in run("nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits").splitlines():
            index, gpu_uuid, name, memory = [part.strip() for part in line.split(",", 3)]
            inventory[int(index)] = {"uuid": gpu_uuid, "name": name, "memory_used_mib": int(float(memory))}
            uuid_to_index[gpu_uuid] = int(index)
    except Exception as exc:
        errors.append(f"GPU inventory failed: {exc}")
    compute = []
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) == 4:
                compute.append({"gpu_uuid": parts[0], "gpu_index": uuid_to_index.get(parts[0]), "pid": parts[1], "process_name": parts[2], "used_memory_mib": parts[3]})
    else:
        errors.append("GPU compute-process query failed")
    targets = TARGETS[args.node_label]
    occupied = [record for record in compute if record.get("gpu_index") in targets]
    if occupied:
        errors.append(f"target GPU has compute process: {occupied}")
    for gpu in targets:
        item = inventory.get(gpu)
        if not item or "H200" not in item["name"].upper():
            errors.append(f"target GPU {gpu} is absent or not H200")
        elif item["memory_used_mib"] > 64:
            errors.append(f"target GPU {gpu} has {item['memory_used_mib']} MiB allocated")
    proof = {
        "schema": "yeto_v13b_node_preflight_v1",
        "status": "PASS" if not errors else "FAIL",
        "checked_at_utc": common.utc_now(),
        "node": args.node_label,
        "source_commit": args.source_commit,
        "manifest_sha256": args.manifest_sha256,
        "isolated_checkout": str(REPO),
        "checkout_head": head,
        "fresh_result_root_absent": not (RESULT_ROOT.exists() or RESULT_ROOT.is_symlink()),
        "fresh_result_target_absent": not RESULT_TARGET.exists(),
        "target_gpus": list(targets),
        "reserved_v11_gpus": list(range(6)) if args.node_label == "h200-n1" else [],
        "gpu_inventory": inventory,
        "compute_processes": compute,
        "target_compute_processes": occupied,
        "errors": errors,
    }
    common.write_json_atomic(args.output, proof)
    print(json.dumps(proof, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
