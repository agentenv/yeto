#!/usr/bin/env python3
"""Capture both H200 nodes' tmux/process/GPU state before v4b/v5 sharing."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


NODES = ("h200-n1", "h200-n2")
V4B_GPUS = tuple(range(6))
V5_INITIAL_GPUS = (6, 7)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote(node: str, command: str, *, allow_nonzero: bool = False) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"root@{node}", command],
        capture_output=True,
        text=True,
    )
    if result.returncode and not allow_nonzero:
        raise RuntimeError(
            f"{node}: {command!r} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def node_state(node: str) -> dict:
    gpu_lines = remote(
        node,
        "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits",
    ).splitlines()
    gpus = []
    errors = []
    for line in gpu_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            errors.append(f"malformed GPU inventory row: {line}")
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "memory_used_mib": int(parts[2]),
                "utilization_percent": int(parts[3]),
            }
        )
    compute = remote(
        node,
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
        "--format=csv,noheader,nounits",
    ).splitlines()
    slot_processes = remote(node, "pgrep -af '[r]un_slot' || true").splitlines()
    tmux = remote(node, "tmux list-sessions 2>/dev/null || true").splitlines()
    v4b_files = int(
        remote(
            node,
            "find -L /root/yeto-results-v4b -type f 2>/dev/null | wc -l",
        )
    )
    if sorted(item["index"] for item in gpus) != list(range(8)):
        errors.append("GPU indices are not exactly 0..7")
    if compute:
        errors.append(f"compute processes already occupy GPUs: {compute}")
    if any(item["memory_used_mib"] > 16 for item in gpus):
        errors.append("one or more GPUs use more than 16 MiB before partition claim")
    if slot_processes:
        errors.append(f"slot controllers already active: {slot_processes}")
    if v4b_files:
        errors.append(f"v4b result root already contains {v4b_files} files")
    return {
        "node": node,
        "captured_at_utc": utc_now(),
        "tmux_sessions": tmux,
        "slot_processes": slot_processes,
        "compute_processes": compute,
        "gpus": gpus,
        "v4b_result_file_count": v4b_files,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing snapshot: {args.output}")
    states = [node_state(node) for node in NODES]
    value = {
        "schema": "yeto_outer_mup_v4b_v5_slot_snapshot_v1",
        "captured_at_utc": utc_now(),
        "partition": {
            "v4b_initial": [
                {"node": node, "gpus": list(V4B_GPUS)} for node in NODES
            ],
            "v5_initial": [
                {"node": node, "gpus": list(V5_INITIAL_GPUS)} for node in NODES
            ],
            "handoff": "each GPU 0..5 starts its v5 queue only after its v4b queue controller exits",
        },
        "nodes": states,
        "status": "PASS" if all(item["status"] == "PASS" for item in states) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "status": value["status"]}))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
