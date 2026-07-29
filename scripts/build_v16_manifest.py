#!/usr/bin/env python3
"""Build the loss-blind 612-cell V16 second-family launch manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import day3_common as common


REGISTRATION_COMMIT = "9bd1ccc43172e4b72c3b252fbde7526470e8a56f"
REGISTRATION_PATH = Path("experiment-specs/outer-mup-v16-pythia-redesign-prereg.json")
BRANCH = "experiment/day3-fleet"
V16_SLOTS = (
    ("h200-n1", 5),
    ("h200-n2", 4),
    ("h200-n1", 6),
    ("h200-n2", 5),
    ("h200-n1", 7),
    ("h200-n2", 6),
    ("h200-n2", 7),
)


def remote_branch_tip() -> str:
    result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"],
        cwd=common.REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    fields = result.stdout.split()
    if len(fields) != 2:
        raise SystemExit(f"cannot resolve public {BRANCH} tip")
    return fields[0]


def file_record(relative: str) -> dict[str, object]:
    path = common.REPO / relative
    if not path.is_file():
        raise SystemExit(f"required V16 execution file is missing: {path}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}


def validate_inputs(inputs: dict, raw_sha256: str) -> None:
    if raw_sha256 != "b77b353dcc8a548c8e4ff31a252ed74919f325482795a15732b4ae9261b08164":
        raise SystemExit("V16 input manifest is not the frozen V13-family input proof")
    if inputs.get("schema") != "yeto_tonight85_v13_ultrachat_inputs_v1":
        raise SystemExit("V16 input proof schema mismatch")
    if inputs.get("status") != "FROZEN":
        raise SystemExit("V16 input proof is not frozen")
    if inputs.get("model", {}).get("canonical_inventory_sha256") != (
        "49840eba946d992e68bded1e7dc3688a8b0186416bd4e2c9b27026d602699b19"
    ):
        raise SystemExit("V16 Pythia model inventory mismatch")
    if inputs.get("files", {}).get("train", {}).get("sha256") != (
        "f9bcb68e84370667cfe2418450c9fcd112cd0cc9236936e48e3aa35f9dd27ace"
    ):
        raise SystemExit("V16 UltraChat train hash mismatch")
    if inputs.get("files", {}).get("eval", {}).get("sha256") != (
        "08ac254552fbd6529c68577c94920e4f33f6e75b5c94c518350c18948ec48df4"
    ):
        raise SystemExit("V16 UltraChat eval hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--syncer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.syncer_sha256) != 64:
        raise SystemExit("syncer SHA-256 must have 64 hex characters")
    head = common.git("rev-parse", "HEAD")
    if remote_branch_tip() != head:
        raise SystemExit("V16 implementation/manifest commit is not public")
    if subprocess.run(
        ["git", "-C", str(common.REPO), "merge-base", "--is-ancestor", REGISTRATION_COMMIT, head]
    ).returncode:
        raise SystemExit("V16 registration commit is not an execution ancestor")
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("tracked worktree changes forbid a V16 manifest")
    inputs = common.read_json(args.input_manifest)
    input_sha = common.sha256_file(args.input_manifest)
    validate_inputs(inputs, input_sha)
    contract = common.read_json(common.REPO / REGISTRATION_PATH)
    seeds = tuple(contract["seeds_and_cells"]["fresh_seeds"])
    if len(seeds) != 17 or contract["seeds_and_cells"]["cell_count"] != 612:
        raise SystemExit("V16 machine contract cardinality mismatch")

    registration = file_record(str(REGISTRATION_PATH))
    registration["git_commit"] = REGISTRATION_COMMIT
    execution_paths = (
        "scripts/day3_common.py",
        "scripts/build_v16_manifest.py",
        "scripts/run_day3_queue.py",
        "scripts/day3_status.py",
        "scripts/day3_conductor.py",
        "scripts/analyze_v16.py",
        "scripts/analyze_v19.py",
        "scripts/run_slot_v3.py",
        "scripts/compare_diloco.py",
        "syncer/src/merge.rs",
        "syncer/src/main.rs",
        "syncer/src/state.rs",
    )
    execution_files = {path: file_record(path) for path in execution_paths}

    queue_records = [
        {
            "queue_id": f"v16-{node}-gpu{gpu}",
            "node": node,
            "gpu": gpu,
            "scientific_cells": 0,
            "retry_groups": [],
        }
        for node, gpu in V16_SLOTS
    ]
    cells = []
    slot_offsets = [0] * len(V16_SLOTS)
    group_index = 0
    curves = contract["grid_construction"]["curves"]
    # T20 has the most sync overhead and was the gatesim-limiting curve, so it
    # is scheduled first. This order is fixed without inspecting V16 outcomes.
    for t in (20, 5, 2):
        h = 2560 // t
        for arm in ("mu0", "mu09"):
            etas = curves[f"T{t}_{arm}"]["etas"]
            if len(etas) != 6:
                raise SystemExit(f"V16 T{t}/{arm} grid is not six-rung")
            for seed in seeds:
                slot_index = group_index % len(V16_SLOTS)
                node, gpu = V16_SLOTS[slot_index]
                queue_id = queue_records[slot_index]["queue_id"]
                retry_group = f"v16-t{t:02d}-{arm}-seed{seed}"
                queue_records[slot_index]["retry_groups"].append(retry_group)
                for eta_index, eta in enumerate(etas):
                    cell = {
                        "program": "v16",
                        "stage": "v16",
                        "cell_id": f"v16-t{t:02d}-{arm}-e{eta_index}-seed{seed}",
                        "queue_id": queue_id,
                        "slot_id": f"{node}-gpu{gpu}",
                        "slot_queue_index": slot_offsets[slot_index],
                        "retry_group_id": retry_group,
                        "assignment": {"node": node, "gpus": [gpu]},
                        "arm": arm,
                        "outer_optimizer": "nesterov",
                        "mu": 0.0 if arm == "mu0" else 0.9,
                        "eta": float(eta),
                        "eta_index": eta_index,
                        "seed": int(seed),
                        "training_seed": int(f"{seed}{seed}"),
                        "model_path": inputs["model"]["path"],
                        "train_path": inputs["files"]["train"]["path"],
                        "eval_path": inputs["files"]["eval"]["path"],
                        "max_rows": 15000,
                        "t": t,
                        "h": h,
                        "s": 2560,
                        "m": 4,
                        "timeout_minutes": 180,
                    }
                    retry_gpus = tuple(slot_gpu for _, slot_gpu in V16_SLOTS)
                    cells.append(common.bind_cell(cell, head, retry_gpus=retry_gpus))
                    slot_offsets[slot_index] += 1
                    queue_records[slot_index]["scientific_cells"] += 1
                group_index += 1
    if len(cells) != 612 or group_index != 102:
        raise SystemExit("internal V16 cardinality mismatch")

    manifest = {
        "schema": "yeto_day3_launch_manifest_v1",
        "status": "AUTHORIZED",
        "program": "v16",
        "created_at_utc": common.utc_now(),
        "source": {"branch": BRANCH, "git_commit": head, "public_remote_tip": head},
        "registration": registration,
        "registration_contract": contract,
        "execution_files": execution_files,
        "syncer": {
            "path": str(common.REMOTE_REPO / "syncer/target/release/yeto-syncer"),
            "sha256": args.syncer_sha256,
        },
        "inputs": {
            "manifest": {
                "path": "/root/yeto-data/tonight85-v13/manifest.json",
                "sha256": input_sha,
            },
            "content": inputs,
        },
        "contract": {
            "T": [2, 5, 20],
            "H_by_T": {"2": 1280, "5": 512, "20": 128},
            "S": 2560,
            "arms": ["mu0", "mu09"],
            "seeds": list(seeds),
            "required_scientific_cells": 612,
            "required_retry_groups": 102,
            "rung_estimator": "median of all 17 finite endpoints",
            "outcome_reads_before_drain": "FORBIDDEN",
        },
        "bootstrap": {"draws": 10000, "seed": 2026081601, "minimum_valid": 7000},
        "result_root": {"symlink": str(common.RESULT_ROOT), "lvm_target": str(common.RESULT_TARGET)},
        "capacity": common.capacity_contract(estimated_peak_mib=36864),
        "environment": {
            "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
            "TMPDIR": "/data/tmp",
            "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
        },
        "queues": queue_records,
        "cells": cells,
        "retry": {
            "attempt_limit": 2,
            "unit": "all six eta rungs sharing T, arm, and seed",
            "attempt2_supersedes_attempt1": True,
            "finite_endpoint_retry_forbidden": True,
        },
    }
    common.write_json_atomic(args.output, manifest)
    digest = common.sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "cells": len(cells),
                "retry_groups": group_index,
                "queues": len(queue_records),
                "source_commit": head,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
