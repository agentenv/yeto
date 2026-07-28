#!/usr/bin/env python3
"""Build the hash-bound 72-cell G13B Pythia/UltraChat regrid manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import tonight85_common as common
from gatesim_v13b import CENTERS


EXECUTION_REPO = Path("/root/yeto-v13b")
RESULT_ROOT = Path("/root/yeto-results-v13b")
RESULT_TARGET = Path("/data/yeto-results-v13b")
MODEL = (
    "/root/yeto-hf-cache/hub/models--EleutherAI--pythia-160m/"
    "snapshots/50f5173d932e8e61f858120bcb800b97af589f46"
)
TRAIN = "/root/yeto-data/tonight85-v13/train.jsonl"
EVAL = "/root/yeto-data/tonight85-v13/eval.jsonl"
SEEDS = (981, 983, 991)
OFFSETS = (-1.5, -0.5, 0.5, 1.5)
SLOTS = (
    ("h200-n1", 6),
    ("h200-n1", 7),
    *(("h200-n2", gpu) for gpu in range(8)),
)
SHUFFLE_SEED = 20_260_743


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    gpu = cell["assignment"]["gpus"][0]
    return [
        "/root/yeto-venv/bin/python",
        str(EXECUTION_REPO / "scripts/compare_diloco.py"),
        "--model", MODEL,
        "--data", TRAIN,
        "--prebound-development-eval", EVAL,
        "--settings", "m4",
        "--tuning", "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget", "1310720",
        "--seq-len", "128",
        "--micro-batch-size", "1",
        "--inner-lr", "0.001",
        "--eval-rows", "1024",
        "--max-rows", "15000",
        "--shuffle-rows-seed", str(cell["seed"]),
        "--eval-split-seed", "331",
        "--training-seed", str(cell["training_seed"]),
        "--device", "cuda",
        "--gpu-slots", "1",
        "--gpu-offset", str(gpu),
        "--delta-correction", "none",
        "--matrix-merge", "rda",
        "--outer-optimizer", "nesterov",
        "--outer-momentum", repr(float(cell["mu"])),
        "--outer-lr", repr(float(cell["eta"])),
        "--fixed-window-microsteps", str(cell["h"]),
        "--fixed-window-tokens", str(cell["h"] * 128),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms", "0,0,0,0",
        "--learner-delay-jitter-ms", "0",
        "--syncer-total-steps", str(4 * int(cell["t"])),
        "--learner-max-steps", "2560",
        "--strict-quorum",
        "--pipeline-depth", "4",
        "--wan-streams", "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every", str(4 * int(cell["t"])),
        "--rho-telemetry",
        "--arm-timeout-min", "180",
        "--work-dir", str(attempt / "work"),
        "--report-dir", str(attempt / "report"),
    ]


def build_cells(source_commit: str) -> list[dict]:
    cells = []
    for t in (2, 5, 20):
        h = 2560 // t
        for arm, mu in (("mu0", 0.0), ("mu09", 0.9)):
            center = CENTERS[(t, arm)]
            for eta_index, offset in enumerate(OFFSETS):
                eta = center * 2.0**offset
                for seed in SEEDS:
                    cells.append(
                        {
                            "program": "v13b",
                            "stage": "v13b_regrid",
                            "t": t,
                            "s": 2560,
                            "h": h,
                            "m": 4,
                            "arm": arm,
                            "mu": mu,
                            "eta": eta,
                            "eta_index": eta_index,
                            "seed": seed,
                            "training_seed": common.training_seed(seed),
                            "model_path": MODEL,
                            "train_path": TRAIN,
                            "eval_path": EVAL,
                            "outer_optimizer": "nesterov",
                            "timeout_minutes": 180,
                            "retry_group_id": f"v13b-T{t}-{arm}-seed{seed}",
                            "cell_id": f"v13b-t{t:02d}-{arm}-e{eta_index}-seed{seed}",
                            "source_git_commit": source_commit,
                            "arm_name": "m4",
                            "expected": {
                                "learner_count": 4,
                                "learner_steps_per_learner": 2560,
                                "outer_steps": 4 * t,
                                "telemetry_rows": 4 * t,
                                "eval_rows": 1024,
                                "fixed_window_microsteps": h,
                                "fixed_window_tokens": h * 128,
                            },
                        }
                    )
    if len(cells) != 72:
        raise RuntimeError(f"expected 72 cells, built {len(cells)}")
    random.Random(SHUFFLE_SEED).shuffle(cells)
    queues = {slot: [] for slot in SLOTS}
    for index, cell in enumerate(cells):
        queues[SLOTS[index % len(SLOTS)]].append(cell)
    bound = []
    for (node, gpu), queue in queues.items():
        for queue_index, cell in enumerate(queue):
            cell["assignment"] = {"node": node, "gpus": [gpu]}
            cell["slot_id"] = f"{node}-gpu{gpu}"
            cell["slot_queue_index"] = queue_index
            command = command_for(cell, 1)
            retry = command_for(cell, 2)
            cell["command"] = command
            cell["command_hash"] = common.canonical_sha256(command)
            cell["attempts"] = [1, 2]
            cell["attempt2_supersedes_attempt1"] = True
            cell["registered_retry_commands"] = [
                {
                    "attempt_number": 2,
                    "command": retry,
                    "command_hash": common.canonical_sha256(retry),
                    "allowed_only_under": "registered loss-blind whole-curve infrastructure retry",
                }
            ]
            bound.append(cell)
    return bound


def validate(manifest: dict) -> None:
    cells = manifest["cells"]
    if len(cells) != 72 or len({cell["cell_id"] for cell in cells}) != 72:
        raise RuntimeError("v13b cell count/identity failure")
    coordinates = {
        (cell["t"], cell["arm"], cell["eta_index"], cell["seed"])
        for cell in cells
    }
    if len(coordinates) != 72:
        raise RuntimeError("v13b scientific coordinate duplication")
    loads = {
        slot: sum(
            cell["assignment"] == {"node": slot[0], "gpus": [slot[1]]}
            for cell in cells
        )
        for slot in SLOTS
    }
    if sorted(loads.values()) != [7] * 8 + [8] * 2:
        raise RuntimeError(f"v13b queue imbalance: {loads}")
    for cell in cells:
        if common.canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash mismatch: {cell['cell_id']}")
        retry = cell["registered_retry_commands"][0]
        if common.canonical_sha256(retry["command"]) != retry["command_hash"]:
            raise RuntimeError(f"retry hash mismatch: {cell['cell_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires a clean tracked registration checkout")
    source_commit = common.git("rev-parse", "HEAD")
    contract_path = common.REPO / "experiment-specs/outer-mup-v13b-pythia-ultrachat-regrid-prereg.json"
    cells = build_cells(source_commit)
    manifest = {
        "schema": "yeto_v13b_launch_manifest_v1",
        "kind": "v13b-regrid",
        "status": "REGISTERED",
        "created_at_utc": common.utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": "experiment/tonight-8.5-lean",
            "execution_repo": str(EXECUTION_REPO),
        },
        "registration": {
            "contract": {
                "path": str(contract_path.relative_to(common.REPO)),
                "sha256": common.sha256_file(contract_path),
            }
        },
        "result_root": str(RESULT_ROOT),
        "result_target": str(RESULT_TARGET),
        "analysis_cutoff": "2026-07-28T08:30:00-07:00",
        "randomization": {
            "seed": SHUFFLE_SEED,
            "policy": "global seeded shuffle then round-robin over ten v11-disjoint queues",
            "slots": [{"node": node, "gpu": gpu} for node, gpu in SLOTS],
        },
        "cells": cells,
    }
    validate(manifest)
    common.write_json_atomic(args.output, manifest)
    digest = common.sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(json.dumps({"manifest": str(args.output), "sha256": digest, "cells": 72, "source_git_commit": source_commit}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
