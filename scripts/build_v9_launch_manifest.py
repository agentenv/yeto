#!/usr/bin/env python3
"""Build the deterministic, hash-bound 28-cell v9 launch manifest."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_common import (  # noqa: E402
    V9Error,
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


CONTRACT_SCHEMA = "yeto_outer_mup_v9_sealed_scale_prereg_v1"
PREDICTION_SCHEMA = "yeto_outer_mup_v9_sealed_predictions_v1"
MANIFEST_SCHEMA = "yeto_outer_mup_v9_launch_manifest_v1"
SEEDS = (901, 907)
NODES = ("h200-n1", "h200-n2")
GPUS = tuple(range(8))
ISLANDS = (
    ("h200-n1", (0, 1, 2, 3)),
    ("h200-n1", (4, 5, 6, 7)),
    ("h200-n2", (0, 1, 2, 3)),
    ("h200-n2", (4, 5, 6, 7)),
)
SHUFFLE_SEED = 20260728
RESULT_ROOT = Path("/root/yeto-results-v9")


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise V9Error(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def model_path(contract: dict, stage: str) -> str:
    key = "smollm2_1p7b" if stage == "stage_1p7b" else "qwen2p5_7b"
    return str(contract["models"][key]["path"])


def command_for(cell: dict, contract: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    command = [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        model_path(contract, cell["stage"]),
        "--data",
        str(contract["machine_inputs"]["training_jsonl"]["path"]),
        "--prebound-development-eval",
        str(contract["machine_inputs"]["development_jsonl"]["path"]),
        "--settings",
        "m4",
        "--tuning",
        "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget",
        str(cell["s"] * 128 * 4),
        "--seq-len",
        "128",
        "--micro-batch-size",
        "1",
        "--inner-lr",
        "0.001",
        "--eval-rows",
        "1024",
        "--max-rows",
        "13758",
        "--shuffle-rows-seed",
        str(cell["seed"]),
        "--eval-split-seed",
        "331",
        "--training-seed",
        str(cell["training_seed"]),
        "--device",
        "cuda",
        "--gpu-slots",
        str(len(cell["assignment"]["gpus"])),
        "--gpu-offset",
        str(min(cell["assignment"]["gpus"])),
        "--delta-correction",
        "none",
        "--matrix-merge",
        "rda",
        "--outer-optimizer",
        "nesterov",
        "--outer-momentum",
        repr(cell["mu"]),
        "--outer-lr",
        repr(cell["eta"]),
        "--fixed-window-microsteps",
        str(cell["h"]),
        "--fixed-window-tokens",
        str(cell["h"] * 128),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms",
        "0,0,0,0",
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        str(4 * cell["t"]),
        "--learner-max-steps",
        str(cell["s"]),
        "--strict-quorum",
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        str(4 * cell["t"]),
        "--rho-telemetry",
    ]
    if cell["outer_bias_correction"]:
        command.append("--outer-bias-correction")
    command.extend(
        [
            "--arm-timeout-min",
            "480" if cell["stage"] == "stage_1p7b" else "720",
            "--work-dir",
            str(attempt / "work"),
            "--report-dir",
            str(attempt / "report"),
        ]
    )
    return command


def target_cells(predictions: dict) -> list[dict]:
    cells = []
    for stage, t, s in (("stage_1p7b", 10, 5120), ("stage_7b", 5, 2560)):
        for arm, target in predictions[stage]["targets"].items():
            etas = target["verification_etas"]
            expected_points = 4 if stage == "stage_1p7b" else 3
            if len(etas) != expected_points:
                raise V9Error(f"{stage}/{arm}: unexpected eta count")
            for eta_index, eta in enumerate(etas):
                for seed in SEEDS:
                    cells.append(
                        {
                            "stage": stage,
                            "t": t,
                            "s": s,
                            "h": s // t,
                            "m": 4,
                            "arm": arm,
                            "mu": float(target["mu"]),
                            "outer_bias_correction": bool(
                                target["outer_bias_correction"]
                            ),
                            "eta": float(eta),
                            "eta_index": eta_index,
                            "seed": seed,
                            "training_seed": int(f"{seed}{seed}"),
                            "retry_group_id": f"{stage}-{arm}-seed{seed}",
                        }
                    )
    if len(cells) != 28:
        raise V9Error(f"expected 28 v9 cells, built {len(cells)}")
    return cells


def assign_cells(cells: list[dict], source_commit: str, contract: dict) -> list[dict]:
    rng = random.Random(SHUFFLE_SEED)
    stage_a = [cell for cell in cells if cell["stage"] == "stage_1p7b"]
    stage_b = [cell for cell in cells if cell["stage"] == "stage_7b"]
    rng.shuffle(stage_a)
    rng.shuffle(stage_b)
    if len(stage_a) != 16 or len(stage_b) != 12:
        raise V9Error("v9 stage cardinality changed")
    single_slots = [(node, (gpu,)) for node in NODES for gpu in GPUS]
    for cell, (node, gpus) in zip(stage_a, single_slots):
        cell["assignment"] = {"node": node, "gpus": list(gpus)}
        cell["slot_id"] = f"{node}-gpu{gpus[0]}"
        cell["slot_queue_index"] = 0
    island_queues = {f"{node}-gpu{gpus[0]}-{gpus[-1]}": [] for node, gpus in ISLANDS}
    island_details = {
        f"{node}-gpu{gpus[0]}-{gpus[-1]}": (node, gpus) for node, gpus in ISLANDS
    }
    for index, cell in enumerate(stage_b):
        slot_id = list(island_queues)[index % len(island_queues)]
        island_queues[slot_id].append(cell)
    for slot_id, queue in island_queues.items():
        node, gpus = island_details[slot_id]
        for queue_index, cell in enumerate(queue):
            cell["assignment"] = {"node": node, "gpus": list(gpus)}
            cell["slot_id"] = slot_id
            cell["slot_queue_index"] = queue_index

    assigned = []
    for global_index, cell in enumerate(stage_a + stage_b):
        scale = "1p7b" if cell["stage"] == "stage_1p7b" else "7b"
        cell["cell_id"] = (
            f"v9-{scale}-t{cell['t']:02d}-{cell['arm']}-"
            f"e{cell['eta_index']}-seed{cell['seed']}"
        )
        cell["global_queue_index"] = global_index
        cell["source_git_commit"] = source_commit
        cell["expected"] = {
            "learner_count": 4,
            "learner_steps_per_learner": cell["s"],
            "outer_steps": 4 * cell["t"],
            "telemetry_rows": 4 * cell["t"],
            "eval_rows": 1024,
            "fixed_window_microsteps": cell["h"],
            "fixed_window_tokens": cell["h"] * 128,
            "learner_gpu_count": len(cell["assignment"]["gpus"]),
        }
        command = command_for(cell, contract, 1)
        retry = command_for(cell, contract, 2)
        cell["command"] = command
        cell["command_hash"] = canonical_sha256(command)
        cell["registered_retry_commands"] = [
            {
                "attempt_number": 2,
                "command": retry,
                "command_hash": canonical_sha256(retry),
                "allowed_only_under": (
                    "loss-blind whole seed-by-arm curve retry authority for an "
                    "enumerated infrastructure reason"
                ),
            }
        ]
        assigned.append(cell)
    return assigned


def command_value(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def validate(cells: list[dict]) -> dict:
    if len(cells) != 28 or len({cell["cell_id"] for cell in cells}) != 28:
        raise V9Error("cell count or identity failure")
    counts = {}
    for cell in cells:
        key = (cell["stage"], cell["arm"], cell["eta"])
        counts[key] = counts.get(key, 0) + 1
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise V9Error(f"initial command hash failure: {cell['cell_id']}")
        retry = cell["registered_retry_commands"][0]
        if canonical_sha256(retry["command"]) != retry["command_hash"]:
            raise V9Error(f"retry command hash failure: {cell['cell_id']}")
        expected_slots = "1" if cell["stage"] == "stage_1p7b" else "4"
        if command_value(cell["command"], "--gpu-slots") != expected_slots:
            raise V9Error(f"GPU slot width failure: {cell['cell_id']}")
        correction = "--outer-bias-correction" in cell["command"]
        if correction != (cell["arm"] == "corrected"):
            raise V9Error(f"bias-correction binding failure: {cell['cell_id']}")
    if any(value != 2 for value in counts.values()) or len(counts) != 14:
        raise V9Error("every stage/arm/eta must contain exactly both seeds")
    a_slots = [cell["slot_id"] for cell in cells if cell["stage"] == "stage_1p7b"]
    if len(set(a_slots)) != 16:
        raise V9Error("1.7B cells must occupy 16 distinct one-GPU slots")
    b_queues = {}
    for cell in cells:
        if cell["stage"] == "stage_7b":
            b_queues.setdefault(cell["slot_id"], []).append(cell)
    if len(b_queues) != 4 or any(len(queue) != 3 for queue in b_queues.values()):
        raise V9Error("7B stage must be four 4-GPU queues of three cells")
    if any(
        sorted(cell["assignment"]["gpus"] for cell in queue)
        != sorted([queue[0]["assignment"]["gpus"]] * len(queue))
        for queue in b_queues.values()
    ):
        raise V9Error("7B queue GPU ownership changed within a queue")
    return {
        "stage_cell_counts": {"stage_1p7b": 16, "stage_7b": 12},
        "stage_1p7b_slots": sorted(set(a_slots)),
        "stage_7b_queues": {
            slot: [
                cell["cell_id"]
                for cell in sorted(queue, key=lambda item: item["slot_queue_index"])
            ]
            for slot, queue in sorted(b_queues.items())
        },
    }


def validate_analysis_binding(contract: dict, predictions: dict) -> None:
    gates = contract.get("analysis_contract", {}).get("gates", {})
    mapping = {
        "G9A_1P7B": "stage_1p7b",
        "G9B_7B": "stage_7b",
    }
    for gate_id, stage in mapping.items():
        gate = gates.get(gate_id)
        if not isinstance(gate, dict):
            raise V9Error(f"contract lacks {gate_id}")
        for arm, target in predictions[stage]["targets"].items():
            if float(target["registered_absolute_error_band_bits"]) != float(
                gate["absolute_error_band_bits"][arm]
            ):
                raise V9Error(f"{gate_id}/{arm} prediction band differs from contract")
    groups = contract["analysis_contract"]["bootstrap"]["groups"]
    if (
        len(groups) != 3
        or sum(int(group["frequency"]) for group in groups) != 10_000
        or {
            tuple(sorted(int(value) for value in group["representative"]))
            for group in groups
        }
        != {(0, 0), (0, 1), (1, 1)}
    ):
        raise V9Error("contract bootstrap groups are not the exact two-seed support")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires a clean tracked registration")
    contract = read_json(args.contract)
    predictions = read_json(args.predictions)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SystemExit("not the v9 preregistration")
    if (
        predictions.get("schema") != PREDICTION_SCHEMA
        or predictions.get("status") != "SEALED"
    ):
        raise SystemExit("not the sealed v9 predictions")
    if predictions.get("verification_loss_seen") is not False:
        raise SystemExit("prediction seal does not assert zero verification exposure")
    try:
        validate_analysis_binding(contract, predictions)
    except V9Error as exc:
        raise SystemExit(str(exc)) from exc
    expected_contract_hash = predictions.get("source_artifacts", {}).get(
        "contract_sha256"
    )
    if expected_contract_hash != sha256_file(args.contract):
        raise SystemExit("predictions bind another v9 contract")
    for label, record in contract.get("frozen_code", {}).items():
        path = REPO / record["path"]
        if sha256_file(path) != record["sha256"]:
            raise SystemExit(f"frozen code hash mismatch: {label}")
    source_commit = git("rev-parse", "HEAD")
    cells = assign_cells(target_cells(predictions), source_commit, contract)
    scheduling = validate(cells)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": "V9_SEALED_SCALE",
        "status": "REGISTERED",
        "created_at_utc": utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": git("branch", "--show-current"),
        },
        "registration": {
            "git_commit": source_commit,
            "must_be_pushed_before_authority": True,
        },
        "contract": {
            "path": str(args.contract.relative_to(REPO)),
            "sha256": sha256_file(args.contract),
        },
        "predictions": {
            "path": str(args.predictions.relative_to(REPO)),
            "sha256": sha256_file(args.predictions),
            "prediction_preimage_canonical_sha256": predictions[
                "prediction_preimage_canonical_sha256"
            ],
            "verification_loss_seen": False,
        },
        "inputs": contract["machine_inputs"],
        "models": contract["models"],
        "analysis_contract": contract["analysis_contract"],
        "retry_contract": contract["retry_contract"],
        "wall_clock": contract["wall_clock"],
        "scheduling": {
            "shuffle_seed": SHUFFLE_SEED,
            "stage_order": ["stage_1p7b", "stage_7b"],
            "stage_7b_requires_stage_1p7b_drain": True,
            **scheduling,
        },
        "cells": cells,
    }
    output = args.output.resolve()
    write_json_atomic(output, manifest)
    digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(output),
                "sha256": digest,
                "cells": len(cells),
                "source_git_commit": source_commit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
