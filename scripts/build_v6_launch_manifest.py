#!/usr/bin/env python3
"""Build the deterministic, hash-bound 900-cell v6 launch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SEEDS = (601, 607, 613, 617, 619)
ARMS = ("mu0", "raw", "corrected")
NODES = ("h200-n1", "h200-n2")
GPUS = tuple(range(8))
SHUFFLE_SEED = 20260727
RESULT_ROOT = Path("/root/yeto-results-v6")
MODEL = Path("/root/yeto-data/model")
EVAL = Path("/root/yeto-data/splits/seed-337/eval.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def arm_fields(arm: str) -> tuple[float, bool]:
    if arm == "mu0":
        return 0.0, False
    if arm == "raw":
        return 0.9, False
    if arm == "corrected":
        return 0.9, True
    raise RuntimeError(f"unknown arm {arm}")


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    timeout_minutes = {2560: 180, 5120: 240, 10240: 360}[cell["s"]]
    command = [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        str(MODEL),
        "--data",
        f"/root/yeto-data/outer-mup-v6/seed-{cell['seed']}/train.jsonl",
        "--prebound-development-eval",
        str(EVAL),
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
        "1",
        "--gpu-offset",
        str(cell["assignment"]["gpu"]),
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
            str(timeout_minutes),
            "--work-dir",
            str(attempt / "work"),
            "--report-dir",
            str(attempt / "report"),
        ]
    )
    return command


def build_cells(contract: dict, source_commit: str) -> list[dict]:
    cells = []
    for coordinate in contract["design"]["factorial_cells"]:
        t = int(coordinate["T"])
        s = int(coordinate["S"])
        h = int(coordinate["H"])
        if s % t or h != s // t:
            raise RuntimeError(f"T{t}/S{s}: nonintegral or mismatched H")
        for arm in ARMS:
            mu, correction = arm_fields(arm)
            etas = coordinate["eta_grids"][arm]["etas"]
            if len(etas) != 5:
                raise RuntimeError(f"T{t}/S{s}/{arm}: expected five etas")
            for eta_index, eta in enumerate(etas):
                for seed in SEEDS:
                    cells.append(
                        {
                            "t": t,
                            "s": s,
                            "h": h,
                            "m": 4,
                            "arm": arm,
                            "mu": mu,
                            "outer_bias_correction": correction,
                            "eta": float(eta),
                            "eta_index": eta_index,
                            "seed": seed,
                            "training_seed": int(f"{seed}{seed}"),
                            "source_git_commit": source_commit,
                            "retry_group_id": f"T{t}-S{s}-{arm}-seed{seed}",
                            "estimated_cost_units": s // 2560,
                        }
                    )
    if len(cells) != 900:
        raise RuntimeError(f"expected 900 cells, built {len(cells)}")

    rng = random.Random(SHUFFLE_SEED)
    ordered = []
    for s in sorted({cell["s"] for cell in cells}, reverse=True):
        stratum = [cell for cell in cells if cell["s"] == s]
        rng.shuffle(stratum)
        ordered.extend(stratum)

    slots = [(node, gpu) for node in NODES for gpu in GPUS]
    loads = {slot: 0 for slot in slots}
    queues = {slot: [] for slot in slots}
    tie_cursor = 0
    for cell in ordered:
        minimum = min(loads.values())
        candidates = [slot for slot in slots if loads[slot] == minimum]
        slot = candidates[tie_cursor % len(candidates)]
        tie_cursor += 1
        loads[slot] += cell["estimated_cost_units"]
        queues[slot].append(cell)

    assigned = []
    global_index = 0
    for node, gpu in slots:
        queue = queues[(node, gpu)]
        for slot_index, cell in enumerate(queue):
            cell["cell_id"] = (
                f"v6-t{cell['t']:02d}-s{cell['s']:05d}-{cell['arm']}-"
                f"e{cell['eta_index']}-seed{cell['seed']}"
            )
            cell["assignment"] = {"node": node, "gpu": gpu}
            cell["slot_queue_index"] = slot_index
            cell["global_queue_index"] = global_index
            cell["expected"] = {
                "learner_count": 4,
                "learner_steps_per_learner": cell["s"],
                "outer_steps": 4 * cell["t"],
                "telemetry_rows": 4 * cell["t"],
                "eval_rows": 1024,
                "fixed_window_microsteps": cell["h"],
                "fixed_window_tokens": cell["h"] * 128,
            }
            command = command_for(cell, 1)
            retry = command_for(cell, 2)
            cell["command"] = command
            cell["command_hash"] = canonical_sha256(command)
            cell["registered_retry_commands"] = [
                {
                    "attempt_number": 2,
                    "command": retry,
                    "command_hash": canonical_sha256(retry),
                    "allowed_only_under": (
                        "loss-blind whole paired-seed curve retry authority for "
                        "an enumerated infrastructure reason"
                    ),
                }
            ]
            assigned.append(cell)
            global_index += 1
    return assigned


def validate(cells: list[dict]) -> dict:
    if len(cells) != 900 or len({cell["cell_id"] for cell in cells}) != 900:
        raise RuntimeError("cell count or identity failure")
    counts = {}
    for cell in cells:
        key = (cell["t"], cell["s"], cell["arm"], cell["eta"])
        counts[key] = counts.get(key, 0) + 1
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash failure: {cell['cell_id']}")
        if cell["s"] % cell["h"] or cell["s"] // cell["h"] != cell["t"]:
            raise RuntimeError(f"H closure failure: {cell['cell_id']}")
        command = cell["command"]
        has_correction = "--outer-bias-correction" in command
        if has_correction != (cell["arm"] == "corrected"):
            raise RuntimeError(f"correction flag failure: {cell['cell_id']}")
    if len(counts) != 180 or any(count != 5 for count in counts.values()):
        raise RuntimeError(f"eta-level seed balance failure: {counts}")

    loads = {}
    queue_lengths = {}
    for node in NODES:
        for gpu in GPUS:
            queue = sorted(
                [
                    cell
                    for cell in cells
                    if cell["assignment"] == {"node": node, "gpu": gpu}
                ],
                key=lambda cell: cell["slot_queue_index"],
            )
            costs = [cell["estimated_cost_units"] for cell in queue]
            if costs != sorted(costs, reverse=True):
                raise RuntimeError(f"longest-first failure: {node}/gpu{gpu}")
            loads[f"{node}/gpu{gpu}"] = sum(costs)
            queue_lengths[f"{node}/gpu{gpu}"] = len(queue)
    if max(loads.values()) - min(loads.values()) > 1:
        raise RuntimeError(f"greedy cost imbalance: {loads}")
    return {"cost_loads": loads, "queue_lengths": queue_lengths}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO / "experiment-specs/outer-mup-v6-factorial-prereg.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires clean tracked registration files")
    source_commit = git("rev-parse", "HEAD")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "yeto_outer_mup_v6_factorial_prereg_v1":
        raise SystemExit("not the v6 factorial contract")
    analyzer = REPO / "scripts/analyze_v6.py"
    contract_md = args.contract.with_suffix(".md")
    hashes = {
        "json_sha256": sha256_file(args.contract),
        "md_sha256": sha256_file(contract_md),
        "analyzer_sha256": sha256_file(analyzer),
    }
    if hashes["analyzer_sha256"] != contract["frozen_analyzer"]["sha256"]:
        raise SystemExit("frozen analyzer hash differs from the contract")
    cells = build_cells(contract, source_commit)
    balance = validate(cells)
    manifest = {
        "schema": "yeto_outer_mup_v6_launch_manifest_v1",
        "manifest_variant": "v6_full_T_by_S_factorial",
        "stage": "V6_FACTORIAL",
        "status": "REGISTERED",
        "created_at_utc": utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": git("branch", "--show-current"),
        },
        "registration": {"git_commit": source_commit},
        "contract": {
            "json_path": str(args.contract.relative_to(REPO)),
            "md_path": str(contract_md.relative_to(REPO)),
            "analyzer_path": str(analyzer.relative_to(REPO)),
            **hashes,
        },
        "inputs": contract["machine_inputs"],
        "workflow_gates": contract["workflow_gates"],
        "scheduling": {
            "policy": "longest-first greedy exact-S work balance",
            "shuffle_seed": SHUFFLE_SEED,
            "cost_units": {"S2560": 1, "S5120": 2, "S10240": 4},
            "slots": 16,
            **balance,
        },
        "wall_clock": contract["wall_clock"],
        "retry_contract": contract["retry_contract"],
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
                "minimum_slot_cost": min(balance["cost_loads"].values()),
                "maximum_slot_cost": max(balance["cost_loads"].values()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
