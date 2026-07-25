#!/usr/bin/env python3
"""Build the EXPLORATORY Lane B finite-T launch manifest and GPU queues."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


LABEL = "EXPLORATORY"
COMMIT = "a886a3996905913d37ec56cc14914878f636283d"
H = 512
M = 4
SEEDS = (401, 409)
ETA0 = 0.0443
RESULT_ROOT = Path("/root/yeto-results-explore")
DATA_ROOT = Path("/root/yeto-data/outer-mup-explore")
EVAL_PATH = Path("/root/yeto-data/splits/seed-337/eval.jsonl")
REPO = Path("/root/yeto")
PYTHON = Path("/root/yeto-venv/bin/python")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def mu_tag(mu: float) -> str:
    return {
        0.0: "0",
        0.5: "0p5",
        0.8: "0p8",
        0.9: "0p9",
        0.95: "0p95",
    }[mu]


def curves() -> list[dict[str, object]]:
    # D predictions are the user-specified registered values, including their
    # requested rounding; the eta centers use these exact registered numbers.
    grid1_d = {1024: 5.263, 5120: 1.535, 10240: 1.139}
    records: list[dict[str, object]] = []
    for s in (1024, 5120, 10240):
        records.append(
            {
                "grid": "grid1_s_variation",
                "h": H,
                "s": s,
                "t": s // H,
                "mu": 0.0,
                "d_pred": 1.0,
                "eta0_reference": ETA0,
                "eta_center": ETA0,
                "d_denominator": "within-S fitted mu=0 eta_star",
            }
        )
        records.append(
            {
                "grid": "grid1_s_variation",
                "h": H,
                "s": s,
                "t": s // H,
                "mu": 0.9,
                "d_pred": grid1_d[s],
                "eta0_reference": ETA0,
                "eta_center": ETA0 * 0.1 * grid1_d[s],
                "d_denominator": "within-S fitted mu=0 eta_star",
            }
        )
    for mu, d_pred in ((0.5, 1.033), (0.8, 1.487), (0.95, 4.417)):
        records.append(
            {
                "grid": "grid2_mu_sweep",
                "h": H,
                "s": 2560,
                "t": 5,
                "mu": mu,
                "d_pred": d_pred,
                "eta0_reference": ETA0,
                "eta_center": ETA0 * (1.0 - mu) * d_pred,
                "d_denominator": "existing S=2560 mu=0 eta_star=0.0443",
            }
        )
    return records


def command_for(cell: dict[str, object], gpu: int) -> list[str]:
    s = int(cell["s"])
    seed = int(cell["seed"])
    outer_steps = 4 * (s // H)
    attempt = RESULT_ROOT / str(cell["cell_id"]) / "attempt-1"
    return [
        str(PYTHON),
        str(REPO / "scripts" / "compare_diloco.py"),
        "--model",
        "/root/yeto-data/model",
        "--data",
        str(DATA_ROOT / f"seed-{seed}" / "train.jsonl"),
        "--prebound-development-eval",
        str(EVAL_PATH),
        "--settings",
        "m4",
        "--tuning",
        "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget",
        str(s * M * 128),
        "--seq-len",
        "128",
        "--micro-batch-size",
        "1",
        "--inner-lr",
        "0.001",
        "--eval-rows",
        "1024",
        "--max-rows",
        "5000",
        "--shuffle-rows-seed",
        str(seed),
        "--eval-split-seed",
        "331",
        "--training-seed",
        int(f"{seed}{seed}").__str__(),
        "--device",
        "cuda",
        "--gpu-slots",
        "1",
        "--gpu-offset",
        str(gpu),
        "--delta-correction",
        "none",
        "--matrix-merge",
        "rda",
        "--outer-optimizer",
        "nesterov",
        "--outer-momentum",
        format(float(cell["mu"]), ".12g"),
        "--outer-lr",
        format(float(cell["eta"]), ".17g"),
        "--fixed-window-microsteps",
        str(H),
        "--fixed-window-tokens",
        str(H * 128),
        "--pad-to-fixed-window-tokens",
        "--freeze-delta-before-delay",
        "--learner-push-delay-ms",
        "0,0,0,0",
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        str(outer_steps),
        "--learner-max-steps",
        str(s),
        "--strict-quorum",
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        str(outer_steps),
        "--rho-telemetry",
        "--arm-timeout-min",
        "180",
        "--work-dir",
        str(attempt / "work"),
        "--report-dir",
        str(attempt / "report"),
    ]


def build() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    curve_order = {
        (10240, 0.0): 0,
        (10240, 0.9): 1,
        (5120, 0.0): 2,
        (5120, 0.9): 3,
        (2560, 0.5): 4,
        (2560, 0.8): 5,
        (2560, 0.95): 6,
        (1024, 0.0): 7,
        (1024, 0.9): 8,
    }
    for curve in curves():
        center = float(curve["eta_center"])
        for eta_index, exponent in enumerate((-0.75, -0.25, 0.25, 0.75)):
            eta = center * (2.0**exponent)
            for seed in SEEDS:
                cell_id = (
                    f"e1x-ft-h{H:04d}-s{int(curve['s']):05d}-"
                    f"mu{mu_tag(float(curve['mu']))}-e{eta_index}-seed{seed}"
                )
                cell = {
                    "label": LABEL,
                    "cell_id": cell_id,
                    **curve,
                    "m": M,
                    "eta_index": eta_index,
                    "eta_ladder_exponent": exponent,
                    "eta": eta,
                    "seed": seed,
                    "training_seed": int(f"{seed}{seed}"),
                    "token_budget": int(curve["s"]) * M * 128,
                    "expected_outer_steps": 4 * (int(curve["s"]) // H),
                    "expected_telemetry_rows": 4 * (int(curve["s"]) // H),
                    "cost_units_s2560": int(curve["s"]) / 2560.0,
                    "curve_order": curve_order[(int(curve["s"]), float(curve["mu"]))],
                }
                cells.append(cell)

    if len(cells) != 72:
        raise AssertionError(f"expected 72 cells, got {len(cells)}")
    if any(int(cell["seed"]) == 307 for cell in cells):
        raise AssertionError("reserved seed 307 entered the grid")

    # Longest-processing-time assignment. Stable curve ordering makes the
    # first 16 cells exactly the S=10240 wave, one per physical GPU.
    workers = [
        {"node": "h200-n1" if index < 8 else "h200-n2", "gpu": index % 8}
        for index in range(16)
    ]
    loads = [0.0] * len(workers)
    queues: list[list[dict[str, object]]] = [[] for _ in workers]
    ordered = sorted(
        cells,
        key=lambda cell: (
            -float(cell["cost_units_s2560"]),
            int(cell["curve_order"]),
            int(cell["eta_index"]),
            int(cell["seed"]),
        ),
    )
    for schedule_index, cell in enumerate(ordered):
        worker_index = min(range(len(workers)), key=lambda index: (loads[index], index))
        assignment = workers[worker_index]
        cell["assignment"] = dict(assignment)
        cell["schedule_index"] = schedule_index
        cell["slot_queue_index"] = len(queues[worker_index])
        cell["command"] = command_for(cell, int(assignment["gpu"]))
        cell["command_hash"] = canonical_sha256(cell["command"])
        queues[worker_index].append(cell)
        loads[worker_index] += float(cell["cost_units_s2560"])

    first_wave = ordered[:16]
    if {int(cell["s"]) for cell in first_wave} != {10240}:
        raise AssertionError("the first 16 scheduled cells are not all S=10240")
    if len({(cell["assignment"]["node"], cell["assignment"]["gpu"]) for cell in first_wave}) != 16:
        raise AssertionError("the first wave does not occupy all 16 GPUs")

    manifest: dict[str, object] = {
        "label": LABEL,
        "schema": "yeto_e1x_lane_b_finite_t_manifest_v1",
        "status": "EXPLORATORY",
        "source_git_commit": COMMIT,
        "result_namespace": "e1x-*",
        "result_root": str(RESULT_ROOT),
        "data_root": str(DATA_ROOT),
        "frozen_eval": str(EVAL_PATH),
        "law": "D(T,mu) ~ 1/(1-mu^T)",
        "eta_ladder": {
            "points": 4,
            "ratio": math.sqrt(2.0),
            "exponents_about_geometric_center": [-0.75, -0.25, 0.25, 0.75],
        },
        "seeds": list(SEEDS),
        "cell_count": len(cells),
        "curve_count": 9,
        "weighted_worker_loads": [
            {**workers[index], "cost_units_s2560": loads[index], "cells": len(queues[index])}
            for index in range(16)
        ],
        "cells": ordered,
    }
    manifest["manifest_canonical_hash"] = canonical_sha256(manifest)
    manifest["queues"] = [
        {
            "node": workers[index]["node"],
            "gpu": workers[index]["gpu"],
            "cell_ids": [cell["cell_id"] for cell in queues[index]],
        }
        for index in range(16)
    ]
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = build()
    manifest_path = output / "manifest.json"
    payload = json_bytes(manifest)
    write_bytes(manifest_path, payload)
    manifest_sha = sha256_bytes(payload)
    write_bytes(manifest_path.with_suffix(".json.sha256"), f"{manifest_sha}  manifest.json\n".encode())

    cells_by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    for queue_record in manifest["queues"]:
        node = str(queue_record["node"])
        gpu = int(queue_record["gpu"])
        queue = {
            "label": LABEL,
            "schema": "yeto_e1x_lane_b_gpu_queue_v1",
            "manifest_sha256": manifest_sha,
            "node": node,
            "gpu": gpu,
            "cells": [cells_by_id[cell_id] for cell_id in queue_record["cell_ids"]],
        }
        write_bytes(output / "queues" / f"{node}-gpu{gpu}.json", json_bytes(queue))

    readme = (
        "# EXPLORATORY — Lane B finite-T launch artifacts\n\n"
        "All cells use the `e1x-*` namespace and `/root/yeto-results-explore`. "
        "These files are exploratory-only and must not be promoted into E1/E1v2 evidence.\n"
    )
    write_bytes(output / "README.md", readme.encode())
    print(
        json.dumps(
            {
                "label": LABEL,
                "manifest": str(manifest_path),
                "sha256": manifest_sha,
                "cells": len(manifest["cells"]),
                "worker_loads": manifest["weighted_worker_loads"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
