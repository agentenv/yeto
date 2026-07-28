#!/usr/bin/env python3
"""Build hash-bound static or conditional tonight-8.5 launch manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import tonight85_common as common


SMOL_MODEL_135 = "/root/yeto-data/model"
SMOL_MODEL_1P7 = (
    "/root/yeto-hf-cache/hub/models--HuggingFaceTB--SmolLM2-1.7B/"
    "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
)
PYTHIA_MODEL = (
    "/root/yeto-hf-cache/hub/models--EleutherAI--pythia-160m/"
    "snapshots/50f5173d932e8e61f858120bcb800b97af589f46"
)
QWEN_27B_MODEL = (
    "/data/yeto-hf-cache/hub/models--Qwen--Qwen3.6-27B/"
    "snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
CAPY_TRAIN = "/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl"
CAPY_EVAL = "/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl"
PYTHIA_TRAIN = "/root/yeto-data/tonight85-v13/train.jsonl"
PYTHIA_EVAL = "/root/yeto-data/tonight85-v13/eval.jsonl"


def cell_base(
    *,
    program: str,
    stage: str,
    t: int,
    s: int,
    h: int,
    arm: str,
    mu: float,
    eta: float,
    eta_index: int,
    seed: int,
    model_path: str,
    train_path: str,
    eval_path: str,
    optimizer: str,
    timeout: int,
) -> dict:
    return {
        "program": program,
        "stage": stage,
        "t": t,
        "s": s,
        "h": h,
        "m": 4,
        "arm": arm,
        "mu": mu,
        "eta": eta,
        "eta_index": eta_index,
        "seed": seed,
        "training_seed": common.training_seed(seed),
        "model_path": model_path,
        "train_path": train_path,
        "eval_path": eval_path,
        "outer_optimizer": optimizer,
        "timeout_minutes": timeout,
        "retry_group_id": f"{program}-T{t}-{arm}-seed{seed}",
    }


def assign_one_gpu(cells: list[dict], *, shuffle_seed: int) -> None:
    rng = random.Random(shuffle_seed)
    rng.shuffle(cells)
    queues = {slot: [] for slot in common.SLOTS}
    for index, cell in enumerate(cells):
        queues[common.SLOTS[index % len(common.SLOTS)]].append(cell)
    for (node, gpu), queue in queues.items():
        for queue_index, cell in enumerate(queue):
            cell["assignment"] = {"node": node, "gpus": [gpu]}
            cell["slot_id"] = f"{node}-gpu{gpu}"
            cell["slot_queue_index"] = queue_index


def scan_cells(source_commit: str) -> list[dict]:
    v12 = common.read_json(
        common.REPO / "experiment-specs/outer-mup-v12-heavy-ball-prereg.json"
    )
    v13 = common.read_json(
        common.REPO / "experiment-specs/outer-mup-v13-pythia-ultrachat-prereg.json"
    )
    by_program = []
    for program, contract, optimizer, model, train, eval_path in (
        ("v12", v12, "heavy-ball", SMOL_MODEL_135, CAPY_TRAIN, CAPY_EVAL),
        ("v13", v13, "nesterov", PYTHIA_MODEL, PYTHIA_TRAIN, PYTHIA_EVAL),
    ):
        cells = []
        for coordinate in contract["design"]["grids"]:
            for arm, mu in (("mu0", 0.0), ("mu09", 0.9)):
                key = "mu0" if arm == "mu0" else "mu0.9"
                for eta_index, eta in enumerate(coordinate[key]["etas"]):
                    for seed in common.SCAN_SEEDS:
                        cell = cell_base(
                            program=program,
                            stage="short_scans",
                            t=int(coordinate["T"]),
                            s=2560,
                            h=int(coordinate["H"]),
                            arm=arm,
                            mu=mu,
                            eta=float(eta),
                            eta_index=eta_index,
                            seed=seed,
                            model_path=model,
                            train_path=train,
                            eval_path=eval_path,
                            optimizer=optimizer,
                            timeout=180,
                        )
                        cell["max_rows"] = 15000 if program == "v13" else 13758
                        cell["cell_id"] = (
                            f"{program}-t{coordinate['T']:02d}-{arm}-e{eta_index}-seed{seed}"
                        )
                        cells.append(cell)
        if len(cells) != 72:
            raise RuntimeError(f"{program}: expected 72 cells")
        by_program.append(cells)

    # Interleave programs before balanced round-robin slot assignment.
    rng12 = random.Random(20_260_712)
    rng13 = random.Random(20_260_713)
    rng12.shuffle(by_program[0])
    rng13.shuffle(by_program[1])
    queues = {slot: [] for slot in common.SLOTS}
    cursors = {"v12": 0, "v13": 0}
    program_cells = {"v12": by_program[0], "v13": by_program[1]}
    for queue_index in range(9):
        for slot_index, slot in enumerate(common.SLOTS):
            program = "v12" if (queue_index + slot_index) % 2 == 0 else "v13"
            cell = program_cells[program][cursors[program]]
            cursors[program] += 1
            queues[slot].append(cell)
    if cursors != {"v12": 72, "v13": 72}:
        raise RuntimeError(f"interleaving did not consume both scans: {cursors}")
    for (node, gpu), queue in queues.items():
        for queue_index, cell in enumerate(queue):
            cell["assignment"] = {"node": node, "gpus": [gpu]}
            cell["slot_id"] = f"{node}-gpu{gpu}"
            cell["slot_queue_index"] = queue_index
            common.bind(cell, source_commit)
    return [cell for queue in queues.values() for cell in queue]


def v11_anchor_cells(source_commit: str) -> list[dict]:
    contract = common.read_json(
        common.REPO / "experiment-specs/outer-mup-v11-ratio-transport-prereg.json"
    )
    cells = []
    for coordinate_id, coordinate in contract["coordinates"].items():
        center = float(coordinate["mu0_placement_center"])
        model = SMOL_MODEL_135 if "135m" in coordinate_id else SMOL_MODEL_1P7
        timeout = 480 if "135m" in coordinate_id else 720
        for eta_index, offset in enumerate(contract["anchor_probe"]["offsets_log2"]):
            eta = center * 2.0 ** float(offset)
            cell = cell_base(
                program="v11_anchor",
                stage="v11_anchor",
                t=int(coordinate["T"]),
                s=int(coordinate["S"]),
                h=512,
                arm="mu0",
                mu=0.0,
                eta=eta,
                eta_index=eta_index,
                seed=967,
                model_path=model,
                train_path=CAPY_TRAIN,
                eval_path=CAPY_EVAL,
                optimizer="nesterov",
                timeout=timeout,
            )
            cell["coordinate_id"] = coordinate_id
            cell["retry_group_id"] = f"v11-anchor-{coordinate_id}"
            cell["cell_id"] = f"v11-{coordinate_id}-anchor-e{eta_index}-seed967"
            cells.append(cell)
    assign_one_gpu(cells, shuffle_seed=20_260_711)
    for cell in cells:
        common.bind(cell, source_commit)
    return cells


def v7_prep_cells(source_commit: str) -> list[dict]:
    amendment = common.read_json(
        common.REPO / "experiment-specs/outer-mup-v7-lean-scope-amendment.json"
    )
    cells = []
    smoke = amendment["retained_wiring_smoke"]
    smoke_cell = {
        "program": "v7_smoke",
        "stage": "v7_smoke",
        "cell_id": "v7-lean-smoke-s00064-h0016-mu0-e0-seed683",
        "t": smoke["T"],
        "s": smoke["S"],
        "h": smoke["H"],
        "m": 2,
        "arm": "mu0",
        "mu": smoke["mu"],
        "eta": smoke["eta"],
        "eta_index": 0,
        "seed": smoke["seed"],
        "training_seed": common.training_seed(smoke["seed"]),
        "model_path": QWEN_27B_MODEL,
        "train_path": CAPY_TRAIN,
        "eval_path": CAPY_EVAL,
        "outer_optimizer": "nesterov",
        "timeout_minutes": 90,
        "retry_group_id": "v7-lean-smoke",
        "island": True,
        "assignment": {"node": "h200-n1", "gpus": list(range(8))},
        "slot_id": "h200-n1-island",
        "slot_queue_index": 0,
    }
    cells.append(common.bind(smoke_cell, source_commit))
    pilot = amendment["pilot"]
    assignments = (("h200-n1", 0), ("h200-n2", 0), ("h200-n1", 1))
    for eta_index, (eta, (node, queue_index)) in enumerate(
        zip(pilot["etas"], assignments)
    ):
        cell = {
            "program": "v7_pilot",
            "stage": "v7_pilot",
            "cell_id": f"v7-lean-pilot-t05-mu0-e{eta_index}-seed691",
            "t": 5,
            "s": 2560,
            "h": 512,
            "m": 2,
            "arm": "mu0",
            "mu": 0.0,
            "eta": float(eta),
            "eta_index": eta_index,
            "seed": 691,
            "training_seed": 691691,
            "model_path": QWEN_27B_MODEL,
            "train_path": CAPY_TRAIN,
            "eval_path": CAPY_EVAL,
            "outer_optimizer": "nesterov",
            "timeout_minutes": 180,
            "retry_group_id": "v7-lean-pilot-all",
            "island": True,
            "assignment": {"node": node, "gpus": list(range(8))},
            "slot_id": f"{node}-island",
            "slot_queue_index": queue_index,
        }
        cells.append(common.bind(cell, source_commit))
    return cells


def v11_truth_cells(source_commit: str, predictions: dict) -> list[dict]:
    contract = common.read_json(
        common.REPO / "experiment-specs/outer-mup-v11-ratio-transport-prereg.json"
    )
    cells = []
    for coordinate_id, prediction in predictions["coordinates"].items():
        coordinate = contract["coordinates"][coordinate_id]
        model = SMOL_MODEL_135 if "135m" in coordinate_id else SMOL_MODEL_1P7
        timeout = 480 if "135m" in coordinate_id else 720
        for eta_index, eta in enumerate(prediction["ground_truth_etas"]):
            for seed in common.V11_TRUTH_SEEDS:
                cell = cell_base(
                    program="v11_truth",
                    stage="v11_truth",
                    t=int(coordinate["T"]),
                    s=int(coordinate["S"]),
                    h=512,
                    arm="raw",
                    mu=0.9,
                    eta=float(eta),
                    eta_index=eta_index,
                    seed=seed,
                    model_path=model,
                    train_path=CAPY_TRAIN,
                    eval_path=CAPY_EVAL,
                    optimizer="nesterov",
                    timeout=timeout,
                )
                cell["coordinate_id"] = coordinate_id
                cell["retry_group_id"] = f"v11-truth-{coordinate_id}-seed{seed}"
                cell["cell_id"] = f"v11-{coordinate_id}-raw-e{eta_index}-seed{seed}"
                cells.append(cell)
    if len(cells) != 20:
        raise RuntimeError("v11 truth must contain 20 cells")
    assign_one_gpu(cells, shuffle_seed=20_260_711_2)
    for cell in cells:
        common.bind(cell, source_commit)
    return cells


def v7_raw_cells(source_commit: str, prediction: dict) -> list[dict]:
    cells = []
    for eta_index, eta in enumerate(prediction["raw_etas"]):
        for seed in (701, 709):
            node = common.NODES[len(cells) % 2]
            queue_index = len(cells) // 2
            cell = {
                "program": "v7_raw",
                "stage": "v7_raw",
                "cell_id": f"v7-lean-t05-raw-e{eta_index}-seed{seed}",
                "t": 5,
                "s": 2560,
                "h": 512,
                "m": 2,
                "arm": "raw",
                "mu": 0.9,
                "eta": float(eta),
                "eta_index": eta_index,
                "seed": seed,
                "training_seed": common.training_seed(seed),
                "model_path": QWEN_27B_MODEL,
                "train_path": CAPY_TRAIN,
                "eval_path": CAPY_EVAL,
                "outer_optimizer": "nesterov",
                "timeout_minutes": 180,
                "retry_group_id": f"v7-lean-raw-seed{seed}",
                "island": True,
                "assignment": {"node": node, "gpus": list(range(8))},
                "slot_id": f"{node}-island",
                "slot_queue_index": queue_index,
            }
            cells.append(common.bind(cell, source_commit))
    return cells


def build(kind: str, source_commit: str, dynamic: dict | None) -> dict:
    if kind == "static":
        cells = (
            scan_cells(source_commit)
            + v11_anchor_cells(source_commit)
            + v7_prep_cells(source_commit)
        )
    elif kind == "v11-truth":
        if dynamic is None:
            raise RuntimeError("v11-truth requires predictions")
        cells = v11_truth_cells(source_commit, dynamic)
    elif kind == "v7-raw":
        if dynamic is None:
            raise RuntimeError("v7-raw requires prediction")
        cells = v7_raw_cells(source_commit, dynamic)
    else:
        raise RuntimeError(f"unknown manifest kind {kind}")
    return {
        "schema": "yeto_tonight85_launch_manifest_v1",
        "kind": kind,
        "status": "REGISTERED",
        "created_at_utc": common.utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": "experiment/tonight-8.5-lean",
        },
        "registration": {
            name: {
                "path": f"experiment-specs/{name}",
                "sha256": common.sha256_file(common.REPO / "experiment-specs" / name),
            }
            for name in (
                "outer-mup-v11-ratio-transport-prereg.json",
                "outer-mup-v12-heavy-ball-prereg.json",
                "outer-mup-v13-pythia-ultrachat-prereg.json",
                "outer-mup-v7-lean-scope-amendment.json",
            )
        },
        "dynamic_artifact_canonical_sha256": common.canonical_sha256(dynamic)
        if dynamic
        else None,
        "result_root": str(common.RESULT_ROOT),
        "result_target": str(common.RESULT_TARGET),
        "analysis_cutoff": "2026-07-28T08:30:00-07:00",
        "cells": cells,
    }


def validate(manifest: dict) -> None:
    ids = [cell["cell_id"] for cell in manifest["cells"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate cell ids")
    for cell in manifest["cells"]:
        if common.canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash mismatch: {cell['cell_id']}")
        retry = cell["registered_retry_commands"][0]
        if common.canonical_sha256(retry["command"]) != retry["command_hash"]:
            raise RuntimeError(f"retry hash mismatch: {cell['cell_id']}")
    counts = {}
    for cell in manifest["cells"]:
        counts[cell["program"]] = counts.get(cell["program"], 0) + 1
    expected = {
        "static": {"v12": 72, "v13": 72, "v11_anchor": 6, "v7_smoke": 1, "v7_pilot": 3},
        "v11-truth": {"v11_truth": 20},
        "v7-raw": {"v7_raw": 4},
    }[manifest["kind"]]
    if counts != expected:
        raise RuntimeError(f"manifest counts {counts} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=("static", "v11-truth", "v7-raw"), required=True
    )
    parser.add_argument("--dynamic", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires a clean tracked worktree")
    source_commit = common.git("rev-parse", "HEAD")
    dynamic = common.read_json(args.dynamic) if args.dynamic else None
    manifest = build(args.kind, source_commit, dynamic)
    if args.dynamic:
        dynamic_path = args.dynamic.resolve()
        try:
            relative = dynamic_path.relative_to(common.REPO)
        except ValueError as exc:
            raise SystemExit(
                "dynamic prediction artifact must live in the Git worktree"
            ) from exc
        manifest["dynamic_artifact"] = {
            "path": str(relative),
            "sha256": common.sha256_file(dynamic_path),
        }
    validate(manifest)
    common.write_json_atomic(args.output, manifest)
    digest = common.sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "kind": args.kind,
                "cells": len(manifest["cells"]),
                "sha256": digest,
                "source_commit": source_commit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
