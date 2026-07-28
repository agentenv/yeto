#!/usr/bin/env python3
"""Build the deterministic 18-cell V10 exact-rate launch manifest."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_common import (  # noqa: E402
    V10Error,
    canonical_sha256,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


REPO = Path(__file__).resolve().parent.parent
BRANCH = "experiment/outer-mup-v10-freshtransfer"
CONTRACT_SCHEMA = "yeto_outer_mup_v10_freshtransfer_prereg_v1"
MANIFEST_SCHEMA = "yeto_outer_mup_v10_freshtransfer_launch_manifest_v1"
RESULT_ROOT = Path("/root/yeto-results-v10")
REMOTE_REPO = Path("/root/yeto-v10")
SEEDS = (941, 947, 953)
NODES = ("h200-n1", "h200-n2")
GPUS = tuple(range(8))
ASSIGNMENT_SEED = 20260730
MODEL_PATH = Path(
    "/root/yeto-hf-cache/hub/models--HuggingFaceTB--SmolLM2-1.7B/"
    "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
)
TRAINING = {
    "path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl",
    "bytes": 64423638,
    "sha256": "e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf",
}
HELDOUT = {
    "path": (
        "/root/yeto-data/outer-mup-v3/scale-s2560/raw/confirmation-audit.jsonl"
    ),
    "bytes": 4774107,
    "sha256": "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b",
    "role": "confirmation-audit reserved shard; never used to select V10 etas",
}
SCALE_MANIFEST = {
    "path": "/root/yeto-data/outer-mup-v3/scale-s2560/manifest.json",
    "bytes": 10317,
    "sha256": "b737381d6fbe1ecdfe1c98ae9a4801ef654f0dc65d743a222b7fbbffadc44a36",
}


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise V10Error(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def artifact(path: str) -> dict:
    target = REPO / path
    return {"path": path, "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    return [
        "/root/yeto-venv/bin/python",
        str(REMOTE_REPO / "scripts/compare_diloco.py"),
        "--model",
        str(MODEL_PATH),
        "--data",
        TRAINING["path"],
        "--prebound-development-eval",
        HELDOUT["path"],
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
        "337",
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
        "0.9",
        "--outer-lr",
        repr(cell["eta"]),
        "--fixed-window-microsteps",
        "512",
        "--fixed-window-tokens",
        "65536",
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
        "--arm-timeout-min",
        "720" if cell["t"] == 40 else "480",
        "--work-dir",
        str(attempt / "work"),
        "--report-dir",
        str(attempt / "report"),
    ]


def design_cells(contract: dict) -> list[dict]:
    prescriptions = contract["prescriptions"]
    roles = (
        ("transfer_t5_to_t20", 5, 20, prescriptions["eta_T5"]),
        ("comparator_t20", None, 20, prescriptions["eta_T20"]),
        ("transfer_t5_to_t40", 5, 40, prescriptions["eta_T5"]),
        ("comparator_t40", None, 40, prescriptions["eta_T40"]),
        ("transfer_t20_to_t5", 20, 5, prescriptions["eta_T20"]),
        ("comparator_t5", None, 5, prescriptions["eta_T5"]),
    )
    cells = []
    for role, source_t, target_t, eta in roles:
        for seed in SEEDS:
            cells.append(
                {
                    "role": role,
                    "source_t": source_t,
                    "t": target_t,
                    "s": target_t * 512,
                    "h": 512,
                    "m": 4,
                    "mu": 0.9,
                    "outer_bias_correction": False,
                    "eta": float(eta),
                    "seed": seed,
                    "training_seed": int(f"{seed}{seed}"),
                    "retry_group_id": f"target-t{target_t}-seed{seed}",
                }
            )
    if len(cells) != 18:
        raise V10Error("V10 design cardinality changed")
    return cells


def assign(cells: list[dict]) -> list[dict]:
    rng = random.Random(ASSIGNMENT_SEED)
    rng.shuffle(cells)
    cells.sort(key=lambda cell: -cell["s"])
    # Interleave nodes/islands so long T40 work is not concentrated in one
    # progressively draining v9 island.
    slots = [
        (node, gpu)
        for gpu in (0, 4, 1, 5, 2, 6, 3, 7)
        for node in NODES
    ]
    loads = {(node, gpu): 0 for node, gpu in slots}
    queues = {(node, gpu): [] for node, gpu in slots}
    for cell in cells:
        minimum = min(loads.values())
        candidates = [slot for slot in slots if loads[slot] == minimum]
        slot = candidates[0]
        queues[slot].append(cell)
        loads[slot] += cell["s"]
    assigned = []
    for node, gpu in slots:
        for queue_index, cell in enumerate(queues[node, gpu]):
            cell["assignment"] = {"node": node, "gpu": gpu}
            cell["slot_id"] = f"{node}-gpu{gpu}"
            cell["slot_queue_index"] = queue_index
            assigned.append(cell)
    return assigned


def materialize(cells: list[dict], source_commit: str) -> list[dict]:
    for index, cell in enumerate(cells):
        cell["cell_id"] = (
            f"v10-t{cell['t']:02d}-{cell['role']}-seed{cell['seed']}"
        )
        cell["global_index"] = index
        cell["source_git_commit"] = source_commit
        cell["expected"] = {
            "learner_count": 4,
            "learner_steps_per_learner": cell["s"],
            "outer_steps": 4 * cell["t"],
            "telemetry_rows": 4 * cell["t"],
            "eval_rows": 1024,
            "fixed_window_microsteps": 512,
            "fixed_window_tokens": 65536,
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
                    "loss-blind paired target-horizon/seed retry authority for a "
                    "registered infrastructure failure"
                ),
            }
        ]
    return cells


def validate(cells: list[dict], contract: dict) -> dict:
    errors = []
    if len(cells) != 18 or len({cell["cell_id"] for cell in cells}) != 18:
        errors.append("cell count or identity failure")
    expected_roles = {
        "transfer_t5_to_t20",
        "comparator_t20",
        "transfer_t5_to_t40",
        "comparator_t40",
        "transfer_t20_to_t5",
        "comparator_t5",
    }
    for role in expected_roles:
        role_cells = [cell for cell in cells if cell["role"] == role]
        if {cell["seed"] for cell in role_cells} != set(SEEDS):
            errors.append(f"role {role} does not have exactly the three fresh seeds")
    for cell in cells:
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            errors.append(f"command hash mismatch {cell['cell_id']}")
        if cell["h"] != 512 or cell["s"] != cell["t"] * 512:
            errors.append(f"fixed-H coordinate mismatch {cell['cell_id']}")
        command = cell["command"]
        if "--outer-bias-correction" in command:
            errors.append(f"corrected arm leaked into {cell['cell_id']}")
        if HELDOUT["path"] not in command or TRAINING["path"] not in command:
            errors.append(f"data stream mismatch {cell['cell_id']}")
    if contract["design"]["cell_count"] != len(cells):
        errors.append("contract cell count differs")
    if errors:
        raise V10Error("; ".join(errors))
    return {
        "status": "PASS",
        "cell_count": len(cells),
        "role_count": len(expected_roles),
        "seed_count": len(SEEDS),
        "slot_count": len({cell["slot_id"] for cell in cells}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = read_json(args.contract)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SystemExit("V10 contract schema mismatch")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise SystemExit("V10 manifest requires a clean registration worktree")
    head = git("rev-parse", "HEAD")
    remote = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    fields = remote.stdout.split()
    if remote.returncode or len(fields) != 2 or fields[0] != head:
        raise SystemExit(f"origin/{BRANCH} is not exact local HEAD {head}")
    v9_contract = read_json(REPO / "experiment-specs/outer-mup-v9-sealed-scale-prereg.json")
    model = v9_contract["models"]["smollm2_1p7b"]
    cells = materialize(assign(design_cells(contract)), head)
    validation = validate(cells, contract)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": "V10_FRESH_TRANSFER",
        "status": "REGISTERED",
        "created_at_utc": utc_now(),
        "source": {"branch": BRANCH, "git_commit": head},
        "registration": {
            "git_commit": head,
            "origin_exact": True,
            "pre_outcome": True,
        },
        "contract": {
            "path": str(args.contract),
            "sha256": sha256_file(args.contract),
        },
        "frozen_artifacts": {
            name: artifact(path)
            for name, path in {
                "contract_json": "experiment-specs/outer-mup-v10-freshtransfer-prereg.json",
                "contract_markdown": "experiment-specs/outer-mup-v10-freshtransfer-prereg.md",
                "analyzer": "scripts/analyze_v10.py",
                "common": "scripts/v10_common.py",
                "gatesim": "scripts/gatesim_v10.py",
                "gatesim_report": "experiment-specs/outer-mup-v10-freshtransfer-gatesim.json",
                "preseal_proof": "experiment-specs/outer-mup-v10-preseal-proof.json",
                "manifest_builder": "scripts/build_v10_launch_manifest.py",
                "slot_controller": "scripts/run_slot_v10.py",
            }.items()
        },
        "inputs": {
            "training_jsonl": TRAINING,
            "heldout_audit_jsonl": HELDOUT,
            "scale_manifest": SCALE_MANIFEST,
        },
        "model": model,
        "environment": {
            "HF_HOME": "/root/yeto-hf-cache",
            "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
            "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
            "TMPDIR": "/data/tmp",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "result_root": {
            "link": "/root/yeto-results-v10",
            "target": "/data/yeto-results-v10",
        },
        "retry_contract": contract["retry_contract"],
        "analysis_contract": contract["gate"],
        "assignment": {
            "seed": ASSIGNMENT_SEED,
            "rule": "longest-first greedy over 16 interleaved node/GPU slots",
            "slot_loads_S": {
                f"{node}-gpu{gpu}": sum(
                    cell["s"]
                    for cell in cells
                    if cell["assignment"] == {"node": node, "gpu": gpu}
                )
                for node in NODES
                for gpu in GPUS
            },
        },
        "validation": validation,
        "cells": cells,
    }
    write_json_atomic(args.output.resolve(), manifest)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(args.output.resolve())}  {args.output.name}\n")
    print(
        json.dumps(
            {
                "status": "REGISTERED",
                "cells": len(cells),
                "manifest_sha256": sha256_file(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
