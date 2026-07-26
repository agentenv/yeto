#!/usr/bin/env python3
"""Build the authorized, hash-bound 48-cell v4 scale launch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SEEDS = (501, 503, 509)
S_GRID = (2560, 10240)
T_BY_S = {2560: 5, 10240: 20}
MU_GRID = (0.0, 0.9)
GPUS = tuple(range(8))
NODES = ("h200-n1", "h200-n2")
SHUFFLE_SEED = 20260726
RESULT_ROOT = Path("/root/yeto-results-v4")
MODEL = Path(
    "/root/yeto-hf-cache/hub/models--HuggingFaceTB--SmolLM2-1.7B/"
    "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
)
TRAIN = Path("/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl")
EVAL = Path("/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl")


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
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
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


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    outer_steps = 4 * cell["t"]
    timeout_minutes = 180 if cell["s"] == 2560 else 420
    return [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        str(MODEL),
        "--data",
        str(TRAIN),
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
        str(cell["mu"]),
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
        str(outer_steps),
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
        str(outer_steps),
        "--rho-telemetry",
        "--arm-timeout-min",
        str(timeout_minutes),
        "--work-dir",
        str(attempt / "work"),
        "--report-dir",
        str(attempt / "report"),
    ]


def build_cells(contract: dict, source_commit: str) -> list[dict]:
    eta_grids = contract["stage"]["eta_grids"]
    cells = []
    for s in S_GRID:
        t = T_BY_S[s]
        for mu in MU_GRID:
            key = f"T{t}_mu{'0' if mu == 0.0 else '0.9'}"
            etas = eta_grids[key]["etas"]
            if len(etas) != 4:
                raise RuntimeError(f"{key}: expected four registered etas")
            for eta_index, eta in enumerate(etas):
                for seed in SEEDS:
                    cells.append(
                        {
                            "s": s,
                            "t": t,
                            "h": 512,
                            "m": 4,
                            "mu": mu,
                            "eta": float(eta),
                            "eta_index": eta_index,
                            "seed": seed,
                            "training_seed": int(f"{seed}{seed}"),
                            "source_git_commit": source_commit,
                            "retry_group_id": f"T{t}-mu{mu}-seed{seed}",
                            "estimated_cost_units": 4 if s == 10240 else 1,
                        }
                    )
    if len(cells) != 48:
        raise RuntimeError(f"expected 48 cells, built {len(cells)}")

    rng = random.Random(SHUFFLE_SEED)
    long_cells = [cell for cell in cells if cell["s"] == 10240]
    short_cells = [cell for cell in cells if cell["s"] == 2560]
    rng.shuffle(long_cells)
    rng.shuffle(short_cells)
    slots = [(node, gpu) for node in NODES for gpu in GPUS]
    loads = {slot: 0 for slot in slots}
    slot_cells = {slot: [] for slot in slots}
    tie_cursor = 0
    for cell in long_cells + short_cells:
        minimum = min(loads.values())
        candidates = [slot for slot in slots if loads[slot] == minimum]
        slot = candidates[tie_cursor % len(candidates)]
        tie_cursor += 1
        loads[slot] += cell["estimated_cost_units"]
        slot_cells[slot].append(cell)

    global_index = 0
    assigned = []
    for node, gpu in slots:
        queue = sorted(
            slot_cells[(node, gpu)],
            key=lambda cell: (-cell["estimated_cost_units"], cell["s"], cell["mu"], cell["eta_index"], cell["seed"]),
        )
        for slot_index, cell in enumerate(queue):
            mu_label = "mu0" if cell["mu"] == 0.0 else "mu09"
            cell["cell_id"] = (
                f"v4-t{cell['t']:02d}-s{cell['s']:05d}-{mu_label}-"
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
                    "allowed_only_under": "loss-blind whole-curve retry authority for an enumerated infrastructure reason",
                }
            ]
            assigned.append(cell)
            global_index += 1
    return assigned


def validate(cells: list[dict]) -> None:
    if len(cells) != 48 or len({cell["cell_id"] for cell in cells}) != 48:
        raise RuntimeError("cell count or identity failure")
    counts = {}
    for cell in cells:
        key = (cell["s"], cell["mu"], cell["eta"])
        counts[key] = counts.get(key, 0) + 1
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash failure: {cell['cell_id']}")
    if len(counts) != 16 or any(count != 3 for count in counts.values()):
        raise RuntimeError(f"curve-cell balance failure: {counts}")
    for node in NODES:
        for gpu in GPUS:
            queue = sorted(
                [cell for cell in cells if cell["assignment"] == {"node": node, "gpu": gpu}],
                key=lambda cell: cell["slot_queue_index"],
            )
            costs = [cell["estimated_cost_units"] for cell in queue]
            if costs != sorted(costs, reverse=True):
                raise RuntimeError(f"longest-first failure: {node}/gpu{gpu}")
    loads = {
        (node, gpu): sum(
            cell["estimated_cost_units"]
            for cell in cells
            if cell["assignment"] == {"node": node, "gpu": gpu}
        )
        for node in NODES
        for gpu in GPUS
    }
    if max(loads.values()) - min(loads.values()) > 1:
        raise RuntimeError(f"greedy cost imbalance: {loads}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO / "experiment-specs/outer-mup-v4-scale-prereg.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires clean tracked registration files")
    source_commit = git("rev-parse", "HEAD")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "yeto_outer_mup_v4_scale_prereg_v1":
        raise SystemExit("not the v4 scale contract")
    analyzer = REPO / "scripts/analyze_v4.py"
    contract_md = args.contract.with_suffix(".md")
    hashes = {
        "json_sha256": sha256_file(args.contract),
        "md_sha256": sha256_file(contract_md),
        "analyzer_sha256": sha256_file(analyzer),
    }
    if hashes["analyzer_sha256"] != contract["frozen_analyzer"]["sha256"]:
        raise SystemExit(
            "frozen analyzer hash differs from the value recorded in the contract: "
            f"{hashes['analyzer_sha256']}"
        )
    cells = build_cells(contract, source_commit)
    validate(cells)
    manifest = {
        "schema": "yeto_outer_mup_v4_scale_launch_manifest_v1",
        "manifest_variant": "v4_scale_raw_tscan",
        "stage": "V4_SCALE",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "source": {"git_commit": source_commit, "branch": git("branch", "--show-current")},
        "registration": {"git_commit": source_commit},
        "contract": {
            "json_path": str(args.contract.relative_to(REPO)),
            "md_path": str(contract_md.relative_to(REPO)),
            "analyzer_path": str(analyzer.relative_to(REPO)),
            **hashes,
        },
        "inputs": contract["materialized_inputs"],
        "pilot": contract["pilot"],
        "scheduling": {
            "policy": "longest-first greedy estimated cost balance",
            "shuffle_seed": SHUFFLE_SEED,
            "cost_units": {"S2560": 1, "S10240": 4},
            "slots": 16,
        },
        "wall_clock": contract["wall_clock"],
        "retry_contract": contract["retry_contract"],
        "cells": cells,
    }
    write_json_atomic(args.output.resolve(), manifest)
    digest = sha256_file(args.output.resolve())
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve()),
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
