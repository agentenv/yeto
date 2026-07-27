#!/usr/bin/env python3
"""Build the hash-bound, deterministic outer-muP v8 launch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO / "experiment-specs/outer-mup-v8-phasediagram-prereg.json"
NODES = ("h200-n1", "h200-n2")
GPUS = tuple(range(8))
SHUFFLE_SEED = 20260728
EXPECTED_CELLS = 900
RESULT_ROOT = Path("/root/yeto-results-v8")
MODEL = Path("/root/yeto-data/model")
EVAL = Path("/root/yeto-data/splits/seed-337/eval.jsonl")
CANONICAL_INPUT_MANIFEST = Path(
    "/root/yeto-data/outer-mup-v8/input-manifest.json"
)
CANONICAL_TOKEN_REPORT = Path(
    "/root/yeto-data/outer-mup-v8/token-counts-smollm2.json"
)
TIMEOUT_MINUTES = {1024: 120, 2560: 180, 5120: 240, 10240: 360, 20480: 480}


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
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def mu_tag(mu: float) -> str:
    if mu == 0.0:
        return "mu0"
    return "mu" + f"{mu:.2f}".replace("0.", "0p").replace(".", "p")


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    command = [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        str(MODEL),
        "--data",
        f"/root/yeto-data/outer-mup-v8/seed-{cell['seed']}/train.jsonl",
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
    ]
    if cell["outer_bias_correction"]:
        command.append("--outer-bias-correction")
    command.extend(
        [
            "--arm-timeout-min",
            str(TIMEOUT_MINUTES[cell["s"]]),
            "--work-dir",
            str(attempt / "work"),
            "--report-dir",
            str(attempt / "report"),
        ]
    )
    return command


def build_cells(contract: dict, source_commit: str) -> tuple[list[dict], dict]:
    runtime = contract["cost_and_scope_rule"]["observed_v3_attempt1_runtime_seconds_by_S"]
    cells = []
    for curve in contract["design"]["curves"]:
        t = int(curve["T"])
        s = int(curve["S"])
        h = int(curve["H"])
        if h != 512 or s != 512 * t:
            raise RuntimeError(f"T{t}/S{s}: fixed-H closure failure")
        etas = [float(value) for value in curve["eta_grid"]]
        if len(etas) != 4:
            raise RuntimeError(f"T{t}/{curve['arm']}/mu{curve['mu']}: not four eta")
        for eta_index, eta in enumerate(etas):
            for seed in contract["common_protocol"]["seeds"]:
                cells.append(
                    {
                        "t": t,
                        "s": s,
                        "h": h,
                        "m": 4,
                        "arm": str(curve["arm"]),
                        "mu": float(curve["mu"]),
                        "outer_bias_correction": bool(
                            curve["outer_bias_correction"]
                        ),
                        "eta": eta,
                        "eta_index": eta_index,
                        "seed": int(seed),
                        "training_seed": int(f"{seed}{seed}"),
                        "source_git_commit": source_commit,
                        "retry_group_id": (
                            f"T{t}-{curve['arm']}-{mu_tag(float(curve['mu']))}"
                        ),
                        "estimated_runtime_seconds": float(runtime[str(s)]["p90"]),
                    }
                )
    if len(cells) != EXPECTED_CELLS:
        raise RuntimeError(f"expected {EXPECTED_CELLS} cells, built {len(cells)}")

    rng = random.Random(SHUFFLE_SEED)
    ordered = []
    for s in sorted({cell["s"] for cell in cells}, reverse=True):
        stratum = [cell for cell in cells if cell["s"] == s]
        rng.shuffle(stratum)
        ordered.extend(stratum)

    slots = [(node, gpu) for node in NODES for gpu in GPUS]
    loads = {slot: 0.0 for slot in slots}
    queues = {slot: [] for slot in slots}
    tie_cursor = 0
    for cell in ordered:
        minimum = min(loads.values())
        candidates = [slot for slot in slots if abs(loads[slot] - minimum) < 1e-9]
        slot = candidates[tie_cursor % len(candidates)]
        tie_cursor += 1
        cell["assignment"] = {"node": slot[0], "gpu": slot[1]}
        loads[slot] += cell["estimated_runtime_seconds"]
        queues[slot].append(cell)

    assigned = []
    global_index = 0
    for node, gpu in slots:
        for slot_index, cell in enumerate(queues[(node, gpu)]):
            cell["cell_id"] = (
                f"v8-t{cell['t']:02d}-{cell['arm']}-{mu_tag(cell['mu'])}-"
                f"e{cell['eta_index']}-seed{cell['seed']}"
            )
            cell["slot_queue_index"] = slot_index
            cell["global_queue_index"] = global_index
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
                        "loss-blind whole 20-cell paired-curve retry authority "
                        "for an enumerated infrastructure reason"
                    ),
                }
            ]
            assigned.append(cell)
            global_index += 1
    load_summary = {
        f"{node}/gpu{gpu}": loads[(node, gpu)] for node, gpu in slots
    }
    return assigned, load_summary


def validate(cells: list[dict]) -> None:
    if len(cells) != EXPECTED_CELLS or len({cell["cell_id"] for cell in cells}) != EXPECTED_CELLS:
        raise RuntimeError("cell count or cell identity failure")
    curve_counts = {}
    for cell in cells:
        key = (cell["t"], cell["arm"], cell["mu"])
        curve_counts[key] = curve_counts.get(key, 0) + 1
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash failure: {cell['cell_id']}")
        has_correction = "--outer-bias-correction" in cell["command"]
        if has_correction != (cell["arm"] == "corrected"):
            raise RuntimeError(f"correction flag failure: {cell['cell_id']}")
        if cell["h"] != 512 or cell["s"] != 512 * cell["t"]:
            raise RuntimeError(f"fixed-H closure failure: {cell['cell_id']}")
    if len(curve_counts) != 45 or any(count != 20 for count in curve_counts.values()):
        raise RuntimeError("curve balance failure")
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
            s_values = [cell["s"] for cell in queue]
            if s_values != sorted(s_values, reverse=True):
                raise RuntimeError(f"longest-first failure: {node}/gpu{gpu}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--token-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires a clean tracked worktree")
    source_commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    remote = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    remote_commit = (
        remote.stdout.split()[0]
        if remote.returncode == 0 and remote.stdout.split()
        else None
    )
    if remote_commit != source_commit:
        raise SystemExit(f"origin/{branch} is {remote_commit}, expected {source_commit}")
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("schema") != "yeto_outer_mup_v8_phasediagram_prereg_v1":
        raise SystemExit("not the v8 phase-diagram contract")
    if contract["gate_feasibility_sim"]["status"] != "PASS_PREOUTCOME":
        raise SystemExit("v8 feasibility gate is not PASS_PREOUTCOME")
    if contract["cost_and_scope_rule"]["effective_cell_count"] != EXPECTED_CELLS:
        raise SystemExit("v8 registered cost rule did not retain the 900-cell grid")
    input_hash = sha256_file(args.input_manifest.resolve())
    token_hash = sha256_file(args.token_report.resolve())
    if input_hash != contract["machine_inputs"]["combined_manifest_sha256"]:
        raise SystemExit("input-manifest hash differs from contract")
    if token_hash != contract["machine_inputs"]["token_capacity_report_sha256"]:
        raise SystemExit("token-capacity hash differs from contract")
    token_report = json.loads(args.token_report.read_text())
    if token_report.get("status") != "PASS" or token_report[
        "minimum_across_all_seeds_and_learners"
    ]["blocks"] < 20480:
        raise SystemExit("token-capacity report does not prove no-wrap capacity")

    cells, loads = build_cells(contract, source_commit)
    validate(cells)
    md_path = contract_path.with_suffix(".md")
    material = {
        "contract_json": sha256_file(contract_path),
        "contract_md": sha256_file(md_path),
        "analyzer": sha256_file(REPO / "scripts/analyze_v8.py"),
        "gate_simulator": sha256_file(REPO / "scripts/simulate_v8_gate_feasibility.py"),
        "gate_simulation": sha256_file(
            REPO / "experiment-specs/outer-mup-v8-phasediagram-gatesim.json"
        ),
        "input_builder": sha256_file(REPO / "scripts/prepare_inputs_v8.py"),
        "input_verifier": sha256_file(REPO / "scripts/verify_inputs_v8.py"),
        "manifest_builder": sha256_file(Path(__file__).resolve()),
        "runner": sha256_file(REPO / "scripts/run_slot_v8.py"),
        "gate_checker": sha256_file(REPO / "scripts/check_v8_gates.py"),
        "launch_authorizer": sha256_file(REPO / "scripts/authorize_v8_launch.py"),
        "retry_authorizer": sha256_file(REPO / "scripts/authorize_v8_retry.py"),
    }
    for name, observed in material.items():
        registered = {
            "analyzer": contract["frozen_analyzer"]["sha256"],
            "gate_simulator": contract["gate_feasibility_sim"]["script"]["sha256"],
            "gate_simulation": contract["gate_feasibility_sim"]["artifact"]["sha256"],
            "input_builder": contract["machine_inputs"]["input_builder"]["sha256"],
            "input_verifier": contract["machine_inputs"]["token_verifier"]["sha256"],
            "manifest_builder": contract["execution_scripts"]["manifest_builder"]["sha256"],
            "runner": contract["execution_scripts"]["runner"]["sha256"],
            "gate_checker": contract["execution_scripts"]["gate_checker"]["sha256"],
            "launch_authorizer": contract["execution_scripts"]["launch_authorizer"]["sha256"],
            "retry_authorizer": contract["execution_scripts"]["retry_authorizer"]["sha256"],
        }.get(name)
        if registered is not None and registered != observed:
            raise SystemExit(f"{name} hash differs from contract")
    manifest = {
        "schema": "yeto_outer_mup_v8_launch_manifest_v1",
        "stage": "V8_PHASE_DIAGRAM",
        "status": "REGISTERED",
        "created_at_utc": utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": branch,
            "pushed_branch_tip": remote_commit,
            "clean": True,
        },
        "registration": {
            "git_commit": source_commit,
            "is_execution_commit": True,
        },
        "contract": {
            "json_path": str(contract_path),
            "json_sha256": material["contract_json"],
            "md_path": str(md_path),
            "md_sha256": material["contract_md"],
            "analyzer_sha256": material["analyzer"],
        },
        "material_hashes": material,
        "inputs": {
            "input_manifest": {
                "path": str(CANONICAL_INPUT_MANIFEST),
                "sha256": input_hash,
            },
            "token_capacity": {
                "report_path": str(CANONICAL_TOKEN_REPORT),
                "report_sha256": token_hash,
                "minimum_complete_blocks": contract["machine_inputs"]["minimum_complete_blocks_per_learner"],
                "required_complete_blocks": 20480,
                "verifier": contract["machine_inputs"]["token_verifier"],
            },
            "input_builder": contract["machine_inputs"]["input_builder"],
            "model": {
                "path": "/root/yeto-data/model",
                "id": contract["common_protocol"]["model"],
                "revision": contract["common_protocol"]["model_revision"],
                "files": {},
            },
        },
        "reuse": contract["reuse_audit"],
        "cost_and_scope_rule": contract["cost_and_scope_rule"],
        "queue_policy": {
            "nodes": list(NODES),
            "gpus_per_node": list(GPUS),
            "shuffle_seed": SHUFFLE_SEED,
            "longest_S_first": True,
            "balance_weight": "registered measured v3 p90 runtime seconds",
            "estimated_load_seconds": loads,
        },
        "retry_contract": contract["evidence_and_retry_contract"],
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
                "output": str(args.output.resolve()),
                "sha256": digest,
                "cells": len(cells),
                "min_load_seconds": min(loads.values()),
                "max_load_seconds": max(loads.values()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
