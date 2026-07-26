#!/usr/bin/env python3
"""Build the authorized, hash-bound 90-cell v5 SNOO launch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from analyze_v5 import CODE_TRUE_MATCH_FACTOR, CONDITIONS, ETA_GRIDS, SEEDS
except ModuleNotFoundError:  # package import in tests
    from scripts.analyze_v5 import CODE_TRUE_MATCH_FACTOR, CONDITIONS, ETA_GRIDS, SEEDS


REPO = Path(__file__).resolve().parent.parent
GPUS = tuple(range(8))
NODES = ("h200-n1", "h200-n2")
SHUFFLE_SEED = 20260726
RESULT_ROOT = Path("/root/yeto-results-v5")
INPUT_MANIFEST = Path("/root/yeto-data/outer-mup-v5/input-manifest.json")
MODEL = Path("/root/yeto-data/model")
MODEL_FILES_MANIFEST = MODEL / "model-files.sha256"
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


def command_for(cell: dict, attempt_number: int, gpu: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    return [
        "/root/yeto-venv/bin/python",
        "/root/yeto/scripts/compare_diloco.py",
        "--model",
        str(MODEL),
        "--data",
        f"/root/yeto-data/outer-mup-v5/seed-{cell['seed']}/train.jsonl",
        "--prebound-development-eval",
        str(EVAL),
        "--settings",
        "m1",
        "--tuning",
        "full",
        "--skip-baseline",
        "--skip-untrained-eval",
        "--token-budget",
        "327680",
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
        str(gpu),
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
        "0",
        "--learner-delay-jitter-ms",
        "0",
        "--syncer-total-steps",
        "20",
        "--learner-max-steps",
        "2560",
        "--strict-quorum",
        "--pipeline-depth",
        "4",
        "--wan-streams",
        "0",
        "--barrier-sync",
        "--version-matched-anchor",
        "--syncer-checkpoint-every",
        "20",
        "--rho-telemetry",
        "--arm-timeout-min",
        "180",
        "--work-dir",
        str(attempt / "work"),
        "--report-dir",
        str(attempt / "report"),
    ]


def build_cells(source_commit: str) -> list[dict]:
    cells = []
    for condition in CONDITIONS:
        mu = 0.9 if condition == "b" else 0.0
        for eta_index, eta in enumerate(ETA_GRIDS[condition]):
            for seed in SEEDS:
                cells.append(
                    {
                        "condition": condition,
                        "condition_role": {
                            "a": "plain AdamW-equivalent mu0 tuning curve",
                            "b": "SNOO-style outer Nesterov mu0.9",
                            "c": "mu0 code-true effective-eta matched control for b",
                        }[condition],
                        "mu": mu,
                        "eta": eta,
                        "eta_index": eta_index,
                        "seed": seed,
                        "training_seed": int(f"{seed}{seed}"),
                        "scale": "135m",
                        "h": 512,
                        "m": 1,
                        "s": 2560,
                        "t": 5,
                        "arm_name": "m1",
                        "source_git_commit": source_commit,
                        "retry_group_id": f"condition-{condition}-seed-{seed}",
                    }
                )
    if len(cells) != 90:
        raise RuntimeError(f"expected 90 cells, built {len(cells)}")

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(cells)
    slots = [(node, gpu) for gpu in GPUS for node in NODES]
    assigned = []
    for global_index, cell in enumerate(cells):
        node, gpu = slots[global_index % len(slots)]
        slot_index = global_index // len(slots)
        cell["cell_id"] = (
            f"v5-snoo-{cell['condition']}-e{cell['eta_index']}-seed{cell['seed']}"
        )
        cell["assignment"] = {"node": node, "gpu": gpu}
        cell["slot_queue_index"] = slot_index
        cell["global_queue_index"] = global_index
        cell["expected"] = {
            "learner_count": 1,
            "learner_steps_per_learner": 2560,
            "outer_steps": 20,
            "telemetry_rows": 20,
            "eval_rows": 1024,
            "training_tokens": 327680,
        }
        command = command_for(cell, 1, gpu)
        retry = command_for(cell, 2, gpu)
        cell["command"] = command
        cell["command_hash"] = canonical_sha256(command)
        cell["registered_retry_commands"] = [
            {
                "attempt_number": 2,
                "command": retry,
                "command_hash": canonical_sha256(retry),
                "allowed_only_under": (
                    "loss-blind whole condition-by-seed six-eta curve retry authority "
                    "for an enumerated infrastructure reason"
                ),
            }
        ]
        assigned.append(cell)
    return assigned


def validate(cells: list[dict]) -> None:
    if len(cells) != 90 or len({cell["cell_id"] for cell in cells}) != 90:
        raise RuntimeError("cell count or identity failure")
    coordinates = {
        (cell["condition"], cell["eta_index"], cell["seed"]) for cell in cells
    }
    if len(coordinates) != 90:
        raise RuntimeError("scientific coordinate duplication")
    loads = {
        (node, gpu): sum(
            cell["assignment"] == {"node": node, "gpu": gpu} for cell in cells
        )
        for node in NODES
        for gpu in GPUS
    }
    if max(loads.values()) - min(loads.values()) > 1:
        raise RuntimeError(f"slot load imbalance: {loads}")
    node_loads = {
        node: sum(cell["assignment"]["node"] == node for cell in cells)
        for node in NODES
    }
    if set(node_loads.values()) != {45}:
        raise RuntimeError(f"node load imbalance: {node_loads}")
    for cell in cells:
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash failure: {cell['cell_id']}")
    for eta_b, eta_c in zip(ETA_GRIDS["b"], ETA_GRIDS["c"]):
        if not math.isclose(eta_c, eta_b * CODE_TRUE_MATCH_FACTOR, rel_tol=0, abs_tol=1e-15):
            raise RuntimeError("condition-c eta grid is not code-true matched to b")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO / "experiment-specs/outer-mup-v5-snoo-interior-prereg.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires clean tracked registration files")
    source_commit = git("rev-parse", "HEAD")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "yeto_outer_mup_v5_snoo_interior_prereg_v1":
        raise SystemExit("not the v5 SNOO interior contract")
    analyzer = REPO / "scripts/analyze_v5.py"
    contract_md = args.contract.with_suffix(".md")
    hashes = {
        "json_sha256": sha256_file(args.contract),
        "md_sha256": sha256_file(contract_md),
        "analyzer_sha256": sha256_file(analyzer),
    }
    if hashes["analyzer_sha256"] != contract["frozen_analyzer"]["sha256"]:
        raise SystemExit("frozen analyzer hash differs from the registered value")
    input_manifest_sha = sha256_file(INPUT_MANIFEST)
    registered_input = contract["materialized_inputs"]["input_manifest"]
    if input_manifest_sha != registered_input["sha256"]:
        raise SystemExit("v5 input manifest hash differs from the registered value")
    input_manifest = json.loads(INPUT_MANIFEST.read_text())
    if sorted(int(seed) for seed in input_manifest.get("seeds", {})) != list(SEEDS):
        raise SystemExit("v5 input manifest seed set mismatch")
    model_manifest_sha = sha256_file(MODEL_FILES_MANIFEST)
    if model_manifest_sha != contract["materialized_inputs"]["model"]["files_manifest_sha256"]:
        raise SystemExit("model files manifest differs from the registered value")

    cells = build_cells(source_commit)
    validate(cells)
    manifest = {
        "schema": "yeto_outer_mup_v5_snoo_launch_manifest_v1",
        "manifest_variant": "v5_snoo_interior",
        "stage": "V5_SNOO_INTERIOR",
        "status": "AUTHORIZED",
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
        "inputs": {
            "seed_manifest": {
                "path": str(INPUT_MANIFEST),
                "sha256": input_manifest_sha,
                "contents": input_manifest,
            },
            "model": {
                "path": str(MODEL),
                "files_manifest_path": str(MODEL_FILES_MANIFEST),
                "files_manifest_sha256": model_manifest_sha,
            },
            "development_eval": contract["materialized_inputs"]["development_eval"],
        },
        "randomization": {
            "policy": "global seeded shuffle then round-robin over interleaved node/GPU slots",
            "seed": SHUFFLE_SEED,
            "slots": 16,
            "node_cells": {"h200-n1": 45, "h200-n2": 45},
        },
        "wall_clock": contract["wall_clock"],
        "retry_contract": contract["retry_contract"],
        "v4_drain_gate": contract["v4_drain_gate"],
        "cells": cells,
    }
    output = args.output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise SystemExit(f"refusing existing launch manifest: {output}")
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
