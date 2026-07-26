#!/usr/bin/env python3
"""Build the hash-bound 44-cell v4c seed-power launch manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from analyze_v4c import ADDED_SEEDS, COMBINED_ETA_GRIDS, T_BY_S
except ModuleNotFoundError:  # package import in tests
    from scripts.analyze_v4c import ADDED_SEEDS, COMBINED_ETA_GRIDS, T_BY_S


REPO = Path(__file__).resolve().parent.parent
EXECUTION_REPO = Path("/root/yeto-v4c")
NODES = ("h200-n1", "h200-n2")
INITIAL_SLOTS = tuple((node, gpu) for gpu in range(6) for node in NODES)
DEFERRED_SLOTS = tuple((node, gpu) for gpu in (6, 7) for node in NODES)
ALL_SLOTS = INITIAL_SLOTS + DEFERRED_SLOTS
SHUFFLE_SEED = 20260726
RESULT_ROOT = Path("/root/yeto-results-v4c")
DEFAULT_V4_MANIFEST = Path(
    "/root/yeto-results-v4/_controller/launch-v4/launch-manifest-v4.json"
)
DEFAULT_V4B_MANIFEST = Path(
    "/root/yeto-results-v4b/_controller/launch-v4b/launch-manifest-v4b.json"
)
DEFAULT_G4B_READOUT = Path("/root/g4b-readout.json")
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


def command_for(cell: dict, attempt_number: int) -> list[str]:
    attempt = RESULT_ROOT / cell["cell_id"] / f"attempt-{attempt_number}"
    outer_steps = 4 * cell["t"]
    timeout_minutes = 180 if cell["s"] == 2560 else 420
    return [
        "/root/yeto-venv/bin/python",
        str(EXECUTION_REPO / "scripts/compare_diloco.py"),
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


def unassigned_cells(source_commit: str) -> list[dict]:
    cells = []
    for (s, mu), etas in COMBINED_ETA_GRIDS.items():
        t = T_BY_S[s]
        for eta_index, eta in enumerate(etas):
            for seed in ADDED_SEEDS:
                cells.append(
                    {
                        "s": s,
                        "t": t,
                        "h": 512,
                        "m": 4,
                        "mu": mu,
                        "eta": eta,
                        "eta_index": eta_index,
                        "seed": seed,
                        "training_seed": int(f"{seed}{seed}"),
                        "source_git_commit": source_commit,
                        "retry_group_id": f"T{t}-mu{mu}-seed{seed}",
                        "estimated_cost_units": 4 if s == 10240 else 1,
                    }
                )
    return cells


def build_cells(source_commit: str) -> list[dict]:
    cells = unassigned_cells(source_commit)
    if len(cells) != 44:
        raise RuntimeError(f"expected 44 cells, built {len(cells)}")
    rng = random.Random(SHUFFLE_SEED)
    long_cells = [cell for cell in cells if cell["s"] == 10240]
    short_cells = [cell for cell in cells if cell["s"] == 2560]
    rng.shuffle(long_cells)
    rng.shuffle(short_cells)
    if len(long_cells) != 24 or len(short_cells) != 20:
        raise RuntimeError("v4c long/short identity mismatch")

    slot_cells = {slot: [] for slot in ALL_SLOTS}
    for slot, cell in zip(INITIAL_SLOTS, long_cells[:12]):
        slot_cells[slot].append(cell)
    for slot, cell in zip(DEFERRED_SLOTS, long_cells[12:16]):
        slot_cells[slot].append(cell)
    for slot, cell in zip(INITIAL_SLOTS[:8], long_cells[16:]):
        slot_cells[slot].append(cell)
    for slot_index, slot in enumerate(INITIAL_SLOTS[8:]):
        slot_cells[slot].extend(short_cells[slot_index * 4 : (slot_index + 1) * 4])
    for slot, cell in zip(DEFERRED_SLOTS, short_cells[16:]):
        slot_cells[slot].append(cell)

    assigned = []
    global_index = 0
    for node, gpu in ALL_SLOTS:
        queue = sorted(
            slot_cells[(node, gpu)],
            key=lambda cell: (
                -cell["estimated_cost_units"],
                cell["s"],
                cell["mu"],
                cell["eta_index"],
                cell["seed"],
            ),
        )
        for queue_index, cell in enumerate(queue):
            mu_label = "mu0" if cell["mu"] == 0.0 else "mu09"
            cell["cell_id"] = (
                f"v4c-t{cell['t']:02d}-s{cell['s']:05d}-{mu_label}-"
                f"e{cell['eta_index']}-seed{cell['seed']}"
            )
            cell["assignment"] = {"node": node, "gpu": gpu}
            cell["claim_wave"] = "initial" if gpu < 6 else "deferred_after_v5b"
            cell["slot_queue_index"] = queue_index
            cell["global_queue_index"] = global_index
            cell["expected"] = {
                "learner_count": 4,
                "learner_steps_per_learner": cell["s"],
                "outer_steps": 4 * cell["t"],
                "telemetry_rows": 4 * cell["t"],
                "eval_rows": 1024,
            }
            initial = command_for(cell, 1)
            retry = command_for(cell, 2)
            cell["command"] = initial
            cell["command_hash"] = canonical_sha256(initial)
            cell["registered_retry_commands"] = [
                {
                    "attempt_number": 2,
                    "command": retry,
                    "command_hash": canonical_sha256(retry),
                    "allowed_only_under": (
                        "loss-blind whole combined-grid curve-by-added-seed "
                        "retry authority for an enumerated infrastructure reason"
                    ),
                }
            ]
            assigned.append(cell)
            global_index += 1
    return assigned


def validate(cells: list[dict]) -> None:
    if len(cells) != 44 or len({cell["cell_id"] for cell in cells}) != 44:
        raise RuntimeError("cell count or identity failure")
    coordinate_counts = {}
    for cell in cells:
        key = (cell["s"], cell["mu"], cell["eta"])
        coordinate_counts[key] = coordinate_counts.get(key, 0) + 1
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash failure: {cell['cell_id']}")
    if len(coordinate_counts) != 22 or set(coordinate_counts.values()) != {2}:
        raise RuntimeError(f"eta/additional-seed balance failure: {coordinate_counts}")
    if sum(cell["s"] == 10240 for cell in cells) != 24:
        raise RuntimeError("long-cell count failure")
    if sum(cell["s"] == 2560 for cell in cells) != 20:
        raise RuntimeError("short-cell count failure")
    node_counts = {
        node: sum(cell["assignment"]["node"] == node for cell in cells)
        for node in NODES
    }
    if node_counts != {"h200-n1": 22, "h200-n2": 22}:
        raise RuntimeError(f"node balance failure: {node_counts}")
    loads = {}
    for node, gpu in ALL_SLOTS:
        queue = sorted(
            [
                cell
                for cell in cells
                if cell["assignment"] == {"node": node, "gpu": gpu}
            ],
            key=lambda cell: cell["slot_queue_index"],
        )
        costs = [cell["estimated_cost_units"] for cell in queue]
        if not queue or costs != sorted(costs, reverse=True):
            raise RuntimeError(f"longest-first failure: {node}/gpu{gpu}")
        loads[(node, gpu)] = sum(costs)
    if any(loads[slot] != 8 for slot in INITIAL_SLOTS):
        raise RuntimeError(f"initial queue load failure: {loads}")
    if any(loads[slot] != 5 for slot in DEFERRED_SLOTS):
        raise RuntimeError(f"deferred queue load failure: {loads}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO / "experiment-specs/outer-mup-v4c-seedpower-prereg.json",
    )
    parser.add_argument("--v4-manifest", type=Path, default=DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4b-manifest", type=Path, default=DEFAULT_V4B_MANIFEST)
    parser.add_argument("--g4b-readout", type=Path, default=DEFAULT_G4B_READOUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires a clean tracked registration checkout")
    source_commit = git("rev-parse", "HEAD")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "yeto_outer_mup_v4c_seedpower_prereg_v1":
        raise SystemExit("not the v4c seed-power contract")
    analyzer = REPO / "scripts/analyze_v4c.py"
    dependency = REPO / "scripts/analyze_v4b.py"
    contract_md = args.contract.with_suffix(".md")
    hashes = {
        "json_sha256": sha256_file(args.contract),
        "md_sha256": sha256_file(contract_md),
        "analyzer_sha256": sha256_file(analyzer),
        "analyzer_dependency_sha256": sha256_file(dependency),
    }
    if hashes["analyzer_sha256"] != contract["frozen_analyzer"]["sha256"]:
        raise SystemExit("frozen analyzer hash differs from registration")
    if (
        hashes["analyzer_dependency_sha256"]
        != contract["frozen_analyzer"]["frozen_dependency"]["sha256"]
    ):
        raise SystemExit("frozen analyzer dependency differs from registration")

    expected = contract["g4b_disclosure"]
    observed_hashes = {
        "v4_manifest_sha256": sha256_file(args.v4_manifest),
        "v4b_manifest_sha256": sha256_file(args.v4b_manifest),
        "g4b_readout_sha256": sha256_file(args.g4b_readout),
    }
    if observed_hashes["v4_manifest_sha256"] != expected["v4_manifest_sha256"]:
        raise SystemExit("v4 manifest hash differs from disclosed evidence")
    if observed_hashes["v4b_manifest_sha256"] != expected["v4b_manifest_sha256"]:
        raise SystemExit("v4b manifest hash differs from disclosed evidence")
    if observed_hashes["g4b_readout_sha256"] != expected["readout_sha256"]:
        raise SystemExit("G4B readout hash differs from disclosed evidence")
    v4_manifest = json.loads(args.v4_manifest.read_text())
    v4b_manifest = json.loads(args.v4b_manifest.read_text())
    g4b = json.loads(args.g4b_readout.read_text())
    if len(v4_manifest.get("cells", [])) != 48 or len(v4b_manifest.get("cells", [])) != 18:
        raise SystemExit("base manifests are incomplete")
    if (
        g4b.get("gate", {}).get("verdict") != "NOT_EVALUABLE"
        or g4b.get("bootstrap", {}).get("valid_replicates") != 7067
        or not all(fit.get("interior") for fit in g4b.get("curve_fits", []))
    ):
        raise SystemExit("G4B readout does not establish the registered trigger")

    cells = build_cells(source_commit)
    validate(cells)
    manifest = {
        "schema": "yeto_outer_mup_v4c_seedpower_launch_manifest_v1",
        "manifest_variant": "v4c_five_seed_combined_grids",
        "stage": "V4C_SEED_POWER",
        "status": "AUTHORIZED",
        "created_at_utc": utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": git("branch", "--show-current"),
            "execution_repo": str(EXECUTION_REPO),
        },
        "registration": {"git_commit": source_commit},
        "contract": {
            "json_path": str(args.contract.relative_to(REPO)),
            "md_path": str(contract_md.relative_to(REPO)),
            "analyzer_path": str(analyzer.relative_to(REPO)),
            "analyzer_dependency_path": str(dependency.relative_to(REPO)),
            **hashes,
        },
        "base_evidence": {
            **observed_hashes,
            "v4_manifest_path": str(args.v4_manifest.resolve()),
            "v4b_manifest_path": str(args.v4b_manifest.resolve()),
            "g4b_readout_path": str(args.g4b_readout.resolve()),
        },
        "inputs": v4_manifest["inputs"],
        "scheduling": {
            "policy": "registered v5b-aware longest-first static queues",
            "shuffle_seed": SHUFFLE_SEED,
            "initial_slots": [
                {"node": node, "gpu": gpu} for node, gpu in INITIAL_SLOTS
            ],
            "deferred_after_v5b_slots": [
                {"node": node, "gpu": gpu} for node, gpu in DEFERRED_SLOTS
            ],
            "estimated_cost_units": {"S2560": 1, "S10240": 4},
            "no_gpu_overlap": True,
        },
        "wall_clock": contract["wall_clock"],
        "retry_contract": contract["retry_contract"],
        "cells": cells,
    }
    output = args.output.resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise SystemExit(f"refusing existing launch manifest: {output}")
    write_json_atomic(output, manifest)
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n")
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
