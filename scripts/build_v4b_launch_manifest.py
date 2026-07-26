#!/usr/bin/env python3
"""Build the hash-bound 18-cell v4b downward-extension manifest."""

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
GPUS = tuple(range(8))
NODES = ("h200-n1", "h200-n2")
SHUFFLE_SEED = 20260726
RESULT_ROOT = Path("/root/yeto-results-v4b")
DEFAULT_V4_MANIFEST = Path(
    "/root/yeto-results-v4/_controller/launch-v4/launch-manifest-v4.json"
)
MODEL = Path(
    "/root/yeto-hf-cache/hub/models--HuggingFaceTB--SmolLM2-1.7B/"
    "snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71"
)
TRAIN = Path("/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl")
EVAL = Path("/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl")
LONG_SLOTS = tuple((node, gpu) for gpu in range(6) for node in NODES)
V5_INITIAL_SLOTS = tuple((node, gpu) for gpu in (6, 7) for node in NODES)


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
    cells = []
    for curve in contract["extension_design"]["curves"]:
        for eta_index, eta in enumerate(curve["new_etas"]):
            extension_step = 2 - eta_index
            for seed in SEEDS:
                cells.append(
                    {
                        "s": int(curve["s"]),
                        "t": int(curve["t"]),
                        "h": 512,
                        "m": 4,
                        "mu": float(curve["mu"]),
                        "eta": float(eta),
                        "eta_index": eta_index,
                        "extension_step_below_v4_bottom": extension_step,
                        "seed": seed,
                        "training_seed": int(f"{seed}{seed}"),
                        "source_git_commit": source_commit,
                        "retry_group_id": (
                            f"T{curve['t']}-mu{curve['mu']}-seed{seed}"
                        ),
                        "estimated_cost_units": 4 if int(curve["s"]) == 10240 else 1,
                    }
                )
    if len(cells) != 18:
        raise RuntimeError(f"expected 18 cells, built {len(cells)}")

    rng = random.Random(SHUFFLE_SEED)
    long_cells = [cell for cell in cells if cell["s"] == 10240]
    short_cells = [cell for cell in cells if cell["s"] == 2560]
    rng.shuffle(long_cells)
    rng.shuffle(short_cells)
    if len(long_cells) != 12 or len(short_cells) != 6:
        raise RuntimeError("v4b long/short cell identity mismatch")

    slot_cells = {slot: [] for slot in LONG_SLOTS}
    for slot, cell in zip(LONG_SLOTS, long_cells):
        slot_cells[slot].append(cell)
    short_slots = LONG_SLOTS[:6]
    for slot, cell in zip(short_slots, short_cells):
        slot_cells[slot].append(cell)

    assigned = []
    global_index = 0
    for node, gpu in LONG_SLOTS:
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
        for slot_index, cell in enumerate(queue):
            mu_label = "mu0" if cell["mu"] == 0.0 else "mu09"
            cell["cell_id"] = (
                f"v4b-t{cell['t']:02d}-s{cell['s']:05d}-{mu_label}-"
                f"d{cell['extension_step_below_v4_bottom']}-seed{cell['seed']}"
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
                    "allowed_only_under": (
                        "loss-blind whole two-new-eta curve-by-seed retry authority "
                        "for an enumerated infrastructure reason"
                    ),
                }
            ]
            assigned.append(cell)
            global_index += 1
    return assigned


def validate(cells: list[dict]) -> None:
    if len(cells) != 18 or len({cell["cell_id"] for cell in cells}) != 18:
        raise RuntimeError("cell count or identity failure")
    coordinates = {(c["s"], c["mu"], c["eta"], c["seed"]) for c in cells}
    if len(coordinates) != 18:
        raise RuntimeError("scientific coordinate duplication")
    long_cells = [cell for cell in cells if cell["s"] == 10240]
    if {tuple(cell["assignment"].values()) for cell in long_cells} != set(LONG_SLOTS):
        raise RuntimeError("the 12 long cells are not first-wave one-per-slot")
    for node, gpu in LONG_SLOTS:
        queue = sorted(
            [
                cell
                for cell in cells
                if cell["assignment"] == {"node": node, "gpu": gpu}
            ],
            key=lambda cell: cell["slot_queue_index"],
        )
        if not queue or queue[0]["s"] != 10240:
            raise RuntimeError(f"longest-first failure: {node}/gpu{gpu}")
        if any(cell["assignment"]["gpu"] in (6, 7) for cell in queue):
            raise RuntimeError("v4b overlaps the registered initial v5 slots")
    node_counts = {
        node: sum(cell["assignment"]["node"] == node for cell in cells)
        for node in NODES
    }
    if node_counts != {"h200-n1": 9, "h200-n2": 9}:
        raise RuntimeError(f"node balance failure: {node_counts}")
    for cell in cells:
        if canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash failure: {cell['cell_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPO / "experiment-specs/outer-mup-v4b-extension-prereg.json",
    )
    parser.add_argument("--v4-manifest", type=Path, default=DEFAULT_V4_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("manifest build requires clean tracked registration files")
    source_commit = git("rev-parse", "HEAD")
    contract = json.loads(args.contract.read_text())
    if contract.get("schema") != "yeto_outer_mup_v4b_extension_prereg_v1":
        raise SystemExit("not the v4b extension contract")
    analyzer = REPO / "scripts/analyze_v4b.py"
    contract_md = args.contract.with_suffix(".md")
    hashes = {
        "json_sha256": sha256_file(args.contract),
        "md_sha256": sha256_file(contract_md),
        "analyzer_sha256": sha256_file(analyzer),
    }
    if hashes["analyzer_sha256"] != contract["frozen_analyzer"]["sha256"]:
        raise SystemExit("frozen analyzer hash differs from the registered value")
    v4_manifest_sha = sha256_file(args.v4_manifest)
    if v4_manifest_sha != contract["v4_disclosure"]["launch_manifest_sha256"]:
        raise SystemExit("base v4 launch manifest hash differs from registration")
    v4_manifest = json.loads(args.v4_manifest.read_text())
    if (
        v4_manifest.get("schema") != "yeto_outer_mup_v4_scale_launch_manifest_v1"
        or len(v4_manifest.get("cells", [])) != 48
    ):
        raise SystemExit("base v4 manifest is not the complete registered stage")

    cells = build_cells(contract, source_commit)
    validate(cells)
    manifest = {
        "schema": "yeto_outer_mup_v4b_extension_launch_manifest_v1",
        "manifest_variant": "v4b_preoutcome_downward_extension",
        "stage": "V4B_EXTENSION",
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
        "base_v4": {
            "manifest_path": str(args.v4_manifest.resolve()),
            "manifest_sha256": v4_manifest_sha,
            "source_git_commit": v4_manifest["source"]["git_commit"],
            "g4_readout_sha256": contract["v4_disclosure"]["g4_readout_sha256"],
        },
        "inputs": v4_manifest["inputs"],
        "scheduling": {
            "policy": "12 S10240 cells first, one per v4b slot; S2560 cells second on six queues",
            "shuffle_seed": SHUFFLE_SEED,
            "v4b_slots": [
                {"node": node, "gpu": gpu} for node, gpu in LONG_SLOTS
            ],
            "v5_initial_slots": [
                {"node": node, "gpu": gpu} for node, gpu in V5_INITIAL_SLOTS
            ],
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
