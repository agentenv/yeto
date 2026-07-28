#!/usr/bin/env python3
"""Build the exact hash-bound v7 main-grid launch manifest after the pilot."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

try:
    import v7_common as common
except ModuleNotFoundError:  # package import in tests
    from scripts import v7_common as common


SHUFFLE_SEED = 20260727


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_cells(pilot: dict, source_commit: str) -> list[dict]:
    variant = pilot["selected_grid"]["variant"]
    grids = pilot["selected_grid"]["eta_grids"]
    cells = []
    for s, t in ((2560, 5), (10240, 20)):
        for mu in (0.0, 0.9):
            key = f"T{t}_mu{'0' if mu == 0.0 else '0.9'}"
            for eta_index, eta in enumerate(grids[key]):
                for seed in common.MAIN_SEEDS:
                    cell = {
                        "s": s,
                        "t": t,
                        "h": 512,
                        "m": 2,
                        "mu": mu,
                        "eta": float(eta),
                        "eta_index": eta_index,
                        "seed": seed,
                        "training_seed": common.training_seed(seed),
                        "fixed_window_tokens": 512 * 4 * 128,
                        "timeout_minutes": 180 if s == 2560 else 420,
                        "estimated_cost_units": 1 if s == 2560 else 4,
                        "retry_group_id": f"T{t}-mu{mu}-seed{seed}",
                        "arm_name": "m2",
                    }
                    mu_label = "mu0" if mu == 0.0 else "mu09"
                    cell["cell_id"] = (
                        f"v7-t{t:02d}-s{s:05d}-{mu_label}-e{eta_index}-seed{seed}"
                    )
                    cell["expected"] = common.expected_for(cell)
                    cells.append(cell)
    expected = 48 if variant == "FULL_48" else 45
    if len(cells) != expected:
        raise RuntimeError(f"{variant}: expected {expected} cells, built {len(cells)}")

    rng = random.Random(SHUFFLE_SEED)
    long_cells = [cell for cell in cells if cell["s"] == 10240]
    short_cells = [cell for cell in cells if cell["s"] == 2560]
    rng.shuffle(long_cells)
    rng.shuffle(short_cells)
    loads = {node: 0 for node in common.NODES}
    queues = {node: [] for node in common.NODES}
    for cell in long_cells + short_cells:
        node = min(common.NODES, key=lambda item: (loads[item], item))
        queues[node].append(cell)
        loads[node] += cell["estimated_cost_units"]

    assigned = []
    for node in common.NODES:
        queue = sorted(
            queues[node],
            key=lambda cell: (
                -cell["estimated_cost_units"],
                cell["t"],
                cell["mu"],
                cell["eta_index"],
                cell["seed"],
            ),
        )
        for index, cell in enumerate(queue):
            cell["assignment"] = {"node": node, "slot": 0}
            cell["slot_queue_index"] = index
            common.bind_commands(cell, source_commit)
            assigned.append(cell)
    return assigned


def validate(cells: list[dict], pilot: dict) -> None:
    variant = pilot["selected_grid"]["variant"]
    expected = 48 if variant == "FULL_48" else 45
    if len(cells) != expected or len({cell["cell_id"] for cell in cells}) != expected:
        raise RuntimeError("v7 main cell count or identity failure")
    counts = {}
    for cell in cells:
        coordinate = (cell["s"], cell["mu"], cell["eta"])
        counts[coordinate] = counts.get(coordinate, 0) + 1
        if common.canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash mismatch: {cell['cell_id']}")
    if any(value != 3 for value in counts.values()):
        raise RuntimeError("every selected eta must have exactly three seeds")
    for node in common.NODES:
        queue = sorted(
            [cell for cell in cells if cell["assignment"]["node"] == node],
            key=lambda cell: cell["slot_queue_index"],
        )
        costs = [cell["estimated_cost_units"] for cell in queue]
        if costs != sorted(costs, reverse=True):
            raise RuntimeError(f"{node}: queue is not longest-first")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-readout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common.verify_frozen_contract()
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("main manifest build requires a clean tracked worktree")
    pilot = json.loads(args.pilot_readout.read_text())
    if (
        pilot.get("schema") != "yeto_outer_mup_v7_pilot_readout_v1"
        or pilot.get("status") != "PASS"
    ):
        raise SystemExit("pilot readout is not a passing frozen v7 pilot")
    source_commit = common.git("rev-parse", "HEAD")
    if pilot.get("source_git_commit") != source_commit:
        raise SystemExit("pilot was not executed at this exact source commit")
    cells = build_cells(pilot, source_commit)
    validate(cells, pilot)
    eta_grids = pilot["selected_grid"]["eta_grids"]
    manifest = {
        "schema": "yeto_outer_mup_v7_27b_lora_launch_manifest_v1",
        "manifest_variant": "v7_27b_lora_raw_tscan",
        "stage": "V7_27B_LORA_GRID",
        "status": "REGISTERED",
        "created_at_utc": utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": common.git("branch", "--show-current"),
        },
        "registration": {"git_commit": source_commit},
        "contract": common.contract_record(),
        "pilot": {
            "path": str(args.pilot_readout.resolve()),
            "sha256": common.sha256_file(args.pilot_readout),
            "selected_eta_star": pilot["pilot_selection"]["selected_eta_star"],
            "projection": pilot["fleet_hour_projection"],
        },
        "grid": {
            "variant": pilot["selected_grid"]["variant"],
            "eta_grids": eta_grids,
        },
        "inputs": {
            "model": {
                "path": str(common.MODEL),
                "revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
                "canonical_inventory_sha256": "32c8f34fa11f07ffde3eedb32435b39a78590ea102b7923bbc1d9b4df7b51c4c",
            },
            "train": {
                "path": str(common.TRAIN),
                "bytes": 64423638,
                "sha256": "e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf",
            },
            "eval": {
                "path": str(common.EVAL),
                "bytes": 4666640,
                "sha256": "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc",
            },
        },
        "scheduling": {
            "policy": "two-node longest-processing-time greedy balance",
            "shuffle_seed": SHUFFLE_SEED,
            "estimated_cost_units": {"S2560": 1, "S10240": 4},
            "parallel_cells": 2,
        },
        "wall_clock": {
            "ceiling_seconds": 108000,
            "start_event": "GRID STARTED",
        },
        "retry_contract": json.loads(common.CONTRACT_JSON.read_text())[
            "retry_contract"
        ],
        "cells": cells,
    }
    common.write_json_atomic(args.output, manifest)
    digest = common.sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve()),
                "sha256": digest,
                "variant": manifest["grid"]["variant"],
                "cells": len(cells),
                "source_git_commit": source_commit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
