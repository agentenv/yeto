#!/usr/bin/env python3
"""Build the hash-bound v7 wiring-smoke and pilot manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import v7_common as common
except ModuleNotFoundError:  # package import in tests
    from scripts import v7_common as common


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prep_cell(
    *,
    cell_id: str,
    stage: str,
    seed: int,
    s: int,
    h: int,
    t: int,
    mu: float,
    eta: float,
    node: str,
    queue_index: int,
    retry_group: str,
    timeout_minutes: int,
    source_commit: str,
) -> dict:
    cell = {
        "cell_id": cell_id,
        "stage": stage,
        "seed": seed,
        "training_seed": common.training_seed(seed),
        "s": s,
        "h": h,
        "t": t,
        "m": 2,
        "mu": mu,
        "eta": eta,
        "fixed_window_tokens": h * 4 * 128,
        "timeout_minutes": timeout_minutes,
        "assignment": {"node": node, "slot": 0},
        "slot_queue_index": queue_index,
        "retry_group_id": retry_group,
        "arm_name": "m2",
    }
    cell["expected"] = common.expected_for(cell)
    return common.bind_commands(cell, source_commit)


def build_cells(source_commit: str) -> list[dict]:
    cells = [
        prep_cell(
            cell_id="v7-smoke-s00064-h0016-mu0-e0-seed683",
            stage="SMOKE",
            seed=common.SMOKE_SEED,
            s=64,
            h=16,
            t=4,
            mu=0.0,
            eta=0.28,
            node="h200-n1",
            queue_index=0,
            retry_group="v7-smoke",
            timeout_minutes=90,
            source_commit=source_commit,
        )
    ]
    pilot_etas = (0.14, 0.28, 0.56)
    assignments = (("h200-n1", 0), ("h200-n2", 0), ("h200-n1", 1))
    for index, (eta, (node, queue_index)) in enumerate(zip(pilot_etas, assignments)):
        cells.append(
            prep_cell(
                cell_id=f"v7-pilot-t05-s02560-mu0-e{index}-seed691",
                stage="PILOT",
                seed=common.PILOT_SEED,
                s=2560,
                h=512,
                t=5,
                mu=0.0,
                eta=eta,
                node=node,
                queue_index=queue_index,
                retry_group="v7-pilot-all",
                timeout_minutes=180,
                source_commit=source_commit,
            )
        )
    return cells


def validate(cells: list[dict]) -> None:
    if len(cells) != 4 or len({cell["cell_id"] for cell in cells}) != 4:
        raise RuntimeError("v7 prep manifest requires one smoke and three pilot cells")
    if [cell["stage"] for cell in cells].count("SMOKE") != 1:
        raise RuntimeError("v7 prep manifest smoke count changed")
    pilot = [cell for cell in cells if cell["stage"] == "PILOT"]
    if sorted(cell["eta"] for cell in pilot) != [0.14, 0.28, 0.56]:
        raise RuntimeError("v7 pilot eta grid changed")
    for cell in cells:
        if common.canonical_sha256(cell["command"]) != cell["command_hash"]:
            raise RuntimeError(f"command hash mismatch: {cell['cell_id']}")
        retry = cell["registered_retry_commands"][0]
        if common.canonical_sha256(retry["command"]) != retry["command_hash"]:
            raise RuntimeError(f"retry command hash mismatch: {cell['cell_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common.verify_frozen_contract()
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("prep manifest build requires a clean tracked worktree")
    source_commit = common.git("rev-parse", "HEAD")
    cells = build_cells(source_commit)
    validate(cells)
    manifest = {
        "schema": "yeto_outer_mup_v7_27b_lora_prep_manifest_v1",
        "stage": "V7_27B_LORA_PREP",
        "status": "REGISTERED",
        "created_at_utc": utc_now(),
        "source": {
            "git_commit": source_commit,
            "branch": common.git("branch", "--show-current"),
        },
        "registration": {"git_commit": source_commit},
        "contract": common.contract_record(),
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
        "retry_contract": json.loads(common.CONTRACT_JSON.read_text())[
            "retry_contract"
        ],
        "stage_dependencies": {
            "SMOKE": "v6 drain proof and two clean node preflights",
            "PILOT": "completed hash-valid SMOKE evidence",
        },
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
                "cells": len(cells),
                "source_git_commit": source_commit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
