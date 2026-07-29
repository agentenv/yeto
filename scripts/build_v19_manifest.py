#!/usr/bin/env python3
"""Build the hash-bound V19 launch manifest after input staging and tests."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import day3_common as common


REGISTRATION_COMMIT = "1314f9eec3c392c78c5bedebe3648145328ebb95"
REGISTRATION_PATH = Path("experiment-specs/outer-mup-v19-discrimination-prereg.md")
BRANCH = "experiment/day3-fleet"
ARMS = {
    "mu0": {
        "outer_optimizer": "nesterov",
        "mu": 0.0,
        "etas": (
            0.004741905838138914,
            0.009483811676277829,
            0.018967623352555658,
            0.037935246705111315,
            0.07587049341022263,
            0.15174098682044526,
        ),
    },
    "nesterov_raw": {
        "outer_optimizer": "nesterov",
        "mu": 0.9,
        "etas": (
            0.00044245751734673343,
            0.0008849150346934669,
            0.0017698300693869337,
            0.0035396601387738674,
            0.007079320277547735,
            0.01415864055509547,
        ),
    },
    "heavy_ball": {
        "outer_optimizer": "heavy-ball",
        "mu": 0.9,
        "etas": (
            0.0004552805106771446,
            0.0009105610213542892,
            0.0018211220427085783,
            0.0036422440854171566,
            0.007284488170834313,
            0.014568976341668627,
        ),
    },
}
PREDICTIONS = {
    "C1_C3_INTERFERENCE": 0.6097701376463005,
    "C4_FROZEN_RATCHET": 0.770027996425627,
    "FORENSICS_SAT_EXP": 1.0050466361633321,
    "C1_PURE": 1.2492299609173685,
}
HB_TARGET = 1.0284809176568759
QUEUE_SLOTS = (
    ("h200-n1", 0),
    ("h200-n2", 0),
    ("h200-n1", 1),
    ("h200-n2", 1),
    ("h200-n1", 2),
    ("h200-n2", 2),
    ("h200-n1", 3),
    ("h200-n2", 3),
    ("h200-n1", 4),
)


def remote_branch_tip() -> str:
    result = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"],
        cwd=common.REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    fields = result.stdout.split()
    if len(fields) != 2:
        raise SystemExit(f"cannot resolve public {BRANCH} tip")
    return fields[0]


def file_record(relative: str) -> dict[str, object]:
    path = common.REPO / relative
    if not path.is_file():
        raise SystemExit(f"required V19 execution file is missing: {path}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def validate_inputs(inputs: dict) -> None:
    if inputs.get("schema") != "yeto_outer_mup_v19_inputs_v1":
        raise SystemExit("V19 input manifest schema mismatch")
    if tuple(int(seed) for seed in inputs.get("seeds", {})) != common.V19_SEEDS:
        raise SystemExit("V19 input manifest seed set/order mismatch")
    if inputs.get("source", {}).get("sha256") != (
        "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409"
    ):
        raise SystemExit("V19 source hash mismatch")
    if inputs.get("reference_audit", {}).get("sha256") != (
        "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b"
    ):
        raise SystemExit("V19 audit hash mismatch")
    if inputs.get("training_pool", {}).get("rows") != 13758:
        raise SystemExit("V19 no-wrap training row count mismatch")
    if inputs.get("model", {}).get("revision") != (
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
    ):
        raise SystemExit("V19 model revision mismatch")
    for seed in common.V19_SEEDS:
        record = inputs["seeds"][str(seed)]
        if record.get("training_seed") != int(f"{seed}{seed}"):
            raise SystemExit(f"V19 training seed mismatch for {seed}")
        if record.get("split_rule", {}).get("train_rows") != 13758:
            raise SystemExit(f"V19 train row mismatch for {seed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--syncer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.syncer_sha256) != 64:
        raise SystemExit("syncer SHA-256 must have 64 hex characters")

    head = common.git("rev-parse", "HEAD")
    if remote_branch_tip() != head:
        raise SystemExit("V19 implementation/manifest commit is not the public branch tip")
    if subprocess.run(
        ["git", "-C", str(common.REPO), "merge-base", "--is-ancestor", REGISTRATION_COMMIT, head]
    ).returncode:
        raise SystemExit("V19 registration commit is not an execution ancestor")
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("tracked worktree changes forbid a V19 manifest")
    inputs = common.read_json(args.input_manifest)
    validate_inputs(inputs)

    registration = file_record(str(REGISTRATION_PATH))
    registration["git_commit"] = REGISTRATION_COMMIT
    execution_paths = (
        "scripts/day3_common.py",
        "scripts/build_v19_manifest.py",
        "scripts/run_day3_queue.py",
        "scripts/analyze_v19.py",
        "scripts/prepare_v19_inputs.py",
        "scripts/run_slot_v3.py",
        "scripts/compare_diloco.py",
        "syncer/src/merge.rs",
        "syncer/src/main.rs",
        "syncer/src/state.rs",
    )
    execution_files = {path: file_record(path) for path in execution_paths}

    cells = []
    queue_records = []
    queue_index = 0
    for arm, arm_spec in ARMS.items():
        for seed in common.V19_SEEDS:
            node, gpu = QUEUE_SLOTS[queue_index]
            queue_id = f"v19-{arm}-seed{seed}"
            queue_records.append(
                {
                    "queue_id": queue_id,
                    "node": node,
                    "gpu": gpu,
                    "scientific_cells": 6,
                    "retry_group_id": queue_id,
                }
            )
            seed_inputs = inputs["seeds"][str(seed)]
            for eta_index, eta in enumerate(arm_spec["etas"]):
                cell = {
                    "program": "v19",
                    "stage": "v19",
                    "cell_id": f"v19-{arm}-e{eta_index}-seed{seed}",
                    "queue_id": queue_id,
                    "slot_id": f"{node}-gpu{gpu}",
                    "slot_queue_index": eta_index,
                    "retry_group_id": queue_id,
                    "assignment": {"node": node, "gpus": [gpu]},
                    "arm": arm,
                    "outer_optimizer": arm_spec["outer_optimizer"],
                    "mu": arm_spec["mu"],
                    "eta": eta,
                    "eta_index": eta_index,
                    "seed": seed,
                    "training_seed": int(f"{seed}{seed}"),
                    "model_path": inputs["model"]["path"],
                    "train_path": seed_inputs["files"]["train"]["path"],
                    "eval_path": seed_inputs["files"]["audit"]["path"],
                    "t": 40,
                    "h": 512,
                    "s": 20480,
                    "m": 4,
                    "timeout_minutes": 360,
                }
                cells.append(common.bind_cell(cell, head))
            queue_index += 1
    if len(cells) != 54 or len(queue_records) != 9:
        raise SystemExit("internal V19 design cardinality mismatch")

    manifest = {
        "schema": "yeto_day3_launch_manifest_v1",
        "status": "AUTHORIZED",
        "program": "v19",
        "created_at_utc": common.utc_now(),
        "source": {
            "branch": BRANCH,
            "git_commit": head,
            "public_remote_tip": head,
        },
        "registration": registration,
        "execution_files": execution_files,
        "implementation_tests": {
            "command": "cargo test heavy_ball --release",
            "expected_tests": [
                "heavy_ball_matches_reference_without_lookahead",
                "heavy_ball_v19_first_40_calls_and_fragment_state_are_exact",
                "heavy_ball_off_path_is_bit_identical_to_production_nesterov",
            ],
            "status": "PASS_BEFORE_MANIFEST",
        },
        "syncer": {
            "path": "/root/yeto/syncer/target/release/yeto-syncer",
            "sha256": args.syncer_sha256,
        },
        "inputs": {
            "manifest": {
                "path": "/root/yeto-data/outer-mup-v19/input-manifest.json",
                "sha256": common.sha256_file(args.input_manifest),
            },
            "content": inputs,
        },
        "contract": {
            "T": 40,
            "mu": 0.9,
            "H": 512,
            "S": 20480,
            "arms": ARMS,
            "seeds": list(common.V19_SEEDS),
            "required_scientific_cells": 54,
            "required_queues": 9,
            "endpoint": "frozen confirmation-audit per-token NLL after exactly S steps",
            "outcome_reads_before_drain": "FORBIDDEN",
        },
        "predictions": PREDICTIONS,
        "heavy_ball_target": HB_TARGET,
        "bootstrap": {"draws": 10000, "seed": 2026072819, "minimum_valid": 7500},
        "result_root": {
            "symlink": str(common.RESULT_ROOT),
            "lvm_target": str(common.RESULT_TARGET),
        },
        "environment": {
            "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
            "TMPDIR": "/data/tmp",
            "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
        },
        "queues": queue_records,
        "cells": cells,
        "retry": {
            "attempt_limit": 2,
            "unit": "all six rungs sharing arm and seed",
            "attempt2_supersedes_attempt1": True,
            "finite_endpoint_retry_forbidden": True,
        },
    }
    common.write_json_atomic(args.output, manifest)
    digest = common.sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "cells": len(cells),
                "queues": len(queue_records),
                "source_commit": head,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
