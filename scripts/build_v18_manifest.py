#!/usr/bin/env python3
"""Build the hash-bound 96-cell V18 FedAdam launch manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import day3_common as common


REGISTRATION_COMMIT = "7580d076a36af581a1e3ca1b7e8774ba234313d9"
REGISTRATION_PATH = Path("experiment-specs/outer-mup-v18-fedadam-prereg.md")
BRANCH = "experiment/day3-fleet"
SEEDS = (1009, 1013, 1019)
PREDICTIONS = {
    2: 0.742459788403,
    5: 0.540601963459,
    20: 0.485783579482,
    40: 0.583982312067,
}
GRIDS = {
    (2, "sgd"): (0.035532313335, 0.056404031567, 0.089535819044, 0.142129253340),
    (2, "fedadam"): (0.026381313840, 0.041877725342, 0.066476745262, 0.105525255360),
    (5, "sgd"): (0.021709590570, 0.034461826909, 0.054704740288, 0.086838362281),
    (5, "fedadam"): (0.011736247288, 0.018630131291, 0.029573490010, 0.046944989153),
    (20, "sgd"): (0.010963109331, 0.017402851285, 0.027625304437, 0.043852437324),
    (20, "fedadam"): (0.005325698493, 0.008454019390, 0.013419919274, 0.021302793972),
    (40, "sgd"): (0.009817647680, 0.015584544255, 0.024738921945, 0.039270590720),
    (40, "fedadam"): (0.005733332591, 0.009101098187, 0.014447092836, 0.022933330365),
}
H_BY_T = {2: 1280, 5: 512, 20: 128, 40: 64}
V18_SLOTS = (
    ("h200-n1", 5),
    ("h200-n2", 4),
    ("h200-n1", 6),
    ("h200-n2", 5),
    ("h200-n1", 7),
    ("h200-n2", 6),
    ("h200-n2", 7),
)
TRAIN = {
    "path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl",
    "sha256": "e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf",
    "rows": 13758,
}
EVAL = {
    "path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/eval.jsonl",
    "sha256": "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc",
    "rows": 1024,
}


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
        raise SystemExit(f"required V18 execution file is missing: {path}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-input-manifest",
        type=Path,
        required=True,
        help="the identical both-node V19 input proof containing the pinned 135M model bytes",
    )
    parser.add_argument("--syncer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.syncer_sha256) != 64:
        raise SystemExit("syncer SHA-256 must have 64 hex characters")
    head = common.git("rev-parse", "HEAD")
    if remote_branch_tip() != head:
        raise SystemExit("V18 implementation/manifest commit is not public")
    if subprocess.run(
        ["git", "-C", str(common.REPO), "merge-base", "--is-ancestor", REGISTRATION_COMMIT, head]
    ).returncode:
        raise SystemExit("V18 registration commit is not an execution ancestor")
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("tracked worktree changes forbid a V18 manifest")
    model_proof = common.read_json(args.model_input_manifest)
    model = model_proof.get("model")
    if model is None or model.get("revision") != "93efa2f097d58c2a74874c7e644dbc9b0cee75a2":
        raise SystemExit("V18 model proof mismatch")

    registration = file_record(str(REGISTRATION_PATH))
    registration["git_commit"] = REGISTRATION_COMMIT
    execution_paths = (
        "scripts/day3_common.py",
        "scripts/build_v18_manifest.py",
        "scripts/run_day3_queue.py",
        "scripts/analyze_v18.py",
        "scripts/run_slot_v3.py",
        "scripts/compare_diloco.py",
        "syncer/src/merge.rs",
        "syncer/src/main.rs",
        "syncer/src/server.rs",
        "syncer/src/state.rs",
    )
    execution_files = {path: file_record(path) for path in execution_paths}

    queue_records = [
        {
            "queue_id": f"v18-{node}-gpu{gpu}",
            "node": node,
            "gpu": gpu,
            "scientific_cells": 0,
            "retry_groups": [],
        }
        for node, gpu in V18_SLOTS
    ]
    cells = []
    slot_offsets = [0] * len(V18_SLOTS)
    group_index = 0
    # Highest-T groups have the greatest sync overhead and are placed first;
    # this ordering is fixed before any V18 endpoint exists.
    for t in (40, 20, 5, 2):
        for arm in ("sgd", "fedadam"):
            for seed in SEEDS:
                slot_index = group_index % len(V18_SLOTS)
                node, gpu = V18_SLOTS[slot_index]
                queue_id = queue_records[slot_index]["queue_id"]
                retry_group = f"v18-t{t:02d}-{arm}-seed{seed}"
                queue_records[slot_index]["retry_groups"].append(retry_group)
                for eta_index, eta in enumerate(GRIDS[(t, arm)]):
                    cell = {
                        "program": "v18",
                        "stage": "v18",
                        "cell_id": f"v18-t{t:02d}-{arm}-e{eta_index}-seed{seed}",
                        "queue_id": queue_id,
                        "slot_id": f"{node}-gpu{gpu}",
                        "slot_queue_index": slot_offsets[slot_index],
                        "retry_group_id": retry_group,
                        "assignment": {"node": node, "gpus": [gpu]},
                        "arm": arm,
                        "outer_optimizer": "nesterov" if arm == "sgd" else "fedadam",
                        "mu": 0.0 if arm == "sgd" else 0.9,
                        "eta": eta,
                        "eta_index": eta_index,
                        "seed": seed,
                        "training_seed": int(f"{seed}{seed}"),
                        "model_path": model["path"],
                        "train_path": TRAIN["path"],
                        "eval_path": EVAL["path"],
                        "t": t,
                        "h": H_BY_T[t],
                        "s": 2560,
                        "m": 4,
                        "timeout_minutes": 180,
                    }
                    cells.append(common.bind_cell(cell, head))
                    slot_offsets[slot_index] += 1
                    queue_records[slot_index]["scientific_cells"] += 1
                group_index += 1
    if len(cells) != 96 or group_index != 24:
        raise SystemExit("internal V18 cardinality mismatch")

    manifest = {
        "schema": "yeto_day3_launch_manifest_v1",
        "status": "AUTHORIZED",
        "program": "v18",
        "created_at_utc": common.utc_now(),
        "source": {"branch": BRANCH, "git_commit": head, "public_remote_tip": head},
        "registration": registration,
        "execution_files": execution_files,
        "implementation_tests": {
            "targeted_commands": [
                "cargo test fedadam --release",
                "cargo test validates_ema_beta_and_restart_threshold --release",
                "cargo test parses_outer_optimizer_contract --release",
            ],
            "status": "PASS_BEFORE_MANIFEST",
            "full_suite_disclosure": "195 passed; one unrelated pre-existing exact-f64 outer_lr_controller assertion failed by 1e-15",
        },
        "syncer": {"path": "/root/yeto/syncer/target/release/yeto-syncer", "sha256": args.syncer_sha256},
        "inputs": {"model": model, "files": {"train": TRAIN, "eval": EVAL}},
        "contract": {
            "arms": ["sgd", "fedadam"],
            "T": [2, 5, 20, 40],
            "H_by_T": H_BY_T,
            "S": 2560,
            "seeds": list(SEEDS),
            "required_scientific_cells": 96,
            "required_retry_groups": 24,
            "fedadam": {
                "beta1_f32": 0.8999999761581421,
                "beta2_f32": 0.9900000095367432,
                "bias_correction": False,
                "epsilon": 0.0,
                "zero_safe": True,
            },
            "outcome_reads_before_drain": "FORBIDDEN",
        },
        "predictions": {str(t): value for t, value in PREDICTIONS.items()},
        "band_bits": 0.35,
        "bootstrap": {"draws": 10000, "seed": 2026072818, "minimum_valid": 7500},
        "result_root": {"symlink": str(common.RESULT_ROOT), "lvm_target": str(common.RESULT_TARGET)},
        "environment": {
            "HF_DATASETS_CACHE": "/data/hf-datasets-cache",
            "TMPDIR": "/data/tmp",
            "HF_HUB_CACHE": "/root/yeto-hf-cache/hub",
        },
        "queues": queue_records,
        "cells": cells,
        "retry": {
            "attempt_limit": 2,
            "unit": "all four eta rungs sharing T, arm, and seed",
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
                "retry_groups": group_index,
                "queues": len(queue_records),
                "source_commit": head,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
