#!/usr/bin/env python3
"""Build the loss-blind 160-cell V14 exact-rate transfer matrix manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import day3_common as common


REGISTRATION_COMMIT = "3b14e87fd5e26f2814c9ea0c8459690e4c5a02d9"
REGISTRATION_PATH = Path("experiment-specs/outer-mup-v14-transfer-matrix-prereg.json")
BRANCH = "experiment/day3-fleet"
V14_SLOTS = (
    ("h200-n1", 1),
    ("h200-n1", 2),
    ("h200-n1", 3),
    ("h200-n1", 4),
    ("h200-n1", 5),
    ("h200-n1", 6),
    ("h200-n1", 7),
)
TRAIN = {
    "path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/train.jsonl",
    "sha256": "e680a29ea8c8fc7c99efdceb4f62e485d3eed1ac2afd15bab43b506cb3f4ecaf",
    "bytes": 64423638,
    "rows": 13758,
}
AUDIT = {
    "path": "/root/yeto-data/outer-mup-v3/scale-s2560/raw/confirmation-audit.jsonl",
    "sha256": "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b",
    "bytes": 4774107,
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
        raise SystemExit(f"required V14 execution file is missing: {path}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": common.sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-proof", type=Path, required=True)
    parser.add_argument("--syncer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.syncer_sha256) != 64:
        raise SystemExit("syncer SHA-256 must have 64 hex characters")
    head = common.git("rev-parse", "HEAD")
    if remote_branch_tip() != head:
        raise SystemExit("V14 implementation/manifest commit is not public")
    if subprocess.run(
        ["git", "-C", str(common.REPO), "merge-base", "--is-ancestor", REGISTRATION_COMMIT, head]
    ).returncode:
        raise SystemExit("V14 registration commit is not an execution ancestor")
    if common.git("status", "--porcelain=v1", "--untracked-files=no"):
        raise SystemExit("tracked worktree changes forbid a V14 manifest")
    model_proof = common.read_json(args.model_proof)
    if model_proof.get("schema") != "yeto_v14_model_proof_v1":
        raise SystemExit("V14 model proof schema mismatch")
    model = model_proof["model"]
    if model.get("revision") != "effd688a12921b4cc83e3312b6feb579f70f9c71":
        raise SystemExit("V14 model revision mismatch")
    contract = common.read_json(common.REPO / REGISTRATION_PATH)

    registration = file_record(str(REGISTRATION_PATH))
    registration["git_commit"] = REGISTRATION_COMMIT
    execution_paths = (
        "scripts/day3_common.py",
        "scripts/build_v14_model_proof.py",
        "scripts/build_v14_manifest.py",
        "scripts/run_day3_queue.py",
        "scripts/day3_status.py",
        "scripts/day3_conductor.py",
        "scripts/analyze_v14.py",
        "scripts/analyze_v19.py",
        "scripts/run_slot_v3.py",
        "scripts/compare_diloco.py",
        "syncer/src/merge.rs",
        "syncer/src/main.rs",
        "syncer/src/state.rs",
    )
    execution_files = {path: file_record(path) for path in execution_paths}
    seeds = tuple(contract["design"]["fresh_seeds"])
    contexts = contract["design"]["contexts"]
    rates = {
        "FIXED_H512": {
            int(key.removeprefix("T")): float(value)
            for key, value in contract["rate_prescriptions"]["fixed_H512"].items()
        },
        "FIXED_S2560": {
            int(key.removeprefix("T")): float(value)
            for key, value in contract["rate_prescriptions"]["fixed_S2560"].items()
        },
    }

    queue_records = [
        {
            "queue_id": f"v14-{node}-gpu{gpu}",
            "node": node,
            "gpu": gpu,
            "scientific_cells": 0,
            "retry_groups": [],
        }
        for node, gpu in V14_SLOTS
    ]
    cells = []
    slot_offsets = [0] * len(V14_SLOTS)
    group_index = 0
    schedule_lookup = {
        context: {int(item["T"]): item for item in items}
        for context, items in contexts.items()
    }
    # Wave 1 starts with the long fixed-H targets, then the cheap fixed-S
    # matrix. Ordering is fixed without reading any V14 endpoint.
    for context in ("FIXED_H512", "FIXED_S2560"):
        for target_t in (40, 20, 5, 2):
            target = schedule_lookup[context][target_t]
            for seed in seeds:
                slot_index = group_index % len(V14_SLOTS)
                node, gpu = V14_SLOTS[slot_index]
                queue_id = queue_records[slot_index]["queue_id"]
                retry_group = f"v14-{context.lower()}-target{target_t:02d}-seed{seed}"
                queue_records[slot_index]["retry_groups"].append(retry_group)
                roles = [("comparator", target_t)] + [
                    ("transfer", source_t)
                    for source_t in (2, 5, 20, 40)
                    if source_t != target_t
                ]
                for role, source_t in roles:
                    role_id = "cmp" if role == "comparator" else f"src{source_t:02d}"
                    cell = {
                        "program": "v14",
                        "stage": "v14",
                        "cell_id": (
                            f"v14-{context.lower()}-target{target_t:02d}-{role_id}-seed{seed}"
                        ),
                        "queue_id": queue_id,
                        "slot_id": f"{node}-gpu{gpu}",
                        "slot_queue_index": slot_offsets[slot_index],
                        "retry_group_id": retry_group,
                        "assignment": {"node": node, "gpus": [gpu]},
                        "arm": role_id,
                        "role": role,
                        "context": context,
                        "source_t": source_t,
                        "target_t": target_t,
                        "outer_optimizer": "nesterov",
                        "mu": 0.9,
                        "eta": rates[context][source_t],
                        "eta_index": 0,
                        "seed": int(seed),
                        "training_seed": int(f"{seed}{seed}"),
                        "model_path": model["path"],
                        "train_path": TRAIN["path"],
                        "eval_path": AUDIT["path"],
                        "max_rows": 13758,
                        "t": target_t,
                        "h": int(target["H"]),
                        "s": int(target["S"]),
                        "m": 4,
                        "timeout_minutes": (
                            720 if int(target["S"]) >= 20480
                            else 480 if int(target["S"]) >= 10240
                            else 240 if int(target["S"]) >= 2560
                            else 180
                        ),
                    }
                    retry_gpus = tuple(slot_gpu for _, slot_gpu in V14_SLOTS)
                    cells.append(common.bind_cell(cell, head, retry_gpus=retry_gpus))
                    slot_offsets[slot_index] += 1
                    queue_records[slot_index]["scientific_cells"] += 1
                group_index += 1
    if len(cells) != 160 or group_index != 40:
        raise SystemExit("internal V14 cardinality mismatch")

    manifest = {
        "schema": "yeto_day3_launch_manifest_v1",
        "status": "AUTHORIZED",
        "program": "v14",
        "created_at_utc": common.utc_now(),
        "source": {"branch": BRANCH, "git_commit": head, "public_remote_tip": head},
        "registration": registration,
        "registration_contract": contract,
        "execution_files": execution_files,
        "syncer": {
            "path": str(common.REMOTE_REPO / "syncer/target/release/yeto-syncer"),
            "sha256": args.syncer_sha256,
        },
        "inputs": {
            "model_proof": {
                "path": "/root/day3-control/v14-model-proof.json",
                "sha256": common.sha256_file(args.model_proof),
            },
            "model": model,
            "files": {"train": TRAIN, "eval": AUDIT},
        },
        "contract": {
            "contexts": ["FIXED_H512", "FIXED_S2560"],
            "T": [2, 5, 20, 40],
            "seeds": list(seeds),
            "required_scientific_cells": 160,
            "required_retry_groups": 40,
            "clip_bits": 0.75,
            "practical_margin_bits": 0.1,
            "student_t_df4": 2.7764451051977987,
            "outcome_reads_before_drain": "FORBIDDEN",
        },
        "result_root": {"symlink": str(common.RESULT_ROOT), "lvm_target": str(common.RESULT_TARGET)},
        "capacity": {
            **common.capacity_contract(estimated_peak_mib=122880),
            "class": "1.7B",
            "launch_authority": "DEFERRED_UNTIL_USER_CLEARS_1P7B_CAPACITY",
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
            "unit": "four cells sharing context, target T, and seed",
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
