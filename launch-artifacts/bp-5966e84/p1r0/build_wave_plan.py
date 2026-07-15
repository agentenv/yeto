#!/usr/bin/env python3
"""Build the deterministic pre-parent-binding P1-R0 wave plan."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from collections import defaultdict
from pathlib import Path


PACKET = Path(__file__).resolve().parent
SOURCE_TREE = Path("/tmp/yeto-prod-8d58208")
SOURCE_COMMIT = "8d58208cacafef12cb95f2642b4fa700531151b4"
MASTER_SEED_HEX = "0728fa50c14f4e52113407ab12e173b7ef4eb3b3b36f192ec7b814dd411223c5"
MASTER_SEED_PREIMAGE = (
    "yeto-best-paper-parallel-cells-v1|"
    "16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80|"
    "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
)
STUDY_ID = "bp-phase-map-p1-r0"
CAMPAIGN_LABEL = "bp-p1r0-5966e84-20260715a"
SCIENCE_ROOT = f"/opt/yeto-science/p1r0/{CAMPAIGN_LABEL}"
REMOTE_REPO = "/tmp/yeto-best-paper"
BUCKET = "gs://yeto-exp2-52-model-training-497007"
SLOTS = ("v0", "v1", "v2", "v3")
RUN_IDS = {
    slot: f"bp-p1r0-w{index + 1}-5966e84-20260715a"
    for index, slot in enumerate(SLOTS)
}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rank(domain: str, token: str) -> str:
    payload = (
        bytes.fromhex(MASTER_SEED_HEX)
        + b"\x00"
        + domain.encode("utf-8")
        + b"\x00"
        + STUDY_ID.encode("utf-8")
        + b"\x00"
        + token.encode("utf-8")
    )
    return sha256_bytes(payload)


def ranked(tokens: list[str], domain: str) -> list[str]:
    return sorted(tokens, key=lambda token: (bytes.fromhex(rank(domain, token)), token.encode()))


def load_source():
    head = subprocess.run(
        ["git", "-C", str(SOURCE_TREE), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(SOURCE_TREE),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != SOURCE_COMMIT or dirty:
        raise SystemExit("P1 plan source checkout is not exact and clean")
    spec = importlib.util.spec_from_file_location(
        "p1r0_run_phase_map", SOURCE_TREE / "scripts" / "run_phase_map.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load exact run_phase_map source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_plan(module):
    argv = [
        "--study-id",
        STUDY_ID,
        "--study-phase",
        "p1_development",
        "--run-dir",
        f"{SCIENCE_ROOT}/phase-map",
        "--artifact-uri",
        f"{BUCKET}/{CAMPAIGN_LABEL}/campaign/phase-map",
        "--git-commit",
        SOURCE_COMMIT,
        "--python-executable",
        "/home/shou/venv/bin/python",
        "--command-repo-root",
        REMOTE_REPO,
        "--image-digest",
        "038098c2b5356c9117f1019bf0d19c8999ab50f259dceb041a57fcf657d2620f",
        "--image-numeric-id",
        "7290368630472593484",
        "--model-path",
        f"{SCIENCE_ROOT}/inputs/model",
        "--model-revision",
        "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
        "--data",
        f"{SCIENCE_ROOT}/inputs/train.parquet",
        "--provider-evidence",
        f"{SCIENCE_ROOT}/provider/provider-evidence.json",
        "--h",
        "16,64,256",
        "--mu",
        "0,.5,.9",
        "--eta",
        ".021875,.04375,.0875,.175",
        "--seed",
        "347",
        "--training-seed",
        "347347",
        "--order-seed",
        "20260714",
        "--eval-split-seed",
        "331",
        "--token-budget",
        "655360",
        "--seq-len",
        "128",
        "--micro-batch-size",
        "1",
        "--inner-lr",
        "0.001",
        "--train-rows",
        "5000",
        "--eval-rows",
        "1024",
        "--confirmation-audit-rows",
        "1024",
        "--device",
        "cuda",
        "--gpu-slots",
        "4",
        "--syncer-checkpoint-every",
        "4",
        "--arm-timeout-min",
        "240",
        "--resource-class",
        "a2-highgpu-4g",
    ]
    args = module.build_parser().parse_args(argv)
    return argv, module.build_plan(args)


def main() -> None:
    if sha256_bytes(MASTER_SEED_PREIMAGE.encode("utf-8")) != MASTER_SEED_HEX:
        raise SystemExit("parallel master-seed preimage does not reproduce the amendment")
    module = load_source()
    argv, plan = source_plan(module)
    write_json(PACKET / "scientific-materialize-argv.prebinding.json", argv)
    write_json(PACKET / "scientific-randomization-plan.json", plan)

    launch_cells = []
    for cell in sorted(plan["cells"], key=lambda item: item["cell_id"].encode("utf-8")):
        launch_cells.append(
            {
                "cell_id": cell["cell_id"],
                "block_id": cell["randomization"]["block_id"],
                "h": cell["H"],
                "mu": cell["mu"],
                "eta": cell["eta"],
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "paired_control_id": cell["paired_control_id"],
                "command_hash": cell["command_hash"],
                "normalized_workload_command_hash": module.sha256_bytes(
                    module.canonical_json(
                        module.normalized_workload_command(cell["command"])
                    )
                ),
            }
        )
    write_json(PACKET / "launch-cells.json", launch_cells)

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    for cell in launch_cells:
        by_group[str(cell["block_id"])].append(cell)
    group_ids = ranked(list(by_group), "wave")
    waves = []
    wave_dir = PACKET / "waves"
    vm_rows: dict[str, list[dict[str, object]]] = {slot: [] for slot in SLOTS}
    for wave_index, group_id in enumerate(group_ids):
        cells = by_group[group_id]
        cell_ids = [str(cell["cell_id"]) for cell in cells]
        arm_order = ranked(cell_ids, f"arm-order|{group_id}|1")
        slot_order = ranked(list(SLOTS), f"slot-order|{group_id}|1")
        assignments = dict(zip(arm_order, slot_order))
        launch_order = ranked(cell_ids, f"launch-order|{group_id}|1")
        cell_by_id = {str(cell["cell_id"]): cell for cell in cells}
        assigned_rows = []
        for cell_id in launch_order:
            slot = assignments[cell_id]
            cell = cell_by_id[cell_id]
            row = {
                **cell,
                "logical_slot": slot,
                "requested_vm_run_id": RUN_IDS[slot],
                "requested_vm_artifact_prefix": f"{BUCKET}/{RUN_IDS[slot]}",
                "arm_order_index": arm_order.index(cell_id),
                "launch_order_index": launch_order.index(cell_id),
                "arm_rank": rank(f"arm-order|{group_id}|1", cell_id),
                "launch_rank": rank(f"launch-order|{group_id}|1", cell_id),
                "slot_rank": rank(f"slot-order|{group_id}|1", slot),
            }
            assigned_rows.append(row)
            vm_rows[slot].append(
                {
                    "wave_index": wave_index,
                    "time_block_index": wave_index,
                    "group_id": group_id,
                    "launch_order_index": row["launch_order_index"],
                    "cell_id": cell_id,
                    "command_hash": cell["command_hash"],
                }
            )
        idle_slot = next(slot for slot in SLOTS if slot not in assignments.values())
        sample = cells[0]
        wave = {
            "schema": "yeto_p1r0_prebound_wave_manifest_v1",
            "status": "PREBOUND_SCHEDULE_ONLY_NOT_LAUNCH_AUTHORITY",
            "stage_code": "p1r0",
            "study_id": STUDY_ID,
            "source_commit": SOURCE_COMMIT,
            "amendment_path": "docs/AMENDMENT-parallel-cells.md",
            "amendment_raw_sha256": "e2c87fd6c2ec0e4b91f488b5771334e0befd175560a3e2ccfcf349be1ee8b3dd",
            "master_seed": MASTER_SEED_HEX,
            "attempt_round": 1,
            "wave_index": wave_index,
            "time_block_index": wave_index,
            "group_id": group_id,
            "group_rank": rank("wave", group_id),
            "h": sample["h"],
            "eta": sample["eta"],
            "seed": sample["seed"],
            "training_seed": sample["training_seed"],
            "assigned_cells_in_dispatch_order": assigned_rows,
            "idle_slot": idle_slot,
            "idle_requested_vm_run_id": RUN_IDS[idle_slot],
            "dispatch_span_limit_seconds": 60,
            "scientific_start_span_limit_seconds": 120,
            "all_assigned_vms_ready_before_dispatch": True,
            "parent_p0b_manifest_sha256": None,
            "parent_p0b_replay_report_sha256": None,
            "launch_eligible": False,
        }
        wave["prebinding_manifest_canonical_sha256"] = sha256_bytes(
            canonical_json(wave)
        )
        waves.append(wave)
        write_json(wave_dir / f"wave-{wave_index + 1:02d}.json", wave)

    vm_dir = PACKET / "vms"
    for slot in SLOTS:
        vm = {
            "schema": "yeto_p1r0_requested_vm_plan_v1",
            "status": "PREBOUND_SCHEDULE_ONLY_NOT_LAUNCH_AUTHORITY",
            "logical_slot": slot,
            "requested_run_id": RUN_IDS[slot],
            "artifact_prefix": f"{BUCKET}/{RUN_IDS[slot]}",
            "controller_state_path": f"/tmp/yeto-p1r0-state/{RUN_IDS[slot]}.json",
            "machine_type": "a2-highgpu-4g",
            "accelerator_count": 4,
            "provisioning_model": "SPOT",
            "termination_action": "DELETE",
            "automatic_restart": False,
            "on_host_maintenance": "TERMINATE",
            "assigned_wave_cells": vm_rows[slot],
            "assigned_cell_count": len(vm_rows[slot]),
            "launch_eligible": False,
        }
        write_json(vm_dir / f"{slot}.json", vm)

    schedule = {
        "schema": "yeto_p1r0_prebound_schedule_v1",
        "status": "PREBOUND_SCHEDULE_ONLY_NOT_LAUNCH_AUTHORITY",
        "stage_code": "p1r0",
        "study_id": STUDY_ID,
        "operator_label_commit": "5966e84",
        "source_commit": SOURCE_COMMIT,
        "scientific_randomization_plan_hash": plan["randomization_plan_hash"],
        "master_seed": MASTER_SEED_HEX,
        "science_root": SCIENCE_ROOT,
        "logical_slots": list(SLOTS),
        "requested_vm_run_ids": RUN_IDS,
        "maximum_concurrent_scientific_cells": 4,
        "maximum_campaign_attached_a100s": 16,
        "realized_cells_per_wave": 3,
        "wave_count": len(waves),
        "launch_cell_count": len(launch_cells),
        "waves": waves,
        "deferred_binding": {
            "p0b_final_manifest_canonical_sha256": None,
            "p0b_replay_report_raw_sha256": None,
            "parallel_roster_hash": None,
            "parallel_plan_hash": None,
            "bound_p1_manifest_canonical_sha256": None,
        },
        "launch_eligible": False,
        "blocking_gates": [
            "P0b must launch, seal, undergo exact-ID teardown, and pass CPU replay",
            "the final P0b manifest and replay hashes must be bound into P1-R0",
            "the amendment-native roster_hash and parallel_plan_hash must be generated",
            "the parallel executor/partial-manifest/aggregator implementation and conformance tests are not present in the source commit",
            "the requested w1..w4 aliases do not satisfy the amendment's normative physical-generation run-ID grammar",
            "byte-identical ICML and statistics reviews plus explicit launch authority are required",
        ],
    }
    schedule["prebinding_schedule_canonical_sha256"] = sha256_bytes(
        canonical_json(schedule)
    )
    write_json(PACKET / "wave-plan.prebinding.json", schedule)


if __name__ == "__main__":
    main()
