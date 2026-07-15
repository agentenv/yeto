from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PARALLEL_SPEC = importlib.util.spec_from_file_location(
    "run_parallel_phase_map", ROOT / "scripts" / "run_parallel_phase_map.py"
)
pexec = importlib.util.module_from_spec(PARALLEL_SPEC)
sys.modules["run_parallel_phase_map"] = pexec
assert PARALLEL_SPEC.loader is not None
PARALLEL_SPEC.loader.exec_module(pexec)

PHASE_SPEC = importlib.util.spec_from_file_location(
    "parallel_test_run_phase_map", ROOT / "scripts" / "run_phase_map.py"
)
phase = importlib.util.module_from_spec(PHASE_SPEC)
sys.modules["parallel_test_run_phase_map"] = phase
assert PHASE_SPEC.loader is not None
PHASE_SPEC.loader.exec_module(phase)

EVAL_HASHES = {
    "development_eval_rows_hash": "a" * 64,
    "development_eval_packed_hash": "b" * 64,
    "development_eval_example_ids_hash": "c" * 64,
    "development_eval_token_ids_hash": "d" * 64,
    "development_eval_source_indices_hash": "e" * 64,
}


def _phase_args(tmp_path: Path):
    return phase.build_parser().parse_args(
        [
            "--study-id",
            "bp-phase-map-p1-r0",
            "--study-phase",
            "p1_development",
            "--run-dir",
            str(tmp_path / "science"),
            "--artifact-uri",
            "gs://bucket/prebinding",
            "--git-commit",
            "a" * 40,
            "--python-executable",
            "/home/shou/venv/bin/python",
            "--command-repo-root",
            Path("/tmp/yeto-best-paper").as_posix(),
            "--image-digest",
            "b" * 64,
            "--image-numeric-id",
            "7290368630472593484",
            "--model-path",
            "/opt/yeto-science/p1r0/prebinding/inputs/model",
            "--model-revision",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "--data",
            "/opt/yeto-science/p1r0/prebinding/inputs/train.parquet",
            "--provider-evidence",
            "/opt/yeto-science/p1r0/prebinding/provider/provider-evidence.json",
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
            "--resource-class",
            "a2-highgpu-4g",
        ]
    )


def _parent_manifest() -> dict:
    cells = []
    for index, mu in enumerate((0.0, 0.5, 0.9)):
        cells.append(
            {
                "cell_id": f"bp-phase-map-p0b-h16-mu{index}-eta0p0875-s337",
                "h": 16,
                "mu": mu,
                "eta": 0.0875,
                "seed": 337,
                "training_seed": 337337,
                "block_id": "bp-phase-map-p0b-block-h16-eta0p0875-s337",
                "paired_control_id": "bp-phase-map-p0b-h16-mu0-eta0p0875-s337",
                "command_hash": f"{index + 1:064x}",
                "normalized_workload_command_hash": f"{index + 11:064x}",
            }
        )
    return {
        "schema_version": "0.2",
        "status": "sealed_results",
        "study_id": "bp-phase-map-p0b",
        "expected_cells": cells,
        "results": [],
    }


def _p1_design(tmp_path: Path):
    scientific = phase.build_plan(_phase_args(tmp_path))
    expected_cells = []
    for cell in scientific["cells"]:
        normalized = phase.sha256_bytes(
            phase.canonical_json(phase.normalized_workload_command(cell["command"]))
        )
        expected_cells.append(
            {
                "cell_id": cell["cell_id"],
                "h": cell["H"],
                "mu": cell["mu"],
                "eta": cell["eta"],
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "block_id": cell["randomization"]["block_id"],
                "paired_control_id": cell["paired_control_id"],
                "command_hash": cell["command_hash"],
                "normalized_workload_command_hash": normalized,
                "expected_learner_count": 4,
                "expected_learner_steps": 1280,
            }
        )
    parent = _parent_manifest()
    bound = {
        "schema_version": "0.2",
        "status": "bound_launch_authority",
        "study_id": "bp-phase-map-p1-r0",
        "lineage": {
            "parent_manifest_sha256": pexec.canonical_sha256(parent),
            "authoritative_prereg_template_sha256": (
                pexec.AUTHORITATIVE_PREREG_TEMPLATE_SHA256
            ),
            "descendant_kind": "initial_bound_p1_r0",
        },
        "expected_cells": expected_cells,
        "results": [],
        "frozen": {
            "cell_command_hashes": {
                cell["cell_id"]: cell["command_hash"] for cell in expected_cells
            },
            "randomization_plan_hash": scientific["randomization_plan_hash"],
            **EVAL_HASHES,
        },
    }
    roster = pexec.build_parallel_roster(
        stage_code="p1r0",
        bound_manifest=bound,
        parent_manifest=parent,
        scientific_plan=scientific,
    )
    plan = pexec.build_parallel_plan(roster)
    return parent, bound, scientific, roster, plan


def _p3_design(tmp_path: Path):
    parent = _parent_manifest()
    cells = []
    expected = []
    for seed_index, seed in enumerate(range(401, 409)):
        group_ids = []
        for arm_index, (h, arm) in enumerate(
            ((16, "short-control"), (16, "short-treatment"), (256, "long-control"), (256, "long-treatment"))
        ):
            cell_id = f"bp-phase-map-p3t-s{seed}-{arm}"
            pair_id = (
                f"bp-phase-map-p3t-s{seed}-short-control"
                if h == 16
                else f"bp-phase-map-p3t-s{seed}-long-control"
            )
            command = [
                "/home/shou/venv/bin/python",
                "/tmp/yeto-best-paper/scripts/train_p3.py",
                "--cell-id",
                cell_id,
                "--work-dir",
                "work",
                "--report-dir",
                "report",
            ]
            command_hash = pexec.canonical_sha256(command)
            normalized_hash = pexec.canonical_sha256(
                ["p3-training", seed, h, arm_index]
            )
            cells.append(
                {
                    "cell_id": cell_id,
                    "H": h,
                    "mu": float(arm_index),
                    "eta": 0.01,
                    "seed": seed,
                    "training_seed": seed * 1000 + 7,
                    "command_hash": command_hash,
                    "normalized_workload_command_hash": normalized_hash,
                    "paired_control_id": pair_id,
                    "randomization": {"block_id": f"p3-source-seed-{seed}"},
                    "target_work": {
                        "tokens": 655360,
                        "microsteps": 5120,
                        "outer_steps": pexec.HORIZON_WORK[h]["outer_steps"],
                        "learner_count": 4,
                        "learner_steps_per_learner": 1280,
                    },
                    "command": command,
                }
            )
            expected.append(
                {
                    "cell_id": cell_id,
                    "h": h,
                    "mu": float(arm_index),
                    "eta": 0.01,
                    "seed": seed,
                    "training_seed": seed * 1000 + 7,
                    "block_id": f"p3-source-seed-{seed}",
                    "paired_control_id": pair_id,
                    "command_hash": command_hash,
                    "normalized_workload_command_hash": normalized_hash,
                }
            )
            group_ids.append(cell_id)
        assert len(group_ids) == 4
    scientific = {
        "schema": "yeto_phase_map_randomization_v1",
        "study_id": "bp-phase-map-p3-training",
        "cells": cells,
    }
    scientific["randomization_plan_hash"] = pexec.canonical_sha256(scientific)
    bound = {
        "schema_version": "0.2",
        "status": "bound_launch_authority",
        "study_id": "bp-phase-map-p3-training",
        "lineage": {
            "parent_manifest_sha256": pexec.canonical_sha256(parent),
            "authoritative_prereg_template_sha256": (
                pexec.AUTHORITATIVE_PREREG_TEMPLATE_SHA256
            ),
            "descendant_kind": "fresh_confirmation_stage",
        },
        "expected_cells": copy.deepcopy(parent["expected_cells"]) + expected,
        "results": copy.deepcopy(parent["results"]),
        "frozen": {
            "cell_command_hashes": {
                **{
                    cell["cell_id"]: cell["command_hash"]
                    for cell in parent["expected_cells"]
                },
                **{cell["cell_id"]: cell["command_hash"] for cell in expected},
            },
            "randomization_plan_hash": scientific["randomization_plan_hash"],
        },
    }
    roster = pexec.build_parallel_roster(
        stage_code="p3t",
        bound_manifest=bound,
        parent_manifest=parent,
        scientific_plan=scientific,
    )
    plan = pexec.build_parallel_plan(roster)
    return parent, bound, scientific, roster, plan


def _prebound_from_plan(plan: dict) -> dict:
    return {
        "schema": "yeto_p1r0_prebound_schedule_v1",
        "stage_code": "p1r0",
        "launch_cell_count": 36,
        "waves": copy.deepcopy(plan["waves"]),
    }


def _iso(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    )


def _inventory_entry(campaign_root: Path, path: Path) -> dict:
    return {
        "path": path.relative_to(campaign_root).as_posix(),
        "sha256": pexec.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _eval_identity(index: int) -> dict:
    digest = f"{index + 1:064x}"
    return {
        "sequence_index": index,
        "sequence_id": digest,
        "input_ids_sha256": digest,
        "labels_sha256": digest,
        "attention_mask_sha256": digest,
        "supervision_weights_sha256": digest,
        "target_token_mask_sha256": digest,
        "sequence_length": 8,
        "supervised_token_count": 4,
    }


def _write_eval_freeze(campaign_root: Path, seed: int) -> tuple[Path, dict]:
    path = campaign_root / "common" / "evaluation" / f"seed-{seed}.json"
    rows = [_eval_identity(0), _eval_identity(1)]
    _write_json(
        path,
        {
            "schema": "yeto_parallel_eval_freeze_v1",
            "seed": seed,
            "supervised_token_count": 8,
            "sequences": rows,
            **EVAL_HASHES,
        },
    )
    return path, {
        "path": path.relative_to(campaign_root).as_posix(),
        "sha256": pexec.sha256_file(path),
        **EVAL_HASHES,
    }


def _work_events(h: int) -> list[dict]:
    updates = []
    prior = {fragment: 0 for fragment in range(4)}
    for outer_step in range(1, pexec.HORIZON_WORK[h]["outer_steps"] + 1):
        fragment = (outer_step - 1) % 4
        updates.append(
            {
                "outer_step": outer_step,
                "fragment": fragment,
                "responders": [
                    {
                        "learner_id": learner,
                        "base_version": prior[fragment],
                        "microsteps": h,
                        "tokens": h * 128,
                        "version_matched_anchor": True,
                    }
                    for learner in range(4)
                ],
            }
        )
        prior[fragment] = outer_step
    return updates


def _barrier_events(updates: list[dict]) -> dict:
    learners = {}
    for learner in range(4):
        pushes = []
        broadcasts = []
        for update in updates:
            responder = update["responders"][learner]
            pushes.append(
                {
                    "outer_step": update["outer_step"],
                    "fragment": update["fragment"],
                    "base_version": responder["base_version"],
                }
            )
            broadcasts.append(
                {
                    "outer_step": update["outer_step"],
                    "fragment": update["fragment"],
                    "pushed_base_version": responder["base_version"],
                    "broadcast_version": update["outer_step"],
                }
            )
        learners[str(learner)] = {
            "initial_fragments": [0, 1, 2, 3],
            "pushes": pushes,
            "broadcasts": broadcasts,
            "inner_steps_while_blocked": [],
        }
    return {"schema": "yeto_parallel_barrier_events_v1", "learners": learners}


def _provider_record(
    identity, index: int, created_at: str, *, zone: str = "us-central1-c"
) -> dict:
    gpu_uuids = [f"GPU-{identity.slot}-{identity.generation}-{gpu}" for gpu in range(4)]
    return {
        "schema": "yeto_parallel_gcp_provider_evidence_v1",
        "project": "model-training-497007",
        "zone": zone,
        "run_id": identity.run_id,
        "campaign_tag": identity.roster_tag,
        "slot": identity.slot,
        "generation": identity.generation,
        "ownership_nonce": identity.ownership_nonce,
        "labels": identity.labels,
        "instance_name": identity.run_id,
        "instance_numeric_id": str(1000 + index),
        "boot_disk_name": f"{identity.run_id}-disk",
        "boot_disk_numeric_id": str(2000 + index),
        "source_image_numeric_id": "7290368630472593484",
        "machine_type": "a2-highgpu-4g",
        "provisioning_model": "SPOT",
        "termination_action": "DELETE",
        "automatic_restart": False,
        "maintenance_action": "TERMINATE",
        "boot_disk_auto_delete": True,
        "creation_timestamp": created_at,
        "cuda_indices": [0, 1, 2, 3],
        "a100_gpu_uuids": gpu_uuids,
        "a100_gpu_names": ["NVIDIA A100-SXM4-40GB"] * 4,
        "learner_gpu_uuid_bijection": {
            str(learner): gpu_uuids[learner] for learner in range(4)
        },
    }


def _fresh_start() -> dict:
    return {
        "same_frozen_initial_model": True,
        "same_seed_and_data_order": True,
        "same_command_and_work_budget": True,
        "resumed": False,
        "prior_optimizer_state_used": False,
        "prior_checkpoint_used": False,
        "prior_tape_used": False,
        "prior_result_used": False,
    }


def _write_completed_evidence(
    *,
    campaign_root: Path,
    attempt_dir: Path,
    cell: dict,
    provider_hash: str,
    attempt: int,
    started_at: str,
    eval_freeze: Path,
    loss: float = 2.0,
) -> dict:
    command = attempt_dir / "command.json"
    attempt_start = attempt_dir / "attempt-start.json"
    learner_steps = attempt_dir / "report" / "learner-steps.json"
    work_events = attempt_dir / "report" / "work-events.json"
    barrier_events = attempt_dir / "report" / "barrier-events.json"
    results = attempt_dir / "report" / "results.json"
    eval_losses = attempt_dir / "report" / "eval-losses.jsonl"
    _write_json(command, cell["command"])
    _write_json(
        attempt_start,
        {
            "attempt_id": f"{cell['cell_id']}-attempt-{attempt}",
            "cell_id": cell["cell_id"],
            "attempt": attempt,
            "started_at_utc": started_at,
            "command_hash": cell["command_hash"],
            "provider_evidence_sha256": provider_hash,
            "fresh_initial_state": True,
            "resumed_from_attempt": None,
            "optimizer_state_input": None,
            "checkpoint_input": None,
            "prior_attempt_artifacts_used": False,
        },
    )
    _write_json(
        learner_steps,
        {
            "schema": "yeto_parallel_learner_steps_v1",
            "learners": {
                str(learner): list(range(1, 1281)) for learner in range(4)
            },
        },
    )
    updates = _work_events(cell["H"])
    _write_json(
        work_events,
        {"schema": "yeto_parallel_work_events_v1", "updates": updates},
    )
    _write_json(barrier_events, _barrier_events(updates))
    _write_json(
        results,
        {
            "schema": "yeto_parallel_cell_result_v1",
            "arm": "m4",
            "runner_exit_code": 0,
            "syncer_exit_code": 0,
            "learner_exit_codes": [0, 0, 0, 0],
            "eval_loss": loss,
        },
    )
    loss_rows = []
    for frozen in [_eval_identity(0), _eval_identity(1)]:
        row = {key: value for key, value in frozen.items() if key != "supervised_token_count"}
        row.update(token_count=4, loss_sum=8.0, loss_per_token=2.0)
        loss_rows.append(row)
    _write_jsonl(eval_losses, loss_rows)
    return {
        "command": _inventory_entry(campaign_root, command),
        "attempt_start": _inventory_entry(campaign_root, attempt_start),
        "learner_steps": _inventory_entry(campaign_root, learner_steps),
        "work_events": _inventory_entry(campaign_root, work_events),
        "barrier_events": _inventory_entry(campaign_root, barrier_events),
        "results": _inventory_entry(campaign_root, results),
        "eval_freeze": _inventory_entry(campaign_root, eval_freeze),
        "eval_losses": _inventory_entry(campaign_root, eval_losses),
    }


def _write_p3_completed_evidence(
    *,
    campaign_root: Path,
    attempt_dir: Path,
    cell: dict,
    provider_hash: str,
    attempt: int,
    started_at: str,
) -> dict:
    command = attempt_dir / "command.json"
    attempt_start = attempt_dir / "attempt-start.json"
    learner_steps = attempt_dir / "report" / "learner-steps.json"
    work_events = attempt_dir / "report" / "work-events.json"
    barrier_events = attempt_dir / "report" / "barrier-events.json"
    results = attempt_dir / "report" / "results.json"
    training_losses = attempt_dir / "report" / "training-losses.jsonl"
    checkpoint = attempt_dir / "work" / "final-checkpoint.json"
    _write_json(command, cell["command"])
    _write_json(
        attempt_start,
        {
            "attempt_id": f"{cell['cell_id']}-attempt-{attempt}",
            "cell_id": cell["cell_id"],
            "attempt": attempt,
            "started_at_utc": started_at,
            "command_hash": cell["command_hash"],
            "provider_evidence_sha256": provider_hash,
            "fresh_initial_state": True,
            "resumed_from_attempt": None,
            "optimizer_state_input": None,
            "checkpoint_input": None,
            "prior_attempt_artifacts_used": False,
        },
    )
    _write_json(
        learner_steps,
        {
            "schema": "yeto_parallel_learner_steps_v1",
            "learners": {
                str(learner): list(range(1, 1281)) for learner in range(4)
            },
        },
    )
    updates = _work_events(cell["H"])
    _write_json(
        work_events,
        {"schema": "yeto_parallel_work_events_v1", "updates": updates},
    )
    _write_json(barrier_events, _barrier_events(updates))
    _write_json(
        results,
        {
            "schema": "yeto_parallel_p3_training_result_v1",
            "loss": None,
            "evaluation_role": "none",
            "runner_exit_code": 0,
            "syncer_exit_code": 0,
            "learner_exit_codes": [0, 0, 0, 0],
        },
    )
    _write_jsonl(
        training_losses,
        [
            {
                "learner_id": learner,
                "learner_step": step,
                "training_loss": 3.0 + learner / 10_000 + step / 1_000_000,
            }
            for learner in range(4)
            for step in range(1, 1281)
        ],
    )
    _write_json(checkpoint, {"tensors": {"weight": [0.0, 1.0], "bias": [0.25]}})
    return {
        "command": _inventory_entry(campaign_root, command),
        "attempt_start": _inventory_entry(campaign_root, attempt_start),
        "learner_steps": _inventory_entry(campaign_root, learner_steps),
        "work_events": _inventory_entry(campaign_root, work_events),
        "barrier_events": _inventory_entry(campaign_root, barrier_events),
        "results": _inventory_entry(campaign_root, results),
        "training_losses": _inventory_entry(campaign_root, training_losses),
        "final_checkpoint": _inventory_entry(campaign_root, checkpoint),
    }


def _campaign_fixture(tmp_path: Path):
    parent, bound, scientific, roster, plan = _p1_design(tmp_path)
    campaign_root = tmp_path / "campaign-artifacts"
    campaign_root.mkdir()
    state_root = tmp_path / "campaign-state"
    registry_controller = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=state_root,
        campaign_artifact_root=str(campaign_root),
    )
    identities = {
        slot: registry_controller.reserve(
            slot, ownership_nonce=f"{index + 1:032x}"
        )
        for index, slot in enumerate(pexec.LOGICAL_SLOTS)
    }
    common = {
        "roster_hash": pexec.roster_hash(roster),
        "parallel_plan_hash": pexec.parallel_plan_hash(plan),
        "bound_manifest_canonical_sha256": pexec.canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific["randomization_plan_hash"],
        "amendment_raw_sha256": pexec.AMENDMENT_RAW_SHA256,
    }
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    controllers = {}
    providers = {}
    for index, (slot, identity) in enumerate(identities.items()):
        root = campaign_root / "vms" / slot / "g1"
        provider = _provider_record(identity, index, _iso(base, -3600))
        provider_path = root / "provider" / "provider-evidence.json"
        _write_json(provider_path, provider)
        registry_controller.update_state(identity, zone=provider["zone"])
        providers[slot] = provider
        controllers[slot] = pexec.VmPartialManifestController(
            identity=identity,
            local_vm_root=root,
            common_bindings=common,
            provider_record_sha256=pexec.sha256_file(provider_path),
        )
    eval_freeze, eval_entry = _write_eval_freeze(campaign_root, 347)
    scientific_by_id = {cell["cell_id"]: cell for cell in scientific["cells"]}
    for wave_index, wave in enumerate(plan["waves"]):
        wave_base = wave_index * 300
        for assignment in wave["assigned_cells_in_dispatch_order"]:
            slot = assignment["logical_slot"]
            identity = identities[slot]
            cell = scientific_by_id[assignment["cell_id"]]
            attempt_dir = (
                campaign_root
                / "vms"
                / slot
                / "g1"
                / "cells"
                / cell["cell_id"]
                / "attempt-1"
            )
            provider_path = (
                campaign_root
                / "vms"
                / slot
                / "g1"
                / "provider"
                / "provider-evidence.json"
            )
            started = _iso(base, wave_base + 10 + assignment["launch_order_index"])
            inventory = _write_completed_evidence(
                campaign_root=campaign_root,
                attempt_dir=attempt_dir,
                cell=cell,
                provider_hash=pexec.sha256_file(provider_path),
                attempt=1,
                started_at=started,
                eval_freeze=eval_freeze,
            )
            row = {
                "status": "COMPLETED",
                "failure_reason": None,
                "loss": 2.0,
                "attempt_id": f"{cell['cell_id']}-attempt-1",
                "cell_id": cell["cell_id"],
                "attempt": 1,
                "group_id": wave["group_id"],
                "retry_round": 1,
                "actual_wave_index": wave_index,
                "time_block_index": wave["time_block_index"],
                "retry_time_block_index": None,
                "available_slot_set": assignment["available_slot_set"],
                "dispatch_batch_index": assignment["dispatch_batch_index"],
                "batch_launch_order_index": assignment[
                    "batch_launch_order_index"
                ],
                "launch_order_index": assignment["launch_order_index"],
                "logical_slot": slot,
                "generation": 1,
                "run_id": identity.run_id,
                "ownership_nonce": identity.ownership_nonce,
                "instance_numeric_id": providers[slot]["instance_numeric_id"],
                "provider_evidence_sha256": pexec.sha256_file(provider_path),
                "attempt_prefix": identity.attempt_prefix(cell["cell_id"], 1),
                "executed_command_hash": cell["command_hash"],
                "fresh_start": _fresh_start(),
                "retry_of": None,
                "retry_reason": None,
                "retry_authorization": None,
                "vm_ready_at": _iso(base, -1800),
                "dispatched_at": _iso(
                    base, wave_base + assignment["launch_order_index"]
                ),
                "scientific_started_at": started,
                "scientific_ended_at": _iso(
                    base, wave_base + 100 + assignment["launch_order_index"]
                ),
                "wave_terminal_prefix_sealed_at": _iso(base, wave_base + 120),
                "artifact_inventory": inventory,
            }
            controllers[slot].append_attempt(row)
    deletion_time = _iso(base, len(plan["waves"]) * 300 + 300)
    for slot, identity in identities.items():
        digest = controllers[slot].hash_lock(
            hash_locked_at_utc=_iso(base, len(plan["waves"]) * 300 + 100)
        )
        provider = providers[slot]
        lifecycle = {
            "schema": "yeto_vm_lifecycle_final_v1",
            "status": "vm_lifecycle_final",
            "zone": provider["zone"],
            "run_id": identity.run_id,
            "slot": slot,
            "generation": 1,
            "ownership_nonce": identity.ownership_nonce,
            "labels": identity.labels,
            "partial_manifest_sha256": digest,
            "instance_numeric_id": provider["instance_numeric_id"],
            "boot_disk_numeric_id": provider["boot_disk_numeric_id"],
            "deletion_requested_at_utc": _iso(
                base, len(plan["waves"]) * 300 + 200
            ),
            "deletion_completed_at_utc": deletion_time,
            "provider_not_found_verification": {
                "instance": {
                    "name": identity.run_id,
                    "provider_id": provider["instance_numeric_id"],
                    "result": "NOT_FOUND",
                },
                "boot_disk": {
                    "name": provider["boot_disk_name"],
                    "provider_id": provider["boot_disk_numeric_id"],
                    "result": "NOT_FOUND",
                },
            },
            "zero_attached_accelerator_proof": {
                "generation_attached_a100s": 0
            },
        }
        _write_json(
            campaign_root / "vms" / slot / "g1" / "manifests" / "vm-lifecycle-final.json",
            lifecycle,
        )
    vm_registry = registry_controller.snapshot()
    census = {
        "schema": "yeto_parallel_final_provider_census_v1",
        "campaign_owned_vm_count": 0,
        "campaign_owned_attached_a100s": 0,
        "queried_at_utc": deletion_time,
    }
    bundle = pexec.CampaignBundle(
        stage_code="p1r0",
        parent_manifest=parent,
        bound_manifest=bound,
        scientific_plan=scientific,
        roster=roster,
        parallel_plan=plan,
        vm_registry=vm_registry,
        evaluation_registry={"347": eval_entry},
        final_provider_census=census,
        campaign_attempt=1,
        campaign_root=campaign_root,
    )
    return bundle


def _p3_campaign_fixture(tmp_path: Path):
    parent, bound, scientific, roster, plan = _p3_design(tmp_path)
    campaign_root = tmp_path / "p3-campaign-artifacts"
    campaign_root.mkdir()
    registry_controller = pexec.CampaignGenerationRegistry(
        stage_code="p3t",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "p3-campaign-state",
        campaign_artifact_root=str(campaign_root),
    )
    identities = {
        slot: registry_controller.reserve(
            slot, ownership_nonce=f"{index + 101:032x}"
        )
        for index, slot in enumerate(pexec.LOGICAL_SLOTS)
    }
    common = {
        "roster_hash": pexec.roster_hash(roster),
        "parallel_plan_hash": pexec.parallel_plan_hash(plan),
        "bound_manifest_canonical_sha256": pexec.canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific["randomization_plan_hash"],
        "amendment_raw_sha256": pexec.AMENDMENT_RAW_SHA256,
    }
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    controllers = {}
    providers = {}
    for index, (slot, identity) in enumerate(identities.items()):
        root = campaign_root / "vms" / slot / "g1"
        provider = _provider_record(identity, index + 20, _iso(base, -3600))
        provider_path = root / "provider" / "provider-evidence.json"
        _write_json(provider_path, provider)
        registry_controller.update_state(identity, zone=provider["zone"])
        providers[slot] = provider
        controllers[slot] = pexec.VmPartialManifestController(
            identity=identity,
            local_vm_root=root,
            common_bindings=common,
            provider_record_sha256=pexec.sha256_file(provider_path),
        )
    scientific_by_id = {cell["cell_id"]: cell for cell in scientific["cells"]}
    for wave_index, wave in enumerate(plan["waves"]):
        wave_base = wave_index * 300
        for assignment in wave["assigned_cells_in_dispatch_order"]:
            slot = assignment["logical_slot"]
            identity = identities[slot]
            cell = scientific_by_id[assignment["cell_id"]]
            attempt_dir = (
                campaign_root
                / "vms"
                / slot
                / "g1"
                / "cells"
                / cell["cell_id"]
                / "attempt-1"
            )
            provider_path = (
                campaign_root
                / "vms"
                / slot
                / "g1"
                / "provider"
                / "provider-evidence.json"
            )
            started = _iso(base, wave_base + 10 + assignment["launch_order_index"])
            inventory = _write_p3_completed_evidence(
                campaign_root=campaign_root,
                attempt_dir=attempt_dir,
                cell=cell,
                provider_hash=pexec.sha256_file(provider_path),
                attempt=1,
                started_at=started,
            )
            row = {
                "status": "COMPLETED",
                "failure_reason": None,
                "loss": None,
                "attempt_id": f"{cell['cell_id']}-attempt-1",
                "cell_id": cell["cell_id"],
                "attempt": 1,
                "group_id": wave["group_id"],
                "retry_round": 1,
                "actual_wave_index": wave_index,
                "time_block_index": wave["time_block_index"],
                "retry_time_block_index": None,
                "available_slot_set": assignment["available_slot_set"],
                "dispatch_batch_index": assignment["dispatch_batch_index"],
                "batch_launch_order_index": assignment[
                    "batch_launch_order_index"
                ],
                "launch_order_index": assignment["launch_order_index"],
                "logical_slot": slot,
                "generation": 1,
                "run_id": identity.run_id,
                "ownership_nonce": identity.ownership_nonce,
                "instance_numeric_id": providers[slot]["instance_numeric_id"],
                "provider_evidence_sha256": pexec.sha256_file(provider_path),
                "attempt_prefix": identity.attempt_prefix(cell["cell_id"], 1),
                "executed_command_hash": cell["command_hash"],
                "fresh_start": _fresh_start(),
                "retry_of": None,
                "retry_reason": None,
                "retry_authorization": None,
                "vm_ready_at": _iso(base, -1800),
                "dispatched_at": _iso(
                    base, wave_base + assignment["launch_order_index"]
                ),
                "scientific_started_at": started,
                "scientific_ended_at": _iso(
                    base, wave_base + 100 + assignment["launch_order_index"]
                ),
                "wave_terminal_prefix_sealed_at": _iso(base, wave_base + 120),
                "io_paths": ["work", "report", "work/final-checkpoint.json"],
                "artifact_inventory": inventory,
            }
            controllers[slot].append_attempt(row)
    deletion_time = _iso(base, len(plan["waves"]) * 300 + 300)
    for slot, identity in identities.items():
        digest = controllers[slot].hash_lock(
            hash_locked_at_utc=_iso(base, len(plan["waves"]) * 300 + 100)
        )
        provider = providers[slot]
        _write_json(
            campaign_root / "vms" / slot / "g1" / "manifests" / "vm-lifecycle-final.json",
            {
                "schema": "yeto_vm_lifecycle_final_v1",
                "status": "vm_lifecycle_final",
                "zone": provider["zone"],
                "run_id": identity.run_id,
                "slot": slot,
                "generation": 1,
                "ownership_nonce": identity.ownership_nonce,
                "labels": identity.labels,
                "partial_manifest_sha256": digest,
                "instance_numeric_id": provider["instance_numeric_id"],
                "boot_disk_numeric_id": provider["boot_disk_numeric_id"],
                "deletion_requested_at_utc": _iso(
                    base, len(plan["waves"]) * 300 + 200
                ),
                "deletion_completed_at_utc": deletion_time,
                "provider_not_found_verification": {
                    "instance": {
                        "provider_id": provider["instance_numeric_id"],
                        "result": "NOT_FOUND",
                    },
                    "boot_disk": {
                        "provider_id": provider["boot_disk_numeric_id"],
                        "result": "NOT_FOUND",
                    },
                },
                "zero_attached_accelerator_proof": {
                    "generation_attached_a100s": 0
                },
            },
        )
    return pexec.CampaignBundle(
        stage_code="p3t",
        parent_manifest=parent,
        bound_manifest=bound,
        scientific_plan=scientific,
        roster=roster,
        parallel_plan=plan,
        vm_registry=registry_controller.snapshot(),
        evaluation_registry={},
        final_provider_census={
            "schema": "yeto_parallel_final_provider_census_v1",
            "campaign_owned_vm_count": 0,
            "campaign_owned_attached_a100s": 0,
            "queried_at_utc": deletion_time,
        },
        campaign_attempt=1,
        campaign_root=campaign_root,
    )


def _rehash_partial_for_artifact(
    bundle, *, slot: str, attempt_id: str, role: str, path: Path
) -> None:
    root = bundle.campaign_root / "vms" / slot / "g1" / "manifests"
    partial_path = root / "vm-partial-manifest.json"
    partial = json.loads(partial_path.read_text())
    row = next(row for row in partial["attempts"] if row["attempt_id"] == attempt_id)
    row["artifact_inventory"][role] = _inventory_entry(bundle.campaign_root, path)
    _write_json(partial_path, partial)
    digest = pexec.sha256_file(partial_path)
    (root / "vm-partial-manifest.sha256").write_text(
        f"{digest}  vm-partial-manifest.json\n"
    )
    lifecycle_path = root / "vm-lifecycle-final.json"
    lifecycle = json.loads(lifecycle_path.read_text())
    lifecycle["partial_manifest_sha256"] = digest
    _write_json(lifecycle_path, lifecycle)


def test_master_seed_rank_matches_golden_and_openssl():
    pexec.verify_master_seed()
    domain = "wave"
    study_id = "bp-phase-map-p1-r0"
    token = "bp-phase-map-p1-r0-block-h256-eta0p021875-s347"
    expected = "00f05c96a7cb2108c865f080ad06b683e926b980362e44d6301a8d4443899722"
    assert pexec.rank_hex(domain, study_id, token) == expected
    payload = (
        bytes.fromhex(pexec.MASTER_SEED_HEX)
        + b"\0"
        + domain.encode()
        + b"\0"
        + study_id.encode()
        + b"\0"
        + token.encode()
    )
    openssl = subprocess.run(
        ["openssl", "dgst", "-sha256", "-binary"],
        input=payload,
        capture_output=True,
        check=True,
    ).stdout.hex()
    assert openssl == expected


def test_plan_is_identical_under_input_row_and_dictionary_permutation(tmp_path):
    parent, bound, scientific, roster, plan = _p1_design(tmp_path)
    permuted_bound = copy.deepcopy(bound)
    permuted_bound["expected_cells"].reverse()
    permuted_bound["frozen"]["cell_command_hashes"] = dict(
        reversed(list(permuted_bound["frozen"]["cell_command_hashes"].items()))
    )
    permuted_scientific = copy.deepcopy(scientific)
    permuted_scientific["cells"].reverse()
    rebuilt_roster = pexec.build_parallel_roster(
        stage_code="p1r0",
        bound_manifest=permuted_bound,
        parent_manifest=parent,
        scientific_plan=permuted_scientific,
    )
    rebuilt_plan = pexec.build_parallel_plan(rebuilt_roster)
    assert pexec.canonical_json(rebuilt_roster) == pexec.canonical_json(roster)
    assert pexec.canonical_json(rebuilt_plan) == pexec.canonical_json(plan)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_roster_rejects_missing_duplicate_or_extra_cell(tmp_path, mutation):
    parent, bound, scientific, _roster, _plan = _p1_design(tmp_path)
    scientific = copy.deepcopy(scientific)
    if mutation == "missing":
        scientific["cells"].pop()
    elif mutation == "duplicate":
        scientific["cells"][-1] = copy.deepcopy(scientific["cells"][0])
    else:
        extra = copy.deepcopy(scientific["cells"][0])
        extra["cell_id"] += "-extra"
        scientific["cells"].append(extra)
    with pytest.raises(pexec.ScheduleError):
        pexec.build_parallel_roster(
            stage_code="p1r0",
            bound_manifest=bound,
            parent_manifest=parent,
            scientific_plan=scientific,
        )


def test_p1_rejects_malformed_three_arm_block(tmp_path):
    _parent, _bound, _scientific, roster, _plan = _p1_design(tmp_path)
    roster = copy.deepcopy(roster)
    roster["launch_cells"][0]["mu"] = 0.5
    with pytest.raises(pexec.ScheduleError, match="mu"):
        pexec.build_parallel_plan(roster)


def test_p3_rejects_malformed_four_cell_seed_group(tmp_path):
    _parent, _bound, _scientific, roster, _plan = _p3_design(tmp_path)
    roster = copy.deepcopy(roster)
    roster["launch_cells"].pop()
    with pytest.raises(pexec.ScheduleError, match="P3 training|four"):
        pexec.build_parallel_plan(roster)


def test_p1_has_twelve_three_cell_waves_and_never_fills_idle_slot(tmp_path):
    _parent, _bound, _scientific, _roster, plan = _p1_design(tmp_path)
    assert len(plan["waves"]) == 12
    assert all(len(wave["assigned_cells_in_dispatch_order"]) == 3 for wave in plan["waves"])
    assert all(len(wave["idle_slots"]) == 1 for wave in plan["waves"])
    assert all(
        len({row["logical_slot"] for row in wave["assigned_cells_in_dispatch_order"]})
        == 3
        for wave in plan["waves"]
    )


def test_revision2_plan_supersedes_old_hash_and_materializes_all_slot_sets(tmp_path):
    _parent, _bound, _scientific, roster, plan = _p1_design(tmp_path)
    legacy = pexec.build_legacy_parallel_plan(roster)
    assert plan["schema"] == "yeto_parallel_plan_v2"
    assert plan["supersedes_parallel_plan_hash"] == pexec.canonical_sha256(legacy)
    assert plan["superseded_full_width_waves"] == legacy["waves"]
    assert len(plan["available_slot_variants"]) == 15
    assert {
        tuple(variant["available_slot_set"])
        for variant in plan["available_slot_variants"]
    } == set(pexec.available_slot_subsets())
    assert plan["binding_function"]["pure_inputs"] == [
        "roster_hash",
        "available_slot_set",
        "wave_index",
        "retry_round",
    ]
    assert plan["binding_function"]["outcome_inputs_forbidden"] is True


@pytest.mark.parametrize(
    "slots,expected_batches",
    [
        (("v0",), 3),
        (("v0", "v2"), 2),
        (("v0", "v1", "v3"), 1),
        (pexec.LOGICAL_SLOTS, 1),
    ],
)
def test_reduced_width_binding_is_deterministic_and_one_cell_per_slot_per_batch(
    tmp_path, slots, expected_batches
):
    _parent, _bound, _scientific, roster, plan = _p1_design(tmp_path)
    group_id = plan["waves"][0]["group_id"]
    first = pexec.wave_for_retry(
        plan, roster, group_id, 1, available_slots=slots
    )
    second = pexec.wave_for_retry(
        plan, roster, group_id, 1, available_slots=reversed(slots)
    )
    assert pexec.canonical_json(first) == pexec.canonical_json(second)
    assert first["available_slot_set"] == sorted(slots)
    assert first["dispatch_batch_count"] == expected_batches
    assert len(first["assigned_cells_in_dispatch_order"]) == 3
    for batch_index in range(expected_batches):
        batch = [
            row
            for row in first["assigned_cells_in_dispatch_order"]
            if row["dispatch_batch_index"] == batch_index
        ]
        assert len({row["logical_slot"] for row in batch}) == len(batch)
        assert all(row["logical_slot"] in slots for row in batch)


def test_p3_has_eight_width_four_seed_waves(tmp_path):
    _parent, _bound, _scientific, _roster, plan = _p3_design(tmp_path)
    assert len(plan["waves"]) == 8
    assert all(len(wave["assigned_cells_in_dispatch_order"]) == 4 for wave in plan["waves"])
    assert all(wave["idle_slots"] == [] for wave in plan["waves"])


@pytest.mark.parametrize("mutation", ["slot", "dispatch", "split"])
def test_prebound_validator_rejects_manual_schedule_changes(tmp_path, mutation):
    _parent, _bound, _scientific, _roster, plan = _p1_design(tmp_path)
    prebound = _prebound_from_plan(plan)
    if mutation == "slot":
        prebound["waves"][0]["assigned_cells_in_dispatch_order"][0][
            "logical_slot"
        ] = "v2"
    elif mutation == "dispatch":
        prebound["waves"][0]["assigned_cells_in_dispatch_order"].reverse()
    else:
        prebound["waves"][0]["assigned_cells_in_dispatch_order"].pop()
    with pytest.raises(pexec.ScheduleError):
        pexec.validate_prebound_p1r0_schedule(prebound, plan)


def test_name_and_assignment_reproduce_from_same_parent_and_seed(tmp_path):
    parent, bound, scientific, roster, plan = _p1_design(tmp_path)
    second_roster = pexec.build_parallel_roster(
        stage_code="p1r0",
        bound_manifest=copy.deepcopy(bound),
        parent_manifest=copy.deepcopy(parent),
        scientific_plan=copy.deepcopy(scientific),
    )
    second_plan = pexec.build_parallel_plan(second_roster)
    digest = pexec.roster_hash(roster)
    assert pexec.roster_hash(second_roster) == digest
    assert pexec.parallel_plan_hash(second_plan) == pexec.parallel_plan_hash(plan)
    assert [
        pexec.physical_run_id("p1r0", digest, 1, slot, 1)
        for slot in pexec.LOGICAL_SLOTS
    ] == [
        pexec.physical_run_id("p1r0", pexec.roster_hash(second_roster), 1, slot, 1)
        for slot in pexec.LOGICAL_SLOTS
    ]


def test_physical_generation_name_grammar_and_monotone_generation(tmp_path):
    _parent, bound, _scientific, roster, _plan = _p1_design(tmp_path)
    digest = pexec.roster_hash(roster)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=digest,
        campaign_attempt=3,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    first = registry.reserve("v2", ownership_nonce="1" * 32)
    second = registry.reserve("v2", ownership_nonce="2" * 32)
    assert first.run_id == f"bp-p1r0-{digest[:16]}-c3-v2-g1"
    assert second.run_id == f"bp-p1r0-{digest[:16]}-c3-v2-g2"
    assert pexec.RUN_ID_RE.fullmatch(second.run_id)


def test_generation_registry_rejects_nonce_reuse(tmp_path):
    _parent, bound, _scientific, roster, _plan = _p1_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    registry.reserve("v0", ownership_nonce="a" * 32)
    with pytest.raises(pexec.LifecycleError, match="reuse"):
        registry.reserve("v1", ownership_nonce="a" * 32)


@pytest.mark.parametrize("zone", pexec.ALLOWED_US_CENTRAL1_ZONES)
def test_provider_record_accepts_each_authorized_us_central1_zone(tmp_path, zone):
    _parent, bound, _scientific, roster, _plan = _p1_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    identity = registry.reserve("v0", ownership_nonce="c" * 32)
    provider = _provider_record(
        identity, 1, "2026-07-15T00:00:00Z", zone=zone
    )
    registry_row = identity.registry_row()
    registry_row["zone"] = zone
    assert pexec.validate_provider_record(provider, registry_row)[
        "instance_numeric_id"
    ] == "1001"


def test_provider_record_rejects_zone_outside_authorized_region(tmp_path):
    _parent, bound, _scientific, roster, _plan = _p1_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    identity = registry.reserve("v0", ownership_nonce="d" * 32)
    provider = _provider_record(
        identity, 1, "2026-07-15T00:00:00Z", zone="us-east4-a"
    )
    with pytest.raises(pexec.LifecycleError, match="project/zone"):
        pexec.validate_provider_record(provider, identity.registry_row())


def test_partial_manifest_is_not_a_campaign_seal_and_is_hash_locked(tmp_path):
    _parent, bound, scientific, roster, plan = _p1_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    identity = registry.reserve("v0", ownership_nonce="b" * 32)
    root = tmp_path / "artifacts" / "vms" / "v0" / "g1"
    provider = root / "provider" / "provider-evidence.json"
    provider.parent.mkdir(parents=True)
    provider.write_text("{}\n")
    controller = pexec.VmPartialManifestController(
        identity=identity,
        local_vm_root=root,
        common_bindings={
            "roster_hash": pexec.roster_hash(roster),
            "parallel_plan_hash": pexec.parallel_plan_hash(plan),
            "bound_manifest_canonical_sha256": pexec.canonical_sha256(bound),
            "scientific_randomization_plan_hash": scientific[
                "randomization_plan_hash"
            ],
            "amendment_raw_sha256": pexec.AMENDMENT_RAW_SHA256,
        },
        provider_record_sha256=pexec.sha256_file(provider),
    )
    digest = controller.hash_lock(hash_locked_at_utc="2026-07-15T00:00:00Z")
    partial = json.loads(controller.manifest_path.read_text())
    assert partial["status"] == "vm_partial_hash_locked"
    assert partial["status"] != "sealed_results"
    assert controller.hash_path.read_text() == f"{digest}  vm-partial-manifest.json\n"
    with pytest.raises(pexec.LifecycleError, match="hash-locked"):
        controller.append_attempt({})


def test_exact_teardown_command_requires_numeric_nonprotected_id(tmp_path):
    argv = pexec.optimizer_harness_delete_argv(
        python_executable="/Users/shou/yeto/.venv/bin/python3",
        state_dir=tmp_path / "state",
        spec_path=tmp_path / "spec.json",
        exact_instance_id="123456",
    )
    assert argv[-3:] == ["--instance-id", "123456", "--yes"]
    with pytest.raises(pexec.LifecycleError):
        pexec.optimizer_harness_delete_argv(
            python_executable="/Users/shou/yeto/.venv/bin/python3",
            state_dir=tmp_path,
            spec_path=tmp_path / "spec.json",
            exact_instance_id=pexec.PROTECTED_INSTANCE_ID,
        )


def test_parallel_harness_launch_binds_32_hex_nonce(tmp_path):
    argv = pexec.optimizer_harness_launch_argv(
        python_executable="/Users/shou/yeto/.venv/bin/python3",
        state_dir=tmp_path / "state",
        spec_path=tmp_path / "spec.json",
        ownership_nonce="c" * 32,
    )
    assert argv[-3:] == ["--ownership-nonce", "c" * 32, "--yes"]


def test_campaign_aggregator_seals_all_36_cells_and_four_vm_partials(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    manifest, seal = pexec.CampaignAggregator(bundle).build_manifest_and_seal(
        sealed_at_utc="2026-07-15T12:00:00Z"
    )
    assert seal["status"] == "sealed_results"
    assert seal["launch_cell_count"] == 36
    assert seal["resolved_launch_cell_count"] == 36
    assert seal["attempt_count"] == 36
    assert len(seal["vm_partial_manifest_hashes"]) == 4
    assert len(seal["vm_lifecycle_record_hashes"]) == 4
    assert len(manifest["analysis_rounds"]) == 36
    assert all(row["status"] == "COMPLETED" for row in manifest["attempts"])


def test_vm_missing_teardown_proof_hard_fails_aggregation(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    lifecycle = (
        bundle.campaign_root
        / "vms"
        / "v2"
        / "g1"
        / "manifests"
        / "vm-lifecycle-final.json"
    )
    lifecycle.unlink()
    with pytest.raises(pexec.LifecycleError, match="teardown evidence"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_vm_partial_manifest_tamper_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    partial = (
        bundle.campaign_root
        / "vms"
        / "v1"
        / "g1"
        / "manifests"
        / "vm-partial-manifest.json"
    )
    partial.write_text(partial.read_text() + " ")
    with pytest.raises(pexec.LifecycleError, match="hash lock"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_missing_exact_numeric_id_in_not_found_proof_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    lifecycle_path = (
        bundle.campaign_root
        / "vms"
        / "v0"
        / "g1"
        / "manifests"
        / "vm-lifecycle-final.json"
    )
    lifecycle = json.loads(lifecycle_path.read_text())
    lifecycle["provider_not_found_verification"]["instance"].pop("provider_id")
    _write_json(lifecycle_path, lifecycle)
    with pytest.raises(pexec.LifecycleError, match="name-only|NOT_FOUND"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_partial_arrival_order_produces_identical_manifest_and_seal_preimage(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    first_manifest, first_seal = pexec.CampaignAggregator(
        bundle
    ).build_manifest_and_seal(sealed_at_utc="2026-07-15T12:00:00Z")
    reversed_registry = copy.deepcopy(bundle.vm_registry)
    reversed_registry["generations"].reverse()
    reordered = pexec.CampaignBundle(
        **{
            **bundle.__dict__,
            "vm_registry": reversed_registry,
        }
    )
    second_manifest, second_seal = pexec.CampaignAggregator(
        reordered
    ).build_manifest_and_seal(sealed_at_utc="2026-07-15T12:00:00Z")
    assert pexec.canonical_json(first_manifest) == pexec.canonical_json(second_manifest)
    assert pexec.canonical_json(first_seal) == pexec.canonical_json(second_seal)


def test_campaign_seal_is_create_only_and_unique(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    aggregator = pexec.CampaignAggregator(bundle)
    first = aggregator.seal(sealed_at_utc="2026-07-15T12:00:00Z")
    assert first["status"] == "sealed_results"
    assert (bundle.campaign_root / "campaign" / "campaign-seal.json").is_file()
    with pytest.raises(pexec.SealError, match="create-only"):
        aggregator.seal(sealed_at_utc="2026-07-15T12:00:00Z")


@pytest.mark.parametrize(
    ("role", "mode"),
    [
        ("command", "delete"),
        ("attempt_start", "corrupt"),
        ("learner_steps", "delete"),
        ("work_events", "corrupt"),
        ("barrier_events", "delete"),
        ("results", "corrupt"),
        ("eval_freeze", "delete"),
        ("eval_losses", "corrupt"),
    ],
)
def test_each_completed_work_artifact_is_required(tmp_path, role, mode):
    bundle = _campaign_fixture(tmp_path)
    partial_path = (
        bundle.campaign_root
        / "vms"
        / "v0"
        / "g1"
        / "manifests"
        / "vm-partial-manifest.json"
    )
    partial = json.loads(partial_path.read_text())
    row = partial["attempts"][0]
    path = bundle.campaign_root / row["artifact_inventory"][role]["path"]
    if mode == "delete":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(pexec.EvidenceError):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def _first_attempt(bundle):
    for slot in pexec.LOGICAL_SLOTS:
        partial_path = (
            bundle.campaign_root
            / "vms"
            / slot
            / "g1"
            / "manifests"
            / "vm-partial-manifest.json"
        )
        partial = json.loads(partial_path.read_text())
        if partial["attempts"]:
            return slot, partial_path, partial, partial["attempts"][0]
    raise AssertionError("fixture has no attempts")


def _rewrite_partial(bundle, slot: str, partial_path: Path, partial: dict) -> None:
    _write_json(partial_path, partial)
    digest = pexec.sha256_file(partial_path)
    hash_path = partial_path.with_name("vm-partial-manifest.sha256")
    hash_path.write_text(f"{digest}  vm-partial-manifest.json\n")
    lifecycle_path = partial_path.with_name("vm-lifecycle-final.json")
    lifecycle = json.loads(lifecycle_path.read_text())
    lifecycle["partial_manifest_sha256"] = digest
    _write_json(lifecycle_path, lifecycle)


@pytest.mark.parametrize("case", ["zero", "one_short"])
def test_zero_or_short_learner_work_cannot_seal(tmp_path, case):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    path = bundle.campaign_root / row["artifact_inventory"]["learner_steps"]["path"]
    evidence = json.loads(path.read_text())
    evidence["learners"]["0"] = [] if case == "zero" else list(range(1, 1280))
    _write_json(path, evidence)
    row["artifact_inventory"]["learner_steps"] = _inventory_entry(
        bundle.campaign_root, path
    )
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.EvidenceError, match="optimizer steps"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_missing_fragment_update_cannot_seal(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    path = bundle.campaign_root / row["artifact_inventory"]["work_events"]["path"]
    evidence = json.loads(path.read_text())
    evidence["updates"].pop()
    _write_json(path, evidence)
    row["artifact_inventory"]["work_events"] = _inventory_entry(bundle.campaign_root, path)
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.EvidenceError, match="event count|fragment"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_completed_endpoint_cannot_seal(tmp_path, constant):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    path = bundle.campaign_root / row["artifact_inventory"]["results"]["path"]
    raw = path.read_text().replace('"eval_loss": 2.0', f'"eval_loss": {constant}')
    path.write_text(raw)
    row["artifact_inventory"]["results"] = _inventory_entry(bundle.campaign_root, path)
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.ParallelPhaseMapError):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_endpoint_per_sequence_mismatch_cannot_seal(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    path = bundle.campaign_root / row["artifact_inventory"]["eval_losses"]["path"]
    losses = [json.loads(line) for line in path.read_text().splitlines()]
    losses[0]["loss_sum"] = 12.0
    losses[0]["loss_per_token"] = 3.0
    _write_jsonl(path, losses)
    row["artifact_inventory"]["eval_losses"] = _inventory_entry(bundle.campaign_root, path)
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.EvidenceError, match="reproduce"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_unresolved_infrastructure_failure_cannot_seal(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    attempt_dir = (
        bundle.campaign_root
        / row["artifact_inventory"]["command"]["path"]
    ).parent
    infra = attempt_dir / "infra-failure.json"
    _write_json(
        infra,
        {
            "attempt_id": row["attempt_id"],
            "failure_reason": "vm_host_gpu_failure",
        },
    )
    row["status"] = "INFRA_FAILURE"
    row["failure_reason"] = "vm_host_gpu_failure"
    row["loss"] = None
    row["artifact_inventory"]["infra_failure"] = _inventory_entry(
        bundle.campaign_root, infra
    )
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(
        pexec.ScheduleError, match="missing actual wave|manual slot swap|mixed group"
    ):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_zero_cell_campaign_cannot_seal(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    for slot in pexec.LOGICAL_SLOTS:
        partial_path = (
            bundle.campaign_root
            / "vms"
            / slot
            / "g1"
            / "manifests"
            / "vm-partial-manifest.json"
        )
        partial = json.loads(partial_path.read_text())
        partial["attempts"] = []
        _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.ScheduleError, match="missing actual wave"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_correctly_evidenced_divergence_is_retained_and_sealable(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    attempt_dir = (
        bundle.campaign_root / row["artifact_inventory"]["command"]["path"]
    ).parent
    tape = attempt_dir / "report" / "tape-prefix.jsonl"
    divergence = attempt_dir / "report" / "scientific-divergence.json"
    _write_jsonl(tape, [{"outer_step": 1, "finite": True}])
    _write_json(
        divergence,
        {
            "schema": "yeto_parallel_scientific_divergence_v1",
            "cell_id": row["cell_id"],
            "attempt_id": row["attempt_id"],
            "last_finite_step": 1,
            "first_nonfinite_event": {"outer_step": 2, "kind": "nonfinite_gradient"},
        },
    )
    row["status"] = "DIVERGED"
    row["loss"] = None
    row["failure_reason"] = None
    row["artifact_inventory"]["tape_prefix"] = _inventory_entry(
        bundle.campaign_root, tape
    )
    row["artifact_inventory"]["scientific_divergence"] = _inventory_entry(
        bundle.campaign_root, divergence
    )
    _rewrite_partial(bundle, slot, partial_path, partial)
    manifest, seal = pexec.CampaignAggregator(bundle).build_manifest_and_seal(
        sealed_at_utc="2026-07-15T12:00:00Z"
    )
    assert seal["status"] == "sealed_results"
    analysis = manifest["analysis_rounds"][row["cell_id"]]
    assert analysis["status"] == "DIVERGED"
    assert analysis["attempt"] == 1


def test_start_span_excess_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    row["scientific_started_at"] = "2026-07-15T00:03:00Z"
    row["scientific_ended_at"] = "2026-07-15T00:03:01Z"
    row["wave_terminal_prefix_sealed_at"] = "2026-07-15T00:03:02Z"
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.ScheduleError, match="start span"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_cross_wave_overlap_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    # Extend one wave-0 attempt beyond the wave-1 first dispatch while keeping
    # its own terminal-prefix seal after the process end.
    slot, partial_path, partial, row = _first_attempt(bundle)
    target_wave = row["actual_wave_index"]
    base = datetime(2026, 7, 15, tzinfo=timezone.utc)
    overlap_end = _iso(base, (target_wave + 1) * 300 + 60)
    overlap_seal = _iso(base, (target_wave + 1) * 300 + 61)
    row["scientific_ended_at"] = overlap_end
    row["wave_terminal_prefix_sealed_at"] = overlap_seal
    # Every row in the same wave must cite the same terminal-prefix seal.
    _rewrite_partial(bundle, slot, partial_path, partial)
    for other_slot in pexec.LOGICAL_SLOTS:
        if other_slot == slot:
            continue
        other_path = (
            bundle.campaign_root
            / "vms"
            / other_slot
            / "g1"
            / "manifests"
            / "vm-partial-manifest.json"
        )
        other = json.loads(other_path.read_text())
        changed = False
        for other_row in other["attempts"]:
                if other_row["actual_wave_index"] == target_wave:
                    other_row["wave_terminal_prefix_sealed_at"] = overlap_seal
                changed = True
        if changed:
            _rewrite_partial(bundle, other_slot, other_path, other)
    with pytest.raises(pexec.ScheduleError, match="overlaps"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_capacity_trace_rejects_five_overlapping_generations():
    rows = [
        {
            "creation_timestamp": "2026-07-15T00:00:00Z",
            "deletion_completed_at_utc": "2026-07-15T01:00:00Z",
        }
        for _ in range(5)
    ]
    with pytest.raises(pexec.LifecycleError, match="four VMs|16 A100s"):
        pexec._validate_generation_capacity(rows)


def test_cross_namespace_attempt_write_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    other_slot = "v1" if slot != "v1" else "v2"
    row["attempt_prefix"] = row["attempt_prefix"].replace(
        f"/vms/{slot}/", f"/vms/{other_slot}/"
    )
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.LifecycleError, match="namespace"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_provider_record_substitution_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    substitute_slot = "v1" if slot != "v1" else "v2"
    substitute = (
        bundle.campaign_root
        / "vms"
        / substitute_slot
        / "g1"
        / "provider"
        / "provider-evidence.json"
    )
    row["provider_evidence_sha256"] = pexec.sha256_file(substitute)
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.LifecycleError, match="substituted provider"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_instance_numeric_id_reuse_across_generations_is_rejected(tmp_path):
    bundle = _campaign_fixture(tmp_path)
    v0_provider_path = (
        bundle.campaign_root / "vms" / "v0" / "g1" / "provider" / "provider-evidence.json"
    )
    reused_id = json.loads(v0_provider_path.read_text())["instance_numeric_id"]
    provider_path = (
        bundle.campaign_root / "vms" / "v1" / "g1" / "provider" / "provider-evidence.json"
    )
    provider = json.loads(provider_path.read_text())
    provider["instance_numeric_id"] = reused_id
    _write_json(provider_path, provider)
    provider_hash = pexec.sha256_file(provider_path)
    partial_path = (
        bundle.campaign_root / "vms" / "v1" / "g1" / "manifests" / "vm-partial-manifest.json"
    )
    partial = json.loads(partial_path.read_text())
    partial["provider_record_sha256"] = provider_hash
    for row in partial["attempts"]:
        row["provider_evidence_sha256"] = provider_hash
        row["instance_numeric_id"] = reused_id
        attempt_start = (
            bundle.campaign_root / row["artifact_inventory"]["attempt_start"]["path"]
        )
        start = json.loads(attempt_start.read_text())
        start["provider_evidence_sha256"] = provider_hash
        _write_json(attempt_start, start)
        row["artifact_inventory"]["attempt_start"] = _inventory_entry(
            bundle.campaign_root, attempt_start
        )
    _rewrite_partial(bundle, "v1", partial_path, partial)
    lifecycle_path = partial_path.with_name("vm-lifecycle-final.json")
    lifecycle = json.loads(lifecycle_path.read_text())
    lifecycle["instance_numeric_id"] = reused_id
    lifecycle["provider_not_found_verification"]["instance"]["provider_id"] = reused_id
    _write_json(lifecycle_path, lifecycle)
    with pytest.raises(pexec.LifecycleError, match="reuse"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


def test_p3_training_seal_builds_exact_32_cell_checkpoint_registry(tmp_path):
    bundle = _p3_campaign_fixture(tmp_path)
    manifest, seal = pexec.CampaignAggregator(bundle).build_manifest_and_seal(
        sealed_at_utc="2026-07-15T12:00:00Z"
    )
    registry = manifest["p3_checkpoint_registry"]
    assert seal["launch_cell_count"] == 32
    assert seal["resolved_launch_cell_count"] == 32
    assert len(registry["cells"]) == 32
    assert len({row["cell_id"] for row in registry["cells"]}) == 32
    assert all(row["loss"] is None for row in manifest["attempts"])
    assert manifest["partial_outcomes_exposed"] is False


def test_p3_missing_checkpoint_blocks_campaign_seal(tmp_path):
    bundle = _p3_campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    checkpoint = bundle.campaign_root / row["artifact_inventory"]["final_checkpoint"]["path"]
    checkpoint.unlink()
    with pytest.raises(pexec.EvidenceError, match="final_checkpoint"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


@pytest.mark.parametrize(
    "forbidden_path",
    [
        "/opt/yeto-science/p3t/tag/development/eval.jsonl",
        "gs://bucket/audit/checkpoints",
        "backup/audit-results.json",
    ],
)
def test_p3_command_backup_and_sync_paths_cannot_expose_evaluation(
    tmp_path, forbidden_path
):
    bundle = _p3_campaign_fixture(tmp_path)
    slot, partial_path, partial, row = _first_attempt(bundle)
    row["io_paths"].append(forbidden_path)
    _rewrite_partial(bundle, slot, partial_path, partial)
    with pytest.raises(pexec.EvidenceError, match="evaluation data"):
        pexec.CampaignAggregator(bundle).build_manifest_and_seal()


class _ScriptedBackend:
    def __init__(self, *, target_slot: str | None = None, target_launch: int | None = None):
        self.target_slot = target_slot
        self.target_launch = target_launch
        self.preempted = False
        self.active = {}
        self.providers = {}
        self.requests = []
        self._provider_index = 0
        self.base = datetime(2026, 7, 15, tzinfo=timezone.utc)

    def provision(self, identity):
        self._provider_index += 1
        provider = _provider_record(
            identity,
            self._provider_index,
            _iso(self.base, -3600 + self._provider_index),
        )
        self.active[identity.run_id] = identity
        self.providers[identity.run_id] = provider
        return provider

    def ready(self, identity, provider_record):
        del identity, provider_record
        return _iso(self.base, -1800)

    def dispatch(self, identity, request):
        self.requests.append((identity, request))
        return _iso(
            self.base,
            request.actual_wave_index * 1000
            + request.dispatch_batch_index * 300
            + request.batch_launch_order_index,
        )

    def collect(self, identity, request):
        should_preempt = (
            not self.preempted
            and request.retry_round == 1
            and (
                (self.target_slot is not None and identity.slot == self.target_slot)
                or (
                    self.target_launch is not None
                    and request.launch_order_index == self.target_launch
                )
            )
        )
        common = {
            "loss": None if should_preempt else 999.0,
            "artifact_inventory": {},
            "scientific_started_at": _iso(
                self.base,
                request.actual_wave_index * 1000
                + request.dispatch_batch_index * 300
                + 10
                + request.batch_launch_order_index,
            ),
            "scientific_ended_at": _iso(
                self.base,
                request.actual_wave_index * 1000
                + request.dispatch_batch_index * 300
                + 100
                + request.batch_launch_order_index,
            ),
        }
        if should_preempt:
            self.preempted = True
            return {
                **common,
                "status": "INFRA_FAILURE",
                "failure_reason": "provider_spot_preemption",
            }
        return {**common, "status": "COMPLETED", "failure_reason": None}

    def finalize_generation(
        self, identity, provider_record, partial_manifest_sha256, *, preempted
    ):
        del preempted
        self.active.pop(identity.run_id, None)
        return {
            "schema": "yeto_vm_lifecycle_final_v1",
            "status": "vm_lifecycle_final",
            "zone": provider_record["zone"],
            "run_id": identity.run_id,
            "slot": identity.slot,
            "generation": identity.generation,
            "ownership_nonce": identity.ownership_nonce,
            "labels": identity.labels,
            "partial_manifest_sha256": partial_manifest_sha256,
            "instance_numeric_id": provider_record["instance_numeric_id"],
            "boot_disk_numeric_id": provider_record["boot_disk_numeric_id"],
            "deletion_requested_at_utc": "2026-07-16T00:00:00Z",
            "deletion_completed_at_utc": "2026-07-16T00:00:01Z",
            "provider_not_found_verification": {
                "instance": {
                    "provider_id": provider_record["instance_numeric_id"],
                    "result": "NOT_FOUND",
                },
                "boot_disk": {
                    "provider_id": provider_record["boot_disk_numeric_id"],
                    "result": "NOT_FOUND",
                },
            },
            "zero_attached_accelerator_proof": {
                "generation_attached_a100s": 0
            },
        }

    def census(self, roster_tag):
        del roster_tag
        return {
            "campaign_owned_vm_count": len(self.active),
            "campaign_owned_attached_a100s": len(self.active) * 4,
        }


def _run_scripted_preemption(tmp_path: Path, *, target_slot=None, target_launch=None):
    _parent, bound, scientific, roster, plan = _p3_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p3t",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    backend = _ScriptedBackend(
        target_slot=target_slot, target_launch=target_launch
    )
    executor = pexec.ParallelWaveExecutor(
        roster=roster,
        parallel_plan=plan,
        scientific_plan=scientific,
        bound_manifest=bound,
        registry=registry,
        campaign_root=tmp_path / "artifacts",
        backend=backend,
        clock=lambda: "2026-07-15T23:59:59Z",
    )
    vm_registry, census = executor.run()
    return roster, plan, backend, vm_registry, census


def test_executor_runs_complete_p1r0_at_width_one_in_deterministic_batches(tmp_path):
    _parent, bound, scientific, roster, plan = _p1_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p1r0",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    backend = _ScriptedBackend()
    executor = pexec.ParallelWaveExecutor(
        roster=roster,
        parallel_plan=plan,
        scientific_plan=scientific,
        bound_manifest=bound,
        registry=registry,
        campaign_root=tmp_path / "artifacts",
        backend=backend,
        available_slots=("v0",),
        clock=lambda: "2026-07-16T12:00:00Z",
    )
    vm_registry, census = executor.run()
    assert census["campaign_owned_vm_count"] == 0
    assert census["campaign_owned_attached_a100s"] == 0
    assert {row["slot"] for row in vm_registry["generations"]} == {"v0"}
    assert len(backend.requests) == 36
    assert all(identity.slot == "v0" for identity, _request in backend.requests)
    for actual_wave_index in range(12):
        requests = [
            request
            for _identity, request in backend.requests
            if request.actual_wave_index == actual_wave_index
        ]
        assert [request.dispatch_batch_index for request in requests] == [0, 1, 2]
        assert all(request.available_slot_set == ("v0",) for request in requests)


@pytest.mark.parametrize("slot", list(pexec.LOGICAL_SLOTS))
def test_preemption_in_every_slot_forces_fresh_whole_wave_retry(tmp_path, slot):
    roster, plan, backend, vm_registry, census = _run_scripted_preemption(
        tmp_path, target_slot=slot
    )
    assert backend.preempted is True
    assert census["campaign_owned_vm_count"] == 0
    assert census["campaign_owned_attached_a100s"] == 0
    assert any(
        row["slot"] == slot and row["generation"] == 2
        for row in vm_registry["generations"]
    )
    first_group = plan["waves"][0]["group_id"]
    group_requests = [
        request for _identity, request in backend.requests if request.group_id == first_group
    ]
    assert [request.retry_round for request in group_requests] == [1, 1, 1, 1, 2, 2, 2, 2]
    retry = group_requests[4:]
    expected_retry = pexec.wave_for_retry(plan, roster, first_group, 2)
    assert [request.cell_id for request in retry] == [
        row["cell_id"] for row in expected_retry["assigned_cells_in_dispatch_order"]
    ]
    assert [request.launch_order_index for request in retry] == [0, 1, 2, 3]
    assert all(request.attempt_prefix.endswith("/attempt-2/") for request in retry)
    assert all(request.fresh_start["resumed"] is False for request in retry)
    assert all(request.retry_of and request.retry_of.endswith("-attempt-1") for request in retry)
    assert all(request.retry_authorization["loss_blind"] is True for request in retry)


@pytest.mark.parametrize("launch_position", [0, 1, 2, 3])
def test_preemption_in_every_launch_position_is_contiguous_and_not_resumed(
    tmp_path, launch_position
):
    _roster, plan, backend, _registry, _census = _run_scripted_preemption(
        tmp_path, target_launch=launch_position
    )
    first_group = plan["waves"][0]["group_id"]
    requests = [
        request for _identity, request in backend.requests if request.group_id == first_group
    ]
    assert len(requests) == 8
    assert [request.actual_wave_index for request in requests] == [0] * 4 + [1] * 4
    assert not any(request.attempt_prefix.endswith("/attempt-1/") for request in requests[4:])


def test_finite_poor_loss_never_triggers_a_retry(tmp_path):
    _parent, bound, scientific, roster, plan = _p3_design(tmp_path)
    registry = pexec.CampaignGenerationRegistry(
        stage_code="p3t",
        study_id=bound["study_id"],
        roster_digest=pexec.roster_hash(roster),
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root=str(tmp_path / "artifacts"),
    )
    backend = _ScriptedBackend()
    executor = pexec.ParallelWaveExecutor(
        roster=roster,
        parallel_plan=plan,
        scientific_plan=scientific,
        bound_manifest=bound,
        registry=registry,
        campaign_root=tmp_path / "artifacts",
        backend=backend,
        clock=lambda: "2026-07-15T23:59:59Z",
    )
    executor.run()
    assert len(backend.requests) == 32
    assert all(request.retry_round == 1 for _identity, request in backend.requests)
