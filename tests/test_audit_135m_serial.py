from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_135m_contract as audit
from scripts import audit_135m_phase_manifest as promotion
from scripts import audit_135m_serial as serial
from scripts import audit_135m_serial_controller as controller
from scripts import build_audit_135m_serial_packet as packet_builder
from scripts import run_parallel_phase_map as parallel


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runtime(tmp_path: Path) -> dict[str, object]:
    return {
        "run_dir": tmp_path / "runs",
        "model_path": tmp_path / "model",
        "python_executable": "/home/shou/venv/bin/python",
        "command_repo_root": Path("/tmp/yeto-best-paper"),
    }


def _parent() -> dict[str, object]:
    cell = {
        "cell_id": "sealed-parent-cell",
        "h": 64,
        "m": 4,
        "mu": 0.0,
        "eta": 0.021875,
        "seed": 347,
        "training_seed": 347347,
        "block_id": "sealed-parent-block",
        "analysis_role": "sealed_parent",
        "pairing_identity_hash": "3" * 64,
        "pairing_command_hash": "4" * 64,
        "command_hash": "1" * 64,
        "normalized_workload_command_hash": "2" * 64,
    }
    return {
        "status": "sealed_results",
        "study_id": "sealed-parent",
        "expected_cells": [cell],
        "results": [{"cell_id": cell["cell_id"], "status": "COMPLETED"}],
        "seed_pairs": {"347": 347347},
        "frozen": {
            "model_id": audit.MODEL_ID,
            "model_revision": audit.MODEL_REVISION,
            "model_hash": audit.MODEL_HASH,
            "data_hash": audit.DATA_HASH,
            "image_id": audit.IMAGE_NUMERIC_ID,
            "image_digest": audit.IMAGE_DIGEST,
            "cell_command_hashes": {cell["cell_id"]: cell["command_hash"]},
            "train_rows_hashes": {"347": "5" * 64},
            "train_source_indices_hashes": {"347": "6" * 64},
            "development_eval_rows_hash": "7" * 64,
            "development_eval_packed_hash": "8" * 64,
            "development_eval_example_ids_hash": "9" * 64,
            "development_eval_token_ids_hash": "a" * 64,
            "development_eval_source_indices_hash": "b" * 64,
        },
        "protocol": {
            "train_rows": audit.TRAIN_ROWS,
            "development_eval_rows": audit.DEVELOPMENT_EVAL_ROWS,
            "audit_eval_rows": audit.AUDIT_EVAL_ROWS,
            "seq_len": audit.SEQ_LEN,
            "micro_batch_size": audit.MICRO_BATCH_SIZE,
            "inner_lr": audit.INNER_LR,
            "token_budget": audit.TOKEN_BUDGET,
            "eval_split_seed": audit.EVAL_SPLIT_SEED,
            "spot_only": True,
            "barrier": True,
            "strict_quorum": True,
            "version_matched": True,
        },
    }


def _seed_registry(plan: dict[str, object]) -> dict[str, object]:
    seeds = sorted({int(cell["seed"]) for cell in plan["cells"]})
    return {
        "schema": "audit_135m_seed_bundle_registry_v1",
        "seeds": {
            str(seed): {
                "train_rows_sha256": f"{seed % 10:x}" * 64,
                "train_source_indices_sha256": f"{(seed + 1) % 10:x}" * 64,
                "parallel_eval_freeze_sha256": f"{(seed + 2) % 10:x}" * 64,
            }
            for seed in seeds
        },
    }


def _design(tmp_path: Path, stage_code: str = "a1d"):
    scientific, _hashes = audit.build_plan(
        stage_code=stage_code,
        study_id=f"audit-{stage_code}-serial-test",
        runtime_config=_runtime(tmp_path),
        order_seed=20260717,
    )
    parent = _parent()
    bound = audit.build_bound_manifest(
        stage_code=stage_code,
        study_id=f"audit-{stage_code}-serial-test",
        git_commit="a" * 40,
        parent=parent,
        expected_parent_hash=audit.canonical_sha256(parent),
        plan=scientific,
        seed_registry=_seed_registry(scientific),
        decision_hashes=scientific["decision_manifest_hashes"],
    )
    roster = parallel.build_parallel_roster(
        stage_code=stage_code,
        bound_manifest=bound,
        parent_manifest=parent,
        scientific_plan=scientific,
    )
    plan = parallel.build_parallel_plan(roster)
    binding = serial.build_serial_binding(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        compatibility_roster=roster,
        compatibility_plan=plan,
    )
    return parent, bound, scientific, roster, plan, binding


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _fresh_start() -> dict[str, bool]:
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


def _schedule(scientific: dict, binding: dict) -> list[dict]:
    by_id = {cell["cell_id"]: cell for cell in scientific["cells"]}
    now = datetime(2026, 7, 17, 20, 0, tzinfo=timezone.utc)
    generation = 1
    rows: list[dict] = []
    inject_retry_cell = binding["blocks"][0]["cells"][1]["cell_id"]
    for block in binding["blocks"]:
        generation_ready = now
        for cell_ref in block["cells"]:
            cell_id = cell_ref["cell_id"]
            statuses = (
                ["INFRA_FAILURE", "COMPLETED"]
                if cell_id == inject_retry_cell
                else ["COMPLETED"]
            )
            prior = None
            for attempt, status in enumerate(statuses, 1):
                if attempt > 1:
                    generation += 1
                    generation_ready = now
                dispatched = now + timedelta(seconds=1)
                started = now + timedelta(seconds=2)
                ended = now + timedelta(seconds=4)
                recorded = now + timedelta(seconds=5)
                cell = by_id[cell_id]
                retry_authorization = None
                retry_of = retry_reason = None
                if prior is not None:
                    retry_of = prior["attempt_id"]
                    retry_reason = prior["failure_reason"]
                    retry_authorization = {
                        "schema": "audit_135m_serial_cell_retry_authorization_v1",
                        "loss_blind": True,
                        "serial_plan_hash": binding["serial_plan_hash"],
                        "cell_id": cell_id,
                        "retry_attempt": attempt,
                        "trigger_attempt_id": retry_of,
                        "trigger_reason": retry_reason,
                        "authorized_at_utc": _iso(now),
                    }
                learner_count = int(cell["target_work"]["learner_count"])
                row = {
                    "attempt_id": f"{cell_id}-attempt-{attempt}",
                    "cell_id": cell_id,
                    "attempt": attempt,
                    "group_id": cell["randomization"]["block_id"],
                    "retry_round": attempt,
                    "actual_wave_index": int(cell_ref["within_block_index"]),
                    "serial_cell_order_index": binding[
                        "cells_in_dispatch_order"
                    ].index(cell_id),
                    "serial_block_order_index": block["block_order_index"],
                    "concurrent_batch_index": 0,
                    "concurrent_batch_slot_set": ["v0"],
                    "time_block_index": block["block_order_index"],
                    "retry_time_block_index": None if attempt == 1 else 1,
                    "available_slot_set": ["v0"],
                    "dispatch_batch_index": 0,
                    "batch_launch_order_index": 0,
                    "launch_order_index": attempt - 1,
                    "logical_slot": "v0",
                    "generation": generation,
                    "run_id": f"run-v0-g{generation}",
                    "ownership_nonce": f"{generation:032x}",
                    "machine_type": "a2-highgpu-1g",
                    "gpu_slots": 1,
                    "instance_numeric_id": str(1000 + generation),
                    "provider_evidence_sha256": f"{generation % 10:x}" * 64,
                    "attempt_prefix": (
                        f"gs://bucket/vms/v0/g{generation}/cells/{cell_id}/"
                        f"attempt-{attempt}/"
                    ),
                    "m": learner_count,
                    "quorum": int(cell["target_work"]["quorum"]),
                    "learner_gpu_slot_map": {
                        str(index): 0 for index in range(learner_count)
                    },
                    "maximum_learners_per_gpu": learner_count,
                    "pairing_identity_hash": cell.get("pairing_identity_hash"),
                    "frozen_command_hash": cell["command_hash"],
                    "executed_command_hash": "a" * 64,
                    "normalized_workload_command_hash": "b" * 64,
                    "fresh_start": _fresh_start(),
                    "retry_of": retry_of,
                    "retry_reason": retry_reason,
                    "retry_authorization": retry_authorization,
                    "vm_ready_at": _iso(generation_ready),
                    "dispatched_at": _iso(dispatched),
                    "scientific_started_at": _iso(started),
                    "scientific_ended_at": _iso(ended),
                    "wave_terminal_prefix_sealed_at": _iso(recorded),
                    "serial_attempt_recorded_at": _iso(recorded),
                    "status": status,
                    "loss": None,
                    "artifact_inventory": {},
                }
                if status == "INFRA_FAILURE":
                    row["failure_reason"] = "provider_spot_preemption"
                else:
                    row.update(
                        {
                            "analysis_loss": 1.0,
                            "analysis_loss_kind": "finite_endpoint",
                            "divergence_retained": False,
                        }
                    )
                rows.append(row)
                prior = row
                now = recorded + timedelta(seconds=2)
        generation += 1
        now += timedelta(seconds=2)
    return rows


def _ratchet_files(
    root: Path, attempts: list[dict], binding: dict, work_reports: list[dict]
) -> None:
    reports = {row["attempt_id"]: row for row in work_reports}
    for row in attempts:
        stem = f"{row['serial_cell_order_index']:04d}-a{row['attempt']:02d}"
        receipt_path = root / "campaign" / "ratchet" / f"{stem}.json"
        remote_uri = f"gs://bucket/campaign/ratchet/{receipt_path.name}"
        receipt = {
            "schema": "audit_135m_serial_attempt_ratchet_receipt_v1",
            "status": "ATTEMPT_RETAINED",
            "loss_exposed": False,
            "serial_plan_hash": binding["serial_plan_hash"],
            "serial_cell_order_index": row["serial_cell_order_index"],
            "cell_id": row["cell_id"],
            "attempt_id": row["attempt_id"],
            "attempt_status": row["status"],
            "cell_resolved": row["status"] in {"COMPLETED", "DIVERGED"},
            "run_id": row["run_id"],
            "generation": row["generation"],
            "instance_numeric_id": row["instance_numeric_id"],
            "provider_evidence_sha256": row["provider_evidence_sha256"],
            "attempt_prefix": row["attempt_prefix"],
            "attempt_canonical_sha256": serial.canonical_sha256(row),
            "artifact_inventory_canonical_sha256": serial.canonical_sha256(
                row["artifact_inventory"]
            ),
            "work_evidence_validated": True,
            "work_evidence_report_canonical_sha256": serial.canonical_sha256(
                reports[row["attempt_id"]]
            ),
            "created_at_utc": row["wave_terminal_prefix_sealed_at"],
            "remote_uri": remote_uri,
        }
        _write_json(receipt_path, receipt)
        ack_path = receipt_path.with_name(receipt_path.stem + ".ack.json")
        _write_json(
            ack_path,
            {
                "schema": "audit_135m_serial_attempt_ratchet_ack_v1",
                "status": "GCS_CREATE_ONLY_ROUNDTRIP_PASS",
                "loss_exposed": False,
                "serial_plan_hash": binding["serial_plan_hash"],
                "cell_id": row["cell_id"],
                "attempt_id": row["attempt_id"],
                "receipt_path": receipt_path.relative_to(root).as_posix(),
                "receipt_raw_sha256": serial.sha256_file(receipt_path),
                "receipt_remote_uri": remote_uri,
                "gcs_create_only": True,
                "gcs_roundtrip_verified": True,
                "verified_at_utc": row["wave_terminal_prefix_sealed_at"],
                "remote_uri": remote_uri.removesuffix(".json") + ".ack.json",
            },
        )


def test_serial_binding_is_deterministic_and_outcome_free(tmp_path: Path) -> None:
    parent, bound, scientific, roster, plan, binding = _design(tmp_path)
    rebuilt = serial.build_serial_binding(
        stage_code="a1d",
        parent=parent,
        bound=bound,
        scientific=scientific,
        compatibility_roster=roster,
        compatibility_plan=plan,
    )
    assert rebuilt == binding
    assert binding["schema"] == serial.SERIAL_BINDING_SCHEMA
    assert binding["execution_mode"] == "serial_single_vm_width_1"
    assert binding["parallel_executor_authorized"] is False
    assert binding["maximum_active_vms"] == 1
    assert binding["maximum_active_a100_equivalent"] == 1
    assert binding["machine_type"] == "a2-highgpu-1g"
    assert binding["cell_count"] == len(scientific["cells"])
    assert binding["cells_in_dispatch_order"] == [
        cell["cell_id"]
        for block in binding["blocks"]
        for cell in block["cells"]
    ]
    assert [block["block_order_index"] for block in binding["blocks"]] == list(
        range(binding["block_count"])
    )
    rendered = json.dumps(binding).casefold()
    assert "analysis_loss" not in rendered
    assert '"results"' not in rendered


def test_serial_runtime_authorization_binds_reviewed_helpers(tmp_path: Path) -> None:
    _parent_value, bound, _scientific, _roster, _plan, binding = _design(tmp_path)
    helpers = {
        "p1_capacity_controller": "1" * 64,
        "gcp_backend_controller": "2" * 64,
    }
    authorization = serial.runtime_authorization(
        binding,
        hard_ceiling_usd=bound["audit_135m_contract"]["hard_ceiling_usd"],
        reviewed_helper_sha256=helpers,
    )
    assert serial.verify_runtime_authorization(authorization, binding) == authorization[
        "authorization_canonical_sha256"
    ]
    tampered = copy.deepcopy(authorization)
    tampered["reviewed_helper_sha256"]["p1_capacity_controller"] = "3" * 64
    with pytest.raises(serial.SerialAuditError, match="authorization"):
        serial.verify_runtime_authorization(tampered, binding)


def test_serial_compatibility_aggregator_requires_frozen_runtime_authorization(
    tmp_path: Path,
) -> None:
    parent, bound, scientific, roster, plan, _binding = _design(tmp_path)
    authorization = {
        "schema": "yeto_audit_135m_runtime_authorization_v1",
        "launch_authorized": True,
        "stage_code": "a1d",
        "audit_135m_design_contract_hash": roster[
            "audit_135m_design_contract_hash"
        ],
        "roster_hash": parallel.roster_hash(roster),
        "parallel_plan_hash": parallel.parallel_plan_hash(plan),
        "bound_manifest_canonical_sha256": parallel.canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific[
            "randomization_plan_hash"
        ],
        "hard_ceiling_usd": roster["hard_ceiling_usd"],
        "spot_only": True,
        "maximum_attached_a100_equivalent": 16,
        "max_idle_before_science_seconds": 600,
    }
    evaluation_registry = {
        str(seed): {
            "path": f"common/evaluation/seed-{seed}.json",
            "sha256": "f" * 64,
            **{field: bound["frozen"][field] for field in parallel.EVAL_BOUND_FIELDS},
        }
        for seed in {cell["seed"] for cell in scientific["cells"]}
    }
    aggregator = serial._compatibility_aggregator(
        stage_code="a1d",
        parent=parent,
        bound=bound,
        scientific=scientific,
        roster=roster,
        parallel_plan=plan,
        vm_registry={},
        evaluation_registry=evaluation_registry,
        final_provider_census={},
        runtime_authorization=authorization,
        campaign_attempt=1,
        campaign_root=tmp_path,
    )
    assert aggregator.runtime_authorization_hash == parallel.canonical_sha256(
        authorization
    )
    with pytest.raises(parallel.LifecycleError, match="runtime authorization"):
        serial._compatibility_aggregator(
            stage_code="a1d",
            parent=parent,
            bound=bound,
            scientific=scientific,
            roster=roster,
            parallel_plan=plan,
            vm_registry={},
            evaluation_registry=evaluation_registry,
            final_provider_census={},
            runtime_authorization={},
            campaign_attempt=1,
            campaign_root=tmp_path,
        )


def test_serial_cell_ratchet_retries_only_infrastructure_failed_cell(
    tmp_path: Path,
) -> None:
    _parent_value, _bound, scientific, _roster, _plan, binding = _design(tmp_path)
    attempts = _schedule(scientific, binding)
    analysis = serial.validate_serial_schedule(
        attempts=attempts, scientific=scientific, binding=binding
    )
    retry_cell = binding["blocks"][0]["cells"][1]["cell_id"]
    first_cell = binding["blocks"][0]["cells"][0]["cell_id"]
    assert [row["cell_id"] for row in attempts].count(first_cell) == 1
    assert [row["cell_id"] for row in attempts].count(retry_cell) == 2
    assert analysis[first_cell]["attempt"] == 1
    assert analysis[retry_cell]["attempt"] == 2
    assert len(analysis) == len(scientific["cells"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.__setitem__(slice(0, 2), reversed(rows[:2])), "reordered"),
        (lambda rows: rows[0].__setitem__("logical_slot", "v1"), "one-slot"),
        (lambda rows: rows[0].__setitem__("machine_type", "a2-highgpu-4g"), "one-slot"),
        (
            lambda rows: rows[0]["fresh_start"].__setitem__("resumed", True),
            "fresh scientific attempt",
        ),
        (
            lambda rows: rows[2].__setitem__("attempt_prefix", rows[1]["attempt_prefix"]),
            "reused or malformed",
        ),
    ],
)
def test_serial_schedule_rejects_invalid_execution(
    tmp_path: Path, mutation, message: str
) -> None:
    _parent_value, _bound, scientific, _roster, _plan, binding = _design(tmp_path)
    attempts = _schedule(scientific, binding)
    mutation(attempts)
    with pytest.raises(serial.SerialAuditError, match=message):
        serial.validate_serial_schedule(
            attempts=attempts, scientific=scientific, binding=binding
        )


def test_serial_schedule_rejects_overlap_completed_peer_retry_and_cross_block_vm(
    tmp_path: Path,
) -> None:
    _parent_value, _bound, scientific, _roster, _plan, binding = _design(tmp_path)
    attempts = _schedule(scientific, binding)
    overlapping = copy.deepcopy(attempts)
    overlapping[1]["dispatched_at"] = overlapping[0]["scientific_started_at"]
    overlapping[1]["scientific_started_at"] = overlapping[0]["scientific_started_at"]
    with pytest.raises(serial.SerialAuditError, match="overlap"):
        serial.validate_serial_schedule(
            attempts=overlapping, scientific=scientific, binding=binding
        )

    completed_retry = copy.deepcopy(attempts)
    completed_retry[1]["status"] = "COMPLETED"
    completed_retry[1].pop("failure_reason")
    with pytest.raises(serial.SerialAuditError, match="later cell"):
        serial.validate_serial_schedule(
            attempts=completed_retry, scientific=scientific, binding=binding
        )

    crossed = copy.deepcopy(attempts)
    first_next_block = next(
        row
        for row in crossed
        if row["serial_block_order_index"] == 1 and row["attempt"] == 1
    )
    prior_block_terminal = [
        row for row in crossed if row["serial_block_order_index"] == 0
    ][-1]
    first_next_block["generation"] = prior_block_terminal["generation"]
    first_next_block["run_id"] = prior_block_terminal["run_id"]
    with pytest.raises(serial.SerialAuditError, match="block boundary"):
        serial.validate_serial_schedule(
            attempts=crossed, scientific=scientific, binding=binding
        )


def test_serial_lifecycle_rejects_two_simultaneous_vms() -> None:
    rows = [
        {
            "machine_type": "a2-highgpu-1g",
            "a100_count": 1,
            "creation_timestamp": "2026-07-17T20:00:00Z",
            "deletion_completed_at_utc": "2026-07-17T20:10:00Z",
        },
        {
            "machine_type": "a2-highgpu-1g",
            "a100_count": 1,
            "creation_timestamp": "2026-07-17T20:09:00Z",
            "deletion_completed_at_utc": "2026-07-17T20:20:00Z",
        },
    ]
    with pytest.raises(serial.SerialAuditError, match="overlap"):
        serial.validate_serial_generation_lifecycles(rows)
    rows[1]["creation_timestamp"] = "2026-07-17T20:10:00Z"
    serial.validate_serial_generation_lifecycles(rows)


def test_serial_generation_gaps_require_exact_transient_provider_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster_hash = "a" * 64
    campaign_root = tmp_path / "campaign-root"
    identities = []
    for generation in (1, 3):
        identity = parallel.GenerationIdentity(
            stage_code="a1d",
            study_id="serial-study",
            roster_hash=roster_hash,
            campaign_attempt=1,
            slot="v0",
            generation=generation,
            ownership_nonce=f"{generation:032x}",
            campaign_state_root=tmp_path / "state",
            campaign_artifact_root="gs://bucket/campaign",
        )
        row = identity.registry_row()
        row.update(
            {
                "region": "us-west4",
                "zone": "us-west4-b",
                "machine_type": "a2-highgpu-1g",
            }
        )
        identities.append(row)
        _write_json(
            campaign_root
            / "vms"
            / "v0"
            / f"g{generation}"
            / "provider"
            / "provider-evidence.json",
            {
                "instance_numeric_id": str(1000 + generation),
                "boot_disk_numeric_id": str(2000 + generation),
            },
        )
    vm_registry = {
        "schema": "yeto_parallel_vm_registry_v1",
        "stage_code": "a1d",
        "study_id": "serial-study",
        "roster_hash": roster_hash,
        "campaign_attempt": 1,
        "generations": identities,
    }
    rows = serial._serial_generation_rows(
        stage_code="a1d",
        study_id="serial-study",
        roster_digest=roster_hash,
        campaign_attempt=1,
        vm_registry=vm_registry,
    )
    assert [row["generation"] for row in rows] == [1, 3]

    monkeypatch.setattr(
        serial.evidence,
        "validate_provider_record",
        lambda provider, _identity: {
            "instance_numeric_id": str(provider["instance_numeric_id"]),
            "boot_disk_numeric_id": str(provider["boot_disk_numeric_id"]),
        },
    )
    transient_identity = parallel.GenerationIdentity(
        stage_code="a1d",
        study_id="serial-study",
        roster_hash=roster_hash,
        campaign_attempt=1,
        slot="v0",
        generation=2,
        ownership_nonce="2" * 32,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root="gs://bucket/campaign",
    )
    run_id = transient_identity.run_id
    lifecycle_path = (
        campaign_root / "common" / "transient-provider-lifecycles" / f"{run_id}.json"
    )
    lifecycle = {
        "schema": "audit_135m_transient_provider_lifecycle_v1",
        "status": "TRANSIENT_PROVIDER_PREEMPTED_AND_EXACT_IDS_ABSENT",
        "stage_code": "a1d",
        "run_id": run_id,
        "slot": "v0",
        "generation": 2,
        "ownership_nonce": "2" * 32,
        "provisioning_model": "SPOT",
        "machine_type": "a2-highgpu-1g",
        "provider_spot_preempted": True,
        "scientific_attempt_started": False,
        "loss_inspected": False,
        "region": "us-west4",
        "zone": "us-west4-b",
        "labels": {
            "campaign": "audit-135m",
            "campaign-tag": roster_hash[:16],
            "draft": "false",
            "logical-slot": "v0",
            "managed-by": "yeto-optimizer-harness",
            "physical-generation": "2",
            "run-id": run_id,
            "stage": "a1d",
        },
        "instance_numeric_id": "1002",
        "boot_disk_name": run_id,
        "boot_disk_numeric_id": "2002",
        "source_image_numeric_id": "7290368630472593484",
        "creation_timestamp": "2026-07-17T20:00:00Z",
        "first_observed_at_utc": "2026-07-17T20:00:01Z",
        "deletion_requested_at_utc": "2026-07-17T20:00:02Z",
        "deletion_completed_at_utc": "2026-07-17T20:00:03Z",
        "teardown_mode": "OPERATOR_EXACT_DELETE",
        "provider_not_found_verification": {
            "instance": {
                "name": run_id,
                "result": "NOT_FOUND",
                "provider_id": "1002",
                "verified_at_utc": "2026-07-17T20:00:03Z",
            },
            "boot_disk": {
                "name": run_id,
                "result": "NOT_FOUND",
                "provider_id": "2002",
                "verified_at_utc": "2026-07-17T20:00:03Z",
            },
        },
        "zero_attached_accelerator_proof": {
            "generation_attached_a100s": 0,
            "instance_numeric_id": "1002",
            "verified_at_utc": "2026-07-17T20:00:03Z",
        },
    }
    _write_json(lifecycle_path, lifecycle)
    transient_registry = {
        "schema": "audit_135m_transient_provider_registry_v1",
        "status": "SEALED_EXACT_ID_EVIDENCE",
        "scientific_attempt_started": False,
        "loss_inspected": False,
        "count": 1,
        "lifecycles": [
            {
                "run_id": run_id,
                "slot": "v0",
                "generation": 2,
                "instance_numeric_id": "1002",
                "boot_disk_numeric_id": "2002",
                "path": lifecycle_path.relative_to(campaign_root).as_posix(),
                "raw_sha256": serial.sha256_file(lifecycle_path),
            }
        ],
    }
    canonical, hashes, intervals = serial._transient_provider_evidence(
        stage_code="a1d",
        roster_digest=roster_hash,
        campaign_attempt=1,
        campaign_root=campaign_root,
        registered_generations=rows,
        transient_provider_registry=transient_registry,
    )
    assert canonical == transient_registry
    assert hashes == [{"slot": "v0", "generation": 2, "sha256": serial.sha256_file(lifecycle_path)}]
    assert intervals[0]["machine_type"] == "a2-highgpu-1g"
    with pytest.raises(serial.SerialAuditError, match="gaps"):
        serial._transient_provider_evidence(
            stage_code="a1d",
            roster_digest=roster_hash,
            campaign_attempt=1,
            campaign_root=campaign_root,
            registered_generations=rows,
            transient_provider_registry={
                **transient_registry,
                "count": 0,
                "lifecycles": [],
            },
        )


def test_serial_ratchet_requires_exact_receipt_ack_and_retry_after_ack(
    tmp_path: Path,
) -> None:
    _parent_value, _bound, scientific, _roster, _plan, binding = _design(tmp_path)
    attempts = _schedule(scientific, binding)
    work_reports = [
        {"attempt_id": row["attempt_id"], "status": row["status"], "report": {}}
        for row in attempts
    ]
    _ratchet_files(tmp_path, attempts, binding, work_reports)
    receipts = serial.validate_ratchet_receipts(
        attempts=attempts,
        work_reports=work_reports,
        binding=binding,
        campaign_root=tmp_path,
    )
    assert len(receipts) == len(attempts)
    ack = next((tmp_path / "campaign" / "ratchet").glob("*.ack.json"))
    ack.unlink()
    with pytest.raises(serial.SerialAuditError):
        serial.validate_ratchet_receipts(
            attempts=attempts,
            work_reports=work_reports,
            binding=binding,
            campaign_root=tmp_path,
        )


def test_a1_serial_forecast_uses_corrected_registered_path_and_filters_zones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _parent_value, _bound, scientific, _roster, _plan, binding = _design(tmp_path)
    west = controller._remaining_forecast_usd(
        scientific=scientific,
        serial_binding=binding,
        resolved_ids=set(),
        region="us-west4",
    )
    east = controller._remaining_forecast_usd(
        scientific=scientific,
        serial_binding=binding,
        resolved_ids=set(),
        region="us-east1",
    )
    assert west == pytest.approx(132.5)
    assert east > west
    monkeypatch.setattr(controller.operations, "_guard_abort_burn", lambda **_kwargs: 0.0)
    monkeypatch.setattr(
        controller.operations,
        "_current_campaign_cost",
        lambda _runtime, _root: (0.0, []),
    )
    eligible = controller._eligible_zones(
        runtime=SimpleNamespace(),
        campaign_root=tmp_path,
        stage_ledger={"estimated_spend_usd": 5.420056},
        scientific=scientific,
        serial_binding=binding,
        resolved_ids=set(),
        ceiling=140.0,
    )
    assert eligible[:2] == ("us-west4-b", "us-west4-a")
    assert "us-east1-b" not in eligible


def test_serial_packet_is_one_slot_and_restores_parallel_builder_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, bound, scientific, _roster, _plan, _binding = _design(tmp_path)
    packet = tmp_path / "packet"
    materialized = packet / "materialized"
    materialized.mkdir(parents=True)
    _write_json(materialized / "bound-manifest.json", bound)
    _write_json(materialized / "scientific-randomization-plan.json", scientific)
    parent_path = tmp_path / "parent.json"
    _write_json(parent_path, parent)
    parallel.bind_campaign_inputs(
        stage_code="a1d",
        parent_manifest_path=parent_path,
        bound_manifest_path=materialized / "bound-manifest.json",
        scientific_plan_path=materialized / "scientific-randomization-plan.json",
        output_dir=packet / "binding",
    )
    original = (
        packet_builder.legacy.AUDIT_BLOCK_WIDTH,
        packet_builder.legacy.MAX_CONCURRENT_BLOCKS,
        packet_builder.legacy.SLOTS,
    )

    def fake_legacy_build(args: argparse.Namespace) -> dict[str, object]:
        _write_json(
            args.packet_root / "identity-plan.json",
            {"hard_ceiling_usd": 140.0},
        )
        _write_json(
            args.packet_root / "review-packet.json",
            {"status": "SEALED_LAUNCH_AUTHORIZED"},
        )
        return {"status": "SEALED_LAUNCH_AUTHORIZED"}

    monkeypatch.setattr(packet_builder.legacy, "build", fake_legacy_build)
    seed_registry = tmp_path / "seed-registry.json"
    _write_json(seed_registry, {})
    result = packet_builder.build(
        argparse.Namespace(
            stage_code="a1d",
            packet_root=packet,
            parent_manifest=parent_path,
            seed_bundle_registry=seed_registry,
            worker_wrapper=Path("worker.py"),
            science_root=Path("/tmp/audit-135m-science/test"),
            initial_zone="us-east1-b",
        )
    )
    assert result["status"] == "SEALED_SERIAL_LAUNCH_AUTHORIZED"
    identity = serial.load_object(packet / "identity-plan.json", "identity")
    review = serial.load_object(packet / "review-packet.json", "review")
    authorization = serial.load_object(
        packet / "serial-runtime-authorization.json", "authorization"
    )
    assert identity["logical_slots"] == ["v0"]
    assert identity["target_width"] == 1
    assert identity["maximum_concurrent_blocks"] == 1
    assert identity["parallel_executor_authorized"] is False
    assert review["status"] == "SEALED_SERIAL_LAUNCH_AUTHORIZED"
    assert review["parallel_executor_authorized"] is False
    assert authorization["reviewed_helper_sha256"] == identity[
        "reviewed_helper_sha256"
    ]
    assert (
        packet_builder.legacy.AUDIT_BLOCK_WIDTH,
        packet_builder.legacy.MAX_CONCURRENT_BLOCKS,
        packet_builder.legacy.SLOTS,
    ) == original


def test_phase_promotion_routes_serial_seal_to_serial_reaggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = tmp_path / "descriptor.json"
    campaign_path = tmp_path / "campaign.json"
    seal_path = tmp_path / "seal.json"
    campaign = {
        "schema": serial.SERIAL_MANIFEST_SCHEMA,
        "parallel_executor_used": False,
        "partial_outcomes_exposed": False,
    }
    seal = {
        "schema": serial.SERIAL_SEAL_SCHEMA,
        "status": "sealed_results",
        "partial_outcomes_exposed": False,
        "work_evidence_all_pass": True,
        "provider_ownership_all_pass": True,
        "exact_id_teardown_all_pass": True,
        "parallel_executor_used": False,
        "completed_cell_ratchet_all_pass": True,
        "generation_lineage_all_pass": True,
        "campaign_manifest_canonical_sha256": promotion.canonical_sha256(campaign),
        "sealed_at_utc": "2026-07-17T20:00:00Z",
    }
    _write_json(descriptor, {"aggregation_authorized": True})
    _write_json(campaign_path, campaign)
    _write_json(seal_path, seal)
    called = []

    def reproduce(path: Path, *, write_seal: bool, sealed_at_utc: str):
        called.append((path, write_seal, sealed_at_utc))
        return seal

    monkeypatch.setattr(promotion.serial, "aggregate_from_descriptor", reproduce)
    assert promotion._verify_campaign(
        descriptor=descriptor,
        campaign_manifest_path=campaign_path,
        campaign_seal_path=seal_path,
    ) == (campaign, seal)
    assert called == [(descriptor.resolve(), False, seal["sealed_at_utc"])]
