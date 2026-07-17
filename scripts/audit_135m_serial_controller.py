#!/usr/bin/env python3
"""Execute one audit-135M suffix on a single Spot 1g VM at a time.

The controller deliberately does not instantiate or call
``run_parallel_phase_map.ParallelWaveExecutor``.  It reuses only reviewed
provider/evidence primitives, dispatches the already-frozen scientific cells
sequentially, and verifies a durable GCS ratchet receipt after every terminal
cell before continuing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import audit_135m_serial as serial
from scripts import audit_135m_campaign_controller as operations


REPO_ROOT = Path(__file__).resolve().parents[1]
P1_SESSION = Path("/private/tmp/yeto-p1r0-launcher/p1-adaptive-session")
P1_CONTROLLER = P1_SESSION / "p1ad_campaign_controller.py"
R0_CONTROLLER = Path("/private/tmp/yeto-p1r0-launcher/p1r0-session/p1r0_controller.py")
GCLOUD_CONFIG = "/private/tmp/yeto-gcloud-admin-codex"
NOTE_PATH = Path("/private/tmp/audit-135m-note.md")
SLOT = "v0"
ZONE_ROTATION = (
    "us-east1-b",
    "us-west4-b",
    "us-west1-b",
    "us-west4-a",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
)
FULL_STAGE_FORECAST_HOURS = {
    "A1": 48 * operations.CELL_HOURS[16] + 48 * operations.CELL_HOURS[256],
    "A3": (
        10 * operations.CELL_HOURS[8]
        + 12 * operations.CELL_HOURS[512]
        + operations.CELL_HOURS[16]
        + operations.FINITE_KERNEL_EXTRA_HOURS[16]
        + operations.CELL_HOURS[64]
        + operations.FINITE_KERNEL_EXTRA_HOURS[64]
        + operations.CELL_HOURS[256]
        + operations.FINITE_KERNEL_EXTRA_HOURS[256]
    ),
    "A4": 80 * operations.CELL_HOURS[16] + 80 * operations.CELL_HOURS[256],
}


class SerialControllerError(RuntimeError):
    pass


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class SerialRuntime:
    """Minimal one-slot runtime using reviewed generation evidence objects."""

    def __init__(
        self,
        *,
        pexec,
        roster: Mapping[str, Any],
        parallel_plan: Mapping[str, Any],
        scientific: Mapping[str, Any],
        bound: Mapping[str, Any],
        registry,
        campaign_root: Path,
        backend,
        compatibility_authorization: Mapping[str, Any],
    ) -> None:
        self.pexec = pexec
        self.roster = dict(roster)
        self.parallel_plan = dict(parallel_plan)
        self.scientific = dict(scientific)
        self.bound = dict(bound)
        self.registry = registry
        self.campaign_root = campaign_root.resolve()
        self.backend = backend
        self.roster_digest = pexec.roster_hash(roster)
        self.parallel_digest = pexec.parallel_plan_hash(parallel_plan)
        self.bound_digest = pexec.canonical_sha256(bound)
        self.scientific_digest = str(scientific["randomization_plan_hash"])
        self.runtime_authorization_hash = pexec.multiseed_runtime_authorization_hash(
            stage_code=str(roster["stage_code"]),
            design_contract_hash=roster["audit_135m_design_contract_hash"],
            roster_digest=self.roster_digest,
            parallel_digest=self.parallel_digest,
            bound_digest=self.bound_digest,
            scientific_digest=self.scientific_digest,
            authorization=compatibility_authorization,
            hard_ceiling_usd=roster["hard_ceiling_usd"],
        )
        self.active: dict[str, Any] = {}
        self.providers: dict[tuple[str, int], dict[str, Any]] = {}
        self.partials: dict[tuple[str, int], Any] = {}
        self.ready_at: dict[tuple[str, int], str] = {}

    def _local_vm_root(self, identity) -> Path:
        return (
            self.campaign_root
            / "vms"
            / identity.slot
            / f"g{identity.generation}"
        )

    def _common_bindings(self) -> dict[str, Any]:
        return {
            "roster_hash": self.roster_digest,
            "parallel_plan_hash": self.parallel_digest,
            "bound_manifest_canonical_sha256": self.bound_digest,
            "scientific_randomization_plan_hash": self.scientific_digest,
            # Compatibility evidence stays on the already accepted revision-1.3
            # schema; the separate serial binding is the sole scheduler authority.
            "amendment_raw_sha256": self.pexec.AMENDMENT_RAW_SHA256,
            "multiseed_runtime_authorization_hash": self.runtime_authorization_hash,
        }

    def _assert_capacity_census(self) -> Mapping[str, Any]:
        census = dict(self.backend.census(self.roster_digest[:16]))
        if (
            int(census.get("campaign_owned_vm_count", -1)) > 1
            or int(census.get("campaign_owned_attached_a100s", -1)) > 1
        ):
            raise SerialControllerError("serial campaign exceeded one VM/A100")
        return census

    def _finalize_identity(self, identity, *, preempted: bool) -> None:
        key = (identity.slot, identity.generation)
        partial = self.partials[key]
        provider = self.providers[key]
        digest = partial.hash_lock(hash_locked_at_utc=operations.utc_now())
        lifecycle = dict(
            self.backend.finalize_generation(
                identity, provider, digest, preempted=preempted
            )
        )
        self.pexec.validate_lifecycle_record(
            lifecycle, identity.registry_row(), provider, digest
        )
        lifecycle_path = (
            self._local_vm_root(identity)
            / "manifests"
            / "vm-lifecycle-final.json"
        )
        self.pexec.write_json_create_only(lifecycle_path, lifecycle)
        self.registry.update_state(
            identity,
            status="vm_lifecycle_final",
            partial_manifest_sha256=digest,
            lifecycle_record_sha256=self.pexec.sha256_file(lifecycle_path),
        )
        self.backend.upload_lifecycles(self.registry.snapshot())
        if self.active.get(identity.slot) == identity:
            del self.active[identity.slot]


def _remaining_forecast_usd(
    *,
    scientific: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    resolved_ids: set[str],
    region: str,
) -> float:
    by_id = {str(row["cell_id"]): row for row in scientific["cells"]}
    current_hours = sum(
        operations._cell_forecast_hours(by_id[cell_id])
        for cell_id in serial_binding["cells_in_dispatch_order"]
        if cell_id not in resolved_ids
    )
    stage_code = str(serial_binding["stage_code"])
    future_hours = sum(
        operations.CELL_HOURS[int(horizon)] * int(count)
        for horizon, count in operations.FUTURE_STAGE_CELL_COUNTS[
            stage_code
        ].items()
    )
    hours = current_hours + future_hours
    price = operations.PRICE_PER_VM_HOUR[region]["a2-highgpu-1g"]
    computed = hours * price * operations.SPOT_PREEMPTION_RESERVE_FACTOR
    audit_stage = str(serial_binding["audit_stage"])
    authority = serial.audit.load_authority()
    registered_upper = float(
        authority["costs"]["blocks"][audit_stage]["range_usd"][1]
    )
    full_hours = FULL_STAGE_FORECAST_HOURS[audit_stage]
    cheapest_price = min(
        row["a2-highgpu-1g"] for row in operations.PRICE_PER_VM_HOUR.values()
    )
    registered_scaled = (
        registered_upper * (hours / full_hours) * (price / cheapest_price)
    )
    return max(computed, registered_scaled)


def _eligible_zones(
    *,
    runtime: SerialRuntime,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    scientific: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    resolved_ids: set[str],
    ceiling: float,
) -> tuple[str, ...]:
    operations._guard_abort_burn(
        executor=runtime,
        campaign_root=campaign_root,
        stage_ledger=stage_ledger,
        phase="serial_zone_eligibility",
    )
    current, _rows = operations._current_campaign_cost(runtime, campaign_root)
    rows = []
    for zone in ZONE_ROTATION:
        region = zone.rsplit("-", 1)[0]
        forecast = _remaining_forecast_usd(
            scientific=scientific,
            serial_binding=serial_binding,
            resolved_ids=resolved_ids,
            region=region,
        )
        projected = float(stage_ledger["estimated_spend_usd"]) + current + forecast
        rows.append(
            {
                "zone": zone,
                "region": region,
                "remaining_forecast_with_preemption_reserve_usd": round(
                    forecast, 6
                ),
                "projected_stage_spend_usd": round(projected, 6),
                "eligible": projected < ceiling,
            }
        )
    eligible = tuple(row["zone"] for row in rows if row["eligible"])
    evidence = {
        "schema": "audit_135m_serial_zone_cost_eligibility_v1",
        "status": "PASS" if eligible else "HARD_CEILING_INFEASIBLE",
        "loss_inspected": False,
        "resolved_cell_count": len(resolved_ids),
        "remaining_cell_count": len(scientific["cells"]) - len(resolved_ids),
        "prior_stage_spend_usd": float(stage_ledger["estimated_spend_usd"]),
        "current_campaign_spend_usd": round(current, 6),
        "hard_ceiling_usd": ceiling,
        "zones": rows,
        "eligible_zone_rotation": list(eligible),
        "evaluated_at_utc": operations.utc_now(),
    }
    _write_atomic(
        campaign_root / "campaign" / "serial-zone-cost-eligibility.json",
        evidence,
    )
    if not eligible:
        cheapest = min(rows, key=lambda row: row["projected_stage_spend_usd"])
        raise operations.HardCeilingStop(
            "serial remaining-work forecast has no cost-eligible zone: cheapest "
            f"{cheapest['zone']} projects ${cheapest['projected_stage_spend_usd']:.6f} "
            f"against the ${ceiling:.2f} ceiling"
        )
    return eligible


def _prelaunch_guard(
    *,
    runtime: SerialRuntime,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    scientific: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    resolved_ids: set[str],
    zone: str,
    ceiling: float,
) -> None:
    operations._guard_abort_burn(
        executor=runtime,
        campaign_root=campaign_root,
        stage_ledger=stage_ledger,
        phase="serial_prelaunch",
    )
    current, _rows = operations._current_campaign_cost(runtime, campaign_root)
    region = zone.rsplit("-", 1)[0]
    forecast = _remaining_forecast_usd(
        scientific=scientific,
        serial_binding=serial_binding,
        resolved_ids=resolved_ids,
        region=region,
    )
    projected = float(stage_ledger["estimated_spend_usd"]) + current + forecast
    evidence_row = {
        "schema": "audit_135m_serial_prelaunch_cost_guard_v1",
        "status": "PASS" if projected < ceiling else "STOP",
        "loss_inspected": False,
        "zone": zone,
        "region": region,
        "resolved_cell_count": len(resolved_ids),
        "remaining_cell_count": len(scientific["cells"]) - len(resolved_ids),
        "prior_stage_spend_usd": float(stage_ledger["estimated_spend_usd"]),
        "current_campaign_spend_usd": round(current, 6),
        "remaining_forecast_with_preemption_reserve_usd": round(forecast, 6),
        "projected_stage_spend_usd": round(projected, 6),
        "hard_ceiling_usd": ceiling,
        "evaluated_at_utc": operations.utc_now(),
    }
    _write_atomic(campaign_root / "campaign" / "serial-live-cost-guard.json", evidence_row)
    if projected >= ceiling:
        raise operations.HardCeilingStop(
            f"serial remaining-work forecast ${projected:.6f} reaches/exceeds "
            f"the ${ceiling:.2f} A1/A3/A4 ceiling"
        )


def _provision_one(
    *,
    base,
    p1,
    pexec,
    runtime: SerialRuntime,
    registry,
    backend,
    campaign_root: Path,
    stage_ledger: Mapping[str, Any],
    scientific: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    resolved_ids: set[str],
    ceiling: float,
) -> Any:
    identity = operations._audit_provisional_identity(
        pexec=pexec, registry=registry, backend=backend, slot=SLOT
    )
    zone_index = 0
    tries = 0
    while True:
        eligible_zones = _eligible_zones(
            runtime=runtime,
            campaign_root=campaign_root,
            stage_ledger=stage_ledger,
            scientific=scientific,
            serial_binding=serial_binding,
            resolved_ids=resolved_ids,
            ceiling=ceiling,
        )
        zone = eligible_zones[zone_index % len(eligible_zones)]
        started = time.time()
        tries += 1
        try:
            _prelaunch_guard(
                runtime=runtime,
                campaign_root=campaign_root,
                stage_ledger=stage_ledger,
                scientific=scientific,
                serial_binding=serial_binding,
                resolved_ids=resolved_ids,
                zone=zone,
                ceiling=ceiling,
            )
            p1.cache_zone_render(backend, identity, zone)
            backend.note(
                f"SERIAL CAPACITY probe {tries} for {identity.slot}/g"
                f"{identity.generation}: {zone}, Spot a2-highgpu-1g only; "
                "one active VM maximum and losses SEALED/BLINDED."
            )
            provider = dict(backend.provision(identity))
            pexec.validate_provider_record(provider, identity.registry_row())
            p1.initialize_registered_generation(
                base=base,
                pexec=pexec,
                executor=runtime,
                registry=registry,
                identity=identity,
                provider=provider,
            )
            p1.mark_ready(
                pexec=pexec,
                executor=runtime,
                registry=registry,
                identity=identity,
                provider=provider,
                active=True,
            )
            runtime._assert_capacity_census()
            backend.note(
                f"SERIAL READY {identity.slot}/g{identity.generation}: exact "
                f"instance {provider['instance_numeric_id']} in {provider['zone']}; "
                "the next unbanked cell dispatches immediately."
            )
            return identity
        except operations.TransientProviderGeneration as transient:
            previous = identity
            zone_index += 1
            identity = operations._audit_provisional_identity(
                pexec=pexec, registry=registry, backend=backend, slot=SLOT
            )
            backend.note(
                f"SERIAL PRE-READY TRANSIENT finalized for {previous.run_id}; "
                f"fresh exact identity {identity.run_id} continues in the next zone."
            )
            continue
        except (operations.HardCeilingStop, operations.AbortBurnStop):
            raise
        except Exception as exc:
            text = str(exc).casefold()
            absent = (
                backend.describe_instance(identity.run_id, check=False) is None
                and not (Path(base.HARNESS_STATE_ROOT) / f"{identity.run_id}.json").exists()
            )
            unsupported = (
                "machine type with name" in text
                and "does not exist in zone" in text
                and absent
            )
            capacity = base._capacity_stockout(exc) or (
                "operation was canceled by user" in text and absent
            )
            if unsupported:
                zone_index += 1
                backend.note(
                    f"SERIAL ZONE-CATALOG SKIP {identity.run_id}: {zone} has no A2 "
                    "shape; no provider identity was created."
                )
                continue
            if capacity and absent:
                zone_index += 1
                next_zone = eligible_zones[zone_index % len(eligible_zones)]
                backend.note(
                    f"SERIAL STOCKOUT {identity.run_id} in {zone}; exact name/nonce "
                    f"remain unused and {next_zone} follows after the 600s cadence."
                )
                remaining = max(0.0, started + 600 - time.time())
                while remaining > 0:
                    time.sleep(min(30.0, remaining))
                    remaining = max(0.0, started + 600 - time.time())
                continue
            raise


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


def _request_for_cell(
    *,
    pexec,
    identity,
    provider: Mapping[str, Any],
    cell: Mapping[str, Any],
    cell_order_index: int,
    block_order_index: int,
    attempt: int,
    prior: Mapping[str, Any] | None,
    serial_binding: Mapping[str, Any],
) -> Any:
    shape = pexec.machine_shape_contract(provider["machine_type"])
    learner_count = int(cell["target_work"]["learner_count"])
    retry_authorization = None
    retry_of = retry_reason = None
    if prior is not None:
        retry_of = str(prior["attempt_id"])
        retry_reason = str(prior["failure_reason"])
        retry_authorization = {
            "schema": "audit_135m_serial_cell_retry_authorization_v1",
            "loss_blind": True,
            "serial_plan_hash": serial_binding["serial_plan_hash"],
            "cell_id": cell["cell_id"],
            "retry_attempt": attempt,
            "trigger_attempt_id": retry_of,
            "trigger_reason": retry_reason,
            "authorized_at_utc": operations.utc_now(),
        }
    return pexec.DispatchRequest(
        group_id=str(cell["randomization"]["block_id"]),
        cell_id=str(cell["cell_id"]),
        retry_round=attempt,
        actual_wave_index=cell_order_index,
        concurrent_batch_index=0,
        concurrent_batch_slot_set=(SLOT,),
        time_block_index=block_order_index,
        retry_time_block_index=(None if attempt == 1 else cell_order_index),
        available_slot_set=(SLOT,),
        dispatch_batch_index=0,
        batch_launch_order_index=0,
        launch_order_index=attempt - 1,
        attempt_prefix=identity.attempt_prefix(str(cell["cell_id"]), attempt),
        learner_count=learner_count,
        quorum=int(cell["target_work"]["quorum"]),
        gpu_slots=shape["gpu_slots"],
        learner_gpu_slot_map=pexec.learner_gpu_slot_map(
            learner_count, shape["gpu_slots"]
        ),
        pairing_identity_hash=cell.get("pairing_identity_hash"),
        command=tuple(cell["command"]),
        command_hash=str(cell["command_hash"]),
        fresh_start=_fresh_start(),
        retry_of=retry_of,
        retry_reason=retry_reason,
        retry_authorization=retry_authorization,
    )


def _attempt_row(
    *,
    pexec,
    runtime: SerialRuntime,
    identity,
    provider: Mapping[str, Any],
    cell: Mapping[str, Any],
    request,
    outcome: Mapping[str, Any],
    dispatched_at: str,
    cell_order_index: int,
    block_order_index: int,
) -> dict[str, Any]:
    result = dict(outcome)
    if result.get("status") in {"COMPLETED", "DIVERGED"}:
        result.update(
            pexec.multiseed_analysis_outcome_fields(
                status=str(result["status"]),
                loss=result.get("loss"),
                divergence_loss_cap=float(runtime.bound["analysis_policy"]["divergence_loss_cap"]),
                checkpoint_only=pexec.audit_checkpoint_only(cell),
            )
        )
    shape = pexec.machine_shape_contract(provider["machine_type"])
    gpu_map = pexec.learner_gpu_slot_map(request.learner_count, shape["gpu_slots"])
    projected = pexec.project_scientific_command_for_machine_type(
        request.command, provider["machine_type"]
    )
    terminal = operations.utc_now()
    row = {
        **result,
        "attempt_id": f"{request.cell_id}-attempt-{request.retry_round}",
        "cell_id": request.cell_id,
        "attempt": request.retry_round,
        "group_id": request.group_id,
        "retry_round": request.retry_round,
        "actual_wave_index": request.actual_wave_index,
        "serial_cell_order_index": cell_order_index,
        "serial_block_order_index": block_order_index,
        "concurrent_batch_index": 0,
        "concurrent_batch_slot_set": [SLOT],
        "time_block_index": block_order_index,
        "retry_time_block_index": request.retry_time_block_index,
        "available_slot_set": [SLOT],
        "dispatch_batch_index": 0,
        "batch_launch_order_index": 0,
        "launch_order_index": request.launch_order_index,
        "logical_slot": SLOT,
        "generation": identity.generation,
        "run_id": identity.run_id,
        "ownership_nonce": identity.ownership_nonce,
        "machine_type": provider["machine_type"],
        "gpu_slots": shape["gpu_slots"],
        "instance_numeric_id": provider["instance_numeric_id"],
        "provider_evidence_sha256": pexec.sha256_file(
            runtime._local_vm_root(identity)
            / "provider"
            / "provider-evidence.json"
        ),
        "attempt_prefix": request.attempt_prefix,
        "m": request.learner_count,
        "quorum": request.quorum,
        "learner_gpu_slot_map": gpu_map,
        "maximum_learners_per_gpu": max(Counter(gpu_map.values()).values()),
        "pairing_identity_hash": request.pairing_identity_hash,
        "frozen_command_hash": request.command_hash,
        "executed_command_hash": pexec.canonical_sha256(projected),
        "normalized_workload_command_hash": pexec.canonical_sha256(
            pexec.normalized_workload_command(projected)
        ),
        "fresh_start": dict(request.fresh_start),
        "retry_of": request.retry_of,
        "retry_reason": request.retry_reason,
        "retry_authorization": (
            None
            if request.retry_authorization is None
            else dict(request.retry_authorization)
        ),
        "vm_ready_at": runtime.ready_at[(SLOT, identity.generation)],
        "dispatched_at": dispatched_at,
        "wave_terminal_prefix_sealed_at": terminal,
        "serial_attempt_recorded_at": terminal,
    }
    pexec.parse_time(row.get("scientific_started_at"), "scientific start")
    pexec.parse_time(row.get("scientific_ended_at"), "scientific end")
    return row


def _validate_attempt_work(
    *,
    pexec,
    stage_code: str,
    row: Mapping[str, Any],
    scientific_cell: Mapping[str, Any],
    roster_cell: Mapping[str, Any],
    campaign_root: Path,
    evaluation_registry: Mapping[str, Any],
    divergence_loss_cap: float,
) -> dict[str, Any]:
    merged_cell = {**dict(roster_cell), **dict(scientific_cell)}
    status = row.get("status")
    if status == "COMPLETED":
        report = pexec.validate_completed_attempt_work(
            stage_code=stage_code,
            row=row,
            cell=merged_cell,
            expected_command=scientific_cell["command"],
            campaign_root=campaign_root,
            evaluation_registry=evaluation_registry,
        )
    elif status == "DIVERGED":
        report = pexec.validate_diverged_attempt(
            row=row,
            cell=merged_cell,
            expected_command=scientific_cell["command"],
            campaign_root=campaign_root,
        )
    elif status == "INFRA_FAILURE":
        pexec.validate_infrastructure_attempt(row, campaign_root)
        report = {"infrastructure_failure": row["failure_reason"]}
    else:
        raise SerialControllerError(
            f"cell {row.get('cell_id')} returned unsealable status {status}"
        )
    if status in {"COMPLETED", "DIVERGED"}:
        expected_analysis = pexec.multiseed_analysis_outcome_fields(
            status=str(status),
            loss=None if status == "DIVERGED" else report.get("loss"),
            divergence_loss_cap=divergence_loss_cap,
            checkpoint_only=pexec.audit_checkpoint_only(merged_cell),
        )
        if any(row.get(key) != value for key, value in expected_analysis.items()):
            raise SerialControllerError(
                "serial attempt analysis fields differ from validated work evidence"
            )
        report = {**report, **expected_analysis}
    return {"attempt_id": row["attempt_id"], "status": status, "report": report}


def _ratchet_sync(
    *,
    backend,
    runtime: SerialRuntime,
    identity,
    row: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    work_report: Mapping[str, Any],
) -> None:
    backend._upload_controller_outer(
        identity, ["manifests/vm-partial-manifest.json"]
    )
    receipt = {
        "schema": "audit_135m_serial_attempt_ratchet_receipt_v1",
        "status": "ATTEMPT_RETAINED",
        "loss_exposed": False,
        "serial_plan_hash": serial_binding["serial_plan_hash"],
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
            row.get("artifact_inventory")
        ),
        "work_evidence_validated": True,
        "work_evidence_report_canonical_sha256": serial.canonical_sha256(
            work_report
        ),
        "created_at_utc": operations.utc_now(),
    }
    local = (
        runtime.campaign_root
        / "campaign"
        / "ratchet"
        / f"{int(row['serial_cell_order_index']):04d}-a{int(row['attempt']):02d}.json"
    )
    uri = (
        runtime.backend.identity_plan["campaign_artifact_root"].rstrip("/")
        + "/ratchet/"
        + local.name
    )
    receipt["remote_uri"] = uri
    serial.write_json_create_only(local, receipt)
    exists = backend.gcloud(
        "storage", "ls", "--all-versions", uri, check=False
    )
    if exists.returncode == 0 and exists.stdout.strip():
        raise SerialControllerError("ratchet receipt URI already exists")
    backend.gcloud(
        "storage",
        "cp",
        str(local),
        uri,
        "--if-generation-match=0",
        timeout=900,
    )
    proof = local.with_suffix(".roundtrip.json")
    backend.gcloud("storage", "cp", uri, str(proof), timeout=900)
    if serial.sha256_file(proof) != serial.sha256_file(local):
        raise SerialControllerError("ratchet receipt GCS round-trip differs")
    proof.unlink()
    ack = {
        "schema": "audit_135m_serial_attempt_ratchet_ack_v1",
        "status": "GCS_CREATE_ONLY_ROUNDTRIP_PASS",
        "loss_exposed": False,
        "serial_plan_hash": serial_binding["serial_plan_hash"],
        "cell_id": row["cell_id"],
        "attempt_id": row["attempt_id"],
        "receipt_path": local.relative_to(runtime.campaign_root).as_posix(),
        "receipt_raw_sha256": serial.sha256_file(local),
        "receipt_remote_uri": uri,
        "gcs_create_only": True,
        "gcs_roundtrip_verified": True,
        "verified_at_utc": operations.utc_now(),
    }
    ack_path = local.with_name(local.stem + ".ack.json")
    ack_uri = uri.removesuffix(".json") + ".ack.json"
    ack["remote_uri"] = ack_uri
    serial.write_json_create_only(ack_path, ack)
    exists = backend.gcloud(
        "storage", "ls", "--all-versions", ack_uri, check=False
    )
    if exists.returncode == 0 and exists.stdout.strip():
        raise SerialControllerError("ratchet acknowledgement URI already exists")
    backend.gcloud(
        "storage",
        "cp",
        str(ack_path),
        ack_uri,
        "--if-generation-match=0",
        timeout=900,
    )
    ack_proof = ack_path.with_suffix(".roundtrip.json")
    backend.gcloud("storage", "cp", ack_uri, str(ack_proof), timeout=900)
    if serial.sha256_file(ack_proof) != serial.sha256_file(ack_path):
        raise SerialControllerError("ratchet acknowledgement GCS round-trip differs")
    ack_proof.unlink()
    backend.note(
        f"SERIAL RATCHET PASS cell {row['cell_id']} attempt {row['attempt']}: "
        f"status={row['status']}, full work evidence and immutable GCS receipt "
        "verified; loss remains SEALED/BLINDED."
    )


def _final_census(runtime: SerialRuntime) -> dict[str, Any]:
    raw = dict(runtime._assert_capacity_census())
    if raw.get("campaign_owned_vm_count") or raw.get(
        "campaign_owned_attached_a100s"
    ):
        raise SerialControllerError("serial final provider census is not zero")
    global_a100 = operations._global_a100_census(runtime.backend)
    if (
        int(global_a100["total_attached_a100_equivalent"])
        > operations.GLOBAL_A100_CEILING
    ):
        raise SerialControllerError("final global A100 census exceeds sixteen")
    instance_result = runtime.backend.gcloud(
        "compute",
        "instances",
        "list",
        "--project=model-training-497007",
        "--format=json",
    )
    instances = json.loads(instance_result.stdout)
    protected = [
        row
        for row in instances
        if str(row.get("id")) == operations.PROTECTED_INSTANCE_ID
    ]
    if len(protected) != 1:
        raise SerialControllerError("protected instance is absent or duplicated at final census")
    return {
        "schema": "yeto_parallel_final_provider_census_v1",
        "campaign_owned_vm_count": 0,
        "campaign_owned_attached_a100s": 0,
        "campaign_owned_instance_ids": [],
        "queried_at_utc": raw.get("queried_at_utc", operations.utc_now()),
        "global_a100_census": global_a100,
        "protected_instance_numeric_id": operations.PROTECTED_INSTANCE_ID,
        "protected_instance_name": protected[0].get("name"),
        "protected_instance_status": protected[0].get("status"),
        "protected_instance_untouched_by_exact_id_teardown": True,
    }


def _upload_campaign(backend, campaign_root: Path, artifact_root: str) -> None:
    backend.gcloud(
        "storage",
        "rsync",
        "--recursive",
        str(campaign_root / "campaign"),
        artifact_root.rstrip("/") + "/campaign",
        timeout=1800,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    packet = args.packet_root.resolve()
    campaign_root = args.campaign_root.resolve()
    if campaign_root.exists() and any(campaign_root.iterdir()):
        raise SerialControllerError(f"campaign root is not empty: {campaign_root}")
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "campaign").mkdir(parents=True, exist_ok=True)

    identity_plan = serial.load_object(packet / "identity-plan.json", "identity plan")
    review = serial.load_object(packet / "review-packet.json", "review packet")
    serial_binding = serial.load_object(
        packet / "binding" / "serial-binding.json", "serial binding"
    )
    serial_authorization = serial.load_object(
        packet / "serial-runtime-authorization.json", "serial authorization"
    )
    if (
        identity_plan.get("execution_mode") != "serial_single_vm_width_1"
        or identity_plan.get("parallel_executor_authorized") is not False
        or review.get("status") != "SEALED_SERIAL_LAUNCH_AUTHORIZED"
        or review.get("parallel_executor_authorized") is not False
        or identity_plan.get("logical_slots") != [SLOT]
        or int(identity_plan.get("target_width", 0)) != 1
        or int(identity_plan.get("target_1g_slot_count", 0)) != 1
    ):
        raise SerialControllerError("packet is not one-slot serial launch authority")
    serial.verify_runtime_authorization(
        serial_authorization,
        serial_binding,
        expected_hard_ceiling_usd=float(identity_plan["hard_ceiling_usd"]),
    )
    current_helper_sha256 = {
        "p1_capacity_controller": serial.sha256_file(P1_CONTROLLER),
        "gcp_backend_controller": serial.sha256_file(R0_CONTROLLER),
    }
    authorized_helper_sha256 = serial.normalize_helper_hashes(
        serial_authorization.get("reviewed_helper_sha256")
    )
    if (
        current_helper_sha256 != authorized_helper_sha256
        or identity_plan.get("reviewed_helper_sha256")
        != authorized_helper_sha256
        or review.get("reviewed_helper_sha256") != authorized_helper_sha256
    ):
        raise SerialControllerError("reviewed serial P1/backend helper hashes differ")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    sys.path.insert(0, str(P1_SESSION))
    sys.path.insert(0, "/private/tmp/yeto-p1r0-launcher")
    from scripts import run_parallel_phase_map as pexec
    import build_audit_135m_launch_packet as packet_builder

    packet_builder.configure_from_identity_plan(identity_plan)
    sys.modules["build_launch_packet"] = packet_builder
    p1 = operations.load_module("audit_135m_serial_p1_capacity", P1_CONTROLLER)
    p1.STAGE_CODE = str(identity_plan["stage_code"])
    p1.SLOTS = (SLOT,)
    p1.ZONE_ROTATION = ZONE_ROTATION
    p1.NOTE_PATH = NOTE_PATH
    base = operations.load_module("audit_135m_serial_gcp_backend", R0_CONTROLLER)
    p1.patch_base(base, identity_plan)
    operations._bind_reviewed_backend_workspace(base)
    base.NOTE_PATH = NOTE_PATH
    base.GCLOUD_CONFIG = GCLOUD_CONFIG
    base.ALLOWED_ZONES = ZONE_ROTATION
    os.environ["P1AD_SCIENCE_ROOT"] = str(identity_plan["science_root"])

    roster = base.load_json(packet / "binding" / "parallel-roster.json")
    plan = base.load_json(packet / "binding" / "parallel-plan.json")
    scientific = base.load_json(
        packet / "materialized" / "scientific-randomization-plan.json"
    )
    bound = base.load_json(packet / "materialized" / "bound-manifest.json")
    parent = base.load_json(packet / "parent" / "parent-manifest.json")
    compatibility_authorization = base.load_json(packet / "runtime-authorization.json")
    seed_registry = base.load_json(packet / "inputs" / "seed-bundle-registry.json")
    expected_binding = serial.build_serial_binding(
        stage_code=str(identity_plan["stage_code"]),
        parent=parent,
        bound=bound,
        scientific=scientific,
        compatibility_roster=roster,
        compatibility_plan=plan,
    )
    if serial.canonical_json(expected_binding) != serial.canonical_json(serial_binding):
        raise SerialControllerError("packet serial binding does not reconstruct")
    evaluation_registry = operations._evaluation_registry(
        packet=packet,
        campaign_root=campaign_root,
        seed_registry=seed_registry,
        bound=bound,
    )
    registry = base.PreboundRegistry(
        pexec,
        prebound_nonces={
            row["slot"]: row["ownership_nonce"]
            for row in identity_plan["generations"]
        },
        stage_code=str(identity_plan["stage_code"]),
        study_id=bound["study_id"],
        roster_digest=identity_plan["roster_hash"],
        campaign_attempt=int(identity_plan["campaign_attempt"]),
        campaign_state_root=Path(identity_plan["campaign_state_root"]),
        campaign_artifact_root=identity_plan["campaign_artifact_root"],
    )
    backend = base.GcpBackend(
        pexec=pexec,
        packet_root=packet,
        campaign_root=campaign_root,
        evaluation_registry=evaluation_registry,
    )
    backend.private = (
        Path(identity_plan["controller_private_root"])
        / identity_plan["roster_tag"]
        / "c1-serial"
    )
    backend.private.mkdir(parents=True, exist_ok=True)
    operations._install_operator_kill_lifecycle_patch(backend)
    operations._install_transient_provider_generation_monitor(
        backend=backend, campaign_root=campaign_root
    )
    operations._install_launch_census_and_direct_fallback_patch(
        base=base, backend=backend, campaign_root=campaign_root
    )
    runtime = SerialRuntime(
        pexec=pexec,
        roster=roster,
        parallel_plan=plan,
        scientific=scientific,
        bound=bound,
        registry=registry,
        campaign_root=campaign_root,
        backend=backend,
        compatibility_authorization=compatibility_authorization,
    )
    ceiling = float(identity_plan["hard_ceiling_usd"])
    audit_stage = str(identity_plan["audit_stage"])
    spend_ledger = operations._load_spend_ledger(
        args.stage_spend_ledger, audit_stage, ceiling
    )
    if float(spend_ledger["pre_science_aborted_launch_spend_usd"]) > 40.0:
        raise operations.AbortBurnStop("pre-existing abort-burn ledger exceeds $40")
    initial = operations._global_a100_census(backend)
    if (
        int(initial["total_attached_a100_equivalent"]) + 1
        > operations.GLOBAL_A100_CEILING
        or runtime._assert_capacity_census()["campaign_owned_vm_count"] != 0
    ):
        raise SerialControllerError(
            "serial launch lacks one A100 of global headroom or has campaign residue"
        )
    backend.note(
        f"SERIAL {identity_plan['stage_code'].upper()} LAUNCH AUTHORIZED: "
        f"{len(scientific['cells'])} frozen cells, width 1, Spot 1g only, "
        f"stage spend ${float(spend_ledger['estimated_spend_usd']):.6f}/"
        f"${ceiling:.2f}, abort burn "
        f"${float(spend_ledger['pre_science_aborted_launch_spend_usd']):.6f}/$40; "
        "parallel executor not instantiated, losses SEALED/BLINDED."
    )

    by_id = {str(row["cell_id"]): row for row in scientific["cells"]}
    roster_by_id = {
        str(row["cell_id"]): row for row in roster["launch_cells"]
    }
    divergence_loss_cap = pexec.multiseed_analysis_loss_cap(
        str(identity_plan["stage_code"]), bound
    )
    if divergence_loss_cap is None:
        raise SerialControllerError("serial audit stage lacks its divergence-loss cap")
    block_by_cell: dict[str, int] = {}
    block_terminal_cells: set[str] = set()
    for block in serial_binding["blocks"]:
        for cell in block["cells"]:
            block_by_cell[cell["cell_id"]] = int(block["block_order_index"])
        block_terminal_cells.add(str(block["cells"][-1]["cell_id"]))
    attempts_by_cell: dict[str, list[dict[str, Any]]] = {
        cell_id: [] for cell_id in serial_binding["cells_in_dispatch_order"]
    }
    resolved: set[str] = set()
    active = None
    lifecycle_lock = threading.RLock()
    watchdog: operations.CostWatchdog | None = None
    controller_error: BaseException | None = None
    completed = False
    checkpoint_preseal_path: Path | None = None
    prediction_freeze_path: Path | None = None
    hidden_authorization_path: Path | None = None
    hidden_root: Path | None = None
    try:
        watchdog = operations.CostWatchdog(
            executor=runtime,
            backend=backend,
            campaign_root=campaign_root,
            stage_ledger=spend_ledger,
            ceiling=ceiling,
            lifecycle_lock=lifecycle_lock,
        )
        watchdog.start()
        for cell_order_index, cell_id in enumerate(
            serial_binding["cells_in_dispatch_order"]
        ):
            cell = by_id[cell_id]
            block_order_index = block_by_cell[cell_id]
            while cell_id not in resolved:
                watchdog.raise_if_triggered()
                if active is None:
                    active = _provision_one(
                        base=base,
                        p1=p1,
                        pexec=pexec,
                        runtime=runtime,
                        registry=registry,
                        backend=backend,
                        campaign_root=campaign_root,
                        stage_ledger=spend_ledger,
                        scientific=scientific,
                        serial_binding=serial_binding,
                        resolved_ids=resolved,
                        ceiling=ceiling,
                    )
                provider = runtime.providers[(SLOT, active.generation)]
                prior_rows = attempts_by_cell[cell_id]
                prior = prior_rows[-1] if prior_rows else None
                attempt = len(prior_rows) + 1
                request = _request_for_cell(
                    pexec=pexec,
                    identity=active,
                    provider=provider,
                    cell=cell,
                    cell_order_index=cell_order_index,
                    block_order_index=block_order_index,
                    attempt=attempt,
                    prior=prior,
                    serial_binding=serial_binding,
                )
                dispatched = backend.dispatch(active, request)
                backend.note(
                    f"SERIAL DISPATCH cell {cell_order_index + 1}/"
                    f"{len(scientific['cells'])} block {block_order_index}: "
                    f"{cell_id} attempt {attempt} -> {active.slot}/g"
                    f"{active.generation}; loss SEALED/BLINDED."
                )
                outcome = dict(backend.collect(active, request))
                row = _attempt_row(
                    pexec=pexec,
                    runtime=runtime,
                    identity=active,
                    provider=provider,
                    cell=cell,
                    request=request,
                    outcome=outcome,
                    dispatched_at=dispatched,
                    cell_order_index=cell_order_index,
                    block_order_index=block_order_index,
                )
                work_report = _validate_attempt_work(
                    pexec=pexec,
                    stage_code=str(identity_plan["stage_code"]),
                    row=row,
                    scientific_cell=cell,
                    roster_cell=roster_by_id[cell_id],
                    campaign_root=campaign_root,
                    evaluation_registry=evaluation_registry,
                    divergence_loss_cap=float(divergence_loss_cap),
                )
                runtime.partials[(SLOT, active.generation)].append_attempt(row)
                attempts_by_cell[cell_id].append(row)
                _ratchet_sync(
                    backend=backend,
                    runtime=runtime,
                    identity=active,
                    row=row,
                    serial_binding=serial_binding,
                    work_report=work_report,
                )
                if row["status"] == "INFRA_FAILURE":
                    operations._finalize_identity(
                        executor=runtime,
                        identity=active,
                        preempted=True,
                        lifecycle_lock=lifecycle_lock,
                    )
                    active = None
                    continue
                if row["status"] not in {"COMPLETED", "DIVERGED"}:
                    raise SerialControllerError(
                        f"cell {cell_id} returned nonterminal status {row['status']}"
                    )
                resolved.add(cell_id)
                print(
                    json.dumps(
                        {
                            "event": "SERIAL_CELL_BANKED",
                            "cell_index": cell_order_index,
                            "cell_count": len(scientific["cells"]),
                            "cell_id": cell_id,
                            "attempt": attempt,
                            "status": row["status"],
                            "development_loss": "SEALED/BLINDED",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if cell_id in block_terminal_cells:
                    operations._finalize_identity(
                        executor=runtime,
                        identity=active,
                        preempted=False,
                        lifecycle_lock=lifecycle_lock,
                    )
                    backend.note(
                        f"SERIAL BLOCK {block_order_index} COMPLETE: all registered "
                        "cells are durably ratcheted and the exact generation has "
                        "instance/disk NOT_FOUND proofs before the next block."
                    )
                    active = None
        if active is not None:
            operations._finalize_identity(
                executor=runtime,
                identity=active,
                preempted=False,
                lifecycle_lock=lifecycle_lock,
            )
            active = None
        watchdog.raise_if_triggered()

        evaluation_role = operations._deferred_evaluation_role(scientific)
        if evaluation_role is not None:
            training_transient_registry = (
                operations._transient_provider_lifecycle_registry(campaign_root)
            )
            checkpoint_value = serial.build_checkpoint_preseal(
                stage_code=str(identity_plan["stage_code"]),
                parent=parent,
                bound=bound,
                scientific=scientific,
                roster=roster,
                parallel_plan=plan,
                serial_binding=serial_binding,
                serial_authorization=serial_authorization,
                compatibility_runtime_authorization=compatibility_authorization,
                vm_registry=registry.snapshot(),
                transient_provider_registry=training_transient_registry,
                evaluation_registry=evaluation_registry,
                campaign_attempt=int(identity_plan["campaign_attempt"]),
                campaign_root=campaign_root,
            )
            checkpoint_preseal_path = (
                campaign_root / "campaign" / "checkpoint-preseal.json"
            )
            serial.write_json_create_only(checkpoint_preseal_path, checkpoint_value)
            if evaluation_role == "development_prediction_endpoint":
                prediction_freeze_path = operations._a3_prediction_freeze(
                    args=args,
                    checkpoint_preseal=checkpoint_preseal_path,
                    bound_manifest=packet / "materialized" / "bound-manifest.json",
                    campaign_root=campaign_root,
                )
            hidden_authorization_path = operations._hidden_authorization(
                checkpoint_preseal=checkpoint_preseal_path,
                bound_manifest=packet / "materialized" / "bound-manifest.json",
                campaign_root=campaign_root,
                evaluation_role=evaluation_role,
                prediction_freeze=prediction_freeze_path,
            )
            batch_attempt = 1
            while True:
                watchdog.raise_if_triggered()
                hidden_identity = _provision_one(
                    base=base,
                    p1=p1,
                    pexec=pexec,
                    runtime=runtime,
                    registry=registry,
                    backend=backend,
                    campaign_root=campaign_root,
                    stage_ledger=spend_ledger,
                    scientific=scientific,
                    serial_binding=serial_binding,
                    resolved_ids=resolved,
                    ceiling=ceiling,
                )
                try:
                    hidden_root = operations._execute_hidden_on_survivor(
                        executor=runtime,
                        backend=backend,
                        identity=hidden_identity,
                        campaign_root=campaign_root,
                        science_root=str(identity_plan["science_root"]),
                        evaluation_role=evaluation_role,
                        first_seed=min(int(seed) for seed in seed_registry["seeds"]),
                        authorization=hidden_authorization_path,
                        checkpoint_preseal=checkpoint_preseal_path,
                        bound_manifest=packet / "materialized" / "bound-manifest.json",
                        batch_attempt=batch_attempt,
                        prediction_freeze=prediction_freeze_path,
                    )
                except operations.HiddenSurvivorPreempted:
                    operations._finalize_identity(
                        executor=runtime,
                        identity=hidden_identity,
                        preempted=True,
                        lifecycle_lock=lifecycle_lock,
                    )
                    batch_attempt += 1
                    continue
                operations._finalize_identity(
                    executor=runtime,
                    identity=hidden_identity,
                    preempted=False,
                    lifecycle_lock=lifecycle_lock,
                )
                break
        completed = True
    except BaseException as exc:
        controller_error = exc
    finally:
        if watchdog is not None:
            watchdog.stop()
            try:
                watchdog.raise_if_triggered()
            except BaseException as exc:
                if controller_error is None:
                    controller_error = exc
        for identity in list(runtime.active.values()):
            try:
                operations._finalize_identity(
                    executor=runtime,
                    identity=identity,
                    preempted=(backend.describe_instance(identity.run_id, check=False) is None),
                    lifecycle_lock=lifecycle_lock,
                )
            except BaseException as exc:
                if controller_error is None:
                    controller_error = exc
        completed = completed and controller_error is None

        census = _final_census(runtime)
        current_cost, generation_costs = operations._current_campaign_cost(
            runtime, campaign_root
        )
        scientific_started = operations._campaign_scientific_attempt_started(
            campaign_root
        )
        prior_abort = float(spend_ledger["pre_science_aborted_launch_spend_usd"])
        updated_abort = (
            prior_abort
            if scientific_started or completed
            else prior_abort + current_cost
        )
        transient_registry = operations._transient_provider_lifecycle_registry(
            campaign_root
        )
        transient_path = campaign_root / "campaign" / "transient-provider-registry.json"
        _write_atomic(transient_path, transient_registry)
        campaign_cost = {
            "schema": "audit_135m_campaign_cost_v1",
            "execution_mode": "serial_single_vm_width_1",
            "stage_code": identity_plan["stage_code"],
            "roster_hash": identity_plan["roster_hash"],
            "completed": completed,
            "scientific_attempt_started": scientific_started,
            "counts_toward_pre_science_abort_burn": (
                not scientific_started and not completed
            ),
            "estimated_cost_usd": round(current_cost, 6),
            "generations": generation_costs,
            "transient_provider_registry": {
                "path": transient_path.relative_to(campaign_root).as_posix(),
                "raw_sha256": serial.sha256_file(transient_path),
                "count": int(transient_registry["count"]),
            },
            "final_zero_census": census,
        }
        updated_ledger = {
            **spend_ledger,
            "abort_burn_kill_usd": 40.0,
            "pre_science_aborted_launch_spend_usd": round(updated_abort, 6),
            "abort_burn_kill_exceeded": updated_abort > 40.0,
            "estimated_spend_usd": round(
                float(spend_ledger["estimated_spend_usd"]) + current_cost, 6
            ),
            "campaigns": [*spend_ledger["campaigns"], campaign_cost],
            "updated_at_utc": operations.utc_now(),
        }
        _write_atomic(args.stage_spend_ledger, updated_ledger)
        _write_atomic(campaign_root / "campaign" / "campaign-cost-final.json", campaign_cost)
        _write_atomic(campaign_root / "campaign" / "final-provider-census.json", census)
        vm_registry = registry.snapshot()
        _write_atomic(campaign_root / "campaign" / "vm-registry.json", vm_registry)
        _write_atomic(
            campaign_root / "campaign" / "evaluation-registry.json",
            evaluation_registry,
        )
        descriptor = {
            "schema": "audit_135m_serial_aggregation_descriptor_v1",
            "aggregation_authorized": False,
            "stage_code": identity_plan["stage_code"],
            "campaign_attempt": int(identity_plan["campaign_attempt"]),
            "campaign_root": str(campaign_root),
            "parent_manifest": str(packet / "parent" / "parent-manifest.json"),
            "bound_manifest": str(packet / "materialized" / "bound-manifest.json"),
            "scientific_plan": str(
                packet / "materialized" / "scientific-randomization-plan.json"
            ),
            "parallel_roster": str(packet / "binding" / "parallel-roster.json"),
            "parallel_plan": str(packet / "binding" / "parallel-plan.json"),
            "serial_binding": str(packet / "binding" / "serial-binding.json"),
            "serial_runtime_authorization": str(
                packet / "serial-runtime-authorization.json"
            ),
            "runtime_authorization": str(packet / "runtime-authorization.json"),
            "vm_registry": str(campaign_root / "campaign" / "vm-registry.json"),
            "transient_provider_registry": str(transient_path),
            "evaluation_registry": str(
                campaign_root / "campaign" / "evaluation-registry.json"
            ),
            "final_provider_census": str(
                campaign_root / "campaign" / "final-provider-census.json"
            ),
        }
        descriptor_path = campaign_root / "campaign" / "aggregation-descriptor.json"
        _write_atomic(descriptor_path, descriptor)
        if completed:
            manifest, seal = serial.aggregate(
                stage_code=str(identity_plan["stage_code"]),
                parent=parent,
                bound=bound,
                scientific=scientific,
                roster=roster,
                parallel_plan=plan,
                serial_binding=serial_binding,
                serial_authorization=serial_authorization,
                compatibility_runtime_authorization=compatibility_authorization,
                vm_registry=vm_registry,
                transient_provider_registry=transient_registry,
                evaluation_registry=evaluation_registry,
                final_provider_census=census,
                campaign_attempt=int(identity_plan["campaign_attempt"]),
                campaign_root=campaign_root,
            )
            manifest_path = campaign_root / "campaign" / "campaign-manifest.json"
            seal_path = campaign_root / "campaign" / "campaign-seal.json"
            if checkpoint_preseal_path is not None:
                preseal = serial.load_object(checkpoint_preseal_path, "checkpoint preseal")
                if (
                    manifest.get("attempts") != preseal.get("attempts")
                    or manifest.get("audit_checkpoint_registry")
                    != preseal.get("audit_checkpoint_registry")
                ):
                    completed = False
                    controller_error = SerialControllerError(
                        "serial final checkpoint registry differs from preseal"
                    )
            if completed:
                serial.write_json_create_only(manifest_path, manifest)
                serial.write_json_create_only(seal_path, seal)
                descriptor.update(
                    {
                        "aggregation_authorized": True,
                        "campaign_manifest": str(manifest_path),
                        "campaign_seal": str(seal_path),
                    }
                )
                _write_atomic(descriptor_path, descriptor)
        result = {
            "schema": "audit_135m_serial_controller_result_v1",
            "status": (
                "SEALED_EXECUTION_AND_EXACT_ID_TEARDOWN_COMPLETE"
                if completed
                else "EXECUTION_ABORTED_AND_EXACT_ID_TEARDOWN_COMPLETE"
            ),
            "execution_mode": "serial_single_vm_width_1",
            "parallel_executor_used": False,
            "stage_code": identity_plan["stage_code"],
            "execution_complete": completed,
            "aggregation_authorized": completed,
            "controller_error_type": (
                None if controller_error is None else type(controller_error).__name__
            ),
            "controller_error_detail": (
                None if controller_error is None else str(controller_error)
            ),
            "resolved_cell_count": len(resolved),
            "launch_cell_count": len(scientific["cells"]),
            "estimated_campaign_cost_usd": campaign_cost["estimated_cost_usd"],
            "estimated_stage_spend_usd": updated_ledger["estimated_spend_usd"],
            "hard_ceiling_usd": ceiling,
            "pre_science_aborted_launch_spend_usd": updated_ledger[
                "pre_science_aborted_launch_spend_usd"
            ],
            "abort_burn_kill_usd": 40.0,
            "final_zero_census": census,
            "checkpoint_preseal": (
                None if checkpoint_preseal_path is None else str(checkpoint_preseal_path)
            ),
            "prediction_freeze": (
                None if prediction_freeze_path is None else str(prediction_freeze_path)
            ),
            "hidden_authorization": (
                None if hidden_authorization_path is None else str(hidden_authorization_path)
            ),
            "hidden_root": None if hidden_root is None else str(hidden_root),
        }
        _write_atomic(campaign_root / "campaign" / "controller-result.json", result)
        _upload_campaign(
            backend, campaign_root, str(identity_plan["campaign_artifact_root"])
        )
    if controller_error is not None:
        raise controller_error
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--stage-spend-ledger", type=Path, required=True)
    parser.add_argument("--a3-mechanical-gate", type=Path)
    parser.add_argument("--a3-historical-phase-manifest", type=Path)
    parser.add_argument("--a3-recapture-campaign-manifest", type=Path)
    parser.add_argument("--a3-recapture-bound-manifest", type=Path)
    parser.add_argument("--a3-recapture-campaign-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (
        OSError,
        KeyError,
        ValueError,
        SerialControllerError,
        serial.SerialAuditError,
        operations.ControllerError,
    ) as exc:
        print(f"audit serial controller error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
