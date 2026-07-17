#!/usr/bin/env python3
"""Serial execution/evidence helpers for the triggered 135M audit fallback.

The scientific commands and work validators remain those frozen by
``audit_135m_contract`` and ``run_phase_map``.  This module replaces only the
cross-VM scheduler: one exact Spot 1g generation is active at a time, cells are
dispatched in their already-materialized block/arm order, and every completed
cell is permanently banked before the next dispatch.

Compatibility roster/plan bytes are retained solely because the reviewed work
evidence validator uses their cell projection.  They are not launch authority,
and this module never instantiates or calls ``ParallelWaveExecutor``.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import audit_135m_contract as audit
from scripts import run_parallel_phase_map as evidence


REPO_ROOT = Path(__file__).resolve().parents[1]
SERIAL_AMENDMENT_PATH = (
    REPO_ROOT / "docs" / "AMENDMENT-audit-135m-serial-fallback.md"
)
SERIAL_BINDING_SCHEMA = "audit_135m_serial_binding_v1"
SERIAL_AUTH_SCHEMA = "audit_135m_serial_runtime_authorization_v1"
SERIAL_MANIFEST_SCHEMA = "audit_135m_serial_campaign_manifest_v1"
SERIAL_SEAL_SCHEMA = "audit_135m_serial_campaign_seal_v1"
SERIAL_HELPER_KEYS = frozenset(
    {"p1_capacity_controller", "gcp_backend_controller"}
)


class SerialAuditError(RuntimeError):
    """Serial binding, schedule, or evidence is not exact."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SerialAuditError(f"{label} is not a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SerialAuditError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise SerialAuditError(f"{label} lacks timezone information")
    return parsed.astimezone(timezone.utc)


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SerialAuditError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SerialAuditError(f"{label} must be a JSON object")
    return value


def write_json_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise SerialAuditError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SerialAuditError(f"{label} must be an object")
    return dict(value)


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SerialAuditError(f"{label} must be an array")
    return value


def normalize_helper_hashes(value: Any) -> dict[str, str]:
    helpers = _mapping(value, "serial reviewed helper hashes")
    if set(helpers) != SERIAL_HELPER_KEYS:
        raise SerialAuditError("serial reviewed helper hash set differs")
    normalized: dict[str, str] = {}
    for key in sorted(helpers):
        digest = helpers[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SerialAuditError(f"serial reviewed helper hash {key} is not SHA-256")
        normalized[key] = digest
    return normalized


def _cell_map(scientific: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in _array(scientific.get("cells"), "scientific cells"):
        row = _mapping(raw, "scientific cell")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in result:
            raise SerialAuditError("scientific cell IDs are missing or duplicated")
        result[cell_id] = row
    if not result:
        raise SerialAuditError("scientific plan has no launch cells")
    return result


def build_serial_binding(
    *,
    stage_code: str,
    parent: Mapping[str, Any],
    bound: Mapping[str, Any],
    scientific: Mapping[str, Any],
    compatibility_roster: Mapping[str, Any],
    compatibility_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the existing randomized cells to a one-VM dispatch order."""

    if stage_code not in audit.STAGE_CODES:
        raise SerialAuditError(f"unsupported serial audit stage {stage_code!r}")
    if scientific.get("stage_code") != stage_code:
        raise SerialAuditError("scientific stage code differs")
    if bound.get("study_id") != scientific.get("study_id"):
        raise SerialAuditError("bound/scientific study identity differs")
    cells = _cell_map(scientific)
    roster_cells = {
        str(row.get("cell_id")): row
        for row in _array(
            compatibility_roster.get("launch_cells"),
            "compatibility launch roster",
        )
        if isinstance(row, Mapping)
    }
    if set(roster_cells) != set(cells):
        raise SerialAuditError("compatibility roster does not cover the scientific suffix")
    if evidence.roster_hash(compatibility_roster) != compatibility_plan.get(
        "roster_hash"
    ):
        raise SerialAuditError("compatibility plan cites a different roster")
    compatibility_plan_hash = evidence.parallel_plan_hash(compatibility_plan)

    ordered = sorted(
        cells.values(),
        key=lambda row: (
            int(_mapping(row.get("randomization"), "cell randomization")["block_order_index"]),
            int(_mapping(row.get("randomization"), "cell randomization")["within_block_index"]),
        ),
    )
    blocks: list[dict[str, Any]] = []
    seen_blocks: set[str] = set()
    for row in ordered:
        randomization = _mapping(row.get("randomization"), "cell randomization")
        block_id = str(randomization.get("block_id"))
        if not blocks or blocks[-1]["block_id"] != block_id:
            if block_id in seen_blocks:
                raise SerialAuditError("materialized block cells are not contiguous")
            seen_blocks.add(block_id)
            blocks.append(
                {
                    "block_id": block_id,
                    "block_order_index": int(randomization["block_order_index"]),
                    "H": int(row["H"]),
                    "M": int(row["M"]),
                    "seed": int(row["seed"]),
                    "training_seed": int(row["training_seed"]),
                    "pairing_identity_hash": row.get("pairing_identity_hash"),
                    "cells": [],
                }
            )
        block = blocks[-1]
        stable = {
            "H": int(row["H"]),
            "M": int(row["M"]),
            "seed": int(row["seed"]),
            "training_seed": int(row["training_seed"]),
            "pairing_identity_hash": row.get("pairing_identity_hash"),
        }
        if any(block[key] != value for key, value in stable.items()):
            raise SerialAuditError(f"block {block_id} changes its pairing identity")
        block["cells"].append(
            {
                "cell_id": str(row["cell_id"]),
                "within_block_index": int(randomization["within_block_index"]),
                "analysis_role": str(row["analysis_role"]),
                "pair_key": str(row["pair_key"]),
                "mu": float(row["mu"]),
                "eta": float(row["eta"]),
                "command_hash": str(row["command_hash"]),
            }
        )
    if [block["block_order_index"] for block in blocks] != list(range(len(blocks))):
        raise SerialAuditError("materialized block order is not contiguous from zero")
    for block in blocks:
        indices = [cell["within_block_index"] for cell in block["cells"]]
        if indices != list(range(len(indices))):
            raise SerialAuditError(
                f"block {block['block_id']} arm order is not contiguous from zero"
            )

    binding = {
        "schema": SERIAL_BINDING_SCHEMA,
        "status": "BOUND_NOT_LAUNCH_AUTHORITY",
        "execution_mode": "serial_single_vm_width_1",
        "stage_code": stage_code,
        "audit_stage": bound["audit_135m_contract"]["audit_stage"],
        "audit_phase": bound["audit_135m_contract"]["audit_phase"],
        "study_id": bound["study_id"],
        "serial_amendment_raw_sha256": sha256_file(SERIAL_AMENDMENT_PATH),
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific[
            "randomization_plan_hash"
        ],
        "compatibility_roster_hash": evidence.roster_hash(compatibility_roster),
        "compatibility_parallel_plan_hash": compatibility_plan_hash,
        "parallel_executor_authorized": False,
        "maximum_active_vms": 1,
        "maximum_active_a100_equivalent": 1,
        "machine_type": "a2-highgpu-1g",
        "spot_only": True,
        "completed_cell_ratchet": True,
        "mid_cell_preemption_retries_only_that_cell": True,
        "partial_state_resume_forbidden": True,
        "completed_cells_never_rerun": True,
        "development_loss_publication_before_stage_seal": False,
        "block_count": len(blocks),
        "cell_count": len(ordered),
        "blocks": blocks,
        "cells_in_dispatch_order": [str(row["cell_id"]) for row in ordered],
    }
    binding["serial_plan_hash"] = canonical_sha256(binding)
    return binding


def runtime_authorization(
    binding: Mapping[str, Any],
    *,
    hard_ceiling_usd: float,
    reviewed_helper_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if binding.get("schema") != SERIAL_BINDING_SCHEMA:
        raise SerialAuditError("serial runtime authorization requires a serial binding")
    value = {
        "schema": SERIAL_AUTH_SCHEMA,
        "launch_authorized": True,
        "execution_mode": binding["execution_mode"],
        "stage_code": binding["stage_code"],
        "study_id": binding["study_id"],
        "serial_amendment_raw_sha256": binding[
            "serial_amendment_raw_sha256"
        ],
        "serial_plan_hash": binding["serial_plan_hash"],
        "parent_manifest_canonical_sha256": binding[
            "parent_manifest_canonical_sha256"
        ],
        "bound_manifest_canonical_sha256": binding[
            "bound_manifest_canonical_sha256"
        ],
        "scientific_randomization_plan_hash": binding[
            "scientific_randomization_plan_hash"
        ],
        "compatibility_roster_hash": binding["compatibility_roster_hash"],
        "compatibility_parallel_plan_hash": binding[
            "compatibility_parallel_plan_hash"
        ],
        "parallel_executor_authorized": False,
        "maximum_active_vms": 1,
        "maximum_active_a100_equivalent": 1,
        "machine_type": "a2-highgpu-1g",
        "spot_only": True,
        "hard_ceiling_usd": float(hard_ceiling_usd),
        "abort_burn_kill_usd": 40.0,
        "max_idle_before_science_seconds": 600,
        "completed_cell_ratchet": True,
        "partial_state_resume_forbidden": True,
        "reviewed_helper_sha256": normalize_helper_hashes(
            reviewed_helper_sha256
        ),
    }
    value["authorization_canonical_sha256"] = canonical_sha256(value)
    return value


def verify_runtime_authorization(
    authorization: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    expected_hard_ceiling_usd: float | None = None,
) -> str:
    hard_ceiling = authorization.get("hard_ceiling_usd")
    if (
        isinstance(hard_ceiling, bool)
        or not isinstance(hard_ceiling, (int, float))
        or not math.isfinite(float(hard_ceiling))
        or float(hard_ceiling) <= 0.0
    ):
        raise SerialAuditError("serial runtime authorization ceiling is not finite")
    if (
        expected_hard_ceiling_usd is not None
        and float(hard_ceiling) != float(expected_hard_ceiling_usd)
    ):
        raise SerialAuditError("serial runtime authorization ceiling differs")
    preimage = dict(authorization)
    digest = preimage.pop("authorization_canonical_sha256", None)
    expected = runtime_authorization(
        binding,
        hard_ceiling_usd=float(hard_ceiling),
        reviewed_helper_sha256=normalize_helper_hashes(
            authorization.get("reviewed_helper_sha256")
        ),
    )
    if dict(authorization) != expected or digest != canonical_sha256(preimage):
        raise SerialAuditError("serial runtime authorization identity/hash differs")
    if authorization.get("launch_authorized") is not True:
        raise SerialAuditError("serial runtime authorization is not launch authority")
    return str(digest)


def _attempt_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("serial_cell_order_index", row.get("actual_wave_index", -1))),
        int(row.get("attempt", 0)),
        int(row.get("generation", 0)),
    )


def validate_serial_schedule(
    *,
    attempts: Sequence[Mapping[str, Any]],
    scientific: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate cell-only ratchets and return authoritative terminal attempts."""

    if (
        binding.get("schema") != SERIAL_BINDING_SCHEMA
        or binding.get("execution_mode") != "serial_single_vm_width_1"
        or binding.get("parallel_executor_authorized") is not False
        or binding.get("maximum_active_vms") != 1
        or binding.get("maximum_active_a100_equivalent") != 1
        or binding.get("machine_type") != "a2-highgpu-1g"
        or binding.get("spot_only") is not True
    ):
        raise SerialAuditError("serial schedule binding is not width-one Spot authority")
    cells = _cell_map(scientific)
    order = list(_array(binding.get("cells_in_dispatch_order"), "serial cell order"))
    if order != [cell_id for cell_id in order if isinstance(cell_id, str)]:
        raise SerialAuditError("serial cell order contains a non-string identity")
    if set(order) != set(cells) or len(order) != len(cells):
        raise SerialAuditError("serial cell order does not cover the exact suffix")
    order_index = {cell_id: index for index, cell_id in enumerate(order)}
    rows = [dict(row) for row in attempts]
    if rows != sorted(rows, key=_attempt_sort_key):
        raise SerialAuditError("serial attempt registry is reordered")
    by_cell: dict[str, list[dict[str, Any]]] = {cell_id: [] for cell_id in order}
    intervals: list[tuple[datetime, datetime, str]] = []
    seen_attempt_ids: set[str] = set()
    seen_attempt_prefixes: set[str] = set()
    generation_blocks: dict[tuple[str, int], str] = {}
    generation_last_end: dict[tuple[str, int], datetime] = {}
    next_cell_index = 0
    for row in rows:
        cell_id = str(row.get("cell_id"))
        if cell_id not in cells:
            raise SerialAuditError("serial attempt references an unknown cell")
        if next_cell_index >= len(order) or cell_id != order[next_cell_index]:
            raise SerialAuditError(
                "serial attempt ran a later cell before the current cell resolved"
            )
        if row.get("group_id") != cells[cell_id]["randomization"]["block_id"]:
            raise SerialAuditError("serial attempt changed its materialized block")
        if int(row.get("serial_cell_order_index", -1)) != order_index[cell_id]:
            raise SerialAuditError("serial attempt changed its cell order index")
        if (
            row.get("logical_slot") != "v0"
            or row.get("machine_type") != "a2-highgpu-1g"
            or row.get("gpu_slots") != 1
            or set(_mapping(row.get("learner_gpu_slot_map"), "learner GPU map").values())
            != {0}
        ):
            raise SerialAuditError("serial attempt escaped the one-slot 1g contract")
        if (
            row.get("concurrent_batch_slot_set") != ["v0"]
            or row.get("available_slot_set") != ["v0"]
            or row.get("concurrent_batch_index") != 0
            or row.get("dispatch_batch_index") != 0
            or row.get("batch_launch_order_index") != 0
        ):
            raise SerialAuditError("serial attempt claims a concurrent slot set")
        attempt = len(by_cell[cell_id]) + 1
        if int(row.get("attempt", 0)) != attempt or int(
            row.get("retry_round", 0)
        ) != attempt:
            raise SerialAuditError("serial cell attempt numbering is not contiguous")
        expected_attempt_id = f"{cell_id}-attempt-{attempt}"
        attempt_id = str(row.get("attempt_id"))
        attempt_prefix = str(row.get("attempt_prefix"))
        if (
            attempt_id != expected_attempt_id
            or attempt_id in seen_attempt_ids
            or attempt_prefix in seen_attempt_prefixes
            or not attempt_prefix.endswith(f"cells/{cell_id}/attempt-{attempt}/")
        ):
            raise SerialAuditError("serial attempt identity/prefix is reused or malformed")
        seen_attempt_ids.add(attempt_id)
        seen_attempt_prefixes.add(attempt_prefix)
        if (
            row.get("frozen_command_hash") != cells[cell_id].get("command_hash")
            or row.get("pairing_identity_hash")
            != cells[cell_id].get("pairing_identity_hash")
        ):
            raise SerialAuditError("serial attempt changed its frozen scientific identity")
        prior = by_cell[cell_id][-1] if by_cell[cell_id] else None
        try:
            evidence._validate_fresh_attempt(row, prior)
        except evidence.ScheduleError as exc:
            raise SerialAuditError(str(exc)) from exc
        dispatched = parse_time(row.get("dispatched_at"), "dispatch time")
        ready = parse_time(row.get("vm_ready_at"), "VM READY time")
        start = parse_time(row.get("scientific_started_at"), "scientific start")
        end = parse_time(row.get("scientific_ended_at"), "scientific end")
        recorded = parse_time(
            row.get("wave_terminal_prefix_sealed_at"),
            "serial terminal-record time",
        )
        if not ready <= dispatched <= start <= end <= recorded:
            raise SerialAuditError("serial attempt chronology is inconsistent")
        generation_key = (str(row.get("run_id")), int(row.get("generation", 0)))
        if generation_key[1] <= 0:
            raise SerialAuditError("serial attempt has an invalid physical generation")
        block_id = str(row.get("group_id"))
        prior_block = generation_blocks.setdefault(generation_key, block_id)
        if prior_block != block_id:
            raise SerialAuditError("one serial VM generation crossed a block boundary")
        prior_generation_end = generation_last_end.get(generation_key)
        if prior_generation_end is None:
            if (dispatched - ready).total_seconds() >= 600:
                raise SerialAuditError("serial READY VM idled for at least 600 seconds")
        elif (
            dispatched < prior_generation_end
            or (dispatched - prior_generation_end).total_seconds() >= 600
        ):
            raise SerialAuditError("serial VM idled or overlapped between cells")
        generation_last_end[generation_key] = end
        if prior is None:
            if any(
                row.get(field) is not None
                for field in ("retry_of", "retry_reason", "retry_authorization")
            ):
                raise SerialAuditError("first serial attempt carries retry lineage")
        else:
            if prior.get("status") != "INFRA_FAILURE":
                raise SerialAuditError("serial retry followed terminal scientific work")
            if int(row.get("generation", 0)) <= int(prior.get("generation", 0)):
                raise SerialAuditError("serial retry did not use a fresh physical generation")
            authorization = _mapping(
                row.get("retry_authorization"), "serial retry authorization"
            )
            expected_authorization = {
                "schema": "audit_135m_serial_cell_retry_authorization_v1",
                "loss_blind": True,
                "serial_plan_hash": binding["serial_plan_hash"],
                "cell_id": cell_id,
                "retry_attempt": attempt,
                "trigger_attempt_id": prior["attempt_id"],
                "trigger_reason": prior["failure_reason"],
            }
            if any(
                authorization.get(key) != value
                for key, value in expected_authorization.items()
            ):
                raise SerialAuditError("serial cell retry authorization differs")
            authorized_at = parse_time(
                authorization.get("authorized_at_utc"),
                "serial retry authorization time",
            )
            prior_recorded = parse_time(
                prior.get("wave_terminal_prefix_sealed_at"),
                "prior terminal-record time",
            )
            if not prior_recorded <= authorized_at <= dispatched:
                raise SerialAuditError(
                    "serial retry authorization is outside the sealed retry window"
                )
            if (
                row.get("retry_of") != prior.get("attempt_id")
                or row.get("retry_reason") != prior.get("failure_reason")
            ):
                raise SerialAuditError("serial cell retry lineage differs")
        intervals.append((start, end, str(row.get("attempt_id"))))
        by_cell[cell_id].append(row)
        status = row.get("status")
        if status == "INFRA_FAILURE":
            if (
                row.get("failure_reason")
                not in evidence.DIRECT_INFRASTRUCTURE_FAILURE_REASONS
            ):
                raise SerialAuditError(
                    "serial retry cites a non-direct infrastructure reason"
                )
        elif status in {"COMPLETED", "DIVERGED"}:
            next_cell_index += 1
        else:
            raise SerialAuditError("serial schedule contains a nonterminal/failed row")
    intervals.sort(key=lambda item: item[0])
    for prior, current in zip(intervals, intervals[1:]):
        if current[0] < prior[1]:
            raise SerialAuditError(
                f"serial attempts overlap: {prior[2]} and {current[2]}"
            )

    if next_cell_index != len(order):
        raise SerialAuditError("serial schedule does not resolve the complete suffix")

    analysis: dict[str, dict[str, Any]] = {}
    for cell_id in order:
        cell_rows = by_cell[cell_id]
        if not cell_rows:
            raise SerialAuditError(f"serial cell {cell_id} has no retained attempt")
        terminal = cell_rows[-1]
        if terminal.get("status") not in {"COMPLETED", "DIVERGED"}:
            raise SerialAuditError(f"serial cell {cell_id} is unresolved")
        analysis[cell_id] = {
            "attempt_id": terminal["attempt_id"],
            "attempt": int(terminal["attempt"]),
            "status": terminal["status"],
            "group_id": terminal["group_id"],
            "retry_round": int(terminal["attempt"]),
            "analysis_loss": terminal.get("analysis_loss"),
            "analysis_loss_kind": terminal.get("analysis_loss_kind"),
            "divergence_retained": terminal.get("divergence_retained"),
        }
    return analysis


def validate_serial_generation_lifecycles(
    lifecycles: Sequence[Mapping[str, Any]],
) -> None:
    intervals: list[tuple[datetime, datetime]] = []
    for row in lifecycles:
        if row.get("machine_type") != "a2-highgpu-1g" or row.get(
            "a100_count"
        ) != 1:
            raise SerialAuditError("serial lifecycle escaped the one-VM 1g contract")
        start = evidence.parse_time(
            row.get("creation_timestamp"), "generation creation"
        )
        end = evidence.parse_time(
            row.get("deletion_completed_at_utc"), "generation deletion"
        )
        if end < start:
            raise SerialAuditError("serial generation deletion precedes creation")
        intervals.append((start, end))
    intervals.sort()
    for prior, current in zip(intervals, intervals[1:]):
        if current[0] < prior[1]:
            raise SerialAuditError("serial physical generations overlap")


def validate_ratchet_receipts(
    *,
    attempts: Sequence[Mapping[str, Any]],
    work_reports: Sequence[Mapping[str, Any]],
    binding: Mapping[str, Any],
    campaign_root: Path,
) -> list[dict[str, Any]]:
    ratchet_root = campaign_root / "campaign" / "ratchet"
    if not ratchet_root.is_dir():
        raise SerialAuditError("serial campaign lacks its durable ratchet directory")
    expected_paths: set[Path] = set()
    result: list[dict[str, Any]] = []
    ack_times: dict[str, datetime] = {}
    reports_by_attempt = {
        str(report.get("attempt_id")): dict(report) for report in work_reports
    }
    if len(reports_by_attempt) != len(work_reports):
        raise SerialAuditError("serial work-evidence reports repeat an attempt")
    if set(reports_by_attempt) != {str(row.get("attempt_id")) for row in attempts}:
        raise SerialAuditError("serial work-evidence reports do not cover every attempt")
    for row in attempts:
        stem = (
            f"{int(row['serial_cell_order_index']):04d}"
            f"-a{int(row['attempt']):02d}"
        )
        receipt_path = ratchet_root / f"{stem}.json"
        ack_path = ratchet_root / f"{stem}.ack.json"
        expected_paths.update({receipt_path, ack_path})
        receipt = load_object(receipt_path, "serial ratchet receipt")
        ack = load_object(ack_path, "serial ratchet acknowledgement")
        receipt_hash = sha256_file(receipt_path)
        expected_receipt = {
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
            "attempt_canonical_sha256": canonical_sha256(row),
            "artifact_inventory_canonical_sha256": canonical_sha256(
                row.get("artifact_inventory")
            ),
            "work_evidence_validated": True,
            "work_evidence_report_canonical_sha256": canonical_sha256(
                reports_by_attempt.get(str(row["attempt_id"]))
            ),
        }
        if any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise SerialAuditError("serial ratchet receipt differs from its attempt")
        created = parse_time(
            receipt.get("created_at_utc"), "serial ratchet receipt creation"
        )
        if created < parse_time(
            row.get("wave_terminal_prefix_sealed_at"),
            "serial terminal-record time",
        ):
            raise SerialAuditError("serial ratchet receipt predates terminal recording")
        remote_uri = receipt.get("remote_uri")
        if (
            not isinstance(remote_uri, str)
            or not remote_uri.startswith("gs://")
            or not remote_uri.endswith(f"/ratchet/{receipt_path.name}")
        ):
            raise SerialAuditError("serial ratchet receipt URI is malformed")
        expected_ack = {
            "schema": "audit_135m_serial_attempt_ratchet_ack_v1",
            "status": "GCS_CREATE_ONLY_ROUNDTRIP_PASS",
            "loss_exposed": False,
            "serial_plan_hash": binding["serial_plan_hash"],
            "cell_id": row["cell_id"],
            "attempt_id": row["attempt_id"],
            "receipt_path": receipt_path.relative_to(campaign_root).as_posix(),
            "receipt_raw_sha256": receipt_hash,
            "receipt_remote_uri": remote_uri,
            "gcs_create_only": True,
            "gcs_roundtrip_verified": True,
        }
        if any(ack.get(key) != value for key, value in expected_ack.items()):
            raise SerialAuditError("serial ratchet acknowledgement differs")
        verified = parse_time(
            ack.get("verified_at_utc"), "serial ratchet acknowledgement time"
        )
        if verified < created:
            raise SerialAuditError("serial ratchet acknowledgement predates its receipt")
        ack_uri = ack.get("remote_uri")
        if (
            not isinstance(ack_uri, str)
            or not ack_uri.startswith("gs://")
            or not ack_uri.endswith(f"/ratchet/{ack_path.name}")
        ):
            raise SerialAuditError("serial ratchet acknowledgement URI is malformed")
        ack_times[str(row["attempt_id"])] = verified
        result.append(
            {
                "cell_id": row["cell_id"],
                "attempt_id": row["attempt_id"],
                "receipt_raw_sha256": receipt_hash,
                "ack_raw_sha256": sha256_file(ack_path),
            }
        )
    actual_paths = {
        path
        for path in ratchet_root.iterdir()
        if path.is_file() and path.suffix == ".json"
    }
    if actual_paths != expected_paths:
        raise SerialAuditError("serial ratchet directory has missing or extra receipts")
    for row in attempts:
        authorization = row.get("retry_authorization")
        if not isinstance(authorization, Mapping):
            continue
        prior_id = str(row.get("retry_of"))
        authorized = parse_time(
            authorization.get("authorized_at_utc"),
            "serial retry authorization time",
        )
        if prior_id not in ack_times or authorized < ack_times[prior_id]:
            raise SerialAuditError("serial retry preceded the prior durable GCS ratchet")
    return result


def _compatibility_aggregator(
    *,
    stage_code: str,
    parent: Mapping[str, Any],
    bound: Mapping[str, Any],
    scientific: Mapping[str, Any],
    roster: Mapping[str, Any],
    parallel_plan: Mapping[str, Any],
    vm_registry: Mapping[str, Any],
    evaluation_registry: Mapping[str, Any],
    final_provider_census: Mapping[str, Any],
    runtime_authorization: Mapping[str, Any],
    campaign_attempt: int,
    campaign_root: Path,
) -> evidence.CampaignAggregator:
    aggregator = evidence.CampaignAggregator(
        evidence.CampaignBundle(
            stage_code=stage_code,
            parent_manifest=parent,
            bound_manifest=bound,
            scientific_plan=scientific,
            roster=roster,
            parallel_plan=parallel_plan,
            vm_registry=vm_registry,
            evaluation_registry=evaluation_registry,
            final_provider_census=final_provider_census,
            campaign_attempt=campaign_attempt,
            campaign_root=campaign_root,
            runtime_authorization=runtime_authorization,
        )
    )
    aggregator._verify_common_bindings()
    return aggregator


def _serial_generation_rows(
    *,
    stage_code: str,
    study_id: str,
    roster_digest: str,
    campaign_attempt: int,
    vm_registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate READY generations while allowing proved pre-READY gaps."""

    registry = _mapping(vm_registry, "serial VM registry")
    if (
        registry.get("schema") != "yeto_parallel_vm_registry_v1"
        or registry.get("stage_code") != stage_code
        or registry.get("study_id") != study_id
        or registry.get("roster_hash") != roster_digest
        or registry.get("campaign_attempt") != campaign_attempt
    ):
        raise SerialAuditError("serial VM registry common identity differs")
    rows = [
        _mapping(row, "serial VM generation row")
        for row in _array(registry.get("generations"), "serial VM generations")
    ]
    if not rows:
        raise SerialAuditError("serial VM registry is empty")
    seen_generations: set[int] = set()
    seen_namespaces: set[str] = set()
    seen_state_paths: set[str] = set()
    for row in rows:
        generation = evidence.require_positive_int(
            row.get("generation"), "serial VM generation"
        )
        if row.get("slot") != "v0" or generation in seen_generations:
            raise SerialAuditError("serial VM registry escaped or reused v0")
        seen_generations.add(generation)
        expected_run_id = evidence.physical_run_id(
            stage_code, roster_digest, campaign_attempt, "v0", generation
        )
        if row.get("run_id") != expected_run_id:
            raise SerialAuditError("serial VM registry run ID grammar differs")
        zone = row.get("zone")
        if zone not in evidence.ALLOWED_US_A100_ZONES:
            raise SerialAuditError("serial VM registry lacks an allowed landed zone")
        if row.get("region") != evidence.region_for_zone(zone):
            raise SerialAuditError("serial VM registry region differs from its zone")
        if row.get("machine_type") != "a2-highgpu-1g":
            raise SerialAuditError("serial VM registry is not 1g shaped")
        nonce = row.get("ownership_nonce")
        if not isinstance(nonce, str) or evidence.NONCE_RE.fullmatch(nonce) is None:
            raise SerialAuditError("serial VM registry nonce is not exact")
        expected_labels = {
            "campaign-tag": roster_digest[:16],
            "logical-slot": "v0",
            "physical-generation": str(generation),
            "run-id": expected_run_id,
            "ownership-nonce": nonce,
        }
        if row.get("labels") != expected_labels:
            raise SerialAuditError("serial VM registry ownership labels differ")
        prefix = row.get("artifact_prefix")
        state_path = row.get("state_path")
        if (
            not isinstance(prefix, str)
            or not prefix.endswith(f"/vms/v0/g{generation}/")
            or prefix in seen_namespaces
            or not isinstance(state_path, str)
            or not state_path.endswith(f"/{expected_run_id}.json")
            or state_path in seen_state_paths
        ):
            raise SerialAuditError("serial VM registry namespaces are not exact")
        expected_paths = {
            "provider_record_path": evidence._join_artifact_root(
                prefix, "provider/provider-evidence.json"
            ),
            "partial_manifest_path": evidence._join_artifact_root(
                prefix, "manifests/vm-partial-manifest.json"
            ),
            "lifecycle_record_path": evidence._join_artifact_root(
                prefix, "manifests/vm-lifecycle-final.json"
            ),
        }
        if any(row.get(field) != value for field, value in expected_paths.items()):
            raise SerialAuditError("serial VM registry outer paths cross namespaces")
        seen_namespaces.add(prefix)
        seen_state_paths.add(state_path)
    return sorted(rows, key=lambda row: int(row["generation"]))


def _transient_provider_evidence(
    *,
    stage_code: str,
    roster_digest: str,
    campaign_attempt: int,
    campaign_root: Path,
    registered_generations: Sequence[Mapping[str, Any]],
    transient_provider_registry: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prove every missing generation is one exact finalized pre-READY VM."""

    registry = _mapping(
        transient_provider_registry, "transient provider registry"
    )
    raw_rows = [
        _mapping(row, "transient provider registry row")
        for row in _array(
            registry.get("lifecycles"), "transient provider lifecycles"
        )
    ]
    if (
        registry.get("schema") != "audit_135m_transient_provider_registry_v1"
        or registry.get("status") != "SEALED_EXACT_ID_EVIDENCE"
        or registry.get("scientific_attempt_started") is not False
        or registry.get("loss_inspected") is not False
        or registry.get("count") != len(raw_rows)
    ):
        raise SerialAuditError("transient provider registry is not a blind exact-ID seal")

    registered_numbers = {int(row["generation"]) for row in registered_generations}
    maximum_registered = max(registered_numbers)
    missing_numbers = set(range(1, maximum_registered + 1)) - registered_numbers
    registered_run_ids = {str(row["run_id"]) for row in registered_generations}
    registered_nonces = {
        str(row["ownership_nonce"]) for row in registered_generations
    }
    registered_instance_ids: set[str] = set()
    registered_disk_ids: set[str] = set()
    for identity in registered_generations:
        generation = int(identity["generation"])
        provider = evidence._load_strict_object(
            campaign_root
            / "vms"
            / "v0"
            / f"g{generation}"
            / "provider"
            / "provider-evidence.json",
            "registered serial provider record",
        )
        provider_summary = evidence.validate_provider_record(provider, identity)
        registered_instance_ids.add(provider_summary["instance_numeric_id"])
        registered_disk_ids.add(provider_summary["boot_disk_numeric_id"])

    seen_numbers: set[int] = set()
    seen_run_ids: set[str] = set()
    seen_nonces: set[str] = set()
    seen_instance_ids: set[str] = set()
    seen_disk_ids: set[str] = set()
    canonical_rows: list[dict[str, Any]] = []
    lifecycle_hashes: list[dict[str, Any]] = []
    lifecycle_intervals: list[dict[str, Any]] = []
    for row in raw_rows:
        generation = evidence.require_positive_int(
            row.get("generation"), "transient generation"
        )
        expected_run_id = evidence.physical_run_id(
            stage_code, roster_digest, campaign_attempt, "v0", generation
        )
        if (
            row.get("slot") != "v0"
            or generation in seen_numbers
            or row.get("run_id") != expected_run_id
        ):
            raise SerialAuditError("transient generation identity is duplicated or escaped")
        expected_relative = (
            f"common/transient-provider-lifecycles/{expected_run_id}.json"
        )
        if row.get("path") != expected_relative:
            raise SerialAuditError("transient lifecycle path differs from its identity")
        lifecycle_path = evidence._safe_campaign_path(
            campaign_root, expected_relative, "transient lifecycle"
        )
        lifecycle = evidence._load_strict_object(
            lifecycle_path, "transient provider lifecycle"
        )
        lifecycle_hash = sha256_file(lifecycle_path)
        instance_id = evidence.require_numeric_id(
            lifecycle.get("instance_numeric_id"), "transient instance numeric ID"
        )
        disk_id = evidence.require_numeric_id(
            lifecycle.get("boot_disk_numeric_id"),
            "transient boot disk numeric ID",
        )
        nonce = lifecycle.get("ownership_nonce")
        if not isinstance(nonce, str) or evidence.NONCE_RE.fullmatch(nonce) is None:
            raise SerialAuditError("transient ownership nonce is not exact")
        expected_registry_row = {
            "run_id": expected_run_id,
            "slot": "v0",
            "generation": generation,
            "instance_numeric_id": instance_id,
            "boot_disk_numeric_id": disk_id,
            "path": expected_relative,
            "raw_sha256": lifecycle_hash,
        }
        if dict(row) != expected_registry_row:
            raise SerialAuditError("transient registry row does not hash-bind its lifecycle")
        if (
            lifecycle.get("schema")
            != "audit_135m_transient_provider_lifecycle_v1"
            or lifecycle.get("status")
            != "TRANSIENT_PROVIDER_PREEMPTED_AND_EXACT_IDS_ABSENT"
            or lifecycle.get("stage_code") != stage_code
            or lifecycle.get("run_id") != expected_run_id
            or lifecycle.get("slot") != "v0"
            or lifecycle.get("generation") != generation
            or lifecycle.get("provisioning_model") != "SPOT"
            or lifecycle.get("machine_type") != "a2-highgpu-1g"
            or lifecycle.get("provider_spot_preempted") is not True
            or lifecycle.get("scientific_attempt_started") is not False
            or lifecycle.get("loss_inspected") is not False
        ):
            raise SerialAuditError("transient lifecycle violates the serial Spot/blind rails")
        zone = lifecycle.get("zone")
        if (
            zone not in evidence.ALLOWED_US_A100_ZONES
            or lifecycle.get("region") != evidence.region_for_zone(zone)
        ):
            raise SerialAuditError("transient lifecycle landed region/zone differs")
        labels = _mapping(lifecycle.get("labels"), "transient ownership labels")
        expected_label_subset = {
            "campaign": "audit-135m",
            "campaign-tag": roster_digest[:16],
            "draft": "false",
            "logical-slot": "v0",
            "managed-by": "yeto-optimizer-harness",
            "physical-generation": str(generation),
            "run-id": expected_run_id,
            "stage": stage_code,
        }
        if any(labels.get(key) != value for key, value in expected_label_subset.items()):
            raise SerialAuditError("transient lifecycle ownership labels differ")
        boot_disk_name = lifecycle.get("boot_disk_name")
        source_image_id = evidence.require_numeric_id(
            lifecycle.get("source_image_numeric_id"),
            "transient source image numeric ID",
            allow_protected=True,
        )
        if boot_disk_name != expected_run_id or source_image_id != "7290368630472593484":
            raise SerialAuditError("transient disk/image identity differs")
        creation = evidence.parse_time(
            lifecycle.get("creation_timestamp"), "transient creation"
        )
        first_observed = evidence.parse_time(
            lifecycle.get("first_observed_at_utc"), "transient first observation"
        )
        completed = evidence.parse_time(
            lifecycle.get("deletion_completed_at_utc"), "transient deletion"
        )
        if first_observed < creation or completed < first_observed:
            raise SerialAuditError("transient lifecycle chronology differs")
        requested_value = lifecycle.get("deletion_requested_at_utc")
        if requested_value is not None:
            requested = evidence.parse_time(
                requested_value, "transient deletion request"
            )
            if requested < creation or completed < requested:
                raise SerialAuditError("transient deletion request chronology differs")
        if lifecycle.get("teardown_mode") not in {
            "OPERATOR_EXACT_DELETE",
            "PROVIDER_SPOT_PREEMPTION_AUTO_DELETE",
        }:
            raise SerialAuditError("transient teardown mode is not exact-ID bounded")
        proofs = _mapping(
            lifecycle.get("provider_not_found_verification"),
            "transient NOT_FOUND proofs",
        )
        instance_proof = _mapping(
            proofs.get("instance"), "transient instance NOT_FOUND proof"
        )
        disk_proof = _mapping(
            proofs.get("boot_disk"), "transient disk NOT_FOUND proof"
        )
        if (
            instance_proof.get("name") != expected_run_id
            or instance_proof.get("result") != "NOT_FOUND"
            or str(instance_proof.get("provider_id")) != instance_id
            or disk_proof.get("name") != boot_disk_name
            or disk_proof.get("result") != "NOT_FOUND"
            or str(disk_proof.get("provider_id")) != disk_id
        ):
            raise SerialAuditError("transient exact instance/disk NOT_FOUND proof differs")
        evidence.parse_time(
            instance_proof.get("verified_at_utc"),
            "transient instance NOT_FOUND time",
        )
        evidence.parse_time(
            disk_proof.get("verified_at_utc"),
            "transient disk NOT_FOUND time",
        )
        zero = _mapping(
            lifecycle.get("zero_attached_accelerator_proof"),
            "transient zero accelerator proof",
        )
        if (
            zero.get("generation_attached_a100s") != 0
            or str(zero.get("instance_numeric_id")) != instance_id
        ):
            raise SerialAuditError("transient lifecycle lacks a zero-A100 proof")
        evidence.parse_time(
            zero.get("verified_at_utc"), "transient zero-A100 proof time"
        )
        if (
            generation in registered_numbers
            or expected_run_id in registered_run_ids
            or nonce in registered_nonces
            or expected_run_id in seen_run_ids
            or nonce in seen_nonces
            or instance_id in registered_instance_ids
            or instance_id in seen_instance_ids
            or disk_id in registered_disk_ids
            or disk_id in seen_disk_ids
        ):
            raise SerialAuditError("transient lifecycle reuses a scientific/provider identity")
        seen_numbers.add(generation)
        seen_run_ids.add(expected_run_id)
        seen_nonces.add(nonce)
        seen_instance_ids.add(instance_id)
        seen_disk_ids.add(disk_id)
        canonical_rows.append(expected_registry_row)
        lifecycle_hashes.append(
            {"slot": "v0", "generation": generation, "sha256": lifecycle_hash}
        )
        lifecycle_intervals.append(
            {
                "creation_timestamp": lifecycle["creation_timestamp"],
                "deletion_completed_at_utc": lifecycle[
                    "deletion_completed_at_utc"
                ],
                "machine_type": "a2-highgpu-1g",
                "a100_count": 1,
            }
        )
    if seen_numbers != missing_numbers:
        raise SerialAuditError(
            "serial generation gaps are not covered exactly once by transient lifecycles"
        )
    canonical_rows.sort(key=lambda row: int(row["generation"]))
    lifecycle_hashes.sort(key=lambda row: int(row["generation"]))
    canonical_registry = {
        "schema": "audit_135m_transient_provider_registry_v1",
        "status": "SEALED_EXACT_ID_EVIDENCE",
        "scientific_attempt_started": False,
        "loss_inspected": False,
        "count": len(canonical_rows),
        "lifecycles": canonical_rows,
    }
    return canonical_registry, lifecycle_hashes, lifecycle_intervals


def _load_serial_evidence(
    *,
    compat: evidence.CampaignAggregator,
    stage_code: str,
    study_id: str,
    roster_digest: str,
    campaign_attempt: int,
    campaign_root: Path,
    vm_registry: Mapping[str, Any],
    transient_provider_registry: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    generations = _serial_generation_rows(
        stage_code=stage_code,
        study_id=study_id,
        roster_digest=roster_digest,
        campaign_attempt=campaign_attempt,
        vm_registry=vm_registry,
    )
    transient_registry, transient_hashes, transient_intervals = (
        _transient_provider_evidence(
            stage_code=stage_code,
            roster_digest=roster_digest,
            campaign_attempt=campaign_attempt,
            campaign_root=campaign_root,
            registered_generations=generations,
            transient_provider_registry=transient_provider_registry,
        )
    )
    attempts, partial_hashes, lifecycle_hashes, registered_intervals = (
        compat._load_vm_evidence(generations)
    )
    validate_serial_generation_lifecycles(
        [*registered_intervals, *transient_intervals]
    )
    return (
        generations,
        transient_registry,
        transient_hashes,
        attempts,
        partial_hashes,
        lifecycle_hashes,
    )


def _serial_common_verify(
    *,
    stage_code: str,
    parent: Mapping[str, Any],
    bound: Mapping[str, Any],
    scientific: Mapping[str, Any],
    roster: Mapping[str, Any],
    parallel_plan: Mapping[str, Any],
    binding: Mapping[str, Any],
    authorization: Mapping[str, Any],
    evaluation_registry: Mapping[str, Any],
) -> str:
    binding_preimage = dict(binding)
    binding_digest = binding_preimage.pop("serial_plan_hash", None)
    if (
        binding_digest != canonical_sha256(binding_preimage)
        or binding.get("serial_amendment_raw_sha256")
        != sha256_file(SERIAL_AMENDMENT_PATH)
    ):
        raise SerialAuditError("serial binding hash/amendment identity differs")
    expected_binding = build_serial_binding(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        compatibility_roster=roster,
        compatibility_plan=parallel_plan,
    )
    if canonical_json(expected_binding) != canonical_json(binding):
        raise SerialAuditError("serial binding does not reconstruct exactly")
    hard_ceiling = float(
        _mapping(bound.get("audit_135m_contract"), "bound audit contract")[
            "hard_ceiling_usd"
        ]
    )
    authorization_hash = verify_runtime_authorization(
        authorization,
        binding,
        expected_hard_ceiling_usd=hard_ceiling,
    )
    frozen = _mapping(bound.get("frozen"), "bound frozen")
    if frozen.get("randomization_plan_hash") != scientific.get(
        "randomization_plan_hash"
    ):
        raise SerialAuditError("bound manifest cites a different scientific plan")
    launch_seeds = {str(cell["seed"]) for cell in _cell_map(scientific).values()}
    if set(evaluation_registry) != launch_seeds:
        raise SerialAuditError("evaluation registry does not cover the serial suffix")
    for seed in launch_seeds:
        entry = _mapping(evaluation_registry[seed], f"evaluation registry seed {seed}")
        for field in evidence.EVAL_BOUND_FIELDS:
            if entry.get(field) != frozen.get(field):
                raise SerialAuditError("evaluation registry differs from the bound surface")
    evidence._validate_inherited_prefix(stage_code, parent, bound)
    return authorization_hash


def build_checkpoint_preseal(
    *,
    stage_code: str,
    parent: Mapping[str, Any],
    bound: Mapping[str, Any],
    scientific: Mapping[str, Any],
    roster: Mapping[str, Any],
    parallel_plan: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    serial_authorization: Mapping[str, Any],
    compatibility_runtime_authorization: Mapping[str, Any],
    vm_registry: Mapping[str, Any],
    transient_provider_registry: Mapping[str, Any],
    evaluation_registry: Mapping[str, Any],
    campaign_attempt: int,
    campaign_root: Path,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    authorization_hash = _serial_common_verify(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        roster=roster,
        parallel_plan=parallel_plan,
        binding=serial_binding,
        authorization=serial_authorization,
        evaluation_registry=evaluation_registry,
    )
    final_census = {
        "schema": "yeto_parallel_final_provider_census_v1",
        "campaign_owned_vm_count": 0,
        "campaign_owned_attached_a100s": 0,
    }
    compat = _compatibility_aggregator(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        roster=roster,
        parallel_plan=parallel_plan,
        vm_registry=vm_registry,
        evaluation_registry=evaluation_registry,
        final_provider_census=final_census,
        runtime_authorization=compatibility_runtime_authorization,
        campaign_attempt=campaign_attempt,
        campaign_root=campaign_root,
    )
    (
        generations,
        canonical_transient_registry,
        transient_lifecycle_hashes,
        attempts,
        _partial_hashes,
        _lifecycle_hashes,
    ) = _load_serial_evidence(
        compat=compat,
        stage_code=stage_code,
        study_id=str(bound["study_id"]),
        roster_digest=evidence.roster_hash(roster),
        campaign_attempt=campaign_attempt,
        campaign_root=campaign_root,
        vm_registry=vm_registry,
        transient_provider_registry=transient_provider_registry,
    )
    analysis = validate_serial_schedule(
        attempts=attempts, scientific=scientific, binding=serial_binding
    )
    work_reports, checkpoint_registry = compat._validate_work(attempts, analysis)
    ratchet_receipts = validate_ratchet_receipts(
        attempts=attempts,
        work_reports=work_reports,
        binding=serial_binding,
        campaign_root=campaign_root,
    )
    if checkpoint_registry is None or checkpoint_registry.get("schema") != (
        "audit_135m_checkpoint_registry_v1"
    ):
        raise SerialAuditError("serial checkpoint preseal lacks its exact registry")
    scientific_by_id = _cell_map(scientific)
    pending = sorted(
        cell_id
        for cell_id, cell in scientific_by_id.items()
        if cell.get("evaluation_mode")
        in {"confirmation_audit_pending", "development_prediction_pending"}
    )
    capture_only = sorted(
        cell_id
        for cell_id, cell in scientific_by_id.items()
        if cell.get("evaluation_mode") == "capture_only_no_endpoint"
    )
    terminal_times = [
        parse_time(row.get("scientific_ended_at"), "scientific end")
        for row in attempts
        if row.get("status") in {"COMPLETED", "DIVERGED"}
    ]
    if not terminal_times:
        raise SerialAuditError("serial checkpoint preseal has no terminal science")
    maximum = max(terminal_times)
    seal_time = sealed_at_utc or utc_now()
    if parse_time(seal_time, "checkpoint preseal time") <= maximum:
        raise SerialAuditError("checkpoint preseal does not follow terminal training")
    value = {
        "schema": "audit_135m_checkpoint_preseal_v1",
        "status": "SEALED_TRAINING_AND_CHECKPOINT_REGISTRY",
        "stage_code": stage_code,
        "study_id": bound["study_id"],
        "campaign_attempt": campaign_attempt,
        "loss_exposed": False,
        "provider_lifecycle_final_pending": True,
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "roster_hash": evidence.roster_hash(roster),
        "parallel_plan_hash": evidence.parallel_plan_hash(parallel_plan),
        "serial_plan_hash": serial_binding["serial_plan_hash"],
        "serial_runtime_authorization_hash": authorization_hash,
        "scientific_randomization_plan_hash": scientific[
            "randomization_plan_hash"
        ],
        "attempts_canonical_sha256": canonical_sha256(attempts),
        "attempts": [deepcopy(row) for row in attempts],
        "training_vm_registry_generations": [
            deepcopy(dict(row)) for row in generations
        ],
        "training_transient_provider_registry": canonical_transient_registry,
        "training_transient_provider_lifecycle_hashes": transient_lifecycle_hashes,
        "analysis_rounds": {
            key: analysis[key] for key in sorted(analysis, key=lambda s: s.encode())
        },
        "work_evidence_reports": work_reports,
        "serial_ratchet_receipts": ratchet_receipts,
        "evaluation_registry": deepcopy(dict(evaluation_registry)),
        "audit_checkpoint_registry": checkpoint_registry,
        "evaluation_required_cell_ids": pending,
        "capture_only_cell_ids": capture_only,
        "partial_outcomes_exposed": False,
        "maximum_training_completion_utc": maximum.isoformat().replace(
            "+00:00", "Z"
        ),
        "sealed_at_utc": seal_time,
    }
    value["preseal_canonical_sha256"] = canonical_sha256(value)
    return value


def aggregate(
    *,
    stage_code: str,
    parent: Mapping[str, Any],
    bound: Mapping[str, Any],
    scientific: Mapping[str, Any],
    roster: Mapping[str, Any],
    parallel_plan: Mapping[str, Any],
    serial_binding: Mapping[str, Any],
    serial_authorization: Mapping[str, Any],
    compatibility_runtime_authorization: Mapping[str, Any],
    vm_registry: Mapping[str, Any],
    transient_provider_registry: Mapping[str, Any],
    evaluation_registry: Mapping[str, Any],
    final_provider_census: Mapping[str, Any],
    campaign_attempt: int,
    campaign_root: Path,
    sealed_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization_hash = _serial_common_verify(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        roster=roster,
        parallel_plan=parallel_plan,
        binding=serial_binding,
        authorization=serial_authorization,
        evaluation_registry=evaluation_registry,
    )
    compat = _compatibility_aggregator(
        stage_code=stage_code,
        parent=parent,
        bound=bound,
        scientific=scientific,
        roster=roster,
        parallel_plan=parallel_plan,
        vm_registry=vm_registry,
        evaluation_registry=evaluation_registry,
        final_provider_census=final_provider_census,
        runtime_authorization=compatibility_runtime_authorization,
        campaign_attempt=campaign_attempt,
        campaign_root=campaign_root,
    )
    (
        generations,
        canonical_transient_registry,
        transient_lifecycle_hashes,
        attempts,
        partial_hashes,
        lifecycle_hashes,
    ) = _load_serial_evidence(
        compat=compat,
        stage_code=stage_code,
        study_id=str(bound["study_id"]),
        roster_digest=evidence.roster_hash(roster),
        campaign_attempt=campaign_attempt,
        campaign_root=campaign_root,
        vm_registry=vm_registry,
        transient_provider_registry=transient_provider_registry,
    )
    analysis = validate_serial_schedule(
        attempts=attempts, scientific=scientific, binding=serial_binding
    )
    work_reports, checkpoint_registry = compat._validate_work(attempts, analysis)
    ratchet_receipts = validate_ratchet_receipts(
        attempts=attempts,
        work_reports=work_reports,
        binding=serial_binding,
        campaign_root=campaign_root,
    )
    census = _mapping(final_provider_census, "final provider census")
    if (
        census.get("schema") != "yeto_parallel_final_provider_census_v1"
        or census.get("campaign_owned_vm_count") != 0
        or census.get("campaign_owned_attached_a100s") != 0
    ):
        raise SerialAuditError("serial final provider census is not zero")
    canonical_registry = deepcopy(dict(vm_registry))
    canonical_registry["generations"] = [deepcopy(dict(row)) for row in generations]
    manifest = {
        "schema": SERIAL_MANIFEST_SCHEMA,
        "execution_mode": serial_binding["execution_mode"],
        "stage_code": stage_code,
        "study_id": bound["study_id"],
        "campaign_attempt": campaign_attempt,
        "serial_amendment_raw_sha256": serial_binding[
            "serial_amendment_raw_sha256"
        ],
        "serial_plan_hash": serial_binding["serial_plan_hash"],
        "serial_runtime_authorization_hash": authorization_hash,
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "roster_hash": evidence.roster_hash(roster),
        "parallel_plan_hash": evidence.parallel_plan_hash(parallel_plan),
        "parallel_executor_used": False,
        "scientific_randomization_plan_hash": scientific[
            "randomization_plan_hash"
        ],
        "vm_registry": canonical_registry,
        "transient_provider_registry": canonical_transient_registry,
        "attempts": [deepcopy(row) for row in attempts],
        "analysis_rounds": {
            key: analysis[key] for key in sorted(analysis, key=lambda s: s.encode())
        },
        "work_evidence_reports": work_reports,
        "serial_ratchet_receipts": ratchet_receipts,
        "evaluation_registry": deepcopy(dict(evaluation_registry)),
        "final_provider_census": deepcopy(census),
        "partial_outcomes_exposed": False,
    }
    if checkpoint_registry is not None:
        manifest["audit_checkpoint_registry"] = checkpoint_registry
    manifest_hash = canonical_sha256(manifest)
    seal_time = sealed_at_utc or utc_now()
    parse_time(seal_time, "serial campaign seal time")
    seal = {
        "schema": SERIAL_SEAL_SCHEMA,
        "status": "sealed_results",
        "execution_mode": serial_binding["execution_mode"],
        "stage_code": stage_code,
        "study_id": bound["study_id"],
        "authoritative_prereg_template_sha256": audit.PREREG_JSON_SHA256,
        "serial_amendment_raw_sha256": serial_binding[
            "serial_amendment_raw_sha256"
        ],
        "serial_plan_hash": serial_binding["serial_plan_hash"],
        "serial_runtime_authorization_hash": authorization_hash,
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "roster_hash": evidence.roster_hash(roster),
        "parallel_plan_hash": evidence.parallel_plan_hash(parallel_plan),
        "parallel_executor_used": False,
        "scientific_randomization_plan_hash": scientific[
            "randomization_plan_hash"
        ],
        "campaign_manifest_canonical_sha256": manifest_hash,
        "vm_registry_canonical_sha256": canonical_sha256(canonical_registry),
        "vm_partial_manifest_hashes": partial_hashes,
        "vm_lifecycle_record_hashes": lifecycle_hashes,
        "transient_provider_lifecycle_hashes": transient_lifecycle_hashes,
        "transient_provider_registry_canonical_sha256": canonical_sha256(
            canonical_transient_registry
        ),
        "cumulative_expected_cell_count": len(bound["expected_cells"]),
        "launch_cell_count": len(scientific["cells"]),
        "resolved_launch_cell_count": len(analysis),
        "attempt_count": len(attempts),
        "completed_cell_ratchet_all_pass": True,
        "serial_ratchet_receipts_canonical_sha256": canonical_sha256(
            ratchet_receipts
        ),
        "one_physical_generation_at_a_time_all_pass": True,
        "work_evidence_all_pass": True,
        "schedule_all_pass": True,
        "provider_ownership_all_pass": True,
        "exact_id_teardown_all_pass": True,
        "generation_lineage_all_pass": True,
        "partial_outcomes_exposed": False,
        "sealed_at_utc": seal_time,
    }
    return manifest, seal


def _descriptor_path(descriptor_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SerialAuditError(f"descriptor {label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = descriptor_path.parent / path
    return path.resolve()


def aggregate_from_descriptor(
    descriptor_path: Path,
    *,
    write_seal: bool,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    descriptor = load_object(descriptor_path, "serial aggregation descriptor")
    if descriptor.get("schema") != "audit_135m_serial_aggregation_descriptor_v1":
        raise SerialAuditError("aggregation descriptor is not serial")

    def obj(field: str) -> dict[str, Any]:
        return load_object(
            _descriptor_path(descriptor_path, descriptor.get(field), field), field
        )

    campaign_root = _descriptor_path(
        descriptor_path, descriptor.get("campaign_root"), "campaign_root"
    )
    manifest, seal = aggregate(
        stage_code=str(descriptor.get("stage_code")),
        parent=obj("parent_manifest"),
        bound=obj("bound_manifest"),
        scientific=obj("scientific_plan"),
        roster=obj("parallel_roster"),
        parallel_plan=obj("parallel_plan"),
        serial_binding=obj("serial_binding"),
        serial_authorization=obj("serial_runtime_authorization"),
        compatibility_runtime_authorization=obj("runtime_authorization"),
        vm_registry=obj("vm_registry"),
        transient_provider_registry=obj("transient_provider_registry"),
        evaluation_registry=obj("evaluation_registry"),
        final_provider_census=obj("final_provider_census"),
        campaign_attempt=int(descriptor.get("campaign_attempt", 0)),
        campaign_root=campaign_root,
        sealed_at_utc=sealed_at_utc,
    )
    if write_seal:
        campaign_dir = campaign_root / "campaign"
        write_json_create_only(campaign_dir / "campaign-manifest.json", manifest)
        write_json_create_only(campaign_dir / "campaign-seal.json", seal)
    return seal
