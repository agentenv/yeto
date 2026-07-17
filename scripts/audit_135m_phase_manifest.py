#!/usr/bin/env python3
"""Promote one sealed audit campaign into append-only cumulative lineage.

The parallel campaign seal is the authority for training work, provider
ownership, and exact-ID teardown.  Checkpoint-only rows remain loss-free in the
training-attempt prefix.  When a stage requires deferred evaluation, this
bridge separately verifies the checkpoint preseal and the complete hidden
batch, then appends one authoritative endpoint row per evaluated cell only
after the batch seal and shared unblind exist.

The parent ``expected_cells`` and ``results`` arrays are immutable ordered
prefixes.  Every infrastructure failure, divergence, completed training
attempt, and superseded retry remains present in the suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts import audit_135m_contract as audit
from scripts import run_parallel_phase_map as parallel


class PromotionError(RuntimeError):
    """A campaign cannot be promoted without changing or exposing its record."""


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
        raise PromotionError(f"{label} is not a UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PromotionError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise PromotionError(f"{label} lacks timezone information")
    return parsed.astimezone(timezone.utc)


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must be a JSON object")
    return value


def write_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise PromotionError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PromotionError(f"{label} must be an array")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionError(f"{label} must be an object")
    return dict(value)


def _expected_cells(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _array(manifest.get("expected_cells"), "expected cells")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _mapping(raw, "expected cell")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in result:
            raise PromotionError("expected cell IDs are missing or duplicated")
        result[cell_id] = row
    return result


def _verify_parent_prefix(
    *, parent: Mapping[str, Any], bound: Mapping[str, Any]
) -> tuple[int, int]:
    parent_cells = _array(parent.get("expected_cells"), "parent expected cells")
    parent_results = _array(parent.get("results"), "parent results")
    bound_cells = _array(bound.get("expected_cells"), "bound expected cells")
    bound_results = _array(bound.get("results"), "bound results")
    if bound_cells[: len(parent_cells)] != parent_cells:
        raise PromotionError("bound manifest changed the parent expected-cell prefix")
    if bound_results != parent_results:
        raise PromotionError("bound manifest changed the parent result prefix")
    lineage = _mapping(bound.get("lineage"), "bound lineage")
    if lineage.get("parent_manifest_sha256") != canonical_sha256(parent):
        raise PromotionError("bound lineage does not cite the exact parent")
    return len(parent_cells), len(parent_results)


def _verify_campaign(
    *,
    descriptor: Path,
    campaign_manifest_path: Path,
    campaign_seal_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor_value = load_object(descriptor, "campaign aggregation descriptor")
    if descriptor_value.get("aggregation_authorized") is not True:
        raise PromotionError("campaign descriptor does not authorize aggregation")
    campaign = load_object(campaign_manifest_path, "campaign manifest")
    seal = load_object(campaign_seal_path, "campaign seal")
    if (
        seal.get("schema") != "yeto_parallel_campaign_seal_v1"
        or seal.get("status") != "sealed_results"
        or seal.get("partial_outcomes_exposed") is not False
        or seal.get("work_evidence_all_pass") is not True
        or seal.get("provider_ownership_all_pass") is not True
        or seal.get("exact_id_teardown_all_pass") is not True
        or campaign.get("partial_outcomes_exposed") is not False
        or seal.get("campaign_manifest_canonical_sha256")
        != canonical_sha256(campaign)
    ):
        raise PromotionError("parallel campaign seal is not a complete blinded PASS")
    reproduced = parallel.aggregate_from_descriptor(
        descriptor.resolve(),
        write_seal=False,
        sealed_at_utc=str(seal["sealed_at_utc"]),
    )
    if reproduced != seal:
        raise PromotionError("read-only aggregation does not reproduce the campaign seal")
    return campaign, seal


def _campaign_root_uri(campaign: Mapping[str, Any]) -> str:
    attempts = _array(campaign.get("attempts"), "campaign attempts")
    prefixes = [
        str(row.get("attempt_prefix"))
        for row in attempts
        if isinstance(row, Mapping) and isinstance(row.get("attempt_prefix"), str)
    ]
    if not prefixes or any("/vms/" not in prefix for prefix in prefixes):
        raise PromotionError("campaign attempt prefixes are malformed")
    roots = {prefix.split("/vms/", 1)[0] for prefix in prefixes}
    if len(roots) != 1:
        raise PromotionError("campaign attempts do not share one artifact root")
    return next(iter(roots))


def _artifact_uri(root_uri: str, entry: Any) -> str | None:
    if entry is None:
        return None
    row = _mapping(entry, "artifact entry")
    relative = row.get("path")
    if not isinstance(relative, str) or not relative:
        raise PromotionError("artifact entry lacks a relative path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts:
        raise PromotionError("artifact entry escapes its campaign root")
    return root_uri.rstrip("/") + "/" + relative


def _provider_lifecycle(
    campaign_root: Path, attempt: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    slot = str(attempt.get("logical_slot"))
    generation = int(attempt.get("generation", 0))
    root = campaign_root / "vms" / slot / f"g{generation}"
    provider_path = root / "provider" / "provider-evidence.json"
    lifecycle_path = root / "manifests" / "vm-lifecycle-final.json"
    provider = load_object(provider_path, "provider evidence")
    lifecycle = load_object(lifecycle_path, "lifecycle evidence")
    parallel.validate_provider_record(provider, provider)
    partial_path = root / "manifests" / "vm-partial-manifest.json"
    partial_hash = sha256_file(partial_path)
    parallel.validate_lifecycle_record(lifecycle, provider, provider, partial_hash)
    if (
        str(provider.get("instance_numeric_id"))
        != str(attempt.get("instance_numeric_id"))
        or provider.get("ownership_nonce") != attempt.get("ownership_nonce")
    ):
        raise PromotionError("attempt provider identity differs from sealed evidence")
    return provider, lifecycle, sha256_file(provider_path), sha256_file(lifecycle_path)


def _training_row(
    *,
    expected: Mapping[str, Any],
    attempt: Mapping[str, Any],
    campaign_root: Path,
    root_uri: str,
) -> dict[str, Any]:
    cell_id = str(attempt.get("cell_id"))
    if (
        cell_id != expected.get("cell_id")
        or attempt.get("frozen_command_hash") != expected.get("command_hash")
        or attempt.get("normalized_workload_command_hash")
        != expected.get("normalized_workload_command_hash")
        or attempt.get("group_id") != expected.get("block_id")
    ):
        raise PromotionError(f"training attempt {attempt.get('attempt_id')} changed its cell")
    status = str(attempt.get("status"))
    mode = str(expected.get("evaluation_mode", "development_endpoint"))
    loss = attempt.get("loss")
    if status == "COMPLETED":
        if mode == "development_endpoint":
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            ):
                raise PromotionError("completed development attempt lacks a finite loss")
        elif loss is not None:
            raise PromotionError("checkpoint-only training attempt exposed an endpoint loss")
    elif status in {"DIVERGED", "INFRA_FAILURE"}:
        if loss is not None:
            raise PromotionError("non-completed training attempt carries a raw loss")
    else:
        raise PromotionError(f"unsupported training attempt status {status!r}")

    provider, lifecycle, provider_hash, lifecycle_hash = _provider_lifecycle(
        campaign_root, attempt
    )
    inventory = _mapping(attempt.get("artifact_inventory"), "attempt inventory")
    expected_work = {
        "tokens": int(expected.get("expected_learner_steps", 0))
        * int(expected.get("expected_learner_count", 0))
        * audit.SEQ_LEN,
        "microsteps": int(expected.get("expected_learner_steps", 0))
        * int(expected.get("expected_learner_count", 0)),
        "learner_count": int(expected.get("expected_learner_count", 0)),
        "quorum": int(expected.get("expected_quorum", 0)),
        "learner_steps_per_learner": int(expected.get("expected_learner_steps", 0)),
        "outer_steps": int(expected.get("expected_outer_steps", 0)),
        "terminal_partial_window_registered": bool(
            expected.get("terminal_partial_window_registered", False)
        ),
        "terminal_partial_window_microsteps": int(
            expected.get("terminal_partial_window_microsteps", 0)
        ),
    }
    return {
        "result_kind": "training_attempt",
        "cell_id": cell_id,
        "attempt_id": str(attempt["attempt_id"]),
        "attempt": int(attempt["attempt"]),
        "status": status,
        "loss": loss,
        "raw_loss": loss,
        "analysis_loss": attempt.get("analysis_loss"),
        "analysis_loss_kind": attempt.get("analysis_loss_kind"),
        "divergence_retained": attempt.get("divergence_retained"),
        "evaluation_mode": mode,
        "evaluation_role": (
            "development"
            if mode == "development_endpoint"
            else "withheld_pending_complete_bound_batch"
            if mode != "capture_only_no_endpoint"
            else "capture_only_no_endpoint"
        ),
        "h": int(expected["h"]),
        "m": int(expected.get("m", 4)),
        "mu": float(expected["mu"]),
        "eta": float(expected["eta"]),
        "seed": int(expected["seed"]),
        "training_seed": int(expected["training_seed"]),
        "audit_stage": expected.get("audit_stage"),
        "audit_phase": expected.get("audit_phase"),
        "analysis_role": expected.get("analysis_role"),
        "pair_key": expected.get("pair_key"),
        "paired_control_id": expected.get("paired_control_id"),
        "block_id": expected.get("block_id"),
        "command_hash": expected.get("command_hash"),
        "normalized_workload_command_hash": expected.get(
            "normalized_workload_command_hash"
        ),
        "work": expected_work,
        "observed_work": deepcopy(attempt.get("observed_work")),
        "failure_reason": attempt.get("failure_reason"),
        "retry_of": attempt.get("retry_of"),
        "retry_reason": attempt.get("retry_reason"),
        "retry_authorization": deepcopy(attempt.get("retry_authorization")),
        "started_at": attempt.get("scientific_started_at"),
        "ended_at": attempt.get("scientific_ended_at"),
        "spot": True,
        "resource_class": provider.get("machine_type"),
        "hardware": {
            "provider": provider.get("provider"),
            "project": provider.get("project"),
            "market": "spot",
            "instance_type": provider.get("machine_type"),
            "region": provider.get("region"),
            "zone": provider.get("zone"),
            "instance_numeric_id": str(provider.get("instance_numeric_id")),
            "boot_disk_numeric_id": str(provider.get("boot_disk_numeric_id")),
            "run_id": provider.get("run_id"),
            "slot": provider.get("slot"),
            "generation": provider.get("generation"),
            "ownership_nonce": provider.get("ownership_nonce"),
            "provider_spot_preempted": lifecycle.get("provider_spot_preempted"),
            "deletion_requested_at_utc": lifecycle.get("deletion_requested_at_utc"),
            "deletion_completed_at_utc": lifecycle.get("deletion_completed_at_utc"),
            "instance_not_found": True,
            "boot_disk_not_found": True,
            "generation_attached_a100s_final": 0,
            "provider_evidence_raw_sha256": provider_hash,
            "lifecycle_evidence_raw_sha256": lifecycle_hash,
        },
        "artifact_inventory": deepcopy(inventory),
        "artifact_uris": {
            role: _artifact_uri(root_uri, entry)
            for role, entry in sorted(inventory.items())
        },
        "parallel_attempt_projection": deepcopy(dict(attempt)),
    }


def _verify_preseal(
    *, preseal_path: Path, campaign: Mapping[str, Any], bound: Mapping[str, Any]
) -> dict[str, Any]:
    value = load_object(preseal_path, "checkpoint preseal")
    preimage = dict(value)
    digest = preimage.pop("preseal_canonical_sha256", None)
    if (
        value.get("schema") != "audit_135m_checkpoint_preseal_v1"
        or value.get("status") != "SEALED_TRAINING_AND_CHECKPOINT_REGISTRY"
        or value.get("loss_exposed") is not False
        or value.get("partial_outcomes_exposed") is not False
        or digest != canonical_sha256(preimage)
        or value.get("bound_manifest_canonical_sha256") != canonical_sha256(bound)
        or value.get("attempts_canonical_sha256")
        != canonical_sha256(value.get("attempts"))
    ):
        raise PromotionError("checkpoint preseal identity or hash differs")
    if campaign.get("audit_checkpoint_registry") != value.get(
        "audit_checkpoint_registry"
    ):
        raise PromotionError(
            "final campaign checkpoint registry differs from the pre-hidden preseal"
        )
    if campaign.get("attempts") != value.get("attempts"):
        raise PromotionError("final campaign training attempts differ from the preseal")
    maximum = parse_time(
        value.get("maximum_training_completion_utc"), "maximum training completion"
    )
    sealed = parse_time(value.get("sealed_at_utc"), "checkpoint preseal time")
    if sealed <= maximum:
        raise PromotionError("checkpoint preseal chronology differs")
    return value


def _verify_private_artifacts(root: Path, bundle: Mapping[str, Any]) -> None:
    rows = _array(bundle.get("private_artifacts"), "hidden private artifacts")
    if bundle.get("private_artifacts_hash") != canonical_sha256(rows):
        raise PromotionError("hidden private-artifact registry hash differs")
    seen: set[str] = set()
    for raw in rows:
        row = _mapping(raw, "hidden private artifact")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in seen:
            raise PromotionError("hidden private-artifact paths are malformed or duplicated")
        seen.add(relative)
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise PromotionError("hidden private artifact escapes its sealed root")
        path = root / Path(*parsed.parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != row.get("sha256")
            or path.stat().st_size != row.get("size_bytes")
        ):
            raise PromotionError("hidden private artifact hash or size differs")


def _verify_hidden(
    *,
    hidden_root: Path,
    authorization_path: Path,
    preseal: Mapping[str, Any],
    bound: Mapping[str, Any],
    prediction_freeze_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle_path = hidden_root / "audit-bundle.json"
    seal_path = hidden_root / "audit-seal.json"
    unblind_path = hidden_root / "shared-unblind.json"
    authorization = load_object(authorization_path, "hidden authorization")
    bundle = load_object(bundle_path, "hidden evaluation bundle")
    seal = load_object(seal_path, "hidden evaluation seal")
    unblind = load_object(unblind_path, "shared unblind")
    auth_preimage = dict(authorization)
    auth_digest = auth_preimage.pop("authorization_canonical_sha256", None)
    bundle_preimage = dict(bundle)
    bundle_digest = bundle_preimage.pop("bundle_canonical_sha256", None)
    if (
        authorization.get("schema")
        != "audit_135m_hidden_evaluation_authorization_v2"
        or authorization.get("status") != "SEALED"
        or authorization.get("loss_blind") is not True
        or auth_digest != canonical_sha256(auth_preimage)
        or authorization.get("checkpoint_preseal_canonical_sha256")
        != canonical_sha256(preseal)
        or authorization.get("bound_manifest_canonical_sha256")
        != canonical_sha256(bound)
        or bundle.get("schema") != "audit_135m_hidden_evaluation_bundle_v2"
        or bundle.get("status") != "SEALED"
        or bundle.get("partial_results_exposed") is not False
        or bundle_digest != canonical_sha256(bundle_preimage)
        or bundle.get("authorization_canonical_sha256")
        != canonical_sha256(authorization)
        or seal.get("schema") != "audit_135m_hidden_evaluation_seal_v2"
        or seal.get("status") != "sealed_results"
        or seal.get("bundle_raw_sha256") != sha256_file(bundle_path)
        or seal.get("bundle_canonical_sha256") != bundle_digest
        or unblind.get("schema") != "audit_135m_shared_unblind_v1"
        or unblind.get("bundle_raw_sha256") != sha256_file(bundle_path)
        or unblind.get("seal_raw_sha256") != sha256_file(seal_path)
    ):
        raise PromotionError("hidden authorization, bundle, seal, or unblind differs")
    prediction_hash = (
        None if prediction_freeze_path is None else sha256_file(prediction_freeze_path)
    )
    if authorization.get("prediction_freeze_sha256") != prediction_hash or bundle.get(
        "prediction_freeze_sha256"
    ) != prediction_hash:
        raise PromotionError("hidden batch prediction-freeze binding differs")
    if prediction_freeze_path is not None:
        prediction = load_object(prediction_freeze_path, "A3 prediction freeze")
        prediction_time = parse_time(
            prediction.get("sealed_at_utc"), "A3 prediction freeze time"
        )
    else:
        prediction_time = parse_time(preseal["sealed_at_utc"], "checkpoint preseal time")
    chronology = [
        parse_time(preseal["sealed_at_utc"], "checkpoint preseal time"),
        prediction_time,
        parse_time(authorization["authorized_at_utc"], "hidden authorization time"),
        parse_time(bundle["batch_started_at_utc"], "hidden batch start"),
        parse_time(bundle["batch_ended_at_utc"], "hidden batch end"),
        parse_time(seal["sealed_at_utc"], "hidden seal time"),
        parse_time(unblind["shared_unblind_at_utc"], "shared unblind time"),
    ]
    if any(right < left for left, right in zip(chronology, chronology[1:])):
        raise PromotionError("hidden evaluation chronology is reordered")
    if chronology[2] <= chronology[1] or chronology[3] <= chronology[2]:
        raise PromotionError("hidden authorization/evaluation did not follow its freeze")
    _verify_private_artifacts(hidden_root, bundle)
    return authorization, bundle, seal, unblind


def _hidden_rows(
    *,
    bundle: Mapping[str, Any],
    authorization: Mapping[str, Any],
    campaign: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    order = _array(authorization.get("evaluation_order"), "hidden evaluation order")
    results = _array(bundle.get("results"), "hidden results")
    if [row.get("cell_id") for row in results if isinstance(row, Mapping)] != order:
        raise PromotionError("hidden result order differs from its authorization")
    analysis = _mapping(campaign.get("analysis_rounds"), "campaign analysis rounds")
    attempts = {
        str(row["attempt_id"]): row
        for row in _array(campaign.get("attempts"), "campaign attempts")
    }
    rows = []
    for order_index, raw in enumerate(results):
        hidden = _mapping(raw, "hidden result")
        cell_id = str(hidden.get("cell_id"))
        cell = expected.get(cell_id)
        selected = analysis.get(cell_id)
        if cell is None or not isinstance(selected, Mapping):
            raise PromotionError("hidden result lies outside the authoritative campaign set")
        training = attempts.get(str(selected.get("attempt_id")))
        if training is None or hidden.get("training_status") != training.get("status"):
            raise PromotionError("hidden result training-attempt binding differs")
        status = str(hidden.get("training_status"))
        loss = hidden.get("audit_loss")
        if status == "COMPLETED":
            if (
                hidden.get("audit_status") != "COMPLETED"
                or isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            ):
                raise PromotionError("completed hidden endpoint is not finite")
        elif status == "DIVERGED":
            if loss is not None or hidden.get("audit_status") != "SCIENTIFIC_DIVERGENCE":
                raise PromotionError("diverged hidden endpoint was changed")
        else:
            raise PromotionError("hidden batch contains a nonterminal training status")
        rows.append(
            {
                "result_kind": "hidden_endpoint_evaluation",
                "cell_id": cell_id,
                "attempt_id": str(training["attempt_id"]) + "::hidden-evaluation",
                "training_attempt_id": training["attempt_id"],
                "attempt": int(training["attempt"]),
                "status": status,
                "loss": loss,
                "raw_loss": loss,
                # JSON has no portable infinity representation.  The terminal
                # status plus the explicit loss kind is the authoritative
                # positive-infinity tuning/inference sentinel.
                "analysis_loss": None if status == "DIVERGED" else loss,
                "analysis_loss_kind": hidden.get("analysis_loss_kind"),
                "evaluation_mode": cell.get("evaluation_mode"),
                "evaluation_role": bundle.get("evaluation_role"),
                "h": int(cell["h"]),
                "m": int(cell.get("m", 4)),
                "mu": float(cell["mu"]),
                "eta": float(cell["eta"]),
                "seed": int(cell["seed"]),
                "training_seed": int(cell["training_seed"]),
                "audit_stage": cell.get("audit_stage"),
                "audit_phase": cell.get("audit_phase"),
                "analysis_role": cell.get("analysis_role"),
                "pair_key": cell.get("pair_key"),
                "paired_control_id": cell.get("paired_control_id"),
                "block_id": cell.get("block_id"),
                "order_index": order_index,
                "started_at": hidden.get("started_at_utc"),
                "ended_at": hidden.get("ended_at_utc"),
                "checkpoint_inventory_canonical_sha256": hidden.get(
                    "checkpoint_inventory_canonical_sha256"
                ),
                "audit_command_hash": hidden.get("audit_command_hash"),
                "per_sequence_sha256": hidden.get("per_sequence_sha256"),
                "sequence_count": hidden.get("sequence_count"),
                "supervised_token_count": hidden.get("supervised_token_count"),
                "hidden_bundle_canonical_sha256": bundle.get(
                    "bundle_canonical_sha256"
                ),
                "hidden_surface": deepcopy(bundle.get("surface")),
            }
        )
    return rows


def _verify_cost(
    *,
    campaign_cost_path: Path,
    stage_ledger_path: Path,
    stage_code: str,
    audit_stage: str,
    roster_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_cost = load_object(campaign_cost_path, "campaign cost")
    ledger = load_object(stage_ledger_path, "stage spend ledger")
    ceiling = audit.HARD_CEILINGS[audit_stage]
    if (
        campaign_cost.get("stage_code") != stage_code
        or campaign_cost.get("roster_hash") != roster_hash
        or campaign_cost.get("completed") is not True
        or _mapping(campaign_cost.get("final_zero_census"), "cost zero census").get(
            "campaign_owned_attached_a100s"
        )
        != 0
        or ledger.get("schema") != "audit_135m_stage_spend_ledger_v1"
        or ledger.get("audit_stage") != audit_stage
        or float(ledger.get("hard_ceiling_usd", math.nan)) != ceiling
        or float(ledger.get("estimated_spend_usd", math.inf)) >= ceiling
    ):
        raise PromotionError("campaign/stage cost ledger violates the hard ceiling")
    return campaign_cost, ledger


def promote(args: argparse.Namespace) -> dict[str, Any]:
    audit.load_authority()
    parent = load_object(args.parent_manifest, "parent manifest")
    bound = load_object(args.bound_manifest, "bound manifest")
    _parent_cell_count, parent_result_count = _verify_parent_prefix(
        parent=parent, bound=bound
    )
    campaign, campaign_seal = _verify_campaign(
        descriptor=args.aggregation_descriptor,
        campaign_manifest_path=args.campaign_manifest,
        campaign_seal_path=args.campaign_seal,
    )
    if (
        campaign.get("bound_manifest_canonical_sha256") != canonical_sha256(bound)
        or campaign.get("parent_manifest_canonical_sha256")
        != canonical_sha256(parent)
    ):
        raise PromotionError("campaign does not bind the supplied parent/bound manifests")
    stage_code = str(campaign["stage_code"])
    audit_stage = str(bound["audit_135m_contract"]["audit_stage"])
    expected = _expected_cells(bound)
    launch_ids = set(_mapping(campaign.get("analysis_rounds"), "analysis rounds"))
    suffix_ids = {
        str(row["cell_id"])
        for row in _array(bound.get("expected_cells"), "bound cells")
        if isinstance(row, Mapping) and row.get("audit_phase") == bound["audit_135m_contract"]["audit_phase"]
    }
    if launch_ids != suffix_ids:
        raise PromotionError("campaign authoritative cell set differs from the bound suffix")

    root_uri = _campaign_root_uri(campaign)
    training_rows = [
        _training_row(
            expected=expected[str(raw["cell_id"])],
            attempt=_mapping(raw, "campaign attempt"),
            campaign_root=args.campaign_root.resolve(),
            root_uri=root_uri,
        )
        for raw in _array(campaign.get("attempts"), "campaign attempts")
    ]
    pending = sorted(
        cell_id
        for cell_id in launch_ids
        if expected[cell_id].get("evaluation_mode")
        in {"confirmation_audit_pending", "development_prediction_pending"}
    )
    hidden_rows: list[dict[str, Any]] = []
    preseal = authorization = bundle = hidden_seal = unblind = None
    if pending:
        required_paths = (
            args.checkpoint_preseal,
            args.hidden_authorization,
            args.hidden_root,
        )
        if any(path is None for path in required_paths):
            raise PromotionError("deferred-evaluation campaign lacks its hidden evidence")
        assert args.checkpoint_preseal is not None
        assert args.hidden_authorization is not None
        assert args.hidden_root is not None
        preseal = _verify_preseal(
            preseal_path=args.checkpoint_preseal,
            campaign=campaign,
            bound=bound,
        )
        if set(preseal.get("evaluation_required_cell_ids", [])) != set(pending):
            raise PromotionError("checkpoint preseal pending-evaluation coverage differs")
        authorization, bundle, hidden_seal, unblind = _verify_hidden(
            hidden_root=args.hidden_root.resolve(),
            authorization_path=args.hidden_authorization,
            preseal=preseal,
            bound=bound,
            prediction_freeze_path=args.prediction_freeze,
        )
        hidden_rows = _hidden_rows(
            bundle=bundle,
            authorization=authorization,
            campaign=campaign,
            expected=expected,
        )
        if {row["cell_id"] for row in hidden_rows} != set(pending):
            raise PromotionError("hidden endpoints do not cover the exact pending set")
    elif any(
        value is not None
        for value in (
            args.checkpoint_preseal,
            args.hidden_authorization,
            args.hidden_root,
            args.prediction_freeze,
        )
    ):
        raise PromotionError("non-deferred campaign supplied unexpected hidden evidence")

    campaign_cost, stage_ledger = _verify_cost(
        campaign_cost_path=args.campaign_cost,
        stage_ledger_path=args.stage_spend_ledger,
        stage_code=stage_code,
        audit_stage=audit_stage,
        roster_hash=str(campaign["roster_hash"]),
    )
    phase = deepcopy(bound)
    inherited = deepcopy(_array(bound.get("results"), "bound results"))
    phase["status"] = "sealed_results"
    phase["launch_authorized"] = False
    phase["results"] = inherited + training_rows + hidden_rows
    if phase["results"][:parent_result_count] != parent.get("results"):
        raise PromotionError("promotion changed the inherited parent result prefix")
    if phase["expected_cells"][: len(parent["expected_cells"])] != parent[
        "expected_cells"
    ]:
        raise PromotionError("promotion changed the inherited parent cell prefix")
    seals = list(parent.get("audit_135m_lineage_seals") or [])
    seal_binding = {
        "schema": "audit_135m_lineage_suffix_seal_v1",
        "stage_code": stage_code,
        "audit_stage": audit_stage,
        "audit_phase": bound["audit_135m_contract"]["audit_phase"],
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "bound_manifest_raw_sha256": sha256_file(args.bound_manifest),
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "campaign_manifest_raw_sha256": sha256_file(args.campaign_manifest),
        "campaign_manifest_canonical_sha256": canonical_sha256(campaign),
        "campaign_seal_raw_sha256": sha256_file(args.campaign_seal),
        "campaign_seal": deepcopy(campaign_seal),
        "checkpoint_preseal_raw_sha256": (
            None if args.checkpoint_preseal is None else sha256_file(args.checkpoint_preseal)
        ),
        "checkpoint_registry_exactly_reproduced_after_teardown": preseal is not None,
        "hidden_authorization_raw_sha256": (
            None
            if args.hidden_authorization is None
            else sha256_file(args.hidden_authorization)
        ),
        "hidden_bundle_raw_sha256": (
            None
            if args.hidden_root is None
            else sha256_file(args.hidden_root / "audit-bundle.json")
        ),
        "hidden_seal_raw_sha256": (
            None
            if args.hidden_root is None
            else sha256_file(args.hidden_root / "audit-seal.json")
        ),
        "shared_unblind_raw_sha256": (
            None
            if args.hidden_root is None
            else sha256_file(args.hidden_root / "shared-unblind.json")
        ),
        "prediction_freeze_raw_sha256": (
            None if args.prediction_freeze is None else sha256_file(args.prediction_freeze)
        ),
        "campaign_cost_raw_sha256": sha256_file(args.campaign_cost),
        "stage_spend_ledger_raw_sha256": sha256_file(args.stage_spend_ledger),
        "estimated_campaign_cost_usd": campaign_cost["estimated_cost_usd"],
        "estimated_stage_spend_usd": stage_ledger["estimated_spend_usd"],
        "hard_ceiling_usd": audit.HARD_CEILINGS[audit_stage],
        "training_attempt_count": len(training_rows),
        "hidden_endpoint_count": len(hidden_rows),
        "exact_id_teardown_all_pass": True,
        "final_zero_provider_census": True,
        "partial_outcomes_exposed": False,
    }
    seals.append(seal_binding)
    phase["audit_135m_lineage_seals"] = seals
    sealed_at = args.sealed_at_utc or utc_now()
    seal_floor = parse_time(campaign_seal["sealed_at_utc"], "campaign seal time")
    if unblind is not None:
        seal_floor = max(
            seal_floor,
            parse_time(unblind["shared_unblind_at_utc"], "shared unblind time"),
        )
    if parse_time(sealed_at, "phase seal time") <= seal_floor:
        raise PromotionError("phase promotion seal does not follow all source seals")
    phase["sealed_at_utc"] = sealed_at
    write_create_only(args.output_manifest, phase)
    attestation = {
        "schema": "audit_135m_parallel_hidden_to_phase_attestation_v1",
        "status": "PASS",
        "stage_code": stage_code,
        "audit_stage": audit_stage,
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "parent_manifest_raw_sha256": sha256_file(args.parent_manifest),
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "bound_manifest_raw_sha256": sha256_file(args.bound_manifest),
        "campaign_manifest_raw_sha256": sha256_file(args.campaign_manifest),
        "campaign_seal_raw_sha256": sha256_file(args.campaign_seal),
        "output_manifest_raw_sha256": sha256_file(args.output_manifest),
        "output_manifest_canonical_sha256": canonical_sha256(phase),
        "inherited_result_prefix_count": parent_result_count,
        "appended_training_attempt_count": len(training_rows),
        "appended_hidden_endpoint_count": len(hidden_rows),
        "output_expected_cell_count": len(phase["expected_cells"]),
        "output_result_count": len(phase["results"]),
        "parent_cells_and_results_are_exact_immutable_prefixes": True,
        "all_training_attempts_retained": True,
        "checkpoint_only_training_rows_remain_loss_null": all(
            row["loss"] is None
            for row in training_rows
            if row["evaluation_mode"] != "development_endpoint"
            and row["status"] == "COMPLETED"
        ),
        "hidden_rows_appended_only_after_complete_bundle_seal": bool(hidden_rows),
        "checkpoint_registry_exactly_reproduced_after_teardown": preseal is not None,
        "cost_and_exact_teardown_pass": True,
    }
    write_create_only(args.output_attestation, attestation)
    return {
        "status": "SEALED_RESULTS",
        "stage_code": stage_code,
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": sha256_file(args.output_manifest),
        "output_attestation": str(args.output_attestation),
        "training_attempt_count": len(training_rows),
        "hidden_endpoint_count": len(hidden_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--bound-manifest", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--campaign-seal", type=Path, required=True)
    parser.add_argument("--aggregation-descriptor", type=Path, required=True)
    parser.add_argument("--campaign-cost", type=Path, required=True)
    parser.add_argument("--stage-spend-ledger", type=Path, required=True)
    parser.add_argument("--checkpoint-preseal", type=Path)
    parser.add_argument("--hidden-authorization", type=Path)
    parser.add_argument("--hidden-root", type=Path)
    parser.add_argument("--prediction-freeze", type=Path)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-attestation", type=Path, required=True)
    parser.add_argument("--sealed-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = promote(args)
    except (
        PromotionError,
        OSError,
        ValueError,
        KeyError,
        audit.AuditContractError,
        parallel.ParallelPhaseMapError,
    ) as exc:
        print(f"audit phase-promotion error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
