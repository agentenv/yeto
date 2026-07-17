#!/usr/bin/env python3
"""Materialize cumulative A1/A3/A4 contracts for the 135M tuned-baseline audit.

This module is intentionally outcome-blind.  It verifies the frozen audit
preregistration, derives only the cells authorized for one registered
sub-stage, binds them as an append-only suffix of a sealed parent manifest,
and emits a non-authorizing runtime contract.  Development selections,
boundary extensions, and precision expansion are accepted only through
separately hashed decision manifests with exact schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from scripts import run_phase_map as phase


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_JSON = REPO_ROOT / "experiment-specs/tuned-baseline-audit-prereg.json"
PREREG_MD = REPO_ROOT / "experiment-specs/tuned-baseline-audit-prereg.md"
PREREG_JSON_SHA256 = "5198d62090ea307a5b8c7151f66088ddf8c57782b00591da93b1465f1c146eb7"
PREREG_MD_SHA256 = "3739da4d89f14e0081cac501b38164a7e5938bd5595b49d0020f9e862da6c804"

STAGE_CODES = frozenset(
    {
        "a1d",
        "a1x",
        "a1c",
        "a3k",
        "a3r0",
        "a3x",
        "a4d",
        "a4b",
        "a4c",
        "a4x",
    }
)
STAGE_TO_AUDIT = {
    "a1d": "A1",
    "a1x": "A1",
    "a1c": "A1",
    "a3k": "A3",
    "a3r0": "A3",
    "a3x": "A3",
    "a4d": "A4",
    "a4b": "A4",
    "a4c": "A4",
    "a4x": "A4",
}
STAGE_PHASE = {
    "a1d": "development_initial",
    "a1x": "development_boundary_extension",
    "a1c": "confirmation",
    "a3k": "matched_eta_kernel_recapture",
    "a3r0": "frontier_initial",
    "a3x": "frontier_boundary_extension",
    "a4d": "development_initial",
    "a4b": "development_boundary_extension",
    "a4c": "confirmation_initial",
    "a4x": "confirmation_precision_expansion",
}
HARD_CEILINGS = {"A1": 140.0, "A3": 31.18, "A4": 138.21}

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_HASH = "43f9494fad3335a9237f7a3093ae1401b7c4d3164c7486542070e2cc04837132"
DATA_HASH = "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409"
IMAGE_NUMERIC_ID = "7290368630472593484"
IMAGE_DIGEST = "038098c2b5356c9117f1019bf0d19c8999ab50f259dceb041a57fcf657d2620f"

TOKEN_BUDGET = 655_360
SEQ_LEN = 128
MICRO_BATCH_SIZE = 1
INNER_LR = 0.001
TRAIN_ROWS = 5_000
DEVELOPMENT_EVAL_ROWS = 1_024
AUDIT_EVAL_ROWS = 1_024
EVAL_SPLIT_SEED = 331
FRAGMENTS = 4
DIVERGENCE_LOSS_CAP = 10.0

A1_DEVELOPMENT_SEEDS = ((359, 359359), (373, 373373))
A1_CONFIRMATION_SEEDS = tuple(
    (seed, int(f"{seed}{seed}"))
    for seed in (383, 397, 409, 421, 433, 443, 457, 461)
)
A3_SEEDS = A1_DEVELOPMENT_SEEDS
A4_DEVELOPMENT_SEEDS = ((2069, 20692069), (2081, 20812081))
A4_CONFIRMATION_SEEDS = tuple(
    (seed, int(f"{seed}{seed}")) for seed in (2083, 2087, 2089)
)
A4_EXPANSION_SEEDS = tuple(
    (seed, int(f"{seed}{seed}")) for seed in (2099, 2111, 2113)
)

A1_GRIDS = {
    (16, 0.0): (0.015467960838455726, 0.021875, 0.030935921676911452),
    (16, 0.9): (0.0013671875, 0.002734375, 0.00546875),
    (256, 0.0): (0.030935921676911452, 0.04375, 0.061871843353822904),
    (256, 0.5): (0.0109375, 0.021875, 0.030935921676911452),
}
A4_GRIDS = {
    (16, 0.0): (0.0109375, 0.021875, 0.04375),
    (16, 0.9): (0.0013671875, 0.002734375, 0.00546875),
    (256, 0.0): (0.021875, 0.04375, 0.0875),
    (256, 0.5): (0.0109375, 0.021875, 0.04375),
}
A3_GRIDS = {
    8: (0.0109375, 0.015467960838455726, 0.021875, 0.030935921676911452),
    512: (0.021875, 0.030935921676911452, 0.04375, 0.061871843353822904, 0.0875),
}
A3_ALLOWED_EXTENSIONS = {
    8: {0.00546875, 0.04375},
    512: {0.015467960838455726, 0.12374368670764582},
}
FIXED_ETA = 0.0875


class AuditContractError(RuntimeError):
    """The requested packet differs from the frozen audit contract."""


@dataclass(frozen=True)
class ArmSpec:
    mu: float
    eta: float
    role: str
    pair_key: str
    finite_kernel_capture: bool = False


@dataclass(frozen=True)
class BlockSpec:
    h: int
    m: int
    seed: int
    training_seed: int
    arms: tuple[ArmSpec, ...]
    terminal_partial_window: bool = False
    evaluation_mode: str = "development_endpoint"


EVALUATION_MODES = frozenset(
    {
        "development_endpoint",
        "confirmation_audit_pending",
        "development_prediction_pending",
        "capture_only_no_endpoint",
    }
)


def _checkpoint_only(mode: str) -> bool:
    if mode not in EVALUATION_MODES:
        raise AuditContractError(f"unsupported audit evaluation mode {mode!r}")
    return mode != "development_endpoint"


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


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise AuditContractError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditContractError(f"{label} must be a JSON object")
    return value


def write_json_create_only(path: Path, value: object) -> None:
    if path.exists():
        raise AuditContractError(f"refusing to overwrite create-only artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_authority() -> dict[str, Any]:
    if sha256_file(PREREG_JSON) != PREREG_JSON_SHA256:
        raise AuditContractError("frozen audit JSON bytes differ")
    if sha256_file(PREREG_MD) != PREREG_MD_SHA256:
        raise AuditContractError("frozen audit Markdown bytes differ")
    authority = read_object(PREREG_JSON, "audit preregistration")
    if (
        authority.get("schema") != "tuned_baseline_audit_prereg_v1"
        or authority.get("revision") != "1.2"
        or authority.get("status", {}).get("no_controller_or_optimizer_zoo_authority")
        is not True
    ):
        raise AuditContractError("audit preregistration identity/status differs")
    expected_costs = {stage: HARD_CEILINGS[stage] for stage in HARD_CEILINGS}
    observed_costs = {
        stage: float(authority["costs"]["blocks"][stage]["hard_ceiling_usd"])
        for stage in expected_costs
    }
    if observed_costs != expected_costs:
        raise AuditContractError("registered 135M hard ceilings differ")
    for stage in expected_costs:
        experiment = authority["experiments"][stage]
        if (
            int(experiment.get("width_cap", 0)) != 2
            or float(experiment.get("abort_burn_kill_usd", -1.0)) != 40.0
        ):
            raise AuditContractError("registered width/abort-burn controls differ")
    return authority


def stage_seed_pairs(stage_code: str) -> tuple[tuple[int, int], ...]:
    return {
        "a1d": A1_DEVELOPMENT_SEEDS,
        "a1x": A1_DEVELOPMENT_SEEDS,
        "a1c": A1_CONFIRMATION_SEEDS,
        "a3k": ((347, 347347),),
        "a3r0": A3_SEEDS,
        "a3x": A3_SEEDS,
        "a4d": A4_DEVELOPMENT_SEEDS,
        "a4b": A4_DEVELOPMENT_SEEDS,
        "a4c": A4_CONFIRMATION_SEEDS,
        "a4x": A4_EXPANSION_SEEDS,
    }[stage_code]


def _selection(
    path: Path | None,
    schema: str,
    *,
    audit_stage: str,
) -> tuple[dict[str, Any], str]:
    if path is None:
        raise AuditContractError(f"{schema} stage requires a decision manifest")
    value = read_object(path, "decision manifest")
    expected_fields = {
        "schema",
        "status",
        "audit_stage",
        "authority_prereg_sha256",
        "development_evidence_canonical_sha256",
        "selection_rule",
        "boundary_extension_status",
        "selected_etas",
        "sealed_at_utc",
    }
    if set(value) != expected_fields:
        raise AuditContractError(f"decision manifest field set differs for {schema}")
    if value.get("schema") != schema or value.get("status") != "SEALED":
        raise AuditContractError(f"decision manifest must be sealed {schema}")
    if (
        value.get("audit_stage") != audit_stage
        or value.get("authority_prereg_sha256") != PREREG_JSON_SHA256
        or value.get("selection_rule") != "lowest_pooled_development_mean"
        or value.get("boundary_extension_status")
        not in {"NOT_REQUIRED", "REQUIRED", "COMPLETED", "UNBRACKETED"}
    ):
        raise AuditContractError(f"decision manifest contract differs for {schema}")
    require_sha256(
        value.get("development_evidence_canonical_sha256"),
        "development evidence canonical hash",
    )
    sealed_at = value.get("sealed_at_utc")
    if not isinstance(sealed_at, str) or not sealed_at.endswith("Z"):
        raise AuditContractError("decision manifest lacks a UTC seal time")
    return value, sha256_file(path)


def _a3_extension_selection(path: Path | None) -> tuple[dict[str, Any], str]:
    schema = "audit_135m_a3_frontier_selection_v1"
    if path is None:
        raise AuditContractError(f"{schema} stage requires a decision manifest")
    value = read_object(path, "A3 frontier decision manifest")
    expected_fields = {
        "schema",
        "status",
        "audit_stage",
        "authority_prereg_sha256",
        "frontier_evidence_canonical_sha256",
        "extension_rule",
        "extension_etas",
        "sealed_at_utc",
    }
    if set(value) != expected_fields:
        raise AuditContractError("A3 frontier decision manifest field set differs")
    if (
        value.get("schema") != schema
        or value.get("status") != "SEALED"
        or value.get("audit_stage") != "A3"
        or value.get("authority_prereg_sha256") != PREREG_JSON_SHA256
        or value.get("extension_rule")
        != "one_registered_outward_boundary_extension_maximum"
    ):
        raise AuditContractError("A3 frontier decision manifest contract differs")
    require_sha256(
        value.get("frontier_evidence_canonical_sha256"),
        "A3 frontier evidence canonical hash",
    )
    sealed_at = value.get("sealed_at_utc")
    if not isinstance(sealed_at, str) or not sealed_at.endswith("Z"):
        raise AuditContractError("A3 frontier decision lacks a UTC seal time")
    return value, sha256_file(path)


def _precision_trigger(path: Path | None) -> tuple[dict[str, Any], str]:
    schema = "audit_135m_a4_precision_trigger_v1"
    if path is None:
        raise AuditContractError("A4 precision expansion requires a trigger manifest")
    value = read_object(path, "A4 precision-trigger manifest")
    expected_fields = {
        "schema",
        "status",
        "audit_stage",
        "authority_prereg_sha256",
        "initial_confirmation_evidence_canonical_sha256",
        "initial_confirmation_complete",
        "precision_expansion_rule",
        "epsilon",
        "triggered_sign_blindly",
        "expansion_required",
        "run_all_registered_expansion_seeds",
        "sealed_at_utc",
    }
    if set(value) != expected_fields:
        raise AuditContractError("A4 precision-trigger manifest field set differs")
    epsilon = value.get("epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise AuditContractError("A4 precision-trigger epsilon is malformed")
    if (
        value.get("schema") != schema
        or value.get("status") != "SEALED"
        or value.get("audit_stage") != "A4"
        or value.get("authority_prereg_sha256") != PREREG_JSON_SHA256
        or value.get("initial_confirmation_complete") is not True
        or value.get("precision_expansion_rule")
        != "adjusted_ci_half_width_exceeds_epsilon_after_complete_initial_seed_block"
        or float(epsilon) != 0.01
        or value.get("triggered_sign_blindly") is not True
        or value.get("expansion_required") is not True
        or value.get("run_all_registered_expansion_seeds") is not True
    ):
        raise AuditContractError("A4 precision-trigger contract differs")
    require_sha256(
        value.get("initial_confirmation_evidence_canonical_sha256"),
        "A4 initial confirmation evidence canonical hash",
    )
    sealed_at = value.get("sealed_at_utc")
    if not isinstance(sealed_at, str) or not sealed_at.endswith("Z"):
        raise AuditContractError("A4 precision trigger lacks a UTC seal time")
    return value, sha256_file(path)


def _selected_eta_map(value: Mapping[str, Any], expected_keys: Iterable[str]) -> dict[str, float]:
    raw = value.get("selected_etas")
    if not isinstance(raw, Mapping) or set(raw) != set(expected_keys):
        raise AuditContractError("decision manifest selected-eta keys differ")
    result: dict[str, float] = {}
    for key, eta in raw.items():
        if isinstance(eta, bool) or not isinstance(eta, (int, float)) or eta <= 0:
            raise AuditContractError(f"selected eta {key} is not positive")
        result[str(key)] = float(eta)
    return result


def _paired_grid_arms(grids: Mapping[tuple[int, float], Sequence[float]], h: int) -> tuple[ArmSpec, ...]:
    mus = (0.0, 0.9) if h == 16 else (0.0, 0.5)
    baseline = tuple(float(value) for value in grids[(h, mus[0])])
    method = tuple(float(value) for value in grids[(h, mus[1])])
    if len(baseline) != len(method):
        raise AuditContractError("paired tuning grids have unequal cardinality")
    arms: list[ArmSpec] = []
    for index, (baseline_eta, method_eta) in enumerate(zip(baseline, method)):
        pair_key = f"rank{index}"
        arms.append(ArmSpec(mus[0], baseline_eta, f"tune_control_{pair_key}", pair_key))
        arms.append(ArmSpec(mus[1], method_eta, f"tune_method_{pair_key}", pair_key))
    return tuple(arms)


def _outward_extension(grid: Sequence[float], winner: float) -> float:
    values = tuple(float(value) for value in grid)
    if winner == values[0]:
        return values[0] * (values[0] / values[1])
    if winner == values[-1]:
        return values[-1] * (values[-1] / values[-2])
    raise AuditContractError("an outward extension was requested for an interior winner")


def _registered_grid_values(
    grid: Sequence[float], *, include_extensions: bool
) -> set[float]:
    values = {float(value) for value in grid}
    if include_extensions:
        values.add(_outward_extension(grid, float(grid[0])))
        values.add(_outward_extension(grid, float(grid[-1])))
    return values


def _validate_a1_selection(
    selected: Mapping[str, float], *, include_extensions: bool
) -> None:
    for h, method_mu in ((16, 0.9), (256, 0.5)):
        entries = (
            (f"H{h}_mu0", A1_GRIDS[(h, 0.0)]),
            (f"H{h}_mu{method_mu:g}", A1_GRIDS[(h, method_mu)]),
        )
        for key, grid in entries:
            if selected[key] not in _registered_grid_values(
                grid, include_extensions=include_extensions
            ):
                raise AuditContractError(f"A1 selected eta {key} is outside its registry")


def _validate_a4_selection(
    selected: Mapping[str, float], *, include_extensions: bool
) -> None:
    for m in (1, 4):
        for h, method_mu in ((16, 0.9), (256, 0.5)):
            entries = (
                (f"M{m}_H{h}_mu0", A4_GRIDS[(h, 0.0)]),
                (f"M{m}_H{h}_mu{method_mu:g}", A4_GRIDS[(h, method_mu)]),
            )
            for key, grid in entries:
                if selected[key] not in _registered_grid_values(
                    grid, include_extensions=include_extensions
                ):
                    raise AuditContractError(
                        f"A4 selected eta {key} is outside its registry"
                    )


def _a1_extension_blocks(decision: Mapping[str, Any]) -> list[BlockSpec]:
    selected = _selected_eta_map(
        decision, ("H16_mu0", "H16_mu0.9", "H256_mu0", "H256_mu0.5")
    )
    _validate_a1_selection(selected, include_extensions=False)
    blocks: list[BlockSpec] = []
    for h, method_mu in ((16, 0.9), (256, 0.5)):
        control_key = f"H{h}_mu0"
        method_key = f"H{h}_mu{method_mu:g}"
        control_boundary = selected[control_key] in {
            A1_GRIDS[(h, 0.0)][0], A1_GRIDS[(h, 0.0)][-1]
        }
        method_boundary = selected[method_key] in {
            A1_GRIDS[(h, method_mu)][0], A1_GRIDS[(h, method_mu)][-1]
        }
        if not (control_boundary or method_boundary):
            continue
        control_winner = selected[control_key]
        method_winner = selected[method_key]
        control_eta = (
            _outward_extension(A1_GRIDS[(h, 0.0)], control_winner)
            if control_boundary
            else _outward_extension(
                A1_GRIDS[(h, 0.0)],
                A1_GRIDS[(h, 0.0)][0]
                if method_winner == A1_GRIDS[(h, method_mu)][0]
                else A1_GRIDS[(h, 0.0)][-1],
            )
        )
        method_eta = (
            _outward_extension(A1_GRIDS[(h, method_mu)], method_winner)
            if method_boundary
            else _outward_extension(
                A1_GRIDS[(h, method_mu)],
                A1_GRIDS[(h, method_mu)][0]
                if control_winner == A1_GRIDS[(h, 0.0)][0]
                else A1_GRIDS[(h, method_mu)][-1],
            )
        )
        arms = (
            ArmSpec(0.0, control_eta, "boundary_control", "boundary"),
            ArmSpec(method_mu, method_eta, "boundary_method", "boundary"),
        )
        for seed, training_seed in A1_DEVELOPMENT_SEEDS:
            blocks.append(BlockSpec(h, 4, seed, training_seed, arms))
    if not blocks:
        raise AuditContractError("A1 boundary-extension stage has no boundary winner")
    return blocks


def _a4_extension_blocks(decision: Mapping[str, Any]) -> list[BlockSpec]:
    keys = [
        f"M{m}_H{h}_mu{mu:g}"
        for m in (1, 4)
        for h, mu in ((16, 0.0), (16, 0.9), (256, 0.0), (256, 0.5))
    ]
    selected = _selected_eta_map(decision, keys)
    _validate_a4_selection(selected, include_extensions=False)
    blocks: list[BlockSpec] = []
    for m in (1, 4):
        for h, method_mu in ((16, 0.9), (256, 0.5)):
            control_key = f"M{m}_H{h}_mu0"
            method_key = f"M{m}_H{h}_mu{method_mu:g}"
            control_grid = A4_GRIDS[(h, 0.0)]
            method_grid = A4_GRIDS[(h, method_mu)]
            control_boundary = selected[control_key] in {control_grid[0], control_grid[-1]}
            method_boundary = selected[method_key] in {method_grid[0], method_grid[-1]}
            if not (control_boundary or method_boundary):
                continue
            direction_low = (
                selected[control_key] == control_grid[0]
                if control_boundary
                else selected[method_key] == method_grid[0]
            )
            control_eta = _outward_extension(
                control_grid,
                selected[control_key]
                if control_boundary
                else (control_grid[0] if direction_low else control_grid[-1]),
            )
            method_eta = _outward_extension(
                method_grid,
                selected[method_key]
                if method_boundary
                else (method_grid[0] if direction_low else method_grid[-1]),
            )
            arms = (
                ArmSpec(0.0, control_eta, "boundary_control", "boundary"),
                ArmSpec(method_mu, method_eta, "boundary_method", "boundary"),
            )
            for seed, training_seed in A4_DEVELOPMENT_SEEDS:
                blocks.append(BlockSpec(h, m, seed, training_seed, arms))
    if not blocks:
        raise AuditContractError("A4 boundary-extension stage has no boundary winner")
    return blocks


def stage_blocks(
    stage_code: str,
    decision_path: Path | None = None,
    precision_trigger_path: Path | None = None,
) -> tuple[list[BlockSpec], dict[str, str]]:
    if stage_code not in STAGE_CODES:
        raise AuditContractError(f"unsupported audit stage code {stage_code!r}")
    decision_hashes: dict[str, str] = {}
    blocks: list[BlockSpec] = []
    if stage_code == "a1d":
        for seed, training_seed in A1_DEVELOPMENT_SEEDS:
            for h in (16, 256):
                blocks.append(BlockSpec(h, 4, seed, training_seed, _paired_grid_arms(A1_GRIDS, h)))
    elif stage_code == "a1x":
        decision, selection_hash = _selection(
            decision_path,
            "audit_135m_a1_development_selection_v1",
            audit_stage="A1",
        )
        if decision["boundary_extension_status"] != "REQUIRED":
            raise AuditContractError("A1 extension requires a REQUIRED boundary decision")
        decision_hashes["selection"] = selection_hash
        blocks = _a1_extension_blocks(decision)
    elif stage_code == "a1c":
        decision, selection_hash = _selection(
            decision_path,
            "audit_135m_a1_development_selection_v1",
            audit_stage="A1",
        )
        if decision["boundary_extension_status"] not in {
            "NOT_REQUIRED",
            "COMPLETED",
            "UNBRACKETED",
        }:
            raise AuditContractError("A1 confirmation precedes its boundary decision")
        decision_hashes["selection"] = selection_hash
        selected = _selected_eta_map(
            decision, ("H16_mu0", "H16_mu0.9", "H256_mu0", "H256_mu0.5")
        )
        _validate_a1_selection(selected, include_extensions=True)
        for seed, training_seed in A1_CONFIRMATION_SEEDS:
            for h, method_mu in ((16, 0.9), (256, 0.5)):
                arms = (
                    ArmSpec(0.0, FIXED_ETA, "fixed_control", "fixed"),
                    ArmSpec(method_mu, FIXED_ETA, "fixed_method", "fixed"),
                    ArmSpec(0.0, selected[f"H{h}_mu0"], "tuned_control", "tuned"),
                    ArmSpec(
                        method_mu,
                        selected[f"H{h}_mu{method_mu:g}"],
                        "tuned_method",
                        "tuned",
                    ),
                )
                blocks.append(
                    BlockSpec(
                        h,
                        4,
                        seed,
                        training_seed,
                        arms,
                        evaluation_mode="confirmation_audit_pending",
                    )
                )
    elif stage_code == "a3k":
        for h in (16, 64, 256):
            blocks.append(
                BlockSpec(
                    h,
                    4,
                    347,
                    347347,
                    (
                        ArmSpec(
                            0.0,
                            0.021875,
                            "kernel_recapture",
                            "self",
                            finite_kernel_capture=True,
                        ),
                    ),
                    evaluation_mode="capture_only_no_endpoint",
                )
            )
    elif stage_code == "a3r0":
        for seed, training_seed in A3_SEEDS:
            for h in (8, 512):
                arms = tuple(
                    ArmSpec(
                        0.0,
                        eta,
                        f"frontier_eta_{index}",
                        f"self_{index}",
                        finite_kernel_capture=(eta == 0.021875),
                    )
                    for index, eta in enumerate(A3_GRIDS[h])
                )
                blocks.append(
                    BlockSpec(
                        h,
                        4,
                        seed,
                        training_seed,
                        arms,
                        terminal_partial_window=(h == 512),
                        evaluation_mode="development_prediction_pending",
                    )
                )
    elif stage_code == "a3x":
        decision, selection_hash = _a3_extension_selection(decision_path)
        decision_hashes["selection"] = selection_hash
        raw = decision.get("extension_etas")
        if not isinstance(raw, Mapping) or not raw:
            raise AuditContractError("A3 extension decision has no extension etas")
        for raw_h, raw_eta in sorted(raw.items(), key=lambda item: int(item[0])):
            h = int(raw_h)
            eta = float(raw_eta)
            if h not in A3_ALLOWED_EXTENSIONS or eta not in A3_ALLOWED_EXTENSIONS[h]:
                raise AuditContractError("A3 boundary extension differs from the registry")
            for seed, training_seed in A3_SEEDS:
                blocks.append(
                    BlockSpec(
                        h,
                        4,
                        seed,
                        training_seed,
                        (ArmSpec(0.0, eta, "frontier_boundary_extension", "self"),),
                        terminal_partial_window=(h == 512),
                        evaluation_mode="development_prediction_pending",
                    )
                )
    elif stage_code == "a4d":
        for seed, training_seed in A4_DEVELOPMENT_SEEDS:
            for m in (1, 4):
                for h in (16, 256):
                    blocks.append(BlockSpec(h, m, seed, training_seed, _paired_grid_arms(A4_GRIDS, h)))
    elif stage_code == "a4b":
        decision, selection_hash = _selection(
            decision_path,
            "audit_135m_a4_development_selection_v1",
            audit_stage="A4",
        )
        if decision["boundary_extension_status"] != "REQUIRED":
            raise AuditContractError("A4 extension requires a REQUIRED boundary decision")
        decision_hashes["selection"] = selection_hash
        blocks = _a4_extension_blocks(decision)
    elif stage_code in {"a4c", "a4x"}:
        decision, selection_hash = _selection(
            decision_path,
            "audit_135m_a4_development_selection_v1",
            audit_stage="A4",
        )
        if decision["boundary_extension_status"] not in {
            "NOT_REQUIRED",
            "COMPLETED",
            "UNBRACKETED",
        }:
            raise AuditContractError("A4 confirmation precedes its boundary decision")
        decision_hashes["selection"] = selection_hash
        keys = [
            f"M{m}_H{h}_mu{mu:g}"
            for m in (1, 4)
            for h, mu in ((16, 0.0), (16, 0.9), (256, 0.0), (256, 0.5))
        ]
        selected = _selected_eta_map(decision, keys)
        _validate_a4_selection(selected, include_extensions=True)
        if stage_code == "a4x":
            _trigger, trigger_hash = _precision_trigger(precision_trigger_path)
            decision_hashes["precision_trigger"] = trigger_hash
            seeds = A4_EXPANSION_SEEDS
        else:
            seeds = A4_CONFIRMATION_SEEDS
        for seed, training_seed in seeds:
            for m in (1, 4):
                for h, method_mu in ((16, 0.9), (256, 0.5)):
                    arms = (
                        ArmSpec(0.0, FIXED_ETA, "fixed_control", "fixed"),
                        ArmSpec(method_mu, FIXED_ETA, "fixed_method", "fixed"),
                        ArmSpec(
                            0.0,
                            selected[f"M{m}_H{h}_mu0"],
                            "tuned_control",
                            "tuned",
                        ),
                        ArmSpec(
                            method_mu,
                            selected[f"M{m}_H{h}_mu{method_mu:g}"],
                            "tuned_method",
                            "tuned",
                        ),
                    )
                    blocks.append(
                        BlockSpec(
                            h,
                            m,
                            seed,
                            training_seed,
                            arms,
                            evaluation_mode="confirmation_audit_pending",
                        )
                    )
    if stage_code != "a4x" and precision_trigger_path is not None:
        raise AuditContractError(
            "precision-trigger manifest is authorized only for A4 precision expansion"
        )
    return blocks, decision_hashes


def _runtime_namespace(config: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        run_dir=Path(str(config["run_dir"])),
        model_path=Path(str(config["model_path"])),
        python_executable=str(config["python_executable"]),
        command_repo_root=Path(str(config["command_repo_root"])),
        token_budget=TOKEN_BUDGET,
        seq_len=SEQ_LEN,
        micro_batch_size=MICRO_BATCH_SIZE,
        inner_lr=INNER_LR,
        eval_rows=DEVELOPMENT_EVAL_ROWS,
        train_rows=TRAIN_ROWS,
        eval_split_seed=EVAL_SPLIT_SEED,
        device="cuda",
        gpu_slots=4,
        syncer_checkpoint_every=4,
        arm_timeout_min=240,
        require_distinct_learner_gpu_uuids=False,
        capture_every_step=False,
    )


def _cell_id(study_id: str, block: BlockSpec, arm: ArmSpec) -> str:
    role = re.sub(r"[^a-z0-9]+", "-", arm.role.lower()).strip("-")
    return (
        f"{study_id}-h{block.h}-m{block.m}-mu{phase.slug_float(arm.mu)}-"
        f"eta{phase.slug_float(arm.eta)}-s{block.seed}-r{role}"
    )


def _target_work(block: BlockSpec) -> dict[str, Any]:
    denominator = block.m * MICRO_BATCH_SIZE * SEQ_LEN
    learner_steps, remainder = divmod(TOKEN_BUDGET, denominator)
    if remainder or learner_steps <= 0:
        raise AuditContractError("token budget does not define exact per-learner work")
    window_rounds, terminal_remainder = divmod(learner_steps, block.h)
    if terminal_remainder:
        if not block.terminal_partial_window:
            raise AuditContractError(
                f"M={block.m}, H={block.h} needs an unregistered partial terminal window"
            )
        window_rounds += 1
    outer_steps = window_rounds * FRAGMENTS
    return {
        "tokens": TOKEN_BUDGET,
        "microsteps": TOKEN_BUDGET // SEQ_LEN,
        "outer_steps": outer_steps,
        "per_fragment_outer_steps": window_rounds,
        "learner_count": block.m,
        "quorum": block.m,
        "learner_steps_per_learner": learner_steps,
        "terminal_partial_window_registered": bool(terminal_remainder),
        "terminal_partial_window_microsteps": terminal_remainder,
    }


def build_plan(
    *,
    stage_code: str,
    study_id: str,
    runtime_config: Mapping[str, Any],
    order_seed: int,
    decision_path: Path | None = None,
    precision_trigger_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    load_authority()
    blocks, decision_hashes = stage_blocks(
        stage_code,
        decision_path,
        precision_trigger_path,
    )
    args = _runtime_namespace(runtime_config)
    rng = random.Random(order_seed)
    rng.shuffle(blocks)
    cells: list[dict[str, Any]] = []
    block_identities: dict[str, dict[str, Any]] = {}
    order_index = 0
    for block_index, block in enumerate(blocks):
        target = _target_work(block)
        block_id = (
            f"{study_id}-block-h{block.h}-m{block.m}-s{block.seed}"
        )
        identity = phase.seed_block_identity(
            args,
            h=block.h,
            seed=block.seed,
            training_seed=block.training_seed,
            learner_count=block.m,
        )
        identity["terminal_partial_window_registered"] = target[
            "terminal_partial_window_registered"
        ]
        identity["terminal_partial_window_microsteps"] = target[
            "terminal_partial_window_microsteps"
        ]
        identity["identity_hash"] = canonical_sha256(identity)
        block_identities[block_id] = identity
        arm_specs = list(block.arms)
        rng.shuffle(arm_specs)
        ids_by_role = {arm.role: _cell_id(study_id, block, arm) for arm in arm_specs}
        controls_by_pair: dict[str, str] = {}
        for arm in arm_specs:
            if arm.mu == 0.0:
                controls_by_pair.setdefault(arm.pair_key, ids_by_role[arm.role])
        block_cells: list[dict[str, Any]] = []
        for within_block_index, arm in enumerate(arm_specs):
            control_id = controls_by_pair.get(arm.pair_key)
            if control_id is None:
                if len(arm_specs) == 1 and arm.mu == 0.0:
                    control_id = ids_by_role[arm.role]
                elif arm.mu == 0.0:
                    control_id = ids_by_role[arm.role]
                else:
                    raise AuditContractError(
                        f"block {block_id} lacks a mu=0 control for {arm.pair_key}"
                    )
            command = phase.compare_command(
                args,
                h=block.h,
                mu=arm.mu,
                eta=arm.eta,
                seed=block.seed,
                training_seed=block.training_seed,
                learner_count=block.m,
                allow_terminal_partial_window=block.terminal_partial_window,
            )
            if arm.finite_kernel_capture:
                command.append("--audit-finite-kernel-capture")
            if _checkpoint_only(block.evaluation_mode):
                command.append("--train-only-sealed-checkpoint")
            cell = {
                "cell_id": ids_by_role[arm.role],
                "H": block.h,
                "M": block.m,
                "mu": arm.mu,
                "eta": arm.eta,
                "seed": block.seed,
                "training_seed": block.training_seed,
                "audit_stage": STAGE_TO_AUDIT[stage_code],
                "audit_phase": STAGE_PHASE[stage_code],
                "analysis_role": arm.role,
                "pair_key": arm.pair_key,
                "evaluation_mode": block.evaluation_mode,
                "finite_kernel_capture_required": arm.finite_kernel_capture,
                "command_hash": canonical_sha256(command),
                "pairing_command_hash": canonical_sha256(
                    phase.normalized_pairing_command(command)
                ),
                "pairing_identity_hash": identity["identity_hash"],
                "paired_control_id": control_id,
                "resource_class": "a2-highgpu-4g",
                "target_work": target,
                "randomization": {
                    "block_id": block_id,
                    "block_order_index": block_index,
                    "within_block_index": within_block_index,
                    "order_index": order_index,
                },
                "command": command,
            }
            block_cells.append(cell)
            cells.append(cell)
            order_index += 1
        if len({cell["cell_id"] for cell in block_cells}) != len(block_cells):
            raise AuditContractError(f"block {block_id} contains duplicate cell IDs")
        if len({cell["pairing_identity_hash"] for cell in block_cells}) != 1:
            raise AuditContractError(f"block {block_id} breaks its pairing identity")
        if len({cell["pairing_command_hash"] for cell in block_cells}) != 1:
            raise AuditContractError(f"block {block_id} changes a non-treatment command field")
        by_id = {cell["cell_id"]: cell for cell in block_cells}
        for cell in block_cells:
            control = by_id.get(cell["paired_control_id"])
            if control is None or float(control["mu"]) != 0.0:
                raise AuditContractError(f"block {block_id} has an invalid live control")
    plan = {
        "schema": "yeto_audit_135m_randomization_v1",
        "study_id": study_id,
        "stage_code": stage_code,
        "audit_stage": STAGE_TO_AUDIT[stage_code],
        "audit_phase": STAGE_PHASE[stage_code],
        "order_seed": order_seed,
        "seed_pairs": {
            str(seed): training for seed, training in stage_seed_pairs(stage_code)
        },
        "learner_counts": sorted({block.m for block in blocks}),
        "atomic_wave_unit": "complete_seed_h_m_comparison_block",
        "block_fields": ["H", "seed", "M"],
        "within_block_fields": ["mu", "eta", "analysis_role"],
        "loss_blind": True,
        "decision_manifest_hashes": decision_hashes,
        "seed_blocks": block_identities,
        "cells": cells,
    }
    plan["randomization_plan_hash"] = canonical_sha256(plan)
    return plan, decision_hashes


def _validate_parent(parent: Mapping[str, Any], expected_hash: str) -> None:
    if canonical_sha256(parent) != require_sha256(expected_hash, "parent canonical hash"):
        raise AuditContractError("parent manifest canonical hash differs")
    if parent.get("status") != "sealed_results":
        raise AuditContractError("audit descendant requires a sealed-results parent")
    frozen = parent.get("frozen")
    protocol = parent.get("protocol")
    if not isinstance(frozen, Mapping) or not isinstance(protocol, Mapping):
        raise AuditContractError("parent lacks frozen protocol bindings")
    expected_frozen = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_hash": MODEL_HASH,
        "data_hash": DATA_HASH,
        "image_id": IMAGE_NUMERIC_ID,
        "image_digest": IMAGE_DIGEST,
    }
    for key, expected in expected_frozen.items():
        if frozen.get(key) != expected:
            raise AuditContractError(f"parent frozen {key} differs")
    expected_protocol = {
        "train_rows": TRAIN_ROWS,
        "development_eval_rows": DEVELOPMENT_EVAL_ROWS,
        "audit_eval_rows": AUDIT_EVAL_ROWS,
        "seq_len": SEQ_LEN,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "inner_lr": INNER_LR,
        "token_budget": TOKEN_BUDGET,
        "eval_split_seed": EVAL_SPLIT_SEED,
        "spot_only": True,
        "barrier": True,
        "strict_quorum": True,
        "version_matched": True,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise AuditContractError(f"parent protocol {key} differs")


def _seed_registry(path: Path, required_seeds: set[int]) -> dict[str, Any]:
    registry = read_object(path, "seed-bundle registry")
    if registry.get("schema") != "audit_135m_seed_bundle_registry_v1":
        raise AuditContractError("seed-bundle registry has the wrong schema")
    seeds = registry.get("seeds")
    if not isinstance(seeds, Mapping) or set(seeds) != {str(seed) for seed in required_seeds}:
        raise AuditContractError("seed-bundle registry does not cover exactly this suffix")
    for seed, entry in seeds.items():
        if not isinstance(entry, Mapping):
            raise AuditContractError(f"seed registry entry {seed} is malformed")
        require_sha256(entry.get("train_rows_sha256"), f"seed {seed} train rows")
        require_sha256(
            entry.get("train_source_indices_sha256"),
            f"seed {seed} train source indices",
        )
        require_sha256(entry.get("parallel_eval_freeze_sha256"), f"seed {seed} eval freeze")
    return registry


def build_bound_manifest(
    *,
    stage_code: str,
    study_id: str,
    git_commit: str,
    parent: Mapping[str, Any],
    expected_parent_hash: str,
    plan: Mapping[str, Any],
    seed_registry: Mapping[str, Any],
    decision_hashes: Mapping[str, str],
) -> dict[str, Any]:
    authority = load_authority()
    _validate_parent(parent, expected_parent_hash)
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise AuditContractError("scientific Git commit must be lowercase 40-hex")
    parent_cells = parent.get("expected_cells")
    parent_results = parent.get("results")
    if not isinstance(parent_cells, list) or not isinstance(parent_results, list):
        raise AuditContractError("parent cells/results are not arrays")
    new_cells = []
    for cell in plan["cells"]:
        new_cells.append(
            {
                "cell_id": cell["cell_id"],
                "h": cell["H"],
                "m": cell["M"],
                "mu": cell["mu"],
                "eta": cell["eta"],
                "seed": cell["seed"],
                "training_seed": cell["training_seed"],
                "block_id": cell["randomization"]["block_id"],
                "paired_control_id": cell["paired_control_id"],
                "pairing_identity_hash": cell["pairing_identity_hash"],
                "pairing_command_hash": cell["pairing_command_hash"],
                "audit_stage": cell["audit_stage"],
                "audit_phase": cell["audit_phase"],
                "analysis_role": cell["analysis_role"],
                "pair_key": cell["pair_key"],
                "evaluation_mode": cell["evaluation_mode"],
                "finite_kernel_capture_required": cell[
                    "finite_kernel_capture_required"
                ],
                "command_hash": cell["command_hash"],
                "normalized_workload_command_hash": canonical_sha256(
                    phase.normalized_workload_command(cell["command"])
                ),
                "expected_learner_count": cell["target_work"]["learner_count"],
                "expected_quorum": cell["target_work"]["quorum"],
                "expected_learner_steps": cell["target_work"][
                    "learner_steps_per_learner"
                ],
                "expected_outer_steps": cell["target_work"]["outer_steps"],
                "terminal_partial_window_registered": cell["target_work"][
                    "terminal_partial_window_registered"
                ],
                "terminal_partial_window_microsteps": cell["target_work"][
                    "terminal_partial_window_microsteps"
                ],
            }
        )
    old_ids = {str(cell.get("cell_id")) for cell in parent_cells if isinstance(cell, Mapping)}
    new_ids = {cell["cell_id"] for cell in new_cells}
    if old_ids & new_ids or len(new_ids) != len(new_cells):
        raise AuditContractError("audit suffix repeats a cumulative cell ID")
    manifest = deepcopy(dict(parent))
    manifest["schema"] = "yeto_audit_135m_cumulative_manifest_v1"
    manifest["status"] = "bound_runtime_contract"
    manifest["launch_authorized"] = False
    manifest["study_id"] = study_id
    manifest["mode"] = "audit_135m_loss_blind_acquisition"
    manifest["expected_cells"] = deepcopy(parent_cells) + new_cells
    manifest["results"] = deepcopy(parent_results)
    seed_pairs = dict(manifest.get("seed_pairs") or {})
    seed_pairs.update(plan["seed_pairs"])
    manifest["seed_pairs"] = seed_pairs
    frozen = dict(manifest.get("frozen") or {})
    train_rows = dict(frozen.get("train_rows_hashes") or {})
    train_indices = dict(frozen.get("train_source_indices_hashes") or {})
    for seed, entry in seed_registry["seeds"].items():
        train_rows[seed] = entry["train_rows_sha256"]
        train_indices[seed] = entry["train_source_indices_sha256"]
    cell_hashes = dict(frozen.get("cell_command_hashes") or {})
    cell_hashes.update({cell["cell_id"]: cell["command_hash"] for cell in plan["cells"]})
    frozen.update(
        {
            "git_commit": git_commit,
            "image_id": IMAGE_NUMERIC_ID,
            "image_digest": IMAGE_DIGEST,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_hash": MODEL_HASH,
            "data_hash": DATA_HASH,
            "train_rows_hashes": train_rows,
            "train_source_indices_hashes": train_indices,
            "cell_command_hashes": cell_hashes,
            "command_hash": canonical_sha256(
                [
                    {"cell_id": cell["cell_id"], "command_hash": cell_hashes[cell["cell_id"]]}
                    for cell in manifest["expected_cells"]
                ]
            ),
            "randomization_plan_hash": plan["randomization_plan_hash"],
            "audit_prereg_json_sha256": PREREG_JSON_SHA256,
            "audit_prereg_markdown_sha256": PREREG_MD_SHA256,
            "decision_manifest_hashes": dict(decision_hashes),
        }
    )
    manifest["frozen"] = frozen
    protocol = dict(manifest.get("protocol") or {})
    protocol.update(
        {
            "tuning": "full",
            "train_rows": TRAIN_ROWS,
            "development_eval_rows": DEVELOPMENT_EVAL_ROWS,
            "audit_eval_rows": AUDIT_EVAL_ROWS,
            "seq_len": SEQ_LEN,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "inner_lr": INNER_LR,
            "fragments": FRAGMENTS,
            "token_budget": TOKEN_BUDGET,
            "learner_counts": plan["learner_counts"],
            "strict_quorum": True,
            "barrier": True,
            "version_matched": True,
            "fixed_window": True,
            "spot_only": True,
            "on_demand_fallback": False,
            "train_all_then_audit_all": True,
        }
    )
    manifest["protocol"] = protocol
    manifest["expected_grid"] = {
        "h": sorted({int(cell["h"]) for cell in manifest["expected_cells"]}),
        "m": sorted({int(cell.get("m", 4)) for cell in manifest["expected_cells"]}),
        "mu": sorted({float(cell["mu"]) for cell in manifest["expected_cells"]}),
        "eta": sorted({float(cell["eta"]) for cell in manifest["expected_cells"]}),
        "seeds": sorted({int(cell["seed"]) for cell in manifest["expected_cells"]}),
    }
    manifest["randomization"] = {
        "unit": "arm",
        "atomic_wave_unit": plan["atomic_wave_unit"],
        "block_fields": plan["block_fields"],
        "within_block_fields": plan["within_block_fields"],
        "block_order": "materialized_pseudorandom_permutation",
        "within_block_order": "materialized_pseudorandom_permutation",
        "loss_blind": True,
        "plan_hash": plan["randomization_plan_hash"],
    }
    manifest["pairing"] = {
        "independent_unit": "complete_training_seed_block",
        "same_initialization": True,
        "same_data_order": True,
        "same_worker_allocation": True,
        "same_work_schedule": True,
        "uniform_runtime_machine_shape_within_block": True,
        "seed_blocks": deepcopy(plan["seed_blocks"]),
    }
    manifest["analysis_policy"] = {
        "divergence_loss_cap": DIVERGENCE_LOSS_CAP,
        "divergence_is_outcome": True,
        "silent_divergence_exclusion_forbidden": True,
    }
    manifest["retry_policy"] = {
        "loss_blind_only": True,
        "rerun_entire_incomplete_block": True,
        "completed_peers_retain_original_rows": True,
        "partial_state_resume_forbidden": True,
        "retain_all_attempts": True,
        "retry_lineage_required": True,
        "direct_infrastructure_failure_reasons": sorted(
            phase.DIRECT_INFRASTRUCTURE_FAILURE_REASONS
        ),
        "peer_retry_reason": phase.PEER_BLOCK_RETRY_REASON,
    }
    audit_stage = STAGE_TO_AUDIT[stage_code]
    design_contract = {
        "schema": "audit_135m_stage_contract_v1",
        "stage_code": stage_code,
        "audit_stage": audit_stage,
        "audit_phase": STAGE_PHASE[stage_code],
        "hard_ceiling_usd": HARD_CEILINGS[audit_stage],
        "spot_only": True,
        "maximum_attached_a100_equivalent": 16,
        "max_idle_before_science_seconds": 600,
        "seed_pairs": plan["seed_pairs"],
        "launch_cell_count": len(plan["cells"]),
        "decision_manifest_hashes": dict(decision_hashes),
        "authority": {
            "json_sha256": PREREG_JSON_SHA256,
            "markdown_sha256": PREREG_MD_SHA256,
            "registered_at_date": authority["registered_at_date"],
        },
    }
    design_hash = canonical_sha256(design_contract)
    manifest["audit_135m_contract"] = design_contract
    manifest["lineage"] = {
        "descendant_kind": f"audit_135m_{stage_code}",
        "parent_manifest_sha256": require_sha256(
            expected_parent_hash, "parent manifest hash"
        ),
        "authoritative_prereg_path": PREREG_JSON.relative_to(REPO_ROOT).as_posix(),
        "authoritative_prereg_template_sha256": PREREG_JSON_SHA256,
        "authoritative_prereg_markdown_sha256": PREREG_MD_SHA256,
        "audit_135m_design_contract_hash": design_hash,
        "cumulative_parent_cells": len(parent_cells),
        "cumulative_parent_results": len(parent_results),
        "append_only_suffix_cells": len(new_cells),
    }
    manifest["launch_authorization_requirement"] = (
        "bind an exact audit_135m runtime authorization to the preregistration, "
        "parent, bound manifest, scientific plan, roster, parallel plan, stage "
        "ceiling, and Spot/accelerator rails before provider mutation"
    )
    return manifest


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    parent = read_object(args.parent_manifest, "sealed cumulative parent")
    plan, decision_hashes = build_plan(
        stage_code=args.stage_code,
        study_id=args.study_id,
        runtime_config={
            "run_dir": args.run_dir,
            "model_path": args.model_path,
            "python_executable": args.python_executable,
            "command_repo_root": args.command_repo_root,
        },
        order_seed=args.order_seed,
        decision_path=args.decision_manifest,
        precision_trigger_path=args.precision_trigger_manifest,
    )
    required_seeds = {int(cell["seed"]) for cell in plan["cells"]}
    registry = _seed_registry(args.seed_bundle_registry, required_seeds)
    bound = build_bound_manifest(
        stage_code=args.stage_code,
        study_id=args.study_id,
        git_commit=args.git_commit,
        parent=parent,
        expected_parent_hash=args.expected_parent_manifest_hash,
        plan=plan,
        seed_registry=registry,
        decision_hashes=decision_hashes,
    )
    output = args.output_dir
    write_json_create_only(output / "scientific-randomization-plan.json", plan)
    write_json_create_only(output / "bound-manifest.json", bound)
    write_json_create_only(output / "seed-bundle-registry.json", registry)
    summary = {
        "schema": "audit_135m_materialization_v1",
        "status": "MATERIALIZED_NOT_AUTHORIZED",
        "stage_code": args.stage_code,
        "audit_stage": STAGE_TO_AUDIT[args.stage_code],
        "audit_phase": STAGE_PHASE[args.stage_code],
        "study_id": args.study_id,
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "scientific_randomization_plan_hash": plan["randomization_plan_hash"],
        "launch_cell_count": len(plan["cells"]),
        "decision_manifest_hashes": decision_hashes,
        "hard_ceiling_usd": HARD_CEILINGS[STAGE_TO_AUDIT[args.stage_code]],
        "launch_authorized": False,
    }
    write_json_create_only(output / "materialization.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-code", choices=sorted(STAGE_CODES), required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-manifest-hash", required=True)
    parser.add_argument("--decision-manifest", type=Path)
    parser.add_argument("--precision-trigger-manifest", type=Path)
    parser.add_argument("--seed-bundle-registry", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--command-repo-root", type=Path, required=True)
    parser.add_argument("--order-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize(args)
    except (AuditContractError, OSError, ValueError) as exc:
        print(f"audit-135m contract error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
