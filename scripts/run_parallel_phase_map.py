#!/usr/bin/env python3
"""Amendment-native parallel orchestration for phase-map scientific cells.

This module is the cross-VM control plane that deliberately sits outside the
scientific command.  It binds a cumulative phase-map descendant to the fixed
parallel rank function, owns physical VM generations, dispatches one cell per
logical slot, retains whole-wave retry lineage, hash-locks one partial manifest
per physical generation, validates exact-ID teardown, and writes the sole
campaign seal.

The implementation is provider-adapter driven.  Importing or materializing a
plan is CPU-only; cloud mutations can occur only through an explicitly supplied
``ParallelExecutionBackend`` used by ``ParallelWaveExecutor``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import secrets
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


MASTER_SEED_HEX = (
    "0728fa50c14f4e52113407ab12e173b7ef4eb3b3b36f192ec7b814dd411223c5"
)
MASTER_SEED_PREIMAGE = (
    "yeto-best-paper-parallel-cells-v1|"
    "16d27bc60deb6d8910bf0111c7fb57c9d0eb5b80|"
    "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
)
AUTHORITATIVE_PREREG_TEMPLATE_SHA256 = (
    "7cba3c62328b4bfe15fffbc523979274e834e8e720e16f70d79621eaf6ebdb7b"
)
AMENDMENT_RAW_SHA256 = (
    "33d2bca5e00737145c7996c236f9f3face255216608c7c8b635a18cdb0fc2542"
)
REVISION_1_2_AMENDMENT_RAW_SHA256 = (
    "33781ad5d4deb29120a2d41f3ccbe2937a5945b97db6400ff1690abeceb520f7"
)
PROTECTED_INSTANCE_ID = "3908640733128066700"
MULTISEED_STAGE_CODES = frozenset(("e1", "e4"))
AUDIT_135M_STAGE_CODES = frozenset(
    ("a1d", "a1x", "a1c", "a3k", "a3r0", "a3x", "a4d", "a4b", "a4c", "a4x")
)
CONTROLLED_STAGE_CODES = MULTISEED_STAGE_CODES | AUDIT_135M_STAGE_CODES
AUDIT_135M_PREREG_SHA256 = (
    "bf66f7b07fb3b5a8ed531320e4ca505df51f75d4d6fd22aa4ab6edc3eafe7bed"
)
ALLOWED_STAGE_CODES = frozenset(
    ("p1r0", "p1ad", "p2", "p3t", *CONTROLLED_STAGE_CODES)
)
LOGICAL_SLOTS = ("v0", "v1", "v2", "v3")
AUDIT_135M_BLOCK_WIDTH = 3
AUDIT_135M_MAX_CONCURRENT_BLOCKS = 5
AUDIT_135M_LOGICAL_SLOTS = tuple(
    f"v{index}"
    for index in range(AUDIT_135M_BLOCK_WIDTH * AUDIT_135M_MAX_CONCURRENT_BLOCKS)
)
ALL_LOGICAL_SLOTS = AUDIT_135M_LOGICAL_SLOTS
AUDIT_135M_CONCURRENT_BLOCK_AMENDMENT_RAW_SHA256 = (
    "ac3fa833fd0cf1a985eca1500656217e2025d6d66dd642a1609cd3006405afb7"
)
REVISION_1_2_ALLOWED_ZONES = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
)
ALLOWED_US_A100_ZONES = (
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-b",
    "us-west1-b",
    "us-west4-a",
    "us-west4-b",
)
AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER = (
    "us-east1-b",
    "us-west4-b",
    "us-west1-b",
    "us-west4-a",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
)
# Compatibility alias for packet/controller code written against Version 1.1.
ALLOWED_US_CENTRAL1_ZONES = REVISION_1_2_ALLOWED_ZONES
ACCEPTED_GENERATION_AMENDMENT_RAW_SHA256S = frozenset(
    (
        REVISION_1_2_AMENDMENT_RAW_SHA256,
        AMENDMENT_RAW_SHA256,
    )
)
MAX_CONCURRENT_CELLS = 4
AUDIT_135M_MAX_CONCURRENT_CELLS = (
    AUDIT_135M_BLOCK_WIDTH * AUDIT_135M_MAX_CONCURRENT_BLOCKS
)
MAX_CAMPAIGN_A100S = 16
A100S_PER_VM = 4
SCIENTIFIC_VM_SHAPE = "a2-highgpu-4g"
FALLBACK_VM_SHAPE = "a2-highgpu-1g"
ALLOWED_SCIENTIFIC_VM_SHAPES = (SCIENTIFIC_VM_SHAPE, FALLBACK_VM_SHAPE)
A100S_PER_VM_BY_SHAPE = {
    SCIENTIFIC_VM_SHAPE: 4,
    FALLBACK_VM_SHAPE: 1,
}
GPU_ALLOCATION_MODE_BY_SHAPE = {
    SCIENTIFIC_VM_SHAPE: "one_learner_per_distinct_a100",
    FALLBACK_VM_SHAPE: "four_learners_packed_one_a100",
}
PACKING_EQUIVALENCE_LOSS = "2.105365492953676"
PACKING_EQUIVALENCE_NORMALIZED_COMMAND_HASH = (
    "155bf0801c2c8bfc71b81ada1f4f5dcb97f5a37395087603bc7aab6517b04faf"
)
RUN_ID_RE = re.compile(
    r"^bp-(p1r0|p1ad|p2|p3t|e1|e4|a1d|a1x|a1c|a3k|a3r0|a3x|a4d|a4b|a4c|a4x)-[0-9a-f]{16}-c[1-9][0-9]*-v(?:[0-9]|1[0-4])-g[1-9][0-9]*$"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
NUMERIC_ID_RE = re.compile(r"[0-9]+\Z")
ATTEMPT_STATUSES = frozenset(("COMPLETED", "DIVERGED", "INFRA_FAILURE", "FAILED"))
DIRECT_INFRASTRUCTURE_FAILURE_REASONS = frozenset(
    (
        "provider_spot_preemption",
        "vm_host_gpu_failure",
        "process_exit_before_scientific_divergence",
        "missing_or_checksum_invalid_required_artifact",
        "pre_unblinding_validator_provenance_failure",
    )
)
PEER_RETRY_REASON = "peer_block_invalidated_by_infra_failure"
FORBIDDEN_RETRY_REASONS = frozenset(
    ("poor_loss", "finite_completed_loss", "scientific_divergence", "post_unblinding_preference")
)
HORIZON_WORK = {
    16: {"outer_steps": 320, "per_fragment_outer_updates": 80},
    64: {"outer_steps": 80, "per_fragment_outer_updates": 20},
    256: {"outer_steps": 20, "per_fragment_outer_updates": 5},
}
EXPECTED_TOKENS = 655_360
EXPECTED_MICROSTEPS = 5_120
EXPECTED_LEARNERS = (0, 1, 2, 3)
EXPECTED_STEPS_PER_LEARNER = 1_280
FRAGMENT_COUNT = 4
BEST_PAPER_V2_FRESH_SEEDS = (383, 397, 409, 421, 433, 443, 457, 461)
BEST_PAPER_V2_ARMS = {
    16: frozenset(
        ((0.0, 0.021875), (0.9, 0.002734375), (0.9, 0.021875))
    ),
    256: frozenset(((0.0, 0.04375), (0.9, 0.0109375), (0.9, 0.04375))),
}


class ParallelPhaseMapError(RuntimeError):
    """Base class for deterministic schedule, lifecycle, or evidence failure."""


class ScheduleError(ParallelPhaseMapError):
    """The roster, plan, wave order, or retry lineage is not amendment-valid."""


class LifecycleError(ParallelPhaseMapError):
    """A VM generation is not exactly owned, bounded, or torn down."""


class EvidenceError(ParallelPhaseMapError):
    """Attempt artifacts do not prove the registered scientific work."""


class SealError(ParallelPhaseMapError):
    """A campaign cannot produce the one create-only scientific seal."""


AUDIT_EVALUATION_MODES = frozenset(
    {
        "development_endpoint",
        "confirmation_audit_pending",
        "development_prediction_pending",
        "capture_only_no_endpoint",
    }
)


def logical_slots_for_stage(stage_code: str) -> tuple[str, ...]:
    return (
        AUDIT_135M_LOGICAL_SLOTS
        if stage_code in AUDIT_135M_STAGE_CODES
        else LOGICAL_SLOTS
    )


def maximum_scientific_vms_for_stage(stage_code: str) -> int:
    return len(logical_slots_for_stage(stage_code))


def maximum_concurrent_cells_for_stage(stage_code: str) -> int:
    return (
        AUDIT_135M_MAX_CONCURRENT_CELLS
        if stage_code in AUDIT_135M_STAGE_CODES
        else MAX_CONCURRENT_CELLS
    )


def audit_checkpoint_only(cell: Mapping[str, Any]) -> bool:
    mode = cell.get("evaluation_mode", "development_endpoint")
    if mode not in AUDIT_EVALUATION_MODES:
        raise EvidenceError(f"unsupported audit evaluation mode {mode!r}")
    return mode != "development_endpoint"


def machine_shape_contract(machine_type: Any) -> dict[str, Any]:
    if machine_type not in ALLOWED_SCIENTIFIC_VM_SHAPES:
        raise LifecycleError("machine type is not an amendment-authorized scientific shape")
    shape = str(machine_type)
    return {
        "machine_type": shape,
        "a100_count": A100S_PER_VM_BY_SHAPE[shape],
        "gpu_slots": A100S_PER_VM_BY_SHAPE[shape],
        "gpu_allocation_mode": GPU_ALLOCATION_MODE_BY_SHAPE[shape],
    }


def region_for_zone(zone: Any) -> str:
    if zone not in ALLOWED_US_A100_ZONES:
        raise LifecycleError("zone is not in the amendment-authorized US A100 set")
    return str(zone).rsplit("-", 1)[0]


def normalized_workload_command(command: Sequence[str]) -> list[str]:
    """Mirror the preregistered hardware-only command normalization."""

    normalized: list[str] = []
    skip_value = False
    role_path_flags = {
        "--model": "<FROZEN_MODEL>",
        "--data": "<PREBOUND_TRAIN>",
        "--prebound-development-eval": "<PREBOUND_DEVELOPMENT_EVAL>",
    }
    for index, token in enumerate(command):
        if not isinstance(token, str):
            raise EvidenceError("scientific command must contain only string argv tokens")
        if skip_value:
            skip_value = False
            continue
        if token == "--gpu-slots":
            skip_value = True
            continue
        if token == "--require-distinct-learner-gpu-uuids":
            continue
        if token in role_path_flags:
            if index + 1 >= len(command):
                raise EvidenceError(f"{token} lacks a value in normalized command")
            normalized.extend((token, role_path_flags[token]))
            skip_value = True
            continue
        normalized.append(token)
    if skip_value:
        raise EvidenceError("--gpu-slots lacks a value in normalized command")
    return normalized


def project_scientific_command_for_machine_type(
    command: Sequence[str], machine_type: Any
) -> list[str]:
    """Apply only the Version 1.2 gpu-slot projection for a landed shape."""

    contract = machine_shape_contract(machine_type)
    projected = list(command)
    positions = [index for index, token in enumerate(projected) if token == "--gpu-slots"]
    if not positions and contract["machine_type"] == SCIENTIFIC_VM_SHAPE:
        return projected
    if len(positions) != 1 or positions[0] + 1 >= len(projected):
        raise EvidenceError("scientific command must contain exactly one --gpu-slots value")
    value_index = positions[0] + 1
    if projected[value_index] != "4":
        raise EvidenceError("frozen scientific command no longer records gpu_slots=4")
    if (
        contract["machine_type"] == FALLBACK_VM_SHAPE
        and "--require-distinct-learner-gpu-uuids" in projected
    ):
        raise EvidenceError("packed 1g execution cannot require distinct learner GPU UUIDs")
    projected[value_index] = str(contract["gpu_slots"])
    return projected


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard/non-finite JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes) -> Any:
    return json.loads(
        payload,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def load_json(path: Path, label: str = "JSON") -> Any:
    try:
        return strict_json_loads(path.read_bytes())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ParallelPhaseMapError(f"cannot read strict {label} {path}: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _pretty_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_bytes_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise SealError(f"create-only artifact already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_json_create_only(path: Path, value: Any) -> None:
    write_bytes_create_only(path, _pretty_json(value))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_pretty_json(value))
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ParallelPhaseMapError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ParallelPhaseMapError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ParallelPhaseMapError(f"{label} must be timezone-aware")
    return parsed


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ParallelPhaseMapError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def require_numeric_id(value: Any, label: str, *, allow_protected: bool = False) -> str:
    rendered = str(value or "")
    if NUMERIC_ID_RE.fullmatch(rendered) is None:
        raise LifecycleError(f"{label} must be an exact numeric provider ID")
    if not allow_protected and rendered == PROTECTED_INSTANCE_ID:
        raise LifecycleError(f"{label} is the protected instance ID")
    return rendered


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParallelPhaseMapError(f"{label} must be a positive integer")
    return value


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParallelPhaseMapError(f"{label} must be a non-negative integer")
    return value


def require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a finite IEEE-754 number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise EvidenceError(f"{label} must be finite")
    return normalized


def _utf8_sort(values: Iterable[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def expected_cell_ids_hash(cell_ids: Iterable[str]) -> str:
    return canonical_sha256(_utf8_sort(str(cell_id) for cell_id in cell_ids))


def verify_master_seed() -> None:
    if sha256_bytes(MASTER_SEED_PREIMAGE.encode("utf-8")) != MASTER_SEED_HEX:
        raise ScheduleError("parallel master-seed preimage does not reproduce the amendment")


def rank_digest(domain: str, study_id: str, token: str) -> bytes:
    if not all(isinstance(value, str) for value in (domain, study_id, token)):
        raise ScheduleError("rank inputs must be UTF-8 strings")
    return hashlib.sha256(
        bytes.fromhex(MASTER_SEED_HEX)
        + b"\x00"
        + domain.encode("utf-8")
        + b"\x00"
        + study_id.encode("utf-8")
        + b"\x00"
        + token.encode("utf-8")
    ).digest()


def rank_hex(domain: str, study_id: str, token: str) -> str:
    return rank_digest(domain, study_id, token).hex()


def rank_order(tokens: Iterable[str], domain: str, study_id: str) -> list[str]:
    unique = list(tokens)
    if len(set(unique)) != len(unique):
        raise ScheduleError(f"rank domain {domain!r} received duplicate tokens")
    return sorted(
        unique,
        key=lambda token: (rank_digest(domain, study_id, token), token.encode("utf-8")),
    )


def verify_amendment_bytes(path: Path) -> None:
    if sha256_file(path) != AMENDMENT_RAW_SHA256:
        raise ScheduleError("adopted parallel amendment bytes differ from the frozen hash")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParallelPhaseMapError(f"{label} must be a JSON object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ParallelPhaseMapError(f"{label} must be a JSON array")
    return value


def _cell_field(cell: Mapping[str, Any], lower: str, upper: str | None = None) -> Any:
    if lower in cell:
        return cell[lower]
    if upper is not None and upper in cell:
        return cell[upper]
    raise ScheduleError(f"cell {cell.get('cell_id')!r} lacks {lower!r}")


def _cell_learner_count(cell: Mapping[str, Any]) -> int:
    value = cell.get("m", cell.get("M", cell.get("expected_learner_count")))
    if value is None:
        target = cell.get("target_work")
        if isinstance(target, Mapping):
            value = target.get("learner_count")
    if value is None:
        return 4
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScheduleError(f"cell {cell.get('cell_id')!r} has invalid learner count")
    return value


def _scientific_cells(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _array(plan.get("cells"), "scientific plan cells")
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"scientific plan cells[{index}]")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in by_id:
            raise ScheduleError("scientific plan has a missing or duplicate cell_id")
        by_id[cell_id] = row
    return by_id


def _bound_cells(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _array(manifest.get("expected_cells"), "bound expected_cells")
    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    coordinates: set[tuple[int, int, float, float, int, int, str | None]] = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"bound expected_cells[{index}]")
        cell_id = row.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in by_id:
            raise ScheduleError("bound manifest has a missing or duplicate cell_id")
        coordinate = (
            int(_cell_field(row, "h", "H")),
            _cell_learner_count(row),
            float(_cell_field(row, "mu")),
            float(_cell_field(row, "eta")),
            int(_cell_field(row, "seed")),
            int(_cell_field(row, "training_seed")),
            str(row.get("analysis_role")) if row.get("analysis_role") is not None else None,
        )
        if coordinate in coordinates:
            raise ScheduleError(f"bound manifest duplicates coordinate {coordinate}")
        coordinates.add(coordinate)
        ordered.append(row)
        by_id[cell_id] = row
    return ordered, by_id


def _launch_ids_for_stage(
    stage_code: str,
    bound_rows: Sequence[Mapping[str, Any]],
    parent_rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    bound_ids = {str(row["cell_id"]) for row in bound_rows}
    parent_ids = {str(row["cell_id"]) for row in parent_rows}
    if stage_code == "p1r0":
        if len(bound_ids) != 36:
            raise ScheduleError("P1-R0 must bind exactly 36 launch cells")
        return bound_ids
    if stage_code in MULTISEED_STAGE_CODES:
        expected = 48 if stage_code == "e1" else 96
        if parent_ids:
            raise ScheduleError(f"{stage_code.upper()} multiseed binding must be parentless")
        if len(bound_ids) != expected:
            raise ScheduleError(
                f"{stage_code.upper()} must bind exactly {expected} launch cells"
            )
        return bound_ids
    if stage_code in AUDIT_135M_STAGE_CODES:
        if not parent_ids.issubset(bound_ids):
            raise ScheduleError("audit cumulative child omits a parent cell ID")
        launch_ids = bound_ids - parent_ids
        if not launch_ids:
            raise ScheduleError("audit launch suffix is empty")
        return launch_ids
    if not parent_ids.issubset(bound_ids):
        raise ScheduleError("cumulative child does not contain every parent cell ID")
    launch_ids = bound_ids - parent_ids
    if not launch_ids:
        raise ScheduleError(f"{stage_code} launch set is empty")
    return launch_ids


def build_parallel_roster(
    *,
    stage_code: str,
    bound_manifest: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    scientific_plan: Mapping[str, Any],
    expected_parent_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the exact ``parallel_roster_v1`` canonical preimage."""

    verify_master_seed()
    if stage_code not in ALLOWED_STAGE_CODES:
        raise ScheduleError(f"stage_code {stage_code!r} is outside the amendment scope")
    parent_hash = canonical_sha256(parent_manifest)
    if expected_parent_manifest_sha256 is not None and parent_hash != require_sha256(
        expected_parent_manifest_sha256, "expected parent manifest canonical SHA-256"
    ):
        raise ScheduleError("P0/parent manifest canonical hash differs from the bound parameter")
    lineage = _mapping(bound_manifest.get("lineage"), "bound lineage")
    if stage_code in MULTISEED_STAGE_CODES:
        if lineage.get("parent_manifest_sha256") is not None:
            raise ScheduleError("multiseed runtime contract must be parentless")
    elif lineage.get("parent_manifest_sha256") != parent_hash:
        raise ScheduleError("bound descendant is not tied to the supplied parent hash")
    prereg_hash = require_sha256(
        lineage.get("authoritative_prereg_template_sha256"),
        "authoritative preregistration template SHA-256",
    )
    expected_prereg_hash = (
        AUDIT_135M_PREREG_SHA256
        if stage_code in AUDIT_135M_STAGE_CODES
        else AUTHORITATIVE_PREREG_TEMPLATE_SHA256
    )
    if prereg_hash != expected_prereg_hash:
        raise ScheduleError("bound descendant cites a different authoritative preregistration")
    study_id = bound_manifest.get("study_id")
    descendant_kind = lineage.get("descendant_kind")
    if not isinstance(study_id, str) or not study_id:
        raise ScheduleError("bound descendant lacks study_id")
    if not isinstance(descendant_kind, str) or not descendant_kind:
        raise ScheduleError("bound descendant lacks descendant_kind")

    bound_rows, bound_by_id = _bound_cells(bound_manifest)
    parent_rows, _parent_by_id = _bound_cells(parent_manifest)
    launch_ids = _launch_ids_for_stage(stage_code, bound_rows, parent_rows)
    plan_by_id = _scientific_cells(scientific_plan)
    if set(plan_by_id) != launch_ids:
        raise ScheduleError(
            "scientific plan cells must equal the derived launch set exactly"
        )
    frozen = _mapping(bound_manifest.get("frozen"), "bound frozen identities")
    command_registry = _mapping(
        frozen.get("cell_command_hashes"), "bound cell command registry"
    )
    if set(command_registry) != {str(row["cell_id"]) for row in bound_rows}:
        raise ScheduleError("bound command registry keys do not equal cumulative expected cells")

    launch_cells: list[dict[str, Any]] = []
    for cell_id in _utf8_sort(launch_ids):
        bound = bound_by_id[cell_id]
        planned = plan_by_id[cell_id]
        block_id = _cell_field(bound, "block_id")
        planned_randomization = _mapping(
            planned.get("randomization"), f"scientific cell {cell_id} randomization"
        )
        command_hash = require_sha256(bound.get("command_hash"), f"{cell_id} command hash")
        if (
            command_registry.get(cell_id) != command_hash
            or planned.get("command_hash") != command_hash
            or planned_randomization.get("block_id") != block_id
        ):
            raise ScheduleError(f"cell {cell_id} command/block identity differs across inputs")
        if stage_code in CONTROLLED_STAGE_CODES and (
            _cell_learner_count(planned) != _cell_learner_count(bound)
            or planned.get("pairing_identity_hash")
            != bound.get("pairing_identity_hash")
            or planned.get("pairing_command_hash")
            != bound.get("pairing_command_hash")
            or planned.get("evaluation_mode", "development_endpoint")
            != bound.get("evaluation_mode", "development_endpoint")
            or planned.get("finite_kernel_capture_required", False)
            != bound.get("finite_kernel_capture_required", False)
        ):
            raise ScheduleError(
                f"cell {cell_id} M or pairing identity differs across inputs"
            )
        normalized_hash = require_sha256(
            bound.get("normalized_workload_command_hash"),
            f"{cell_id} normalized workload command hash",
        )
        if planned.get("normalized_workload_command_hash") not in (None, normalized_hash):
            raise ScheduleError(f"cell {cell_id} normalized command hash differs")
        planned_command = _array(planned.get("command"), f"{cell_id} scientific command")
        if canonical_sha256(planned_command) != command_hash:
            raise ScheduleError(f"cell {cell_id} command bytes do not reproduce command_hash")
        launch_cells.append(
            {
                "cell_id": cell_id,
                "block_id": block_id,
                "h": int(_cell_field(bound, "h", "H")),
                "m": _cell_learner_count(bound),
                "mu": float(_cell_field(bound, "mu")),
                "eta": float(_cell_field(bound, "eta")),
                "seed": int(_cell_field(bound, "seed")),
                "training_seed": int(_cell_field(bound, "training_seed")),
                "paired_control_id": bound.get("paired_control_id"),
                "pairing_identity_hash": bound.get("pairing_identity_hash"),
                "pairing_command_hash": bound.get("pairing_command_hash"),
                "expected_learner_count": bound.get(
                    "expected_learner_count", _cell_learner_count(bound)
                ),
                "expected_learner_steps": bound.get("expected_learner_steps"),
                "audit_stage": bound.get("audit_stage"),
                "audit_phase": bound.get("audit_phase"),
                "analysis_role": bound.get("analysis_role"),
                "pair_key": bound.get("pair_key"),
                "evaluation_mode": bound.get(
                    "evaluation_mode", "development_endpoint"
                ),
                "finite_kernel_capture_required": bound.get(
                    "finite_kernel_capture_required", False
                ),
                "terminal_partial_window_registered": bound.get(
                    "terminal_partial_window_registered", False
                ),
                "terminal_partial_window_microsteps": bound.get(
                    "terminal_partial_window_microsteps", 0
                ),
                "command_hash": command_hash,
                "normalized_workload_command_hash": normalized_hash,
            }
        )
    roster = {
        "schema": "parallel_roster_v1",
        "stage_code": stage_code,
        "study_id": study_id,
        "descendant_kind": descendant_kind,
        "authoritative_prereg_template_sha256": prereg_hash,
        "parent_manifest_sha256": parent_hash,
        "parent_expected_cell_ids_hash": expected_cell_ids_hash(
            str(row["cell_id"]) for row in parent_rows
        ),
        "cumulative_expected_cell_ids_hash": expected_cell_ids_hash(
            str(row["cell_id"]) for row in bound_rows
        ),
        "launch_cells": launch_cells,
    }
    if stage_code in MULTISEED_STAGE_CODES:
        design_hash = require_sha256(
            lineage.get("best_paper_v2_design_contract_hash"),
            "best-paper v2 design contract hash",
        )
        if bound_manifest.get("launch_authorized") is not False:
            raise ScheduleError(
                "local multiseed contract must not claim launch authorization"
            )
        roster["best_paper_v2_design_contract_hash"] = design_hash
        roster["launch_authorized"] = False
    elif stage_code in AUDIT_135M_STAGE_CODES:
        design_hash = require_sha256(
            lineage.get("audit_135m_design_contract_hash"),
            "audit-135m design contract hash",
        )
        contract = _mapping(
            bound_manifest.get("audit_135m_contract"), "audit-135m contract"
        )
        launch_count = require_positive_int(
            contract.get("launch_cell_count"), "audit launch cell count"
        )
        hard_ceiling = require_finite(
            contract.get("hard_ceiling_usd"), "audit hard ceiling"
        )
        if (
            launch_count != len(launch_cells)
            or hard_ceiling <= 0.0
            or bound_manifest.get("launch_authorized") is not False
        ):
            raise ScheduleError("audit contract count/ceiling/authorization differs")
        roster.update(
            {
                "audit_135m_design_contract_hash": design_hash,
                "audit_stage": contract.get("audit_stage"),
                "audit_phase": contract.get("audit_phase"),
                "hard_ceiling_usd": hard_ceiling,
                "launch_cell_count": launch_count,
                "launch_authorized": False,
            }
        )
    expected_fields = {
        "schema",
        "stage_code",
        "study_id",
        "descendant_kind",
        "authoritative_prereg_template_sha256",
        "parent_manifest_sha256",
        "parent_expected_cell_ids_hash",
        "cumulative_expected_cell_ids_hash",
        "launch_cells",
    }
    if stage_code in MULTISEED_STAGE_CODES:
        expected_fields.update(
            ("best_paper_v2_design_contract_hash", "launch_authorized")
        )
    elif stage_code in AUDIT_135M_STAGE_CODES:
        expected_fields.update(
            (
                "audit_135m_design_contract_hash",
                "audit_stage",
                "audit_phase",
                "hard_ceiling_usd",
                "launch_cell_count",
                "launch_authorized",
            )
        )
    if set(roster) != expected_fields:
        raise AssertionError("parallel roster field set drifted")
    return roster


def roster_hash(roster: Mapping[str, Any]) -> str:
    if roster.get("schema") != "parallel_roster_v1":
        raise ScheduleError("parallel roster has the wrong schema")
    return canonical_sha256(roster)


def _group_launch_cells(
    roster: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    stage_code = str(roster.get("stage_code"))
    rows = _array(roster.get("launch_cells"), "roster launch_cells")
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        cell = _mapping(raw, "roster launch cell")
        if stage_code == "p3t":
            group_id = f"p3t-seed-{int(cell['seed'])}"
        else:
            group_id = str(cell.get("block_id", ""))
        if not group_id:
            raise ScheduleError("launch cell lacks a group ID")
        by_group[group_id].append(cell)

    if stage_code in ("p1r0", "p1ad", "p2"):
        for group_id, cells in by_group.items():
            if len(cells) != 3:
                raise ScheduleError(f"{group_id} must contain exactly three cells")
            if {float(cell["mu"]) for cell in cells} != {0.0, 0.5, 0.9}:
                raise ScheduleError(f"{group_id} does not contain mu={{0,.5,.9}}")
            for field in ("h", "eta", "seed", "training_seed"):
                if len({cell[field] for cell in cells}) != 1:
                    raise ScheduleError(f"{group_id} has mismatched {field}")
        if stage_code == "p1r0" and (len(rows), len(by_group)) != (36, 12):
            raise ScheduleError("P1-R0 must materialize 12 three-cell waves")
    elif stage_code in MULTISEED_STAGE_CODES:
        expected_m = {4} if stage_code == "e1" else {1, 16}
        for group_id, cells in by_group.items():
            if len(cells) != 3 or len({str(cell["cell_id"]) for cell in cells}) != 3:
                raise ScheduleError(f"{group_id} must contain exactly three distinct arms")
            for field in ("h", "m", "seed", "training_seed"):
                if len({cell[field] for cell in cells}) != 1:
                    raise ScheduleError(f"{group_id} has mismatched {field}")
            h = int(cells[0]["h"])
            if h not in BEST_PAPER_V2_ARMS:
                raise ScheduleError(f"{group_id} has unauthorized H={h}")
            observed_arms = frozenset(
                (float(cell["mu"]), float(cell["eta"])) for cell in cells
            )
            if observed_arms != BEST_PAPER_V2_ARMS[h]:
                raise ScheduleError(
                    f"{group_id} does not contain the frozen mixed-eta E1/E4 arms"
                )
            pairing_hashes = {cell.get("pairing_identity_hash") for cell in cells}
            pairing_commands = {cell.get("pairing_command_hash") for cell in cells}
            if (
                len(pairing_hashes) != 1
                or None in pairing_hashes
                or len(pairing_commands) != 1
                or None in pairing_commands
            ):
                raise ScheduleError(f"{group_id} breaks a paired-block identity")
            controls = [cell for cell in cells if float(cell["mu"]) == 0.0]
            if len(controls) != 1 or {
                cell.get("paired_control_id") for cell in cells
            } != {controls[0]["cell_id"]}:
                raise ScheduleError(f"{group_id} lacks one shared live control")
        observed_seeds = {int(cell["seed"]) for cell in rows}
        observed_m = {int(cell["m"]) for cell in rows}
        expected_cells = 48 if stage_code == "e1" else 96
        expected_groups = 16 if stage_code == "e1" else 32
        if observed_seeds != set(BEST_PAPER_V2_FRESH_SEEDS):
            raise ScheduleError(f"{stage_code.upper()} seed registry differs")
        if observed_m != expected_m:
            raise ScheduleError(f"{stage_code.upper()} M variants differ")
        if (len(rows), len(by_group)) != (expected_cells, expected_groups):
            raise ScheduleError(
                f"{stage_code.upper()} must materialize {expected_groups} atomic seed blocks"
            )
    elif stage_code in AUDIT_135M_STAGE_CODES:
        expected_roles = {
            "a1d": {
                "tune_control_rank0",
                "tune_method_rank0",
                "tune_control_rank1",
                "tune_method_rank1",
                "tune_control_rank2",
                "tune_method_rank2",
            },
            "a1x": {"boundary_control", "boundary_method"},
            "a1c": {"fixed_control", "fixed_method", "tuned_control", "tuned_method"},
            "a3k": {"kernel_recapture"},
            "a3x": {"frontier_boundary_extension"},
            "a4d": {
                "tune_control_rank0",
                "tune_method_rank0",
                "tune_control_rank1",
                "tune_method_rank1",
                "tune_control_rank2",
                "tune_method_rank2",
            },
            "a4b": {"boundary_control", "boundary_method"},
            "a4c": {"fixed_control", "fixed_method", "tuned_control", "tuned_method"},
            "a4x": {"fixed_control", "fixed_method", "tuned_control", "tuned_method"},
        }
        expected_seeds = {
            "a1d": {359, 373},
            "a1x": {359, 373},
            "a1c": {383, 397, 409, 421, 433, 443, 457, 461},
            "a3k": {347},
            "a3r0": {359, 373},
            "a3x": {359, 373},
            "a4d": {2069, 2081},
            "a4b": {2069, 2081},
            "a4c": {2083, 2087, 2089},
            "a4x": {2099, 2111, 2113},
        }[stage_code]
        expected_m = {
            "a1d": {4},
            "a1x": {4},
            "a1c": {4},
            "a3k": {4},
            "a3r0": {4},
            "a3x": {4},
            "a4d": {1, 4},
            "a4b": {1, 4},
            "a4c": {1, 4},
            "a4x": {1, 4},
        }[stage_code]
        for group_id, cells in by_group.items():
            if len({str(cell["cell_id"]) for cell in cells}) != len(cells):
                raise ScheduleError(f"{group_id} contains a duplicate audit arm")
            for field in ("h", "m", "seed", "training_seed"):
                if len({cell[field] for cell in cells}) != 1:
                    raise ScheduleError(f"{group_id} has mismatched audit {field}")
            if len({cell.get("pairing_identity_hash") for cell in cells}) != 1:
                raise ScheduleError(f"{group_id} breaks its audit pairing identity")
            if len({cell.get("pairing_command_hash") for cell in cells}) != 1:
                raise ScheduleError(f"{group_id} changes a non-treatment command field")
            by_id = {str(cell["cell_id"]): cell for cell in cells}
            for cell in cells:
                control = by_id.get(str(cell.get("paired_control_id")))
                if control is None or float(control["mu"]) != 0.0:
                    raise ScheduleError(f"{group_id} has an invalid live control")
            roles = {str(cell.get("analysis_role")) for cell in cells}
            if stage_code == "a3r0":
                expected = {
                    f"frontier_eta_{index}"
                    for index in range(4 if int(cells[0]["h"]) == 8 else 5)
                }
                if roles != expected:
                    raise ScheduleError(f"{group_id} has the wrong A3 frontier roles")
            elif roles != expected_roles[stage_code]:
                raise ScheduleError(f"{group_id} has the wrong audit analysis roles")
        if {int(cell["seed"]) for cell in rows} != expected_seeds:
            raise ScheduleError(f"{stage_code} seed registry differs")
        if {int(cell["m"]) for cell in rows} != expected_m:
            raise ScheduleError(f"{stage_code} learner-count registry differs")
        if len(rows) != require_positive_int(
            roster.get("launch_cell_count"), "audit roster launch count"
        ):
            raise ScheduleError("audit roster launch count differs from its cells")
        fixed_counts = {
            "a1d": 24,
            "a1c": 64,
            "a3k": 3,
            "a3r0": 18,
            "a4d": 48,
            "a4c": 48,
            "a4x": 48,
        }
        if stage_code in fixed_counts and len(rows) != fixed_counts[stage_code]:
            raise ScheduleError(f"{stage_code} must launch exactly {fixed_counts[stage_code]} cells")
        if stage_code == "a1x" and len(rows) not in (4, 8):
            raise ScheduleError("A1 boundary extension must add four or eight cells")
        if stage_code == "a3x" and len(rows) not in (2, 4):
            raise ScheduleError("A3 boundary extension must add two or four cells")
        if stage_code == "a4b" and (len(rows) < 4 or len(rows) > 16 or len(rows) % 4):
            raise ScheduleError("A4 boundary extension must add 4..16 cells by complete pairs")
    elif stage_code == "p3t":
        for group_id, cells in by_group.items():
            if len(cells) != 4 or len({cell["cell_id"] for cell in cells}) != 4:
                raise ScheduleError(f"{group_id} must contain exactly four distinct cells")
            if len({int(cell["seed"]) for cell in cells}) != 1:
                raise ScheduleError(f"{group_id} mixes shuffle seeds")
            pair_ids = [cell.get("paired_control_id") for cell in cells]
            if any(not isinstance(value, str) or not value for value in pair_ids):
                raise ScheduleError(f"{group_id} lacks frozen paired-control identities")
        if (len(rows), len(by_group)) != (32, 8):
            raise ScheduleError("P3 training must materialize eight four-cell seed waves")
    return dict(by_group)


def build_legacy_wave_assignment(
    *,
    study_id: str,
    group_id: str,
    cells: Sequence[Mapping[str, Any]],
    retry_round: int,
    wave_index: int,
    time_block_index: int,
) -> dict[str, Any]:
    require_positive_int(retry_round, "retry round")
    cell_ids = [str(cell["cell_id"]) for cell in cells]
    arm_domain = f"arm-order|{group_id}|{retry_round}"
    slot_domain = f"slot-order|{group_id}|{retry_round}"
    launch_domain = f"launch-order|{group_id}|{retry_round}"
    arm_order = rank_order(cell_ids, arm_domain, study_id)
    slot_order = rank_order(LOGICAL_SLOTS, slot_domain, study_id)
    assignment = {
        cell_id: slot_order[index % len(slot_order)]
        for index, cell_id in enumerate(arm_order)
    }
    launch_order = rank_order(cell_ids, launch_domain, study_id)
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    assigned = []
    for launch_index, cell_id in enumerate(launch_order):
        slot = assignment[cell_id]
        cell = by_id[cell_id]
        assigned.append(
            {
                "cell_id": cell_id,
                "logical_slot": slot,
                "arm_order_index": arm_order.index(cell_id),
                "launch_order_index": launch_index,
                "command_hash": cell["command_hash"],
                "m": _cell_learner_count(cell),
                "pairing_identity_hash": cell.get("pairing_identity_hash"),
                "arm_rank": rank_hex(arm_domain, study_id, cell_id),
                "slot_rank": rank_hex(slot_domain, study_id, slot),
                "launch_rank": rank_hex(launch_domain, study_id, cell_id),
            }
        )
    active_slots = {row["logical_slot"] for row in assigned}
    idle_slots = [slot for slot in LOGICAL_SLOTS if slot not in active_slots]
    return {
        "group_id": group_id,
        "wave_index": wave_index,
        "time_block_index": time_block_index,
        "retry_round": retry_round,
        "group_rank": rank_hex("wave", study_id, group_id),
        "assigned_cells_in_dispatch_order": assigned,
        "idle_slots": idle_slots,
        "dispatch_span_limit_seconds": 60,
        "scientific_start_span_limit_seconds": 120,
    }


def build_legacy_parallel_plan(
    roster: Mapping[str, Any], *, expected_roster_hash: str | None = None
) -> dict[str, Any]:
    actual_roster_hash = roster_hash(roster)
    if expected_roster_hash is not None and actual_roster_hash != require_sha256(
        expected_roster_hash, "expected roster hash"
    ):
        raise ScheduleError("parallel roster hash differs from the bound value")
    stage_code = str(roster["stage_code"])
    study_id = str(roster["study_id"])
    by_group = _group_launch_cells(roster)
    group_ids = rank_order(by_group, "wave", study_id)
    waves = [
        build_legacy_wave_assignment(
            study_id=study_id,
            group_id=group_id,
            cells=by_group[group_id],
            retry_round=1,
            wave_index=index,
            time_block_index=index,
        )
        for index, group_id in enumerate(group_ids)
    ]
    return {
        "schema": "yeto_parallel_plan_v1",
        "stage_code": stage_code,
        "study_id": study_id,
        "roster_hash": actual_roster_hash,
        "master_seed": MASTER_SEED_HEX,
        "logical_slots": list(LOGICAL_SLOTS),
        "capacity": {
            "maximum_concurrent_scientific_cells": MAX_CONCURRENT_CELLS,
            "maximum_campaign_owned_attached_a100s": MAX_CAMPAIGN_A100S,
            "a100s_per_scientific_vm": A100S_PER_VM,
            "scientific_vm_shape": SCIENTIFIC_VM_SHAPE,
            "active_scientific_cells_per_vm": 1,
        },
        "retry_derivation": {
            "whole_atomic_group": True,
            "domains": [
                "arm-order|<group_id>|<retry_round>",
                "slot-order|<group_id>|<retry_round>",
                "launch-order|<group_id>|<retry_round>",
            ],
            "fresh_attempt_directories": True,
            "resume_forbidden": True,
            "retry_inserted_immediately": True,
        },
        "waves": waves,
    }


def _normalize_available_slots(
    available_slots: Iterable[str],
    *,
    logical_slots: Sequence[str] = LOGICAL_SLOTS,
) -> tuple[str, ...]:
    allowed = tuple(logical_slots)
    requested = set(available_slots)
    values = tuple(slot for slot in allowed if slot in requested)
    if (
        not values
        or len(values) != len(requested)
        or any(slot not in allowed for slot in requested)
    ):
        raise ScheduleError(
            "available slot set must be a nonempty subset of the plan logical slots"
        )
    return values


def available_slot_subsets(
    logical_slots: Sequence[str] = LOGICAL_SLOTS,
) -> tuple[tuple[str, ...], ...]:
    slots = tuple(logical_slots)
    return tuple(
        subset
        for width in range(1, len(slots) + 1)
        for subset in itertools.combinations(slots, width)
    )


def audit_concurrent_block_slots(
    *,
    contract_hash: str,
    block_index: int,
    available_slots: Iterable[str],
) -> tuple[str, ...]:
    """Bind one audit block to a deterministic three-slot lane.

    The only scientific-schedule inputs are the frozen audit contract hash,
    the planned block index, and the exact loss-blind slot set made available
    by the capacity controller.  Batches are contiguous in planned-block
    order, contain at most five blocks, and use exactly three disjoint slots
    per block.
    """

    contract = require_sha256(contract_hash, "audit design contract hash")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ScheduleError("audit concurrent block index must be nonnegative")
    slots = _normalize_available_slots(
        available_slots, logical_slots=AUDIT_135M_LOGICAL_SLOTS
    )
    if (
        len(slots) % AUDIT_135M_BLOCK_WIDTH
        or len(slots) > AUDIT_135M_MAX_CONCURRENT_CELLS
    ):
        raise ScheduleError(
            "audit concurrent slot set must contain one to five complete three-slot lanes"
        )
    block_count = len(slots) // AUDIT_135M_BLOCK_WIDTH
    slot_order = rank_order(
        slots,
        f"audit-concurrent-slot-order|{contract}|{','.join(slots)}",
        contract,
    )
    lane_order = rank_order(
        [str(index) for index in range(block_count)],
        f"audit-concurrent-lane-order|{contract}|{','.join(slots)}",
        contract,
    )
    lane_index = int(lane_order[block_index % block_count])
    start = lane_index * AUDIT_135M_BLOCK_WIDTH
    return _normalize_available_slots(
        slot_order[start : start + AUDIT_135M_BLOCK_WIDTH],
        logical_slots=AUDIT_135M_LOGICAL_SLOTS,
    )


def build_wave_assignment(
    *,
    study_id: str,
    roster_digest: str,
    group_id: str,
    cells: Sequence[Mapping[str, Any]],
    available_slots: Iterable[str],
    retry_round: int,
    wave_index: int,
    time_block_index: int,
    logical_slots: Sequence[str] = LOGICAL_SLOTS,
) -> dict[str, Any]:
    """Build the revision-2 deterministic reduced-width assignment.

    The binding key is exactly the roster hash, canonical available-slot set,
    planned wave index, and retry round.  Scientific outcomes never enter the
    function.  When the available width is smaller than the atomic group, cells
    are split into deterministic dispatch batches with at most one cell per
    logical slot in a batch; the complete group remains loss-blind until every
    batch is terminal.
    """

    require_positive_int(retry_round, "retry round")
    roster_value = require_sha256(roster_digest, "roster hash")
    slot_universe = tuple(logical_slots)
    slots = _normalize_available_slots(
        available_slots, logical_slots=slot_universe
    )
    subset_token = ",".join(slots)
    binding_domain = (
        f"available-slot-binding-v2|{roster_value}|{subset_token}|"
        f"{wave_index}|{retry_round}"
    )
    cell_ids = [str(cell["cell_id"]) for cell in cells]
    arm_domain = binding_domain + "|arm"
    slot_domain = binding_domain + "|slot"
    launch_domain = binding_domain + "|launch"
    arm_order = rank_order(cell_ids, arm_domain, study_id)
    slot_order = rank_order(slots, slot_domain, study_id)
    launch_order = rank_order(cell_ids, launch_domain, study_id)
    arm_position = {cell_id: index for index, cell_id in enumerate(arm_order)}
    assignment = {
        cell_id: slot_order[arm_position[cell_id] % len(slot_order)]
        for cell_id in cell_ids
    }
    batch_index = {
        cell_id: arm_position[cell_id] // len(slot_order) for cell_id in cell_ids
    }
    by_id = {str(cell["cell_id"]): cell for cell in cells}
    launch_position = {cell_id: index for index, cell_id in enumerate(launch_order)}
    ordered_ids = sorted(
        cell_ids,
        key=lambda cell_id: (
            batch_index[cell_id],
            launch_position[cell_id],
            cell_id.encode("utf-8"),
        ),
    )
    batch_positions: dict[int, int] = defaultdict(int)
    assigned = []
    for launch_index, cell_id in enumerate(ordered_ids):
        slot = assignment[cell_id]
        batch = batch_index[cell_id]
        within_batch = batch_positions[batch]
        batch_positions[batch] += 1
        cell = by_id[cell_id]
        assigned.append(
            {
                "cell_id": cell_id,
                "logical_slot": slot,
                "available_slot_set": list(slots),
                "dispatch_batch_index": batch,
                "batch_launch_order_index": within_batch,
                "arm_order_index": arm_position[cell_id],
                "launch_order_index": launch_index,
                "command_hash": cell["command_hash"],
                "m": _cell_learner_count(cell),
                "pairing_identity_hash": cell.get("pairing_identity_hash"),
                "binding_domain": binding_domain,
                "arm_rank": rank_hex(arm_domain, study_id, cell_id),
                "slot_rank": rank_hex(slot_domain, study_id, slot),
                "launch_rank": rank_hex(launch_domain, study_id, cell_id),
            }
        )
    unavailable = [slot for slot in slot_universe if slot not in slots]
    unused_available = [
        slot for slot in slots if slot not in {row["logical_slot"] for row in assigned}
    ]
    return {
        "group_id": group_id,
        "wave_index": wave_index,
        "time_block_index": time_block_index,
        "retry_round": retry_round,
        "available_slot_set": list(slots),
        "available_width": len(slots),
        "binding_key": {
            "roster_hash": roster_value,
            "available_slot_set": list(slots),
            "wave_index": wave_index,
            "retry_round": retry_round,
        },
        "group_rank": rank_hex("wave", study_id, group_id),
        "assigned_cells_in_dispatch_order": assigned,
        "dispatch_batch_count": max(batch_index.values(), default=0) + 1,
        "idle_slots": unavailable + unused_available,
        "unavailable_slots": unavailable,
        "dispatch_span_limit_seconds_per_batch": 60,
        "scientific_start_span_limit_seconds_per_batch": 120,
        "whole_group_loss_blind_until_terminal": True,
        "runtime_pairing_requirements": {
            "uniform_machine_type_within_block": True,
            "uniform_gpu_slots_within_block": True,
            "learner_gpu_assignment": "learner_id mod gpu_slots",
            "same_pairing_identity_hash": len(
                {cell.get("pairing_identity_hash") for cell in cells}
            )
            == 1,
        },
    }


def build_revision_1_1_parallel_plan(
    roster: Mapping[str, Any], *, expected_roster_hash: str | None = None
) -> dict[str, Any]:
    actual_roster_hash = roster_hash(roster)
    if expected_roster_hash is not None and actual_roster_hash != require_sha256(
        expected_roster_hash, "expected roster hash"
    ):
        raise ScheduleError("parallel roster hash differs from the bound value")
    stage_code = str(roster["stage_code"])
    study_id = str(roster["study_id"])
    by_group = _group_launch_cells(roster)
    group_ids = rank_order(by_group, "wave", study_id)
    legacy = build_legacy_parallel_plan(
        roster, expected_roster_hash=actual_roster_hash
    )

    def waves_for(slots: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            build_wave_assignment(
                study_id=study_id,
                roster_digest=actual_roster_hash,
                group_id=group_id,
                cells=by_group[group_id],
                available_slots=slots,
                retry_round=1,
                wave_index=index,
                time_block_index=index,
            )
            for index, group_id in enumerate(group_ids)
        ]

    variants = [
        {
            "available_slot_set": list(slots),
            "available_width": len(slots),
            "waves": waves_for(slots),
        }
        for slots in available_slot_subsets()
    ]
    full_width = next(
        variant for variant in variants if variant["available_slot_set"] == list(LOGICAL_SLOTS)
    )
    audit_capacity = stage_code in AUDIT_135M_STAGE_CODES
    return {
        "schema": "yeto_parallel_plan_v2",
        "stage_code": stage_code,
        "study_id": study_id,
        "roster_hash": actual_roster_hash,
        "master_seed": MASTER_SEED_HEX,
        "logical_slots": list(LOGICAL_SLOTS),
        "supersedes_parallel_plan_hash": canonical_sha256(legacy),
        "superseded_full_width_waves": legacy["waves"],
        "binding_function": {
            "revision": 2,
            "pure_inputs": [
                "roster_hash",
                "available_slot_set",
                "wave_index",
                "retry_round",
            ],
            "outcome_inputs_forbidden": True,
            "available_slot_sets_materialized": len(available_slot_subsets()),
            "reduced_width_dispatch_batches": True,
        },
        "capacity": {
            "minimum_available_scientific_vms": 1,
            "maximum_available_scientific_vms": 4,
            "maximum_concurrent_scientific_cells": MAX_CONCURRENT_CELLS,
            "maximum_campaign_owned_attached_a100s": MAX_CAMPAIGN_A100S,
            "a100s_per_scientific_vm": A100S_PER_VM,
            "scientific_vm_shape": SCIENTIFIC_VM_SHAPE,
            "active_scientific_cells_per_vm": 1,
            "allowed_zones": list(
                AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER
                if audit_capacity
                else REVISION_1_2_ALLOWED_ZONES
            ),
            **(
                {
                    "zone_fallback_order": list(
                        AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER
                    )
                }
                if audit_capacity
                else {}
            ),
            "quota_scope": (
                "provider-recorded per-region preemptible A100 quota"
                if audit_capacity
                else "us-central1 regional preemptible A100 quota"
            ),
        },
        "retry_derivation": {
            "whole_atomic_group": True,
            "binding_key_recomputed_for_retry_round": True,
            "fresh_attempt_directories": True,
            "resume_forbidden": True,
            "retry_inserted_immediately": True,
        },
        "available_slot_variants": variants,
        "waves": full_width["waves"],
    }


def build_parallel_plan(
    roster: Mapping[str, Any], *, expected_roster_hash: str | None = None
) -> dict[str, Any]:
    """Build the Version 1.2 plan without changing Version 1.1 assignments.

    The reduced-width binding domains and every wave byte remain those of the
    pre-outcome Version 1.1 plan. Version 1.2 changes only the reviewed capacity
    contract by admitting a provider-recorded 1g packing fallback.
    """

    previous = build_revision_1_1_parallel_plan(
        roster, expected_roster_hash=expected_roster_hash
    )
    plan = deepcopy(previous)
    plan["schema"] = "yeto_parallel_plan_v3"
    plan["supersedes_revision_1_1_parallel_plan_hash"] = canonical_sha256(previous)
    audit_capacity = str(roster.get("stage_code")) in AUDIT_135M_STAGE_CODES
    plan["capacity"] = {
        "contract_revision": 2,
        "minimum_available_scientific_vms": 1,
        "maximum_available_scientific_vms": 4,
        "maximum_concurrent_scientific_cells": MAX_CONCURRENT_CELLS,
        "maximum_campaign_owned_attached_a100s": MAX_CAMPAIGN_A100S,
        "preferred_scientific_vm_shape": SCIENTIFIC_VM_SHAPE,
        "fallback_scientific_vm_shape": FALLBACK_VM_SHAPE,
        "shape_fallback_order": list(ALLOWED_SCIENTIFIC_VM_SHAPES),
        "fallback_trigger": "provider_capacity_stockout_before_creation",
        "a100s_per_scientific_vm_by_shape": dict(A100S_PER_VM_BY_SHAPE),
        "gpu_allocation_mode_by_shape": dict(GPU_ALLOCATION_MODE_BY_SHAPE),
        "active_scientific_cells_per_vm": 1,
        "allowed_zones": list(
            AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER
            if audit_capacity
            else REVISION_1_2_ALLOWED_ZONES
        ),
        **(
            {
                "zone_fallback_order": list(
                    AUDIT_135M_SURVIVAL_WEIGHTED_ZONE_ORDER
                )
            }
            if audit_capacity
            else {}
        ),
        "quota_scope": (
            "provider-recorded per-region preemptible A100 quota"
            if audit_capacity
            else "us-central1 regional preemptible A100 quota"
        ),
        "packing_equivalence_evidence": {
            "p0a_machine_type": FALLBACK_VM_SHAPE,
            "p0a_gpu_slots": 1,
            "p0b_machine_type": SCIENTIFIC_VM_SHAPE,
            "p0b_gpu_slots": 4,
            "mu0_bit_identical_loss": PACKING_EQUIVALENCE_LOSS,
            "normalized_workload_command_hash": (
                PACKING_EQUIVALENCE_NORMALIZED_COMMAND_HASH
            ),
        },
        "learner_counts": sorted(
            {_cell_learner_count(cell) for cell in roster["launch_cells"]}
        ),
        "maximum_learners_per_gpu_by_shape": {
            str(count): {
                SCIENTIFIC_VM_SHAPE: math.ceil(
                    count / A100S_PER_VM_BY_SHAPE[SCIENTIFIC_VM_SHAPE]
                ),
                FALLBACK_VM_SHAPE: count,
            }
            for count in sorted(
                {_cell_learner_count(cell) for cell in roster["launch_cells"]}
            )
        },
    }
    if audit_capacity:
        revision_1_2_hash = canonical_sha256(plan)
        by_group = _group_launch_cells(roster)
        contract_hash = require_sha256(
            roster.get("audit_135m_design_contract_hash"),
            "audit-135M design contract hash",
        )
        concurrent_waves: list[dict[str, Any]] = []
        batch_rows: list[dict[str, Any]] = []
        wave_count = len(plan["waves"])
        for batch_start in range(0, wave_count, AUDIT_135M_MAX_CONCURRENT_BLOCKS):
            batch_size = min(
                AUDIT_135M_MAX_CONCURRENT_BLOCKS, wave_count - batch_start
            )
            batch_slots = AUDIT_135M_LOGICAL_SLOTS[
                : batch_size * AUDIT_135M_BLOCK_WIDTH
            ]
            assignments = []
            for block_index in range(batch_start, batch_start + batch_size):
                prior_wave = plan["waves"][block_index]
                group_id = str(prior_wave["group_id"])
                lane = audit_concurrent_block_slots(
                    contract_hash=contract_hash,
                    block_index=block_index,
                    available_slots=batch_slots,
                )
                wave = build_wave_assignment(
                    study_id=str(roster["study_id"]),
                    roster_digest=roster_hash(roster),
                    group_id=group_id,
                    cells=by_group[group_id],
                    available_slots=lane,
                    retry_round=1,
                    wave_index=block_index,
                    time_block_index=block_index,
                    logical_slots=AUDIT_135M_LOGICAL_SLOTS,
                )
                wave["audit_concurrent_block_index"] = block_index
                wave["audit_registered_concurrent_batch_index"] = (
                    batch_start // AUDIT_135M_MAX_CONCURRENT_BLOCKS
                )
                wave["audit_registered_batch_slot_set"] = list(batch_slots)
                concurrent_waves.append(wave)
                assignments.append(
                    {
                        "block_index": block_index,
                        "group_id": group_id,
                        "logical_slots": list(lane),
                    }
                )
            batch_rows.append(
                {
                    "registered_concurrent_batch_index": (
                        batch_start // AUDIT_135M_MAX_CONCURRENT_BLOCKS
                    ),
                    "block_indices": list(
                        range(batch_start, batch_start + batch_size)
                    ),
                    "available_slot_set": list(batch_slots),
                    "assignments": assignments,
                }
            )
        plan.pop("available_slot_variants", None)
        plan["schema"] = "yeto_parallel_plan_v4"
        plan["supersedes_revision_1_2_parallel_plan_hash"] = revision_1_2_hash
        plan["logical_slots"] = list(AUDIT_135M_LOGICAL_SLOTS)
        plan["waves"] = concurrent_waves
        plan["audit_concurrent_block_binding"] = {
            "schema": "audit_135m_concurrent_block_binding_v1",
            "amendment_raw_sha256": (
                AUDIT_135M_CONCURRENT_BLOCK_AMENDMENT_RAW_SHA256
            ),
            "design_contract_hash": contract_hash,
            "block_width": AUDIT_135M_BLOCK_WIDTH,
            "maximum_concurrent_blocks": AUDIT_135M_MAX_CONCURRENT_BLOCKS,
            "maximum_1g_scientific_vms": len(AUDIT_135M_LOGICAL_SLOTS),
            "pure_inputs": [
                "audit_135m_design_contract_hash",
                "planned_block_index",
                "available_slot_set",
            ],
            "outcome_inputs_forbidden": True,
            "registered_full_width_batches": batch_rows,
        }
        plan["capacity"].update(
            {
                "contract_revision": 3,
                "maximum_available_scientific_vms": len(
                    AUDIT_135M_LOGICAL_SLOTS
                ),
                "maximum_concurrent_scientific_cells": (
                    AUDIT_135M_MAX_CONCURRENT_CELLS
                ),
                "maximum_concurrent_atomic_blocks": (
                    AUDIT_135M_MAX_CONCURRENT_BLOCKS
                ),
                "scientific_vms_per_atomic_block": AUDIT_135M_BLOCK_WIDTH,
                "capacity_degrades_only_from_loss_blind_provider_census": True,
            }
        )
    return plan


def parallel_plan_hash(plan: Mapping[str, Any]) -> str:
    if plan.get("schema") not in (
        "yeto_parallel_plan_v1",
        "yeto_parallel_plan_v2",
        "yeto_parallel_plan_v3",
        "yeto_parallel_plan_v4",
    ):
        raise ScheduleError("parallel plan has the wrong schema")
    return canonical_sha256(plan)


def multiseed_analysis_loss_cap(
    stage_code: str, bound_manifest: Mapping[str, Any]
) -> float | None:
    """Return the exact frozen divergence cap for controlled audit stages."""

    if stage_code not in CONTROLLED_STAGE_CODES:
        return None
    policy = bound_manifest.get("analysis_policy")
    if not isinstance(policy, Mapping) or set(policy) != {
        "divergence_loss_cap",
        "divergence_is_outcome",
        "silent_divergence_exclusion_forbidden",
    }:
        raise ScheduleError("multiseed bound contract lacks the exact analysis policy")
    cap = policy.get("divergence_loss_cap")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not math.isfinite(float(cap))
        or float(cap) <= 0.0
        or policy.get("divergence_is_outcome") is not True
        or policy.get("silent_divergence_exclusion_forbidden") is not True
    ):
        raise ScheduleError("multiseed divergence policy is not a positive capped outcome")
    return float(cap)


def multiseed_analysis_outcome_fields(
    *,
    status: str,
    loss: Any,
    divergence_loss_cap: float | None,
    checkpoint_only: bool = False,
) -> dict[str, Any]:
    """Return the exact analysis fields for one scientifically resolved E1/E4 row."""

    if divergence_loss_cap is None or status not in ("COMPLETED", "DIVERGED"):
        return {}
    if checkpoint_only and status == "COMPLETED":
        if loss is not None:
            raise EvidenceError(
                "checkpoint-only acquisition must complete with a null raw loss"
            )
        return {
            "analysis_loss": None,
            "analysis_loss_kind": "sealed_checkpoint_pending_bound_evaluation",
            "divergence_retained": False,
        }
    if status == "DIVERGED":
        if loss is not None:
            raise EvidenceError("DIVERGED multiseed outcome must retain a null raw loss")
        analysis_loss = divergence_loss_cap
        analysis_loss_kind = "capped_divergence_endpoint_nll"
        retained = True
    else:
        analysis_loss = require_finite(loss, "completed endpoint loss")
        analysis_loss_kind = "finite_endpoint_nll"
        retained = False
    return {
        "analysis_loss": analysis_loss,
        "analysis_loss_kind": analysis_loss_kind,
        "divergence_retained": retained,
    }


def multiseed_runtime_authorization_hash(
    *,
    stage_code: str,
    design_contract_hash: Any,
    roster_digest: str,
    parallel_digest: str,
    bound_digest: str,
    scientific_digest: str,
    authorization: Mapping[str, Any] | None,
    hard_ceiling_usd: Any = None,
) -> str | None:
    """Validate and hash the exact external launch authorization."""

    if stage_code not in CONTROLLED_STAGE_CODES:
        return None
    common = {
        "launch_authorized": True,
        "stage_code": stage_code,
        "roster_hash": require_sha256(roster_digest, "roster hash"),
        "parallel_plan_hash": require_sha256(parallel_digest, "parallel plan hash"),
        "bound_manifest_canonical_sha256": require_sha256(
            bound_digest, "bound manifest canonical hash"
        ),
        "scientific_randomization_plan_hash": require_sha256(
            scientific_digest, "scientific randomization plan hash"
        ),
    }
    if stage_code in MULTISEED_STAGE_CODES:
        expected = {
            "schema": "yeto_multiseed_runtime_authorization_v1",
            **common,
            "best_paper_v2_design_contract_hash": require_sha256(
                design_contract_hash, "best-paper v2 design contract hash"
            ),
        }
    else:
        ceiling = require_finite(hard_ceiling_usd, "audit hard ceiling")
        if ceiling <= 0.0:
            raise LifecycleError("audit hard ceiling must be positive")
        expected = {
            "schema": "yeto_audit_135m_runtime_authorization_v1",
            **common,
            "audit_135m_design_contract_hash": require_sha256(
                design_contract_hash, "audit-135m design contract hash"
            ),
            "hard_ceiling_usd": ceiling,
            "spot_only": True,
            "maximum_attached_a100_equivalent": 16,
            "max_idle_before_science_seconds": 600,
        }
    if (
        not isinstance(authorization, Mapping)
        or set(authorization) != set(expected)
        or any(authorization.get(key) != value for key, value in expected.items())
    ):
        raise LifecycleError(
            "controlled execution lacks an exact runtime authorization"
        )
    return canonical_sha256(authorization)


def learner_gpu_slot_map(learner_count: int, gpu_slots: int) -> dict[str, int]:
    """Materialize the exact round-robin learner-to-local-GPU assignment."""

    require_positive_int(learner_count, "learner count")
    require_positive_int(gpu_slots, "GPU slot count")
    return {
        str(learner_id): learner_id % gpu_slots
        for learner_id in range(learner_count)
    }


def wave_for_retry(
    plan: Mapping[str, Any],
    roster: Mapping[str, Any],
    group_id: str,
    retry_round: int,
    available_slots: Iterable[str] | None = None,
) -> dict[str, Any]:
    by_group = _group_launch_cells(roster)
    if group_id not in by_group:
        raise ScheduleError(f"unknown retry group {group_id}")
    initial = [wave for wave in plan["waves"] if wave["group_id"] == group_id]
    if len(initial) != 1:
        raise ScheduleError(f"parallel plan does not contain exactly one {group_id} wave")
    if plan.get("schema") == "yeto_parallel_plan_v1":
        return build_legacy_wave_assignment(
            study_id=str(plan["study_id"]),
            group_id=group_id,
            cells=by_group[group_id],
            retry_round=retry_round,
            wave_index=int(initial[0]["wave_index"]),
            time_block_index=int(initial[0]["time_block_index"]),
        )
    logical_slots = logical_slots_for_stage(str(plan["stage_code"]))
    if available_slots is None:
        if plan.get("schema") == "yeto_parallel_plan_v4":
            slots = _array(
                initial[0].get("available_slot_set"),
                "audit registered wave slot set",
            )
        else:
            slots = LOGICAL_SLOTS
    else:
        slots = available_slots
    return build_wave_assignment(
        study_id=str(plan["study_id"]),
        roster_digest=str(plan["roster_hash"]),
        group_id=group_id,
        cells=by_group[group_id],
        available_slots=slots,
        retry_round=retry_round,
        wave_index=int(initial[0]["wave_index"]),
        time_block_index=int(initial[0]["time_block_index"]),
        logical_slots=logical_slots,
    )


def validate_prebound_p1r0_schedule(
    prebound: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    if plan.get("stage_code") != "p1r0":
        raise ScheduleError("prebound P1-R0 validation requires a P1-R0 final plan")
    if prebound.get("stage_code") != "p1r0" or prebound.get("launch_cell_count") != 36:
        raise ScheduleError("prebound artifact is not the registered 36-cell P1-R0 schedule")
    prebound_waves = _array(prebound.get("waves"), "prebound waves")
    final_waves = _array(
        plan.get("superseded_full_width_waves", plan.get("waves")),
        "parallel plan superseded full-width waves",
    )
    if len(prebound_waves) != len(final_waves):
        raise ScheduleError("prebound and final P1-R0 wave counts differ")
    if len(prebound_waves) != 12:
        raise ScheduleError("prebound P1-R0 schedule must contain 12 waves")
    for index, (old, new) in enumerate(zip(prebound_waves, final_waves)):
        if (
            old.get("group_id") != new.get("group_id")
            or old.get("wave_index") != index
            or old.get("time_block_index") != index
        ):
            raise ScheduleError(f"prebound wave {index} time-block identity differs")
        old_rows = _array(
            old.get("assigned_cells_in_dispatch_order"),
            f"prebound wave {index} assignments",
        )
        new_rows = _array(
            new.get("assigned_cells_in_dispatch_order"),
            f"final wave {index} assignments",
        )
        old_projection = [
            (row.get("cell_id"), row.get("logical_slot"), row.get("launch_order_index"))
            for row in old_rows
        ]
        new_projection = [
            (row.get("cell_id"), row.get("logical_slot"), row.get("launch_order_index"))
            for row in new_rows
        ]
        if old_projection != new_projection:
            raise ScheduleError(f"prebound wave {index} assignment/dispatch order differs")


def physical_run_id(
    stage_code: str,
    roster_digest: str,
    campaign_attempt: int,
    slot: str,
    generation: int,
) -> str:
    if stage_code not in ALLOWED_STAGE_CODES:
        raise LifecycleError("physical run ID has an unauthorized stage code")
    require_sha256(roster_digest, "roster hash")
    require_positive_int(campaign_attempt, "campaign attempt")
    require_positive_int(generation, "physical generation")
    if slot not in logical_slots_for_stage(stage_code):
        raise LifecycleError(f"unknown logical slot {slot!r}")
    run_id = (
        f"bp-{stage_code}-{roster_digest[:16]}-c{campaign_attempt}-{slot}-g{generation}"
    )
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise LifecycleError(f"physical run ID does not match the amendment grammar: {run_id}")
    return run_id


def _join_artifact_root(root: str, suffix: str) -> str:
    return root.rstrip("/") + "/" + suffix.lstrip("/")


@dataclass(frozen=True)
class GenerationIdentity:
    stage_code: str
    study_id: str
    roster_hash: str
    campaign_attempt: int
    slot: str
    generation: int
    ownership_nonce: str
    campaign_state_root: Path
    campaign_artifact_root: str

    @property
    def roster_tag(self) -> str:
        return self.roster_hash[:16]

    @property
    def run_id(self) -> str:
        return physical_run_id(
            self.stage_code,
            self.roster_hash,
            self.campaign_attempt,
            self.slot,
            self.generation,
        )

    @property
    def state_path(self) -> Path:
        return self.campaign_state_root / f"{self.run_id}.json"

    @property
    def artifact_prefix(self) -> str:
        return _join_artifact_root(
            self.campaign_artifact_root,
            f"vms/{self.slot}/g{self.generation}/",
        )

    @property
    def provider_record_path(self) -> str:
        return _join_artifact_root(self.artifact_prefix, "provider/provider-evidence.json")

    @property
    def partial_manifest_path(self) -> str:
        return _join_artifact_root(
            self.artifact_prefix, "manifests/vm-partial-manifest.json"
        )

    @property
    def lifecycle_record_path(self) -> str:
        return _join_artifact_root(
            self.artifact_prefix, "manifests/vm-lifecycle-final.json"
        )

    @property
    def labels(self) -> dict[str, str]:
        return {
            "campaign-tag": self.roster_tag,
            "logical-slot": self.slot,
            "physical-generation": str(self.generation),
            "run-id": self.run_id,
            "ownership-nonce": self.ownership_nonce,
        }

    def attempt_prefix(self, cell_id: str, attempt: int) -> str:
        require_positive_int(attempt, "cell attempt")
        if not isinstance(cell_id, str) or not cell_id or "/" in cell_id:
            raise LifecycleError("cell_id is unsafe for an attempt namespace")
        return _join_artifact_root(
            self.artifact_prefix, f"cells/{cell_id}/attempt-{attempt}/"
        )

    def registry_row(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "generation": self.generation,
            "run_id": self.run_id,
            "ownership_nonce": self.ownership_nonce,
            "state_path": str(self.state_path),
            "artifact_prefix": self.artifact_prefix,
            "provider_record_path": self.provider_record_path,
            "partial_manifest_path": self.partial_manifest_path,
            "lifecycle_record_path": self.lifecycle_record_path,
            "labels": self.labels,
        }


class CampaignGenerationRegistry:
    """Create-only physical-generation registry with monotone slot generations."""

    def __init__(
        self,
        *,
        stage_code: str,
        study_id: str,
        roster_digest: str,
        campaign_attempt: int,
        campaign_state_root: Path,
        campaign_artifact_root: str,
    ) -> None:
        if stage_code not in ALLOWED_STAGE_CODES:
            raise LifecycleError("generation registry stage is not amendment-authorized")
        self.stage_code = stage_code
        self.study_id = study_id
        self.roster_hash = require_sha256(roster_digest, "roster hash")
        self.campaign_attempt = require_positive_int(campaign_attempt, "campaign attempt")
        self.campaign_state_root = campaign_state_root.resolve()
        self.campaign_artifact_root = campaign_artifact_root.rstrip("/")
        self._identities: dict[tuple[str, int], GenerationIdentity] = {}

    @property
    def identities(self) -> tuple[GenerationIdentity, ...]:
        return tuple(
            self._identities[key]
            for key in sorted(self._identities, key=lambda item: (item[0], item[1]))
        )

    def reserve(
        self,
        slot: str,
        *,
        ownership_nonce: str | None = None,
    ) -> GenerationIdentity:
        if slot not in logical_slots_for_stage(self.stage_code):
            raise LifecycleError(f"unknown logical slot {slot!r}")
        prior = [generation for (seen_slot, generation) in self._identities if seen_slot == slot]
        generation = max(prior, default=0) + 1
        nonce = ownership_nonce or secrets.token_hex(16)
        if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
            raise LifecycleError("ownership nonce must be exactly 32 lowercase hex characters")
        if any(identity.ownership_nonce == nonce for identity in self._identities.values()):
            raise LifecycleError("ownership nonce reuse is forbidden")
        identity = GenerationIdentity(
            stage_code=self.stage_code,
            study_id=self.study_id,
            roster_hash=self.roster_hash,
            campaign_attempt=self.campaign_attempt,
            slot=slot,
            generation=generation,
            ownership_nonce=nonce,
            campaign_state_root=self.campaign_state_root,
            campaign_artifact_root=self.campaign_artifact_root,
        )
        state = {
            "schema": "yeto_parallel_vm_generation_state_v1",
            "status": "generation_reserved_preprovision",
            "stage_code": self.stage_code,
            "study_id": self.study_id,
            "roster_hash": self.roster_hash,
            "campaign_attempt": self.campaign_attempt,
            **identity.registry_row(),
            "reserved_at_utc": utc_now(),
        }
        write_json_create_only(identity.state_path, state)
        self._identities[(slot, generation)] = identity
        return identity

    def update_state(self, identity: GenerationIdentity, **updates: Any) -> None:
        registered = self._identities.get((identity.slot, identity.generation))
        if registered != identity:
            raise LifecycleError("cannot update an unregistered VM generation")
        state = _mapping(load_json(identity.state_path, "generation state"), "generation state")
        if state.get("run_id") != identity.run_id or state.get("ownership_nonce") != identity.ownership_nonce:
            raise LifecycleError("generation state ownership changed")
        state.update(updates)
        write_json_atomic(identity.state_path, state)

    def snapshot(self) -> dict[str, Any]:
        generations = []
        for identity in self.identities:
            row = identity.registry_row()
            state = _mapping(load_json(identity.state_path, "generation state"), "generation state")
            if state.get("region") is not None:
                row["region"] = state["region"]
            if state.get("zone") is not None:
                row["zone"] = state["zone"]
            if state.get("machine_type") is not None:
                row["machine_type"] = state["machine_type"]
            generations.append(row)
        return {
            "schema": "yeto_parallel_vm_registry_v1",
            "stage_code": self.stage_code,
            "study_id": self.study_id,
            "roster_hash": self.roster_hash,
            "campaign_attempt": self.campaign_attempt,
            "generations": generations,
        }


def _relative_vm_path(identity: GenerationIdentity, full_path: str) -> PurePosixPath:
    prefix = identity.artifact_prefix.rstrip("/") + "/"
    if not full_path.startswith(prefix):
        raise LifecycleError("generation path escapes its owned artifact prefix")
    relative = PurePosixPath(full_path[len(prefix) :])
    if relative.is_absolute() or ".." in relative.parts or str(relative) in ("", "."):
        raise LifecycleError("generation path is not normalized beneath its prefix")
    return relative


class VmPartialManifestController:
    """Append attempts for one exact physical generation, then hash-lock once."""

    def __init__(
        self,
        *,
        identity: GenerationIdentity,
        local_vm_root: Path,
        common_bindings: Mapping[str, Any],
        provider_record_sha256: str,
    ) -> None:
        self.identity = identity
        self.local_vm_root = local_vm_root.resolve()
        self.manifest_path = self.local_vm_root / "manifests" / "vm-partial-manifest.json"
        self.hash_path = self.local_vm_root / "manifests" / "vm-partial-manifest.sha256"
        require_sha256(provider_record_sha256, "provider record SHA-256")
        header = {
            "schema": "yeto_vm_partial_manifest_v1",
            "status": "collecting_attempts",
            "stage_code": identity.stage_code,
            "study_id": identity.study_id,
            "run_id": identity.run_id,
            "slot": identity.slot,
            "generation": identity.generation,
            "ownership_nonce": identity.ownership_nonce,
            "artifact_prefix": identity.artifact_prefix,
            "provider_record_sha256": provider_record_sha256,
            "roster_hash": common_bindings["roster_hash"],
            "parallel_plan_hash": common_bindings["parallel_plan_hash"],
            "bound_manifest_canonical_sha256": common_bindings[
                "bound_manifest_canonical_sha256"
            ],
            "scientific_randomization_plan_hash": common_bindings[
                "scientific_randomization_plan_hash"
            ],
            "amendment_raw_sha256": common_bindings["amendment_raw_sha256"],
            "attempts": [],
            "partial_outcomes_exposed": False,
        }
        if common_bindings.get("multiseed_runtime_authorization_hash") is not None:
            header["multiseed_runtime_authorization_hash"] = require_sha256(
                common_bindings.get("multiseed_runtime_authorization_hash"),
                "multiseed runtime authorization hash",
            )
        write_json_create_only(self.manifest_path, header)

    def _load_collecting(self) -> dict[str, Any]:
        if self.hash_path.exists():
            raise LifecycleError("VM partial manifest is already hash-locked")
        value = _mapping(load_json(self.manifest_path, "VM partial manifest"), "VM partial manifest")
        if value.get("status") != "collecting_attempts":
            raise LifecycleError("VM partial manifest is not appendable")
        return value

    def append_attempt(self, attempt: Mapping[str, Any]) -> None:
        value = self._load_collecting()
        row = deepcopy(dict(attempt))
        if (
            row.get("run_id") != self.identity.run_id
            or row.get("logical_slot") != self.identity.slot
            or row.get("generation") != self.identity.generation
            or row.get("ownership_nonce") != self.identity.ownership_nonce
        ):
            raise LifecycleError("attempt row does not belong to this physical generation")
        _relative_vm_path(self.identity, str(row.get("attempt_prefix", "")))
        attempts = _array(value.get("attempts"), "partial attempts")
        order_key = (
            require_nonnegative_int(row.get("actual_wave_index"), "actual wave index"),
            require_nonnegative_int(row.get("launch_order_index"), "launch order index"),
        )
        if attempts:
            prior = attempts[-1]
            prior_key = (int(prior["actual_wave_index"]), int(prior["launch_order_index"]))
            if order_key <= prior_key:
                raise LifecycleError("VM partial attempts are not append-only wave/launch ordered")
        attempts.append(row)
        value["attempts"] = attempts
        write_json_atomic(self.manifest_path, value)

    def hash_lock(self, *, hash_locked_at_utc: str | None = None) -> str:
        value = self._load_collecting()
        value["status"] = "vm_partial_hash_locked"
        value["hash_locked_at_utc"] = hash_locked_at_utc or utc_now()
        write_json_atomic(self.manifest_path, value)
        digest = sha256_file(self.manifest_path)
        write_bytes_create_only(
            self.hash_path,
            f"{digest}  vm-partial-manifest.json\n".encode("utf-8"),
        )
        return digest


def validate_provider_record(
    record: Mapping[str, Any], identity: Mapping[str, Any]
) -> dict[str, Any]:
    if record.get("schema") != "yeto_parallel_gcp_provider_evidence_v1":
        raise LifecycleError("provider record has the wrong schema")
    expected_pairs = {
        "run_id": identity.get("run_id"),
        "slot": identity.get("slot"),
        "generation": identity.get("generation"),
        "ownership_nonce": identity.get("ownership_nonce"),
    }
    for field, expected in expected_pairs.items():
        if record.get(field) != expected:
            raise LifecycleError(f"provider record {field} differs from VM registry")
    labels = _mapping(record.get("labels"), "provider labels")
    expected_labels = _mapping(identity.get("labels"), "VM registry labels")
    if any(labels.get(key) != value for key, value in expected_labels.items()):
        raise LifecycleError("provider labels do not prove exact generation ownership")
    instance_id = require_numeric_id(record.get("instance_numeric_id"), "instance numeric ID")
    disk_id = require_numeric_id(record.get("boot_disk_numeric_id"), "boot disk numeric ID")
    source_image_id = require_numeric_id(
        record.get("source_image_numeric_id"), "source image numeric ID", allow_protected=True
    )
    if (
        record.get("project") != "model-training-497007"
        or record.get("zone") not in ALLOWED_US_A100_ZONES
        or record.get("campaign_tag") != expected_labels.get("campaign-tag")
        or record.get("instance_name") != identity.get("run_id")
        or not isinstance(record.get("boot_disk_name"), str)
        or not record.get("boot_disk_name")
        or source_image_id != "7290368630472593484"
    ):
        raise LifecycleError("provider record project/zone/name/image identity differs")
    zone = str(record.get("zone"))
    region = region_for_zone(zone)
    recorded_region = record.get("region")
    identity_region = identity.get("region")
    if recorded_region is None:
        if identity_region is not None or zone not in REVISION_1_2_ALLOWED_ZONES:
            raise LifecycleError("provider record omits the required landed region")
    elif recorded_region != region:
        raise LifecycleError("provider record region does not match its landed zone")
    if identity_region is not None and recorded_region != identity_region:
        raise LifecycleError("provider record region differs from physical identity")
    if identity.get("zone") is not None and record.get("zone") != identity.get("zone"):
        raise LifecycleError("provider record zone differs from the landed physical identity")
    shape = machine_shape_contract(record.get("machine_type"))
    if (
        identity.get("machine_type") is not None
        and shape["machine_type"] != identity.get("machine_type")
    ):
        raise LifecycleError(
            "provider record machine type differs from the landed physical identity"
        )
    required_contract = {
        "provisioning_model": "SPOT",
        "termination_action": "DELETE",
        "automatic_restart": False,
        "maintenance_action": "TERMINATE",
        "boot_disk_auto_delete": True,
    }
    for field, expected in required_contract.items():
        if record.get(field) != expected:
            raise LifecycleError(f"provider record does not prove {field}={expected!r}")
    parse_time(record.get("creation_timestamp"), "provider creation timestamp")
    cuda_indices = _array(record.get("cuda_indices"), "provider CUDA indices")
    gpu_uuids = _array(record.get("a100_gpu_uuids"), "provider A100 UUIDs")
    gpu_names = _array(record.get("a100_gpu_names"), "provider A100 names")
    bijection = _mapping(record.get("learner_gpu_uuid_bijection"), "learner/GPU bijection")
    if record.get("gpu_allocation_mode") != shape["gpu_allocation_mode"]:
        raise LifecycleError("provider record GPU allocation mode differs from machine shape")
    if cuda_indices != list(range(shape["a100_count"])):
        raise LifecycleError("provider record CUDA inventory differs from machine shape")
    if (
        len(gpu_uuids) != shape["a100_count"]
        or len(set(gpu_uuids)) != shape["a100_count"]
        or any(not isinstance(value, str) or not value.startswith("GPU-") for value in gpu_uuids)
        or len(gpu_names) != shape["a100_count"]
        or any(not isinstance(value, str) or "A100" not in value.upper() for value in gpu_names)
    ):
        raise LifecycleError("provider record A100 inventory differs from machine shape")
    if set(bijection) != {"0", "1", "2", "3"}:
        raise LifecycleError("provider learner-to-GPU UUID mapping lacks four learners")
    assigned = list(bijection.values())
    if shape["machine_type"] == SCIENTIFIC_VM_SHAPE:
        if len(set(assigned)) != 4 or set(assigned) != set(gpu_uuids):
            raise LifecycleError("4g provider mapping is not a four-A100 learner bijection")
    elif set(assigned) != {gpu_uuids[0]}:
        raise LifecycleError("1g provider mapping does not pack four learners on one A100")
    return {
        "instance_numeric_id": instance_id,
        "boot_disk_numeric_id": disk_id,
        "source_image_numeric_id": source_image_id,
        "creation_timestamp": record["creation_timestamp"],
        "region": region,
        "machine_type": shape["machine_type"],
        "a100_count": shape["a100_count"],
        "gpu_slots": shape["gpu_slots"],
    }


def validate_lifecycle_record(
    lifecycle: Mapping[str, Any],
    identity: Mapping[str, Any],
    provider_record: Mapping[str, Any],
    partial_manifest_sha256: str,
) -> dict[str, Any]:
    if lifecycle.get("schema") != "yeto_vm_lifecycle_final_v1":
        raise LifecycleError("VM lifecycle record has the wrong schema")
    if lifecycle.get("status") != "vm_lifecycle_final":
        raise LifecycleError("VM lifecycle record is not final")
    for field in ("run_id", "slot", "generation", "ownership_nonce"):
        if lifecycle.get(field) != identity.get(field):
            raise LifecycleError(f"VM lifecycle {field} differs from the registry")
    if (
        lifecycle.get("zone") not in ALLOWED_US_A100_ZONES
        or lifecycle.get("zone") != provider_record.get("zone")
        or (
            identity.get("zone") is not None
            and lifecycle.get("zone") != identity.get("zone")
        )
    ):
        raise LifecycleError("VM lifecycle landed zone differs from physical identity")
    zone = str(lifecycle.get("zone"))
    region = region_for_zone(zone)
    lifecycle_region = lifecycle.get("region")
    provider_region = provider_record.get("region")
    identity_region = identity.get("region")
    legacy_region_omission = (
        zone in REVISION_1_2_ALLOWED_ZONES
        and lifecycle_region is None
        and provider_region is None
        and identity_region is None
    )
    if not legacy_region_omission and (
        lifecycle_region != region
        or provider_region != region
        or (identity_region is not None and lifecycle_region != identity_region)
    ):
        raise LifecycleError("VM lifecycle landed region differs from physical identity")
    shape = machine_shape_contract(lifecycle.get("machine_type"))
    if (
        shape["machine_type"] != provider_record.get("machine_type")
        or (
            identity.get("machine_type") is not None
            and shape["machine_type"] != identity.get("machine_type")
        )
    ):
        raise LifecycleError("VM lifecycle machine type differs from physical identity")
    if lifecycle.get("partial_manifest_sha256") != partial_manifest_sha256:
        raise LifecycleError("VM lifecycle does not bind the hash-locked partial manifest")
    instance_id = require_numeric_id(
        lifecycle.get("instance_numeric_id"), "lifecycle instance numeric ID"
    )
    disk_id = require_numeric_id(
        lifecycle.get("boot_disk_numeric_id"), "lifecycle boot disk numeric ID"
    )
    if (
        instance_id != str(provider_record.get("instance_numeric_id"))
        or disk_id != str(provider_record.get("boot_disk_numeric_id"))
    ):
        raise LifecycleError("lifecycle exact IDs differ from the provider record")
    labels = _mapping(lifecycle.get("labels"), "lifecycle ownership labels")
    if labels != _mapping(identity.get("labels"), "registry ownership labels"):
        raise LifecycleError("lifecycle ownership labels differ from the registry")
    requested = parse_time(lifecycle.get("deletion_requested_at_utc"), "deletion request")
    completed = parse_time(lifecycle.get("deletion_completed_at_utc"), "deletion completion")
    if completed < requested:
        raise LifecycleError("deletion completion precedes deletion request")
    proofs = _mapping(
        lifecycle.get("provider_not_found_verification"), "provider NOT_FOUND proofs"
    )
    instance_proof = _mapping(proofs.get("instance"), "instance NOT_FOUND proof")
    disk_proof = _mapping(proofs.get("boot_disk"), "disk NOT_FOUND proof")
    if (
        instance_proof.get("result") != "NOT_FOUND"
        or str(instance_proof.get("provider_id")) != instance_id
        or disk_proof.get("result") != "NOT_FOUND"
        or str(disk_proof.get("provider_id")) != disk_id
    ):
        raise LifecycleError("exact instance/disk NOT_FOUND proof is missing or name-only")
    zero = _mapping(lifecycle.get("zero_attached_accelerator_proof"), "zero accelerator proof")
    if zero.get("generation_attached_a100s") != 0:
        raise LifecycleError("lifecycle does not prove zero A100s attached to the exact generation")
    return {
        "instance_numeric_id": instance_id,
        "boot_disk_numeric_id": disk_id,
        "creation_timestamp": provider_record["creation_timestamp"],
        "deletion_completed_at_utc": lifecycle["deletion_completed_at_utc"],
        "region": region,
        "machine_type": shape["machine_type"],
        "a100_count": shape["a100_count"],
    }


def _safe_campaign_path(campaign_root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise EvidenceError(f"{label} path must be a nonempty campaign-relative string")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) in ("", "."):
        raise EvidenceError(f"{label} path escapes the campaign artifact root")
    root = campaign_root.resolve()
    path = (root / Path(*parsed.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvidenceError(f"{label} path escapes the campaign artifact root") from exc
    return path


def verify_artifact_inventory(
    campaign_root: Path, row: Mapping[str, Any]
) -> dict[str, Path]:
    inventory = _mapping(row.get("artifact_inventory"), "attempt artifact inventory")
    verified: dict[str, Path] = {}
    seen_paths: set[Path] = set()
    for role, raw in inventory.items():
        if not isinstance(role, str) or not role:
            raise EvidenceError("artifact inventory role must be nonempty")
        entry = _mapping(raw, f"artifact inventory {role}")
        path = _safe_campaign_path(campaign_root, entry.get("path"), role)
        if path in seen_paths:
            raise EvidenceError("two artifact roles cite the same path")
        seen_paths.add(path)
        if not path.is_file() or path.is_symlink():
            raise EvidenceError(f"required artifact {role} is missing or unsafe")
        expected_size = require_nonnegative_int(entry.get("size_bytes"), f"{role} size")
        if expected_size != path.stat().st_size:
            raise EvidenceError(f"required artifact {role} size differs")
        expected_hash = require_sha256(entry.get("sha256"), f"{role} SHA-256")
        if sha256_file(path) != expected_hash:
            raise EvidenceError(f"required artifact {role} checksum differs")
        verified[role] = path
    return verified


def read_jsonl_strict(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise EvidenceError(f"cannot read {label}: {exc}") from exc
    if not lines:
        raise EvidenceError(f"{label} must be nonempty")
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise EvidenceError(f"{label}:{line_number} is blank")
        try:
            value = strict_json_loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"{label}:{line_number} is invalid strict JSON") from exc
        if not isinstance(value, dict):
            raise EvidenceError(f"{label}:{line_number} must be an object")
        rows.append(value)
    return rows


def _load_strict_object(path: Path, label: str) -> dict[str, Any]:
    value = load_json(path, label)
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def _expected_work_for_cell(cell: Mapping[str, Any]) -> dict[str, int]:
    h = int(_cell_field(cell, "h", "H"))
    target = cell.get("target_work")
    if target is not None:
        target = _mapping(target, "scientific target work")
        learner_count = require_positive_int(
            target.get("learner_count"), "scientific learner count"
        )
        learner_steps = require_positive_int(
            target.get("learner_steps_per_learner"),
            "scientific learner steps per learner",
        )
        tokens = require_positive_int(target.get("tokens"), "scientific tokens")
        microsteps = require_positive_int(
            target.get("microsteps"), "scientific microsteps"
        )
        outer_steps = require_positive_int(
            target.get("outer_steps"), "scientific outer steps"
        )
        partial_registered = target.get("terminal_partial_window_registered") is True
        partial_steps = target.get("terminal_partial_window_microsteps", 0)
        if isinstance(partial_steps, bool) or not isinstance(partial_steps, int):
            raise EvidenceError("terminal partial-window work is malformed")
        full_windows, remainder = divmod(learner_steps, h)
        expected_rounds = full_windows + bool(remainder)
        if (
            _cell_learner_count(cell) != learner_count
            or (remainder and not partial_registered)
            or (not remainder and partial_registered)
            or partial_steps != remainder
            or outer_steps != expected_rounds * FRAGMENT_COUNT
            or outer_steps % FRAGMENT_COUNT
            or microsteps != learner_count * learner_steps
            or tokens % microsteps
        ):
            raise EvidenceError("scientific plan target work is not internally exact")
        return {
            "h": h,
            "tokens": tokens,
            "microsteps": microsteps,
            "tokens_per_microstep": tokens // microsteps,
            "outer_steps": outer_steps,
            "per_fragment_outer_updates": outer_steps // FRAGMENT_COUNT,
            "learner_count": learner_count,
            "learner_steps_per_learner": learner_steps,
            "terminal_partial_window_registered": partial_registered,
            "terminal_partial_window_microsteps": partial_steps,
        }
    if h not in HORIZON_WORK and h not in BEST_PAPER_V2_ARMS:
        raise EvidenceError(f"cell {cell.get('cell_id')} has unauthorized horizon H={h}")
    return {
        "h": h,
        "tokens": EXPECTED_TOKENS,
        "microsteps": EXPECTED_MICROSTEPS,
        "tokens_per_microstep": EXPECTED_TOKENS // EXPECTED_MICROSTEPS,
        "outer_steps": HORIZON_WORK[h]["outer_steps"],
        "per_fragment_outer_updates": HORIZON_WORK[h][
            "per_fragment_outer_updates"
        ],
        "learner_count": len(EXPECTED_LEARNERS),
        "learner_steps_per_learner": EXPECTED_STEPS_PER_LEARNER,
    }


def _validate_command_artifact(
    path: Path, expected_command: Sequence[Any], expected_hash: str
) -> None:
    command = load_json(path, "executed scientific command")
    if not isinstance(command, list) or any(not isinstance(token, str) for token in command):
        raise EvidenceError("executed command artifact must be an argv string array")
    if command != list(expected_command) or canonical_sha256(command) != expected_hash:
        raise EvidenceError("executed command bytes differ from the registered cell command")


def _validate_shape_projected_command(
    *,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    expected_command: Sequence[str],
    command_path: Path,
) -> str:
    machine_type = row.get("machine_type")
    projected = project_scientific_command_for_machine_type(
        expected_command, machine_type
    )
    frozen_hash = require_sha256(cell.get("command_hash"), "frozen command hash")
    executed_hash = canonical_sha256(projected)
    if "--gpu-slots" not in expected_command:
        if (
            row.get("frozen_command_hash") != frozen_hash
            or require_sha256(row.get("executed_command_hash"), "executed command hash")
            != frozen_hash
        ):
            raise EvidenceError("attempt command differs from the frozen nonprojected command")
        _validate_command_artifact(command_path, projected, frozen_hash)
        return frozen_hash
    normalized_hash = canonical_sha256(normalized_workload_command(projected))
    if (
        row.get("frozen_command_hash") != frozen_hash
        or require_sha256(row.get("executed_command_hash"), "executed command hash")
        != executed_hash
        or require_sha256(
            row.get("normalized_workload_command_hash"),
            "normalized workload command hash",
        )
        != cell.get("normalized_workload_command_hash")
        or normalized_hash != cell.get("normalized_workload_command_hash")
        or row.get("gpu_slots")
        != machine_shape_contract(machine_type)["gpu_slots"]
    ):
        raise EvidenceError("attempt command projection differs from its landed machine shape")
    _validate_command_artifact(command_path, projected, executed_hash)
    return executed_hash


def _validate_attempt_start(
    path: Path,
    *,
    cell_id: str,
    attempt: int,
    command_hash: str,
    provider_hash: str,
) -> None:
    start = _load_strict_object(path, "attempt-start evidence")
    expected = {
        "attempt_id": f"{cell_id}-attempt-{attempt}",
        "cell_id": cell_id,
        "attempt": attempt,
        "command_hash": command_hash,
        "provider_evidence_sha256": provider_hash,
        "fresh_initial_state": True,
        "resumed_from_attempt": None,
        "optimizer_state_input": None,
        "checkpoint_input": None,
        "prior_attempt_artifacts_used": False,
    }
    if any(start.get(key) != value for key, value in expected.items()):
        raise EvidenceError("attempt-start evidence permits a resume or changes frozen identity")
    parse_time(start.get("started_at_utc"), "attempt start evidence timestamp")


def _validate_learner_steps(
    path: Path, expected_steps: int, learner_count: int = 4
) -> None:
    evidence = _load_strict_object(path, "learner step evidence")
    if evidence.get("schema") != "yeto_parallel_learner_steps_v1":
        raise EvidenceError("learner step evidence has the wrong schema")
    learners = _mapping(evidence.get("learners"), "learner step map")
    expected_learners = tuple(range(learner_count))
    if set(learners) != {str(value) for value in expected_learners}:
        raise EvidenceError(
            f"learner IDs must be exactly 0..{learner_count - 1}"
        )
    expected = list(range(1, expected_steps + 1))
    for learner in expected_learners:
        steps = _array(learners[str(learner)], f"learner {learner} steps")
        if steps != expected:
            raise EvidenceError(
                f"learner {learner} optimizer steps are not exactly 1..{expected_steps}"
            )


def _validate_work_events(path: Path, expected: Mapping[str, int]) -> list[dict[str, Any]]:
    evidence = _load_strict_object(path, "work event evidence")
    if evidence.get("schema") != "yeto_parallel_work_events_v1":
        raise EvidenceError("work event evidence has the wrong schema")
    updates = _array(evidence.get("updates"), "work updates")
    if len(updates) != expected["outer_steps"]:
        raise EvidenceError("work event count differs from the horizon outer-step target")
    fragments: Counter[int] = Counter()
    total_microsteps = 0
    total_tokens = 0
    prior_fragment_commit = {fragment: 0 for fragment in range(FRAGMENT_COUNT)}
    for outer_step, raw in enumerate(updates, 1):
        update = _mapping(raw, f"work update {outer_step}")
        fragment = update.get("fragment")
        if (
            update.get("outer_step") != outer_step
            or isinstance(fragment, bool)
            or fragment not in range(FRAGMENT_COUNT)
        ):
            raise EvidenceError("work updates are missing, reordered, or have an invalid fragment")
        fragments[int(fragment)] += 1
        responders = _array(update.get("responders"), f"work update {outer_step} responders")
        if len(responders) != expected["learner_count"]:
            raise EvidenceError("work update lacks the exact full learner quorum")
        by_learner: dict[int, dict[str, Any]] = {}
        per_fragment_index = fragments[int(fragment)]
        expected_window_steps = (
            int(expected["terminal_partial_window_microsteps"])
            if expected.get("terminal_partial_window_registered")
            and per_fragment_index == expected["per_fragment_outer_updates"]
            else expected["h"]
        )
        for raw_responder in responders:
            responder = _mapping(raw_responder, "work responder")
            learner_id = responder.get("learner_id")
            if (
                isinstance(learner_id, bool)
                or learner_id not in range(expected["learner_count"])
                or learner_id in by_learner
            ):
                raise EvidenceError("work update responder IDs do not match M")
            if (
                responder.get("microsteps") != expected_window_steps
                or responder.get("tokens")
                != expected_window_steps * expected["tokens_per_microstep"]
                or responder.get("base_version") != prior_fragment_commit[int(fragment)]
                or responder.get("version_matched_anchor") is not True
            ):
                raise EvidenceError("work responder does not prove one exact fixed H-window")
            by_learner[int(learner_id)] = responder
            total_microsteps += int(responder["microsteps"])
            total_tokens += int(responder["tokens"])
        prior_fragment_commit[int(fragment)] = outer_step
    if fragments != Counter(
        {
            fragment: expected["per_fragment_outer_updates"]
            for fragment in range(FRAGMENT_COUNT)
        }
    ):
        raise EvidenceError("one or more fragments lack the exact registered update count")
    if total_microsteps // FRAGMENT_COUNT != expected["microsteps"]:
        raise EvidenceError("aggregate microsteps differ from the registered work")
    if total_tokens // FRAGMENT_COUNT != expected["tokens"]:
        raise EvidenceError("observed training tokens differ from the registered work")
    return updates


def _validate_barrier_events(
    path: Path,
    updates: Sequence[Mapping[str, Any]],
    learner_count: int = 4,
) -> None:
    evidence = _load_strict_object(path, "barrier event evidence")
    if evidence.get("schema") != "yeto_parallel_barrier_events_v1":
        raise EvidenceError("barrier event evidence has the wrong schema")
    learners = _mapping(evidence.get("learners"), "barrier learner map")
    expected_learners = tuple(range(learner_count))
    if set(learners) != {str(value) for value in expected_learners}:
        raise EvidenceError("barrier evidence learner IDs do not match M")
    for learner in expected_learners:
        row = _mapping(learners[str(learner)], f"barrier learner {learner}")
        if row.get("initial_fragments") != [0, 1, 2, 3]:
            raise EvidenceError("barrier evidence lacks the four initial broadcasts")
        pushes = _array(row.get("pushes"), f"learner {learner} pushes")
        broadcasts = _array(row.get("broadcasts"), f"learner {learner} broadcasts")
        learner_expected = [
            {
                "outer_step": int(update["outer_step"]),
                "fragment": int(update["fragment"]),
                "base_version": int(
                    next(
                        responder
                        for responder in update["responders"]
                        if responder["learner_id"] == learner
                    )["base_version"]
                ),
            }
            for update in updates
        ]
        if pushes != learner_expected:
            raise EvidenceError("barrier pushes do not biject to the work event tape")
        expected_broadcasts = [
            {
                "outer_step": item["outer_step"],
                "fragment": item["fragment"],
                "pushed_base_version": item["base_version"],
                "broadcast_version": item["outer_step"],
            }
            for item in learner_expected
        ]
        if broadcasts != expected_broadcasts:
            raise EvidenceError("barrier broadcasts do not release the exact pushed versions")
        if row.get("inner_steps_while_blocked") != []:
            raise EvidenceError("barrier evidence records learner work while blocked")


EVAL_IDENTITY_FIELDS = (
    "sequence_index",
    "sequence_id",
    "input_ids_sha256",
    "labels_sha256",
    "attention_mask_sha256",
    "supervision_weights_sha256",
    "target_token_mask_sha256",
    "sequence_length",
)
EVAL_BOUND_FIELDS = (
    "development_eval_rows_hash",
    "development_eval_packed_hash",
    "development_eval_example_ids_hash",
    "development_eval_token_ids_hash",
    "development_eval_source_indices_hash",
)


def _validate_finite_eval(
    freeze_path: Path,
    losses_path: Path,
    result_loss: float,
    *,
    evaluation_binding: Mapping[str, Any],
) -> None:
    if sha256_file(freeze_path) != require_sha256(
        evaluation_binding.get("sha256"), "frozen development evaluation SHA-256"
    ):
        raise EvidenceError("attempt cites a different frozen development evaluation")
    freeze = _load_strict_object(freeze_path, "frozen development evaluation")
    if freeze.get("schema") != "yeto_parallel_eval_freeze_v1":
        raise EvidenceError("frozen development evaluation has the wrong schema")
    if any(freeze.get(field) != evaluation_binding.get(field) for field in EVAL_BOUND_FIELDS):
        raise EvidenceError("frozen development evaluation identities differ from the bound descendant")
    frozen_rows = _array(freeze.get("sequences"), "frozen evaluation sequences")
    losses = read_jsonl_strict(losses_path, "per-sequence development losses")
    if not frozen_rows or len(losses) != len(frozen_rows):
        raise EvidenceError("per-sequence loss coverage differs from the frozen rows")
    total_loss = 0.0
    total_tokens = 0
    for index, (raw_frozen, loss_row) in enumerate(
        zip(frozen_rows, losses)
    ):
        frozen = _mapping(raw_frozen, f"frozen evaluation sequence {index}")
        for field in EVAL_IDENTITY_FIELDS:
            if loss_row.get(field) != frozen.get(field):
                raise EvidenceError(f"evaluation row {index} identity differs at {field}")
        token_count = loss_row.get("token_count")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
            or token_count != frozen.get("supervised_token_count")
        ):
            raise EvidenceError("every locked development row needs positive target tokens")
        loss_sum = require_finite(loss_row.get("loss_sum"), "per-sequence loss_sum")
        loss_per_token = require_finite(
            loss_row.get("loss_per_token"), "per-sequence loss_per_token"
        )
        if not math.isclose(
            loss_per_token,
            loss_sum / token_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise EvidenceError("per-sequence loss arithmetic mismatch")
        total_loss += loss_sum
        total_tokens += token_count
    if total_tokens != freeze.get("supervised_token_count"):
        raise EvidenceError("development target-token total differs from the freeze")
    aggregate = total_loss / total_tokens
    if not math.isclose(aggregate, result_loss, rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError("per-sequence losses do not reproduce the endpoint")


def _validate_results(
    path: Path,
    expected_loss: Any,
    *,
    learner_count: int = 4,
) -> float:
    result = _load_strict_object(path, "cell result evidence")
    arm_name = f"m{learner_count}"
    if (
        result.get("schema") != "yeto_parallel_cell_result_v1"
        or result.get("arm") != arm_name
    ):
        raise EvidenceError("cell result evidence has the wrong schema/arm")
    if (
        result.get("runner_exit_code") != 0
        or result.get("syncer_exit_code") != 0
        or result.get("learner_exit_codes") != [0] * learner_count
    ):
        raise EvidenceError("runner, syncer, and every configured learner must exit zero")
    endpoint = require_finite(result.get("eval_loss"), "development endpoint")
    row_loss = require_finite(expected_loss, "partial-manifest development endpoint")
    if not math.isclose(endpoint, row_loss, rel_tol=0.0, abs_tol=0.0):
        raise EvidenceError("partial-manifest endpoint differs from the result artifact")
    return endpoint


def _forbidden_p3_text(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _forbidden_p3_text(key) or _forbidden_p3_text(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_forbidden_p3_text(item) for item in value)
    if isinstance(value, str):
        lowered = value.casefold()
        return "audit" in lowered or "development" in lowered or "eval" in lowered
    return False


def _validate_training_losses(path: Path) -> None:
    rows = read_jsonl_strict(path, "P3 training-loss tape")
    if len(rows) != EXPECTED_STEPS_PER_LEARNER * 4:
        raise EvidenceError("P3 training-loss tape must contain exactly 5,120 values")
    seen: set[tuple[int, int]] = set()
    for row in rows:
        learner = row.get("learner_id")
        step = row.get("learner_step")
        if (
            isinstance(learner, bool)
            or learner not in EXPECTED_LEARNERS
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step not in range(1, EXPECTED_STEPS_PER_LEARNER + 1)
            or (learner, step) in seen
        ):
            raise EvidenceError("P3 training-loss coordinates are missing or duplicated")
        require_finite(row.get("training_loss"), "P3 training loss")
        seen.add((int(learner), int(step)))
    if len(seen) != EXPECTED_STEPS_PER_LEARNER * 4:
        raise EvidenceError("P3 training-loss tape lacks exact learner/step coverage")


def _validate_checkpoint(path: Path) -> None:
    if path.stat().st_size <= 0:
        raise EvidenceError("P3 final checkpoint is empty")
    if path.suffix.casefold() == ".json":
        value = load_json(path, "P3 checkpoint")

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, list):
                for nested in item:
                    visit(nested)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                if not math.isfinite(float(item)):
                    raise EvidenceError("P3 checkpoint contains a non-finite tensor value")

        visit(value)
        return
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - exercised only on real checkpoints.
        raise EvidenceError(f"cannot load P3 checkpoint for finite-tensor validation: {exc}") from exc

    def visit_tensor(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if not bool(torch.isfinite(item).all().item()):
                raise EvidenceError("P3 checkpoint contains a non-finite tensor")
        elif isinstance(item, Mapping):
            for nested in item.values():
                visit_tensor(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit_tensor(nested)

    visit_tensor(value)


def _validate_audit_checkpoint_inventory(
    path: Path, *, cell_id: str, attempt: int
) -> dict[str, Any]:
    value = _load_strict_object(path, "audit checkpoint inventory")
    digest = value.get("inventory_canonical_sha256")
    preimage = dict(value)
    preimage.pop("inventory_canonical_sha256", None)
    if (
        value.get("schema") != "audit_135m_checkpoint_inventory_v1"
        or value.get("cell_id") != cell_id
        or value.get("attempt") != attempt
        or value.get("loss_exposed") is not False
        or digest != canonical_sha256(preimage)
    ):
        raise EvidenceError("audit checkpoint inventory identity/hash differs")
    rows = _array(value.get("files"), "audit checkpoint files")
    if not rows:
        raise EvidenceError("audit checkpoint inventory is empty")
    attempt_root = path.parents[2].resolve()
    seen: set[str] = set()
    has_checkpoint = False
    has_export = False
    for raw in rows:
        row = _mapping(raw, "audit checkpoint file")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise EvidenceError("audit checkpoint inventory path is missing/duplicated")
        seen.add(relative)
        candidate = (attempt_root / relative).resolve()
        try:
            candidate.relative_to(attempt_root)
        except ValueError as exc:
            raise EvidenceError("audit checkpoint file escapes its attempt") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise EvidenceError("audit checkpoint file is missing or unsafe")
        if (
            row.get("sha256") != sha256_file(candidate)
            or row.get("size_bytes") != candidate.stat().st_size
        ):
            raise EvidenceError("audit checkpoint file hash/size differs")
        has_checkpoint = has_checkpoint or relative.endswith("/state.ckpt")
        has_export = has_export or "/export/" in f"/{relative}"
    if not has_checkpoint or not has_export:
        raise EvidenceError("audit checkpoint inventory lacks state.ckpt or export files")
    return {
        "checkpoint_inventory_canonical_sha256": str(digest),
        "checkpoint_file_count": len(rows),
    }


def _validate_audit_finite_kernel_capture(
    path: Path,
    *,
    expected: Mapping[str, int],
    raw_tape_path: Path,
) -> dict[str, Any]:
    value = _load_strict_object(path, "audit finite-kernel capture")
    digest = value.get("capture_canonical_sha256")
    preimage = dict(value)
    preimage.pop("capture_canonical_sha256", None)
    v_h = value.get("V_H_psd")
    if (
        value.get("schema") != "audit_135m_finite_kernel_capture_v1"
        or value.get("status") != "SEALED"
        or value.get("loss_exposed") is not False
        or digest != canonical_sha256(preimage)
        or value.get("expected_outer_steps") != expected["outer_steps"]
        or value.get("observed_outer_steps") != expected["outer_steps"]
        or value.get("K_H") != expected["outer_steps"]
        or value.get("learner_count") != expected["learner_count"]
        or value.get("updates_per_fragment")
        != expected["per_fragment_outer_updates"]
        or value.get("state_transition_replay_exact") is not True
        or value.get("all_registered_updates_covered") is not True
        or value.get("large_capture_cleanup_complete") is not True
        or value.get("event_tape_sha256") != sha256_file(raw_tape_path)
        or isinstance(v_h, bool)
        or not isinstance(v_h, (int, float))
        or not math.isfinite(float(v_h))
        or float(v_h) <= 0.0
    ):
        raise EvidenceError("audit finite-kernel capture identity/coverage differs")
    updates = _array(
        value.get("ordered_update_registry"), "finite-kernel update registry"
    )
    if (
        len(updates) != expected["outer_steps"]
        or value.get("ordered_update_registry_hash") != canonical_sha256(updates)
        or [row.get("step") for row in updates]
        != list(range(1, expected["outer_steps"] + 1))
    ):
        raise EvidenceError("audit finite-kernel ordered updates differ")
    return {
        "finite_kernel_capture_canonical_sha256": str(digest),
        "finite_kernel_K_H": expected["outer_steps"],
        "finite_kernel_V_H_psd": float(v_h),
    }


def validate_completed_attempt_work(
    *,
    stage_code: str,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    expected_command: Sequence[str],
    campaign_root: Path,
    evaluation_registry: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = verify_artifact_inventory(campaign_root, row)
    required = {
        "command",
        "attempt_start",
        "learner_steps",
        "work_events",
        "barrier_events",
        "results",
    }
    checkpoint_only = audit_checkpoint_only(cell)
    if stage_code == "p3t":
        required.update(("training_losses", "final_checkpoint"))
        forbidden = {role for role in inventory if "eval" in role or "audit" in role or "development" in role}
        if forbidden:
            raise EvidenceError("P3 training artifact inventory exposes evaluation surfaces")
    elif checkpoint_only:
        required.add("checkpoint_inventory")
    else:
        required.update(("eval_freeze", "eval_losses"))
    kernel_required = cell.get("finite_kernel_capture_required") is True
    if kernel_required:
        required.update(("finite_kernel_capture", "raw_tape"))
    missing = required - set(inventory)
    if missing:
        raise EvidenceError(f"completed attempt lacks required artifacts: {sorted(missing)}")
    if kernel_required and "finite_kernel_capture" not in inventory:
        raise EvidenceError("registered finite-kernel cell lacks its compact capture")
    if not kernel_required and "finite_kernel_capture" in inventory:
        raise EvidenceError("unregistered cell emitted a finite-kernel capture")

    command_hash = _validate_shape_projected_command(
        row=row,
        cell=cell,
        expected_command=expected_command,
        command_path=inventory["command"],
    )
    _validate_attempt_start(
        inventory["attempt_start"],
        cell_id=str(row["cell_id"]),
        attempt=int(row["attempt"]),
        command_hash=command_hash,
        provider_hash=require_sha256(
            row.get("provider_evidence_sha256"), "attempt provider evidence hash"
        ),
    )
    expected = _expected_work_for_cell(cell)
    _validate_learner_steps(
        inventory["learner_steps"],
        expected["learner_steps_per_learner"],
        expected["learner_count"],
    )
    updates = _validate_work_events(inventory["work_events"], expected)
    _validate_barrier_events(
        inventory["barrier_events"], updates, expected["learner_count"]
    )

    if stage_code == "p3t":
        result = _load_strict_object(inventory["results"], "P3 result evidence")
        if (
            result.get("schema") != "yeto_parallel_p3_training_result_v1"
            or result.get("loss") is not None
            or row.get("loss") is not None
            or result.get("evaluation_role") != "none"
            or result.get("runner_exit_code") != 0
            or result.get("syncer_exit_code") != 0
            or result.get("learner_exit_codes") != [0, 0, 0, 0]
        ):
            raise EvidenceError("P3 completed row violates the training-only result contract")
        if _forbidden_p3_text(expected_command) or _forbidden_p3_text(row.get("io_paths", [])):
            raise EvidenceError("P3 training command or synchronization path names evaluation data")
        _validate_training_losses(inventory["training_losses"])
        _validate_checkpoint(inventory["final_checkpoint"])
        return {
            "tokens": EXPECTED_TOKENS,
            "microsteps": EXPECTED_MICROSTEPS,
            "learner_steps_per_learner": EXPECTED_STEPS_PER_LEARNER,
            "loss": None,
            "checkpoint_sha256": sha256_file(inventory["final_checkpoint"]),
        }

    if checkpoint_only:
        result = _load_strict_object(
            inventory["results"], "audit checkpoint-only result evidence"
        )
        expected_learner_exits = [0] * expected["learner_count"]
        if (
            result.get("schema") != "audit_135m_checkpoint_only_result_v1"
            or result.get("loss") is not None
            or row.get("loss") is not None
            or result.get("evaluation_role") != "none"
            or result.get("runner_exit_code") != 0
            or result.get("syncer_exit_code") != 0
            or result.get("learner_exit_codes") != expected_learner_exits
        ):
            raise EvidenceError(
                "audit checkpoint-only row violates train-all-then-audit-all"
            )
        checkpoint = _validate_audit_checkpoint_inventory(
            inventory["checkpoint_inventory"],
            cell_id=str(row["cell_id"]),
            attempt=int(row["attempt"]),
        )
        kernel = (
            _validate_audit_finite_kernel_capture(
                inventory["finite_kernel_capture"],
                expected=expected,
                raw_tape_path=inventory["raw_tape"],
            )
            if kernel_required
            else {}
        )
        return {
            "tokens": expected["tokens"],
            "microsteps": expected["microsteps"],
            "learner_count": expected["learner_count"],
            "learner_steps_per_learner": expected["learner_steps_per_learner"],
            "loss": None,
            **checkpoint,
            **kernel,
        }

    endpoint = _validate_results(
        inventory["results"],
        row.get("loss"),
        learner_count=expected["learner_count"],
    )
    seed_key = str(cell["seed"])
    eval_entry = _mapping(
        evaluation_registry.get(seed_key), f"evaluation registry seed {seed_key}"
    )
    expected_eval_path = _safe_campaign_path(
        campaign_root, eval_entry.get("path"), "evaluation registry"
    )
    if inventory["eval_freeze"].resolve() != expected_eval_path.resolve():
        raise EvidenceError("attempt uses a different frozen evaluation path")
    _validate_finite_eval(
        inventory["eval_freeze"],
        inventory["eval_losses"],
        endpoint,
        evaluation_binding=eval_entry,
    )
    return {
        "tokens": expected["tokens"],
        "microsteps": expected["microsteps"],
        "learner_count": expected["learner_count"],
        "learner_steps_per_learner": expected["learner_steps_per_learner"],
        "loss": endpoint,
    }


def validate_diverged_attempt(
    *,
    row: Mapping[str, Any],
    cell: Mapping[str, Any],
    expected_command: Sequence[str],
    campaign_root: Path,
) -> dict[str, Any]:
    inventory = verify_artifact_inventory(campaign_root, row)
    required = {"command", "attempt_start", "tape_prefix", "scientific_divergence"}
    if not required.issubset(inventory):
        raise EvidenceError("DIVERGED attempt lacks command/tape/divergence evidence")
    command_hash = _validate_shape_projected_command(
        row=row,
        cell=cell,
        expected_command=expected_command,
        command_path=inventory["command"],
    )
    _validate_attempt_start(
        inventory["attempt_start"],
        cell_id=str(row["cell_id"]),
        attempt=int(row["attempt"]),
        command_hash=command_hash,
        provider_hash=require_sha256(row.get("provider_evidence_sha256"), "provider hash"),
    )
    prefix = read_jsonl_strict(inventory["tape_prefix"], "scientific tape prefix")
    for index, event in enumerate(prefix, 1):
        if event.get("outer_step") != index:
            raise EvidenceError("DIVERGED tape prefix is reordered or gapped")
    divergence = _load_strict_object(
        inventory["scientific_divergence"], "scientific divergence record"
    )
    if (
        divergence.get("schema") != "yeto_parallel_scientific_divergence_v1"
        or divergence.get("cell_id") != row.get("cell_id")
        or divergence.get("attempt_id") != row.get("attempt_id")
        or require_nonnegative_int(divergence.get("last_finite_step"), "last finite step")
        != len(prefix)
        or not isinstance(divergence.get("first_nonfinite_event"), dict)
    ):
        raise EvidenceError("scientific divergence record is incomplete")
    if row.get("loss") is not None or row.get("failure_reason") not in (None, ""):
        raise EvidenceError("DIVERGED row must retain null loss and no infrastructure reason")
    return {"diverged": True, "last_finite_step": len(prefix)}


def validate_infrastructure_attempt(
    row: Mapping[str, Any], campaign_root: Path
) -> None:
    inventory = verify_artifact_inventory(campaign_root, row)
    if "infra_failure" not in inventory:
        raise EvidenceError("INFRA_FAILURE row lacks its hashed infrastructure record")
    failure = _load_strict_object(inventory["infra_failure"], "infrastructure failure")
    reason = row.get("failure_reason")
    if (
        reason not in DIRECT_INFRASTRUCTURE_FAILURE_REASONS
        or failure.get("failure_reason") != reason
        or failure.get("attempt_id") != row.get("attempt_id")
    ):
        raise EvidenceError("infrastructure failure is not a frozen direct reason")
    if reason == "provider_spot_preemption":
        provider_id = require_numeric_id(
            failure.get("preempted_instance_numeric_id"),
            "preempted instance numeric ID",
        )
        if provider_id != str(row.get("instance_numeric_id")):
            raise EvidenceError("Spot preemption record cites a different exact instance")


def _generation_local_root(campaign_root: Path, slot: str, generation: int) -> Path:
    if slot not in ALL_LOGICAL_SLOTS:
        raise LifecycleError("VM registry has an unknown logical slot")
    require_positive_int(generation, "VM registry generation")
    return campaign_root / "vms" / slot / f"g{generation}"


def _attempt_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("actual_wave_index", -1)),
        int(row.get("launch_order_index", -1)),
        str(row.get("cell_id", "")),
    )


def _wave_manifest_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ScheduleError("cannot hash an empty wave manifest")
    ordered = [deepcopy(dict(row)) for row in sorted(rows, key=_attempt_sort_key)]
    return canonical_sha256(
        {
            "schema": "yeto_parallel_terminal_wave_v1",
            "group_id": ordered[0]["group_id"],
            "retry_round": ordered[0]["retry_round"],
            "actual_wave_index": ordered[0]["actual_wave_index"],
            "attempts": ordered,
        }
    )


def _validate_generation_capacity(
    lifecycles: Sequence[Mapping[str, Any]], *, stage_code: str = "p1r0"
) -> None:
    events: list[tuple[datetime, int, int]] = []
    for row in lifecycles:
        start = parse_time(row.get("creation_timestamp"), "generation creation")
        end = parse_time(row.get("deletion_completed_at_utc"), "generation deletion")
        if end < start:
            raise LifecycleError("generation deletion precedes creation")
        shape = machine_shape_contract(row.get("machine_type"))
        if row.get("a100_count") != shape["a100_count"]:
            raise LifecycleError("generation capacity row disagrees with its machine shape")
        # End events sort first at equal timestamps because half-open intervals
        # stop consuming capacity exactly at deletion completion.
        events.append((start, 1, shape["a100_count"]))
        events.append((end, -1, -shape["a100_count"]))
    active_vms = 0
    active_a100s = 0
    for _timestamp, vm_delta, gpu_delta in sorted(
        events, key=lambda item: (item[0], 0 if item[1] < 0 else 1)
    ):
        active_vms += vm_delta
        active_a100s += gpu_delta
        if active_vms < 0 or active_a100s < 0:
            raise LifecycleError("generation lifecycle capacity trace is inconsistent")
        if (
            active_vms > maximum_scientific_vms_for_stage(stage_code)
            or active_a100s > MAX_CAMPAIGN_A100S
        ):
            raise LifecycleError(
                "generation lifecycle exceeded the stage VM limit or 16 A100s"
            )


def _validate_scientific_concurrency(
    rows: Sequence[Mapping[str, Any]], *, stage_code: str
) -> None:
    events: list[tuple[datetime, int, str, str]] = []
    for row in rows:
        if row.get("status") not in ATTEMPT_STATUSES:
            raise ScheduleError("attempt has a status outside the frozen vocabulary")
        start = parse_time(row.get("scientific_started_at"), "scientific start")
        end = parse_time(row.get("scientific_ended_at"), "scientific end")
        if end < start:
            raise ScheduleError("scientific end precedes start")
        events.append((start, 1, str(row["logical_slot"]), str(row["cell_id"])))
        events.append((end, -1, str(row["logical_slot"]), str(row["cell_id"])))
    active_slots: set[str] = set()
    active_cells: set[str] = set()
    for _timestamp, delta, slot, cell_id in sorted(
        events, key=lambda item: (item[0], 0 if item[1] < 0 else 1)
    ):
        if delta < 0:
            if slot not in active_slots or cell_id not in active_cells:
                raise ScheduleError("scientific interval trace closes an inactive cell")
            active_slots.remove(slot)
            active_cells.remove(cell_id)
        else:
            if slot in active_slots:
                raise ScheduleError("two scientific cells overlap in one logical slot")
            if cell_id in active_cells:
                raise ScheduleError("two attempts of one cell overlap")
            active_slots.add(slot)
            active_cells.add(cell_id)
            if len(active_cells) > maximum_concurrent_cells_for_stage(stage_code):
                raise ScheduleError("scientific concurrency exceeded the stage limit")


def _validate_wave_timing(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ScheduleError("actual wave is empty")
    slot_sets = {
        tuple(_array(row.get("available_slot_set"), "available slot set"))
        for row in rows
    }
    if len(slot_sets) != 1:
        raise ScheduleError("one atomic wave mixes available-slot sets")
    available = _normalize_available_slots(
        next(iter(slot_sets)), logical_slots=ALL_LOGICAL_SLOTS
    )
    if any(str(row.get("logical_slot")) not in available for row in rows):
        raise ScheduleError("wave row is assigned outside its available-slot set")
    batches: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        batches[require_nonnegative_int(row.get("dispatch_batch_index"), "dispatch batch")].append(row)
    if sorted(batches) != list(range(len(batches))):
        raise ScheduleError("wave dispatch batches are not contiguous from zero")
    prior_batch_end: datetime | None = None
    for batch_index in sorted(batches):
        batch = sorted(
            batches[batch_index],
            key=lambda row: require_nonnegative_int(
                row.get("batch_launch_order_index"), "batch launch order"
            ),
        )
        if [int(row["batch_launch_order_index"]) for row in batch] != list(
            range(len(batch))
        ):
            raise ScheduleError("batch launch order is not contiguous from zero")
        if len({str(row["logical_slot"]) for row in batch}) != len(batch):
            raise ScheduleError("two cells share one logical slot within a dispatch batch")
        dispatches = [
            parse_time(row.get("dispatched_at"), "dispatch timestamp") for row in batch
        ]
        starts = [
            parse_time(row.get("scientific_started_at"), "scientific start timestamp")
            for row in batch
        ]
        ends = [
            parse_time(row.get("scientific_ended_at"), "scientific end timestamp")
            for row in batch
        ]
        readies = [
            parse_time(row.get("vm_ready_at"), "VM READY timestamp") for row in batch
        ]
        if dispatches != sorted(dispatches):
            raise ScheduleError("batch dispatch timestamps do not follow committed order")
        if (max(dispatches) - min(dispatches)).total_seconds() > 60:
            raise ScheduleError("dispatch-batch span exceeds 60 seconds")
        if (max(starts) - min(starts)).total_seconds() > 120:
            raise ScheduleError("dispatch-batch scientific-start span exceeds 120 seconds")
        if max(readies) > min(dispatches):
            raise ScheduleError("one or more assigned VMs were not READY before dispatch")
        if prior_batch_end is not None and min(dispatches) < prior_batch_end:
            raise ScheduleError("a reduced-width batch overlaps its prior batch")
        prior_batch_end = max(ends)


def _validate_fresh_attempt(row: Mapping[str, Any], prior: Mapping[str, Any] | None) -> None:
    fresh = _mapping(row.get("fresh_start"), "fresh-attempt declaration")
    required = {
        "same_frozen_initial_model": True,
        "same_seed_and_data_order": True,
        "same_command_and_work_budget": True,
        "resumed": False,
        "prior_optimizer_state_used": False,
        "prior_checkpoint_used": False,
        "prior_tape_used": False,
        "prior_result_used": False,
    }
    if any(fresh.get(key) != value for key, value in required.items()):
        raise ScheduleError("retry attempt is not a fresh scientific attempt")
    if prior is not None and row.get("attempt_prefix") == prior.get("attempt_prefix"):
        raise ScheduleError("retry reuses a prior create-only attempt prefix")


def _validate_retry_authorization(
    *,
    row: Mapping[str, Any],
    prior: Mapping[str, Any],
    prior_wave_rows: Sequence[Mapping[str, Any]],
    parallel_digest: str,
    retry_first_dispatch: datetime,
) -> None:
    authorization = _mapping(row.get("retry_authorization"), "retry authorization")
    expected = {
        "loss_blind": True,
        "parallel_plan_hash": parallel_digest,
        "group_id": row.get("group_id"),
        "retry_round": row.get("retry_round"),
        "prior_wave_manifest_canonical_sha256": _wave_manifest_hash(prior_wave_rows),
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise ScheduleError("retry authorization is incomplete or not bound to the prior wave")
    authorized_at = parse_time(
        authorization.get("authorized_at_utc"), "retry authorization timestamp"
    )
    prior_sealed_at = parse_time(
        prior_wave_rows[0].get("wave_terminal_prefix_sealed_at"),
        "prior wave terminal-prefix seal",
    )
    if authorized_at < prior_sealed_at or authorized_at > retry_first_dispatch:
        raise ScheduleError("retry authorization is not between prior sealing and dispatch")
    if row.get("retry_of") != prior.get("attempt_id"):
        raise ScheduleError("retry_of does not cite the immediately prior cell attempt")
    if prior.get("status") in ("COMPLETED", "DIVERGED"):
        if row.get("retry_reason") != PEER_RETRY_REASON:
            raise ScheduleError("terminal retry peer lacks the sole peer retry reason")
    elif prior.get("status") == "INFRA_FAILURE":
        if row.get("retry_reason") != prior.get("failure_reason"):
            raise ScheduleError("direct-failure retry reason differs from the prior failure")
    else:
        raise ScheduleError("FAILED rows may not be retried")


def _validate_concurrent_attempt_schedule(
    *,
    attempts: Sequence[Mapping[str, Any]],
    roster: Mapping[str, Any],
    plan: Mapping[str, Any],
    parallel_digest: str,
) -> dict[str, dict[str, Any]]:
    ordered = sorted(attempts, key=_attempt_sort_key)
    if ordered != list(attempts):
        raise ScheduleError("canonical attempt registry is not actual-wave/launch ordered")
    actual_indices = sorted({int(row["actual_wave_index"]) for row in ordered})
    if actual_indices != list(range(len(actual_indices))):
        raise ScheduleError("actual wave indices are not contiguous from zero")
    by_wave: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_wave[int(row["actual_wave_index"])].append(row)
    planned_by_group = {
        str(wave["group_id"]): (index, wave)
        for index, wave in enumerate(plan["waves"])
    }
    if len(planned_by_group) != len(plan["waves"]):
        raise ScheduleError("concurrent plan repeats a planned atomic group")
    state = {
        group_id: {"retry_round": 1, "prior_rows": None, "resolved": False}
        for group_id in planned_by_group
    }
    final_analysis: dict[str, dict[str, Any]] = {}
    contract_hash = require_sha256(
        roster.get("audit_135m_design_contract_hash"),
        "audit-135M design contract hash",
    )
    batch_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for actual_index in actual_indices:
        rows = by_wave[actual_index]
        group_ids = {str(row.get("group_id")) for row in rows}
        if len(group_ids) != 1:
            raise ScheduleError("one actual wave mixes concurrent atomic groups")
        group_id = next(iter(group_ids))
        if group_id not in planned_by_group:
            raise ScheduleError("actual concurrent wave names an unknown group")
        block_index, planned_wave = planned_by_group[group_id]
        group_state = state[group_id]
        if group_state["resolved"]:
            raise ScheduleError("resolved concurrent block has an extra attempt")
        retry_round = int(group_state["retry_round"])
        lane_sets = {
            tuple(_array(row.get("available_slot_set"), "available slot set"))
            for row in rows
        }
        batch_slot_sets = {
            tuple(
                _array(
                    row.get("concurrent_batch_slot_set"),
                    "concurrent batch slot set",
                )
            )
            for row in rows
        }
        batch_indices = {
            require_nonnegative_int(
                row.get("concurrent_batch_index"),
                "concurrent batch index",
            )
            for row in rows
        }
        if len(lane_sets) != 1 or len(batch_slot_sets) != 1 or len(batch_indices) != 1:
            raise ScheduleError("one atomic block mixes concurrent binding fields")
        lane = _normalize_available_slots(
            next(iter(lane_sets)), logical_slots=AUDIT_135M_LOGICAL_SLOTS
        )
        batch_slots = _normalize_available_slots(
            next(iter(batch_slot_sets)), logical_slots=AUDIT_135M_LOGICAL_SLOTS
        )
        batch_index = next(iter(batch_indices))
        if len(lane) != AUDIT_135M_BLOCK_WIDTH or any(
            slot not in batch_slots for slot in lane
        ):
            raise ScheduleError("concurrent atomic block lacks one complete lane")
        expected_lane = audit_concurrent_block_slots(
            contract_hash=contract_hash,
            block_index=block_index,
            available_slots=batch_slots,
        )
        if lane != expected_lane:
            raise ScheduleError("concurrent block-to-slot binding is not deterministic")
        expected = wave_for_retry(
            plan,
            roster,
            group_id,
            retry_round,
            available_slots=lane,
        )
        expected_rows = expected["assigned_cells_in_dispatch_order"]
        if len(rows) != len(expected_rows):
            raise ScheduleError("actual concurrent wave splits or mixes an atomic group")
        actual_projection = [
            (
                row.get("cell_id"),
                row.get("logical_slot"),
                row.get("launch_order_index"),
                row.get("dispatch_batch_index"),
                row.get("batch_launch_order_index"),
                row.get("retry_round"),
            )
            for row in rows
        ]
        expected_projection = [
            (
                row["cell_id"],
                row["logical_slot"],
                row["launch_order_index"],
                row["dispatch_batch_index"],
                row["batch_launch_order_index"],
                retry_round,
            )
            for row in expected_rows
        ]
        if actual_projection != expected_projection:
            raise ScheduleError(
                "manual concurrent slot swap, dispatch reorder, or mixed group detected"
            )
        if len({row.get("machine_type") for row in rows}) != 1 or len(
            {row.get("gpu_slots") for row in rows}
        ) != 1:
            raise ScheduleError(
                "paired concurrent block does not use one hardware/GPU packing profile"
            )
        if len({row.get("pairing_identity_hash") for row in rows}) != 1:
            raise ScheduleError("paired concurrent block identity differs across arms")
        packing_maps = set()
        for row in rows:
            learner_count = require_positive_int(row.get("m"), "attempt learner count")
            gpu_slots = require_positive_int(row.get("gpu_slots"), "attempt GPU slots")
            expected_gpu_map = learner_gpu_slot_map(learner_count, gpu_slots)
            if (
                row.get("learner_gpu_slot_map") != expected_gpu_map
                or row.get("maximum_learners_per_gpu")
                != max(Counter(expected_gpu_map.values()).values())
            ):
                raise ScheduleError(
                    "paired concurrent block has an invalid learner/GPU packing map"
                )
            packing_maps.add(canonical_json(expected_gpu_map))
        if len(packing_maps) != 1:
            raise ScheduleError(
                "paired concurrent block does not share one learner/GPU packing map"
            )
        if any(
            row.get("time_block_index") != planned_wave["time_block_index"]
            or row.get("attempt") != retry_round
            or row.get("retry_time_block_index")
            != (None if retry_round == 1 else actual_index)
            or row.get("attempt_id")
            != f"{row.get('cell_id')}-attempt-{retry_round}"
            for row in rows
        ):
            raise ScheduleError("concurrent wave loses group/time-block/attempt identity")
        _validate_wave_timing(rows)
        prior_rows = group_state["prior_rows"]
        for row in rows:
            _validate_fresh_attempt(
                row,
                None
                if prior_rows is None
                else next(
                    prior
                    for prior in prior_rows
                    if prior["cell_id"] == row["cell_id"]
                ),
            )
        statuses = {str(row.get("status")) for row in rows}
        if not statuses.issubset(ATTEMPT_STATUSES) or "FAILED" in statuses:
            raise ScheduleError("concurrent block has a nonsealable terminal status")
        has_infra = "INFRA_FAILURE" in statuses
        if has_infra and "DIVERGED" in statuses:
            raise ScheduleError("mixed DIVERGED/INFRA_FAILURE block cannot retry")
        if prior_rows is not None:
            authorizations = {
                canonical_json(row.get("retry_authorization")) for row in rows
            }
            if len(authorizations) != 1:
                raise ScheduleError("concurrent block retry lacks one authorization")
            first_dispatch = min(
                parse_time(row.get("dispatched_at"), "retry dispatch")
                for row in rows
            )
            for row in rows:
                prior = next(
                    prior
                    for prior in prior_rows
                    if prior["cell_id"] == row["cell_id"]
                )
                _validate_retry_authorization(
                    row=row,
                    prior=prior,
                    prior_wave_rows=prior_rows,
                    parallel_digest=parallel_digest,
                    retry_first_dispatch=first_dispatch,
                )
        elif any(
            row.get("retry_of") not in (None, "")
            or row.get("retry_reason") not in (None, "")
            or row.get("retry_authorization") not in (None, {})
            for row in rows
        ):
            raise ScheduleError("initial concurrent block declares retry lineage")
        seal_time = parse_time(
            rows[0].get("wave_terminal_prefix_sealed_at"),
            "wave terminal-prefix seal",
        )
        if any(
            parse_time(row.get("wave_terminal_prefix_sealed_at"), "wave seal")
            != seal_time
            for row in rows
        ) or max(
            parse_time(row.get("scientific_ended_at"), "scientific end")
            for row in rows
        ) > seal_time:
            raise ScheduleError("concurrent wave terminal-prefix seal is invalid")
        batch_records[batch_index].append(
            {
                "group_id": group_id,
                "block_index": block_index,
                "batch_slots": batch_slots,
                "first_dispatch": min(
                    parse_time(row.get("dispatched_at"), "dispatch") for row in rows
                ),
                "sealed_at": seal_time,
            }
        )
        if has_infra:
            if any(
                row.get("status") not in ("INFRA_FAILURE", "COMPLETED")
                for row in rows
            ):
                raise ScheduleError("concurrent infrastructure retry precondition differs")
            group_state["prior_rows"] = list(rows)
            group_state["retry_round"] = retry_round + 1
            continue
        for row in rows:
            if row.get("status") not in ("COMPLETED", "DIVERGED"):
                raise ScheduleError("concurrent analysis round is unresolved")
            final_analysis[str(row["cell_id"])] = {
                "attempt_id": row["attempt_id"],
                "attempt": row["attempt"],
                "status": row["status"],
                "group_id": group_id,
                "retry_round": retry_round,
                "analysis_loss": row.get("analysis_loss"),
                "analysis_loss_kind": row.get("analysis_loss_kind"),
                "divergence_retained": row.get("divergence_retained"),
            }
        group_state["resolved"] = True
    if any(not value["resolved"] for value in state.values()):
        raise ScheduleError("concurrent attempt registry omits a planned block")
    batch_indices = sorted(batch_records)
    if batch_indices != list(range(len(batch_indices))):
        raise ScheduleError("concurrent batch indices are not contiguous from zero")
    prior_batch_seal: datetime | None = None
    for batch_index in batch_indices:
        records = batch_records[batch_index]
        slot_sets = {record["batch_slots"] for record in records}
        block_indices = sorted({record["block_index"] for record in records})
        if len(slot_sets) != 1:
            raise ScheduleError("one concurrent batch mixes capacity slot sets")
        batch_slots = next(iter(slot_sets))
        if (
            len(block_indices) > AUDIT_135M_MAX_CONCURRENT_BLOCKS
            or block_indices
            != list(range(min(block_indices), min(block_indices) + len(block_indices)))
            or len(batch_slots) != len(block_indices) * AUDIT_135M_BLOCK_WIDTH
        ):
            raise ScheduleError("concurrent batch block/slot cardinality differs")
        first_dispatch = min(record["first_dispatch"] for record in records)
        if prior_batch_seal is not None and first_dispatch < prior_batch_seal:
            raise ScheduleError("a later concurrent batch overlaps an unsealed prior batch")
        prior_batch_seal = max(record["sealed_at"] for record in records)
    launch_ids = {str(row["cell_id"]) for row in roster["launch_cells"]}
    if set(final_analysis) != launch_ids:
        raise ScheduleError("concurrent analysis rounds do not cover every launch cell")
    _validate_scientific_concurrency(
        ordered, stage_code=str(roster.get("stage_code"))
    )
    return final_analysis


def validate_attempt_schedule(
    *,
    attempts: Sequence[Mapping[str, Any]],
    roster: Mapping[str, Any],
    plan: Mapping[str, Any],
    parallel_digest: str,
) -> dict[str, dict[str, Any]]:
    if plan.get("schema") == "yeto_parallel_plan_v4":
        return _validate_concurrent_attempt_schedule(
            attempts=attempts,
            roster=roster,
            plan=plan,
            parallel_digest=parallel_digest,
        )
    ordered = sorted(attempts, key=_attempt_sort_key)
    if ordered != list(attempts):
        raise ScheduleError("canonical attempt registry is not actual-wave/launch ordered")
    actual_indices = sorted({int(row["actual_wave_index"]) for row in ordered})
    if actual_indices != list(range(len(actual_indices))):
        raise ScheduleError("actual wave indices are not contiguous from zero")
    by_wave: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_wave[int(row["actual_wave_index"])].append(row)

    final_analysis: dict[str, dict[str, Any]] = {}
    actual_index = 0
    previous_wave_sealed_at: datetime | None = None
    for planned_wave in plan["waves"]:
        group_id = str(planned_wave["group_id"])
        retry_round = 1
        prior_rows: list[Mapping[str, Any]] | None = None
        while True:
            if actual_index not in by_wave:
                raise ScheduleError(f"missing actual wave for planned group {group_id}")
            rows = by_wave[actual_index]
            available_sets = {
                tuple(_array(row.get("available_slot_set"), "available slot set"))
                for row in rows
            }
            if len(available_sets) != 1:
                raise ScheduleError("actual wave mixes available-slot sets")
            available_slots = _normalize_available_slots(
                next(iter(available_sets)),
                logical_slots=logical_slots_for_stage(
                    str(roster.get("stage_code"))
                ),
            )
            expected = wave_for_retry(
                plan,
                roster,
                group_id,
                retry_round,
                available_slots=available_slots,
            )
            expected_rows = expected["assigned_cells_in_dispatch_order"]
            if len(rows) != len(expected_rows):
                raise ScheduleError("actual wave splits or mixes an atomic group")
            actual_projection = [
                (
                    row.get("cell_id"),
                    row.get("logical_slot"),
                    row.get("launch_order_index"),
                    row.get("dispatch_batch_index"),
                    row.get("batch_launch_order_index"),
                    row.get("retry_round"),
                )
                for row in rows
            ]
            expected_projection = [
                (
                    row["cell_id"],
                    row["logical_slot"],
                    row["launch_order_index"],
                    row["dispatch_batch_index"],
                    row["batch_launch_order_index"],
                    retry_round,
                )
                for row in expected_rows
            ]
            if actual_projection != expected_projection:
                raise ScheduleError("manual slot swap, dispatch reorder, or mixed group detected")
            if str(roster.get("stage_code")) in CONTROLLED_STAGE_CODES:
                if len({row.get("machine_type") for row in rows}) != 1 or len(
                    {row.get("gpu_slots") for row in rows}
                ) != 1:
                    raise ScheduleError(
                        "paired seed block does not use one hardware/GPU packing profile"
                    )
                if len({row.get("pairing_identity_hash") for row in rows}) != 1:
                    raise ScheduleError("paired seed block identity differs across arms")
                packing_maps = set()
                for row in rows:
                    learner_count = require_positive_int(
                        row.get("m"), "attempt learner count"
                    )
                    gpu_slots = require_positive_int(
                        row.get("gpu_slots"), "attempt GPU slots"
                    )
                    expected_gpu_map = learner_gpu_slot_map(
                        learner_count, gpu_slots
                    )
                    if (
                        row.get("learner_gpu_slot_map") != expected_gpu_map
                        or row.get("maximum_learners_per_gpu")
                        != max(Counter(expected_gpu_map.values()).values())
                    ):
                        raise ScheduleError(
                            "paired seed block has an invalid learner/GPU packing map"
                        )
                    packing_maps.add(canonical_json(expected_gpu_map))
                if len(packing_maps) != 1:
                    raise ScheduleError(
                        "paired seed block does not share one learner/GPU packing map"
                    )
            if any(
                row.get("group_id") != group_id
                or row.get("time_block_index") != planned_wave["time_block_index"]
                or row.get("attempt") != retry_round
                or row.get("retry_time_block_index")
                != (None if retry_round == 1 else actual_index)
                or row.get("attempt_id")
                != f"{row.get('cell_id')}-attempt-{retry_round}"
                for row in rows
            ):
                raise ScheduleError("wave rows do not preserve group/time-block/attempt identity")
            if previous_wave_sealed_at is not None:
                first_dispatch = min(
                    parse_time(row.get("dispatched_at"), "next wave dispatch") for row in rows
                )
                if first_dispatch < previous_wave_sealed_at:
                    raise ScheduleError("a later or retry wave overlaps an unsealed prior wave")
            _validate_wave_timing(rows)
            for row in rows:
                _validate_fresh_attempt(
                    row,
                    None
                    if prior_rows is None
                    else next(
                        prior for prior in prior_rows if prior["cell_id"] == row["cell_id"]
                    ),
                )
            statuses = {str(row.get("status")) for row in rows}
            if not statuses.issubset(ATTEMPT_STATUSES):
                raise ScheduleError("wave contains a status outside the frozen vocabulary")
            if "FAILED" in statuses:
                raise ScheduleError("FAILED is nonretryable and blocks a campaign seal")
            has_infra = "INFRA_FAILURE" in statuses
            if has_infra and "DIVERGED" in statuses:
                raise ScheduleError("mixed DIVERGED/INFRA_FAILURE wave cannot retry or seal")
            if prior_rows is not None:
                authorizations = {
                    canonical_json(row.get("retry_authorization")) for row in rows
                }
                if len(authorizations) != 1:
                    raise ScheduleError("whole-group retry rows do not share one authorization")
                for row in rows:
                    prior = next(
                        prior for prior in prior_rows if prior["cell_id"] == row["cell_id"]
                    )
                    _validate_retry_authorization(
                        row=row,
                        prior=prior,
                        prior_wave_rows=prior_rows,
                        parallel_digest=parallel_digest,
                        retry_first_dispatch=min(
                            parse_time(item.get("dispatched_at"), "retry dispatch")
                            for item in rows
                        ),
                    )
            else:
                if any(
                    row.get("retry_of") not in (None, "")
                    or row.get("retry_reason") not in (None, "")
                    or row.get("retry_authorization") not in (None, {})
                    for row in rows
                ):
                    raise ScheduleError("initial wave declares retry lineage")
            previous_wave_sealed_at = parse_time(
                rows[0].get("wave_terminal_prefix_sealed_at"), "wave terminal-prefix seal"
            )
            if any(
                parse_time(row.get("wave_terminal_prefix_sealed_at"), "wave seal")
                != previous_wave_sealed_at
                for row in rows
            ):
                raise ScheduleError("wave rows do not share one terminal-prefix seal time")
            if max(
                parse_time(row.get("scientific_ended_at"), "scientific end") for row in rows
            ) > previous_wave_sealed_at:
                raise ScheduleError("wave terminal prefix sealed before every attempt ended")

            actual_index += 1
            if has_infra:
                if any(
                    row.get("status") != "INFRA_FAILURE"
                    and row.get("status") != "COMPLETED"
                    for row in rows
                ):
                    raise ScheduleError("infrastructure retry precondition is not met")
                for row in rows:
                    if row.get("status") == "INFRA_FAILURE" and row.get(
                        "failure_reason"
                    ) not in DIRECT_INFRASTRUCTURE_FAILURE_REASONS:
                        raise ScheduleError("retry cites a non-direct infrastructure reason")
                prior_rows = list(rows)
                retry_round += 1
                continue
            for row in rows:
                if row.get("status") not in ("COMPLETED", "DIVERGED"):
                    raise ScheduleError("analysis round is not scientifically resolved")
                final_analysis[str(row["cell_id"])] = {
                    "attempt_id": row["attempt_id"],
                    "attempt": row["attempt"],
                    "status": row["status"],
                    "group_id": group_id,
                    "retry_round": retry_round,
                }
                if str(roster.get("stage_code")) in CONTROLLED_STAGE_CODES:
                    final_analysis[str(row["cell_id"])].update(
                        {
                            "analysis_loss": row.get("analysis_loss"),
                            "analysis_loss_kind": row.get("analysis_loss_kind"),
                            "divergence_retained": row.get("divergence_retained"),
                        }
                    )
            break
    if actual_index != len(by_wave):
        raise ScheduleError("attempt registry contains an extra or reordered wave")
    launch_ids = {str(row["cell_id"]) for row in roster["launch_cells"]}
    if set(final_analysis) != launch_ids:
        raise ScheduleError("analysis rounds do not cover every launch cell exactly once")
    _validate_scientific_concurrency(
        ordered, stage_code=str(roster.get("stage_code"))
    )
    return final_analysis


def _validate_inherited_prefix(
    stage_code: str,
    parent_manifest: Mapping[str, Any],
    bound_manifest: Mapping[str, Any],
) -> None:
    if stage_code == "p1r0" or stage_code in MULTISEED_STAGE_CODES:
        return
    parent_cells = _array(parent_manifest.get("expected_cells"), "parent expected cells")
    child_cells = _array(bound_manifest.get("expected_cells"), "child expected cells")
    parent_results = _array(parent_manifest.get("results"), "parent results")
    child_results = _array(bound_manifest.get("results"), "child inherited results")
    if child_cells[: len(parent_cells)] != parent_cells:
        raise ScheduleError("inherited expected cells are not an exact immutable prefix")
    if child_results[: len(parent_results)] != parent_results:
        raise ScheduleError("inherited result rows are not an exact immutable prefix")


@dataclass(frozen=True)
class CampaignBundle:
    stage_code: str
    parent_manifest: Mapping[str, Any]
    bound_manifest: Mapping[str, Any]
    scientific_plan: Mapping[str, Any]
    roster: Mapping[str, Any]
    parallel_plan: Mapping[str, Any]
    vm_registry: Mapping[str, Any]
    evaluation_registry: Mapping[str, Any]
    final_provider_census: Mapping[str, Any]
    campaign_attempt: int
    campaign_root: Path
    runtime_authorization: Mapping[str, Any] | None = None


class CampaignAggregator:
    """Read-only VM namespace verifier and sole campaign-manifest/seal writer."""

    def __init__(self, bundle: CampaignBundle) -> None:
        self.bundle = bundle
        self.campaign_root = bundle.campaign_root.resolve()
        self.campaign_dir = self.campaign_root / "campaign"
        self.roster_digest = roster_hash(bundle.roster)
        self.parallel_digest = parallel_plan_hash(bundle.parallel_plan)
        self.bound_digest = canonical_sha256(bundle.bound_manifest)
        self.scientific_digest = require_sha256(
            bundle.scientific_plan.get("randomization_plan_hash"),
            "scientific randomization plan hash",
        )
        self.divergence_loss_cap = multiseed_analysis_loss_cap(
            bundle.stage_code, bundle.bound_manifest
        )
        self.runtime_authorization_hash = multiseed_runtime_authorization_hash(
            stage_code=bundle.stage_code,
            design_contract_hash=(
                bundle.roster.get("best_paper_v2_design_contract_hash")
                or bundle.roster.get("audit_135m_design_contract_hash")
            ),
            roster_digest=self.roster_digest,
            parallel_digest=self.parallel_digest,
            bound_digest=self.bound_digest,
            scientific_digest=self.scientific_digest,
            authorization=bundle.runtime_authorization,
            hard_ceiling_usd=bundle.roster.get("hard_ceiling_usd"),
        )

    def _verify_common_bindings(self) -> None:
        if self.bundle.stage_code not in ALLOWED_STAGE_CODES:
            raise ScheduleError("campaign stage is outside the amendment")
        rebuilt_roster = build_parallel_roster(
            stage_code=self.bundle.stage_code,
            bound_manifest=self.bundle.bound_manifest,
            parent_manifest=self.bundle.parent_manifest,
            scientific_plan=self.bundle.scientific_plan,
        )
        if canonical_json(rebuilt_roster) != canonical_json(self.bundle.roster):
            raise ScheduleError("campaign roster differs from deterministic reconstruction")
        rebuilt_plan = build_parallel_plan(
            self.bundle.roster, expected_roster_hash=self.roster_digest
        )
        if canonical_json(rebuilt_plan) != canonical_json(self.bundle.parallel_plan):
            raise ScheduleError("campaign parallel plan differs from deterministic reconstruction")
        frozen = _mapping(self.bundle.bound_manifest.get("frozen"), "bound frozen")
        if frozen.get("randomization_plan_hash") != self.scientific_digest:
            raise ScheduleError("bound descendant cites a different scientific plan")
        if self.bundle.stage_code == "p3t":
            if self.bundle.evaluation_registry not in ({}, None):
                raise EvidenceError("P3 training must not bind a development/audit evaluation registry")
        else:
            launch_seeds = {
                str(cell["seed"]) for cell in self.bundle.roster["launch_cells"]
            }
            if set(self.bundle.evaluation_registry) != launch_seeds:
                raise EvidenceError("evaluation registry does not cover exactly the launch seeds")
            for seed in launch_seeds:
                entry = _mapping(
                    self.bundle.evaluation_registry[seed],
                    f"evaluation registry seed {seed}",
                )
                require_sha256(entry.get("sha256"), "evaluation registry raw SHA-256")
                if any(entry.get(field) != frozen.get(field) for field in EVAL_BOUND_FIELDS):
                    raise EvidenceError(
                        "evaluation registry identities differ from the bound descendant"
                    )
        _validate_inherited_prefix(
            self.bundle.stage_code,
            self.bundle.parent_manifest,
            self.bundle.bound_manifest,
        )
        if self.bundle.campaign_attempt <= 0:
            raise LifecycleError("campaign attempt must be positive")

    def _generation_rows(self) -> list[dict[str, Any]]:
        registry = _mapping(self.bundle.vm_registry, "VM registry")
        if (
            registry.get("schema") != "yeto_parallel_vm_registry_v1"
            or registry.get("stage_code") != self.bundle.stage_code
            or registry.get("roster_hash") != self.roster_digest
            or registry.get("campaign_attempt") != self.bundle.campaign_attempt
        ):
            raise LifecycleError("VM registry common identity differs from the campaign")
        rows = [
            _mapping(row, "VM generation registry row")
            for row in _array(registry.get("generations"), "VM generations")
        ]
        if not rows:
            raise LifecycleError("VM registry is empty")
        seen_keys: set[tuple[str, int]] = set()
        seen_namespaces: set[str] = set()
        seen_state_paths: set[str] = set()
        for row in rows:
            slot = row.get("slot")
            generation = require_positive_int(row.get("generation"), "VM generation")
            if (
                slot not in logical_slots_for_stage(self.bundle.stage_code)
                or (str(slot), generation) in seen_keys
            ):
                raise LifecycleError("VM registry repeats or misnames a slot/generation")
            seen_keys.add((str(slot), generation))
            expected_run_id = physical_run_id(
                self.bundle.stage_code,
                self.roster_digest,
                self.bundle.campaign_attempt,
                str(slot),
                generation,
            )
            if row.get("run_id") != expected_run_id:
                raise LifecycleError("VM registry run ID violates physical-generation grammar")
            if row.get("zone") not in ALLOWED_US_A100_ZONES:
                raise LifecycleError("VM registry lacks an admissible landed US A100 zone")
            recorded_region = row.get("region")
            if recorded_region is None:
                if row.get("zone") not in REVISION_1_2_ALLOWED_ZONES:
                    raise LifecycleError("VM registry omits the required landed region")
            elif recorded_region != region_for_zone(row.get("zone")):
                raise LifecycleError("VM registry region does not match its landed zone")
            machine_shape_contract(row.get("machine_type"))
            nonce = row.get("ownership_nonce")
            if not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
                raise LifecycleError("VM registry nonce is not a fresh 128-bit hex token")
            expected_labels = {
                "campaign-tag": self.roster_digest[:16],
                "logical-slot": str(slot),
                "physical-generation": str(generation),
                "run-id": expected_run_id,
                "ownership-nonce": nonce,
            }
            if row.get("labels") != expected_labels:
                raise LifecycleError("VM registry ownership labels are not exact")
            prefix = row.get("artifact_prefix")
            state_path = row.get("state_path")
            expected_suffix = f"/vms/{slot}/g{generation}/"
            if (
                not isinstance(prefix, str)
                or not prefix.endswith(expected_suffix)
                or prefix in seen_namespaces
                or not isinstance(state_path, str)
                or not state_path.endswith(f"/{expected_run_id}.json")
                or state_path in seen_state_paths
            ):
                raise LifecycleError("VM registry namespaces are not pairwise-disjoint/exact")
            expected_paths = {
                "provider_record_path": _join_artifact_root(
                    prefix, "provider/provider-evidence.json"
                ),
                "partial_manifest_path": _join_artifact_root(
                    prefix, "manifests/vm-partial-manifest.json"
                ),
                "lifecycle_record_path": _join_artifact_root(
                    prefix, "manifests/vm-lifecycle-final.json"
                ),
            }
            if any(row.get(field) != value for field, value in expected_paths.items()):
                raise LifecycleError("VM registry outer artifact paths cross namespaces")
            seen_namespaces.add(prefix)
            seen_state_paths.add(state_path)
        for slot in logical_slots_for_stage(self.bundle.stage_code):
            generations = sorted(
                generation for seen_slot, generation in seen_keys if seen_slot == slot
            )
            if generations and generations != list(range(1, max(generations) + 1)):
                raise LifecycleError("physical generations are gapped or reused in one slot")
        return sorted(rows, key=lambda row: (row["slot"], row["generation"]))

    def _load_vm_evidence(
        self, generation_rows: Sequence[Mapping[str, Any]]
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        attempts: list[dict[str, Any]] = []
        partial_hashes: list[dict[str, Any]] = []
        lifecycle_hashes: list[dict[str, Any]] = []
        lifecycle_intervals: list[dict[str, Any]] = []
        run_ids: set[str] = set()
        nonces: set[str] = set()
        instance_ids: set[str] = set()
        disk_ids: set[str] = set()
        for identity in generation_rows:
            slot = str(identity["slot"])
            generation = int(identity["generation"])
            local_root = _generation_local_root(self.campaign_root, slot, generation)
            provider_path = local_root / "provider" / "provider-evidence.json"
            partial_path = local_root / "manifests" / "vm-partial-manifest.json"
            partial_hash_path = local_root / "manifests" / "vm-partial-manifest.sha256"
            lifecycle_path = local_root / "manifests" / "vm-lifecycle-final.json"
            if not all(path.is_file() for path in (provider_path, partial_path, partial_hash_path, lifecycle_path)):
                raise LifecycleError(
                    f"VM {slot}/g{generation} lacks provider, partial hash-lock, or teardown evidence"
                )
            provider = _load_strict_object(provider_path, "provider record")
            provider_summary = validate_provider_record(provider, identity)
            provider_hash = sha256_file(provider_path)
            partial = _load_strict_object(partial_path, "VM partial manifest")
            partial_hash = sha256_file(partial_path)
            expected_sidecar = f"{partial_hash}  vm-partial-manifest.json\n"
            if partial_hash_path.read_text() != expected_sidecar:
                raise LifecycleError("VM partial-manifest hash lock is missing or tampered")
            expected_partial = {
                "status": "vm_partial_hash_locked",
                "stage_code": self.bundle.stage_code,
                "study_id": self.bundle.bound_manifest["study_id"],
                "run_id": identity["run_id"],
                "slot": slot,
                "generation": generation,
                "ownership_nonce": identity["ownership_nonce"],
                "provider_record_sha256": provider_hash,
                "roster_hash": self.roster_digest,
                "parallel_plan_hash": self.parallel_digest,
                "bound_manifest_canonical_sha256": self.bound_digest,
                "scientific_randomization_plan_hash": self.scientific_digest,
                "partial_outcomes_exposed": False,
            }
            if self.runtime_authorization_hash is not None:
                expected_partial["multiseed_runtime_authorization_hash"] = (
                    self.runtime_authorization_hash
                )
            if partial.get("status") == "sealed_results":
                raise LifecycleError("VM partial manifest may not masquerade as sealed_results")
            if any(partial.get(key) != value for key, value in expected_partial.items()):
                raise LifecycleError("VM partial manifest common identity differs")
            generation_amendment = partial.get("amendment_raw_sha256")
            if generation_amendment not in ACCEPTED_GENERATION_AMENDMENT_RAW_SHA256S:
                raise LifecycleError("VM partial manifest cites an unauthorized amendment")
            if (
                generation_amendment == REVISION_1_2_AMENDMENT_RAW_SHA256
                and provider.get("zone") not in REVISION_1_2_ALLOWED_ZONES
            ):
                raise LifecycleError("Version 1.2 generation evidence escaped us-central1")
            if generation_amendment == AMENDMENT_RAW_SHA256 and (
                identity.get("region") is None or provider.get("region") is None
            ):
                raise LifecycleError("Version 1.3 generation evidence omits landed region")
            partial_attempts = [
                _mapping(row, "VM partial attempt")
                for row in _array(partial.get("attempts"), "VM partial attempts")
            ]
            if partial_attempts != sorted(partial_attempts, key=_attempt_sort_key):
                raise LifecycleError("VM partial attempts are not append-only ordered")
            for row in partial_attempts:
                if (
                    row.get("run_id") != identity["run_id"]
                    or row.get("logical_slot") != slot
                    or row.get("generation") != generation
                    or row.get("ownership_nonce") != identity["ownership_nonce"]
                    or row.get("machine_type") != identity.get("machine_type")
                    or row.get("provider_evidence_sha256") != provider_hash
                    or str(row.get("instance_numeric_id"))
                    != provider_summary["instance_numeric_id"]
                ):
                    raise LifecycleError("attempt cites a substituted provider/generation")
                prefix = str(row.get("attempt_prefix", ""))
                expected_suffix = (
                    f"cells/{row.get('cell_id')}/attempt-{row.get('attempt')}/"
                )
                if not prefix.startswith(str(identity["artifact_prefix"]).rstrip("/") + "/") or not prefix.endswith(expected_suffix):
                    raise LifecycleError("attempt writes outside its generation namespace")
                attempts.append(dict(row))
            lifecycle = _load_strict_object(lifecycle_path, "VM lifecycle final")
            if (
                generation_amendment == AMENDMENT_RAW_SHA256
                and lifecycle.get("region") is None
            ):
                raise LifecycleError("Version 1.3 lifecycle evidence omits landed region")
            lifecycle_summary = validate_lifecycle_record(
                lifecycle, identity, provider, partial_hash
            )
            lifecycle_intervals.append(lifecycle_summary)
            partial_hashes.append(
                {"slot": slot, "generation": generation, "sha256": partial_hash}
            )
            lifecycle_hashes.append(
                {
                    "slot": slot,
                    "generation": generation,
                    "sha256": sha256_file(lifecycle_path),
                }
            )
            if identity["run_id"] in run_ids or identity["ownership_nonce"] in nonces:
                raise LifecycleError("VM registry reuses a run ID or ownership nonce")
            if (
                provider_summary["instance_numeric_id"] in instance_ids
                or provider_summary["boot_disk_numeric_id"] in disk_ids
            ):
                raise LifecycleError("VM generations reuse an instance or disk numeric ID")
            run_ids.add(str(identity["run_id"]))
            nonces.add(str(identity["ownership_nonce"]))
            instance_ids.add(provider_summary["instance_numeric_id"])
            disk_ids.add(provider_summary["boot_disk_numeric_id"])
        _validate_generation_capacity(
            lifecycle_intervals, stage_code=self.bundle.stage_code
        )
        return attempts, partial_hashes, lifecycle_hashes, lifecycle_intervals

    def _validate_work(
        self,
        attempts: Sequence[Mapping[str, Any]],
        analysis_rounds: Mapping[str, Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        scientific_by_id = _scientific_cells(self.bundle.scientific_plan)
        roster_by_id = {
            str(row["cell_id"]): row for row in self.bundle.roster["launch_cells"]
        }
        work_reports: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        for row in attempts:
            cell_id = str(row.get("cell_id"))
            if cell_id not in scientific_by_id or cell_id not in roster_by_id:
                raise EvidenceError("attempt cell is absent from the launch roster")
            scientific = scientific_by_id[cell_id]
            merged_cell = {**roster_by_id[cell_id], **scientific}
            status = row.get("status")
            if status == "COMPLETED":
                report = validate_completed_attempt_work(
                    stage_code=self.bundle.stage_code,
                    row=row,
                    cell=merged_cell,
                    expected_command=scientific["command"],
                    campaign_root=self.campaign_root,
                    evaluation_registry=self.bundle.evaluation_registry,
                )
                if self.bundle.stage_code == "p3t":
                    checkpoints.append(
                        {
                            "cell_id": cell_id,
                            "attempt_id": row["attempt_id"],
                            "status": "COMPLETED",
                            "checkpoint_sha256": report["checkpoint_sha256"],
                        }
                    )
                elif audit_checkpoint_only(merged_cell):
                    checkpoints.append(
                        {
                            "cell_id": cell_id,
                            "attempt_id": row["attempt_id"],
                            "status": "COMPLETED",
                            "checkpoint_inventory_canonical_sha256": report[
                                "checkpoint_inventory_canonical_sha256"
                            ],
                            "checkpoint_file_count": report[
                                "checkpoint_file_count"
                            ],
                            "evaluation_mode": merged_cell["evaluation_mode"],
                            "finite_kernel_capture_canonical_sha256": report.get(
                                "finite_kernel_capture_canonical_sha256"
                            ),
                            "finite_kernel_K_H": report.get("finite_kernel_K_H"),
                            "finite_kernel_V_H_psd": report.get(
                                "finite_kernel_V_H_psd"
                            ),
                        }
                    )
            elif status == "DIVERGED":
                report = validate_diverged_attempt(
                    row=row,
                    cell=merged_cell,
                    expected_command=scientific["command"],
                    campaign_root=self.campaign_root,
                )
                if self.bundle.stage_code == "p3t":
                    checkpoints.append(
                        {
                            "cell_id": cell_id,
                            "attempt_id": row["attempt_id"],
                            "status": "DIVERGED",
                            "checkpoint_sha256": None,
                        }
                    )
                elif audit_checkpoint_only(merged_cell):
                    checkpoints.append(
                        {
                            "cell_id": cell_id,
                            "attempt_id": row["attempt_id"],
                            "status": "DIVERGED",
                            "checkpoint_inventory_canonical_sha256": None,
                            "checkpoint_file_count": 0,
                            "evaluation_mode": merged_cell["evaluation_mode"],
                            "finite_kernel_capture_canonical_sha256": None,
                            "finite_kernel_K_H": None,
                            "finite_kernel_V_H_psd": None,
                        }
                    )
            elif status == "INFRA_FAILURE":
                validate_infrastructure_attempt(row, self.campaign_root)
                report = {"infrastructure_failure": row["failure_reason"]}
            else:
                raise EvidenceError("FAILED attempt blocks a campaign seal")
            if self.divergence_loss_cap is not None and status in (
                "COMPLETED",
                "DIVERGED",
            ):
                expected_analysis = multiseed_analysis_outcome_fields(
                    status=str(status),
                    loss=None if status == "DIVERGED" else report.get("loss"),
                    divergence_loss_cap=self.divergence_loss_cap,
                    checkpoint_only=audit_checkpoint_only(merged_cell),
                )
                if (
                    row.get("analysis_loss") != expected_analysis["analysis_loss"]
                    or row.get("analysis_loss_kind")
                    != expected_analysis["analysis_loss_kind"]
                    or row.get("divergence_retained")
                    is not expected_analysis["divergence_retained"]
                ):
                    raise EvidenceError(
                        "multiseed analysis row does not retain the exact capped-loss policy"
                    )
                report = {**report, **expected_analysis}
            work_reports.append(
                {"attempt_id": row["attempt_id"], "status": status, "report": report}
            )
        for cell_id, analysis in analysis_rounds.items():
            matches = [row for row in attempts if row.get("attempt_id") == analysis["attempt_id"]]
            if len(matches) != 1 or matches[0].get("status") not in ("COMPLETED", "DIVERGED"):
                raise EvidenceError(f"analysis round for {cell_id} lacks one terminal work proof")
        checkpoint_registry = None
        if self.bundle.stage_code == "p3t":
            if len(checkpoints) != 32 or {row["cell_id"] for row in checkpoints} != set(
                analysis_rounds
            ):
                raise EvidenceError("P3 checkpoint registry does not cover exactly 32 cells")
            checkpoints.sort(key=lambda row: row["cell_id"].encode("utf-8"))
            checkpoint_registry = {
                "schema": "yeto_p3_checkpoint_registry_v1",
                "cells": checkpoints,
                "checkpoint_registry_hash": canonical_sha256(checkpoints),
            }
        elif checkpoints:
            expected_checkpoint_cells = {
                cell_id
                for cell_id, cell in scientific_by_id.items()
                if audit_checkpoint_only(cell)
            }
            if {row["cell_id"] for row in checkpoints} != expected_checkpoint_cells:
                raise EvidenceError(
                    "audit checkpoint registry does not cover the exact checkpoint-only suffix"
                )
            checkpoints.sort(key=lambda row: row["cell_id"].encode("utf-8"))
            checkpoint_registry = {
                "schema": "audit_135m_checkpoint_registry_v1",
                "loss_exposed": False,
                "cells": checkpoints,
                "checkpoint_registry_hash": canonical_sha256(checkpoints),
            }
        return work_reports, checkpoint_registry

    def build_manifest_and_seal(
        self, *, sealed_at_utc: str | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._verify_common_bindings()
        generations = self._generation_rows()
        attempts, partial_hashes, lifecycle_hashes, _lifecycle_intervals = (
            self._load_vm_evidence(generations)
        )
        attempts = sorted(attempts, key=_attempt_sort_key)
        analysis_rounds = validate_attempt_schedule(
            attempts=attempts,
            roster=self.bundle.roster,
            plan=self.bundle.parallel_plan,
            parallel_digest=self.parallel_digest,
        )
        work_reports, checkpoint_registry = self._validate_work(attempts, analysis_rounds)
        census = _mapping(self.bundle.final_provider_census, "final provider census")
        if (
            census.get("schema") != "yeto_parallel_final_provider_census_v1"
            or census.get("campaign_owned_vm_count") != 0
            or census.get("campaign_owned_attached_a100s") != 0
        ):
            raise LifecycleError("final provider census is not zero for the campaign")
        canonical_vm_registry = deepcopy(dict(self.bundle.vm_registry))
        canonical_vm_registry["generations"] = [deepcopy(dict(row)) for row in generations]
        manifest = {
            "schema": "yeto_parallel_campaign_manifest_v1",
            "stage_code": self.bundle.stage_code,
            "study_id": self.bundle.bound_manifest["study_id"],
            "campaign_attempt": self.bundle.campaign_attempt,
            "amendment_raw_sha256": AMENDMENT_RAW_SHA256,
            "bound_manifest_canonical_sha256": self.bound_digest,
            "parent_manifest_canonical_sha256": canonical_sha256(
                self.bundle.parent_manifest
            ),
            "roster_hash": self.roster_digest,
            "parallel_plan_hash": self.parallel_digest,
            "scientific_randomization_plan_hash": self.scientific_digest,
            "vm_registry": canonical_vm_registry,
            "attempts": [deepcopy(dict(row)) for row in attempts],
            "analysis_rounds": {
                key: analysis_rounds[key] for key in _utf8_sort(analysis_rounds)
            },
            "work_evidence_reports": work_reports,
            "evaluation_registry": deepcopy(dict(self.bundle.evaluation_registry)),
            "final_provider_census": deepcopy(dict(census)),
            "partial_outcomes_exposed": False,
        }
        if checkpoint_registry is not None:
            if self.bundle.stage_code == "p3t":
                manifest["p3_checkpoint_registry"] = checkpoint_registry
            else:
                manifest["audit_checkpoint_registry"] = checkpoint_registry
        if self.runtime_authorization_hash is not None:
            manifest.update(
                {
                    "multiseed_runtime_authorization_hash": (
                        self.runtime_authorization_hash
                    ),
                    "analysis_policy": deepcopy(
                        dict(self.bundle.bound_manifest["analysis_policy"])
                    ),
                }
            )
        manifest_hash = canonical_sha256(manifest)
        vm_registry_hash = canonical_sha256(canonical_vm_registry)
        bound_rows, _bound_by_id = _bound_cells(self.bundle.bound_manifest)
        launch_count = len(self.bundle.roster["launch_cells"])
        seal_time = sealed_at_utc or utc_now()
        parse_time(seal_time, "campaign seal timestamp")
        seal = {
            "schema": "yeto_parallel_campaign_seal_v1",
            "status": "sealed_results",
            "stage_code": self.bundle.stage_code,
            "study_id": self.bundle.bound_manifest["study_id"],
            "authoritative_prereg_template_sha256": AUTHORITATIVE_PREREG_TEMPLATE_SHA256,
            "amendment_raw_sha256": AMENDMENT_RAW_SHA256,
            "bound_manifest_canonical_sha256": self.bound_digest,
            "roster_hash": self.roster_digest,
            "parallel_plan_hash": self.parallel_digest,
            "scientific_randomization_plan_hash": self.scientific_digest,
            "campaign_manifest_canonical_sha256": manifest_hash,
            "vm_registry_canonical_sha256": vm_registry_hash,
            "vm_partial_manifest_hashes": partial_hashes,
            "vm_lifecycle_record_hashes": lifecycle_hashes,
            "cumulative_expected_cell_count": len(bound_rows),
            "launch_cell_count": launch_count,
            "resolved_launch_cell_count": len(analysis_rounds),
            "attempt_count": len(attempts),
            "work_evidence_all_pass": True,
            "schedule_all_pass": True,
            "provider_ownership_all_pass": True,
            "exact_id_teardown_all_pass": True,
            "partial_outcomes_exposed": False,
            "sealed_at_utc": seal_time,
        }
        if self.runtime_authorization_hash is not None:
            seal.update(
                {
                    "multiseed_runtime_authorization_hash": (
                        self.runtime_authorization_hash
                    ),
                    "analysis_policy_canonical_sha256": canonical_sha256(
                        self.bundle.bound_manifest["analysis_policy"]
                    ),
                }
            )
        expected_seal_fields = {
            "schema",
            "status",
            "stage_code",
            "study_id",
            "authoritative_prereg_template_sha256",
            "amendment_raw_sha256",
            "bound_manifest_canonical_sha256",
            "roster_hash",
            "parallel_plan_hash",
            "scientific_randomization_plan_hash",
            "campaign_manifest_canonical_sha256",
            "vm_registry_canonical_sha256",
            "vm_partial_manifest_hashes",
            "vm_lifecycle_record_hashes",
            "cumulative_expected_cell_count",
            "launch_cell_count",
            "resolved_launch_cell_count",
            "attempt_count",
            "work_evidence_all_pass",
            "schedule_all_pass",
            "provider_ownership_all_pass",
            "exact_id_teardown_all_pass",
            "partial_outcomes_exposed",
            "sealed_at_utc",
        }
        if self.runtime_authorization_hash is not None:
            expected_seal_fields.update(
                {
                    "multiseed_runtime_authorization_hash",
                    "analysis_policy_canonical_sha256",
                }
            )
        if set(seal) != expected_seal_fields:
            raise AssertionError("campaign seal field set drifted")
        return manifest, seal

    def seal(self, *, sealed_at_utc: str | None = None) -> dict[str, Any]:
        manifest_path = self.campaign_dir / "campaign-manifest.json"
        seal_path = self.campaign_dir / "campaign-seal.json"
        if manifest_path.exists() or seal_path.exists():
            raise SealError("campaign manifest/seal is create-only and already exists")
        manifest, seal = self.build_manifest_and_seal(sealed_at_utc=sealed_at_utc)
        write_json_create_only(manifest_path, manifest)
        # If seal creation fails, retain the immutable unsealed campaign
        # manifest at its original create-only path.  A later attempt must use
        # a new descendant/namespace rather than repair or overwrite it.
        write_json_create_only(seal_path, seal)
        return seal


def build_audit_checkpoint_preseal(
    *,
    stage_code: str,
    parent_manifest: Mapping[str, Any],
    bound_manifest: Mapping[str, Any],
    scientific_plan: Mapping[str, Any],
    roster: Mapping[str, Any],
    parallel_plan: Mapping[str, Any],
    vm_registry: Mapping[str, Any],
    evaluation_registry: Mapping[str, Any],
    campaign_attempt: int,
    campaign_root: Path,
    runtime_authorization: Mapping[str, Any],
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Seal complete checkpoint/work coverage before hidden batch evaluation.

    This is deliberately not a provider-lifecycle seal: one already-working
    Spot generation may remain alive to execute the bound hidden batch.  The
    final campaign seal must later reproduce this exact checkpoint registry and
    add exact-ID teardown evidence.
    """

    if stage_code not in AUDIT_135M_STAGE_CODES:
        raise ScheduleError("checkpoint preseal is restricted to audit-135M stages")
    root = campaign_root.resolve()
    generations = _array(vm_registry.get("generations"), "preseal VM generations")
    attempts: list[dict[str, Any]] = []
    partial_rows = []
    expected_common = {
        "stage_code": stage_code,
        "study_id": bound_manifest.get("study_id"),
        "roster_hash": roster_hash(roster),
        "parallel_plan_hash": parallel_plan_hash(parallel_plan),
        "bound_manifest_canonical_sha256": canonical_sha256(bound_manifest),
        "scientific_randomization_plan_hash": scientific_plan.get(
            "randomization_plan_hash"
        ),
        "partial_outcomes_exposed": False,
    }
    for raw_identity in generations:
        identity = _mapping(raw_identity, "preseal generation")
        slot = str(identity.get("slot"))
        generation = require_positive_int(
            identity.get("generation"), "preseal generation number"
        )
        local_root = _generation_local_root(root, slot, generation)
        provider_path = local_root / "provider" / "provider-evidence.json"
        partial_path = local_root / "manifests" / "vm-partial-manifest.json"
        if not provider_path.is_file() or not partial_path.is_file():
            raise LifecycleError(
                f"preseal generation {slot}/g{generation} lacks provider/partial evidence"
            )
        provider = _load_strict_object(provider_path, "preseal provider evidence")
        validate_provider_record(provider, identity)
        partial = _load_strict_object(partial_path, "preseal partial manifest")
        if partial.get("status") not in {
            "collecting_attempts",
            "vm_partial_hash_locked",
        } or any(partial.get(key) != value for key, value in expected_common.items()):
            raise LifecycleError("checkpoint preseal partial identity differs")
        rows = [
            _mapping(row, "preseal partial attempt")
            for row in _array(partial.get("attempts"), "preseal partial attempts")
        ]
        if rows != sorted(rows, key=_attempt_sort_key):
            raise LifecycleError("checkpoint preseal partial attempts are reordered")
        attempts.extend(rows)
        partial_rows.append(
            {
                "slot": slot,
                "generation": generation,
                "status": partial["status"],
                "partial_manifest_raw_sha256": sha256_file(partial_path),
                "attempt_count": len(rows),
            }
        )
    attempts.sort(key=_attempt_sort_key)
    analysis_rounds = validate_attempt_schedule(
        attempts=attempts,
        roster=roster,
        plan=parallel_plan,
        parallel_digest=parallel_plan_hash(parallel_plan),
    )
    bundle = CampaignBundle(
        stage_code=stage_code,
        parent_manifest=parent_manifest,
        bound_manifest=bound_manifest,
        scientific_plan=scientific_plan,
        roster=roster,
        parallel_plan=parallel_plan,
        vm_registry=vm_registry,
        evaluation_registry=evaluation_registry,
        final_provider_census={
            "schema": "yeto_parallel_final_provider_census_v1",
            "campaign_owned_vm_count": 0,
            "campaign_owned_attached_a100s": 0,
        },
        campaign_attempt=campaign_attempt,
        campaign_root=root,
        runtime_authorization=runtime_authorization,
    )
    aggregator = CampaignAggregator(bundle)
    work_reports, checkpoint_registry = aggregator._validate_work(
        attempts, analysis_rounds
    )
    if checkpoint_registry is None or checkpoint_registry.get("schema") != (
        "audit_135m_checkpoint_registry_v1"
    ):
        raise EvidenceError("audit checkpoint preseal lacks an exact registry")
    scientific_by_id = _scientific_cells(scientific_plan)
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
    seal_time = sealed_at_utc or utc_now()
    seal_datetime = parse_time(seal_time, "checkpoint preseal timestamp")
    terminal_times = [
        parse_time(row.get("scientific_ended_at"), "preseal scientific end")
        for row in attempts
        if row.get("status") in ("COMPLETED", "DIVERGED")
    ]
    if not terminal_times:
        raise EvidenceError("checkpoint preseal has no terminal scientific work")
    maximum_completion_datetime = max(terminal_times)
    if seal_datetime <= maximum_completion_datetime:
        raise EvidenceError(
            "checkpoint preseal must follow the maximum terminal training completion"
        )
    maximum_completion = maximum_completion_datetime.isoformat().replace(
        "+00:00", "Z"
    )
    value = {
        "schema": "audit_135m_checkpoint_preseal_v1",
        "status": "SEALED_TRAINING_AND_CHECKPOINT_REGISTRY",
        "stage_code": stage_code,
        "study_id": bound_manifest["study_id"],
        "campaign_attempt": campaign_attempt,
        "loss_exposed": False,
        "provider_lifecycle_final_pending": True,
        "bound_manifest_canonical_sha256": canonical_sha256(bound_manifest),
        "parent_manifest_canonical_sha256": canonical_sha256(parent_manifest),
        "roster_hash": roster_hash(roster),
        "parallel_plan_hash": parallel_plan_hash(parallel_plan),
        "scientific_randomization_plan_hash": scientific_plan[
            "randomization_plan_hash"
        ],
        "multiseed_runtime_authorization_hash": aggregator.runtime_authorization_hash,
        "attempts_canonical_sha256": canonical_sha256(attempts),
        "attempts": [deepcopy(dict(row)) for row in attempts],
        "analysis_rounds": {
            key: analysis_rounds[key] for key in _utf8_sort(analysis_rounds)
        },
        "work_evidence_reports": work_reports,
        "evaluation_registry": deepcopy(dict(evaluation_registry)),
        "audit_checkpoint_registry": checkpoint_registry,
        "evaluation_required_cell_ids": pending,
        "capture_only_cell_ids": capture_only,
        "partial_manifests": partial_rows,
        "partial_outcomes_exposed": False,
        "maximum_training_completion_utc": maximum_completion,
        "sealed_at_utc": seal_time,
    }
    value["preseal_canonical_sha256"] = canonical_sha256(value)
    return value


@dataclass(frozen=True)
class DispatchRequest:
    group_id: str
    cell_id: str
    retry_round: int
    actual_wave_index: int
    concurrent_batch_index: int
    concurrent_batch_slot_set: tuple[str, ...]
    time_block_index: int
    retry_time_block_index: int | None
    available_slot_set: tuple[str, ...]
    dispatch_batch_index: int
    batch_launch_order_index: int
    launch_order_index: int
    attempt_prefix: str
    learner_count: int
    quorum: int
    gpu_slots: int
    learner_gpu_slot_map: Mapping[str, int]
    pairing_identity_hash: str | None
    command: tuple[str, ...]
    command_hash: str
    fresh_start: Mapping[str, Any]
    retry_of: str | None
    retry_reason: str | None
    retry_authorization: Mapping[str, Any] | None


class ParallelExecutionBackend(Protocol):
    """Provider/cell adapter used by ``ParallelWaveExecutor``.

    ``dispatch`` must return after issuing a nonblocking start message.  It may
    not wait for scientific completion; ``collect`` performs that wait.
    """

    def provision(self, identity: GenerationIdentity) -> Mapping[str, Any]: ...

    def ready(
        self, identity: GenerationIdentity, provider_record: Mapping[str, Any]
    ) -> str: ...

    def dispatch(self, identity: GenerationIdentity, request: DispatchRequest) -> str: ...

    def collect(
        self, identity: GenerationIdentity, request: DispatchRequest
    ) -> Mapping[str, Any]: ...

    def finalize_generation(
        self,
        identity: GenerationIdentity,
        provider_record: Mapping[str, Any],
        partial_manifest_sha256: str,
        *,
        preempted: bool,
    ) -> Mapping[str, Any]: ...

    def census(self, roster_tag: str) -> Mapping[str, Any]: ...


class ParallelWaveExecutor:
    """Drive sequential atomic waves across at most four physical VMs."""

    def __init__(
        self,
        *,
        roster: Mapping[str, Any],
        parallel_plan: Mapping[str, Any],
        scientific_plan: Mapping[str, Any],
        bound_manifest: Mapping[str, Any],
        registry: CampaignGenerationRegistry,
        campaign_root: Path,
        backend: ParallelExecutionBackend,
        available_slots: Sequence[str] | None = None,
        runtime_authorization: Mapping[str, Any] | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.roster = roster
        self.plan = parallel_plan
        self.scientific_plan = scientific_plan
        self.bound_manifest = bound_manifest
        self.registry = registry
        self.campaign_root = campaign_root.resolve()
        self.backend = backend
        self.clock = clock
        stage_logical_slots = logical_slots_for_stage(str(roster.get("stage_code")))
        self.available_slots = _normalize_available_slots(
            stage_logical_slots if available_slots is None else available_slots,
            logical_slots=stage_logical_slots,
        )
        self.roster_digest = roster_hash(roster)
        self.parallel_digest = parallel_plan_hash(parallel_plan)
        self.bound_digest = canonical_sha256(bound_manifest)
        self.scientific_digest = require_sha256(
            scientific_plan.get("randomization_plan_hash"),
            "scientific randomization plan hash",
        )
        if registry.roster_hash != self.roster_digest:
            raise LifecycleError("execution registry is bound to a different roster")
        stage_code = str(roster.get("stage_code"))
        self.divergence_loss_cap = multiseed_analysis_loss_cap(
            stage_code, bound_manifest
        )
        self.runtime_authorization_hash = multiseed_runtime_authorization_hash(
            stage_code=stage_code,
            design_contract_hash=(
                roster.get("best_paper_v2_design_contract_hash")
                or roster.get("audit_135m_design_contract_hash")
            ),
            roster_digest=self.roster_digest,
            parallel_digest=self.parallel_digest,
            bound_digest=self.bound_digest,
            scientific_digest=self.scientific_digest,
            authorization=runtime_authorization,
            hard_ceiling_usd=roster.get("hard_ceiling_usd"),
        )
        self.active: dict[str, GenerationIdentity] = {}
        self.providers: dict[tuple[str, int], dict[str, Any]] = {}
        self.partials: dict[tuple[str, int], VmPartialManifestController] = {}
        self.ready_at: dict[tuple[str, int], str] = {}
        self._scientific_by_id = _scientific_cells(scientific_plan)

    def _common_bindings(self) -> dict[str, Any]:
        return {
            "roster_hash": self.roster_digest,
            "parallel_plan_hash": self.parallel_digest,
            "bound_manifest_canonical_sha256": self.bound_digest,
            "scientific_randomization_plan_hash": self.scientific_digest,
            "amendment_raw_sha256": AMENDMENT_RAW_SHA256,
            **(
                {"multiseed_runtime_authorization_hash": self.runtime_authorization_hash}
                if self.runtime_authorization_hash is not None
                else {}
            ),
        }

    def _assert_capacity_census(self) -> Mapping[str, Any]:
        census = _mapping(self.backend.census(self.roster_digest[:16]), "provider census")
        vm_count = require_nonnegative_int(
            census.get("campaign_owned_vm_count"), "campaign-owned VM count"
        )
        a100_count = require_nonnegative_int(
            census.get("campaign_owned_attached_a100s"),
            "campaign-owned attached A100 count",
        )
        maximum_vms = maximum_scientific_vms_for_stage(
            str(self.roster.get("stage_code"))
        )
        if vm_count > maximum_vms or a100_count > MAX_CAMPAIGN_A100S:
            raise LifecycleError(
                "provider census exceeds the stage VM limit or 16 A100s"
            )
        return census

    def _local_vm_root(self, identity: GenerationIdentity) -> Path:
        return _generation_local_root(
            self.campaign_root, identity.slot, identity.generation
        )

    def _provision_slot(self, slot: str) -> GenerationIdentity:
        self._assert_capacity_census()
        identity = self.registry.reserve(slot)
        provider = dict(self.backend.provision(identity))
        validate_provider_record(provider, identity.registry_row())
        local_root = self._local_vm_root(identity)
        provider_path = local_root / "provider" / "provider-evidence.json"
        write_json_create_only(provider_path, provider)
        provider_hash = sha256_file(provider_path)
        partial = VmPartialManifestController(
            identity=identity,
            local_vm_root=local_root,
            common_bindings=self._common_bindings(),
            provider_record_sha256=provider_hash,
        )
        ready_at = self.backend.ready(identity, provider)
        parse_time(ready_at, "VM READY time")
        key = (slot, identity.generation)
        self.active[slot] = identity
        self.providers[key] = provider
        self.partials[key] = partial
        self.ready_at[key] = ready_at
        self.registry.update_state(
            identity,
            status="ready",
            provider_record_sha256=provider_hash,
            ready_at_utc=ready_at,
            region=provider["region"],
            zone=provider["zone"],
            machine_type=provider["machine_type"],
        )
        self._assert_capacity_census()
        return identity

    def provision_initial_generations(self) -> None:
        if self.active:
            raise LifecycleError("initial generations were already provisioned")
        for slot in self.available_slots:
            self._provision_slot(slot)

    def _finalize_identity(
        self, identity: GenerationIdentity, *, preempted: bool
    ) -> None:
        key = (identity.slot, identity.generation)
        partial = self.partials[key]
        provider = self.providers[key]
        digest = partial.hash_lock(hash_locked_at_utc=self.clock())
        lifecycle = dict(
            self.backend.finalize_generation(
                identity,
                provider,
                digest,
                preempted=preempted,
            )
        )
        validate_lifecycle_record(lifecycle, identity.registry_row(), provider, digest)
        lifecycle_path = (
            self._local_vm_root(identity)
            / "manifests"
            / "vm-lifecycle-final.json"
        )
        write_json_create_only(lifecycle_path, lifecycle)
        self.registry.update_state(
            identity,
            status="vm_lifecycle_final",
            partial_manifest_sha256=digest,
            lifecycle_record_sha256=sha256_file(lifecycle_path),
        )
        if self.active.get(identity.slot) == identity:
            del self.active[identity.slot]

    def _replace_preempted_slots(self, slots: Iterable[str]) -> None:
        for slot in sorted(set(slots)):
            identity = self.active.get(slot)
            if identity is None:
                raise LifecycleError("preempted slot has no active physical generation")
            self._finalize_identity(identity, preempted=True)
            self._assert_capacity_census()
            self._provision_slot(slot)

    @staticmethod
    def _fresh_start() -> dict[str, Any]:
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

    def _paired_available_slots(
        self,
        *,
        group_id: str,
        retry_round: int,
        candidate_slots: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        """Choose one deterministic shape-homogeneous slot set for a seed block."""

        stage_logical_slots = logical_slots_for_stage(
            str(self.roster.get("stage_code"))
        )
        slots = _normalize_available_slots(
            self.active if candidate_slots is None else candidate_slots,
            logical_slots=stage_logical_slots,
        )
        if any(slot not in self.active for slot in slots):
            raise LifecycleError("candidate block lane contains a non-READY slot")
        if str(self.roster.get("stage_code")) not in CONTROLLED_STAGE_CODES:
            return slots
        by_shape: dict[str, list[str]] = defaultdict(list)
        for slot in slots:
            identity = self.active[slot]
            provider = self.providers[(slot, identity.generation)]
            by_shape[str(provider["machine_type"])].append(slot)
        maximum_width = max(len(shape_slots) for shape_slots in by_shape.values())
        candidates = [
            f"{shape}|{','.join(sorted(shape_slots))}"
            for shape, shape_slots in by_shape.items()
            if len(shape_slots) == maximum_width
        ]
        domain = f"paired-shape|{group_id}|{retry_round}"
        selected = rank_order(candidates, domain, str(self.roster["study_id"]))[0]
        _shape, _separator, slot_text = selected.partition("|")
        selected_slots = tuple(slot_text.split(","))
        return _normalize_available_slots(
            selected_slots, logical_slots=stage_logical_slots
        )

    def _execute_wave(
        self,
        *,
        planned_wave: Mapping[str, Any],
        retry_round: int,
        actual_wave_index: int,
        concurrent_batch_index: int = 0,
        available_slots: Iterable[str] | None = None,
        concurrent_batch_slot_set: Iterable[str] | None = None,
        prior_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        self._assert_capacity_census()
        paired_slots = self._paired_available_slots(
            group_id=str(planned_wave["group_id"]),
            retry_round=retry_round,
            candidate_slots=available_slots,
        )
        batch_slots = _normalize_available_slots(
            paired_slots
            if concurrent_batch_slot_set is None
            else concurrent_batch_slot_set,
            logical_slots=logical_slots_for_stage(
                str(self.roster.get("stage_code"))
            ),
        )
        if any(slot not in batch_slots for slot in paired_slots):
            raise ScheduleError("atomic block lane is outside its concurrent batch slot set")
        wave = wave_for_retry(
            self.plan,
            self.roster,
            str(planned_wave["group_id"]),
            retry_round,
            available_slots=paired_slots,
        )
        requests: list[tuple[GenerationIdentity, DispatchRequest]] = []
        prior_by_cell = (
            {} if prior_rows is None else {str(row["cell_id"]): row for row in prior_rows}
        )
        prior_wave_hash = None if prior_rows is None else _wave_manifest_hash(prior_rows)
        authorization_time = None if prior_rows is None else self.clock()
        for assignment in wave["assigned_cells_in_dispatch_order"]:
            slot = str(assignment["logical_slot"])
            identity = self.active.get(slot)
            if identity is None:
                raise LifecycleError(f"wave assignment {slot} has no READY generation")
            cell_id = str(assignment["cell_id"])
            scientific = self._scientific_by_id[cell_id]
            provider = self.providers[(slot, identity.generation)]
            shape = machine_shape_contract(provider["machine_type"])
            learner_count = _cell_learner_count(scientific)
            prior = prior_by_cell.get(cell_id)
            if prior is None:
                retry_of = retry_reason = None
                retry_authorization = None
            else:
                retry_of = str(prior["attempt_id"])
                retry_reason = (
                    PEER_RETRY_REASON
                    if prior["status"] in ("COMPLETED", "DIVERGED")
                    else str(prior["failure_reason"])
                )
                retry_authorization = {
                    "loss_blind": True,
                    "parallel_plan_hash": self.parallel_digest,
                    "group_id": planned_wave["group_id"],
                    "retry_round": retry_round,
                    "prior_wave_manifest_canonical_sha256": prior_wave_hash,
                    "authorized_at_utc": authorization_time,
                }
            request = DispatchRequest(
                group_id=str(planned_wave["group_id"]),
                cell_id=cell_id,
                retry_round=retry_round,
                actual_wave_index=actual_wave_index,
                concurrent_batch_index=concurrent_batch_index,
                concurrent_batch_slot_set=batch_slots,
                time_block_index=int(planned_wave["time_block_index"]),
                retry_time_block_index=(
                    None if retry_round == 1 else actual_wave_index
                ),
                available_slot_set=tuple(assignment["available_slot_set"]),
                dispatch_batch_index=int(assignment["dispatch_batch_index"]),
                batch_launch_order_index=int(
                    assignment["batch_launch_order_index"]
                ),
                launch_order_index=int(assignment["launch_order_index"]),
                attempt_prefix=identity.attempt_prefix(cell_id, retry_round),
                learner_count=learner_count,
                quorum=int(
                    _mapping(
                        scientific.get("target_work"),
                        f"scientific cell {cell_id} target work",
                    ).get("quorum", _cell_learner_count(scientific))
                ),
                gpu_slots=shape["gpu_slots"],
                learner_gpu_slot_map=learner_gpu_slot_map(
                    learner_count, shape["gpu_slots"]
                ),
                pairing_identity_hash=scientific.get("pairing_identity_hash"),
                command=tuple(scientific["command"]),
                command_hash=str(scientific["command_hash"]),
                fresh_start=self._fresh_start(),
                retry_of=retry_of,
                retry_reason=retry_reason,
                retry_authorization=retry_authorization,
            )
            requests.append((identity, request))

        dispatch_times: dict[str, str] = {}
        outcomes_by_cell: dict[str, dict[str, Any]] = {}
        batch_indices = sorted({request.dispatch_batch_index for _identity, request in requests})
        if batch_indices != list(range(len(batch_indices))):
            raise ScheduleError("dispatch batch indices are not contiguous from zero")
        for batch_index in batch_indices:
            batch = [
                (identity, request)
                for identity, request in requests
                if request.dispatch_batch_index == batch_index
            ]
            batch.sort(key=lambda item: item[1].batch_launch_order_index)
            if len({identity.slot for identity, _request in batch}) != len(batch):
                raise ScheduleError("a reduced-width dispatch batch repeats one logical slot")
            for identity, request in batch:
                dispatched_at = self.backend.dispatch(identity, request)
                parse_time(dispatched_at, "backend dispatch time")
                dispatch_times[request.cell_id] = dispatched_at
            with ThreadPoolExecutor(max_workers=min(4, len(batch))) as pool:
                futures = [
                    pool.submit(self.backend.collect, identity, request)
                    for identity, request in batch
                ]
                batch_outcomes = [dict(future.result()) for future in futures]
            for (_identity, request), outcome in zip(batch, batch_outcomes):
                outcomes_by_cell[request.cell_id] = outcome
        terminal_prefix_sealed_at = self.clock()
        rows: list[dict[str, Any]] = []
        for identity, request in requests:
            outcome = outcomes_by_cell[request.cell_id]
            if outcome.get("resumed") is True or outcome.get("resume_source") not in (None, ""):
                raise ScheduleError("backend attempted to resume a prior scientific attempt")
            status = outcome.get("status")
            if status not in ATTEMPT_STATUSES:
                raise ScheduleError("backend returned a status outside the frozen vocabulary")
            if status == "INFRA_FAILURE" and outcome.get(
                "failure_reason"
            ) not in DIRECT_INFRASTRUCTURE_FAILURE_REASONS:
                raise ScheduleError("backend returned a non-direct infrastructure reason")
            if self.divergence_loss_cap is not None and status in (
                "COMPLETED",
                "DIVERGED",
            ):
                expected_analysis = multiseed_analysis_outcome_fields(
                    status=str(status),
                    loss=outcome.get("loss"),
                    divergence_loss_cap=self.divergence_loss_cap,
                    checkpoint_only=audit_checkpoint_only(scientific),
                )
                if outcome.get("analysis_loss") not in (
                    None,
                    expected_analysis["analysis_loss"],
                ) or outcome.get("analysis_loss_kind") not in (
                    None,
                    expected_analysis["analysis_loss_kind"],
                ):
                    raise ScheduleError(
                        "backend returned an analysis value outside the frozen multiseed policy"
                    )
                if outcome.get("divergence_retained") not in (
                    None,
                    expected_analysis["divergence_retained"],
                ):
                    raise ScheduleError(
                        "backend attempted to change multiseed divergence retention"
                    )
                outcome.update(expected_analysis)
            provider = self.providers[(identity.slot, identity.generation)]
            provider_hash = sha256_file(
                self._local_vm_root(identity)
                / "provider"
                / "provider-evidence.json"
            )
            machine_type = provider["machine_type"]
            shape = machine_shape_contract(machine_type)
            expected_gpu_map = learner_gpu_slot_map(
                request.learner_count, shape["gpu_slots"]
            )
            if (
                request.gpu_slots != shape["gpu_slots"]
                or dict(request.learner_gpu_slot_map) != expected_gpu_map
            ):
                raise ScheduleError(
                    "dispatch request learner/GPU packing differs from the landed shape"
                )
            projected_command = project_scientific_command_for_machine_type(
                request.command, machine_type
            )
            row = {
                **outcome,
                "attempt_id": f"{request.cell_id}-attempt-{request.retry_round}",
                "cell_id": request.cell_id,
                "attempt": request.retry_round,
                "group_id": request.group_id,
                "retry_round": request.retry_round,
                "actual_wave_index": request.actual_wave_index,
                "concurrent_batch_index": request.concurrent_batch_index,
                "concurrent_batch_slot_set": list(
                    request.concurrent_batch_slot_set
                ),
                "time_block_index": request.time_block_index,
                "retry_time_block_index": request.retry_time_block_index,
                "available_slot_set": list(request.available_slot_set),
                "dispatch_batch_index": request.dispatch_batch_index,
                "batch_launch_order_index": request.batch_launch_order_index,
                "launch_order_index": request.launch_order_index,
                "logical_slot": identity.slot,
                "generation": identity.generation,
                "run_id": identity.run_id,
                "ownership_nonce": identity.ownership_nonce,
                "machine_type": machine_type,
                "gpu_slots": shape["gpu_slots"],
                "instance_numeric_id": provider["instance_numeric_id"],
                "provider_evidence_sha256": provider_hash,
                "attempt_prefix": request.attempt_prefix,
                "m": request.learner_count,
                "quorum": request.quorum,
                "learner_gpu_slot_map": expected_gpu_map,
                "maximum_learners_per_gpu": max(
                    Counter(expected_gpu_map.values()).values()
                ),
                "pairing_identity_hash": request.pairing_identity_hash,
                "frozen_command_hash": request.command_hash,
                "executed_command_hash": canonical_sha256(projected_command),
                "normalized_workload_command_hash": canonical_sha256(
                    normalized_workload_command(projected_command)
                ),
                "fresh_start": dict(request.fresh_start),
                "retry_of": request.retry_of,
                "retry_reason": request.retry_reason,
                "retry_authorization": (
                    None
                    if request.retry_authorization is None
                    else dict(request.retry_authorization)
                ),
                "vm_ready_at": self.ready_at[(identity.slot, identity.generation)],
                "dispatched_at": dispatch_times[request.cell_id],
                "wave_terminal_prefix_sealed_at": terminal_prefix_sealed_at,
            }
            parse_time(row.get("scientific_started_at"), "backend scientific start")
            parse_time(row.get("scientific_ended_at"), "backend scientific end")
            rows.append(row)

        rows.sort(key=lambda row: int(row["launch_order_index"]))
        if str(self.roster.get("stage_code")) in CONTROLLED_STAGE_CODES:
            if len({row["machine_type"] for row in rows}) != 1 or len(
                {row["gpu_slots"] for row in rows}
            ) != 1:
                raise ScheduleError(
                    "paired seed block mixed machine shapes or GPU-slot allocations"
                )
            if len({row.get("pairing_identity_hash") for row in rows}) != 1:
                raise ScheduleError("paired seed block mixed pairing identities")
            if len(
                {canonical_json(row.get("learner_gpu_slot_map")) for row in rows}
            ) != 1:
                raise ScheduleError("paired seed block mixed learner/GPU packing maps")
        terminal_time = parse_time(
            terminal_prefix_sealed_at, "wave terminal-prefix seal time"
        )
        if max(
            parse_time(row.get("scientific_ended_at"), "scientific end")
            for row in rows
        ) > terminal_time:
            raise ScheduleError("backend terminal-prefix seal precedes scientific completion")
        for row in rows:
            key = (str(row["logical_slot"]), int(row["generation"]))
            self.partials[key].append_attempt(row)
        _validate_wave_timing(rows)
        return rows

    def run(self) -> tuple[dict[str, Any], Mapping[str, Any]]:
        if not self.active:
            self.provision_initial_generations()
        actual_wave_index = 0
        for planned_wave in self.plan["waves"]:
            retry_round = 1
            prior_rows: list[dict[str, Any]] | None = None
            while True:
                rows = self._execute_wave(
                    planned_wave=planned_wave,
                    retry_round=retry_round,
                    actual_wave_index=actual_wave_index,
                    prior_rows=prior_rows,
                )
                actual_wave_index += 1
                statuses = {row["status"] for row in rows}
                if "FAILED" in statuses or (
                    "DIVERGED" in statuses and "INFRA_FAILURE" in statuses
                ):
                    raise ScheduleError("mixed/nonretryable terminal wave blocks the campaign")
                preempted_slots = [
                    str(row["logical_slot"])
                    for row in rows
                    if row["status"] == "INFRA_FAILURE"
                    and row.get("failure_reason") == "provider_spot_preemption"
                ]
                if preempted_slots:
                    self._replace_preempted_slots(preempted_slots)
                if "INFRA_FAILURE" in statuses:
                    prior_rows = rows
                    retry_round += 1
                    continue
                break
        for identity in list(self.active.values()):
            self._finalize_identity(identity, preempted=False)
        census = self._assert_capacity_census()
        if (
            census.get("campaign_owned_vm_count") != 0
            or census.get("campaign_owned_attached_a100s") != 0
        ):
            raise LifecycleError("campaign teardown completed without a zero provider census")
        return self.registry.snapshot(), census


def optimizer_harness_launch_argv(
    *,
    python_executable: str,
    state_dir: Path,
    spec_path: Path,
    ownership_nonce: str,
) -> list[str]:
    if NONCE_RE.fullmatch(ownership_nonce) is None:
        raise LifecycleError("parallel optimizer-harness launch requires a 128-bit nonce")
    return [
        python_executable,
        "-m",
        "yeto.optimizer_harness",
        "--state-dir",
        str(state_dir),
        "launch",
        str(spec_path),
        "--ownership-nonce",
        ownership_nonce,
        "--yes",
    ]


def optimizer_harness_delete_argv(
    *,
    python_executable: str,
    state_dir: Path,
    spec_path: Path,
    exact_instance_id: str,
) -> list[str]:
    instance_id = require_numeric_id(exact_instance_id, "exact teardown instance ID")
    return [
        python_executable,
        "-m",
        "yeto.optimizer_harness",
        "--state-dir",
        str(state_dir),
        "delete",
        str(spec_path),
        "--instance-id",
        instance_id,
        "--yes",
    ]


def bind_campaign_inputs(
    *,
    stage_code: str,
    parent_manifest_path: Path,
    bound_manifest_path: Path,
    scientific_plan_path: Path,
    output_dir: Path,
    prebound_schedule_path: Path | None = None,
) -> dict[str, Any]:
    parent = _mapping(load_json(parent_manifest_path, "parent manifest"), "parent manifest")
    bound = _mapping(load_json(bound_manifest_path, "bound manifest"), "bound manifest")
    scientific = _mapping(
        load_json(scientific_plan_path, "scientific randomization plan"),
        "scientific randomization plan",
    )
    roster = build_parallel_roster(
        stage_code=stage_code,
        bound_manifest=bound,
        parent_manifest=parent,
        scientific_plan=scientific,
    )
    digest = roster_hash(roster)
    plan = build_parallel_plan(roster, expected_roster_hash=digest)
    plan_digest = parallel_plan_hash(plan)
    runtime_logical_slots = (
        tuple(
            slot
            for slot in AUDIT_135M_LOGICAL_SLOTS
            if any(
                slot in wave.get("audit_registered_batch_slot_set", [])
                for wave in plan["waves"]
            )
        )
        if stage_code in AUDIT_135M_STAGE_CODES
        else LOGICAL_SLOTS
    )
    if prebound_schedule_path is not None:
        prebound = _mapping(
            load_json(prebound_schedule_path, "prebound schedule"),
            "prebound schedule",
        )
        validate_prebound_p1r0_schedule(prebound, plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_create_only(output_dir / "parallel-roster.json", roster)
    write_json_create_only(output_dir / "parallel-plan.json", plan)
    materialization = {
        "schema": "yeto_parallel_binding_v2",
        "stage_code": stage_code,
        "parent_manifest_canonical_sha256": canonical_sha256(parent),
        "bound_manifest_canonical_sha256": canonical_sha256(bound),
        "scientific_randomization_plan_hash": scientific["randomization_plan_hash"],
        "roster_hash": digest,
        "roster_tag": digest[:16],
        "supersedes_parallel_plan_hash": plan[
            "supersedes_parallel_plan_hash"
        ],
        "parallel_plan_hash": plan_digest,
        "science_root": f"/opt/yeto-science/{stage_code}/{digest[:16]}",
        "physical_generation_run_ids": {
            slot: physical_run_id(stage_code, digest, 1, slot, 1)
            for slot in runtime_logical_slots
        },
    }
    if stage_code in MULTISEED_STAGE_CODES:
        materialization.update(
            {
                "launch_authorized": False,
                "best_paper_v2_design_contract_hash": roster[
                    "best_paper_v2_design_contract_hash"
                ],
                "runtime_authorization_required_fields": {
                    "schema": "yeto_multiseed_runtime_authorization_v1",
                    "launch_authorized": True,
                    "stage_code": stage_code,
                    "best_paper_v2_design_contract_hash": roster[
                        "best_paper_v2_design_contract_hash"
                    ],
                    "roster_hash": digest,
                    "parallel_plan_hash": plan_digest,
                    "bound_manifest_canonical_sha256": canonical_sha256(bound),
                    "scientific_randomization_plan_hash": scientific[
                        "randomization_plan_hash"
                    ],
                },
            }
        )
    elif stage_code in AUDIT_135M_STAGE_CODES:
        materialization.update(
            {
                "launch_authorized": False,
                "logical_slots": list(runtime_logical_slots),
                "audit_135m_design_contract_hash": roster[
                    "audit_135m_design_contract_hash"
                ],
                "hard_ceiling_usd": roster["hard_ceiling_usd"],
                "runtime_authorization_required_fields": {
                    "schema": "yeto_audit_135m_runtime_authorization_v1",
                    "launch_authorized": True,
                    "stage_code": stage_code,
                    "audit_135m_design_contract_hash": roster[
                        "audit_135m_design_contract_hash"
                    ],
                    "roster_hash": digest,
                    "parallel_plan_hash": plan_digest,
                    "bound_manifest_canonical_sha256": canonical_sha256(bound),
                    "scientific_randomization_plan_hash": scientific[
                        "randomization_plan_hash"
                    ],
                    "hard_ceiling_usd": roster["hard_ceiling_usd"],
                    "spot_only": True,
                    "maximum_attached_a100_equivalent": 16,
                    "max_idle_before_science_seconds": 600,
                },
            }
        )
    write_json_create_only(output_dir / "parallel-binding.json", materialization)
    return materialization


def _resolve_descriptor_path(descriptor_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ParallelPhaseMapError(f"descriptor {label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = descriptor_path.parent / path
    return path.resolve()


def aggregate_from_descriptor(
    descriptor_path: Path, *, write_seal: bool, sealed_at_utc: str | None = None
) -> dict[str, Any]:
    descriptor = _mapping(
        load_json(descriptor_path, "campaign aggregation descriptor"),
        "campaign aggregation descriptor",
    )

    def object_from(field: str) -> dict[str, Any]:
        return _mapping(
            load_json(
                _resolve_descriptor_path(descriptor_path, descriptor.get(field), field),
                field,
            ),
            field,
        )

    bundle = CampaignBundle(
        stage_code=str(descriptor.get("stage_code")),
        parent_manifest=object_from("parent_manifest"),
        bound_manifest=object_from("bound_manifest"),
        scientific_plan=object_from("scientific_plan"),
        roster=object_from("parallel_roster"),
        parallel_plan=object_from("parallel_plan"),
        vm_registry=object_from("vm_registry"),
        evaluation_registry=object_from("evaluation_registry"),
        final_provider_census=object_from("final_provider_census"),
        campaign_attempt=require_positive_int(
            descriptor.get("campaign_attempt"), "descriptor campaign attempt"
        ),
        campaign_root=_resolve_descriptor_path(
            descriptor_path, descriptor.get("campaign_root"), "campaign_root"
        ),
        runtime_authorization=(
            object_from("runtime_authorization")
            if descriptor.get("runtime_authorization") is not None
            else None
        ),
    )
    aggregator = CampaignAggregator(bundle)
    if write_seal:
        return aggregator.seal(sealed_at_utc=sealed_at_utc)
    _manifest, seal = aggregator.build_manifest_and_seal(
        sealed_at_utc=sealed_at_utc
    )
    return seal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    bind = subparsers.add_parser("bind", help="bind a final parent into roster/plan bytes")
    bind.add_argument("--stage-code", choices=sorted(ALLOWED_STAGE_CODES), required=True)
    bind.add_argument("--parent-manifest", type=Path, required=True)
    bind.add_argument("--bound-manifest", type=Path, required=True)
    bind.add_argument("--scientific-plan", type=Path, required=True)
    bind.add_argument("--prebound-schedule", type=Path)
    bind.add_argument("--output-dir", type=Path, required=True)
    aggregate = subparsers.add_parser(
        "aggregate", help="validate all VM partials/lifecycles and optionally seal"
    )
    aggregate.add_argument("--descriptor", type=Path, required=True)
    aggregate.add_argument("--write-seal", action="store_true")
    aggregate.add_argument("--sealed-at-utc")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "bind":
            result = bind_campaign_inputs(
                stage_code=args.stage_code,
                parent_manifest_path=args.parent_manifest,
                bound_manifest_path=args.bound_manifest,
                scientific_plan_path=args.scientific_plan,
                prebound_schedule_path=args.prebound_schedule,
                output_dir=args.output_dir,
            )
        else:
            result = aggregate_from_descriptor(
                args.descriptor,
                write_seal=args.write_seal,
                sealed_at_utc=args.sealed_at_utc,
            )
    except (OSError, ValueError, ParallelPhaseMapError) as exc:
        print(f"parallel phase-map error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
