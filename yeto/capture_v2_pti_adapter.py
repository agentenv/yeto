"""Authority-preserving PTI-SGD adapter for capture-v2 syncer boundaries.

Only a verified boundary may create a PTI event.  The public API accepts no
event sequence, fragment, version, shape, or stock bytes, so callers cannot
relabel causal evidence.  Evidence that cannot identify the preregistered
production path yields ``UNIDENTIFIABLE`` and never emits a sealed action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import re
import struct
from typing import Any

from .capture_v2_policy import (
    LoadedSealedOuterAction,
    PolicyDefinitionRef,
    SealedOuterActionRef,
    load_sealed_outer_action,
    load_policy_definition,
    publish_sealed_outer_action,
)
from .capture_v2_store import CaptureObjectStore, CaptureStoreError, ObjectRef
from .capture_v2_syncer import (
    BoundaryConfig,
    LoadedSyncerBoundary,
    SyncerBoundaryRef,
    load_syncer_boundary,
    memoryless_outer_update_f32le,
)
from .pti_sgd import PTIEvent, PTILedgerEntry, PTIResult, PTISGD


SCHEMA = "yeto.capture-v2-pti-decision-proof"
SCHEMA_VERSION = 1
IDENTIFIED = "IDENTIFIED"
UNIDENTIFIABLE = "UNIDENTIFIABLE"

PRODUCTION_MERGE_CONFIG = BoundaryConfig("rda", {"weighted": True})
PRODUCTION_OUTER_LR_F64_BITS = struct.pack(">d", 0.28).hex()
PRODUCTION_OUTER_MOMENTUM_F64_BITS = "0000000000000000"
REQUIRED_POLICY_CAPABILITIES = ("same_fragment_history", "global_boundary_state")

_AMENDMENT_1_DESCRIPTOR = {
    "algorithm": "pti-sgd",
    "candidate_arithmetic": "coordinate-order-ieee754-f32",
    "coefficient": {"denominator": 4, "numerator": -1},
    "interlock_length": 3,
    "ledger": "hash-chained-causal-prequential",
    "tail_close": "unresolved-shadow-non-action",
}
PTI_AMENDMENT_1_SHA256 = hashlib.sha256(
    (
        json.dumps(
            _AMENDMENT_1_DESCRIPTOR,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
).hexdigest()


class PTIAdapterError(CaptureStoreError):
    """An identified adapter artifact is internally inconsistent."""


@dataclass(frozen=True)
class PTIAdapterResult:
    """Identified sealed action, or a no-action evidence abstention."""

    status: str
    reason: str
    action: SealedOuterActionRef | None = None
    decision: ObjectRef | None = None
    selected_pseudo_gradient: ObjectRef | None = None
    resulting_fragment: ObjectRef | None = None
    pti_result: PTIResult | None = None


@dataclass(frozen=True)
class LoadedPTIAdapterAction:
    """A generic sealed action whose PTI decision proof is cross-verified."""

    action: LoadedSealedOuterAction
    proof: dict[str, Any]


class CaptureV2PTIPolicyState:
    """Causal PTI history bound to one authoritative capture session."""

    def __init__(self) -> None:
        self._pti = PTISGD()
        self._capture_session_uuid: str | None = None
        self._last_commit_seq: int | None = None
        self._last_boundary_sha256: str | None = None
        self._fragment_layouts: dict[int, tuple[int, str]] = {}
        self._fragment_versions: dict[int, int] = {}

    @property
    def capture_session_uuid(self) -> str | None:
        return self._capture_session_uuid

    @property
    def last_commit_seq(self) -> int | None:
        return self._last_commit_seq

    @property
    def last_boundary_sha256(self) -> str | None:
        return self._last_boundary_sha256

    @property
    def ledger_head(self) -> str:
        return self._pti.ledger_head

    @property
    def ledger(self) -> tuple[PTILedgerEntry, ...]:
        return self._pti.ledger


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PTIAdapterError(f"decision proof is not canonical JSON: {exc}") from exc


def _read_exact(store: CaptureObjectStore, ref: ObjectRef, context: str) -> bytes:
    path = store.verify_object(ref)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PTIAdapterError(f"cannot read {context} CAS object: {exc}") from exc
    if len(raw) != ref.bytes or hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise PTIAdapterError(f"{context} CAS object changed after verification")
    return raw


def _unidentifiable(reason: str) -> PTIAdapterResult:
    return PTIAdapterResult(status=UNIDENTIFIABLE, reason=reason)


def _production_config_reason(boundary: LoadedSyncerBoundary) -> str | None:
    merge_parameters = boundary.merge_config.parameters
    if (
        boundary.merge_config.name != "rda"
        or not isinstance(merge_parameters, dict)
        or set(merge_parameters) != {"weighted"}
        or merge_parameters["weighted"] is not True
    ):
        return "unsupported_merge_config"
    if boundary.outer_config.name != "nesterov":
        return "unsupported_outer_optimizer"
    parameters = boundary.outer_config.parameters
    if not isinstance(parameters, dict) or set(parameters) != {
        "lr_f64_bits",
        "momentum_f64_bits",
    }:
        return "unsupported_outer_config_fields"
    if parameters["lr_f64_bits"] != PRODUCTION_OUTER_LR_F64_BITS:
        return "unsupported_outer_lr"
    if parameters["momentum_f64_bits"] != PRODUCTION_OUTER_MOMENTUM_F64_BITS:
        return "unsupported_outer_momentum"
    return None


def _continuity_reason(
    state: CaptureV2PTIPolicyState, boundary: LoadedSyncerBoundary
) -> str | None:
    identity = boundary.identity
    if (
        state._capture_session_uuid is not None
        and identity.capture_session_uuid != state._capture_session_uuid
    ):
        return "capture_session_discontinuity"
    if (
        state._last_commit_seq is not None
        and identity.commit_seq != state._last_commit_seq + 1
    ):
        return "authoritative_commit_sequence_discontinuity"
    expected_layout = state._fragment_layouts.get(identity.fragment_id)
    observed_layout = (
        boundary.fragment_format.fragment_numel,
        boundary.fragment_format.tensor_layout_sha256,
    )
    if expected_layout is not None and observed_layout != expected_layout:
        return "fragment_layout_discontinuity"
    previous_version = state._fragment_versions.get(identity.fragment_id)
    if previous_version is not None and identity.pre_fragment_version != previous_version:
        return "fragment_version_discontinuity"
    return None


def _factual_stock_reason(
    store: CaptureObjectStore, boundary: LoadedSyncerBoundary
) -> tuple[str | None, bytes | None, bytes | None]:
    pre = _read_exact(store, boundary.pre_fragment, "pre-fragment")
    stock = _read_exact(store, boundary.stock_pseudo_gradient, "stock-gradient")
    post = _read_exact(store, boundary.post_fragment, "post-fragment")
    broadcast = _read_exact(store, boundary.broadcast, "broadcast")
    try:
        factual = memoryless_outer_update_f32le(
            pre, stock, PRODUCTION_OUTER_LR_F64_BITS
        )
    except (CaptureStoreError, TypeError):
        return "invalid_flat_f32_outer_update", None, None
    if factual != post:
        return "stock_path_post_mismatch", None, None
    if broadcast != post:
        return "stock_path_broadcast_mismatch", None, None
    values = struct.unpack(f"<{boundary.fragment_format.fragment_numel}f", stock)
    if not all(math.isfinite(value) for value in values):
        return "nonfinite_stock_pseudo_gradient", None, None
    return None, pre, stock


def _object_value(role: str, ref: ObjectRef) -> dict[str, Any]:
    return {"role": role, "sha256": ref.sha256, "bytes": ref.bytes}


def process_authoritative_boundary(
    store: CaptureObjectStore,
    *,
    state: CaptureV2PTIPolicyState,
    policy: PolicyDefinitionRef,
    boundary: SyncerBoundaryRef,
    action_manifest_id: str,
) -> PTIAdapterResult:
    """Derive, process, prove, and seal one action from boundary authority only."""

    if not isinstance(state, CaptureV2PTIPolicyState):
        raise TypeError("state must be CaptureV2PTIPolicyState")
    if not isinstance(policy, PolicyDefinitionRef):
        raise TypeError("policy must be PolicyDefinitionRef")
    if not isinstance(boundary, SyncerBoundaryRef):
        raise TypeError("boundary must be SyncerBoundaryRef")

    try:
        loaded_boundary = load_syncer_boundary(store, boundary)
        loaded_policy = load_policy_definition(store, policy)
    except CaptureStoreError as exc:
        return _unidentifiable(f"invalid_authoritative_evidence:{exc}")

    config_reason = _production_config_reason(loaded_boundary)
    if config_reason is not None:
        return _unidentifiable(config_reason)
    missing = sorted(
        set(REQUIRED_POLICY_CAPABILITIES) - set(loaded_policy.capabilities)
    )
    if missing:
        return _unidentifiable(f"missing_policy_capabilities:{','.join(missing)}")
    continuity_reason = _continuity_reason(state, loaded_boundary)
    if continuity_reason is not None:
        return _unidentifiable(continuity_reason)
    try:
        factual_reason, pre_raw, stock_raw = _factual_stock_reason(
            store, loaded_boundary
        )
    except CaptureStoreError as exc:
        return _unidentifiable(f"invalid_fragment_evidence:{exc}")
    if factual_reason is not None or pre_raw is None or stock_raw is None:
        return _unidentifiable(factual_reason or "invalid_fragment_evidence")

    identity = loaded_boundary.identity
    candidate_policy = copy.deepcopy(state._pti)
    event = PTIEvent(
        sequence=identity.commit_seq,
        fragment=identity.fragment_id,
        version=identity.post_fragment_version,
        shape=(loaded_boundary.fragment_format.fragment_numel,),
        stock_sha256=loaded_boundary.stock_pseudo_gradient.sha256,
        stock_raw=stock_raw,
    )
    pti_result = candidate_policy.process(event)
    selected = store.put_bytes(pti_result.action.raw).ref
    if selected.sha256 != pti_result.action.action_sha256:
        raise PTIAdapterError("PTI action hash differs from selected CAS object")
    resulting_raw = memoryless_outer_update_f32le(
        pre_raw, pti_result.action.raw, PRODUCTION_OUTER_LR_F64_BITS
    )
    resulting = store.put_bytes(resulting_raw).ref
    if not pti_result.action.used_nonstock:
        if selected != loaded_boundary.stock_pseudo_gradient:
            raise PTIAdapterError("PTI stock fallback lost exact stock object identity")
        if resulting != loaded_boundary.post_fragment:
            raise PTIAdapterError("PTI stock fallback lost exact post-fragment identity")

    proof = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "amendment_1_sha256": PTI_AMENDMENT_1_SHA256,
        "boundary": {
            "manifest_id": boundary.manifest.manifest_id,
            "sha256": boundary.manifest.sha256,
            "bytes": boundary.manifest.bytes,
        },
        "policy": {
            "manifest_id": policy.manifest.manifest_id,
            "sha256": policy.manifest.sha256,
            "bytes": policy.manifest.bytes,
            "config_sha256": loaded_policy.config.sha256,
        },
        "causal_continuity": {
            "capture_session_uuid": identity.capture_session_uuid,
            "previous_boundary_sha256": state._last_boundary_sha256,
            "previous_commit_seq": state._last_commit_seq,
        },
        "event": {
            "sequence": identity.commit_seq,
            "fragment": identity.fragment_id,
            "version": identity.post_fragment_version,
        },
        "fragment_format": {
            "encoding": "ieee754-f32le-flat",
            "fragment_numel": loaded_boundary.fragment_format.fragment_numel,
            "tensor_layout_sha256": (
                loaded_boundary.fragment_format.tensor_layout_sha256
            ),
        },
        "stock_pseudo_gradient": _object_value(
            "syncer/stock-pseudo-gradient",
            loaded_boundary.stock_pseudo_gradient,
        ),
        "selected_pseudo_gradient": _object_value(
            "action/selected-pseudo-gradient", selected
        ),
        "resulting_fragment": _object_value("action/resulting-fragment", resulting),
        "outer_update": {
            "algorithm": "coordinate-order-f32-memoryless-sgd",
            "lr_f64_bits": PRODUCTION_OUTER_LR_F64_BITS,
            "momentum_f64_bits": PRODUCTION_OUTER_MOMENTUM_F64_BITS,
        },
        "pti": {
            "action_sha256": pti_result.action.action_sha256,
            "decision_sha256": pti_result.action.decision_sha256,
            "previous_ledger_sha256": pti_result.ledger.previous_ledger_sha256,
            "ledger_sha256": pti_result.ledger.ledger_sha256,
            "used_nonstock": pti_result.action.used_nonstock,
            "reason": pti_result.action.reason,
        },
    }
    decision = store.put_bytes(_canonical_json_bytes(proof)).ref
    action_kind = "nonstock" if pti_result.action.used_nonstock else "stock_fallback"
    action = publish_sealed_outer_action(
        store,
        action_manifest_id,
        policy=policy,
        boundary=boundary,
        fragment_id=identity.fragment_id,
        required_capabilities=("global_boundary_state",),
        stock_pseudo_gradient=loaded_boundary.stock_pseudo_gradient,
        selected_pseudo_gradient=selected,
        outer_lr_f64_bits=PRODUCTION_OUTER_LR_F64_BITS,
        resulting_fragment=resulting,
        decision=decision,
        config_sha256=loaded_policy.config.sha256,
        action_kind=action_kind,
        action_reason=f"pti_amendment_1:{pti_result.action.reason}",
        fallback_reason=(
            f"pti_fail_closed:{pti_result.action.reason}"
            if action_kind == "stock_fallback"
            else None
        ),
    )

    state._pti = candidate_policy
    state._capture_session_uuid = identity.capture_session_uuid
    state._last_commit_seq = identity.commit_seq
    state._last_boundary_sha256 = loaded_boundary.manifest_sha256
    state._fragment_layouts[identity.fragment_id] = (
        loaded_boundary.fragment_format.fragment_numel,
        loaded_boundary.fragment_format.tensor_layout_sha256,
    )
    state._fragment_versions[identity.fragment_id] = identity.post_fragment_version
    return PTIAdapterResult(
        status=IDENTIFIED,
        reason=pti_result.action.reason,
        action=action,
        decision=decision,
        selected_pseudo_gradient=selected,
        resulting_fragment=resulting,
        pti_result=pti_result,
    )


def load_decision_proof(store: CaptureObjectStore, ref: ObjectRef) -> dict[str, Any]:
    """Strictly decode one adapter proof and verify its own frozen schema digest."""

    raw = _read_exact(store, ref, "PTI decision")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PTIAdapterError(f"cannot decode PTI decision proof: {exc}") from exc
    expected_keys = {
        "schema",
        "schema_version",
        "amendment_1_sha256",
        "boundary",
        "policy",
        "causal_continuity",
        "event",
        "fragment_format",
        "stock_pseudo_gradient",
        "selected_pseudo_gradient",
        "resulting_fragment",
        "outer_update",
        "pti",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise PTIAdapterError("PTI decision proof fields are malformed")
    if value.get("schema") != SCHEMA:
        raise PTIAdapterError("PTI decision proof schema is malformed")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PTIAdapterError("PTI decision proof version is unsupported")
    if value.get("amendment_1_sha256") != PTI_AMENDMENT_1_SHA256:
        raise PTIAdapterError("PTI decision proof amendment digest differs")
    if _canonical_json_bytes(value) != raw:
        raise PTIAdapterError("PTI decision proof is not canonical JSON")
    continuity = value["causal_continuity"]
    if not isinstance(continuity, dict) or set(continuity) != {
        "capture_session_uuid",
        "previous_boundary_sha256",
        "previous_commit_seq",
    }:
        raise PTIAdapterError("PTI decision proof causal continuity is malformed")
    return value


def load_authoritative_pti_action(
    store: CaptureObjectStore, ref: SealedOuterActionRef
) -> LoadedPTIAdapterAction:
    """Reverify a generic action and every adapter-specific proof cross-reference."""

    action = load_sealed_outer_action(store, ref)
    proof = load_decision_proof(store, action.decision)
    config_reason = _production_config_reason(action.boundary)
    if config_reason is not None:
        raise PTIAdapterError(
            f"PTI sealed action boundary is not production-identifiable: {config_reason}"
        )
    missing = sorted(
        set(REQUIRED_POLICY_CAPABILITIES) - set(action.policy.capabilities)
    )
    if missing:
        raise PTIAdapterError(
            f"PTI sealed action policy is missing capabilities: {missing}"
        )
    if action.stock_pseudo_gradient != action.boundary.stock_pseudo_gradient:
        raise PTIAdapterError("PTI sealed action stock object is cross-wired")
    expected_boundary = {
        "manifest_id": action.boundary_ref.manifest.manifest_id,
        "sha256": action.boundary_ref.manifest.sha256,
        "bytes": action.boundary_ref.manifest.bytes,
    }
    expected_policy = {
        "manifest_id": action.policy_ref.manifest.manifest_id,
        "sha256": action.policy_ref.manifest.sha256,
        "bytes": action.policy_ref.manifest.bytes,
        "config_sha256": action.policy.config.sha256,
    }
    expected_event = {
        "sequence": action.boundary.identity.commit_seq,
        "fragment": action.boundary.identity.fragment_id,
        "version": action.boundary.identity.post_fragment_version,
    }
    expected_format = {
        "encoding": "ieee754-f32le-flat",
        "fragment_numel": action.boundary.fragment_format.fragment_numel,
        "tensor_layout_sha256": (
            action.boundary.fragment_format.tensor_layout_sha256
        ),
    }
    expected_outer = {
        "algorithm": "coordinate-order-f32-memoryless-sgd",
        "lr_f64_bits": PRODUCTION_OUTER_LR_F64_BITS,
        "momentum_f64_bits": PRODUCTION_OUTER_MOMENTUM_F64_BITS,
    }
    exact_bindings = (
        (proof.get("boundary"), expected_boundary, "boundary manifest"),
        (proof.get("policy"), expected_policy, "policy manifest"),
        (proof.get("event"), expected_event, "derived event"),
        (proof.get("fragment_format"), expected_format, "fragment layout"),
        (
            proof.get("stock_pseudo_gradient"),
            _object_value("syncer/stock-pseudo-gradient", action.stock_pseudo_gradient),
            "stock pseudo-gradient",
        ),
        (
            proof.get("selected_pseudo_gradient"),
            _object_value(
                "action/selected-pseudo-gradient", action.selected_pseudo_gradient
            ),
            "selected pseudo-gradient",
        ),
        (
            proof.get("resulting_fragment"),
            _object_value("action/resulting-fragment", action.resulting_fragment),
            "resulting fragment",
        ),
        (proof.get("outer_update"), expected_outer, "outer update"),
    )
    for observed, expected, context in exact_bindings:
        if observed != expected:
            raise PTIAdapterError(f"PTI decision proof {context} is cross-wired")
    pti = proof.get("pti")
    if not isinstance(pti, dict) or set(pti) != {
        "action_sha256",
        "decision_sha256",
        "previous_ledger_sha256",
        "ledger_sha256",
        "used_nonstock",
        "reason",
    }:
        raise PTIAdapterError("PTI decision proof ledger fields are malformed")
    if pti["action_sha256"] != action.selected_pseudo_gradient.sha256:
        raise PTIAdapterError("PTI decision proof action hash is cross-wired")
    if pti["used_nonstock"] != (action.action_kind == "nonstock"):
        raise PTIAdapterError("PTI decision proof action kind is cross-wired")
    if action.action_reason != f"pti_amendment_1:{pti['reason']}":
        raise PTIAdapterError("PTI decision proof action reason is cross-wired")
    for key in ("decision_sha256", "previous_ledger_sha256", "ledger_sha256"):
        value = pti[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise PTIAdapterError(f"PTI decision proof {key} is malformed")

    pre = _read_exact(store, action.boundary.pre_fragment, "pre-fragment")
    selected = _read_exact(
        store, action.selected_pseudo_gradient, "selected pseudo-gradient"
    )
    factual_result = memoryless_outer_update_f32le(
        pre, selected, PRODUCTION_OUTER_LR_F64_BITS
    )
    if hashlib.sha256(factual_result).hexdigest() != action.resulting_fragment.sha256:
        raise PTIAdapterError(
            "PTI selected action does not reproduce the sealed resulting fragment"
        )
    return LoadedPTIAdapterAction(action=action, proof=proof)


__all__ = [
    "CaptureV2PTIPolicyState",
    "IDENTIFIED",
    "LoadedPTIAdapterAction",
    "PRODUCTION_MERGE_CONFIG",
    "PRODUCTION_OUTER_LR_F64_BITS",
    "PRODUCTION_OUTER_MOMENTUM_F64_BITS",
    "PTIAdapterError",
    "PTIAdapterResult",
    "PTI_AMENDMENT_1_SHA256",
    "UNIDENTIFIABLE",
    "load_decision_proof",
    "load_authoritative_pti_action",
    "process_authoritative_boundary",
]
