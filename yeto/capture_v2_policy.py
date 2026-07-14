"""Generic capture-v2 optimizer-policy, sealed-action, and outcome contracts.

The module defines immutable identities and cross-reference checks only.  It
does not implement optimizer formulas, choose an action, write live training
state, or interpret evaluation results.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable

from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ManifestRef,
    ObjectRef,
)
from .capture_v2_syncer import (
    LoadedSyncerBoundary,
    SyncerBoundaryRef,
    load_syncer_boundary,
)


POLICY_SCHEMA = "yeto.capture-v2-optimizer-policy"
ACTION_SCHEMA = "yeto.capture-v2-sealed-outer-action"
OUTCOME_SCHEMA = "yeto.capture-v2-policy-outcome"
SCHEMA_VERSION = 1

CAPABILITIES = (
    "same_fragment_history",
    "midpoint_adam",
    "global_boundary_state",
    "model_autograd",
    "proposal_stream",
    "worker_restore",
    "crn_train_k8",
)
_CAPABILITY_ORDER = {name: index for index, name in enumerate(CAPABILITIES)}

POLICY_SOURCE_ROLE = "policy/source"
POLICY_CONFIG_ROLE = "policy/config"
STOCK_GRADIENT_ROLE = "action/stock-pseudo-gradient"
SELECTED_GRADIENT_ROLE = "action/selected-pseudo-gradient"
RESULTING_FRAGMENT_ROLE = "action/resulting-fragment"
K0_ROLE = "outcome/k0"
K8_ROLE = "outcome/k8"
EVALUATION_ROLE = "outcome/evaluation"

_POLICY_ID_RE = re.compile(r"[a-z][a-z0-9._-]{0,63}\Z")
_SEMVER_RE = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
_SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_F64_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")


class PolicyContractError(CaptureStoreError):
    """A policy, action, or outcome violates the immutable contract."""


@dataclass(frozen=True)
class PolicyDefinitionRef:
    """Content identity of one frozen optimizer-policy definition."""

    manifest: ManifestRef


@dataclass(frozen=True)
class LoadedPolicyDefinition:
    """Verified frozen policy identity and exact source/config objects."""

    manifest_id: str
    manifest_sha256: str
    policy_id: str
    policy_version: str
    source_commit: str
    source: ObjectRef
    config: ObjectRef
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class SealedOuterActionRef:
    """Content identity of a sealed action with no outcome fields."""

    manifest: ManifestRef


@dataclass
class LoadedSealedOuterAction:
    """Verified sealed action, policy, and syncer-boundary binding."""

    manifest_id: str
    manifest_sha256: str
    policy: LoadedPolicyDefinition
    policy_ref: PolicyDefinitionRef
    boundary: LoadedSyncerBoundary
    boundary_ref: SyncerBoundaryRef
    fragment_id: int
    required_capabilities: tuple[str, ...]
    stock_pseudo_gradient: ObjectRef
    selected_pseudo_gradient: ObjectRef
    outer_lr_f64_bits: str
    resulting_fragment: ObjectRef
    decision_sha256: str
    config_sha256: str
    action_kind: str
    action_reason: str
    fallback_reason: str | None


@dataclass(frozen=True)
class PolicyOutcomeRef:
    """Content identity of one immutable outcome appended beside an action."""

    manifest: ManifestRef


@dataclass
class LoadedPolicyOutcome:
    """Verified evaluation objects and finite losses bound to one action."""

    manifest_id: str
    manifest_sha256: str
    action: LoadedSealedOuterAction
    action_ref: SealedOuterActionRef
    k0: ObjectRef
    k0_loss: float
    k8: ObjectRef
    k8_loss: float
    evaluation: ObjectRef
    evaluation_loss: float


def _policy_id(value: Any) -> str:
    if not isinstance(value, str) or _POLICY_ID_RE.fullmatch(value) is None:
        raise PolicyContractError("policy_id must be canonical lowercase identifier")
    return value


def _policy_version(value: Any) -> str:
    if not isinstance(value, str) or _SEMVER_RE.fullmatch(value) is None:
        raise PolicyContractError("policy_version must be canonical MAJOR.MINOR.PATCH")
    return value


def _source_commit(value: Any) -> str:
    if not isinstance(value, str) or _SOURCE_COMMIT_RE.fullmatch(value) is None:
        raise PolicyContractError("source_commit must be a lowercase 40-hex commit")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PolicyContractError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _f64_bits(value: Any, context: str) -> str:
    if not isinstance(value, str) or _F64_BITS_RE.fullmatch(value) is None:
        raise PolicyContractError(f"{context} must be exactly 16 lowercase hex digits")
    return value


def _reason(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise PolicyContractError(f"{context} must be a non-empty bounded string")
    return value


def _exact_ref(value: Any, context: str) -> ObjectRef:
    if not isinstance(value, ObjectRef):
        raise TypeError(f"{context} must be an ObjectRef")
    return value


def _manifest_descriptor(ref: ManifestRef) -> dict[str, Any]:
    if not isinstance(ref, ManifestRef):
        raise TypeError("cross-reference must contain a ManifestRef")
    return {
        "manifest_id": ref.manifest_id,
        "sha256": ref.sha256,
        "bytes": ref.bytes,
    }


def _parse_manifest_descriptor(value: Any, context: str) -> ManifestRef:
    if not isinstance(value, dict) or set(value) != {"manifest_id", "sha256", "bytes"}:
        raise PolicyContractError(f"{context} manifest fields are malformed")
    try:
        return ManifestRef(value["manifest_id"], value["sha256"], value["bytes"], False)
    except CaptureStoreError as exc:
        raise PolicyContractError(
            f"{context} manifest reference is malformed: {exc}"
        ) from exc


def _object_descriptor(role: str, ref: ObjectRef) -> dict[str, Any]:
    return {"role": role, "sha256": ref.sha256, "bytes": ref.bytes}


def _parse_object_descriptor(value: Any, role: str, context: str) -> ObjectRef:
    if not isinstance(value, dict) or set(value) != {"role", "sha256", "bytes"}:
        raise PolicyContractError(f"{context} object fields are malformed")
    if value["role"] != role:
        raise PolicyContractError(f"{context} object role is noncanonical")
    try:
        return ObjectRef(value["sha256"], value["bytes"])
    except CaptureStoreError as exc:
        raise PolicyContractError(
            f"{context} object reference is malformed: {exc}"
        ) from exc


def _capabilities(values: Iterable[str], context: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be an iterable of capability names")
    try:
        items = list(values)
    except TypeError as exc:
        raise TypeError(f"{context} must be an iterable of capability names") from exc
    if any(
        not isinstance(item, str) or item not in _CAPABILITY_ORDER for item in items
    ):
        raise PolicyContractError(f"{context} contains an unknown capability")
    if len(set(items)) != len(items):
        raise PolicyContractError(f"{context} contains duplicate capabilities")
    return tuple(sorted(items, key=_CAPABILITY_ORDER.__getitem__))


def publish_policy_definition(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    policy_id: str,
    policy_version: str,
    source_commit: str,
    source: ObjectRef,
    config: ObjectRef,
    capabilities: Iterable[str],
) -> PolicyDefinitionRef:
    """Publish one frozen policy definition with exact source/config identity."""

    source = _exact_ref(source, "policy source")
    config = _exact_ref(config, "policy config")
    metadata = {
        "schema": POLICY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy_id": _policy_id(policy_id),
        "policy_version": _policy_version(policy_version),
        "source_commit": _source_commit(source_commit),
        "source": _object_descriptor(POLICY_SOURCE_ROLE, source),
        "config": _object_descriptor(POLICY_CONFIG_ROLE, config),
        "capabilities": list(_capabilities(capabilities, "policy capabilities")),
    }
    manifest = store.publish_manifest(
        manifest_id,
        [
            ManifestEntry(POLICY_SOURCE_ROLE, source),
            ManifestEntry(POLICY_CONFIG_ROLE, config),
        ],
        metadata=metadata,
    )
    return PolicyDefinitionRef(manifest)


def load_policy_definition(
    store: CaptureObjectStore, ref: PolicyDefinitionRef | ManifestRef | str
) -> LoadedPolicyDefinition:
    """Strictly verify one frozen policy definition and both exact objects."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, PolicyDefinitionRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    expected_keys = {
        "schema",
        "schema_version",
        "policy_id",
        "policy_version",
        "source_commit",
        "source",
        "config",
        "capabilities",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise PolicyContractError("policy manifest metadata fields are malformed")
    if metadata["schema"] != POLICY_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise PolicyContractError("policy manifest uses an unsupported schema")
    source = _parse_object_descriptor(
        metadata["source"], POLICY_SOURCE_ROLE, "policy source"
    )
    config = _parse_object_descriptor(
        metadata["config"], POLICY_CONFIG_ROLE, "policy config"
    )
    if not isinstance(metadata["capabilities"], list):
        raise PolicyContractError("policy capabilities must be an array")
    capabilities = _capabilities(metadata["capabilities"], "policy capabilities")
    if list(metadata["capabilities"]) != list(capabilities):
        raise PolicyContractError("policy capabilities are not in canonical order")
    rows = manifest["objects"]
    if [row["role"] for row in rows] != [POLICY_SOURCE_ROLE, POLICY_CONFIG_ROLE]:
        raise PolicyContractError("policy object roles differ from canonical order")
    objects = {row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in rows}
    if objects[POLICY_SOURCE_ROLE] != source or objects[POLICY_CONFIG_ROLE] != config:
        raise PolicyContractError(
            "policy source/config metadata cross-reference mismatch"
        )
    manifest_sha256 = (
        ref.manifest.sha256
        if isinstance(ref, PolicyDefinitionRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedPolicyDefinition(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha256,
        policy_id=_policy_id(metadata["policy_id"]),
        policy_version=_policy_version(metadata["policy_version"]),
        source_commit=_source_commit(metadata["source_commit"]),
        source=source,
        config=config,
        capabilities=capabilities,
    )


def _validate_action_semantics(
    action_kind: Any,
    stock: ObjectRef,
    selected: ObjectRef,
    action_reason: Any,
    fallback_reason: Any,
) -> tuple[str, str, str | None]:
    if action_kind == "stock_fallback":
        if selected != stock:
            raise PolicyContractError(
                "stock fallback selected pseudo-gradient must reference the exact stock object"
            )
        fallback = _reason(fallback_reason, "fallback_reason")
    elif action_kind == "nonstock":
        if selected == stock:
            raise PolicyContractError(
                "nonstock selected pseudo-gradient must differ from the stock object"
            )
        if fallback_reason is not None:
            raise PolicyContractError("nonstock action cannot contain fallback_reason")
        fallback = None
    else:
        raise PolicyContractError("action_kind must be 'stock_fallback' or 'nonstock'")
    return action_kind, _reason(action_reason, "action_reason"), fallback


def _validate_boundary_action_weights(boundary: LoadedSyncerBoundary) -> None:
    for responder in boundary.responders:
        weight = struct.unpack(">d", bytes.fromhex(responder.weight_f64_bits))[0]
        if not math.isfinite(weight) or weight <= 0.0:
            raise PolicyContractError(
                "sealed action requires every responder weight_f64_bits to decode "
                "to a finite strictly positive f64"
            )


def publish_sealed_outer_action(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    policy: PolicyDefinitionRef,
    boundary: SyncerBoundaryRef,
    fragment_id: int,
    required_capabilities: Iterable[str],
    stock_pseudo_gradient: ObjectRef,
    selected_pseudo_gradient: ObjectRef,
    outer_lr_f64_bits: str,
    resulting_fragment: ObjectRef,
    decision_sha256: str,
    config_sha256: str,
    action_kind: str,
    action_reason: str,
    fallback_reason: str | None,
) -> SealedOuterActionRef:
    """Seal an action independently of all future outcome measurements."""

    if not isinstance(policy, PolicyDefinitionRef):
        raise TypeError("policy must be PolicyDefinitionRef")
    if not isinstance(boundary, SyncerBoundaryRef):
        raise TypeError("boundary must be SyncerBoundaryRef")
    loaded_policy = load_policy_definition(store, policy)
    loaded_boundary = load_syncer_boundary(store, boundary)
    _validate_boundary_action_weights(loaded_boundary)
    if type(fragment_id) is not int or fragment_id < 0:
        raise PolicyContractError("fragment_id must be a non-negative integer")
    if fragment_id != loaded_boundary.identity.fragment_id:
        raise PolicyContractError("action fragment_id differs from syncer boundary")
    required = _capabilities(required_capabilities, "required capabilities")
    missing = sorted(set(required) - set(loaded_policy.capabilities))
    if missing:
        raise PolicyContractError(f"policy is missing required capabilities: {missing}")
    stock = _exact_ref(stock_pseudo_gradient, "stock_pseudo_gradient")
    selected = _exact_ref(selected_pseudo_gradient, "selected_pseudo_gradient")
    result = _exact_ref(resulting_fragment, "resulting_fragment")
    action_kind, action_reason, fallback_reason = _validate_action_semantics(
        action_kind, stock, selected, action_reason, fallback_reason
    )
    config_sha256 = _sha256(config_sha256, "config_sha256")
    if config_sha256 != loaded_policy.config.sha256:
        raise PolicyContractError(
            "action config_sha256 differs from policy config object"
        )

    metadata = {
        "schema": ACTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "policy": _manifest_descriptor(policy.manifest),
        "boundary": _manifest_descriptor(boundary.manifest),
        "fragment_id": fragment_id,
        "required_capabilities": list(required),
        "stock_pseudo_gradient": _object_descriptor(STOCK_GRADIENT_ROLE, stock),
        "selected_pseudo_gradient": _object_descriptor(
            SELECTED_GRADIENT_ROLE, selected
        ),
        "outer_lr_f64_bits": _f64_bits(outer_lr_f64_bits, "outer_lr_f64_bits"),
        "resulting_fragment": _object_descriptor(RESULTING_FRAGMENT_ROLE, result),
        "decision_sha256": _sha256(decision_sha256, "decision_sha256"),
        "config_sha256": config_sha256,
        "action_kind": action_kind,
        "action_reason": action_reason,
        "fallback_reason": fallback_reason,
    }
    manifest = store.publish_manifest(
        manifest_id,
        [
            ManifestEntry(STOCK_GRADIENT_ROLE, stock),
            ManifestEntry(SELECTED_GRADIENT_ROLE, selected),
            ManifestEntry(RESULTING_FRAGMENT_ROLE, result),
        ],
        metadata=metadata,
    )
    return SealedOuterActionRef(manifest)


def load_sealed_outer_action(
    store: CaptureObjectStore, ref: SealedOuterActionRef | ManifestRef | str
) -> LoadedSealedOuterAction:
    """Verify a sealed action, its policy/boundary, and all exact objects."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, SealedOuterActionRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    expected_keys = {
        "schema",
        "schema_version",
        "policy",
        "boundary",
        "fragment_id",
        "required_capabilities",
        "stock_pseudo_gradient",
        "selected_pseudo_gradient",
        "outer_lr_f64_bits",
        "resulting_fragment",
        "decision_sha256",
        "config_sha256",
        "action_kind",
        "action_reason",
        "fallback_reason",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise PolicyContractError(
            "sealed action metadata fields are malformed or contain outcome fields"
        )
    if metadata["schema"] != ACTION_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise PolicyContractError("sealed action uses an unsupported schema")
    policy_manifest = _parse_manifest_descriptor(metadata["policy"], "action policy")
    boundary_manifest = _parse_manifest_descriptor(
        metadata["boundary"], "action boundary"
    )
    policy_ref = PolicyDefinitionRef(policy_manifest)
    boundary_ref = SyncerBoundaryRef(boundary_manifest)
    policy = load_policy_definition(store, policy_ref)
    boundary = load_syncer_boundary(store, boundary_ref)
    _validate_boundary_action_weights(boundary)
    fragment_id = metadata["fragment_id"]
    if type(fragment_id) is not int or fragment_id < 0:
        raise PolicyContractError("action fragment_id must be a non-negative integer")
    if fragment_id != boundary.identity.fragment_id:
        raise PolicyContractError("action fragment_id differs from syncer boundary")
    if not isinstance(metadata["required_capabilities"], list):
        raise PolicyContractError("required capabilities must be an array")
    required = _capabilities(
        metadata["required_capabilities"],
        "required capabilities",
    )
    if list(metadata["required_capabilities"]) != list(required):
        raise PolicyContractError("required capabilities are not in canonical order")
    missing = sorted(set(required) - set(policy.capabilities))
    if missing:
        raise PolicyContractError(f"policy is missing required capabilities: {missing}")

    stock = _parse_object_descriptor(
        metadata["stock_pseudo_gradient"],
        STOCK_GRADIENT_ROLE,
        "stock pseudo-gradient",
    )
    selected = _parse_object_descriptor(
        metadata["selected_pseudo_gradient"],
        SELECTED_GRADIENT_ROLE,
        "selected pseudo-gradient",
    )
    result = _parse_object_descriptor(
        metadata["resulting_fragment"],
        RESULTING_FRAGMENT_ROLE,
        "resulting fragment",
    )
    action_kind, action_reason, fallback_reason = _validate_action_semantics(
        metadata["action_kind"],
        stock,
        selected,
        metadata["action_reason"],
        metadata["fallback_reason"],
    )
    config_sha256 = _sha256(metadata["config_sha256"], "action config_sha256")
    if config_sha256 != policy.config.sha256:
        raise PolicyContractError(
            "action config_sha256 differs from policy config object"
        )

    rows = manifest["objects"]
    expected_roles = [
        STOCK_GRADIENT_ROLE,
        SELECTED_GRADIENT_ROLE,
        RESULTING_FRAGMENT_ROLE,
    ]
    if [row["role"] for row in rows] != expected_roles:
        raise PolicyContractError(
            "sealed action object roles differ from canonical order"
        )
    objects = {row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in rows}
    if (
        objects[STOCK_GRADIENT_ROLE] != stock
        or objects[SELECTED_GRADIENT_ROLE] != selected
        or objects[RESULTING_FRAGMENT_ROLE] != result
    ):
        raise PolicyContractError(
            "sealed action metadata/object cross-reference mismatch"
        )
    manifest_sha256 = (
        ref.manifest.sha256
        if isinstance(ref, SealedOuterActionRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedSealedOuterAction(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha256,
        policy=policy,
        policy_ref=policy_ref,
        boundary=boundary,
        boundary_ref=boundary_ref,
        fragment_id=fragment_id,
        required_capabilities=required,
        stock_pseudo_gradient=stock,
        selected_pseudo_gradient=selected,
        outer_lr_f64_bits=_f64_bits(
            metadata["outer_lr_f64_bits"], "action outer_lr_f64_bits"
        ),
        resulting_fragment=result,
        decision_sha256=_sha256(metadata["decision_sha256"], "action decision_sha256"),
        config_sha256=config_sha256,
        action_kind=action_kind,
        action_reason=action_reason,
        fallback_reason=fallback_reason,
    )


def _loss(value: Any, context: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise PolicyContractError(f"{context} must be a finite JSON float")
    return value


def publish_policy_outcome(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    action: SealedOuterActionRef,
    k0: ObjectRef,
    k0_loss: float,
    k8: ObjectRef,
    k8_loss: float,
    evaluation: ObjectRef,
    evaluation_loss: float,
) -> PolicyOutcomeRef:
    """Append one immutable outcome manifest without modifying action bytes."""

    if not isinstance(action, SealedOuterActionRef):
        raise TypeError("action must be SealedOuterActionRef")
    load_sealed_outer_action(store, action)
    k0 = _exact_ref(k0, "k0")
    k8 = _exact_ref(k8, "k8")
    evaluation = _exact_ref(evaluation, "evaluation")
    metadata = {
        "schema": OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "action": _manifest_descriptor(action.manifest),
        "k0": {**_object_descriptor(K0_ROLE, k0), "loss": _loss(k0_loss, "k0_loss")},
        "k8": {**_object_descriptor(K8_ROLE, k8), "loss": _loss(k8_loss, "k8_loss")},
        "evaluation": {
            **_object_descriptor(EVALUATION_ROLE, evaluation),
            "loss": _loss(evaluation_loss, "evaluation_loss"),
        },
    }
    manifest = store.publish_manifest(
        manifest_id,
        [
            ManifestEntry(K0_ROLE, k0),
            ManifestEntry(K8_ROLE, k8),
            ManifestEntry(EVALUATION_ROLE, evaluation),
        ],
        metadata=metadata,
    )
    return PolicyOutcomeRef(manifest)


def _parse_outcome_value(
    value: Any, role: str, context: str
) -> tuple[ObjectRef, float]:
    if not isinstance(value, dict) or set(value) != {"role", "sha256", "bytes", "loss"}:
        raise PolicyContractError(f"{context} fields are malformed")
    ref = _parse_object_descriptor(
        {key: value[key] for key in ("role", "sha256", "bytes")}, role, context
    )
    return ref, _loss(value["loss"], f"{context} loss")


def load_policy_outcome(
    store: CaptureObjectStore, ref: PolicyOutcomeRef | ManifestRef | str
) -> LoadedPolicyOutcome:
    """Verify one outcome, its sealed action, finite losses, and exact objects."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, PolicyOutcomeRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    expected_keys = {
        "schema",
        "schema_version",
        "action",
        "k0",
        "k8",
        "evaluation",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise PolicyContractError("policy outcome metadata fields are malformed")
    if metadata["schema"] != OUTCOME_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise PolicyContractError("policy outcome uses an unsupported schema")
    action_manifest = _parse_manifest_descriptor(metadata["action"], "outcome action")
    action_ref = SealedOuterActionRef(action_manifest)
    action = load_sealed_outer_action(store, action_ref)
    k0, k0_loss = _parse_outcome_value(metadata["k0"], K0_ROLE, "outcome k0")
    k8, k8_loss = _parse_outcome_value(metadata["k8"], K8_ROLE, "outcome k8")
    evaluation, evaluation_loss = _parse_outcome_value(
        metadata["evaluation"], EVALUATION_ROLE, "outcome evaluation"
    )
    rows = manifest["objects"]
    expected_roles = [K0_ROLE, K8_ROLE, EVALUATION_ROLE]
    if [row["role"] for row in rows] != expected_roles:
        raise PolicyContractError(
            "policy outcome object roles differ from canonical order"
        )
    objects = {row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in rows}
    if (
        objects[K0_ROLE] != k0
        or objects[K8_ROLE] != k8
        or objects[EVALUATION_ROLE] != evaluation
    ):
        raise PolicyContractError(
            "policy outcome metadata/object cross-reference mismatch"
        )
    manifest_sha256 = (
        ref.manifest.sha256
        if isinstance(ref, PolicyOutcomeRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedPolicyOutcome(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha256,
        action=action,
        action_ref=action_ref,
        k0=k0,
        k0_loss=k0_loss,
        k8=k8,
        k8_loss=k8_loss,
        evaluation=evaluation,
        evaluation_loss=evaluation_loss,
    )
