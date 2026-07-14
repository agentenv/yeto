"""Pre-outcome authority and durable evidence for capture-v2 CRN evaluation.

The generic policy contracts intentionally permit many optimizer experiments.
This module is narrower: it freezes the one PTI/SGD-0.28 paired evaluation
before callbacks run, retains the complete A/B and B/A isolation trace, and
publishes both arms atomically.  Generic ``PolicyOutcomeRef`` records are not
scientifically admissible substitutes for the paired artifact defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import struct
from typing import Any, Sequence

from .capture_v2_policy import (
    LoadedSealedOuterAction,
    SealedOuterActionRef,
    load_sealed_outer_action,
)
from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestEntry,
    ManifestRef,
    ObjectRef,
)
from .capture_v2_syncer import (
    LoadedResponder,
    LoadedSyncerBoundary,
    SyncerBoundaryRef,
    load_syncer_boundary,
)


PLAN_SCHEMA = "yeto.capture-v2-crn-evaluation-plan"
CAMPAIGN_INDEX_SCHEMA = "yeto.capture-v2-crn-campaign-index"
ATTESTATION_SCHEMA = "yeto.capture-v2-crn-isolation-attestation"
PAIRED_OUTCOME_SCHEMA = "yeto.capture-v2-crn-paired-outcome"
SCHEMA_VERSION = 1

FUTURE_GROUP_COUNT = 8
HORIZONS = (0, 8)
SCHEDULE = (
    ("stock-candidate", ("stock", "candidate")),
    ("candidate-stock", ("candidate", "stock")),
)
TRACE_SCHEDULE = (
    ("stock-candidate", 0, "stock"),
    ("stock-candidate", 1, "candidate"),
    ("candidate-stock", 0, "candidate"),
    ("candidate-stock", 1, "stock"),
)
REQUIRED_CAPABILITIES = frozenset({"worker_restore", "crn_train_k8"})
PTI_OUTER_LR_F64_BITS = struct.pack(">d", 0.28).hex()
PTI_ZERO_MOMENTUM_F64_BITS = struct.pack(">d", 0.0).hex()
PTI_MEMORYLESS_OUTER_CONFIG_NAMES = frozenset({"sgd", "stock-sgd"})

PLAN_EVALUATION_ROLE = "crn-plan/evaluation"
PLAN_EVALUATOR_SOURCE_ROLE = "crn-plan/evaluator-source"
PLAN_EVALUATOR_CONFIG_ROLE = "crn-plan/evaluator-config"
PAIRED_EVALUATION_ROLE = "crn-pair/evaluation"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_F64_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")
_SOURCE_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_MANIFEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IMAGE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}\Z")


class CRNAuthorityError(CaptureStoreError):
    """A CRN plan, attestation, or paired outcome is not authoritative."""


@dataclass(frozen=True)
class CRNEvaluationPlanRef:
    """Content identity of one pre-outcome CRN evaluation plan."""

    manifest: ManifestRef


@dataclass(frozen=True)
class CRNCampaignIndexRef:
    """Content identity of the precommitted authoritative plan set."""

    manifest: ManifestRef


@dataclass(frozen=True)
class EvaluatorProvenance:
    """Exact code/config/image identity of the evaluator backend."""

    source_commit: str
    image_id: str
    source: ObjectRef
    config: ObjectRef


@dataclass(frozen=True)
class LoadedCRNEvaluationPlan:
    """Strictly verified plan and all selected immutable inputs."""

    manifest_id: str
    manifest_sha256: str
    boundary_ref: SyncerBoundaryRef
    boundary: LoadedSyncerBoundary
    stock_ref: SealedOuterActionRef
    stock: LoadedSealedOuterAction
    candidate_ref: SealedOuterActionRef
    candidate: LoadedSealedOuterAction
    responder_index: int
    responder: LoadedResponder
    evaluation: ObjectRef
    future_groups: tuple[ObjectRef, ...]
    expected_cuda_rng_streams: int
    evaluator: EvaluatorProvenance
    attestation_manifest_id: str
    paired_outcome_manifest_id: str


@dataclass(frozen=True)
class LoadedCRNCampaignIndex:
    """Verified ordered authority over a closed set of plans."""

    manifest_id: str
    manifest_sha256: str
    campaign_id: str
    plans: tuple[CRNEvaluationPlanRef, ...]


@dataclass(frozen=True)
class CRNEvaluationEvidence:
    """One exact k0 or k8 evaluation receipt retained in an attestation."""

    step: int
    evaluation: ObjectRef
    artifact: ObjectRef
    state_sha256: str
    loss_f64_bits: str


@dataclass(frozen=True)
class CRNGroupEvidence:
    """One exact future-group receipt retained in an attestation."""

    group_index: int
    future_group: ObjectRef
    batch_sha256: str
    state_sha256: str


@dataclass(frozen=True)
class CRNArmEvidence:
    """One complete arm execution at one position in the frozen schedule."""

    order: str
    position: int
    arm: str
    action: ManifestRef
    restore_state_sha256: str
    applied_state_sha256: str
    k0: CRNEvaluationEvidence
    groups: tuple[CRNGroupEvidence, ...]
    k8: CRNEvaluationEvidence
    final_state_sha256: str


@dataclass(frozen=True)
class CRNIsolationAttestationRef:
    """Content identity of a complete four-trace isolation attestation."""

    manifest: ManifestRef


@dataclass(frozen=True)
class LoadedCRNIsolationAttestation:
    """Verified durable isolation evidence bound to an authorized plan."""

    manifest_id: str
    manifest_sha256: str
    campaign_index_ref: CRNCampaignIndexRef
    plan_ref: CRNEvaluationPlanRef
    plan: LoadedCRNEvaluationPlan
    traces: tuple[CRNArmEvidence, ...]


@dataclass(frozen=True)
class CRNPairedOutcomeRef:
    """Atomic stock/candidate outcome identity for the PTI campaign."""

    manifest: ManifestRef


@dataclass(frozen=True)
class CRNArmOutcome:
    """One arm of an attested atomic pair."""

    action: ManifestRef
    k0_artifact: ObjectRef
    k0_loss_f64_bits: str
    k8_artifact: ObjectRef
    k8_loss_f64_bits: str

    @property
    def k0_loss(self) -> float:
        return struct.unpack(">d", bytes.fromhex(self.k0_loss_f64_bits))[0]

    @property
    def k8_loss(self) -> float:
        return struct.unpack(">d", bytes.fromhex(self.k8_loss_f64_bits))[0]


@dataclass(frozen=True)
class LoadedCRNPairedOutcome:
    """Scientifically admissible, attested stock/candidate outcome pair."""

    manifest_id: str
    manifest_sha256: str
    campaign_index_ref: CRNCampaignIndexRef
    plan_ref: CRNEvaluationPlanRef
    attestation_ref: CRNIsolationAttestationRef
    evaluation: ObjectRef
    stock: CRNArmOutcome
    candidate: CRNArmOutcome

    @property
    def scientifically_admissible(self) -> bool:
        return True


def _manifest_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or _MANIFEST_ID_RE.fullmatch(value) is None:
        raise CRNAuthorityError(f"{context} is not a canonical manifest_id")
    return value


def _campaign_id(value: Any) -> str:
    if not isinstance(value, str) or _CAMPAIGN_ID_RE.fullmatch(value) is None:
        raise CRNAuthorityError("campaign_id is malformed")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CRNAuthorityError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _source_commit(value: Any) -> str:
    if not isinstance(value, str) or _SOURCE_COMMIT_RE.fullmatch(value) is None:
        raise CRNAuthorityError("evaluator source_commit must be lowercase 40-hex")
    return value


def _image_id(value: Any) -> str:
    if not isinstance(value, str) or _IMAGE_ID_RE.fullmatch(value) is None:
        raise CRNAuthorityError("evaluator image_id is malformed")
    return value


def _exact_nonnegative_int(value: Any, context: str) -> int:
    if type(value) is not int or value < 0:
        raise CRNAuthorityError(f"{context} must be a non-negative integer")
    if value > 2**63 - 1:
        raise CRNAuthorityError(f"{context} exceeds signed int64")
    return value


def _manifest_descriptor(ref: ManifestRef) -> dict[str, Any]:
    if not isinstance(ref, ManifestRef):
        raise TypeError("manifest descriptor requires ManifestRef")
    return {"manifest_id": ref.manifest_id, "sha256": ref.sha256, "bytes": ref.bytes}


def _parse_manifest_descriptor(value: Any, context: str) -> ManifestRef:
    if not isinstance(value, dict) or set(value) != {"manifest_id", "sha256", "bytes"}:
        raise CRNAuthorityError(f"{context} manifest descriptor is malformed")
    try:
        return ManifestRef(value["manifest_id"], value["sha256"], value["bytes"], False)
    except CaptureStoreError as exc:
        raise CRNAuthorityError(
            f"{context} manifest descriptor is invalid: {exc}"
        ) from exc


def _same_manifest(left: ManifestRef, right: ManifestRef) -> bool:
    return _manifest_descriptor(left) == _manifest_descriptor(right)


def _object_descriptor(role: str, ref: ObjectRef) -> dict[str, Any]:
    if not isinstance(ref, ObjectRef):
        raise TypeError("object descriptor requires ObjectRef")
    return {"role": role, "sha256": ref.sha256, "bytes": ref.bytes}


def _parse_object_descriptor(value: Any, role: str, context: str) -> ObjectRef:
    if not isinstance(value, dict) or set(value) != {"role", "sha256", "bytes"}:
        raise CRNAuthorityError(f"{context} object descriptor is malformed")
    if value["role"] != role:
        raise CRNAuthorityError(f"{context} object role is noncanonical")
    try:
        return ObjectRef(value["sha256"], value["bytes"])
    except CaptureStoreError as exc:
        raise CRNAuthorityError(
            f"{context} object descriptor is invalid: {exc}"
        ) from exc


def _plan_group_role(index: int) -> str:
    return f"crn-plan/future-groups/{index}"


def _attestation_artifact_role(trace_index: int, step: int) -> str:
    return f"crn-attestation/traces/{trace_index}/k{step}-artifact"


def _paired_artifact_role(arm: str, step: int) -> str:
    return f"crn-pair/{arm}/k{step}-artifact"


def _require_exact_lr(
    boundary: LoadedSyncerBoundary, *actions: LoadedSealedOuterAction
) -> None:
    parameters = boundary.outer_config.parameters
    if not isinstance(parameters, dict):
        raise CRNAuthorityError("PTI boundary outer parameters must be a mapping")
    boundary_bits = parameters.get("lr_f64_bits")
    if boundary.outer_config.name == "nesterov":
        if set(parameters) != {"lr_f64_bits", "momentum_f64_bits"}:
            raise CRNAuthorityError(
                "PTI Nesterov compatibility requires only LR and momentum bits"
            )
        if parameters["momentum_f64_bits"] != PTI_ZERO_MOMENTUM_F64_BITS:
            raise CRNAuthorityError(
                "PTI Nesterov compatibility requires exact +0.0 f64 momentum"
            )
    elif boundary.outer_config.name in PTI_MEMORYLESS_OUTER_CONFIG_NAMES:
        if set(parameters) != {"lr_f64_bits"}:
            raise CRNAuthorityError("PTI SGD outer config contains noncanonical state")
    else:
        raise CRNAuthorityError("PTI plan requires a memoryless SGD outer boundary")
    if boundary_bits != PTI_OUTER_LR_F64_BITS:
        raise CRNAuthorityError("PTI boundary outer LR must be exactly f64 0.28")
    for action in actions:
        if action.outer_lr_f64_bits != PTI_OUTER_LR_F64_BITS:
            raise CRNAuthorityError(
                "stock, candidate, and boundary outer LR must all be exact f64 0.28"
            )


def _require_action_capabilities(action: LoadedSealedOuterAction, context: str) -> None:
    required = set(action.required_capabilities)
    declared = set(action.policy.capabilities)
    if not REQUIRED_CAPABILITIES <= required or not REQUIRED_CAPABILITIES <= declared:
        raise CRNAuthorityError(
            f"{context} lacks required worker_restore+crn_train_k8 capability binding"
        )


def _load_plan_inputs(
    store: CaptureObjectStore,
    *,
    boundary_ref: SyncerBoundaryRef,
    stock_ref: SealedOuterActionRef,
    candidate_ref: SealedOuterActionRef,
    responder_index: int,
    evaluation: ObjectRef,
    expected_cuda_rng_streams: int,
    evaluator: EvaluatorProvenance,
) -> tuple[
    LoadedSyncerBoundary,
    LoadedSealedOuterAction,
    LoadedSealedOuterAction,
    LoadedResponder,
    tuple[ObjectRef, ...],
]:
    if not isinstance(boundary_ref, SyncerBoundaryRef):
        raise TypeError("boundary must be SyncerBoundaryRef")
    if not isinstance(stock_ref, SealedOuterActionRef):
        raise TypeError("stock_action must be SealedOuterActionRef")
    if not isinstance(candidate_ref, SealedOuterActionRef):
        raise TypeError("candidate_action must be SealedOuterActionRef")
    responder_index = _exact_nonnegative_int(responder_index, "responder_index")
    expected_cuda_rng_streams = _exact_nonnegative_int(
        expected_cuda_rng_streams, "expected_cuda_rng_streams"
    )
    if not isinstance(evaluation, ObjectRef):
        raise TypeError("evaluation must be ObjectRef")
    if not isinstance(evaluator, EvaluatorProvenance):
        raise TypeError("evaluator must be EvaluatorProvenance")
    store.verify_object(evaluation)
    store.verify_object(evaluator.source)
    store.verify_object(evaluator.config)
    _source_commit(evaluator.source_commit)
    _image_id(evaluator.image_id)

    boundary = load_syncer_boundary(store, boundary_ref)
    stock = load_sealed_outer_action(store, stock_ref)
    candidate = load_sealed_outer_action(store, candidate_ref)
    for action, context in ((stock, "stock action"), (candidate, "candidate action")):
        if not _same_manifest(action.boundary_ref.manifest, boundary_ref.manifest):
            raise CRNAuthorityError(f"{context} is cross-wired to another boundary")
        _require_action_capabilities(action, context)
    if stock.action_kind != "stock_fallback":
        raise CRNAuthorityError("stock action must be an exact stock_fallback")
    if candidate.action_kind != "nonstock":
        raise CRNAuthorityError("candidate action must be nonstock")
    if stock.stock_pseudo_gradient != candidate.stock_pseudo_gradient:
        raise CRNAuthorityError(
            "stock and candidate actions use different stock objects"
        )
    _require_exact_lr(boundary, stock, candidate)

    if responder_index >= len(boundary.responders):
        raise CRNAuthorityError("responder_index is absent from the boundary")
    responder = boundary.responders[responder_index]
    endpoint = responder.endpoint
    captured = endpoint.future_groups
    expected_indices = list(range(FUTURE_GROUP_COUNT))
    if captured.state != "complete" or list(captured.refs) != expected_indices:
        raise CRNAuthorityError(
            "selected endpoint lacks exact future groups 0 through 7"
        )
    if len(endpoint.rng.torch_cuda) != expected_cuda_rng_streams:
        raise CRNAuthorityError(
            "selected endpoint CUDA RNG stream count differs from the frozen plan"
        )
    groups = tuple(captured.refs[index] for index in expected_indices)
    for index, ref in enumerate(groups):
        store.verify_object(ref)
        if not isinstance(ref, ObjectRef):  # Defensive after endpoint validation.
            raise CRNAuthorityError(f"future group {index} is not an ObjectRef")
    return boundary, stock, candidate, responder, groups


def _responder_value(index: int, responder: LoadedResponder) -> dict[str, Any]:
    identity = responder.endpoint.identity
    return {
        "index": index,
        "endpoint": _manifest_descriptor(responder.endpoint_ref.manifest),
        "learner_id": identity.learner_id,
        "rank": identity.rank,
        "window_uuid": identity.window_uuid,
        "payload_sha256": responder.payload_sha256,
    }


def _schedule_value() -> list[dict[str, Any]]:
    return [
        {"order_index": index, "order": order, "arms": list(arms)}
        for index, (order, arms) in enumerate(SCHEDULE)
    ]


def publish_crn_evaluation_plan(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    boundary: SyncerBoundaryRef,
    stock_action: SealedOuterActionRef,
    candidate_action: SealedOuterActionRef,
    responder_index: int,
    evaluation: ObjectRef,
    expected_cuda_rng_streams: int,
    evaluator: EvaluatorProvenance,
    attestation_manifest_id: str,
    paired_outcome_manifest_id: str,
) -> CRNEvaluationPlanRef:
    """Seal every outcome-affecting CRN choice before any callback can run."""

    manifest_id = _manifest_id(manifest_id, "plan manifest_id")
    attestation_manifest_id = _manifest_id(
        attestation_manifest_id, "attestation manifest_id"
    )
    paired_outcome_manifest_id = _manifest_id(
        paired_outcome_manifest_id, "paired outcome manifest_id"
    )
    if len({manifest_id, attestation_manifest_id, paired_outcome_manifest_id}) != 3:
        raise CRNAuthorityError("plan, attestation, and paired outcome IDs must differ")
    loaded = _load_plan_inputs(
        store,
        boundary_ref=boundary,
        stock_ref=stock_action,
        candidate_ref=candidate_action,
        responder_index=responder_index,
        evaluation=evaluation,
        expected_cuda_rng_streams=expected_cuda_rng_streams,
        evaluator=evaluator,
    )
    loaded_boundary, _stock, _candidate, responder, groups = loaded
    entries = [ManifestEntry(PLAN_EVALUATION_ROLE, evaluation)]
    entries.extend(
        ManifestEntry(_plan_group_role(index), ref) for index, ref in enumerate(groups)
    )
    entries.extend(
        [
            ManifestEntry(PLAN_EVALUATOR_SOURCE_ROLE, evaluator.source),
            ManifestEntry(PLAN_EVALUATOR_CONFIG_ROLE, evaluator.config),
        ]
    )
    metadata = {
        "schema": PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "boundary": _manifest_descriptor(boundary.manifest),
        "actions": {
            "stock": _manifest_descriptor(stock_action.manifest),
            "candidate": _manifest_descriptor(candidate_action.manifest),
        },
        "responder": _responder_value(responder_index, responder),
        "evaluation": _object_descriptor(PLAN_EVALUATION_ROLE, evaluation),
        "future_groups": [
            {
                "index": index,
                **_object_descriptor(_plan_group_role(index), ref),
            }
            for index, ref in enumerate(groups)
        ],
        "expected_cuda_rng_streams": expected_cuda_rng_streams,
        "horizons": list(HORIZONS),
        "schedule": _schedule_value(),
        "evaluator": {
            "source_commit": evaluator.source_commit,
            "image_id": evaluator.image_id,
            "source": _object_descriptor(PLAN_EVALUATOR_SOURCE_ROLE, evaluator.source),
            "config": _object_descriptor(PLAN_EVALUATOR_CONFIG_ROLE, evaluator.config),
        },
        "attestation_manifest_id": attestation_manifest_id,
        "paired_outcome_manifest_id": paired_outcome_manifest_id,
    }
    # Loading the boundary above is part of publication, not an unused check:
    # it proves the selected responder belongs to the exact boundary.
    if loaded_boundary.identity.fragment_id != _stock.fragment_id:
        raise CRNAuthorityError("action fragment differs from plan boundary")
    ref = store.publish_manifest(manifest_id, entries, metadata=metadata)
    return CRNEvaluationPlanRef(ref)


def _plan_object_roles() -> list[str]:
    roles = [PLAN_EVALUATION_ROLE]
    roles.extend(_plan_group_role(index) for index in range(FUTURE_GROUP_COUNT))
    roles.extend([PLAN_EVALUATOR_SOURCE_ROLE, PLAN_EVALUATOR_CONFIG_ROLE])
    return roles


def load_crn_evaluation_plan(
    store: CaptureObjectStore, ref: CRNEvaluationPlanRef | ManifestRef | str
) -> LoadedCRNEvaluationPlan:
    """Load a plan only after recomputing every boundary/action/object binding."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, CRNEvaluationPlanRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    expected_keys = {
        "schema",
        "schema_version",
        "boundary",
        "actions",
        "responder",
        "evaluation",
        "future_groups",
        "expected_cuda_rng_streams",
        "horizons",
        "schedule",
        "evaluator",
        "attestation_manifest_id",
        "paired_outcome_manifest_id",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise CRNAuthorityError("CRN plan metadata fields are malformed")
    if metadata["schema"] != PLAN_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise CRNAuthorityError("CRN plan uses an unsupported schema")
    if metadata["horizons"] != list(HORIZONS):
        raise CRNAuthorityError("CRN plan horizons must be exactly [0, 8]")
    if metadata["schedule"] != _schedule_value():
        raise CRNAuthorityError("CRN plan arm schedule is not canonical")

    boundary_manifest = _parse_manifest_descriptor(
        metadata["boundary"], "plan boundary"
    )
    actions = metadata["actions"]
    if not isinstance(actions, dict) or set(actions) != {"stock", "candidate"}:
        raise CRNAuthorityError("CRN plan actions are malformed")
    stock_manifest = _parse_manifest_descriptor(actions["stock"], "plan stock action")
    candidate_manifest = _parse_manifest_descriptor(
        actions["candidate"], "plan candidate action"
    )
    boundary_ref = SyncerBoundaryRef(boundary_manifest)
    stock_ref = SealedOuterActionRef(stock_manifest)
    candidate_ref = SealedOuterActionRef(candidate_manifest)

    responder_value = metadata["responder"]
    responder_keys = {
        "index",
        "endpoint",
        "learner_id",
        "rank",
        "window_uuid",
        "payload_sha256",
    }
    if not isinstance(responder_value, dict) or set(responder_value) != responder_keys:
        raise CRNAuthorityError("CRN plan responder fields are malformed")
    responder_index = _exact_nonnegative_int(
        responder_value["index"], "plan responder index"
    )
    endpoint_manifest = _parse_manifest_descriptor(
        responder_value["endpoint"], "plan responder endpoint"
    )
    expected_cuda = _exact_nonnegative_int(
        metadata["expected_cuda_rng_streams"], "plan expected CUDA RNG streams"
    )

    object_rows = manifest["objects"]
    expected_roles = _plan_object_roles()
    if [row["role"] for row in object_rows] != expected_roles:
        raise CRNAuthorityError("CRN plan object roles are noncanonical")
    objects = {
        row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in object_rows
    }
    evaluation = _parse_object_descriptor(
        metadata["evaluation"], PLAN_EVALUATION_ROLE, "plan evaluation"
    )
    if objects[PLAN_EVALUATION_ROLE] != evaluation:
        raise CRNAuthorityError("CRN plan evaluation object cross-reference mismatch")
    group_rows = metadata["future_groups"]
    if not isinstance(group_rows, list) or len(group_rows) != FUTURE_GROUP_COUNT:
        raise CRNAuthorityError("CRN plan must contain exactly eight future-group rows")
    groups: list[ObjectRef] = []
    for index, row in enumerate(group_rows):
        role = _plan_group_role(index)
        if not isinstance(row, dict) or set(row) != {
            "index",
            "role",
            "sha256",
            "bytes",
        }:
            raise CRNAuthorityError(f"CRN plan future-group row {index} is malformed")
        if type(row["index"]) is not int or row["index"] != index:
            raise CRNAuthorityError("CRN plan future-group indices are noncanonical")
        group = _parse_object_descriptor(
            {key: row[key] for key in ("role", "sha256", "bytes")},
            role,
            f"plan future group {index}",
        )
        if objects[role] != group:
            raise CRNAuthorityError(
                f"plan future group {index} cross-reference mismatch"
            )
        groups.append(group)

    evaluator_value = metadata["evaluator"]
    if not isinstance(evaluator_value, dict) or set(evaluator_value) != {
        "source_commit",
        "image_id",
        "source",
        "config",
    }:
        raise CRNAuthorityError("CRN plan evaluator provenance fields are malformed")
    source = _parse_object_descriptor(
        evaluator_value["source"], PLAN_EVALUATOR_SOURCE_ROLE, "evaluator source"
    )
    config = _parse_object_descriptor(
        evaluator_value["config"], PLAN_EVALUATOR_CONFIG_ROLE, "evaluator config"
    )
    if (
        objects[PLAN_EVALUATOR_SOURCE_ROLE] != source
        or objects[PLAN_EVALUATOR_CONFIG_ROLE] != config
    ):
        raise CRNAuthorityError("CRN plan evaluator object cross-reference mismatch")
    evaluator = EvaluatorProvenance(
        _source_commit(evaluator_value["source_commit"]),
        _image_id(evaluator_value["image_id"]),
        source,
        config,
    )

    loaded = _load_plan_inputs(
        store,
        boundary_ref=boundary_ref,
        stock_ref=stock_ref,
        candidate_ref=candidate_ref,
        responder_index=responder_index,
        evaluation=evaluation,
        expected_cuda_rng_streams=expected_cuda,
        evaluator=evaluator,
    )
    boundary, stock, candidate, responder, actual_groups = loaded
    if not _same_manifest(endpoint_manifest, responder.endpoint_ref.manifest):
        raise CRNAuthorityError("CRN plan responder endpoint cross-reference mismatch")
    if responder_value != _responder_value(responder_index, responder):
        raise CRNAuthorityError("CRN plan responder identity cross-reference mismatch")
    if tuple(groups) != actual_groups:
        raise CRNAuthorityError(
            "CRN plan future groups differ from the selected endpoint"
        )

    attestation_id = _manifest_id(
        metadata["attestation_manifest_id"], "plan attestation manifest ID"
    )
    paired_id = _manifest_id(
        metadata["paired_outcome_manifest_id"], "plan paired outcome manifest ID"
    )
    if len({manifest["manifest_id"], attestation_id, paired_id}) != 3:
        raise CRNAuthorityError("plan, attestation, and paired outcome IDs must differ")
    manifest_sha = (
        ref.manifest.sha256
        if isinstance(ref, CRNEvaluationPlanRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedCRNEvaluationPlan(
        manifest_id=manifest["manifest_id"],
        manifest_sha256=manifest_sha,
        boundary_ref=boundary_ref,
        boundary=boundary,
        stock_ref=stock_ref,
        stock=stock,
        candidate_ref=candidate_ref,
        candidate=candidate,
        responder_index=responder_index,
        responder=responder,
        evaluation=evaluation,
        future_groups=actual_groups,
        expected_cuda_rng_streams=expected_cuda,
        evaluator=evaluator,
        attestation_manifest_id=attestation_id,
        paired_outcome_manifest_id=paired_id,
    )


def publish_crn_campaign_index(
    store: CaptureObjectStore,
    manifest_id: str,
    *,
    campaign_id: str,
    plans: Sequence[CRNEvaluationPlanRef],
) -> CRNCampaignIndexRef:
    """Freeze the only plan digests admissible for one campaign denominator."""

    manifest_id = _manifest_id(manifest_id, "campaign index manifest_id")
    campaign_id = _campaign_id(campaign_id)
    if isinstance(plans, (str, bytes)) or not isinstance(plans, Sequence) or not plans:
        raise CRNAuthorityError("campaign index requires a non-empty plan sequence")
    loaded: list[tuple[CRNEvaluationPlanRef, LoadedCRNEvaluationPlan]] = []
    for plan in plans:
        if not isinstance(plan, CRNEvaluationPlanRef):
            raise TypeError("campaign plans must be CRNEvaluationPlanRef values")
        loaded.append((plan, load_crn_evaluation_plan(store, plan)))
    digests = [plan.manifest.sha256 for plan, _ in loaded]
    boundaries = [item.boundary_ref.manifest.sha256 for _, item in loaded]
    outcomes = [item.paired_outcome_manifest_id for _, item in loaded]
    if len(set(digests)) != len(digests):
        raise CRNAuthorityError("campaign index repeats a plan digest")
    if len(set(boundaries)) != len(boundaries):
        raise CRNAuthorityError(
            "campaign index contains competing plans for one boundary"
        )
    if len(set(outcomes)) != len(outcomes):
        raise CRNAuthorityError("campaign index repeats a reserved paired outcome ID")
    metadata = {
        "schema": CAMPAIGN_INDEX_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "plans": [
            {
                "index": index,
                "plan": _manifest_descriptor(plan.manifest),
                "boundary_sha256": item.boundary_ref.manifest.sha256,
                "paired_outcome_manifest_id": item.paired_outcome_manifest_id,
            }
            for index, (plan, item) in enumerate(loaded)
        ],
    }
    return CRNCampaignIndexRef(
        store.publish_manifest(manifest_id, [], metadata=metadata)
    )


def load_crn_campaign_index(
    store: CaptureObjectStore, ref: CRNCampaignIndexRef | ManifestRef | str
) -> LoadedCRNCampaignIndex:
    """Load and revalidate every plan in a precommitted campaign index."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, CRNCampaignIndexRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema",
        "schema_version",
        "campaign_id",
        "plans",
    }:
        raise CRNAuthorityError("campaign index metadata fields are malformed")
    if metadata["schema"] != CAMPAIGN_INDEX_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise CRNAuthorityError("campaign index uses an unsupported schema")
    if manifest["objects"]:
        raise CRNAuthorityError("campaign index must not contain object rows")
    campaign_id = _campaign_id(metadata["campaign_id"])
    rows = metadata["plans"]
    if not isinstance(rows, list) or not rows:
        raise CRNAuthorityError("campaign index plans must be a non-empty array")
    plans: list[CRNEvaluationPlanRef] = []
    boundaries: set[str] = set()
    outcomes: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "index",
            "plan",
            "boundary_sha256",
            "paired_outcome_manifest_id",
        }:
            raise CRNAuthorityError(f"campaign plan row {index} is malformed")
        if type(row["index"]) is not int or row["index"] != index:
            raise CRNAuthorityError("campaign plan indices are noncanonical")
        plan_ref = CRNEvaluationPlanRef(
            _parse_manifest_descriptor(row["plan"], f"campaign plan {index}")
        )
        plan = load_crn_evaluation_plan(store, plan_ref)
        if row["boundary_sha256"] != plan.boundary_ref.manifest.sha256:
            raise CRNAuthorityError("campaign plan boundary summary mismatch")
        if row["paired_outcome_manifest_id"] != plan.paired_outcome_manifest_id:
            raise CRNAuthorityError("campaign plan outcome-ID summary mismatch")
        if plan.boundary_ref.manifest.sha256 in boundaries:
            raise CRNAuthorityError("campaign index contains competing boundary plans")
        if plan.paired_outcome_manifest_id in outcomes:
            raise CRNAuthorityError("campaign index repeats a paired outcome ID")
        boundaries.add(plan.boundary_ref.manifest.sha256)
        outcomes.add(plan.paired_outcome_manifest_id)
        plans.append(plan_ref)
    manifest_sha = (
        ref.manifest.sha256
        if isinstance(ref, CRNCampaignIndexRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedCRNCampaignIndex(
        manifest["manifest_id"], manifest_sha, campaign_id, tuple(plans)
    )


def load_authorized_crn_plan(
    store: CaptureObjectStore,
    campaign_index: CRNCampaignIndexRef,
    plan: CRNEvaluationPlanRef,
) -> LoadedCRNEvaluationPlan:
    """Require exact plan membership in the supplied precommitted authority."""

    if not isinstance(campaign_index, CRNCampaignIndexRef):
        raise TypeError("campaign_index must be CRNCampaignIndexRef")
    if not isinstance(plan, CRNEvaluationPlanRef):
        raise TypeError("plan must be CRNEvaluationPlanRef")
    index = load_crn_campaign_index(store, campaign_index)
    if not any(_same_manifest(item.manifest, plan.manifest) for item in index.plans):
        raise CRNAuthorityError("CRN plan is not authorized by the campaign index")
    return load_crn_evaluation_plan(store, plan)


def _loss_bits(value: Any, context: str) -> str:
    if not isinstance(value, str) or _F64_BITS_RE.fullmatch(value) is None:
        raise CRNAuthorityError(
            f"{context} must be exactly 16 lowercase f64 hex digits"
        )
    decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(decoded):
        raise CRNAuthorityError(f"{context} must decode to a finite f64")
    return value


def _evaluation_value(
    evidence: CRNEvaluationEvidence, *, trace_index: int
) -> dict[str, Any]:
    role = _attestation_artifact_role(trace_index, evidence.step)
    return {
        "step": evidence.step,
        "evaluation": _object_descriptor(PLAN_EVALUATION_ROLE, evidence.evaluation),
        "artifact": _object_descriptor(role, evidence.artifact),
        "state_sha256": evidence.state_sha256,
        "loss_f64_bits": evidence.loss_f64_bits,
    }


def _group_value(evidence: CRNGroupEvidence) -> dict[str, Any]:
    return {
        "group_index": evidence.group_index,
        "future_group": _object_descriptor(
            _plan_group_role(evidence.group_index), evidence.future_group
        ),
        "batch_sha256": evidence.batch_sha256,
        "state_sha256": evidence.state_sha256,
    }


def _trace_value(evidence: CRNArmEvidence, trace_index: int) -> dict[str, Any]:
    return {
        "trace_index": trace_index,
        "order": evidence.order,
        "position": evidence.position,
        "arm": evidence.arm,
        "action": _manifest_descriptor(evidence.action),
        "restore_state_sha256": evidence.restore_state_sha256,
        "applied_state_sha256": evidence.applied_state_sha256,
        "k0": _evaluation_value(evidence.k0, trace_index=trace_index),
        "groups": [_group_value(group) for group in evidence.groups],
        "k8": _evaluation_value(evidence.k8, trace_index=trace_index),
        "final_state_sha256": evidence.final_state_sha256,
    }


def _normalized_trace(evidence: CRNArmEvidence) -> tuple[Any, ...]:
    return (
        evidence.arm,
        _manifest_descriptor(evidence.action),
        evidence.restore_state_sha256,
        evidence.applied_state_sha256,
        evidence.k0,
        evidence.groups,
        evidence.k8,
        evidence.final_state_sha256,
    )


def _validate_traces(
    store: CaptureObjectStore,
    plan: LoadedCRNEvaluationPlan,
    traces: Sequence[CRNArmEvidence],
) -> tuple[CRNArmEvidence, ...]:
    if isinstance(traces, (str, bytes)) or not isinstance(traces, Sequence):
        raise TypeError("traces must be a sequence")
    values = tuple(traces)
    if len(values) != len(TRACE_SCHEDULE):
        raise CRNAuthorityError("isolation attestation requires exactly four traces")
    expected_actions = {
        "stock": plan.stock_ref.manifest,
        "candidate": plan.candidate_ref.manifest,
    }
    for trace_index, (trace, expected_schedule) in enumerate(
        zip(values, TRACE_SCHEDULE, strict=True)
    ):
        if not isinstance(trace, CRNArmEvidence):
            raise TypeError("traces must contain CRNArmEvidence")
        if type(trace.position) is not int:
            raise CRNAuthorityError("isolation trace position must be an exact integer")
        if (trace.order, trace.position, trace.arm) != expected_schedule:
            raise CRNAuthorityError(
                "isolation traces do not follow the frozen schedule"
            )
        if not _same_manifest(trace.action, expected_actions[trace.arm]):
            raise CRNAuthorityError("isolation trace action is cross-wired")
        for value, context in (
            (trace.restore_state_sha256, "restore state hash"),
            (trace.applied_state_sha256, "applied state hash"),
            (trace.final_state_sha256, "final state hash"),
        ):
            _sha256(value, context)
        for evaluation, context in ((trace.k0, "k0"), (trace.k8, "k8")):
            if not isinstance(evaluation, CRNEvaluationEvidence):
                raise TypeError("trace evaluations must be CRNEvaluationEvidence")
            store.verify_object(evaluation.artifact)
            _sha256(evaluation.state_sha256, f"{context} state hash")
            _loss_bits(evaluation.loss_f64_bits, f"{context} loss bits")
        if trace.k0.step != 0 or trace.k8.step != 8:
            raise CRNAuthorityError("isolation trace evaluations must be k0 and k8")
        if (
            trace.k0.evaluation != plan.evaluation
            or trace.k8.evaluation != plan.evaluation
        ):
            raise CRNAuthorityError("isolation trace uses a non-plan evaluation object")
        if trace.k0.state_sha256 != trace.applied_state_sha256:
            raise CRNAuthorityError("k0 state hash differs from applied state hash")
        if trace.k8.state_sha256 != trace.final_state_sha256:
            raise CRNAuthorityError("k8 state hash differs from final state hash")
        if len(trace.groups) != FUTURE_GROUP_COUNT:
            raise CRNAuthorityError("isolation trace must contain exactly eight groups")
        for index, group in enumerate(trace.groups):
            if not isinstance(group, CRNGroupEvidence):
                raise TypeError("trace groups must be CRNGroupEvidence")
            if type(group.group_index) is not int or group.group_index != index:
                raise CRNAuthorityError("trace group indices are noncanonical")
            if group.future_group != plan.future_groups[index]:
                raise CRNAuthorityError("trace future-group object differs from plan")
            _sha256(group.batch_sha256, "trace batch hash")
            _sha256(group.state_sha256, "trace group state hash")
        if trace.groups[-1].state_sha256 != trace.final_state_sha256:
            raise CRNAuthorityError(
                "last future-group state differs from the final arm state"
            )

    restores = {trace.restore_state_sha256 for trace in values}
    if len(restores) != 1:
        raise CRNAuthorityError("four arm restores do not share one exact state hash")
    stock = [trace for trace in values if trace.arm == "stock"]
    candidate = [trace for trace in values if trace.arm == "candidate"]
    if len(stock) != 2 or _normalized_trace(stock[0]) != _normalized_trace(stock[1]):
        raise CRNAuthorityError("stock trace differs across A/B and B/A order")
    if len(candidate) != 2 or _normalized_trace(candidate[0]) != _normalized_trace(
        candidate[1]
    ):
        raise CRNAuthorityError("candidate trace differs across A/B and B/A order")
    return values


def publish_crn_isolation_attestation(
    store: CaptureObjectStore,
    *,
    campaign_index: CRNCampaignIndexRef,
    plan: CRNEvaluationPlanRef,
    traces: Sequence[CRNArmEvidence],
) -> CRNIsolationAttestationRef:
    """Publish all four traces only after recomputing every isolation invariant."""

    loaded_plan = load_authorized_crn_plan(store, campaign_index, plan)
    values = _validate_traces(store, loaded_plan, traces)
    entries: list[ManifestEntry] = []
    for trace_index, trace in enumerate(values):
        entries.extend(
            [
                ManifestEntry(
                    _attestation_artifact_role(trace_index, 0), trace.k0.artifact
                ),
                ManifestEntry(
                    _attestation_artifact_role(trace_index, 8), trace.k8.artifact
                ),
            ]
        )
    metadata = {
        "schema": ATTESTATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "campaign_index": _manifest_descriptor(campaign_index.manifest),
        "plan": _manifest_descriptor(plan.manifest),
        "traces": [
            _trace_value(trace, trace_index) for trace_index, trace in enumerate(values)
        ],
    }
    ref = store.publish_manifest(
        loaded_plan.attestation_manifest_id, entries, metadata=metadata
    )
    return CRNIsolationAttestationRef(ref)


def _parse_evaluation(
    value: Any,
    *,
    trace_index: int,
    expected_step: int,
) -> CRNEvaluationEvidence:
    if not isinstance(value, dict) or set(value) != {
        "step",
        "evaluation",
        "artifact",
        "state_sha256",
        "loss_f64_bits",
    }:
        raise CRNAuthorityError("attestation evaluation row is malformed")
    if type(value["step"]) is not int or value["step"] != expected_step:
        raise CRNAuthorityError("attestation evaluation step is noncanonical")
    evaluation = _parse_object_descriptor(
        value["evaluation"], PLAN_EVALUATION_ROLE, "attestation evaluation input"
    )
    artifact = _parse_object_descriptor(
        value["artifact"],
        _attestation_artifact_role(trace_index, expected_step),
        "attestation evaluation artifact",
    )
    return CRNEvaluationEvidence(
        expected_step,
        evaluation,
        artifact,
        _sha256(value["state_sha256"], "attestation evaluation state hash"),
        _loss_bits(value["loss_f64_bits"], "attestation loss bits"),
    )


def _parse_group(value: Any, expected_index: int) -> CRNGroupEvidence:
    if not isinstance(value, dict) or set(value) != {
        "group_index",
        "future_group",
        "batch_sha256",
        "state_sha256",
    }:
        raise CRNAuthorityError("attestation group row is malformed")
    if type(value["group_index"]) is not int or value["group_index"] != expected_index:
        raise CRNAuthorityError("attestation group index is noncanonical")
    return CRNGroupEvidence(
        expected_index,
        _parse_object_descriptor(
            value["future_group"],
            _plan_group_role(expected_index),
            "attestation future group",
        ),
        _sha256(value["batch_sha256"], "attestation batch hash"),
        _sha256(value["state_sha256"], "attestation group state hash"),
    )


def load_crn_isolation_attestation(
    store: CaptureObjectStore,
    ref: CRNIsolationAttestationRef | ManifestRef | str,
) -> LoadedCRNIsolationAttestation:
    """Reload all trace rows and recompute the cross-order isolation proof."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, CRNIsolationAttestationRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema",
        "schema_version",
        "campaign_index",
        "plan",
        "traces",
    }:
        raise CRNAuthorityError("isolation attestation metadata fields are malformed")
    if metadata["schema"] != ATTESTATION_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise CRNAuthorityError("isolation attestation uses an unsupported schema")
    campaign_ref = CRNCampaignIndexRef(
        _parse_manifest_descriptor(metadata["campaign_index"], "attestation campaign")
    )
    plan_ref = CRNEvaluationPlanRef(
        _parse_manifest_descriptor(metadata["plan"], "attestation plan")
    )
    plan = load_authorized_crn_plan(store, campaign_ref, plan_ref)
    if manifest["manifest_id"] != plan.attestation_manifest_id:
        raise CRNAuthorityError(
            "attestation manifest ID differs from the reserved plan ID"
        )
    rows = metadata["traces"]
    if not isinstance(rows, list) or len(rows) != len(TRACE_SCHEDULE):
        raise CRNAuthorityError("attestation must contain exactly four trace rows")
    traces: list[CRNArmEvidence] = []
    for trace_index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "trace_index",
            "order",
            "position",
            "arm",
            "action",
            "restore_state_sha256",
            "applied_state_sha256",
            "k0",
            "groups",
            "k8",
            "final_state_sha256",
        }:
            raise CRNAuthorityError(f"attestation trace row {trace_index} is malformed")
        if type(row["trace_index"]) is not int or row["trace_index"] != trace_index:
            raise CRNAuthorityError("attestation trace indices are noncanonical")
        groups_value = row["groups"]
        if (
            not isinstance(groups_value, list)
            or len(groups_value) != FUTURE_GROUP_COUNT
        ):
            raise CRNAuthorityError(
                "attestation trace must contain exactly eight groups"
            )
        traces.append(
            CRNArmEvidence(
                order=row["order"],
                position=row["position"],
                arm=row["arm"],
                action=_parse_manifest_descriptor(row["action"], "trace action"),
                restore_state_sha256=_sha256(
                    row["restore_state_sha256"], "trace restore state hash"
                ),
                applied_state_sha256=_sha256(
                    row["applied_state_sha256"], "trace applied state hash"
                ),
                k0=_parse_evaluation(
                    row["k0"], trace_index=trace_index, expected_step=0
                ),
                groups=tuple(
                    _parse_group(group, index)
                    for index, group in enumerate(groups_value)
                ),
                k8=_parse_evaluation(
                    row["k8"], trace_index=trace_index, expected_step=8
                ),
                final_state_sha256=_sha256(
                    row["final_state_sha256"], "trace final state hash"
                ),
            )
        )
    values = _validate_traces(store, plan, traces)
    expected_roles = [
        role
        for trace_index in range(len(TRACE_SCHEDULE))
        for role in (
            _attestation_artifact_role(trace_index, 0),
            _attestation_artifact_role(trace_index, 8),
        )
    ]
    object_rows = manifest["objects"]
    if [row["role"] for row in object_rows] != expected_roles:
        raise CRNAuthorityError("attestation object roles are noncanonical")
    objects = {
        row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in object_rows
    }
    for trace_index, trace in enumerate(values):
        if objects[_attestation_artifact_role(trace_index, 0)] != trace.k0.artifact:
            raise CRNAuthorityError("attestation k0 artifact cross-reference mismatch")
        if objects[_attestation_artifact_role(trace_index, 8)] != trace.k8.artifact:
            raise CRNAuthorityError("attestation k8 artifact cross-reference mismatch")
    manifest_sha = (
        ref.manifest.sha256
        if isinstance(ref, CRNIsolationAttestationRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedCRNIsolationAttestation(
        manifest["manifest_id"],
        manifest_sha,
        campaign_ref,
        plan_ref,
        plan,
        values,
    )


def _canonical_arm_trace(
    attestation: LoadedCRNIsolationAttestation, arm: str
) -> CRNArmEvidence:
    return next(trace for trace in attestation.traces if trace.arm == arm)


def _arm_outcome_value(arm: str, trace: CRNArmEvidence) -> dict[str, Any]:
    return {
        "action": _manifest_descriptor(trace.action),
        "k0": {
            **_object_descriptor(_paired_artifact_role(arm, 0), trace.k0.artifact),
            "loss_f64_bits": trace.k0.loss_f64_bits,
        },
        "k8": {
            **_object_descriptor(_paired_artifact_role(arm, 8), trace.k8.artifact),
            "loss_f64_bits": trace.k8.loss_f64_bits,
        },
    }


def publish_crn_paired_outcome(
    store: CaptureObjectStore,
    *,
    attestation: CRNIsolationAttestationRef,
) -> CRNPairedOutcomeRef:
    """Atomically publish both arms, deriving all values from the attestation."""

    if not isinstance(attestation, CRNIsolationAttestationRef):
        raise TypeError("attestation must be CRNIsolationAttestationRef")
    loaded = load_crn_isolation_attestation(store, attestation)
    plan = loaded.plan
    stock = _canonical_arm_trace(loaded, "stock")
    candidate = _canonical_arm_trace(loaded, "candidate")
    entries = [ManifestEntry(PAIRED_EVALUATION_ROLE, plan.evaluation)]
    for arm, trace in (("stock", stock), ("candidate", candidate)):
        entries.extend(
            [
                ManifestEntry(_paired_artifact_role(arm, 0), trace.k0.artifact),
                ManifestEntry(_paired_artifact_role(arm, 8), trace.k8.artifact),
            ]
        )
    metadata = {
        "schema": PAIRED_OUTCOME_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "campaign_index": _manifest_descriptor(loaded.campaign_index_ref.manifest),
        "plan": _manifest_descriptor(loaded.plan_ref.manifest),
        "attestation": _manifest_descriptor(attestation.manifest),
        "evaluation": _object_descriptor(PAIRED_EVALUATION_ROLE, plan.evaluation),
        "arms": {
            "stock": _arm_outcome_value("stock", stock),
            "candidate": _arm_outcome_value("candidate", candidate),
        },
    }
    ref = store.publish_manifest(
        plan.paired_outcome_manifest_id, entries, metadata=metadata
    )
    return CRNPairedOutcomeRef(ref)


def _parse_arm_outcome(value: Any, arm: str) -> CRNArmOutcome:
    if not isinstance(value, dict) or set(value) != {"action", "k0", "k8"}:
        raise CRNAuthorityError(f"paired {arm} outcome fields are malformed")
    action = _parse_manifest_descriptor(value["action"], f"paired {arm} action")
    parsed: list[tuple[ObjectRef, str]] = []
    for step in HORIZONS:
        row = value[f"k{step}"]
        if not isinstance(row, dict) or set(row) != {
            "role",
            "sha256",
            "bytes",
            "loss_f64_bits",
        }:
            raise CRNAuthorityError(f"paired {arm} k{step} row is malformed")
        artifact = _parse_object_descriptor(
            {key: row[key] for key in ("role", "sha256", "bytes")},
            _paired_artifact_role(arm, step),
            f"paired {arm} k{step} artifact",
        )
        parsed.append((artifact, _loss_bits(row["loss_f64_bits"], "paired loss bits")))
    return CRNArmOutcome(action, parsed[0][0], parsed[0][1], parsed[1][0], parsed[1][1])


def load_crn_paired_outcome(
    store: CaptureObjectStore, ref: CRNPairedOutcomeRef | ManifestRef | str
) -> LoadedCRNPairedOutcome:
    """Accept a paired outcome only when its full durable attestation verifies."""

    manifest_ref: ManifestRef | str = (
        ref.manifest if isinstance(ref, CRNPairedOutcomeRef) else ref
    )
    manifest = store.load_manifest(manifest_ref)
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema",
        "schema_version",
        "campaign_index",
        "plan",
        "attestation",
        "evaluation",
        "arms",
    }:
        raise CRNAuthorityError("paired outcome metadata fields are malformed")
    if metadata["schema"] != PAIRED_OUTCOME_SCHEMA or (
        type(metadata["schema_version"]) is not int
        or metadata["schema_version"] != SCHEMA_VERSION
    ):
        raise CRNAuthorityError("paired outcome uses an unsupported schema")
    campaign_ref = CRNCampaignIndexRef(
        _parse_manifest_descriptor(metadata["campaign_index"], "paired campaign")
    )
    plan_ref = CRNEvaluationPlanRef(
        _parse_manifest_descriptor(metadata["plan"], "paired plan")
    )
    attestation_ref = CRNIsolationAttestationRef(
        _parse_manifest_descriptor(metadata["attestation"], "paired attestation")
    )
    attestation = load_crn_isolation_attestation(store, attestation_ref)
    if not _same_manifest(
        campaign_ref.manifest, attestation.campaign_index_ref.manifest
    ):
        raise CRNAuthorityError("paired outcome campaign differs from attestation")
    if not _same_manifest(plan_ref.manifest, attestation.plan_ref.manifest):
        raise CRNAuthorityError("paired outcome plan differs from attestation")
    plan = load_authorized_crn_plan(store, campaign_ref, plan_ref)
    if manifest["manifest_id"] != plan.paired_outcome_manifest_id:
        raise CRNAuthorityError("paired outcome ID differs from the reserved plan ID")
    evaluation = _parse_object_descriptor(
        metadata["evaluation"], PAIRED_EVALUATION_ROLE, "paired evaluation"
    )
    if evaluation != plan.evaluation:
        raise CRNAuthorityError("paired outcome evaluation differs from plan")
    arms = metadata["arms"]
    if not isinstance(arms, dict) or set(arms) != {"stock", "candidate"}:
        raise CRNAuthorityError("paired outcome arms are malformed")
    stock = _parse_arm_outcome(arms["stock"], "stock")
    candidate = _parse_arm_outcome(arms["candidate"], "candidate")
    expected_stock = _canonical_arm_trace(attestation, "stock")
    expected_candidate = _canonical_arm_trace(attestation, "candidate")
    if arms["stock"] != _arm_outcome_value("stock", expected_stock):
        raise CRNAuthorityError("paired stock outcome differs from attested trace")
    if arms["candidate"] != _arm_outcome_value("candidate", expected_candidate):
        raise CRNAuthorityError("paired candidate outcome differs from attested trace")
    expected_roles = [
        PAIRED_EVALUATION_ROLE,
        _paired_artifact_role("stock", 0),
        _paired_artifact_role("stock", 8),
        _paired_artifact_role("candidate", 0),
        _paired_artifact_role("candidate", 8),
    ]
    object_rows = manifest["objects"]
    if [row["role"] for row in object_rows] != expected_roles:
        raise CRNAuthorityError("paired outcome object roles are noncanonical")
    objects = {
        row["role"]: ObjectRef(row["sha256"], row["bytes"]) for row in object_rows
    }
    expected_objects = {
        PAIRED_EVALUATION_ROLE: evaluation,
        _paired_artifact_role("stock", 0): stock.k0_artifact,
        _paired_artifact_role("stock", 8): stock.k8_artifact,
        _paired_artifact_role("candidate", 0): candidate.k0_artifact,
        _paired_artifact_role("candidate", 8): candidate.k8_artifact,
    }
    if objects != expected_objects:
        raise CRNAuthorityError("paired outcome object cross-reference mismatch")
    manifest_sha = (
        ref.manifest.sha256
        if isinstance(ref, CRNPairedOutcomeRef)
        else ref.sha256
        if isinstance(ref, ManifestRef)
        else ref
    )
    return LoadedCRNPairedOutcome(
        manifest["manifest_id"],
        manifest_sha,
        campaign_ref,
        plan_ref,
        attestation_ref,
        evaluation,
        stock,
        candidate,
    )


__all__ = [
    "ATTESTATION_SCHEMA",
    "CAMPAIGN_INDEX_SCHEMA",
    "CRNArmEvidence",
    "CRNArmOutcome",
    "CRNAuthorityError",
    "CRNCampaignIndexRef",
    "CRNEvaluationEvidence",
    "CRNEvaluationPlanRef",
    "CRNGroupEvidence",
    "CRNIsolationAttestationRef",
    "CRNPairedOutcomeRef",
    "EvaluatorProvenance",
    "LoadedCRNCampaignIndex",
    "LoadedCRNEvaluationPlan",
    "LoadedCRNIsolationAttestation",
    "LoadedCRNPairedOutcome",
    "PAIRED_OUTCOME_SCHEMA",
    "PLAN_SCHEMA",
    "PTI_OUTER_LR_F64_BITS",
    "PTI_ZERO_MOMENTUM_F64_BITS",
    "load_authorized_crn_plan",
    "load_crn_campaign_index",
    "load_crn_evaluation_plan",
    "load_crn_isolation_attestation",
    "load_crn_paired_outcome",
    "publish_crn_campaign_index",
    "publish_crn_evaluation_plan",
    "publish_crn_isolation_attestation",
    "publish_crn_paired_outcome",
]
