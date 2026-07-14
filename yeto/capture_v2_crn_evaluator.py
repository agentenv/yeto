"""Policy-agnostic isolated CRN evaluation over sealed capture-v2 actions.

The evaluator is orchestration and validation only.  A backend callback owns
restoration, action application, training, and evaluation.  This module does
not implement an optimizer, mutate live training state, or depend on PyTorch.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Protocol

from .capture_v2_crn_plan import (
    CRNArmEvidence,
    CRNCampaignIndexRef,
    CRNEvaluationEvidence,
    CRNEvaluationPlanRef,
    CRNGroupEvidence,
    CRNIsolationAttestationRef,
    CRNPairedOutcomeRef,
    EvaluatorProvenance,
    load_authorized_crn_plan,
    publish_crn_isolation_attestation,
    publish_crn_paired_outcome,
)
from .capture_v2_policy import (
    LoadedSealedOuterAction,
)
from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestRef,
    ObjectRef,
)


FUTURE_GROUP_COUNT = 8

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_F64_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")


class CRNEvaluationError(CaptureStoreError):
    """The CRN evaluation is incomplete, cross-wired, or nondeterministic."""


@dataclass(frozen=True)
class ManifestIdentity:
    """Immutable manifest identity without publication-local insertion state."""

    manifest_id: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class VerifiedArtifact:
    """Exact verified CAS object bytes passed immutably to a backend callback."""

    ref: ObjectRef
    data: bytes


@dataclass(frozen=True)
class RestoreRequest:
    """Request one fresh worker restore for one position in an arm order."""

    order: str
    position: int
    boundary: ManifestIdentity
    endpoint: ManifestIdentity


@dataclass(frozen=True)
class RestoreReceipt:
    """Backend-owned fresh branch plus the identities it restored."""

    branch: object
    boundary: ManifestIdentity
    endpoint: ManifestIdentity
    state_sha256: str


@dataclass(frozen=True)
class ApplyActionRequest:
    """Exact sealed action material to apply to a freshly restored branch."""

    action: ManifestIdentity
    selected_pseudo_gradient: ObjectRef
    resulting_fragment: VerifiedArtifact


@dataclass(frozen=True)
class ApplyActionReceipt:
    """Backend acknowledgement of the exact action and resulting state."""

    action: ManifestIdentity
    selected_pseudo_gradient: ObjectRef
    resulting_fragment: ObjectRef
    state_sha256: str


@dataclass(frozen=True)
class EvaluationRequest:
    """Read-only fixed-object evaluation request at k0 or k8."""

    action: ManifestIdentity
    step: int
    evaluation: VerifiedArtifact


@dataclass(frozen=True)
class EvaluationReceipt:
    """Exact evaluation artifact, state identity, and IEEE-754 loss bits."""

    action: ManifestIdentity
    step: int
    evaluation: ObjectRef
    artifact: ObjectRef
    state_sha256: str
    loss_f64_bits: str


@dataclass(frozen=True)
class TrainGroupRequest:
    """One exact future group in the fixed ordered k8 CRN sequence."""

    action: ManifestIdentity
    group_index: int
    future_group: VerifiedArtifact


@dataclass(frozen=True)
class TrainGroupReceipt:
    """Backend acknowledgement of one consumed group and resulting state."""

    action: ManifestIdentity
    group_index: int
    future_group: ObjectRef
    batch_sha256: str
    state_sha256: str


class IsolatedCRNBackend(Protocol):
    """Policy-agnostic callbacks required by the isolated evaluator."""

    def provenance(self) -> EvaluatorProvenance:
        """Return the exact sealed evaluator code, image, and config identity."""

    def restore(self, request: RestoreRequest) -> RestoreReceipt:
        """Create an independent fresh branch from the verified endpoint."""

    def state_sha256(self, branch: object) -> str:
        """Return a stable digest of all branch state that may affect results."""

    def apply_action(
        self, branch: object, request: ApplyActionRequest
    ) -> ApplyActionReceipt:
        """Apply exactly the sealed action material to the branch."""

    def evaluate(self, branch: object, request: EvaluationRequest) -> EvaluationReceipt:
        """Evaluate the fixed object without mutating the branch."""

    def train_group(
        self, branch: object, request: TrainGroupRequest
    ) -> TrainGroupReceipt:
        """Consume exactly one ordered captured future group."""


@dataclass(frozen=True)
class ArmTrace:
    """Order-independent evidence retained for one sealed action arm."""

    action: ManifestIdentity
    restore_state_sha256: str
    applied_state_sha256: str
    k0: EvaluationReceipt
    groups: tuple[TrainGroupReceipt, ...]
    k8: EvaluationReceipt
    final_state_sha256: str


@dataclass(frozen=True)
class CRNEvaluationResult:
    """Atomic paired result plus its complete durable isolation proof."""

    campaign_index: CRNCampaignIndexRef
    plan: CRNEvaluationPlanRef
    attestation: CRNIsolationAttestationRef
    paired_outcome: CRNPairedOutcomeRef
    stock_trace: ArmTrace
    candidate_trace: ArmTrace


@dataclass(frozen=True)
class _Inputs:
    campaign_index_ref: CRNCampaignIndexRef
    plan_ref: CRNEvaluationPlanRef
    boundary: ManifestIdentity
    endpoint: ManifestIdentity
    stock: LoadedSealedOuterAction
    candidate: LoadedSealedOuterAction
    stock_identity: ManifestIdentity
    candidate_identity: ManifestIdentity
    stock_result: VerifiedArtifact
    candidate_result: VerifiedArtifact
    evaluation: VerifiedArtifact
    future_groups: tuple[VerifiedArtifact, ...]
    evaluator: EvaluatorProvenance


def _manifest_identity(ref: ManifestRef) -> ManifestIdentity:
    return ManifestIdentity(ref.manifest_id, ref.sha256, ref.bytes)


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CRNEvaluationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _loss_bits(value: Any, context: str) -> tuple[str, float]:
    if not isinstance(value, str) or _F64_BITS_RE.fullmatch(value) is None:
        raise CRNEvaluationError(
            f"{context} must be exactly 16 lowercase f64 hex digits"
        )
    loss = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(loss):
        raise CRNEvaluationError(f"{context} must decode to a finite f64")
    return value, loss


def _verified_artifact(
    store: CaptureObjectStore, ref: ObjectRef, context: str
) -> VerifiedArtifact:
    if not isinstance(ref, ObjectRef):
        raise TypeError(f"{context} must be an ObjectRef")
    path = store.verify_object(ref)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CRNEvaluationError(f"cannot read {context}: {exc}") from exc
    if len(data) != ref.bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
        raise CRNEvaluationError(f"{context} changed while it was being read")
    return VerifiedArtifact(ref, data)


def _load_inputs(
    store: CaptureObjectStore,
    *,
    campaign_index_ref: CRNCampaignIndexRef,
    plan_ref: CRNEvaluationPlanRef,
) -> _Inputs:
    plan = load_authorized_crn_plan(store, campaign_index_ref, plan_ref)
    responder = plan.responder
    future_groups = tuple(
        _verified_artifact(store, ref, f"future group {index}")
        for index, ref in enumerate(plan.future_groups)
    )
    return _Inputs(
        campaign_index_ref=campaign_index_ref,
        plan_ref=plan_ref,
        boundary=_manifest_identity(plan.boundary_ref.manifest),
        endpoint=_manifest_identity(responder.endpoint_ref.manifest),
        stock=plan.stock,
        candidate=plan.candidate,
        stock_identity=_manifest_identity(plan.stock_ref.manifest),
        candidate_identity=_manifest_identity(plan.candidate_ref.manifest),
        stock_result=_verified_artifact(
            store, plan.stock.resulting_fragment, "stock resulting fragment"
        ),
        candidate_result=_verified_artifact(
            store, plan.candidate.resulting_fragment, "candidate resulting fragment"
        ),
        evaluation=_verified_artifact(
            store, plan.evaluation, "fixed evaluation object"
        ),
        future_groups=future_groups,
        evaluator=plan.evaluator,
    )


def _backend_call(context: str, callback: Any, *args: Any) -> Any:
    try:
        return callback(*args)
    except Exception as exc:
        raise CRNEvaluationError(f"backend {context} failed: {exc}") from exc


def _stable_state_hash(
    backend: IsolatedCRNBackend, branch: object, context: str
) -> str:
    first = _sha256(
        _backend_call(f"{context} state hash", backend.state_sha256, branch),
        f"{context} state hash",
    )
    second = _sha256(
        _backend_call(f"{context} repeated state hash", backend.state_sha256, branch),
        f"{context} repeated state hash",
    )
    if first != second:
        raise CRNEvaluationError(
            f"backend state hash is nondeterministic or mutating at {context}"
        )
    return first


def _validate_restore(
    value: Any,
    request: RestoreRequest,
    backend: IsolatedCRNBackend,
    seen_branch_objects: set[int],
) -> tuple[object, str]:
    if not isinstance(value, RestoreReceipt):
        raise CRNEvaluationError("backend restore returned the wrong receipt type")
    if value.boundary != request.boundary or value.endpoint != request.endpoint:
        raise CRNEvaluationError("backend restore receipt is cross-wired")
    branch_identity = id(value.branch)
    if branch_identity in seen_branch_objects:
        raise CRNEvaluationError("backend reused a branch instead of a fresh restore")
    seen_branch_objects.add(branch_identity)
    state = _stable_state_hash(backend, value.branch, "fresh restore")
    if _sha256(value.state_sha256, "restore receipt state hash") != state:
        raise CRNEvaluationError("backend restore receipt state hash mismatch")
    return value.branch, state


def _validate_apply(
    value: Any,
    request: ApplyActionRequest,
    backend: IsolatedCRNBackend,
    branch: object,
) -> str:
    if not isinstance(value, ApplyActionReceipt):
        raise CRNEvaluationError(
            "backend action application returned the wrong receipt type"
        )
    if (
        value.action != request.action
        or value.selected_pseudo_gradient != request.selected_pseudo_gradient
        or value.resulting_fragment != request.resulting_fragment.ref
    ):
        raise CRNEvaluationError("backend action application receipt is cross-wired")
    state = _stable_state_hash(backend, branch, "action application")
    if _sha256(value.state_sha256, "action receipt state hash") != state:
        raise CRNEvaluationError("backend action application state hash mismatch")
    return state


def _validate_evaluation(
    store: CaptureObjectStore,
    value: Any,
    request: EvaluationRequest,
    expected_state: str,
    backend: IsolatedCRNBackend,
    branch: object,
) -> EvaluationReceipt:
    if not isinstance(value, EvaluationReceipt):
        raise CRNEvaluationError("backend evaluation returned the wrong receipt type")
    if (
        value.action != request.action
        or type(value.step) is not int
        or value.step != request.step
        or value.evaluation != request.evaluation.ref
    ):
        raise CRNEvaluationError("backend evaluation receipt is cross-wired")
    if not isinstance(value.artifact, ObjectRef):
        raise CRNEvaluationError("backend evaluation artifact must be an ObjectRef")
    store.verify_object(value.artifact)
    _loss_bits(value.loss_f64_bits, f"k{request.step} loss bits")
    after = _stable_state_hash(backend, branch, f"k{request.step} evaluation")
    if after != expected_state:
        raise CRNEvaluationError(
            f"backend evaluation mutated branch state at k{request.step}"
        )
    if _sha256(value.state_sha256, "evaluation receipt state hash") != after:
        raise CRNEvaluationError("backend evaluation receipt state hash mismatch")
    return value


def _validate_train_group(
    value: Any,
    request: TrainGroupRequest,
    backend: IsolatedCRNBackend,
    branch: object,
) -> TrainGroupReceipt:
    if not isinstance(value, TrainGroupReceipt):
        raise CRNEvaluationError("backend train_group returned the wrong receipt type")
    if (
        value.action != request.action
        or type(value.group_index) is not int
        or value.group_index != request.group_index
        or value.future_group != request.future_group.ref
    ):
        raise CRNEvaluationError(
            "backend future-group receipt is missing or cross-wired"
        )
    state = _stable_state_hash(backend, branch, f"future group {request.group_index}")
    _sha256(value.batch_sha256, "future-group batch hash")
    if _sha256(value.state_sha256, "future-group state hash") != state:
        raise CRNEvaluationError("backend future-group state hash mismatch")
    return value


def _run_arm(
    store: CaptureObjectStore,
    backend: IsolatedCRNBackend,
    *,
    restore_request: RestoreRequest,
    action: LoadedSealedOuterAction,
    action_identity: ManifestIdentity,
    result: VerifiedArtifact,
    evaluation: VerifiedArtifact,
    future_groups: tuple[VerifiedArtifact, ...],
    seen_branch_objects: set[int],
) -> tuple[ArmTrace, object]:
    restore = _backend_call("restore", backend.restore, restore_request)
    branch, restore_state = _validate_restore(
        restore, restore_request, backend, seen_branch_objects
    )

    apply_request = ApplyActionRequest(
        action_identity, action.selected_pseudo_gradient, result
    )
    apply_receipt = _backend_call(
        "action application", backend.apply_action, branch, apply_request
    )
    applied_state = _validate_apply(apply_receipt, apply_request, backend, branch)

    k0_request = EvaluationRequest(action_identity, 0, evaluation)
    k0_receipt = _backend_call("k0 evaluation", backend.evaluate, branch, k0_request)
    k0 = _validate_evaluation(
        store, k0_receipt, k0_request, applied_state, backend, branch
    )

    groups: list[TrainGroupReceipt] = []
    for group_index, future_group in enumerate(future_groups):
        request = TrainGroupRequest(action_identity, group_index, future_group)
        receipt = _backend_call(
            f"future group {group_index}", backend.train_group, branch, request
        )
        groups.append(_validate_train_group(receipt, request, backend, branch))

    if len(groups) != FUTURE_GROUP_COUNT:
        raise CRNEvaluationError("backend did not consume exactly 8 future groups")
    trained_state = _stable_state_hash(backend, branch, "post-k8 training")
    k8_request = EvaluationRequest(action_identity, 8, evaluation)
    k8_receipt = _backend_call("k8 evaluation", backend.evaluate, branch, k8_request)
    k8 = _validate_evaluation(
        store, k8_receipt, k8_request, trained_state, backend, branch
    )
    final_state = _stable_state_hash(backend, branch, "completed arm")
    return (
        ArmTrace(
            action=action_identity,
            restore_state_sha256=restore_state,
            applied_state_sha256=applied_state,
            k0=k0,
            groups=tuple(groups),
            k8=k8,
            final_state_sha256=final_state,
        ),
        branch,
    )


def _assert_same_trace(first: ArmTrace, second: ArmTrace, context: str) -> None:
    if first != second:
        raise CRNEvaluationError(
            f"backend nondeterminism or branch leakage changed the {context} trace "
            "across A/B and B/A order"
        )


def _manifest_ref(identity: ManifestIdentity) -> ManifestRef:
    return ManifestRef(
        identity.manifest_id,
        identity.sha256,
        identity.bytes,
        False,
    )


def _attestation_trace(
    *, order: str, position: int, arm: str, trace: ArmTrace
) -> CRNArmEvidence:
    """Convert a validated in-memory trace to the durable authority schema."""

    return CRNArmEvidence(
        order=order,
        position=position,
        arm=arm,
        action=_manifest_ref(trace.action),
        restore_state_sha256=trace.restore_state_sha256,
        applied_state_sha256=trace.applied_state_sha256,
        k0=CRNEvaluationEvidence(
            step=trace.k0.step,
            evaluation=trace.k0.evaluation,
            artifact=trace.k0.artifact,
            state_sha256=trace.k0.state_sha256,
            loss_f64_bits=trace.k0.loss_f64_bits,
        ),
        groups=tuple(
            CRNGroupEvidence(
                group_index=group.group_index,
                future_group=group.future_group,
                batch_sha256=group.batch_sha256,
                state_sha256=group.state_sha256,
            )
            for group in trace.groups
        ),
        k8=CRNEvaluationEvidence(
            step=trace.k8.step,
            evaluation=trace.k8.evaluation,
            artifact=trace.k8.artifact,
            state_sha256=trace.k8.state_sha256,
            loss_f64_bits=trace.k8.loss_f64_bits,
        ),
        final_state_sha256=trace.final_state_sha256,
    )


def evaluate_isolated_crn_pair(
    store: CaptureObjectStore,
    *,
    campaign_index: CRNCampaignIndexRef,
    plan: CRNEvaluationPlanRef,
    backend: IsolatedCRNBackend,
) -> CRNEvaluationResult:
    """Evaluate one pre-authorized plan in both orders, then publish atomically.

    No callback runs until the campaign membership and the plan's boundary,
    actions, responder endpoint, fixed evaluation object, future groups,
    schedule, horizons, evaluator provenance, CUDA count, and reserved output
    identities have all been verified.
    """

    inputs = _load_inputs(
        store,
        campaign_index_ref=campaign_index,
        plan_ref=plan,
    )
    provenance_callback = getattr(backend, "provenance", None)
    if not callable(provenance_callback):
        raise CRNEvaluationError("backend lacks sealed evaluator provenance")
    actual_provenance = _backend_call("provenance", provenance_callback)
    if not isinstance(actual_provenance, EvaluatorProvenance):
        raise CRNEvaluationError("backend provenance returned the wrong type")
    if actual_provenance != inputs.evaluator:
        raise CRNEvaluationError("backend provenance differs from the sealed plan")
    seen_branch_objects: set[int] = set()
    completed: list[tuple[object, str]] = []
    traces: dict[tuple[str, str], ArmTrace] = {}
    arm_values = {
        "stock": (inputs.stock, inputs.stock_identity, inputs.stock_result),
        "candidate": (
            inputs.candidate,
            inputs.candidate_identity,
            inputs.candidate_result,
        ),
    }
    orders = (
        ("stock-candidate", ("stock", "candidate")),
        ("candidate-stock", ("candidate", "stock")),
    )
    restore_baseline: str | None = None
    for order, arms in orders:
        for position, arm in enumerate(arms):
            action, action_identity, result = arm_values[arm]
            trace, branch = _run_arm(
                store,
                backend,
                restore_request=RestoreRequest(
                    order, position, inputs.boundary, inputs.endpoint
                ),
                action=action,
                action_identity=action_identity,
                result=result,
                evaluation=inputs.evaluation,
                future_groups=inputs.future_groups,
                seen_branch_objects=seen_branch_objects,
            )
            if restore_baseline is None:
                restore_baseline = trace.restore_state_sha256
            elif trace.restore_state_sha256 != restore_baseline:
                raise CRNEvaluationError(
                    "fresh worker restores do not have identical state hashes"
                )
            traces[(order, arm)] = trace
            completed.append((branch, trace.final_state_sha256))

    stock_trace = traces[("stock-candidate", "stock")]
    candidate_trace = traces[("stock-candidate", "candidate")]
    _assert_same_trace(stock_trace, traces[("candidate-stock", "stock")], "stock arm")
    _assert_same_trace(
        candidate_trace,
        traces[("candidate-stock", "candidate")],
        "candidate arm",
    )

    for branch, expected_state in completed:
        if (
            _stable_state_hash(backend, branch, "final isolation audit")
            != expected_state
        ):
            raise CRNEvaluationError(
                "a completed branch was mutated through shared or leaked backend state"
            )

    # Callbacks cannot alter any sealed input without this second verification
    # failing before an outcome manifest is published.
    _load_inputs(
        store,
        campaign_index_ref=campaign_index,
        plan_ref=plan,
    )
    for trace in traces.values():
        store.verify_object(trace.k0.artifact)
        store.verify_object(trace.k8.artifact)

    durable_traces = (
        _attestation_trace(
            order="stock-candidate",
            position=0,
            arm="stock",
            trace=traces[("stock-candidate", "stock")],
        ),
        _attestation_trace(
            order="stock-candidate",
            position=1,
            arm="candidate",
            trace=traces[("stock-candidate", "candidate")],
        ),
        _attestation_trace(
            order="candidate-stock",
            position=0,
            arm="candidate",
            trace=traces[("candidate-stock", "candidate")],
        ),
        _attestation_trace(
            order="candidate-stock",
            position=1,
            arm="stock",
            trace=traces[("candidate-stock", "stock")],
        ),
    )
    attestation = publish_crn_isolation_attestation(
        store,
        campaign_index=inputs.campaign_index_ref,
        plan=inputs.plan_ref,
        traces=durable_traces,
    )
    paired_outcome = publish_crn_paired_outcome(store, attestation=attestation)
    return CRNEvaluationResult(
        campaign_index=inputs.campaign_index_ref,
        plan=inputs.plan_ref,
        attestation=attestation,
        paired_outcome=paired_outcome,
        stock_trace=stock_trace,
        candidate_trace=candidate_trace,
    )
