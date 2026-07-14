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

from .capture_v2_policy import (
    LoadedSealedOuterAction,
    PolicyOutcomeRef,
    SealedOuterActionRef,
    load_sealed_outer_action,
    publish_policy_outcome,
)
from .capture_v2_store import (
    CaptureObjectStore,
    CaptureStoreError,
    ManifestRef,
    ObjectRef,
)
from .capture_v2_syncer import SyncerBoundaryRef, load_syncer_boundary


REQUIRED_CAPABILITIES = frozenset({"worker_restore", "crn_train_k8"})
FUTURE_GROUP_COUNT = 8

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_F64_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")
_MANIFEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


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
    """Published outcomes and the two verified canonical arm traces."""

    stock_outcome: PolicyOutcomeRef
    candidate_outcome: PolicyOutcomeRef
    stock_trace: ArmTrace
    candidate_trace: ArmTrace


@dataclass(frozen=True)
class _Inputs:
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


def _validate_manifest_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or _MANIFEST_ID_RE.fullmatch(value) is None:
        raise CRNEvaluationError(f"{context} is not a canonical manifest_id")
    return value


def _same_manifest(left: ManifestRef, right: ManifestRef) -> bool:
    return (
        left.manifest_id == right.manifest_id
        and left.sha256 == right.sha256
        and left.bytes == right.bytes
    )


def _require_capabilities(action: LoadedSealedOuterAction, context: str) -> None:
    required = set(action.required_capabilities)
    declared = set(action.policy.capabilities)
    if not REQUIRED_CAPABILITIES <= required or not REQUIRED_CAPABILITIES <= declared:
        raise CRNEvaluationError(
            f"{context} lacks complete worker_restore+crn_train_k8 capabilities"
        )


def _load_inputs(
    store: CaptureObjectStore,
    *,
    boundary_ref: SyncerBoundaryRef,
    stock_ref: SealedOuterActionRef,
    candidate_ref: SealedOuterActionRef,
    responder_index: int,
    evaluation_ref: ObjectRef,
) -> _Inputs:
    if not isinstance(boundary_ref, SyncerBoundaryRef):
        raise TypeError("boundary must be a SyncerBoundaryRef")
    if not isinstance(stock_ref, SealedOuterActionRef):
        raise TypeError("stock_action must be a SealedOuterActionRef")
    if not isinstance(candidate_ref, SealedOuterActionRef):
        raise TypeError("candidate_action must be a SealedOuterActionRef")
    if type(responder_index) is not int or responder_index < 0:
        raise CRNEvaluationError("responder_index must be a non-negative integer")

    boundary = load_syncer_boundary(store, boundary_ref)
    stock = load_sealed_outer_action(store, stock_ref)
    candidate = load_sealed_outer_action(store, candidate_ref)
    for action, context in ((stock, "stock action"), (candidate, "candidate action")):
        if not _same_manifest(action.boundary_ref.manifest, boundary_ref.manifest):
            raise CRNEvaluationError(f"{context} is cross-wired to another boundary")
        _require_capabilities(action, context)
    if stock.action_kind != "stock_fallback":
        raise CRNEvaluationError("stock action must be a sealed stock_fallback")
    if candidate.action_kind != "nonstock":
        raise CRNEvaluationError("candidate action must be a sealed nonstock action")
    if stock.stock_pseudo_gradient != candidate.stock_pseudo_gradient:
        raise CRNEvaluationError(
            "stock and candidate actions use different stock objects"
        )
    if responder_index >= len(boundary.responders):
        raise CRNEvaluationError("responder_index is absent from the boundary")

    responder = boundary.responders[responder_index]
    captured = responder.endpoint.future_groups
    expected_indices = list(range(FUTURE_GROUP_COUNT))
    if captured.state != "complete" or list(captured.refs) != expected_indices:
        raise CRNEvaluationError(
            "selected worker restore lacks the complete ordered 8 future groups"
        )
    future_groups = tuple(
        _verified_artifact(store, captured.refs[index], f"future group {index}")
        for index in expected_indices
    )
    return _Inputs(
        boundary=_manifest_identity(boundary_ref.manifest),
        endpoint=_manifest_identity(responder.endpoint_ref.manifest),
        stock=stock,
        candidate=candidate,
        stock_identity=_manifest_identity(stock_ref.manifest),
        candidate_identity=_manifest_identity(candidate_ref.manifest),
        stock_result=_verified_artifact(
            store, stock.resulting_fragment, "stock resulting fragment"
        ),
        candidate_result=_verified_artifact(
            store, candidate.resulting_fragment, "candidate resulting fragment"
        ),
        evaluation=_verified_artifact(store, evaluation_ref, "fixed evaluation object"),
        future_groups=future_groups,
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


def evaluate_isolated_crn_pair(
    store: CaptureObjectStore,
    *,
    boundary: SyncerBoundaryRef,
    stock_action: SealedOuterActionRef,
    candidate_action: SealedOuterActionRef,
    responder_index: int,
    evaluation: ObjectRef,
    backend: IsolatedCRNBackend,
    stock_outcome_manifest_id: str,
    candidate_outcome_manifest_id: str,
) -> CRNEvaluationResult:
    """Evaluate sealed stock/candidate actions in both orders, then publish.

    No callback runs until the boundary, both actions, the selected worker
    restore, the fixed evaluation object, and all eight future groups have been
    verified.  The fixed evaluation object's k8 loss is also recorded as the
    outcome's final ``evaluation_loss``.
    """

    stock_outcome_manifest_id = _validate_manifest_id(
        stock_outcome_manifest_id, "stock outcome manifest id"
    )
    candidate_outcome_manifest_id = _validate_manifest_id(
        candidate_outcome_manifest_id, "candidate outcome manifest id"
    )
    if stock_outcome_manifest_id == candidate_outcome_manifest_id:
        raise CRNEvaluationError("stock and candidate outcome manifest ids must differ")

    inputs = _load_inputs(
        store,
        boundary_ref=boundary,
        stock_ref=stock_action,
        candidate_ref=candidate_action,
        responder_index=responder_index,
        evaluation_ref=evaluation,
    )
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
        boundary_ref=boundary,
        stock_ref=stock_action,
        candidate_ref=candidate_action,
        responder_index=responder_index,
        evaluation_ref=evaluation,
    )
    for trace in (stock_trace, candidate_trace):
        store.verify_object(trace.k0.artifact)
        store.verify_object(trace.k8.artifact)

    _, stock_k0_loss = _loss_bits(stock_trace.k0.loss_f64_bits, "stock k0 loss bits")
    _, stock_k8_loss = _loss_bits(stock_trace.k8.loss_f64_bits, "stock k8 loss bits")
    _, candidate_k0_loss = _loss_bits(
        candidate_trace.k0.loss_f64_bits, "candidate k0 loss bits"
    )
    _, candidate_k8_loss = _loss_bits(
        candidate_trace.k8.loss_f64_bits, "candidate k8 loss bits"
    )

    stock_outcome = publish_policy_outcome(
        store,
        stock_outcome_manifest_id,
        action=stock_action,
        k0=stock_trace.k0.artifact,
        k0_loss=stock_k0_loss,
        k8=stock_trace.k8.artifact,
        k8_loss=stock_k8_loss,
        evaluation=inputs.evaluation.ref,
        evaluation_loss=stock_k8_loss,
    )
    candidate_outcome = publish_policy_outcome(
        store,
        candidate_outcome_manifest_id,
        action=candidate_action,
        k0=candidate_trace.k0.artifact,
        k0_loss=candidate_k0_loss,
        k8=candidate_trace.k8.artifact,
        k8_loss=candidate_k8_loss,
        evaluation=inputs.evaluation.ref,
        evaluation_loss=candidate_k8_loss,
    )
    return CRNEvaluationResult(
        stock_outcome=stock_outcome,
        candidate_outcome=candidate_outcome,
        stock_trace=stock_trace,
        candidate_trace=candidate_trace,
    )
