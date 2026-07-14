from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, replace

import pytest
import torch

from yeto.capture_v2_crn_evaluator import (
    ApplyActionReceipt,
    ApplyActionRequest,
    CRNEvaluationError,
    EvaluationReceipt,
    EvaluationRequest,
    ManifestIdentity,
    RestoreReceipt,
    RestoreRequest,
    TrainGroupReceipt,
    TrainGroupRequest,
    evaluate_isolated_crn_pair,
)
from yeto.capture_v2_endpoint import (
    EndpointIdentity,
    FutureGroupRefs,
    InputProvenance,
    publish_future_group_envelope,
    publish_learner_endpoint,
)
from yeto.capture_v2_policy import (
    CAPABILITIES,
    SealedOuterActionRef,
    load_policy_outcome,
    publish_policy_definition,
    publish_sealed_outer_action,
)
from yeto.capture_v2_store import CaptureObjectStore, CaptureStoreError, ObjectRef
from yeto.capture_v2_syncer import (
    BoundaryConfig,
    ResponderEndpointRef,
    SyncerBoundaryIdentity,
    SyncerBoundaryRef,
    publish_syncer_boundary,
)
from yeto.capture_v2_tensor_pack import publish_tensor_pack


SESSION = "12345678-1234-5678-9234-567812345678"
COMPLETE_CAPABILITIES = ("worker_restore", "crn_train_k8")


def _digest(*values: str | bytes) -> str:
    digest = hashlib.sha256()
    for value in values:
        raw = value.encode() if isinstance(value, str) else value
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _object(store: CaptureObjectStore, raw: bytes) -> ObjectRef:
    return store.put_bytes(raw).ref


@dataclass(frozen=True)
class _Fixture:
    store: CaptureObjectStore
    boundary: SyncerBoundaryRef
    stock_action: SealedOuterActionRef
    candidate_action: SealedOuterActionRef
    evaluation: ObjectRef
    future_groups: tuple[ObjectRef, ...]
    stock_result: ObjectRef
    candidate_result: ObjectRef


def _fixture(
    tmp_path,
    *,
    store: CaptureObjectStore | None = None,
    suffix: str = "one",
    future_state: str = "complete",
    stock_required=COMPLETE_CAPABILITIES,
    candidate_required=COMPLETE_CAPABILITIES,
) -> _Fixture:
    store = store or CaptureObjectStore(tmp_path / "cas")
    future_count = 8 if future_state == "complete" else 7
    numeric_suffix = sum(suffix.encode())
    identity = EndpointIdentity(
        capture_session_uuid=SESSION,
        learner_id=0,
        rank=0,
        local_step=100,
        active_fragment_id=0,
        window_uuid=f"00000000-0000-4000-8000-{numeric_suffix:012d}",
    )
    future_groups = tuple(
        publish_future_group_envelope(
            store,
            capture_session_uuid=identity.capture_session_uuid,
            window_uuid=identity.window_uuid,
            learner_id=identity.learner_id,
            rank=identity.rank,
            group_index=index,
            group_id=f"{suffix}-batch-{index}",
            data_iterator_position=1000 + index,
            content=f"{suffix} future group {index}".encode(),
        )
        for index in range(future_count)
    )
    pack = publish_tensor_pack(
        store,
        f"{suffix}-fragment-pack",
        fragment_id=0,
        trainable={"model.weight": torch.tensor([1.0, -2.0])},
        optimizer={"model.weight/exp_avg": torch.tensor([0.25, -0.5])},
        clocks={"optimizer_steps": 7},
        metadata={"fixture": suffix},
    )
    endpoint = publish_learner_endpoint(
        store,
        f"{suffix}-endpoint",
        identity=identity,
        input_provenance=InputProvenance(
            object=_object(store, f"{suffix} provenance".encode()),
            source_commit="a" * 40,
            image_id="capture-v2-fake-backend",
            model_sha256="b" * 64,
            data_sha256="c" * 64,
            config_sha256="d" * 64,
        ),
        fragment_packs={0: pack},
        fragment_versions=[7],
        mode="train",
        model_buffers=_object(store, f"{suffix} model buffers".encode()),
        scheduler={"last_epoch": 7},
        scaler=None,
        python_rng=_object(store, f"{suffix} python rng".encode()),
        numpy_rng=_object(store, f"{suffix} numpy rng".encode()),
        torch_cpu_rng=_object(store, f"{suffix} torch cpu rng".encode()),
        torch_cuda_rng={0: _object(store, f"{suffix} torch cuda rng 0".encode())},
        future_groups=FutureGroupRefs(
            future_state,
            {index: ref for index, ref in enumerate(future_groups)},
            None if future_state == "complete" else "future group 7 is absent",
        ),
    )
    post_fragment = _object(store, f"{suffix} post fragment".encode())
    boundary = publish_syncer_boundary(
        store,
        f"{suffix}-boundary",
        identity=SyncerBoundaryIdentity(
            capture_session_uuid=SESSION,
            commit_id=f"{suffix}-commit-0008",
            commit_seq=8,
            fragment_id=0,
            pre_fragment_version=7,
            post_fragment_version=8,
        ),
        responders=[
            ResponderEndpointRef(
                endpoint=endpoint,
                weight_f64_bits=struct.pack(">d", 128.0).hex(),
                payload=_object(store, f"{suffix} responder payload".encode()),
            )
        ],
        pre_fragment=_object(store, f"{suffix} pre fragment".encode()),
        post_fragment=post_fragment,
        outer_state=_object(store, f"{suffix} outer state".encode()),
        broadcast=_object(store, f"{suffix} broadcast".encode()),
        merge_config=BoundaryConfig("rda", {"weighted": True}),
        outer_config=BoundaryConfig(
            "stock-sgd",
            {"lr_f64_bits": struct.pack(">d", 0.28).hex()},
        ),
    )
    source = _object(store, f"{suffix} policy source".encode())
    config = _object(store, f"{suffix} policy config".encode())
    policy = publish_policy_definition(
        store,
        f"{suffix}-policy",
        policy_id=f"candidate-{suffix}",
        policy_version="1.0.0",
        source_commit="e" * 40,
        source=source,
        config=config,
        capabilities=CAPABILITIES,
    )
    stock = _object(store, f"{suffix} exact stock pseudo-gradient".encode())
    candidate = _object(store, f"{suffix} candidate pseudo-gradient".encode())
    stock_result = post_fragment
    candidate_result = _object(store, f"{suffix} candidate resulting fragment".encode())
    common = {
        "policy": policy,
        "boundary": boundary,
        "fragment_id": 0,
        "outer_lr_f64_bits": struct.pack(">d", 0.28).hex(),
        "decision": _object(store, f"{suffix} decision bytes".encode()),
        "config_sha256": config.sha256,
    }
    stock_action = publish_sealed_outer_action(
        store,
        f"{suffix}-stock-action",
        **common,
        required_capabilities=stock_required,
        stock_pseudo_gradient=stock,
        selected_pseudo_gradient=stock,
        resulting_fragment=stock_result,
        action_kind="stock_fallback",
        action_reason="seal the stock comparison arm",
        fallback_reason="intentional stock CRN control arm",
    )
    candidate_action = publish_sealed_outer_action(
        store,
        f"{suffix}-candidate-action",
        **common,
        required_capabilities=candidate_required,
        stock_pseudo_gradient=stock,
        selected_pseudo_gradient=candidate,
        resulting_fragment=candidate_result,
        action_kind="nonstock",
        action_reason="seal the candidate comparison arm",
        fallback_reason=None,
    )
    return _Fixture(
        store=store,
        boundary=boundary,
        stock_action=stock_action,
        candidate_action=candidate_action,
        evaluation=_object(store, f"{suffix} fixed evaluation object".encode()),
        future_groups=future_groups,
        stock_result=stock_result,
        candidate_result=candidate_result,
    )


class _Branch:
    def __init__(self, state_box: list[str], order: str, position: int):
        self.state_box = state_box
        self.order = order
        self.position = position
        self.action: ManifestIdentity | None = None

    @property
    def state(self) -> str:
        return self.state_box[0]

    @state.setter
    def state(self, value: str) -> None:
        self.state_box[0] = value


class _FakeBackend:
    def __init__(
        self,
        fixture: _Fixture,
        *,
        fault: str | None = None,
    ):
        self.fixture = fixture
        self.store = fixture.store
        self.fault = fault
        self.calls: list[tuple] = []
        self.branches: list[_Branch] = []
        self.shared_state = [_digest("fresh worker restore")]
        self.state_hash_calls = 0
        self.corrupted = False
        self.manifest_count_during_callbacks: list[int] = []

    def _callback_guard(self) -> None:
        self.manifest_count_during_callbacks.append(
            len(list(self.store.manifests_dir.iterdir()))
        )
        for action in (self.fixture.stock_action, self.fixture.candidate_action):
            assert self.store.manifest_path(action.manifest.sha256).is_file()

    def restore(self, request: RestoreRequest) -> RestoreReceipt:
        self._callback_guard()
        self.calls.append(("restore", request.order, request.position))
        initial = _digest("fresh worker restore")
        if self.fault == "shared-state-leak":
            self.shared_state[0] = initial
            state_box = self.shared_state
        else:
            state_box = [initial]
        if (
            self.fault == "restore-nondeterminism"
            and request.order == "candidate-stock"
        ):
            state_box[0] = _digest(initial, request.order)
        if self.fault == "reuse-branch" and self.branches:
            branch = self.branches[0]
            branch.state = initial
            branch.order = request.order
            branch.position = request.position
        else:
            branch = _Branch(state_box, request.order, request.position)
            self.branches.append(branch)
        return RestoreReceipt(
            branch=branch,
            boundary=request.boundary,
            endpoint=request.endpoint,
            state_sha256=branch.state,
        )

    def state_sha256(self, branch: object) -> str:
        assert isinstance(branch, _Branch)
        self.state_hash_calls += 1
        if self.fault == "unstable-state-hash":
            return _digest(branch.state, str(self.state_hash_calls % 2))
        return branch.state

    def apply_action(
        self, branch: object, request: ApplyActionRequest
    ) -> ApplyActionReceipt:
        self._callback_guard()
        assert isinstance(branch, _Branch)
        self.calls.append(
            (
                "apply",
                request.action.sha256,
                request.resulting_fragment.ref,
            )
        )
        branch.action = request.action
        branch.state = _digest("applied", branch.state, request.resulting_fragment.data)
        action = request.action
        if self.fault == "crosswire-action-application":
            action = replace(action, manifest_id="crosswired-action")
        return ApplyActionReceipt(
            action=action,
            selected_pseudo_gradient=request.selected_pseudo_gradient,
            resulting_fragment=request.resulting_fragment.ref,
            state_sha256=branch.state,
        )

    def evaluate(self, branch: object, request: EvaluationRequest) -> EvaluationReceipt:
        self._callback_guard()
        assert isinstance(branch, _Branch)
        self.calls.append(
            ("evaluate", request.action.sha256, request.step, request.evaluation.ref)
        )
        if self.fault == "evaluation-mutation":
            branch.state = _digest(branch.state, "mutated by evaluation")
        artifact_parts: list[str | bytes] = [
            "evaluation artifact",
            request.action.sha256,
            str(request.step),
            branch.state,
            request.evaluation.data,
        ]
        if (
            self.fault == "artifact-nondeterminism"
            and branch.order == "candidate-stock"
        ):
            artifact_parts.append(branch.order)
        artifact = _object(self.store, _digest(*artifact_parts).encode())
        if self.fault == "missing-evaluation-object":
            artifact = ObjectRef("f" * 64, 17)

        base = 1.25 if request.action.manifest_id.endswith("stock-action") else 1.0
        loss = base if request.step == 0 else base - 0.125
        if self.fault == "loss-nondeterminism" and branch.order == "candidate-stock":
            loss += 0.03125
        loss_bits = struct.pack(">d", loss).hex()
        if self.fault == "nonfinite-loss":
            loss_bits = struct.pack(">d", float("nan")).hex()
        action = request.action
        if self.fault == "crosswire-evaluation":
            action = replace(action, manifest_id="crosswired-evaluation")

        if (
            self.fault == "corrupt-sealed-action"
            and request.step == 8
            and branch.order == "candidate-stock"
            and not self.corrupted
        ):
            path = self.store.manifest_path(request.action.sha256)
            raw = bytearray(path.read_bytes())
            raw[len(raw) // 2] ^= 1
            path.write_bytes(raw)
            self.corrupted = True

        return EvaluationReceipt(
            action=action,
            step=request.step,
            evaluation=request.evaluation.ref,
            artifact=artifact,
            state_sha256=branch.state,
            loss_f64_bits=loss_bits,
        )

    def train_group(
        self, branch: object, request: TrainGroupRequest
    ) -> TrainGroupReceipt:
        self._callback_guard()
        assert isinstance(branch, _Branch)
        self.calls.append(
            (
                "train",
                request.action.sha256,
                request.group_index,
                request.future_group.ref,
            )
        )
        if self.fault == "backend-exception" and request.group_index == 3:
            raise RuntimeError("synthetic missing group")
        state_parts = [branch.state, request.future_group.data]
        if self.fault == "state-nondeterminism" and branch.order == "candidate-stock":
            state_parts.append(branch.order.encode())
        branch.state = _digest(*state_parts)
        batch_hash = hashlib.sha256(request.future_group.data).hexdigest()
        if self.fault == "batch-nondeterminism" and branch.order == "candidate-stock":
            batch_hash = _digest(batch_hash, branch.order)
        group_index = request.group_index
        if self.fault == "missing-group-receipt" and request.group_index == 3:
            group_index = 4
        return TrainGroupReceipt(
            action=request.action,
            group_index=group_index,
            future_group=request.future_group.ref,
            batch_sha256=batch_hash,
            state_sha256=branch.state,
        )


def _evaluate(fixture: _Fixture, backend: _FakeBackend):
    return evaluate_isolated_crn_pair(
        fixture.store,
        boundary=fixture.boundary,
        stock_action=fixture.stock_action,
        candidate_action=fixture.candidate_action,
        responder_index=0,
        evaluation=fixture.evaluation,
        backend=backend,
        stock_outcome_manifest_id="crn-stock-outcome",
        candidate_outcome_manifest_id="crn-candidate-outcome",
    )


def test_isolated_crn_runs_both_orders_and_only_then_publishes_outcomes(tmp_path):
    fixture = _fixture(tmp_path)
    backend = _FakeBackend(fixture)
    manifest_count_before = len(list(fixture.store.manifests_dir.iterdir()))

    result = _evaluate(fixture, backend)

    assert backend.calls[:4] != []
    assert [call for call in backend.calls if call[0] == "restore"] == [
        ("restore", "stock-candidate", 0),
        ("restore", "stock-candidate", 1),
        ("restore", "candidate-stock", 0),
        ("restore", "candidate-stock", 1),
    ]
    assert [call[2] for call in backend.calls if call[0] == "apply"] == [
        fixture.stock_result,
        fixture.candidate_result,
        fixture.candidate_result,
        fixture.stock_result,
    ]
    evaluation_calls = [call for call in backend.calls if call[0] == "evaluate"]
    assert [call[2] for call in evaluation_calls] == [0, 8, 0, 8, 0, 8, 0, 8]
    assert {call[3] for call in evaluation_calls} == {fixture.evaluation}

    train_calls = [call for call in backend.calls if call[0] == "train"]
    assert len(train_calls) == 4 * 8
    for start in range(0, len(train_calls), 8):
        assert [call[2] for call in train_calls[start : start + 8]] == list(range(8))
        assert [call[3] for call in train_calls[start : start + 8]] == list(
            fixture.future_groups
        )
    assert set(backend.manifest_count_during_callbacks) == {manifest_count_before}
    assert len(list(fixture.store.manifests_dir.iterdir())) == manifest_count_before + 2

    stock = load_policy_outcome(fixture.store, result.stock_outcome)
    candidate = load_policy_outcome(fixture.store, result.candidate_outcome)
    assert stock.action_ref.manifest.sha256 == fixture.stock_action.manifest.sha256
    assert candidate.action_ref.manifest.sha256 == (
        fixture.candidate_action.manifest.sha256
    )
    assert stock.evaluation == candidate.evaluation == fixture.evaluation
    assert struct.pack(">d", stock.k0_loss).hex() == result.stock_trace.k0.loss_f64_bits
    assert struct.pack(">d", stock.k8_loss).hex() == result.stock_trace.k8.loss_f64_bits
    assert stock.evaluation_loss == stock.k8_loss
    assert candidate.evaluation_loss == candidate.k8_loss
    assert len(result.stock_trace.groups) == len(result.candidate_trace.groups) == 8
    fixture.store.audit()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("reuse-branch", "reused a branch"),
        ("shared-state-leak", "completed branch was mutated"),
        ("restore-nondeterminism", "fresh worker restores"),
        ("evaluation-mutation", "evaluation mutated branch state"),
        ("crosswire-action-application", "application receipt is cross-wired"),
        ("crosswire-evaluation", "evaluation receipt is cross-wired"),
        ("missing-group-receipt", "missing or cross-wired"),
        ("loss-nondeterminism", "nondeterminism or branch leakage"),
        ("batch-nondeterminism", "nondeterminism or branch leakage"),
        ("state-nondeterminism", "nondeterminism or branch leakage"),
        ("artifact-nondeterminism", "nondeterminism or branch leakage"),
        ("unstable-state-hash", "state hash is nondeterministic"),
        ("nonfinite-loss", "decode to a finite f64"),
        ("missing-evaluation-object", "regular non-symlink"),
        ("backend-exception", "backend future group 3 failed"),
        ("corrupt-sealed-action", "manifest SHA-256 mismatch"),
    ],
)
def test_adversarial_backend_fails_before_outcome_publication(tmp_path, fault, message):
    fixture = _fixture(tmp_path)
    backend = _FakeBackend(fixture, fault=fault)
    manifest_count_before = len(list(fixture.store.manifests_dir.iterdir()))

    with pytest.raises(CaptureStoreError, match=message):
        _evaluate(fixture, backend)

    assert len(list(fixture.store.manifests_dir.iterdir())) == manifest_count_before


def test_incomplete_worker_future_groups_cannot_seal_crn_action(tmp_path):
    with pytest.raises(
        CaptureStoreError, match="complete canonical future groups 0..7"
    ):
        _fixture(tmp_path, future_state="incomplete")


@pytest.mark.parametrize("arm", ["stock", "candidate"])
def test_missing_crn_capability_fails_before_first_callback(tmp_path, arm):
    fixture = _fixture(
        tmp_path,
        stock_required=("worker_restore",) if arm == "stock" else COMPLETE_CAPABILITIES,
        candidate_required=("worker_restore",)
        if arm == "candidate"
        else COMPLETE_CAPABILITIES,
    )
    backend = _FakeBackend(fixture)

    with pytest.raises(
        CRNEvaluationError, match=r"complete worker_restore\+crn_train_k8"
    ):
        _evaluate(fixture, backend)

    assert backend.calls == []


def test_action_role_swap_and_boundary_cross_wire_fail_before_callbacks(tmp_path):
    fixture = _fixture(tmp_path, suffix="one")
    backend = _FakeBackend(fixture)
    with pytest.raises(CRNEvaluationError, match="stock action must"):
        evaluate_isolated_crn_pair(
            fixture.store,
            boundary=fixture.boundary,
            stock_action=fixture.candidate_action,
            candidate_action=fixture.stock_action,
            responder_index=0,
            evaluation=fixture.evaluation,
            backend=backend,
            stock_outcome_manifest_id="stock-outcome",
            candidate_outcome_manifest_id="candidate-outcome",
        )
    assert backend.calls == []

    second = _fixture(tmp_path, store=fixture.store, suffix="two")
    with pytest.raises(CRNEvaluationError, match="cross-wired to another boundary"):
        evaluate_isolated_crn_pair(
            fixture.store,
            boundary=fixture.boundary,
            stock_action=fixture.stock_action,
            candidate_action=second.candidate_action,
            responder_index=0,
            evaluation=fixture.evaluation,
            backend=backend,
            stock_outcome_manifest_id="stock-outcome",
            candidate_outcome_manifest_id="candidate-outcome",
        )
    assert backend.calls == []


@pytest.mark.parametrize(
    ("responder_index", "stock_id", "candidate_id", "message"),
    [
        (True, "stock-outcome", "candidate-outcome", "non-negative integer"),
        (1, "stock-outcome", "candidate-outcome", "absent from the boundary"),
        (0, "bad/id", "candidate-outcome", "canonical manifest_id"),
        (0, "same-outcome", "same-outcome", "must differ"),
    ],
)
def test_invalid_evaluator_inputs_fail_before_callbacks(
    tmp_path, responder_index, stock_id, candidate_id, message
):
    fixture = _fixture(tmp_path)
    backend = _FakeBackend(fixture)

    with pytest.raises(CRNEvaluationError, match=message):
        evaluate_isolated_crn_pair(
            fixture.store,
            boundary=fixture.boundary,
            stock_action=fixture.stock_action,
            candidate_action=fixture.candidate_action,
            responder_index=responder_index,
            evaluation=fixture.evaluation,
            backend=backend,
            stock_outcome_manifest_id=stock_id,
            candidate_outcome_manifest_id=candidate_id,
        )

    assert backend.calls == []
