from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar

import pytest

from yeto.protocol import FinalManifest
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.dense_sweep_wire import (
    DenseSweepConfig,
    PendingDenseWirePolicy,
)
from yeto.rl.local_learner import (
    ComponentIdentity,
    ParameterLayout,
    ParameterSpec,
)
from yeto.rl.miles import (
    get_current_published_policy_identity,
    set_current_published_policy_identity,
)
from yeto.rl.miles_chunked_full_parameter import (
    ReferencedPolicyCut,
    StoredAuthoritativeFragment,
)
from yeto.rl.miles_full_parameter_dense import (
    MilesDenseFullParameterConfig,
    MilesFullParameterDenseSync,
)
from yeto.rl.trajectory_evidence import TrajectoryBatchEvidence


@dataclass(frozen=True)
class _OptimizerProof:
    topology: str
    role: str
    installed_policy_version: int
    local_step_generation: int
    last_rollout_id: int
    scheduler_num_steps: int
    populated_parameter_count: int
    optimizer_state_tensor_count: int
    optimizer_state_scalar_count: int
    selected_wire_name: str
    selected_state_sha256: str
    model_master_parameter_count: int


class _Remote:
    def __init__(self, function):
        self.function = function

    async def remote(self, *args, **kwargs):
        value = self.function(*args, **kwargs)
        if hasattr(value, "__await__"):
            return await value
        return value


class _Sink:
    def __init__(self, events: list[object]) -> None:
        self.events = events
        self.released = False

    def __call__(self, fragment_id, version, payload):
        assert payload == b"\0\0\0\0"
        return _fragment(fragment_id, version)

    def release_all(self) -> int:
        if self.released:
            return 0
        self.released = True
        self.events.append("sink-release")
        return 1


class _Adapter:
    def __init__(self, layout: ParameterLayout, events: list[object]) -> None:
        self.layout = layout
        self.events = events
        self.anchor = _cut(0, 0, "a")
        self.local = _cut(0, 1, "b")
        self.target = _cut(1, 0, "c")

    def fragment_sink(self, reference):
        self.events.append(("sink", reference.policy_version))
        return _Sink(self.events)

    def fragment_parts(self, cut, fragment_id):
        assert cut is self.anchor
        assert fragment_id == 0
        return lambda: (b"\0\0\0\0",)

    def verify_initial_fragments(self, reference, fragments):
        assert reference is self.anchor
        assert tuple(fragment.version for fragment in fragments) == (0,)
        self.events.append("initial-verified")

    async def record_grpo_local_step(
        self,
        actor_model,
        *,
        anchor,
        rollout_id,
        learner_id,
        learner_generation,
        trajectories,
    ):
        assert anchor is self.anchor
        assert rollout_id == 0
        assert learner_id == 0
        assert learner_generation == 7
        assert trajectories.behavior_policy_hash == self.anchor.policy_hash
        self.events.append("receipt")
        return LocalStepReceipt(
            algorithm="grpo",
            learner_id=learner_id,
            learner_generation=learner_generation,
            base_policy_version=rollout_id,
            base_policy_hash=anchor.policy_hash,
            input_batch_hash=trajectories.input_batch_hash,
            trajectory_ids=("trajectory-0",),
            trained_tokens=11,
            optimizer_steps=1,
            optimizer_step_succeeded=True,
            parameter_layout_hash=self.layout.layout_hash,
        )

    async def capture(self, actor_model, *, policy_version, local_step_generation):
        assert policy_version == 0
        assert local_step_generation == 1
        self.events.append("local-captured")
        return self.local

    def delta_parts(self, anchor, local, fragment_id):
        assert anchor is self.anchor
        assert local is self.local
        assert fragment_id == 0
        return lambda: (b"\0\0\0\0",)

    def assemble_target(self, reference, *, target_policy_version, fragments):
        assert reference is self.anchor
        assert target_policy_version == 1
        assert tuple(fragment.version for fragment in fragments) == (1,)
        self.events.append("target-assembled")
        return self.target

    async def apply(self, actor_model, target, *, commit_token):
        assert target is self.target
        assert len(commit_token) == 64
        actor_model.applied = True
        self.events.append(("apply", commit_token))
        return 1

    def release(self, cut):
        self.events.append(
            ("cut-release", cut.policy_version, cut.local_step_generation)
        )
        return 1


class _Wire:
    instances: ClassVar[list[_Wire]] = []

    def __init__(self, layout, config, *, client):
        del client
        self.layout = layout
        self.config = config
        self.policy_version = None
        self.commits = []
        self.closed = False
        self.exchange_call = None
        type(self).instances.append(self)

    def start(self, initial_parts, *, policy_version, payload_sink):
        assert policy_version == 0
        assert set(initial_parts) == {0}
        assert b"".join(initial_parts[0]()) == b"\0\0\0\0"
        self.policy_version = 0
        return (payload_sink(0, 0, b"\0\0\0\0"),)

    def exchange(self, **kwargs):
        assert kwargs["base_policy_version"] == 0
        assert kwargs["trained_tokens"] == 11
        assert len(kwargs["sweep_update_id"]) == 64
        assert set(kwargs["delta_parts"]) == {0}
        assert b"".join(kwargs["delta_parts"][0]()) == b"\0\0\0\0"
        self.exchange_call = kwargs
        payload = kwargs["payload_sink"](0, 1, b"\0\0\0\0")
        return PendingDenseWirePolicy(
            kwargs["sweep_update_id"],
            1,
            (1,),
            (payload,),
            (),
            FinalManifest(1, (1,)),
        )

    def commit_applied(self, pending):
        assert pending.policy_version == 1
        self.commits.append(pending)
        self.policy_version = 1

    def close(self):
        self.closed = True


class _Actor:
    def __init__(self) -> None:
        self.applied = False

    async def full_parameter_optimizer_states(self):
        # Global application must advance policy metadata while preserving all
        # Adam/scheduler evidence.  This mirrors Miles' real bounded proof.
        return (
            _OptimizerProof(
                topology="tp0",
                role="actor",
                installed_policy_version=1 if self.applied else 0,
                local_step_generation=0 if self.applied else 1,
                last_rollout_id=0,
                scheduler_num_steps=1,
                populated_parameter_count=1,
                optimizer_state_tensor_count=2,
                optimizer_state_scalar_count=2,
                selected_wire_name="actor::tp0::weight",
                selected_state_sha256="d" * 64,
                model_master_parameter_count=1,
            ),
        )


def _cut(version: int, generation: int, digest: str) -> ReferencedPolicyCut:
    return ReferencedPolicyCut(
        policy_version=version,
        local_step_generation=generation,
        layout_hash="e" * 64,
        policy_hash=digest * 64,
        content_hash=digest * 64,
        transport_cut=SimpleNamespace(name=f"v{version}g{generation}"),
    )


def _fragment(fragment_id: int, version: int) -> StoredAuthoritativeFragment:
    return StoredAuthoritativeFragment(
        fragment_id=fragment_id,
        version=version,
        parameter_layout_hash="e" * 64,
        topology="tp0",
        plan_hash="f" * 64,
        descriptor=SimpleNamespace(fragment_id=fragment_id),
        refs=[object()],
        wire_payload_hash="1" * 64,
    )


def _config() -> MilesDenseFullParameterConfig:
    component = ComponentIdentity("actor", "2" * 40, "3" * 64)
    return MilesDenseFullParameterConfig(
        component=component,
        wire=DenseSweepConfig(
            ("127.0.0.1", 1),
            learner_id=0,
            learner_generation=7,
            policy_rounds=1,
        ),
        learner_generations=(7,),
        minimum_fragments=1,
        training_contract_hash="8" * 64,
        expected_layout_hash="9" * 64,
    )


def _layout() -> ParameterLayout:
    return ParameterLayout.create(
        algorithm="grpo",
        components=(_config().component,),
        specs=(ParameterSpec("actor", "weight", (1,), "float32", 1, "tp0"),),
        num_fragments=1,
        fragment_strategy="owner_affine",
    )


async def _initialized_sync(monkeypatch, tmp_path, *, fail_target_publication: bool):
    import yeto.rl.miles_full_parameter_dense as module

    events = []
    adapter = _Adapter(_layout(), events)

    class _AdapterFactory:
        @staticmethod
        async def initialize(*args, **kwargs):
            del args, kwargs
            return adapter, adapter.anchor

    _Wire.instances.clear()
    monkeypatch.setattr(module, "MilesChunkedFullParameterAdapter", _AdapterFactory)
    monkeypatch.setattr(module, "DenseSweepWire", _Wire)
    monkeypatch.setattr(module, "SyncerClient", lambda *args, **kwargs: object())

    published_tokens = []

    def publish(token):
        published_tokens.append(token)
        if fail_target_publication and token == "yeto:1":
            raise RuntimeError("inference token publication failed")

    engine = SimpleNamespace(update_weight_version=_Remote(publish))
    rollout_args = SimpleNamespace()
    rollout_manager = SimpleNamespace(
        get_updatable_engines_and_lock=_Remote(
            lambda: SimpleNamespace(rollout_engines=(engine,))
        ),
        set_external_policy_identity=_Remote(
            lambda version, digest: set_current_published_policy_identity(
                rollout_args,
                policy_version=version,
                policy_hash=digest,
            )
        ),
    )
    args = SimpleNamespace(
        start_rollout_id=0,
        num_rollout=1,
        yeto_rl_trajectory_evidence_dir="/not-read",
        yeto_rl_event_tape=str(tmp_path / "events.jsonl"),
        yeto_rl_learner_id=0,
    )
    actor = _Actor()
    sync = MilesFullParameterDenseSync(args, _config())
    await sync.initialize(actor_model=actor, rollout_manager=rollout_manager)
    evidence = TrajectoryBatchEvidence(
        rollout_id=0,
        behavior_policy_hash=adapter.anchor.policy_hash,
        input_batch_hash="4" * 64,
        trained_tokens=11,
        envelopes=(SimpleNamespace(trajectory_id="trajectory-0"),),
    )
    sync._trajectory_evidence = lambda rollout_id: evidence
    return (
        sync,
        args,
        rollout_args,
        actor,
        adapter,
        _Wire.instances[0],
        events,
        published_tokens,
    )


@pytest.mark.asyncio
async def test_terminal_h1_lifecycle_commits_only_after_inference_publication(
    monkeypatch,
    tmp_path,
):
    sync, args, rollout_args, actor, adapter, wire, events, published = await _initialized_sync(
        monkeypatch,
        tmp_path,
        fail_target_publication=False,
    )

    with pytest.raises(RuntimeError, match="no current published policy identity"):
        get_current_published_policy_identity(args, expected_policy_version=0)
    await sync.after_inference_publication(rollout_id=None, actor_model=actor)
    assert published == ["yeto:0"]
    assert get_current_published_policy_identity(args, expected_policy_version=0) == (
        0,
        adapter.anchor.policy_hash,
    )
    assert get_current_published_policy_identity(
        rollout_args,
        expected_policy_version=0,
    ) == (0, adapter.anchor.policy_hash)

    assert await sync.after_local_train(
        rollout_id=0,
        actor_model=actor,
        rollout_data=object(),
    )
    assert "receipt" in events
    assert wire.exchange_call is not None
    assert wire.commits == []
    assert sync.current is adapter.anchor
    assert sync.pending_publication is not None

    await sync.after_inference_publication(rollout_id=0, actor_model=actor)
    assert published == ["yeto:0", "yeto:1"]
    assert get_current_published_policy_identity(args, expected_policy_version=1) == (
        1,
        adapter.target.policy_hash,
    )
    assert get_current_published_policy_identity(
        rollout_args,
        expected_policy_version=1,
    ) == (1, adapter.target.policy_hash)
    assert len(wire.commits) == 1
    assert sync.current is adapter.target
    assert sync.pending_publication is None
    assert sync.finished

    await sync.finalize()
    assert wire.closed
    assert sync.current is None
    assert ("cut-release", 0, 0) in events
    assert ("cut-release", 0, 1) in events
    assert ("cut-release", 1, 0) in events


@pytest.mark.asyncio
async def test_failed_inference_publication_keeps_wire_pending_and_blocks_finalize(
    monkeypatch,
    tmp_path,
):
    sync, args, rollout_args, actor, adapter, wire, _, published = await _initialized_sync(
        monkeypatch,
        tmp_path,
        fail_target_publication=True,
    )
    await sync.after_inference_publication(rollout_id=None, actor_model=actor)
    assert await sync.after_local_train(
        rollout_id=0,
        actor_model=actor,
        rollout_data=object(),
    )

    pending = sync.pending_publication
    with pytest.raises(RuntimeError, match="inference token publication failed"):
        await sync.after_inference_publication(rollout_id=0, actor_model=actor)

    assert published == ["yeto:0", "yeto:1"]
    assert wire.commits == []
    assert sync.pending_publication is pending
    assert sync.current is adapter.anchor
    assert not sync.finished
    assert get_current_published_policy_identity(args, expected_policy_version=0) == (
        0,
        adapter.anchor.policy_hash,
    )
    assert get_current_published_policy_identity(
        rollout_args,
        expected_policy_version=0,
    ) == (0, adapter.anchor.policy_hash)
    with pytest.raises(RuntimeError, match="stopped before"):
        await sync.finalize()
    assert not wire.closed
