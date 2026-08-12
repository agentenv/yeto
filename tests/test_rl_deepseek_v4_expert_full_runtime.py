from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yeto.rl import deepseek_v4_expert_full_runtime as runtime
from yeto.rl.deepseek_v4_expert_clone import (
    EXPERT_PARALLEL_SIZE,
    NUM_LAYERS,
    ORIGINAL_EXPERTS_PER_RANK,
    TRAINING_EXPERTS_PER_RANK,
)
from yeto.rl.deepseek_v4_expert_full_runtime import (
    filter_selected_expert_tasks,
    install_on_arguments,
    install_on_lora_utils,
    make_hybrid_trainable_state,
    selected_expert_hf_name,
)


def _expert_name(
    expert: int,
    projection: str = "gate_proj",
    *,
    layer: int = 0,
) -> str:
    return (
        f"base_model.model.model.layers.{layer}.mlp.experts."
        f"{expert}.{projection}.weight"
    )


def _expert_tasks(expert: int, *, layer: int = 0) -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(
            mapping=SimpleNamespace(
                hf_param={
                    "gate": _expert_name(expert, layer=layer),
                    "up": _expert_name(expert, "up_proj", layer=layer),
                }
            )
        ),
        SimpleNamespace(
            mapping=SimpleNamespace(
                hf_param=_expert_name(expert, "down_proj", layer=layer)
            )
        ),
    )


@pytest.mark.parametrize(
    ("ref_load", "hf_checkpoint", "expected"),
    (
        ("/models/trainer-bf16", "/models/rollout-fp8", "/models/trainer-bf16"),
        (None, "/models/rollout-bf16", "/models/rollout-bf16"),
    ),
)
def test_actor_bridge_uses_trainer_reference_layout(
    monkeypatch, ref_load, hf_checkpoint, expected
):
    calls = []
    bridge = object()

    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            return bridge

    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_bridge = ModuleType("megatron.bridge")
    megatron_bridge.AutoBridge = AutoBridge
    megatron.bridge = megatron_bridge
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.bridge", megatron_bridge)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            ref_load=ref_load,
            hf_checkpoint=hf_checkpoint,
            yeto_rl_trust_remote_code=True,
        )
    )

    assert runtime._actor_bridge(actor) is bridge
    assert runtime._actor_bridge(actor) is bridge
    assert calls == [(expected, {"trust_remote_code": True})]


def test_actor_bridge_requires_a_huggingface_source():
    actor = SimpleNamespace(
        args=SimpleNamespace(
            ref_load=None,
            hf_checkpoint=None,
            yeto_rl_trust_remote_code=False,
        )
    )

    with pytest.raises(RuntimeError, match="no HuggingFace checkpoint"):
        runtime._actor_bridge(actor)


def test_collective_expert_tasks_have_identical_ep8_topology():
    expected_count = NUM_LAYERS * 4 * 2
    for rank in range(EXPERT_PARALLEL_SIZE):
        tasks = [
            task
            for layer in range(NUM_LAYERS)
            for expert in range(
                rank * TRAINING_EXPERTS_PER_RANK,
                (rank + 1) * TRAINING_EXPERTS_PER_RANK,
            )
            for task in _expert_tasks(expert, layer=layer)
        ]

        selected = runtime.filter_collective_expert_tasks(tasks, expert_count=16)

        physical_ids = [
            int(runtime._mapping_hf_names(task)[0].split(".experts.")[1].split(".")[0])
            for task in selected
        ]
        assert len(selected) == expected_count
        assert {expert % TRAINING_EXPERTS_PER_RANK for expert in physical_ids} == {
            ORIGINAL_EXPERTS_PER_RANK + offset for offset in range(4)
        }
        assert {expert // TRAINING_EXPERTS_PER_RANK for expert in physical_ids} == {
            rank
        }


@pytest.mark.parametrize(
    ("ref_load", "hf_checkpoint", "expected_source"),
    (
        ("/models/trainer-bf16", "/models/rollout-fp8", "/models/trainer-bf16"),
        (None, "/models/rollout-bf16", "/models/rollout-bf16"),
    ),
)
def test_weight_iterator_filters_after_collective_export(
    monkeypatch, ref_load, hf_checkpoint, expected_source
):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "5")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    rank = EXPERT_PARALLEL_SIZE - 1
    tasks = [
        task
        for expert in range(
            rank * TRAINING_EXPERTS_PER_RANK,
            (rank + 1) * TRAINING_EXPERTS_PER_RANK,
        )
        for task in _expert_tasks(expert)
    ]

    class Bridge:
        def get_conversion_tasks(self, _model):
            return tasks

        def export_hf_weights(self, _model, **kwargs):
            self.exported_tasks = tuple(kwargs["conversion_tasks"])
            return (
                (_expert_name(expert, projection), torch.ones(1), "megatron-name")
                for expert in range(256, 288)
                for projection in ("gate_proj", "up_proj", "down_proj")
            )

    task_bridge = Bridge()
    bridge_calls = []

    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(path, **kwargs):
            bridge_calls.append((path, kwargs))
            return task_bridge

    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_bridge = ModuleType("megatron.bridge")
    megatron_bridge.AutoBridge = AutoBridge
    megatron.bridge = megatron_bridge
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.bridge", megatron_bridge)

    postprocess_sources = []

    class Iterator:
        def get_hf_weight_chunks(self, _weights, weight_type="base"):
            yield ("original", weight_type)

        def _postprocess_and_quantize(self, named_weights, _weight_type):
            postprocess_sources.append(self.args.hf_checkpoint)
            return named_weights

    class RolloutBridge:
        def get_conversion_tasks(self, _model):
            raise AssertionError("FP8 rollout Bridge must not discover trainer tasks")

        def export_hf_weights(self, _model, **_kwargs):
            raise AssertionError("FP8 rollout Bridge must not export trainer weights")

    iterator_module = SimpleNamespace(
        HfWeightIteratorBridge=Iterator,
        strip_param_name_prefix=lambda name: name,
        megatron_bridge_utils=SimpleNamespace(
            patch_megatron_model=lambda _model: nullcontext()
        ),
        _process_conversion_tasks=lambda selected, _weights: list(selected),
        is_lora_weight_name=lambda _name: False,
        get_atomic_update_groups=lambda _args, _model_name: (),
        _stream_atomic_units=lambda items, _groups: (
            [(name, weight)] for name, weight, _megatron_name in items
        ),
        _chunk_atomic_units_by_size=lambda units, chunk_size: units,
    )
    runtime.install_on_weight_iterator(iterator_module)
    iterator = Iterator()
    rollout_bridge = RolloutBridge()
    iterator._bridge = rollout_bridge
    iterator.model = object()
    iterator.args = SimpleNamespace(
        update_weight_buffer_size=1,
        ref_load=ref_load,
        hf_checkpoint=hf_checkpoint,
        yeto_rl_trust_remote_code=True,
    )
    iterator.model_name = "deepseek-v4"

    chunks = list(iterator.get_hf_weight_chunks({}, weight_type="base"))
    list(iterator.get_hf_weight_chunks({}, weight_type="base"))

    task_experts = [
        int(runtime._mapping_hf_names(task)[0].split(".experts.")[1].split(".")[0])
        for task in task_bridge.exported_tasks
    ]
    assert len(task_bridge.exported_tasks) == 4 * 2
    assert {expert % TRAINING_EXPERTS_PER_RANK for expert in task_experts} == {
        ORIGINAL_EXPERTS_PER_RANK + offset for offset in range(4)
    }
    assert {name for chunk in chunks for name, _weight in chunk} == {
        _expert_name(expert, projection)
        for expert in range(256, 261)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    assert bridge_calls == [(expected_source, {"trust_remote_code": True})]
    assert postprocess_sources == [hf_checkpoint, hf_checkpoint]
    assert iterator._bridge is rollout_bridge


def test_selected_expert_task_filter_keeps_only_the_requested_safe_clone_prefix():
    tasks = [
        # Bridge task names carry trainer-physical IDs.  Physical 283 is
        # logical original 255; physical 32 is logical clone 256.
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(283))),
        SimpleNamespace(
            mapping=SimpleNamespace(
                hf_param={
                    "gate": _expert_name(32),
                    "up": _expert_name(32, "up_proj"),
                }
            )
        ),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(143))),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(176))),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(256))),
        SimpleNamespace(mapping=SimpleNamespace(hf_param="model.layers.0.self_attn.q_proj.weight")),
    ]

    selected = filter_selected_expert_tasks(tasks, expert_count=16)

    assert selected == tasks[1:3]
    assert selected_expert_hf_name(_expert_name(256), expert_count=16)
    assert selected_expert_hf_name(_expert_name(271), expert_count=16)
    assert not selected_expert_hf_name(_expert_name(272), expert_count=16)


@dataclass(frozen=True)
class _State:
    policy_version: int
    layout_hash: str
    tensors: dict[str, torch.Tensor]
    train_rollout_kl: float | None = None
    ess_ratio: float | None = None
    pg_clipfrac: float | None = None
    train_seconds: float | None = None


class _TrainableStateModule:
    TrainableState = _State

    @staticmethod
    def _layout_hash(tensors):
        return "layout:" + ",".join(sorted(tensors))


def _tiny_hybrid_tensors() -> dict[str, torch.Tensor]:
    return {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.tensor([0.25, 0.5], dtype=torch.float32),
        _expert_name(256): torch.tensor([1.0, 2.0], dtype=torch.float32),
        _expert_name(256, "up_proj"): torch.tensor(
            [3.0, 4.0], dtype=torch.float32
        ),
        _expert_name(256, "down_proj"): torch.tensor(
            [5.0, 6.0], dtype=torch.float32
        ),
    }


class _FakeRay:
    @dataclass(frozen=True)
    class Ref:
        index: int

    def __init__(self):
        self.objects = []

    def put(self, value):
        self.objects.append(value)
        return self.Ref(len(self.objects) - 1)


def test_owner_sharded_export_merge_requires_exact_disjoint_coverage(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    tensors = _tiny_hybrid_tensors()
    names = tuple(tensors)
    root = runtime._TrainableStateFragment(
        source_rank=0,
        policy_version=4,
        expected_names=names,
        tensors={names[0]: tensors[names[0]], names[1]: tensors[names[1]]},
        train_rollout_kl=0.125,
        ess_ratio=0.875,
        pg_clipfrac=0.25,
        train_seconds=3.5,
    )
    peer = runtime._TrainableStateFragment(
        source_rank=7,
        policy_version=4,
        expected_names=names,
        tensors={names[2]: tensors[names[2]], names[3]: tensors[names[3]]},
    )

    state = runtime._merge_export_fragments(
        _TrainableStateModule,
        [root, None, None, None, None, None, None, peer],
    )

    assert state.policy_version == 4
    assert set(state.tensors) == set(names)
    assert state.layout_hash == _TrainableStateModule._layout_hash(tensors)
    assert state.train_rollout_kl == 0.125
    assert state.ess_ratio == 0.875
    assert state.pg_clipfrac == 0.25
    assert state.train_seconds == 3.5

    missing = runtime._TrainableStateFragment(
        source_rank=7,
        policy_version=4,
        expected_names=names,
        tensors={names[2]: tensors[names[2]]},
    )
    with pytest.raises(RuntimeError, match="do not exactly cover"):
        runtime._merge_export_fragments(
            _TrainableStateModule,
            [root, None, None, None, None, None, None, missing],
        )

    duplicate = runtime._TrainableStateFragment(
        source_rank=7,
        policy_version=4,
        expected_names=names,
        tensors={
            names[1]: tensors[names[1]],
            names[2]: tensors[names[2]],
            names[3]: tensors[names[3]],
        },
    )
    with pytest.raises(RuntimeError, match="duplicate tensor"):
        runtime._merge_export_fragments(
            _TrainableStateModule,
            [root, None, None, None, None, None, None, duplicate],
        )

    with pytest.raises(RuntimeError, match="source rank"):
        runtime._merge_export_fragments(_TrainableStateModule, [peer, root])

    peer_with_metrics = runtime._TrainableStateFragment(
        source_rank=7,
        policy_version=4,
        expected_names=names,
        tensors={names[2]: tensors[names[2]], names[3]: tensors[names[3]]},
        train_seconds=1.0,
    )
    with pytest.raises(RuntimeError, match="nonzero rank"):
        runtime._merge_export_fragments(
            _TrainableStateModule,
            [root, None, None, None, None, None, None, peer_with_metrics],
        )


@pytest.mark.parametrize(("rank", "retained"), ((0, True), (1, False)))
def test_expert_export_retains_only_the_minimum_global_owner(
    monkeypatch, rank, retained
):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    name = _expert_name(256)
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=(SimpleNamespace(name=name, shape=(2,)),)
        )
    )
    monkeypatch.setattr(runtime, "_expert_views", lambda _actor: {name: value})
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local_meta: output.__setitem__(slice(None), [local_meta] * 2),
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda broadcast_value, *, src: broadcast_value.copy_(value),
    )

    tensors = runtime._export_experts(
        actor,
        retain=True,
        canonical_sources_only=True,
    )

    assert (name in tensors) is retained
    if retained:
        assert torch.equal(tensors[name], value)


def test_expert_export_rejects_disagreeing_dp_replica(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    name = _expert_name(256)
    source_value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    replica_value = torch.tensor([9.0, 10.0], dtype=torch.float32)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=(SimpleNamespace(name=name, shape=(2,)),)
        )
    )
    monkeypatch.setattr(
        runtime,
        "_expert_views",
        lambda _actor: {name: replica_value},
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local_meta: output.__setitem__(slice(None), [local_meta] * 2),
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda value, *, src: value.copy_(source_value),
    )

    with pytest.raises(RuntimeError, match="DP replicas disagree"):
        runtime._export_experts(
            actor,
            retain=True,
            canonical_sources_only=True,
        )


def test_apply_transport_chunks_below_cap_and_round_trips_state(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    state = make_hybrid_trainable_state(
        _TrainableStateModule,
        9,
        _tiny_hybrid_tensors(),
        train_rollout_kl=0.2,
        ess_ratio=0.8,
        pg_clipfrac=0.3,
        train_seconds=4.5,
    )
    ray = _FakeRay()

    manifest, refs = runtime._chunk_state_for_ray(
        state,
        ray_module=ray,
        max_chunk_bytes=16,
    )

    assert manifest.chunk_tensor_bytes == (16, 16)
    assert all(size <= 16 for size in manifest.chunk_tensor_bytes)
    assert refs == (_FakeRay.Ref(0), _FakeRay.Ref(1))
    resolved = [ray.objects[ref.index] for ref in refs]
    rebuilt = runtime._state_from_chunk_manifest(
        _TrainableStateModule,
        manifest,
        chunks=resolved,
    )
    assert rebuilt.policy_version == state.policy_version
    assert rebuilt.layout_hash == state.layout_hash
    assert rebuilt.train_rollout_kl == state.train_rollout_kl
    assert rebuilt.ess_ratio == state.ess_ratio
    assert rebuilt.pg_clipfrac == state.pg_clipfrac
    assert rebuilt.train_seconds == state.train_seconds
    assert set(rebuilt.tensors) == set(state.tensors)
    assert all(
        torch.equal(rebuilt.tensors[name], state.tensors[name])
        for name in state.tensors
    )

    oversized = _State(
        1,
        "layout:oversized",
        {"oversized": torch.zeros(5, dtype=torch.float32)},
    )
    with pytest.raises(ValueError, match="exceeds the Ray chunk cap"):
        runtime._chunk_state_for_ray(
            oversized,
            ray_module=ray,
            max_chunk_bytes=16,
        )


@pytest.mark.parametrize("corruption", ("missing", "duplicate"))
def test_apply_transport_rejects_missing_and_duplicate_tensors(
    monkeypatch, corruption
):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    state = make_hybrid_trainable_state(
        _TrainableStateModule,
        2,
        _tiny_hybrid_tensors(),
    )
    ray = _FakeRay()
    manifest, refs = runtime._chunk_state_for_ray(
        state,
        ray_module=ray,
        max_chunk_bytes=16,
    )
    chunks = [dict(ray.objects[ref.index]) for ref in refs]
    repeated_name = next(iter(chunks[0]))
    if corruption == "missing":
        chunks[0].pop(repeated_name)
        manifest = runtime._ChunkedStateManifest(
            policy_version=manifest.policy_version,
            layout_hash=manifest.layout_hash,
            tensor_names=manifest.tensor_names,
            chunk_tensor_names=manifest.chunk_tensor_names,
            chunk_tensor_bytes=(
                manifest.chunk_tensor_bytes[0]
                - runtime._tensor_bytes(repeated_name, state.tensors[repeated_name]),
                manifest.chunk_tensor_bytes[1],
            ),
        )
        pattern = "does not match its name manifest"
    else:
        chunks[1][repeated_name] = chunks[0][repeated_name]
        manifest = runtime._ChunkedStateManifest(
            policy_version=manifest.policy_version,
            layout_hash=manifest.layout_hash,
            tensor_names=manifest.tensor_names,
            chunk_tensor_names=(
                manifest.chunk_tensor_names[0],
                manifest.chunk_tensor_names[1] + (repeated_name,),
            ),
            chunk_tensor_bytes=(
                manifest.chunk_tensor_bytes[0],
                manifest.chunk_tensor_bytes[1]
                + runtime._tensor_bytes(repeated_name, chunks[0][repeated_name]),
            ),
        )
        pattern = "duplicate tensor"
    with pytest.raises(RuntimeError, match=pattern):
        runtime._state_from_chunk_manifest(
            _TrainableStateModule,
            manifest,
            chunks=chunks,
        )


def test_v1_actor_group_merges_export_shards_and_sends_top_level_chunk_refs(
    monkeypatch,
):
    import asyncio

    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    tensors = _tiny_hybrid_tensors()
    names = tuple(tensors)
    fragments = (
        runtime._TrainableStateFragment(
            source_rank=0,
            policy_version=6,
            expected_names=names,
            tensors={names[0]: tensors[names[0]], names[1]: tensors[names[1]]},
        ),
        runtime._TrainableStateFragment(
            source_rank=1,
            policy_version=6,
            expected_names=names,
            tensors={names[2]: tensors[names[2]], names[3]: tensors[names[3]]},
        ),
    )

    trainable_state = ModuleType(
        "miles.backends.megatron_utils.trainable_state"
    )
    trainable_state.TrainableState = _State
    trainable_state._layout_hash = _TrainableStateModule._layout_hash
    megatron_utils = ModuleType("miles.backends.megatron_utils")
    megatron_utils.__path__ = []
    megatron_utils.trainable_state = trainable_state
    backends = ModuleType("miles.backends")
    backends.__path__ = []
    backends.megatron_utils = megatron_utils
    miles = ModuleType("miles")
    miles.__path__ = []
    miles.backends = backends
    monkeypatch.setitem(sys.modules, "miles", miles)
    monkeypatch.setitem(sys.modules, "miles.backends", backends)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils", megatron_utils)
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.trainable_state",
        trainable_state,
    )

    fake_ray = _FakeRay()
    ray = ModuleType("ray")
    ray.put = fake_ray.put
    monkeypatch.setitem(sys.modules, "ray", ray)

    calls = []

    class RemoteMethod:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, *args, **kwargs):
            calls.append((self.rank, args, kwargs))

            async def result():
                return 1 if self.rank[1] == "begin" else 4

            return result()

    class Actor:
        def __init__(self, rank):
            self.begin_chunked_trainable_state = RemoteMethod(
                (rank, "begin")
            )
            self.apply_trainable_state_chunk = RemoteMethod((rank, "chunk"))
            self.finish_chunked_trainable_state = RemoteMethod(
                (rank, "finish")
            )

    class RayTrainGroup:
        pass

    group_module = SimpleNamespace(RayTrainGroup=RayTrainGroup)
    runtime.install_on_actor_group(group_module)
    group = RayTrainGroup()
    group._actor_handles = [Actor(0), Actor(1)]

    async def broadcast(name):
        assert name == "export_trainable_state"
        return fragments

    group._broadcast = broadcast
    state = asyncio.run(group.export_trainable_state())
    assert set(state.tensors) == set(names)

    reset = asyncio.run(
        group.apply_trainable_state(state, reset_optimizer=True)
    )
    assert reset == 4
    assert len(fake_ray.objects) == 1
    assert [call[0][1] for call in calls] == [
        "begin",
        "begin",
        "chunk",
        "chunk",
        "finish",
        "finish",
    ]
    assert isinstance(calls[0][1][0], runtime._ChunkedStateManifest)
    assert calls[0][2] == {"reset_optimizer": True}
    assert calls[1][1][0] is calls[0][1][0]
    assert calls[1][2] == {"reset_optimizer": True}
    assert calls[2][1] == (0, _FakeRay.Ref(0))
    assert calls[3][1] == (0, None)
    assert calls[2][2] == calls[3][2] == {}
    assert calls[4][1:] == ((), {})
    assert calls[5][1:] == ((), {})


def _actor_module_with_value_preserving_restore():
    class MegatronTrainRayActor:
        def __init__(self):
            self.model = torch.nn.Module()
            frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
            frozen._yeto_expert_full_configured = True
            self.model.register_parameter("frozen_expert", frozen)

        def _switch_model(self, target_tag):
            if target_tag == "actor":
                with torch.no_grad():
                    self.model.frozen_expert.copy_(self.model.frozen_expert)
            return target_tag

    return SimpleNamespace(
        MegatronTrainRayActor=MegatronTrainRayActor,
        export_external_trainable_state=lambda *_args, **_kwargs: None,
        apply_external_trainable_state=lambda *_args, **_kwargs: 0,
    )


def test_actor_restore_refreshes_frozen_expert_version_baseline():
    actor_module = _actor_module_with_value_preserving_restore()
    runtime.install_on_actor(actor_module)
    actor = actor_module.MegatronTrainRayActor()
    runtime._assert_frozen_experts_unchanged(actor)
    previous_version = actor.model.frozen_expert._version

    assert actor._switch_model("actor") == "actor"

    assert actor.model.frozen_expert._version > previous_version
    runtime._assert_frozen_experts_unchanged(actor)


def test_actor_restore_keeps_guard_for_later_frozen_expert_modification():
    actor_module = _actor_module_with_value_preserving_restore()
    runtime.install_on_actor(actor_module)
    actor = actor_module.MegatronTrainRayActor()
    actor._switch_model("actor")

    with torch.no_grad():
        actor.model.frozen_expert.add_(1)

    with pytest.raises(RuntimeError, match="frozen original or unselected"):
        runtime._assert_frozen_experts_unchanged(actor)


def test_chunk_apply_lifecycle_rejects_out_of_order_and_publishes_only_at_finish(
    monkeypatch,
):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    tensors = _tiny_hybrid_tensors()
    specs = tuple(
        SimpleNamespace(name=name, shape=tuple(tensor.shape))
        for name, tensor in sorted(tensors.items())
    )
    state = make_hybrid_trainable_state(_TrainableStateModule, 11, tensors)
    ray = _FakeRay()
    manifest, refs = runtime._chunk_state_for_ray(
        state,
        ray_module=ray,
        max_chunk_bytes=16,
    )
    chunks = [ray.objects[ref.index] for ref in refs]
    actor = SimpleNamespace(
        args=SimpleNamespace(yeto_rl_expected_specs=specs),
        optimizer=object(),
        weights_backuper=SimpleNamespace(backup=lambda _name: None),
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _status, *, src: None,
    )
    applied = []
    finished = []
    monkeypatch.setattr(
        runtime,
        "_apply_policy_specs",
        lambda _actor, _state, chunk_specs: applied.append(
            tuple(spec.name for spec in chunk_specs)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_finish_hybrid_apply",
        lambda _module, _actor, **kwargs: finished.append(kwargs) or len(specs),
    )

    assert runtime.begin_chunked_hybrid_apply(
        _TrainableStateModule,
        actor,
        manifest,
        reset_optimizer=True,
    ) == len(chunks)
    assert not finished
    with pytest.raises(RuntimeError, match="out of order"):
        runtime.apply_chunked_hybrid_state(actor, 1, chunks[1])
    with pytest.raises(RuntimeError, match="missing state chunks"):
        runtime.finish_chunked_hybrid_apply(_TrainableStateModule, actor)
    assert not finished

    assert runtime.apply_chunked_hybrid_state(actor, 0, chunks[0]) == 2
    with pytest.raises(RuntimeError, match="out of order"):
        runtime.apply_chunked_hybrid_state(actor, 0, chunks[0])
    assert runtime.apply_chunked_hybrid_state(actor, 1, chunks[1]) == 2
    assert not finished
    assert runtime.finish_chunked_hybrid_apply(
        _TrainableStateModule,
        actor,
    ) == len(specs)
    assert finished == [{"policy_version": 11, "reset_optimizer": True}]
    assert applied == list(manifest.chunk_tensor_names)
    assert not hasattr(actor, "_yeto_chunk_apply")


@pytest.mark.parametrize(
    ("corruption", "pattern"),
    (("nan", "contains NaN or Inf"), ("shape", "wrong shape")),
)
def test_chunk_apply_broadcasts_rank_zero_validation_failure(
    monkeypatch, corruption, pattern
):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    monkeypatch.setattr(runtime, "NUM_LAYERS", 1)
    tensors = _tiny_hybrid_tensors()
    specs = tuple(
        SimpleNamespace(name=name, shape=tuple(tensor.shape))
        for name, tensor in sorted(tensors.items())
    )
    state = make_hybrid_trainable_state(_TrainableStateModule, 1, tensors)
    ray = _FakeRay()
    manifest, refs = runtime._chunk_state_for_ray(
        state,
        ray_module=ray,
        max_chunk_bytes=16,
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(yeto_rl_expected_specs=specs),
    )
    runtime.begin_chunked_hybrid_apply(
        _TrainableStateModule,
        actor,
        manifest,
        reset_optimizer=False,
    )
    corrupt = dict(ray.objects[refs[0].index])
    first_name = next(iter(corrupt))
    if corruption == "nan":
        corrupt[first_name] = torch.tensor([float("nan"), 0.0])
    else:
        corrupt[first_name] = corrupt[first_name].reshape(2, 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda _status, *, src: None,
    )
    with pytest.raises(RuntimeError, match=pattern):
        runtime.apply_chunked_hybrid_state(actor, 0, corrupt)
    assert actor._yeto_chunk_apply.next_chunk == 0


def test_hybrid_state_accepts_attention_lora_and_exact_expert_full_contract(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "16")
    tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.ones(1, dtype=torch.float32),
    }
    for layer in range(43):
        for expert in range(256, 272):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                tensors[
                    "base_model.model.model.layers."
                    f"{layer}.mlp.experts.{expert}.{projection}.weight"
                ] = torch.ones(1, dtype=torch.float32)

    state = make_hybrid_trainable_state(
        _TrainableStateModule,
        3,
        tensors,
        train_seconds=1.5,
    )

    assert state.policy_version == 3
    assert state.tensors is not tensors
    assert state.tensors[next(iter(tensors))] is tensors[next(iter(tensors))]
    assert state.train_seconds == 1.5
    assert state.layout_hash == _TrainableStateModule._layout_hash(tensors)

    invalid = dict(tensors)
    invalid[_expert_name(272)] = torch.ones(1)
    with pytest.raises(ValueError, match="outside the selected clone policy"):
        make_hybrid_trainable_state(_TrainableStateModule, 3, invalid)


def test_lora_factory_is_wrapped_for_attention_plus_expert_full(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "16")

    class LoraModule:
        @staticmethod
        def create_lora_instance(args):
            return SimpleNamespace(marker=args.marker)

    install_on_lora_utils(LoraModule)
    result = LoraModule.create_lora_instance(SimpleNamespace(marker="attention"))

    assert result.marker == "attention"
    assert result._configure_kwargs["expert_count"] == 16
    assert LoraModule._yeto_expert_full_installed


def test_arguments_hook_only_restores_the_required_distributed_optimizer():
    class ArgumentsModule:
        @staticmethod
        def set_default_megatron_args(args):
            return args

    args = SimpleNamespace(
        optimizer="adam",
        use_distributed_optimizer=False,
        accumulate_allreduce_grads_in_fp32=True,
        optimizer_cpu_offload=True,
    )

    install_on_arguments(ArgumentsModule)
    result = ArgumentsModule.set_default_megatron_args(args)

    assert result.use_distributed_optimizer is True
    assert result.accumulate_allreduce_grads_in_fp32 is True
    assert result.optimizer_cpu_offload is True


def test_attention_mapping_retains_remote_pipeline_sides(monkeypatch):
    names = (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight",
    )
    local = torch.nn.Parameter(torch.ones(1))

    def side(name, parameter):
        return SimpleNamespace(
            mapping=SimpleNamespace(hf_param=name),
            param_weight=parameter,
        )

    tasks = {
        "local": [
            SimpleNamespace(
                linear_in_task=side(names[0], local),
                linear_out_task=side("ignored", None),
            )
        ],
        "remote": [
            SimpleNamespace(
                linear_in_task=side(names[1], None),
                linear_out_task=side("ignored", None),
            )
        ],
    }
    bridge = SimpleNamespace(
        _model_bridge=SimpleNamespace(
            build_adapter_conversion_tasks=lambda _model: tasks
        )
    )
    monkeypatch.setattr(runtime, "_actor_bridge", lambda _actor: bridge)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=tuple(
                SimpleNamespace(name=name) for name in names
            )
        ),
        model=[],
    )

    sides = runtime._attention_sides(actor)

    assert set(sides) == set(names)
    assert sides[names[0]].param_weight is local
    assert sides[names[1]].param_weight is None


def test_attention_export_unions_pipeline_stage_tensors(monkeypatch):
    names = (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight",
    )
    local = torch.tensor([1.001, 2.003], dtype=torch.float32)
    remote = torch.tensor([3.005, 4.007], dtype=torch.float32)
    local_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    local_parameter.main_param = local
    calls = []

    class Mapping:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def megatron_to_hf(self, parameter, module):
            calls.append((self.name, parameter, module))
            return {self.name: self.value}

    local_module = object()
    sides = {
        names[0]: SimpleNamespace(
            mapping=Mapping(names[0], local),
            param_weight=local_parameter,
            megatron_module=local_module,
        ),
        names[1]: SimpleNamespace(
            mapping=Mapping(names[1], remote),
            param_weight=None,
            megatron_module=None,
        ),
    }

    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=tuple(
                SimpleNamespace(name=name, shape=(2,)) for name in names
            )
        ),
        model=[],
    )
    monkeypatch.setattr(runtime, "_attention_sides", lambda _actor: sides)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    def all_gather_object(output, local_meta):
        output[:] = [
            local_meta,
            {names[1]: ((2,), "torch.float32")},
        ]

    def broadcast(value, *, src):
        if src == 1:
            value.copy_(remote)

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    original_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda *args, **kwargs: original_device("cpu")
        if args and args[0] == "cuda"
        else original_device(*args, **kwargs),
    )

    tensors = runtime._export_attention(actor, retain=True)

    assert [call[0] for call in calls] == list(names)
    assert calls[0][1].data_ptr() == local.data_ptr()
    assert calls[0][2] is local_module
    assert calls[1][1:] == (None, None)
    assert set(tensors) == set(names)
    assert torch.equal(tensors[names[0]], local.float())
    assert torch.equal(tensors[names[1]], remote.float())


def test_attention_export_uses_global_name_on_later_pipeline_stage(monkeypatch):
    name = "base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight"
    value = torch.tensor([5.001, 6.003], dtype=torch.float32)
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    parameter.main_param = value

    class Mapping:
        megatron_param = "decoder.layers.0.self_attention.q_proj.linear_in.weight"

        @staticmethod
        def megatron_to_hf(parameter, module):
            assert parameter.data_ptr() == value.data_ptr()
            assert module == "local-stage-1"
            return {name: value}

    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=(SimpleNamespace(name=name, shape=(2,)),)
        ),
        model=[],
    )
    side = SimpleNamespace(
        mapping=Mapping(),
        param_weight=parameter,
        megatron_module="local-stage-1",
    )
    monkeypatch.setattr(runtime, "_attention_sides", lambda _actor: {name: side})
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local_meta: output.__setitem__(0, local_meta),
    )
    monkeypatch.setattr(torch.distributed, "broadcast", lambda value, *, src: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    original_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda *args, **kwargs: original_device("cpu")
        if args and args[0] == "cuda"
        else original_device(*args, **kwargs),
    )

    tensors = runtime._export_attention(actor, retain=True)

    assert set(tensors) == {name}
    assert torch.equal(tensors[name], value.float())


def test_attention_export_retains_rank_zero_pp_broadcast(monkeypatch):
    name = "base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight"
    value = torch.tensor([7, 8], dtype=torch.bfloat16)

    class Mapping:
        @staticmethod
        def megatron_to_hf(parameter, module):
            assert parameter is None
            assert module is None
            return {name: value}

    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=(SimpleNamespace(name=name, shape=(2,)),)
        ),
        model=[],
    )
    side = SimpleNamespace(
        mapping=Mapping(),
        param_weight=None,
        megatron_module=None,
    )
    monkeypatch.setattr(runtime, "_attention_sides", lambda _actor: {name: side})
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local_meta: output.__setitem__(0, local_meta),
    )
    monkeypatch.setattr(torch.distributed, "broadcast", lambda value, *, src: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    original_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda *args, **kwargs: original_device("cpu")
        if args and args[0] == "cuda"
        else original_device(*args, **kwargs),
    )

    tensors = runtime._export_attention(actor, retain=True)

    assert set(tensors) == {name}
    assert torch.equal(tensors[name], value.float())


def test_attention_export_validates_dp_replica_on_non_root_stage(monkeypatch):
    names = (
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight",
    )
    values = (
        torch.tensor([1, 2], dtype=torch.float32),
        torch.tensor([3, 4], dtype=torch.float32),
    )
    local_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    local_parameter.main_param = values[1]

    class Mapping:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def megatron_to_hf(self, _parameter, _module):
            return {self.name: self.value}

    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=tuple(
                SimpleNamespace(name=name, shape=(2,)) for name in names
            )
        ),
        model=[],
    )
    sides = {
        names[0]: SimpleNamespace(
            mapping=Mapping(names[0], values[0]),
            param_weight=None,
            megatron_module=None,
        ),
        names[1]: SimpleNamespace(
            mapping=Mapping(names[1], values[1]),
            param_weight=local_parameter,
            megatron_module="local-stage-1",
        ),
    }
    monkeypatch.setattr(runtime, "_attention_sides", lambda _actor: sides)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    def all_gather_object(output, local_meta):
        output[:] = [
            {names[0]: ((2,), "torch.float32")},
            {names[1]: ((2,), "torch.float32")},
            {names[0]: ((2,), "torch.float32")},
            local_meta,
        ]

    def broadcast(value, *, src):
        value.copy_(values[src])

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    original_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda *args, **kwargs: original_device("cpu")
        if args and args[0] == "cuda"
        else original_device(*args, **kwargs),
    )

    assert runtime._export_attention(actor, retain=False) == {}


def test_expert_views_use_global_bridge_layer_on_pipeline_stage(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    canonical_gate = _expert_name(256).replace("layers.0", "layers.22")
    canonical_up = _expert_name(256, "up_proj").replace("layers.0", "layers.22")
    physical_gate = _expert_name(32).replace("layers.0", "layers.22")
    physical_up = _expert_name(32, "up_proj").replace("layers.0", "layers.22")
    parameter = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    parameter.main_param = torch.arange(12).reshape(4, 3).float() + 0.001
    parameter._yeto_expert_full = True
    parameter._yeto_expert_id = 256
    parameter._yeto_expert_layer = 0
    parameter._yeto_expert_branch = "linear_fc1"
    chunk = torch.nn.Module()
    chunk.register_parameter("local_stage_expert", parameter)
    task = SimpleNamespace(
        mapping=SimpleNamespace(
            hf_param={"gate": physical_gate, "up": physical_up}
        ),
        param_weight=parameter,
    )
    bridge = SimpleNamespace(get_conversion_tasks=lambda _model: [task])
    monkeypatch.setattr(runtime, "_actor_bridge", lambda _actor: bridge)
    monkeypatch.setattr(runtime, "_attention_sides", lambda _actor: {})
    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_expected_specs=(
                SimpleNamespace(name=canonical_gate),
                SimpleNamespace(name=canonical_up),
            )
        ),
        model=[chunk],
    )

    views = runtime._expert_views(actor)

    expected_gate, expected_up = parameter.main_param.chunk(2, dim=0)
    assert set(views) == {canonical_gate, canonical_up}
    assert torch.equal(views[canonical_gate], expected_gate)
    assert torch.equal(views[canonical_up], expected_up)


def test_apply_on_remote_rank_updates_fp32_masters_under_no_grad(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    attention_name = (
        "base_model.model.model.layers.22.self_attn.q_proj.lora_A.weight"
    )
    expert_name = _expert_name(256).replace("layers.0", "layers.22")
    specs = (
        SimpleNamespace(name=attention_name, shape=(2,)),
        SimpleNamespace(name=expert_name, shape=(2,)),
    )
    incoming = {
        attention_name: torch.tensor([1.001, 2.003], dtype=torch.float32),
        expert_name: torch.tensor([3.005, 4.007], dtype=torch.float32),
    }
    attention_parameter = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.bfloat16)
    )
    attention_parameter.main_param = torch.nn.Parameter(torch.zeros(2))
    expert_parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    expert_parameter.main_param = torch.nn.Parameter(torch.zeros(2))

    class Mapping:
        @staticmethod
        def hf_to_megatron(value, module):
            assert module == "stage-1"
            assert attention_parameter.dtype == torch.float32
            return value

    side = SimpleNamespace(
        mapping=Mapping(),
        param_weight=attention_parameter,
        megatron_module="stage-1",
    )
    copied = []

    class OptimizerChild:
        def __init__(self):
            self.state = {"momentum": object()}

        def _copy_main_params_to_model_params(self):
            copied.append(True)
            with torch.no_grad():
                attention_parameter.copy_(attention_parameter.main_param)
                expert_parameter.copy_(expert_parameter.main_param)

    child = OptimizerChild()
    backups = []
    actor = SimpleNamespace(
        args=SimpleNamespace(yeto_rl_expected_specs=specs),
        optimizer=SimpleNamespace(chained_optimizers=(child,)),
        weights_backuper=SimpleNamespace(backup=backups.append),
    )
    aligned = []
    module = SimpleNamespace(
        _layout_hash=lambda _tensors: "layout",
        _align_scheduler=lambda _actor, version: aligned.append(version),
    )
    monkeypatch.setattr(runtime, "_attention_sides", lambda _actor: {attention_name: side})
    monkeypatch.setattr(
        runtime,
        "_expert_views",
        lambda _actor: {expert_name: expert_parameter.main_param.view(2)},
    )
    monkeypatch.setattr(runtime, "_assert_frozen_experts_unchanged", lambda _actor: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def broadcast_object_list(header, *, src):
        assert src == 0
        header[0] = (7, "layout")

    values = iter(incoming.values())

    def broadcast(value, *, src):
        assert src == 0
        value.copy_(next(values))

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    original_device = torch.device
    monkeypatch.setattr(
        torch,
        "device",
        lambda *args, **kwargs: original_device("cpu")
        if args and args[0] == "cuda"
        else original_device(*args, **kwargs),
    )

    reset_count = runtime.apply_hybrid_trainable_state(
        module,
        actor,
        None,
        reset_optimizer=True,
    )

    assert reset_count == 2
    assert torch.equal(attention_parameter.main_param, incoming[attention_name])
    assert torch.equal(expert_parameter.main_param, incoming[expert_name])
    assert attention_parameter.dtype == torch.bfloat16
    assert copied == [True]
    assert aligned == [7]
    assert child.state == {}
    assert backups == ["actor"]
