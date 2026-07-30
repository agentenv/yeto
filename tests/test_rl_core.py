from __future__ import annotations

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import yeto.rl.miles as miles
from yeto.fragments import MERGE_AVG
from yeto.rl.core import (
    CanonicalTensorSpec,
    LocalRoundStats,
    StrictRlInvariantError,
    build_avg_layout,
    canonical_layout_hash,
    canonical_state,
    flat_tensor,
    policy_delta,
    tensors_from_flat,
)
from yeto.rl.bridge import BridgeConfig, StrictRlBridge
from yeto.protocol import PullRequest
from yeto.rl.export import adapter_targets, derive_peft_lora_specs
from yeto.rl.miles import MilesIslandRuntime


def tensors():
    return {
        "base_model.model.z.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        "base_model.model.a.lora_A.weight": torch.tensor([[1.0, 2.0]]),
    }


MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64


def state(version, values, **kwargs):
    return canonical_state(
        version,
        values,
        base_model_revision=kwargs.pop("base_model_revision", MODEL_REVISION),
        lora_config_hash=kwargs.pop("lora_config_hash", LORA_CONFIG_HASH),
        **kwargs,
    )


def test_canonical_lora_is_sorted_f32_cpu_and_one_avg_fragment():
    canonical = state(7, tensors())
    assert [spec.name for spec in canonical.specs] == sorted(tensors())
    assert all(value.dtype == torch.float32 for value in canonical.tensors.values())
    assert all(spec.dtype == "float32" for spec in canonical.specs)
    assert canonical.layout_hash == canonical_layout_hash(canonical.specs)
    layout = build_avg_layout(canonical.specs)
    assert len(layout.fragments) == 1
    assert layout.fragments[0].merge_mode == MERGE_AVG


def test_flat_round_trip_and_delta_use_the_exact_tensor_contract():
    base = state(2, tensors())
    flat = flat_tensor(base.tensors, base.specs)
    rebuilt = tensors_from_flat(flat, base.specs)
    assert all(torch.equal(base.tensors[name], rebuilt[name]) for name in rebuilt)
    local = state(
        2,
        {name: value + 0.5 for name, value in base.tensors.items()},
        expected_specs=base.specs,
    )
    assert torch.equal(policy_delta(local, base), torch.full_like(flat, 0.5))


def test_policy_delta_rejects_canonical_identity_mismatch():
    base = state(2, tensors())
    local = state(2, tensors(), base_model_revision="c" * 40)
    with pytest.raises(ValueError, match="identities differ"):
        policy_delta(local, base)
    with pytest.raises(ValueError, match="layout hash changed"):
        state(2, tensors(), layout_hash="d" * 64)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_canonical_lora_rejects_non_finite_values(bad):
    values = tensors()
    values["base_model.model.a.lora_A.weight"][0, 0] = bad
    with pytest.raises(ValueError, match="NaN or Inf"):
        state(0, values)


def test_canonical_lora_rejects_non_peft_names_and_shape_drift():
    with pytest.raises(ValueError, match="canonical PEFT"):
        state(0, {"model.weight": torch.ones(2)})
    canonical = state(0, tensors())
    changed = tensors()
    changed["base_model.model.a.lora_A.weight"] = torch.ones(2, 1)
    with pytest.raises(ValueError, match="names, shapes, or dtypes"):
        state(0, changed, expected_specs=canonical.specs)


def test_avg_layout_rejects_duplicate_names():
    spec = CanonicalTensorSpec(
        "base_model.model.x.lora_A.weight", (1, 2), "float32", 2
    )
    with pytest.raises(ValueError, match="unique"):
        build_avg_layout((spec, spec))


def test_two_model_configs_use_the_same_generic_peft_path(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    configs = (
        transformers.LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        ),
        transformers.OPTConfig(
            vocab_size=32,
            hidden_size=8,
            ffn_dim=16,
            num_hidden_layers=1,
            num_attention_heads=2,
        ),
    )
    layouts = []
    for index, config in enumerate(configs):
        model_dir = tmp_path / str(index)
        config.save_pretrained(model_dir)
        specs = derive_peft_lora_specs(
            str(model_dir),
            None,
            rank=2,
            targets="all-linear",
        )
        layouts.append(build_avg_layout(specs))
        assert specs
    assert all(layout.fragments[0].merge_mode == MERGE_AVG for layout in layouts)


def test_auto_targets_follow_existing_moe_semantics_from_actual_config(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=2,
    )
    model_dir = tmp_path / "moe-marked"
    config.save_pretrained(model_dir)

    specs = derive_peft_lora_specs(
        str(model_dir),
        None,
        rank=2,
        targets="auto",
    )

    assert set(adapter_targets(specs)) == {"q_proj", "k_proj", "v_proj", "o_proj"}


class _VersionedRuntime:
    def __init__(self):
        self.state = state(0, tensors())

    def initialize(self):
        return self.state

    def apply_global_policy(self, state):
        self.state = state

    def run_local_round(self, **kwargs):
        return LocalRoundStats(
            island_id=0,
            local_round_id=1,
            base_policy_version=0,
            active_groups=1,
            completed_groups=1,
            cancelled_groups=0,
            completed_trajectories=1,
            action_tokens=1,
            tool_wait_seconds=0.0,
            group_p50_seconds=0.1,
            group_p95_seconds=0.1,
            group_p99_seconds=0.1,
            reward_mean=1.0,
            reward_std=0.0,
            zero_variance_group_ratio=1.0,
            mean_kl=None,
            ess_ratio=None,
            clip_fraction=0.0,
            delta_l2_norm=0.0,
            rollout_seconds=0.1,
            train_seconds=0.1,
        )

    def export_local_policy(self):
        return state(self.state.policy_version + 1, self.state.tensors)

    def record_local_round(self, stats):
        pass

    def shutdown(self):
        pass


def _bridge_config(tmp_path="/tmp/yeto-rl-test-events.jsonl"):
    canonical = state(0, tensors())
    return BridgeConfig(
        syncer_addr=("127.0.0.1", 1),
        learner_id=0,
        global_rounds=1,
        groups_per_round=1,
        samples_per_group=1,
        local_optimizer_steps=1,
        expected_specs=canonical.specs,
        base_model_revision=canonical.base_model_revision,
        lora_config_hash=canonical.lora_config_hash,
        layout_hash=canonical.layout_hash,
        event_tape=str(tmp_path),
        wan_streams=0,
    )


def test_policy_delta_rejects_exported_policy_version_drift():
    runtime = _VersionedRuntime()
    bridge = StrictRlBridge(runtime, _bridge_config())
    bridge.current = bridge.initial
    with pytest.raises(ValueError, match="versions differ"):
        bridge._run_round(PullRequest(0, 1, 1))


def test_local_round_event_contains_every_init_metric(tmp_path):
    runtime = _VersionedRuntime()
    runtime.export_local_policy = lambda: state(
        0,
        {name: value + 1 for name, value in runtime.state.tensors.items()},
    )
    event_tape = tmp_path / "events.jsonl"
    bridge = StrictRlBridge(runtime, _bridge_config(event_tape))
    bridge.current = bridge.initial
    pushed = []
    bridge.client = SimpleNamespace(push_fragment=lambda *values: pushed.append(values))

    bridge._run_round(PullRequest(0, 1, 1))

    event = json.loads(event_tape.read_text())
    expected = {
        "rl/active_groups",
        "rl/completed_groups",
        "rl/cancelled_groups",
        "rl/completed_trajectories",
        "rl/action_tokens",
        "rl/tool_wait_seconds",
        "rl/rollout_seconds",
        "rl/group_p50_seconds",
        "rl/group_p95_seconds",
        "rl/group_p99_seconds",
        "rl/reward_mean",
        "rl/reward_std",
        "rl/zero_variance_group_ratio",
        "rl/global_policy_version",
        "rl/rollout_policy_version",
        "rl/mixed_version_group_count",
        "rl/local_delta_norm",
        "rl/current_vs_rollout_kl",
        "rl/ess_ratio",
        "rl/clip_fraction",
        "sync/bytes_sent",
    }
    assert expected <= event.keys()
    assert event["sync/bytes_sent"] == 48 + len(pushed[0][-1])


def test_miles_admission_pause_aborts_inflight_rollouts():
    modes = []

    class PauseGeneration:
        async def remote(self, mode):
            modes.append(mode)

    class Engine:
        pause_generation = PauseGeneration()

    runtime = object.__new__(MilesIslandRuntime)

    async def engines():
        return [Engine()]

    runtime._engines = engines
    asyncio.run(runtime._pause_rollout())
    assert modes == ["abort"]


def test_miles_bridge_propagates_attention_backend(monkeypatch):
    class AutoBridge:
        def to_megatron_provider(self):
            return SimpleNamespace(attention_backend="auto")

    megatron = types.ModuleType("megatron")
    bridge = types.ModuleType("megatron.bridge")
    training = types.ModuleType("megatron.bridge.training")
    config = types.ModuleType("megatron.bridge.training.config")

    class DistributedDataParallelConfig:
        def __init__(self, **kwargs):
            self.use_distributed_optimizer = kwargs["use_distributed_optimizer"]

    bridge.AutoBridge = AutoBridge
    config.DistributedDataParallelConfig = DistributedDataParallelConfig
    training.config = config
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training", training)
    monkeypatch.setitem(sys.modules, "megatron.bridge.training.config", config)
    installed = []
    monkeypatch.setattr(miles, "_install_train_metric_capture", lambda: None)
    monkeypatch.setattr(miles, "_install_colocated_lora_ipc_sync", lambda: None)
    monkeypatch.setattr(miles, "install_miles_actor_adapter", lambda: installed.append(True))

    args = SimpleNamespace(
        attention_backend="unfused", use_distributed_optimizer=True
    )
    miles.configure_miles_bridge(args)

    assert AutoBridge().to_megatron_provider().attention_backend == "unfused"
    assert not args.use_distributed_optimizer
    assert not config.DistributedDataParallelConfig(
        use_distributed_optimizer=True
    ).use_distributed_optimizer
    assert installed == [True]


def test_miles_actor_calls_handle_rank0_export_and_all_rank_results():
    runtime = object.__new__(MilesIslandRuntime)
    runtime.args = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=2)

    class Actors:
        def __init__(self, results):
            self.results = results

        async def _broadcast(self, method, *args):
            return self.results

    runtime.actor_model = Actors([{"policy": 1}, None])
    assert asyncio.run(runtime._actor_call("export", rank0=True)) == {"policy": 1}
    runtime.actor_model = Actors([(4, "hash"), (4, "hash")])
    assert asyncio.run(runtime._actor_call("apply")) == (4, "hash")
    runtime.actor_model = Actors([3, 4])
    with pytest.raises(RuntimeError, match="ranks disagree"):
        asyncio.run(runtime._actor_call("steps"))


def test_miles_waits_for_the_complete_multinode_ray_island(monkeypatch):
    canonical = state(0, tensors())
    args = SimpleNamespace(
        actor_num_nodes=2,
        actor_num_gpus_per_node=4,
        offload_train=True,
        yeto_rl_base_model_revision=canonical.base_model_revision,
        yeto_rl_lora_config_hash=canonical.lora_config_hash,
        yeto_rl_layout_hash=canonical.layout_hash,
    )
    resource_counts = iter((4, 8))
    ray = types.ModuleType("ray")
    ray.is_initialized = lambda: True
    ray.cluster_resources = lambda: {"GPU": next(resource_counts)}
    actor = SimpleNamespace()

    async def broadcast(method, *values):
        assert method == "yeto_rl_export_policy"
        return [canonical.tensors] + [None] * 7

    async def onload():
        pass

    async def offload():
        pass

    actor._broadcast = broadcast
    actor.onload = onload
    actor.offload = offload

    async def create_training_models(*values):
        return actor, None

    placement = types.ModuleType("miles.ray.placement_group")
    placement.create_placement_groups = lambda _args: {"rollout": object()}
    placement.create_rollout_manager = lambda *values: (object(), None)
    placement.create_training_models = create_training_models
    external_miles = types.ModuleType("miles")
    ray_package = types.ModuleType("miles.ray")
    ray_package.placement_group = placement
    external_miles.ray = ray_package
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "miles", external_miles)
    monkeypatch.setitem(sys.modules, "miles.ray", ray_package)
    monkeypatch.setitem(sys.modules, "miles.ray.placement_group", placement)
    monkeypatch.setattr(miles, "install_miles_actor_adapter", lambda: None)

    async def no_wait(_seconds):
        pass

    monkeypatch.setattr(miles.asyncio, "sleep", no_wait)
    runtime = MilesIslandRuntime(args)
    try:
        initialized = asyncio.run(runtime._initialize())
    finally:
        runtime.loop.close()

    assert initialized.layout_hash == canonical.layout_hash


def test_miles_keeps_non_offloaded_trainer_resident():
    calls = []

    async def onload():
        calls.append("onload")

    async def offload():
        calls.append("offload")

    runtime = object.__new__(MilesIslandRuntime)
    runtime.args = SimpleNamespace(offload_train=False)
    runtime.actor_model = SimpleNamespace(onload=onload, offload=offload)
    runtime._trainer_awake = True

    asyncio.run(runtime._onload_trainer())
    asyncio.run(runtime._offload_trainer())
    assert runtime._trainer_awake
    assert calls == []


def test_miles_collects_every_data_parallel_rollout_shard(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda value: value))
    runtime = object.__new__(MilesIslandRuntime)
    runtime.args = SimpleNamespace(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        expert_model_parallel_size=2,
    )
    shards = [
        SimpleNamespace(inner={"sample_indices": [0]}),
        SimpleNamespace(inner={"sample_indices": [1]}),
        SimpleNamespace(inner={"sample_indices": [2]}),
        SimpleNamespace(inner={"sample_indices": [3]}),
    ]
    assert runtime._rollout_batches({"data_ref": shards}) == [
        {"sample_indices": [0]},
        {"sample_indices": [1]},
        {"sample_indices": [2]},
        {"sample_indices": [3]},
    ]


def test_miles_rollout_lifecycle_metrics_use_real_task_completion(monkeypatch):
    class Sample:
        def __init__(self, status):
            self.status = SimpleNamespace(value=status)

    class GenerateState:
        def __init__(self):
            self.pendings = set()

        def submit_generate_tasks(self, groups):
            async def finish(group):
                await asyncio.sleep(0)
                return group

            self.pendings.update(asyncio.create_task(finish(group)) for group in groups)

    upstream = types.ModuleType("miles.rollout.sglang_rollout")
    upstream.GenerateState = GenerateState
    rollout = types.ModuleType("miles.rollout")
    rollout.sglang_rollout = upstream
    package = types.ModuleType("miles")
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)

    def generate(*_args, **_kwargs):
        async def run():
            state = GenerateState()
            state.submit_generate_tasks(
                [[Sample("completed")], [Sample("aborted")]]
            )
            await asyncio.gather(*state.pendings)

        asyncio.run(run())
        return object()

    _, lifecycle = miles._run_rollout_with_metrics(
        generate, SimpleNamespace(), 0, object(), False
    )
    assert lifecycle["active"] == 0
    assert lifecycle["peak_active"] == 2
    assert lifecycle["cancelled"] == 1
    assert len(lifecycle["durations"]) == 2


def test_miles_train_metric_capture_returns_rank_zero_values(monkeypatch):
    args = SimpleNamespace()

    def log_train_step(*_values, **_kwargs):
        return {
            "train/train_rollout_kl": 0.1,
            "train/ess_ratio": 0.9,
            "train/pg_clipfrac": 0.2,
        }

    model = types.ModuleType("miles.backends.megatron_utils.model")
    model.log_train_step = log_train_step
    megatron_utils = types.ModuleType("miles.backends.megatron_utils")
    megatron_utils.model = model
    backends = types.ModuleType("miles.backends")
    backends.megatron_utils = megatron_utils
    package = types.ModuleType("miles")
    package.backends = backends
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.backends", backends)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils", megatron_utils)
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.model", model)

    miles._install_train_metric_capture()
    model.log_train_step(args=args)
    actor = SimpleNamespace(args=args)
    assert miles._actor_train_metrics(actor) == {
        "train/train_rollout_kl": 0.1,
        "train/ess_ratio": 0.9,
        "train/pg_clipfrac": 0.2,
    }
    assert miles._actor_train_metrics(actor) is None


def test_miles_keeps_colocated_lora_ipc_storage_alive_until_transfer(monkeypatch):
    events = []
    update = types.ModuleType(
        "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"
    )

    def send(*_values, **_kwargs):
        events.append("send")
        return ["ref"], "storage"

    update._send_to_colocated_engine = send
    common = types.ModuleType("miles.backends.megatron_utils.update_weight.common")

    def check(results, *, is_lora):
        assert results == ["done"] and is_lora
        events.append("check")

    common._check_weight_sync_results = check
    weight = types.ModuleType("miles.backends.megatron_utils.update_weight")
    weight.update_weight_from_tensor = update
    megatron_utils = types.ModuleType("miles.backends.megatron_utils")
    megatron_utils.update_weight = weight
    backends = types.ModuleType("miles.backends")
    backends.megatron_utils = megatron_utils
    package = types.ModuleType("miles")
    package.backends = backends
    ray = types.ModuleType("ray")

    def get(refs):
        assert refs == ["ref"]
        events.append("get")
        return ["done"]

    ray.get = get
    for name, module in {
        "miles": package,
        "miles.backends": backends,
        "miles.backends.megatron_utils": megatron_utils,
        "miles.backends.megatron_utils.update_weight": weight,
        "miles.backends.megatron_utils.update_weight.update_weight_from_tensor": update,
        "miles.backends.megatron_utils.update_weight.common": common,
        "ray": ray,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )

    miles._install_colocated_lora_ipc_sync()
    refs, storage = update._send_to_colocated_engine(
        [], ipc_gather_group="group", lora_config={}
    )

    assert refs == ["ref"] and storage == "storage"
    assert events == ["send", "get", "check", ("barrier", "group")]


def test_miles_round_stats_use_checkpoint_rollout_and_train_metrics(
    tmp_path, monkeypatch
):
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    sglang.srt = srt
    srt.constants = constants
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)

    checkpoint = tmp_path / "island.pt"
    args = SimpleNamespace(
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        advantage_estimator="grpo",
        yeto_rl_model="org/model",
        yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION,
        yeto_rl_data_revision="d" * 40,
        expert_model_parallel_size=1,
        yeto_rl_layout_hash="c" * 64,
        lr=1e-4,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH,
        n_samples_per_prompt=2,
        num_steps_per_rollout=1,
        over_sampling_batch_size=2,
        yeto_rl_reward_sha256="e" * 64,
        rollout_batch_size=2,
        seq_length=128,
        seed=7,
        rollout_max_response_len=16,
        custom_generate_function_path=None,
        use_session_server=False,
        tito_model="default",
        yeto_rl_completed_groups_path=str(checkpoint),
        yeto_rl_learner_id=0,
    )
    torch.save(
        {
            "schema_version": miles._ISLAND_CHECKPOINT_SCHEMA,
            "config": miles._island_checkpoint_config(args),
            "policy_version": 3,
            "rollout_metrics": {
                "active_groups": 3.0,
                "cancelled_groups": 1.0,
                "tool_wait_seconds": 4.0,
                "group_p50_seconds": 5.0,
                "group_p95_seconds": 6.0,
                "group_p99_seconds": 7.0,
            },
        },
        checkpoint,
    )

    class Remote:
        def __init__(self, result=None):
            self.result = result

        async def remote(self, *_args, **_kwargs):
            return self.result

    runtime = object.__new__(MilesIslandRuntime)
    runtime.args = args
    runtime._policy_version = 3
    runtime._rollout_id = 3
    runtime._rollout_offloaded = False
    runtime.rollout_manager = SimpleNamespace(
        generate=Remote(object()),
        offload=Remote(),
    )

    async def train(*_args):
        pass

    runtime.actor_model = SimpleNamespace(train=train)

    async def no_op():
        pass

    runtime._pause_rollout = no_op
    runtime._rollout_batches = lambda _pack: [
        {
            "weight_versions": [["yeto:3"]] * 4,
            "response_lengths": [1, 2, 3, 4],
            "sample_indices": [0, 1, 2, 3],
            "raw_reward": [1.0, 1.0, 0.0, 2.0],
        }
    ]
    optimizer_steps = iter((10, 11))

    async def actor_call(method, *_args, **_kwargs):
        if method == "yeto_rl_optimizer_steps":
            return next(optimizer_steps)
        assert method == "yeto_rl_train_metrics"
        return {
            "train/train_rollout_kl": 0.1,
            "train/ess_ratio": 0.8,
            "train/pg_clipfrac": 0.25,
        }

    runtime._actor_call = actor_call
    stats = asyncio.run(runtime._run_local_round(3, 2, 2, 1))
    assert stats.active_groups == 3
    assert stats.cancelled_groups == 1
    assert stats.tool_wait_seconds == 4.0
    assert stats.zero_variance_group_ratio == 0.5
    assert stats.mean_kl == 0.1
    assert stats.ess_ratio == 0.8
    assert stats.clip_fraction == 0.25


def test_island_checkpoint_restores_only_complete_same_policy_groups(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "island.pt"
    args = SimpleNamespace(
        actor_num_gpus_per_node=1,
        actor_num_nodes=1,
        advantage_estimator="grpo",
        yeto_rl_model="org/model",
        yeto_rl_data="org/data",
        yeto_rl_base_model_revision=MODEL_REVISION,
        yeto_rl_data_revision="d" * 40,
        expert_model_parallel_size=1,
        yeto_rl_layout_hash="c" * 64,
        lr=1e-4,
        yeto_rl_lora_config_hash=LORA_CONFIG_HASH,
        n_samples_per_prompt=2,
        num_steps_per_rollout=1,
        over_sampling_batch_size=2,
        yeto_rl_reward_sha256="e" * 64,
        rollout_batch_size=1,
        seq_length=128,
        seed=7,
        rollout_max_response_len=16,
        custom_generate_function_path=None,
        use_session_server=False,
        tito_model="default",
        yeto_rl_completed_groups_path=str(checkpoint),
    )

    def sample(status, version, index=0):
        value = SimpleNamespace(
            status=SimpleNamespace(value=status),
            weight_versions=[f"yeto:{version}"],
            index=index,
        )
        value.to_dict = lambda: {
            "status": status,
            "weight_versions": [f"yeto:{version}"],
            "index": index,
        }
        return value

    class Sample:
        @staticmethod
        def from_dict(value):
            version = int(value["weight_versions"][0].split(":", 1)[1])
            return sample(value["status"], version, value["index"])

    package = types.ModuleType("miles")
    utils = types.ModuleType("miles.utils")
    sample_types = types.ModuleType("miles.utils.types")
    sample_types.Sample = Sample
    package.utils = utils
    utils.types = sample_types
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.utils", utils)
    monkeypatch.setitem(sys.modules, "miles.utils.types", sample_types)

    complete = [sample("completed", 3), sample("truncated", 3)]
    incomplete = [sample("aborted", 3), sample("completed", 3)]

    class DataSource:
        def __init__(self, buffer=None):
            self.buffer = list(buffer or [])

        def get_samples(self, _count):
            return []

        def add_samples(self, groups):
            self.buffer.extend(groups)

    source = DataSource([complete, incomplete])
    miles._save_completed_groups(args, 3, 4, source, {"reward": 1.5})
    assert source.buffer == [complete]
    payload = torch.load(checkpoint, weights_only=True)
    for name, value in {
        "model": "org/model",
        "dataset": "org/data",
        "seq_length": 128,
        "seed": 7,
    }.items():
        assert payload["config"][name] == value
    assert payload["completed_groups"][0][0]["status"] == "completed"
    assert payload["completed_groups"][0][1]["status"] == "truncated"
    assert "tensors" not in payload and "local_lora" not in payload
    assert not list(tmp_path.glob(".island.pt.tmp-*"))

    restored = DataSource()
    miles._restore_completed_groups(args, 3, restored)
    assert len(restored.buffer) == 1
    args.seed = 8
    incompatible = DataSource()
    miles._restore_completed_groups(args, 3, incompatible)
    assert incompatible.buffer == []
    args.seed = 7
    stale = DataSource()
    miles._restore_completed_groups(args, 4, stale)
    assert stale.buffer == []

    used = [sample("completed", 3, 10), sample("completed", 3, 11)]
    unused = [sample("completed", 3, 20), sample("truncated", 3, 21)]
    incomplete = [sample("aborted", 3, 30), sample("completed", 3, 31)]
    upstream = types.ModuleType("miles.rollout.sglang_rollout")

    def generate(_args, _rollout_id, data_source, evaluation=False):
        assert not evaluation
        miles.queue_completed_groups(
            _args, [used, unused, incomplete], data_source.get_samples
        )
        return SimpleNamespace(samples=[used], metrics={"reward": 2.0})

    upstream.generate_rollout = generate

    class GenerateState:
        def submit_generate_tasks(self, _samples):
            pass

    upstream.GenerateState = GenerateState
    rollout = types.ModuleType("miles.rollout")
    rollout.sglang_rollout = upstream
    package = types.ModuleType("miles")
    package.rollout = rollout
    monkeypatch.setitem(sys.modules, "miles", package)
    monkeypatch.setitem(sys.modules, "miles.rollout", rollout)
    monkeypatch.setitem(sys.modules, "miles.rollout.sglang_rollout", upstream)
    checkpoint.unlink()
    source = DataSource()

    miles.generate_rollout(args, 3, source)

    assert source.buffer == [unused]
    payload = torch.load(checkpoint, weights_only=True)
    assert [sample["index"] for sample in payload["completed_groups"][0]] == [20, 21]
    assert payload["rollout_metrics"] == {
        "reward": 2.0,
        "active_groups": 0.0,
        "cancelled_groups": 0.0,
        "tool_wait_seconds": 0.0,
        "group_p50_seconds": 0.0,
        "group_p95_seconds": 0.0,
        "group_p99_seconds": 0.0,
    }


def test_miles_apply_resets_optimizer_and_restores_scheduler_progress(monkeypatch):
    name = "base_model.model.layer.lora_A.weight"
    parameter = torch.nn.Parameter(torch.zeros(1, 2))
    parameter.main_param = torch.zeros(1, 2, dtype=torch.float32)

    class Mapping:
        def hf_to_megatron(self, value, _module):
            return value

        def megatron_to_hf(self, value, _module):
            return {name: value}

    side = SimpleNamespace(
        mapping=Mapping(),
        megatron_module=None,
        param_weight=parameter,
    )
    optimizer_state = {
        "exp_avg": torch.ones_like(parameter.main_param),
        "exp_avg_sq": torch.ones_like(parameter.main_param),
        "step": torch.tensor(9.0),
    }
    inner = SimpleNamespace(
        param_groups=[{"step": 9}],
        state={parameter.main_param: optimizer_state},
    )
    unrelated = torch.nn.Parameter(torch.ones(1))
    unrelated_state = {"step": torch.tensor(4.0)}
    inner.state[unrelated] = unrelated_state

    def copy_main_to_model():
        parameter.copy_(parameter.main_param)

    child = SimpleNamespace(
        optimizer=inner,
        _copy_main_params_to_model_params=copy_main_to_model,
    )
    optimizer = SimpleNamespace(chained_optimizers=[child])
    scheduler_steps = []

    class Scheduler:
        num_steps = 0

        def step(self, increment):
            scheduler_steps.append(increment)
            self.num_steps += increment

    scheduler = Scheduler()
    backups = []
    applied = torch.tensor([[3.0, 4.0]])
    state_fn = state(2, {name: applied})
    actor = SimpleNamespace(
        args=SimpleNamespace(
            yeto_rl_base_model_revision=MODEL_REVISION,
            yeto_rl_lora_config_hash=LORA_CONFIG_HASH,
            yeto_rl_layout_hash=state_fn.layout_hash,
            global_batch_size=64,
            num_steps_per_rollout=1,
        ),
        model=[SimpleNamespace(start_param_sync=lambda **_kwargs: pytest.fail(
            "replicated LoRA apply must not start distributed optimizer sync"
        ))],
        optimizer=optimizer,
        opt_param_scheduler=scheduler,
        weights_backuper=SimpleNamespace(backup=backups.append),
    )
    monkeypatch.setattr(miles, "_adapter_sides", lambda _actor: [(name, side)])
    cache_releases = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_releases.append(True))

    reset_count, applied_hash = miles._actor_apply_policy(actor, {name: applied}, 2)
    assert actor.optimizer is optimizer
    assert actor.opt_param_scheduler is scheduler
    assert scheduler.num_steps == 128
    assert scheduler_steps == [128]
    assert inner.param_groups[0]["step"] == 9
    assert parameter.main_param not in inner.state
    assert inner.state[unrelated] is unrelated_state
    assert reset_count == 1
    assert applied_hash == miles.policy_hash(state_fn)
    assert torch.equal(parameter.main_param, applied)
    assert torch.equal(parameter, applied)
    assert backups == ["actor"]
    assert cache_releases == [True]

    scheduler.num_steps = 192
    with pytest.raises(RuntimeError, match="ahead of the committed policy"):
        miles._actor_apply_policy(actor, {name: applied}, 2)


def test_miles_apply_hash_mismatch_is_a_strict_failure(monkeypatch):
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    sglang.srt = srt
    srt.constants = constants
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)

    runtime = object.__new__(MilesIslandRuntime)
    runtime._trainer_awake = False
    runtime._rollout_offloaded = True

    async def no_op(*_args, **_kwargs):
        pass

    runtime._pause_rollout = no_op
    runtime.actor_model = SimpleNamespace(onload=no_op)

    async def actor_call(*_args, **_kwargs):
        return 1, "wrong-policy-hash"

    runtime._actor_call = actor_call
    with pytest.raises(StrictRlInvariantError) as failure:
        asyncio.run(runtime._apply_global_policy(state(0, tensors())))
    assert failure.value.metric == "policy_hash_mismatch_after_apply"


def test_miles_policy_apply_event_has_only_namespaced_policy_hash(monkeypatch):
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    sglang.srt = srt
    srt.constants = constants
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)

    async def no_op(*_args, **_kwargs):
        pass

    class Remote:
        async def remote(self, *_args, **_kwargs):
            pass

    state_fn = state(1, tensors())
    events = []
    runtime = object.__new__(MilesIslandRuntime)
    runtime.args = SimpleNamespace(yeto_rl_learner_id=0, offload_train=True)
    runtime._trainer_awake = False
    runtime._rollout_offloaded = True
    runtime._optimizer_reset_count = 0
    runtime._pause_rollout = no_op
    runtime._resume_rollout = no_op
    runtime._set_rollout_version = no_op
    runtime.actor_model = SimpleNamespace(
        onload=no_op,
        offload=no_op,
        update_weights=no_op,
    )
    runtime.rollout_manager = SimpleNamespace(
        onload_weights=Remote(),
        onload_kv=Remote(),
    )

    async def actor_call(*_args, **_kwargs):
        return 2, miles.policy_hash(state_fn)

    runtime._actor_call = actor_call
    runtime._append_event = events.append
    asyncio.run(runtime._apply_global_policy(state_fn))

    assert events[0]["sync/global_policy_hash"] == miles.policy_hash(state_fn)
    assert "policy_hash" not in events[0]
    assert "trainer_ranks" not in events[0]


@pytest.mark.parametrize(
    "versions",
    [["not-a-policy-token"], ["yeto:3", "yeto:4"]],
)
def test_miles_rejects_invalid_or_mixed_rollout_versions(monkeypatch, versions):
    constants = types.ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "graph"
    constants.GPU_MEMORY_TYPE_KV_CACHE = "kv"
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    sglang.srt = srt
    srt.constants = constants
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.constants", constants)

    class Generate:
        async def remote(self, _rollout_id):
            return object()

    runtime = object.__new__(MilesIslandRuntime)
    runtime._policy_version = 3
    runtime._rollout_id = 3
    runtime.rollout_manager = SimpleNamespace(generate=Generate())

    async def no_op():
        pass

    runtime._pause_rollout = no_op
    runtime._rollout_batches = lambda _pack: [
        {"weight_versions": [versions]}
    ]
    with pytest.raises(StrictRlInvariantError) as failure:
        asyncio.run(runtime._run_local_round(3, 1, 1, 1))
    assert failure.value.metric == "mixed_version_group_count"
