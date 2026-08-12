"""Host-only contract tests for the pinned DeepSeek-V4 runtime validator."""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_validator():
    name = "validate_deepseek_v4_bridge_runtime"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "scripts" / f"{name}.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def test_pp2_attention_topology_is_deliberately_two_gpu_only():
    args = validator._args(["--model", "/model", "--pipeline-parallel", "2"])

    assert args.pipeline_parallel == 2
    assert validator._parallel_sizes(2, 2) == (1, 2, 1)
    assert validator._parallel_sizes(8, 1) == (8, 1, 8)
    with pytest.raises(ValueError, match="exactly two ranks"):
        validator._parallel_sizes(4, 2)


def test_pp2_expert_full_component_requires_exact_smoke_topology():
    args = validator._args(
        [
            "--model",
            "/model",
            "--pipeline-parallel",
            "2",
            "--experts",
            "288",
            "--expert-full-count",
            "16",
        ]
    )

    assert args.expert_full_count == 16
    assert validator._parallel_sizes(
        16,
        2,
        expert_full_count=16,
    ) == (8, 2, 8)
    assert validator._pipeline_layer_counts(4, 2) == (2, 2)
    assert validator._pipeline_layer_counts(5, 2) == (3, 2)
    assert validator._pipeline_layer_counts(43, 2) == (22, 21)
    with pytest.raises(ValueError, match="exactly 16 ranks"):
        validator._parallel_sizes(2, 2, expert_full_count=16)
    with pytest.raises(ValueError, match="requires PP2"):
        validator._parallel_sizes(16, 1, expert_full_count=16)


def test_full16_transport_geometry_bounds_each_canonical_owner(monkeypatch):
    calls = []

    def expert_full_specs(model_config, **kwargs):
        calls.append((model_config, kwargs))
        return [
            SimpleNamespace(shape=(4096, 2048))
            for _layer in range(43)
            for _expert in range(16)
            for _projection in range(3)
        ]

    module = ModuleType("yeto.rl.deepseek_v4_expert_full")
    module.expert_full_specs = expert_full_specs
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model_config = object()
    contract = SimpleNamespace(
        selection_sha256="selection",
        selection_contract_sha256="contract",
    )

    total, stage_shards = validator._full16_transport_geometry(
        model_config,
        contract,
    )

    assert total == 69_256_347_648
    assert stage_shards == (8_858_370_048, 8_455_716_864)
    assert calls == [
        (
            model_config,
            {
                "expert_count": 16,
                "expected_selection_sha256": "selection",
                "expected_selection_contract_sha256": "contract",
            },
        )
    ]


def test_pp2_router_modes_include_every_tp_replica():
    base = [(True, False)] * 3 + [(False, True)]

    assert validator._expected_router_modes(
        layers=4,
        pipeline_parallel=1,
        tensor_parallel=8,
    ) == base
    pp2 = validator._expected_router_modes(
        layers=4,
        pipeline_parallel=2,
        tensor_parallel=8,
    )
    assert len(pp2) == 32
    assert pp2.count((True, False)) == 24
    assert pp2.count((False, True)) == 8

    uneven = validator._expected_router_modes(
        layers=5,
        pipeline_parallel=2,
        tensor_parallel=8,
    )
    assert len(uneven) == 40
    assert uneven.count((True, False)) == 24
    assert uneven.count((False, True)) == 16


def test_pp2_expert_full_cli_accepts_ddp_real_optimizer_lifecycle_mode():
    args = validator._args(
        [
            "--model",
            "/model",
            "--pipeline-parallel",
            "2",
            "--experts",
            "288",
            "--expert-full-count",
            "16",
            "--wrap-with-ddp",
            "--build-real-optimizer",
            "--validate-sleep-process-groups",
            "--task-bridge-model",
            "/bf16-model",
        ]
    )

    assert args.layers == 4
    assert args.wrap_with_ddp is True
    assert args.build_real_optimizer is True
    assert args.validate_sleep_process_groups is True
    assert args.task_bridge_model == "/bf16-model"


def test_task_bridge_actor_keeps_provider_and_training_sources_separate():
    provider_bridge = object()
    models = [object()]
    specs = (object(),)

    actor = validator._expert_coverage_actor(
        provider_bridge,
        models,
        specs,
        model_path="/fp8-model",
        task_bridge_model="/bf16-model",
    )

    assert actor.model is models
    assert actor.args.hf_checkpoint == "/fp8-model"
    assert actor.args.ref_load == "/bf16-model"
    assert actor.args.yeto_rl_expected_specs is specs
    assert actor.args.yeto_rl_trust_remote_code is True
    assert not hasattr(actor, "_yeto_expert_full_bridge")

    legacy_actor = validator._expert_coverage_actor(
        provider_bridge,
        models,
        specs,
        model_path="/fp8-model",
        task_bridge_model=None,
    )
    assert legacy_actor._yeto_expert_full_bridge is provider_bridge


def test_task_coverage_resolves_separate_bridge_through_runtime(monkeypatch):
    captured = {}

    class Model:
        @staticmethod
        def parameters():
            return []

        @staticmethod
        def named_parameters():
            return []

    class TaskBridge:
        @staticmethod
        def get_conversion_tasks(models):
            captured["task_models"] = models
            return []

    def actor_bridge(actor):
        captured["actor"] = actor
        assert not hasattr(actor, "_yeto_expert_full_bridge")
        return TaskBridge()

    expert_runtime = ModuleType("yeto.rl.deepseek_v4_expert_full_runtime")
    expert_runtime._actor_bridge = actor_bridge
    expert_runtime.filter_selected_expert_tasks = lambda tasks, expert_count: tasks
    yeto_rl = ModuleType("yeto.rl")
    yeto_rl.deepseek_v4_expert_full_runtime = expert_runtime
    yeto = ModuleType("yeto")
    yeto.rl = yeto_rl

    parallel_state = SimpleNamespace(
        get_pipeline_model_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 8,
        get_expert_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_world_size=lambda: 8,
    )
    megatron_core = ModuleType("megatron.core")
    megatron_core.parallel_state = parallel_state
    megatron = ModuleType("megatron")
    megatron.core = megatron_core
    for name, module in {
        "megatron": megatron,
        "megatron.core": megatron_core,
        "yeto": yeto,
        "yeto.rl": yeto_rl,
        "yeto.rl.deepseek_v4_expert_full_runtime": expert_runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(validator.torch.distributed, "get_rank", lambda: 0)

    provider_bridge = object()
    models = [Model()]
    actor, record = validator._expert_task_coverage_snapshot(
        provider_bridge,
        models,
        {},
        model_path="/fp8-model",
        task_bridge_model="/bf16-model",
    )

    assert captured["actor"] is actor
    assert captured["task_models"] is models
    assert actor.args.hf_checkpoint == "/fp8-model"
    assert actor.args.ref_load == "/bf16-model"
    assert record["flagged_parameters"] == 0
    assert record["task_parameters"] == 0


def test_parallel_coordinates_are_name_only_topology_fields():
    parallel_state = SimpleNamespace(
        get_pipeline_model_parallel_rank=lambda: 1,
        get_pipeline_model_parallel_world_size=lambda: 2,
        get_tensor_model_parallel_rank=lambda: 7,
        get_tensor_model_parallel_world_size=lambda: 8,
        get_expert_model_parallel_rank=lambda: 3,
        get_expert_model_parallel_world_size=lambda: 8,
    )

    assert validator._parallel_coordinates(parallel_state) == {
        "pipeline_rank": 1,
        "pipeline_world_size": 2,
        "tensor_rank": 7,
        "tensor_world_size": 8,
        "expert_rank": 3,
        "expert_world_size": 8,
    }


def test_sleep_process_group_failure_reloads_groups_before_raising(monkeypatch):
    events = []
    state = {"destroyed": False}

    def destroy_process_groups():
        events.append("destroy")
        state["destroyed"] = True

    def reload_process_groups():
        events.append("reload")
        state["destroyed"] = False

    reloadable = ModuleType("miles.utils.reloadable_process_group")
    reloadable.destroy_process_groups = destroy_process_groups
    reloadable.reload_process_groups = reload_process_groups
    miles_utils = ModuleType("miles.utils")
    miles_utils.reloadable_process_group = reloadable
    miles = ModuleType("miles")
    miles.utils = miles_utils

    parallel_state = object()
    megatron_core = ModuleType("megatron.core")
    megatron_core.parallel_state = parallel_state
    megatron = ModuleType("megatron")
    megatron.core = megatron_core

    def expert_views(_actor):
        events.append("expert_views")
        raise RuntimeError(
            "expert-full conversion tasks do not cover local parameters"
        )

    expert_runtime = ModuleType("yeto.rl.deepseek_v4_expert_full_runtime")
    expert_runtime._expert_views = expert_views
    yeto_rl = ModuleType("yeto.rl")
    yeto_rl.deepseek_v4_expert_full_runtime = expert_runtime
    yeto = ModuleType("yeto")
    yeto.rl = yeto_rl

    for name, module in {
        "miles": miles,
        "miles.utils": miles_utils,
        "miles.utils.reloadable_process_group": reloadable,
        "megatron": megatron,
        "megatron.core": megatron_core,
        "yeto": yeto,
        "yeto.rl": yeto_rl,
        "yeto.rl.deepseek_v4_expert_full_runtime": expert_runtime,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def coordinates(_parallel_state):
        return {
            "pipeline_rank": 7 if state["destroyed"] else 1,
            "pipeline_world_size": 16 if state["destroyed"] else 2,
            "tensor_rank": 7,
            "tensor_world_size": 16 if state["destroyed"] else 8,
            "expert_rank": 7,
            "expert_world_size": 16 if state["destroyed"] else 8,
        }

    def snapshot(*_args, **_kwargs):
        events.append("snapshot")
        return object(), {
            "world_rank": 7,
            **coordinates(parallel_state),
            "flagged_parameters": 1,
            "selected_tasks": 0,
            "task_parameters": 0,
            "optimizer_master_parameters": 1,
            "missing_parameters": ["decoder.layers.0.mlp.experts.weight1"],
            "missing_task_mappings": {},
            "unexpected_task_parameters": 0,
        }

    def recovered(*_args, **_kwargs):
        events.append("recovery")
        assert state["destroyed"] is False
        return {"status": "ok"}

    monkeypatch.setattr(validator, "_parallel_coordinates", coordinates)
    monkeypatch.setattr(validator, "_expert_task_coverage_snapshot", snapshot)
    monkeypatch.setattr(validator, "_pp2_expert_task_coverage", recovered)
    monkeypatch.setattr(validator.torch.distributed, "get_rank", lambda: 7)
    monkeypatch.setattr(validator.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        validator.torch.distributed,
        "all_gather_object",
        lambda output, value: output.__setitem__(0, value),
    )

    with pytest.raises(AssertionError, match="groups were destroyed"):
        validator._sleep_process_group_task_coverage(
            object(),
            [object()],
            {},
            pipeline_layer_counts=(2, 2),
        )

    assert state["destroyed"] is False
    assert events == [
        "destroy",
        "snapshot",
        "expert_views",
        "reload",
        "recovery",
    ]


def test_expert_full_component_requires_runtime_hooks_before_python_start(
    monkeypatch,
):
    args = Namespace(
        expert_full_count=16,
        expert_full_lr=1e-6,
        pipeline_parallel=2,
        experts=288,
    )
    for name in (
        "YETO_DSV4_EXPERT_CLONE",
        "YETO_DSV4_EXPERT_FULL",
        "YETO_DSV4_EXPERT_FULL_COUNT",
        "YETO_DSV4_EXPERT_FULL_LR",
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="environment mismatch"):
        validator._validate_expert_full_component_environment(args)

    monkeypatch.setenv("YETO_DSV4_EXPERT_CLONE", "1")
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL", "1")
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "16")
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_LR", "1e-06")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")
    validator._validate_expert_full_component_environment(args)


def test_real_optimizer_builder_uses_production_distributed_contract(monkeypatch):
    captured = {}

    class OptimizerConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs
            self.__dict__.update(kwargs)

    optimizer_module = ModuleType("megatron.core.optimizer")
    optimizer_module.OptimizerConfig = OptimizerConfig
    core_module = ModuleType("megatron.core")
    core_module.optimizer = optimizer_module
    megatron_module = ModuleType("megatron")
    megatron_module.core = core_module

    def get_megatron_optimizer(**kwargs):
        captured["call"] = kwargs
        return "optimizer"

    model_module = ModuleType("miles.backends.megatron_utils.model")
    model_module.get_megatron_optimizer = get_megatron_optimizer
    utils_module = ModuleType("miles.backends.megatron_utils")
    utils_module.model = model_module
    backends_module = ModuleType("miles.backends")
    backends_module.megatron_utils = utils_module
    miles_module = ModuleType("miles")
    miles_module.backends = backends_module
    for name, module in {
        "megatron": megatron_module,
        "megatron.core": core_module,
        "megatron.core.optimizer": optimizer_module,
        "miles": miles_module,
        "miles.backends": backends_module,
        "miles.backends.megatron_utils": utils_module,
        "miles.backends.megatron_utils.model": model_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    models = [object()]
    result = validator._build_real_optimizer(
        models,
        Namespace(expert_full_lr=1e-6),
    )

    assert result == "optimizer"
    assert captured["config_kwargs"] == {
        "optimizer": "adam",
        "lr": 1e-6,
        "min_lr": 1e-6,
        "weight_decay": 0.0,
        "use_distributed_optimizer": True,
        "bf16": True,
        "clip_grad": 1.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "adam_eps": 1e-8,
    }
    assert captured["call"]["model_chunks"] is models
    assert captured["call"]["use_gloo_process_groups"] is False
    assert captured["call"]["config"].timers is None


def test_local_expert_task_coverage_exposes_missing_flagged_parameter():
    covered = SimpleNamespace(_yeto_expert_full=True)
    missing = SimpleNamespace(_yeto_expert_full=True)

    class Model:
        @staticmethod
        def named_parameters():
            return [("covered", covered), ("missing", missing)]

    selected = [
        SimpleNamespace(param_weight=covered),
        SimpleNamespace(param_weight=None),
    ]
    omitted = SimpleNamespace(
        param_weight=missing,
        mapping=SimpleNamespace(
            hf_param="model.layers.2.mlp.experts.32.down_proj.weight"
        ),
    )
    record = validator._local_expert_task_coverage(
        [Model()],
        selected,
        all_tasks=[*selected, omitted],
    )

    assert record == {
        "flagged_parameters": 2,
        "selected_tasks": 2,
        "task_parameters": 1,
        "missing_parameters": ["missing"],
        "missing_task_mappings": {
            "missing": ["model.layers.2.mlp.experts.32.down_proj.weight"]
        },
        "unexpected_task_parameters": 0,
    }


def test_pp2_runtime_targets_require_pipeline_local_wildcards():
    wildcard = [
        "decoder.layers.*.self_attention.linear_q_down_proj",
        "decoder.layers.*.self_attention.indexer.linear_wq_b",
    ]

    assert validator._runtime_targets(
        wildcard,
        layers=4,
        pipeline_parallel=2,
    ) == wildcard
    with pytest.raises(ValueError, match="pipeline-local wildcards"):
        validator._runtime_targets(
            ["decoder.layers.22.self_attention.indexer.linear_wq_b"],
            layers=4,
            pipeline_parallel=2,
        )


def test_pp1_runtime_target_selection_is_unchanged():
    targets = [
        "decoder.layers.0.self_attention.linear_q",
        "decoder.layers.3.self_attention.linear_q",
        "decoder.layers.4.self_attention.linear_q",
    ]

    assert validator._runtime_targets(
        targets,
        layers=4,
        pipeline_parallel=1,
    ) == targets[:2]
