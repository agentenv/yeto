"""DeepSeek V4 expert-full patches on the ports Miles layout (P0 1.7b.2)."""

from __future__ import annotations

import os
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles.models.deepseek_v4 import expert_full_runtime, ports_runtime
from yeto.rl.adapters.miles.models.deepseek_v4.expert_clone import (
    EXPERT_PARALLEL_SIZE,
    ORIGINAL_EXPERTS_PER_RANK,
    TRAINING_EXPERTS_PER_RANK,
)

ROOT = Path(__file__).resolve().parents[1]


def _expert_name(expert: int, projection: str = "gate_proj") -> str:
    return f"base_model.model.model.layers.0.mlp.experts.{expert}.{projection}.weight"


def _expert_tasks(expert: int) -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(mapping=SimpleNamespace(
            hf_param={"gate": _expert_name(expert), "up": _expert_name(expert, "up_proj")})),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(expert, "down_proj"))),
    )


def test_lora_bridge_factory_unfreezes_the_selected_clone_experts(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "16")
    seen = []

    class LoraBridge:
        @staticmethod
        def create_lora_instance(args, *, target_modules):
            seen.append(target_modules)
            return SimpleNamespace(marker=args.marker)

    ports_runtime.install_on_lora_bridge(LoraBridge)
    ports_runtime.install_on_lora_bridge(LoraBridge)  # idempotent: one wrapper
    lora = LoraBridge.create_lora_instance(SimpleNamespace(marker="attention"), target_modules=["q"])

    assert seen == [["q"]]
    assert lora.marker == "attention"
    assert lora._configure_kwargs == {"expert_count": 16}


def test_model_optimizer_gets_the_expert_learning_rate_group(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_LR", "2e-6")
    optimizer_module = ModuleType("megatron.core.optimizer")
    optimizer_module.get_standard_config_overrides = lambda _config: {}
    config_module = ModuleType("megatron.core.optimizer.optimizer_config")
    config_module.ParamKey = lambda *, attr: attr
    monkeypatch.setitem(sys.modules, optimizer_module.__name__, optimizer_module)
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setattr(expert_full_runtime, "_validate_complete_optimizer_masters", lambda *_: None)
    calls = []

    class Model:
        @staticmethod
        def get_megatron_optimizer(**kwargs):
            calls.append(kwargs)
            return "optimizer"

    ports_runtime.install_on_model(Model)
    with pytest.raises(RuntimeError, match="replicated FP32"):
        Model.get_megatron_optimizer(config=SimpleNamespace(use_distributed_optimizer=True), model_chunks=[])
    result = Model.get_megatron_optimizer(
        config=SimpleNamespace(use_distributed_optimizer=False), model_chunks=[], use_gloo_process_groups=False
    )

    assert result == "optimizer"
    assert calls[0]["config_overrides"] == {"_yeto_expert_full": {"max_lr": 2e-6, "min_lr": 2e-6}}
    assert calls[0]["use_gloo_process_groups"] is False


class _TaskBridge:
    def __init__(self, tasks):
        self.tasks = tasks
        self.exported_tasks = ()

    def get_conversion_tasks(self, _model):
        return self.tasks

    def export_hf_weights(self, _model, **kwargs):
        self.exported_tasks = tuple(kwargs["conversion_tasks"])
        self.export_kwargs = kwargs
        return (
            (_expert_name(expert, projection), torch.ones(1), (f"megatron.{expert}.{projection}",))
            for expert in range(256, 288)
            for projection in ("gate_proj", "up_proj", "down_proj")
        )


def _iterator(monkeypatch, task_bridge):
    monkeypatch.setattr(expert_full_runtime, "NUM_LAYERS", 1)
    monkeypatch.setattr(expert_full_runtime, "weight_iterator_task_bridge", lambda _iterator: task_bridge)
    monkeypatch.setattr(ports_runtime, "weight_iterator_task_bridge", lambda _iterator: task_bridge)

    class Iterator:
        model = object()

        def _iter_hf_param_units(self, _weights, *, materialize):
            raise AssertionError("the stock base iterator must be replaced")

        @staticmethod
        def _source_name_kwargs(_export_fn):
            return {"with_megatron_names": True}

        def _postprocess_and_quantize(self, named_weights, weight_type):
            assert weight_type == "base"
            return named_weights

    module = SimpleNamespace(
        HfWeightIteratorBridge=Iterator,
        strip_param_name_prefix=lambda name: name,
        megatron_bridge_utils=SimpleNamespace(patch_megatron_model=lambda _model: nullcontext()),
        _process_conversion_tasks=lambda selected, _weights: list(selected),
        is_lora_weight_name=lambda name: ".lora_" in name,
    )
    ports_runtime.install_on_weight_iterator(module)
    return Iterator()


def _physical_expert(task) -> int:
    names = task.mapping.hf_param
    name = names["gate"] if isinstance(names, dict) else names
    return int(name.split(".experts.")[1].split(".")[0])


def _rank_tasks(rank: int):
    return [
        task
        for expert in range(rank * TRAINING_EXPERTS_PER_RANK, (rank + 1) * TRAINING_EXPERTS_PER_RANK)
        for task in _expert_tasks(expert)
    ]


def test_publish_exports_only_the_selected_clone_experts(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "5")
    task_bridge = _TaskBridge(_rank_tasks(EXPERT_PARALLEL_SIZE - 1))
    iterator = _iterator(monkeypatch, task_bridge)

    units = list(iterator._iter_hf_param_units({}, materialize=True))

    offsets = {
        _physical_expert(task) % TRAINING_EXPERTS_PER_RANK for task in task_bridge.exported_tasks
    }
    assert offsets == {ORIGINAL_EXPERTS_PER_RANK + offset for offset in range(4)}
    assert task_bridge.export_kwargs["merge_adapter_weights"] is False
    assert task_bridge.export_kwargs["with_megatron_names"] is True
    assert {name for unit in units for name, _weight in unit} == {
        _expert_name(expert, projection)
        for expert in range(256, 261)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def test_non_sender_ranks_run_the_export_without_yielding(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "5")
    task_bridge = _TaskBridge(_rank_tasks(0))
    iterator = _iterator(monkeypatch, task_bridge)

    assert list(iterator._iter_hf_param_units({}, materialize=False)) == []
    assert len(task_bridge.exported_tasks) == expert_full_runtime.expected_expert_task_count(5)


def test_publish_refuses_an_unexpected_expert_task_count(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "5")
    iterator = _iterator(monkeypatch, _TaskBridge(_rank_tasks(0)[:-1]))

    with pytest.raises(RuntimeError, match="conversion tasks"):
        list(iterator._iter_hf_param_units({}, materialize=True))


def test_install_arms_only_ports_module_names(monkeypatch):
    armed = []
    monkeypatch.setattr(ports_runtime, "install_or_defer", lambda name, installer: armed.append(name))

    ports_runtime.install()

    assert armed == [
        "miles.backends.megatron_utils.lora.bridge",
        "miles.backends.megatron_utils.model",
        "miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge",
    ]


@pytest.mark.parametrize(("flag", "expected"), [
    ("YETO_DSV4_EXPERT_FULL_PORTS", "ports"),
    ("YETO_DSV4_EXPERT_FULL", "legacy"),
])
def test_sitecustomize_picks_the_runtime_for_the_miles_layout(monkeypatch, flag, expected):
    calls = []
    monkeypatch.setattr(ports_runtime, "install", lambda: calls.append("ports"))
    monkeypatch.setattr(expert_full_runtime, "install", lambda: calls.append("legacy"))
    for name in ("YETO_DSV4_EXPERT_CLONE", "YETO_DSV4_CLONE_ONLY_LORA", "YETO_DSV4_EXPERT_FULL",
                 "YETO_DSV4_EXPERT_FULL_PORTS", "YETO_TMS_POST_PAUSE_IDLE_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(flag, "1")

    runpy.run_path(str(ROOT / "sitecustomize.py"), run_name="_sitecustomize_dsv4_test")

    assert calls == [expected]
    assert os.environ[flag] == "1"


@pytest.mark.parametrize("helper", [
    "miles.backends.megatron_utils.bridge_lora_helpers",
    "miles.backends.megatron_utils.lora.bridge",
])
def test_v4_bridge_registers_in_the_lora_bridge_module_of_either_layout(monkeypatch, helper):
    from yeto.rl.adapters.miles.models.deepseek_v4 import bridge

    installed = []
    monkeypatch.setattr(bridge, "install_on_miles_bridge_helpers", installed.append)
    module = ModuleType(helper)
    monkeypatch.setitem(sys.modules, helper, module)

    bridge.install_deepseek_v4_actor_bridge_hook()

    assert installed == [module]


def test_sitecustomize_refuses_both_expert_full_runtimes(monkeypatch):
    for name in ("YETO_DSV4_EXPERT_CLONE", "YETO_DSV4_CLONE_ONLY_LORA", "YETO_TMS_POST_PAUSE_IDLE_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL", "1")
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_PORTS", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        runpy.run_path(str(ROOT / "sitecustomize.py"), run_name="_sitecustomize_both_test")
