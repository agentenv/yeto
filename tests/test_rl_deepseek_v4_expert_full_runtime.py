from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from yeto.rl import deepseek_v4_expert_full_runtime as runtime
from yeto.rl.deepseek_v4_expert_full_runtime import (
    filter_selected_expert_tasks,
    install_on_arguments,
    install_on_lora_utils,
    make_hybrid_trainable_state,
    selected_expert_hf_name,
)


def _expert_name(expert: int, projection: str = "gate_proj") -> str:
    return (
        "base_model.model.model.layers.0.mlp.experts."
        f"{expert}.{projection}.weight"
    )


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


def test_expert_views_use_global_bridge_layer_on_pipeline_stage(monkeypatch):
    monkeypatch.setenv("YETO_DSV4_EXPERT_FULL_COUNT", "1")
    canonical_gate = _expert_name(256).replace("layers.0", "layers.22")
    canonical_up = _expert_name(256, "up_proj").replace("layers.0", "layers.22")
    physical_gate = _expert_name(32).replace("layers.0", "layers.22")
    physical_up = _expert_name(32, "up_proj").replace("layers.0", "layers.22")
    parameter = torch.nn.Parameter(torch.arange(12).reshape(4, 3).float())
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

    expected_gate, expected_up = parameter.chunk(2, dim=0)
    assert set(views) == {canonical_gate, canonical_up}
    assert torch.equal(views[canonical_gate], expected_gate)
    assert torch.equal(views[canonical_up], expected_up)
