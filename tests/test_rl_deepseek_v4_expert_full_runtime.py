from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

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
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(255))),
        SimpleNamespace(
            mapping=SimpleNamespace(
                hf_param={
                    "gate": _expert_name(256),
                    "up": _expert_name(256, "up_proj"),
                }
            )
        ),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(271))),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=_expert_name(272))),
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
