from __future__ import annotations

import pytest
import torch

from yeto.rl.deepseek_v4_clone_lora import (
    assert_original_expert_lora_zero,
    configure_clone_only_grouped_lora,
    wrap_clone_only_lora,
)


class _FakeGroupedAdapter(torch.nn.Module):
    def __init__(self, base_linear_name: str, local_experts: int = 36):
        super().__init__()
        self.base_linear_name = base_linear_name
        self.num_local_experts = local_experts
        self.linear_in = torch.nn.Module()
        self.linear_out = torch.nn.Module()
        self.linear_in.weight = torch.nn.Parameter(
            torch.ones(local_experts, 3, 5)
        )
        self.linear_out.weight = torch.nn.Parameter(
            torch.ones(local_experts, 7, 3)
        )


class _Fixture(torch.nn.Module):
    def __init__(self, local_experts: int = 36):
        super().__init__()
        self.fc1 = _FakeGroupedAdapter(
            "decoder.layers.2.mlp.experts.linear_fc1", local_experts
        )
        self.fc2 = _FakeGroupedAdapter(
            "decoder.layers.2.mlp.experts.linear_fc2", local_experts
        )
        self.attention = _FakeGroupedAdapter(
            "decoder.layers.2.self_attention.linear_q", local_experts
        )


def test_clone_only_lora_zeros_and_gradient_masks_original_slices():
    model = _Fixture()
    records = configure_clone_only_grouped_lora(
        model,
        expert_parallel_rank=7,
        expert_parallel_size=8,
        adapter_type=_FakeGroupedAdapter,
    )

    assert len(records) == 2
    assert {record.trainable_clone_ids for record in records} == {
        tuple(range(256, 288))
    }
    for adapter in (model.fc1, model.fc2):
        for weight in (adapter.linear_in.weight, adapter.linear_out.weight):
            assert torch.count_nonzero(weight[:4]).item() == 0
            assert torch.all(weight[4:] == 1)
        (adapter.linear_in.weight.sum() + adapter.linear_out.weight.sum()).backward()
        assert torch.count_nonzero(adapter.linear_in.weight.grad[:4]).item() == 0
        assert torch.count_nonzero(adapter.linear_out.weight.grad[:4]).item() == 0
        assert torch.all(adapter.linear_in.weight.grad[4:] == 1)
        assert torch.all(adapter.linear_out.weight.grad[4:] == 1)
    assert_original_expert_lora_zero(model)


def test_non_clone_ep_ranks_remain_exact_zero_after_adamw():
    model = _Fixture()
    configure_clone_only_grouped_lora(
        model,
        expert_parallel_rank=0,
        expert_parallel_size=8,
        adapter_type=_FakeGroupedAdapter,
    )
    parameters = [model.fc1.linear_in.weight, model.fc1.linear_out.weight]
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.1)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()

    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in parameters)
    assert_original_expert_lora_zero(model)


def test_clone_only_lora_rejects_wrong_layout_and_double_install():
    with pytest.raises(RuntimeError, match="expected 36"):
        configure_clone_only_grouped_lora(
            _Fixture(local_experts=35),
            expert_parallel_rank=7,
            expert_parallel_size=8,
            adapter_type=_FakeGroupedAdapter,
        )

    model = _Fixture()
    configure_clone_only_grouped_lora(
        model,
        expert_parallel_rank=7,
        expert_parallel_size=8,
        adapter_type=_FakeGroupedAdapter,
    )
    with pytest.raises(RuntimeError, match="installed twice"):
        configure_clone_only_grouped_lora(
            model,
            expert_parallel_rank=7,
            expert_parallel_size=8,
            adapter_type=_FakeGroupedAdapter,
        )


def test_lora_proxy_configures_transformed_model_and_delegates_methods():
    model = _Fixture()

    class Inner:
        marker = "inner"

        def __call__(self, value, *args, **kwargs):
            return value

        def set_params_to_save(self, value):
            self.saved = value

    proxy = wrap_clone_only_lora(
        Inner(),
        configure_kwargs={
            "expert_parallel_rank": 7,
            "expert_parallel_size": 8,
            "adapter_type": _FakeGroupedAdapter,
        },
    )
    transformed = proxy(model)
    assert transformed is model
    assert proxy.marker == "inner"
    assert len(proxy.clone_records) == 2
    proxy.set_params_to_save(model)
    assert proxy._inner.saved is model
