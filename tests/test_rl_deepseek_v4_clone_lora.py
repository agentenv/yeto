from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

import yeto.rl.deepseek_v4_clone_lora as clone_lora
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


@pytest.mark.parametrize("expert_parallel_rank", [0, 3, 7])
def test_clone_only_lora_balances_masks_and_logical_ids(expert_parallel_rank):
    model = _Fixture()
    records = configure_clone_only_grouped_lora(
        model,
        expert_parallel_rank=expert_parallel_rank,
        expert_parallel_size=8,
        adapter_type=_FakeGroupedAdapter,
    )

    assert len(records) == 2
    assert {record.trainable_clone_ids for record in records} == {
        tuple(
            range(
                256 + expert_parallel_rank * 4,
                256 + (expert_parallel_rank + 1) * 4,
            )
        )
    }
    assert {record.local_expert_ids for record in records} == {
        (
            *range(expert_parallel_rank * 32, (expert_parallel_rank + 1) * 32),
            *range(
                256 + expert_parallel_rank * 4,
                256 + (expert_parallel_rank + 1) * 4,
            ),
        )
    }
    for adapter in (model.fc1, model.fc2):
        for weight in (adapter.linear_in.weight, adapter.linear_out.weight):
            assert torch.count_nonzero(weight[:32]).item() == 0
            assert torch.all(weight[32:] == 1)
        (adapter.linear_in.weight.sum() + adapter.linear_out.weight.sum()).backward()
        assert torch.count_nonzero(adapter.linear_in.weight.grad[:32]).item() == 0
        assert torch.count_nonzero(adapter.linear_out.weight.grad[:32]).item() == 0
        assert torch.all(adapter.linear_in.weight.grad[32:] == 1)
        assert torch.all(adapter.linear_out.weight.grad[32:] == 1)
    assert_original_expert_lora_zero(model)


def test_original_slices_remain_exact_zero_after_adamw():
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

    assert all(
        torch.count_nonzero(parameter[:32]).item() == 0 for parameter in parameters
    )
    assert all(torch.count_nonzero(parameter[32:]).item() > 0 for parameter in parameters)
    assert_original_expert_lora_zero(model)


_FAKE_EXPERT_LORA = re.compile(
    r"^(?P<prefix>base_model\.model\.model\.layers\."
    r"(?P<layer>\d+)\.mlp\.experts\.)"
    r"(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"lora_(?P<side>A|B)\.weight$"
)


def _fake_miles_trainable_state_module():
    def expert_name(match, expert, projection, side):
        return (
            f"{match.group('prefix')}{expert}.{projection}.lora_{side}.weight"
        )

    return SimpleNamespace(
        _EXPERT_LORA=_FAKE_EXPERT_LORA,
        _expert_name=expert_name,
        _sparse_expert_updates=lambda *_args, **_kwargs: "unpatched",
        _assert_original_packed_masters_zero=lambda *_args, **_kwargs: None,
    )


@pytest.mark.parametrize(
    ("expert_parallel_rank", "logical_clones"),
    [
        (0, tuple(range(256, 260))),
        (7, tuple(range(284, 288))),
    ],
)
def test_miles_sparse_apply_reads_each_ranks_logical_clones(
    expert_parallel_rank,
    logical_clones,
):
    module = _fake_miles_trainable_state_module()
    clone_lora.install_on_trainable_state(module)
    parameter = torch.nn.Parameter(torch.zeros(36, 2, 3))
    parameter.main_param = torch.zeros_like(parameter, dtype=torch.float32)
    side = SimpleNamespace(param_weight=parameter)
    representative = (
        "base_model.model.model.layers.2.mlp.experts.0."
        "down_proj.lora_B.weight"
    )
    tensors = {
        "base_model.model.model.layers.2.mlp.experts."
        f"{expert}.down_proj.lora_B.weight": torch.full((2, 3), expert)
        for expert in logical_clones
    }

    master, original_count, updates = module._sparse_expert_updates(
        representative,
        side,
        tensors,
        expert_parallel_rank=expert_parallel_rank,
        expert_parallel_size=8,
    )

    assert master.data_ptr() == parameter.main_param.data_ptr()
    assert original_count == 32
    assert tuple(index for index, _value in updates) == (32, 33, 34, 35)
    assert tuple(int(value[0, 0]) for _index, value in updates) == logical_clones


@pytest.mark.parametrize("expert_parallel_rank", [0, 7])
def test_miles_zero_check_covers_each_ranks_first_32_physical_slots(
    expert_parallel_rank,
):
    module = _fake_miles_trainable_state_module()
    clone_lora.install_on_trainable_state(module)
    parameter = torch.nn.Parameter(torch.zeros(36, 1, 1))
    parameter.main_param = torch.zeros_like(parameter, dtype=torch.float32)
    parameter.main_param[32:] = 1
    side = SimpleNamespace(param_weight=parameter)
    sides = (("base_model.model.model.layers.2.mlp.experts.0.down_proj.lora_B.weight", side),)

    module._assert_original_packed_masters_zero(
        sides,
        expert_parallel_rank=expert_parallel_rank,
    )
    parameter.main_param[31] = 1
    with pytest.raises(RuntimeError, match="original packed expert"):
        module._assert_original_packed_masters_zero(
            sides,
            expert_parallel_rank=expert_parallel_rank,
        )


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
