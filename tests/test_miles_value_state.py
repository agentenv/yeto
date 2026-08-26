from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.megatron.miles_value_state import (
    inverse_tp_partition,
    normalize_miles_partition,
    reconstruct_fp32_from_bf16_remainder,
    select_master_param_storage,
    split_fp32_to_bf16_remainder,
)


def _miles_gather_with_stride(
    shards: list[torch.Tensor], partition_dim: int, partition_stride: int
) -> torch.Tensor:
    """Literal CPU reference for Miles update_weight/common.py."""
    if partition_stride == 1:
        return torch.cat(shards, dim=partition_dim)
    chunks_per_rank = [
        shard.chunk(partition_stride, dim=partition_dim) for shard in shards
    ]
    interleaved = [
        chunks_per_rank[rank][stride]
        for stride in range(partition_stride)
        for rank in range(len(shards))
    ]
    return torch.cat(interleaved, dim=partition_dim)


@pytest.mark.parametrize(
    ("shape", "partition_dim", "partition_stride"),
    [
        ((16, 7), 0, 1),
        ((7, 16), 1, 1),
        ((32, 7), 0, 2),
        ((7, 32), 1, 2),
        ((3, 32, 5), -2, 2),
    ],
)
def test_inverse_tp_partition_round_trips_miles_gather(
    shape, partition_dim, partition_stride
):
    full = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(
        shape
    )
    shards = [
        inverse_tp_partition(
            full,
            tp_rank=rank,
            tp_size=4,
            partition_dim=partition_dim,
            partition_stride=partition_stride,
        )
        for rank in range(4)
    ]

    restored = _miles_gather_with_stride(shards, partition_dim, partition_stride)

    assert torch.equal(restored, full)
    assert all(shard.is_contiguous() for shard in shards)


def test_stride_two_assigns_gate_and_up_chunks_to_same_tp_rank():
    # Canonical fused FC1 ordering is gate TP0..3 followed by up TP0..3.
    full = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    shards = [
        inverse_tp_partition(
            full,
            tp_rank=rank,
            tp_size=4,
            partition_dim=0,
            partition_stride=2,
        )
        for rank in range(4)
    ]

    for rank, shard in enumerate(shards):
        expected = torch.cat((full[rank : rank + 1], full[4 + rank : 5 + rank]))
        assert torch.equal(shard, expected)


def test_normalize_miles_partition_matches_fc1_and_fc2_rules():
    assert normalize_miles_partition(
        "decoder.layers.0.mlp.linear_fc1.weight",
        swiglu=True,
        partition_stride=1,
        partition_dim=0,
    ) == (2, 0)
    assert normalize_miles_partition(
        "decoder.layers.0.mlp.linear_fc1.weight",
        swiglu=True,
        partition_stride=2,
        partition_dim=0,
    ) == (2, 0)
    assert normalize_miles_partition(
        "decoder.layers.0.mlp.linear_fc2.weight",
        swiglu=True,
        partition_stride=1,
        partition_dim=0,
    ) == (1, 1)
    assert normalize_miles_partition(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        swiglu=True,
        partition_stride=1,
        partition_dim=0,
    ) == (1, 0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tp_rank": -1}, "tp_rank"),
        ({"tp_rank": 4}, "tp_rank"),
        ({"tp_rank": True}, "tp_rank"),
        ({"tp_size": 0}, "tp_size"),
        ({"tp_size": 4.0}, "tp_size"),
        ({"partition_dim": 2}, "partition_dim"),
        ({"partition_dim": -3}, "partition_dim"),
        ({"partition_stride": 0}, "partition_stride"),
        ({"partition_stride": 3}, "partition_stride"),
    ],
)
def test_inverse_tp_partition_rejects_invalid_metadata(kwargs, error):
    arguments = {
        "tp_rank": 0,
        "tp_size": 4,
        "partition_dim": 0,
        "partition_stride": 1,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=error):
        inverse_tp_partition(torch.zeros(16, 4), **arguments)


def test_inverse_tp_partition_rejects_ambiguous_tensor_layouts():
    with pytest.raises(ValueError, match="contiguous"):
        inverse_tp_partition(
            torch.zeros(16, 4).t(),
            tp_rank=0,
            tp_size=4,
            partition_dim=0,
            partition_stride=1,
        )
    with pytest.raises(ValueError, match="divisible"):
        inverse_tp_partition(
            torch.zeros(14, 4),
            tp_rank=0,
            tp_size=4,
            partition_dim=0,
            partition_stride=1,
        )
    with pytest.raises(ValueError, match="at least one dimension"):
        inverse_tp_partition(
            torch.tensor(1.0),
            tp_rank=0,
            tp_size=1,
            partition_dim=0,
            partition_stride=1,
        )


def test_normalize_miles_partition_fails_closed_for_unknown_stride():
    with pytest.raises(ValueError, match="only SwiGLU"):
        normalize_miles_partition(
            "decoder.layers.0.self_attention.linear_qkv.weight",
            swiglu=True,
            partition_stride=2,
            partition_dim=0,
        )
    with pytest.raises(ValueError, match="linear_fc2"):
        normalize_miles_partition(
            "decoder.layers.0.mlp.linear_fc2.weight",
            swiglu=True,
            partition_stride=2,
            partition_dim=0,
        )
    with pytest.raises(ValueError, match="SwiGLU"):
        normalize_miles_partition(
            "decoder.layers.0.mlp.linear_fc1.weight",
            swiglu=True,
            partition_stride=3,
            partition_dim=0,
        )


def _signed_int32_words(words: list[int]) -> torch.Tensor:
    signed = [word if word < (1 << 31) else word - (1 << 32) for word in words]
    return torch.tensor(signed, dtype=torch.int32)


def _signed_int16_words(words: list[int]) -> torch.Tensor:
    signed = [word if word < (1 << 15) else word - (1 << 16) for word in words]
    return torch.tensor(signed, dtype=torch.int16)


def test_te_remainder_split_matches_adam_cu_bit_operations():
    words = [
        0x00000000,  # +0
        0x80000000,  # -0
        0x3F800000,  # +1
        0xBF800000,  # -1
        0x3F807FFF,  # low word below the rounding boundary
        0x3F808000,  # exact boundary: TE increments even high bits (not RNE)
        0x3F80FFFF,
        0x7F7FFFFF,  # finite FP32 rounds to BF16 infinity, but remains recoverable
        0x7FC01234,  # NaN payload
        0xFFFF8000,  # rounded high word wraps modulo 16 bits
    ]
    authoritative = _signed_int32_words(words).view(torch.float32)

    rounded_high, remainder = split_fp32_to_bf16_remainder(authoritative)

    expected_low = [word & 0xFFFF for word in words]
    expected_high = [
        (((word >> 16) & 0xFFFF) + int((word & 0x8000) != 0)) & 0xFFFF
        for word in words
    ]
    assert rounded_high.dtype == torch.bfloat16
    assert remainder.dtype == torch.int16
    assert rounded_high.is_contiguous()
    assert remainder.is_contiguous()
    assert torch.equal(rounded_high.view(torch.int16), _signed_int16_words(expected_high))
    assert torch.equal(remainder, _signed_int16_words(expected_low))

    # 0x3f80 is even. IEEE RNE would leave it unchanged at the exact tie,
    # whereas TE v2.17 adam.cu unconditionally increments it to 0x3f81.
    assert rounded_high.view(torch.int16)[5].item() == 0x3F81


def test_te_remainder_reconstructs_every_fp32_bit_including_specials():
    generator = torch.Generator().manual_seed(20260825)
    random_bits = torch.randint(
        -(1 << 31),
        (1 << 31) - 1,
        (20_000,),
        dtype=torch.int32,
        generator=generator,
    )
    edge_bits = _signed_int32_words(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
            0xFFFFFFFF,
        ]
    )
    source_bits = torch.cat((edge_bits, random_bits)).contiguous()
    authoritative = source_bits.view(torch.float32)

    rounded_high, remainder = split_fp32_to_bf16_remainder(authoritative)
    reconstructed = reconstruct_fp32_from_bf16_remainder(rounded_high, remainder)

    assert torch.equal(reconstructed.view(torch.int32), source_bits)


def test_te_remainder_exhausts_all_low_words_and_rounding_decisions():
    low_words = torch.arange(1 << 16, dtype=torch.int32)
    high_words = torch.tensor([0x0000, 0x3F80, 0x8000, 0xFFFF], dtype=torch.int32)
    source_bits = torch.bitwise_or(
        torch.bitwise_left_shift(high_words[:, None], 16), low_words[None, :]
    ).contiguous()
    authoritative = source_bits.view(torch.float32)

    rounded_high, remainder = split_fp32_to_bf16_remainder(authoritative)
    reconstructed = reconstruct_fp32_from_bf16_remainder(rounded_high, remainder)

    expected_high = torch.bitwise_and(
        high_words[:, None] + low_words[None, :].ge(0x8000), 0xFFFF
    ).to(torch.int16)
    assert torch.equal(rounded_high.view(torch.int16), expected_high)
    assert torch.equal(reconstructed.view(torch.int32), source_bits)


def test_te_remainder_helpers_do_not_mutate_or_alias_authoritative_input():
    authoritative = torch.tensor([1.0001, -3.25, 0.0], dtype=torch.float32)
    before = authoritative.clone()

    rounded_high, remainder = split_fp32_to_bf16_remainder(authoritative)
    rounded_high.zero_()
    remainder.zero_()

    assert torch.equal(authoritative, before)


def test_te_remainder_helpers_fail_closed_on_dtype_shape_and_layout():
    with pytest.raises(TypeError, match="torch.float32"):
        split_fp32_to_bf16_remainder(torch.ones(4, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        split_fp32_to_bf16_remainder(torch.ones(4, 3).t())
    with pytest.raises(ValueError, match="empty"):
        split_fp32_to_bf16_remainder(torch.empty(0, dtype=torch.float32))

    high = torch.ones(4, dtype=torch.bfloat16)
    remainder = torch.zeros(4, dtype=torch.int16)
    with pytest.raises(TypeError, match="bfloat16"):
        reconstruct_fp32_from_bf16_remainder(high.float(), remainder)
    with pytest.raises(TypeError, match="int16"):
        reconstruct_fp32_from_bf16_remainder(high, remainder.int())
    with pytest.raises(ValueError, match="does not match"):
        reconstruct_fp32_from_bf16_remainder(high, remainder.reshape(2, 2))


def _fake_te_optimizer(model_param: torch.Tensor, master_param: torch.Tensor, **flags):
    state = {
        "master_param": master_param,
        "exp_avg": torch.randn(model_param.shape, dtype=torch.float32),
        "exp_avg_sq": torch.rand(model_param.shape, dtype=torch.float32),
        "step": 17,
    }
    optimizer = SimpleNamespace(
        master_weights=flags.get("master_weights", True),
        master_weight_dtype=flags.get("master_weight_dtype", torch.float32),
        store_param_remainders=flags.get("store_param_remainders", False),
        state={model_param: state},
    )
    return optimizer, state


@pytest.mark.parametrize(
    ("store_remainders", "master_dtype", "expected_kind"),
    [
        (False, torch.float32, "fp32_master"),
        (True, torch.int16, "bf16_remainder"),
    ],
)
def test_select_master_storage_uses_effective_te_flags_without_mutating_moments(
    store_remainders, master_dtype, expected_kind
):
    model_param = torch.zeros(8, dtype=torch.bfloat16)
    master_param = torch.zeros(8, dtype=master_dtype)
    optimizer, state = _fake_te_optimizer(
        model_param,
        master_param,
        store_param_remainders=store_remainders,
    )
    exp_avg_before = state["exp_avg"].clone()
    exp_avg_sq_before = state["exp_avg_sq"].clone()
    state_keys_before = tuple(state)

    selected = select_master_param_storage(optimizer, model_param)

    assert selected.kind == expected_kind
    assert selected.tensor is master_param
    assert tuple(state) == state_keys_before
    assert torch.equal(state["exp_avg"], exp_avg_before)
    assert torch.equal(state["exp_avg_sq"], exp_avg_sq_before)
    assert state["step"] == 17


@pytest.mark.parametrize(
    ("flags", "master_dtype", "message"),
    [
        ({"master_weights": False}, torch.float32, "does not own"),
        ({"master_weight_dtype": torch.float16}, torch.float16, "FP32 TE master"),
        ({"store_param_remainders": True}, torch.float32, "torch.int16"),
        ({"store_param_remainders": False}, torch.int16, "FP32 master"),
    ],
)
def test_select_master_storage_rejects_inconsistent_flags_and_state(
    flags, master_dtype, message
):
    model_param = torch.zeros(8, dtype=torch.bfloat16)
    optimizer, _ = _fake_te_optimizer(
        model_param, torch.zeros(8, dtype=master_dtype), **flags
    )

    with pytest.raises(ValueError, match=message):
        select_master_param_storage(optimizer, model_param)


def test_select_master_storage_requires_initialized_matching_state():
    model_param = torch.zeros(8, dtype=torch.bfloat16)
    optimizer, state = _fake_te_optimizer(model_param, torch.zeros(8))

    with pytest.raises(ValueError, match="no initialized state"):
        select_master_param_storage(
            SimpleNamespace(
                master_weights=True,
                master_weight_dtype=torch.float32,
                store_param_remainders=False,
                state={},
            ),
            model_param,
        )
    with pytest.raises(ValueError, match="no initialized 'master_param'"):
        select_master_param_storage(optimizer, model_param, optimizer_state={})
    with pytest.raises(ValueError, match="does not match"):
        select_master_param_storage(
            optimizer,
            model_param,
            optimizer_state={**state, "master_param": torch.zeros(4)},
        )


def test_select_master_storage_rejects_non_bf16_model_param():
    model_param = torch.zeros(8, dtype=torch.float32)
    optimizer, _ = _fake_te_optimizer(model_param, torch.zeros(8))
    with pytest.raises(ValueError, match="BF16 model"):
        select_master_param_storage(optimizer, model_param)
