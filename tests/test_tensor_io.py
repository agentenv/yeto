import struct

import pytest
import torch

from yeto.fragments import build_layout
from yeto.protocol import DTYPE_BF16, DTYPE_F32, DTYPE_Q4, bulk_dtype
from yeto.tensor_io import (
    Q4_BLOCK,
    dequantize_q4,
    fragment_flat,
    pack_tensor,
    quantize_q4,
)


def test_bulk_dtype_maps_q4_to_bf16():
    assert bulk_dtype(DTYPE_Q4) == DTYPE_BF16
    assert bulk_dtype(DTYPE_BF16) == DTYPE_BF16
    assert bulk_dtype(DTYPE_F32) == DTYPE_F32


def test_q4_golden_bytes_match_rust_test_vector():
    # Must stay in sync with syncer/src/protocol.rs q4_golden_vector.
    data = quantize_q4(torch.tensor([1.0, -0.5, 0.25, 0.0]))
    assert len(data) == 4 + Q4_BLOCK // 2
    assert data[:4] == struct.pack("<f", 1.0)  # block scale = absmax
    assert data[4] == 0xE7  # 1.0 -> level 7; -0.5 -> sign|level 6
    assert data[5] == 0x05  # 0.25 -> level 5; 0.0 -> zero
    assert all(b == 0 for b in data[6:])


def test_unquantized_delta_golden_bytes_preserve_sign():
    delta = torch.tensor([-1.5, 2.0], dtype=torch.float32)
    assert pack_tensor(delta, DTYPE_F32) == struct.pack("<2f", -1.5, 2.0)
    assert pack_tensor(delta, DTYPE_BF16) == bytes.fromhex("c0bf0040")


def test_q4_roundtrip_error_bounds():
    torch.manual_seed(0)
    flat = torch.randn(3 * Q4_BLOCK + 17)  # non-multiple of the block size
    back = dequantize_q4(quantize_q4(flat), flat.numel())
    assert back.shape == flat.shape
    for b in range(0, flat.numel(), Q4_BLOCK):
        block = flat[b : b + Q4_BLOCK]
        got = back[b : b + Q4_BLOCK]
        absmax = block.abs().max()
        nonzero = got != 0
        # E3M0 rounds in log space: nearest level is within a factor sqrt(2).
        ratio = (got[nonzero] / block[nonzero]).abs()
        assert ratio.max() <= 2**0.5 + 1e-3
        assert ratio.min() >= 2**-0.5 - 1e-3
        # Values quantized to zero were below the smallest level's midpoint.
        if (~nonzero).any():
            assert block[~nonzero].abs().max() <= absmax * 2**-6.5 + 1e-7


def test_q4_zero_and_constant_blocks():
    assert dequantize_q4(quantize_q4(torch.zeros(Q4_BLOCK)), Q4_BLOCK).eq(0).all()
    const = torch.full((Q4_BLOCK,), -3.0)
    assert dequantize_q4(quantize_q4(const), Q4_BLOCK).eq(const).all()


def test_q4_rejects_bad_length():
    with pytest.raises(ValueError):
        dequantize_q4(b"\x00" * 131, 4)


def test_fragment_flat_matches_layout_order():
    params = {
        "model.layers.0.a.weight": torch.arange(4.0),
        "model.layers.1.b.weight": torch.arange(4.0, 6.0),
    }
    layout = build_layout([(n, p.numel()) for n, p in params.items()], 1)
    frag = layout.fragments[0]
    flat = fragment_flat(frag, params)
    expected = torch.cat([params[n].reshape(-1) for n, _ in frag.tensors])
    assert torch.equal(flat, expected)
