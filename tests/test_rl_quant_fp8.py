"""FP8 E4M3 per-channel quantization (S26 E6 tier 2): rule, error bound, reference."""

from __future__ import annotations

import pytest
import torch

from yeto.rl.quant.fp8 import FP8_DTYPE, FP8_MAX, dequantize_fp8_per_channel, quantize_fp8_per_channel


def _weight(rows: int = 16, cols: int = 64, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(rows, cols, generator=g) * torch.logspace(-3, 1, rows).unsqueeze(1)
    return w.to(torch.bfloat16)


def test_scale_is_row_amax_over_448_and_values_follow_the_cast():
    w = _weight()
    q, scale = quantize_fp8_per_channel(w)
    assert q.dtype == FP8_DTYPE and scale.dtype == torch.float32 and scale.shape == (16, 1)
    amax = w.float().abs().amax(dim=1, keepdim=True)
    assert torch.equal(scale, amax / FP8_MAX)
    expected = torch.clamp(w.float() / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    assert torch.equal(q.view(torch.uint8), expected.view(torch.uint8))
    # every row reaches the FP8 maximum exactly once its amax is scaled
    assert torch.equal(q.float().abs().amax(dim=1), torch.full((16,), FP8_MAX))


def test_round_trip_error_is_within_half_an_e4m3_step():
    w = _weight(seed=3)
    q, scale = quantize_fp8_per_channel(w)
    back = dequantize_fp8_per_channel(q, scale)
    x = w.float() / scale
    # E4M3 has 3 mantissa bits: a normal value rounds within 2**-4 relative; below the
    # smallest normal (2**-6) the step is 2**-9 absolute.
    bound = torch.maximum(x.abs() * 2.0**-4, torch.full_like(x, 2.0**-10)) * scale
    assert torch.all((back - w.float()).abs() <= bound + 1e-12)


def test_rows_are_independent_so_row_slices_match():
    w = _weight(seed=5)
    q, scale = quantize_fp8_per_channel(w)
    q2, scale2 = quantize_fp8_per_channel(w[4:9])
    assert torch.equal(q[4:9].view(torch.uint8), q2.view(torch.uint8)) and torch.equal(scale[4:9], scale2)


def test_zero_row_gets_epsilon_scale_and_zero_codes():
    w = _weight()
    w[2] = 0
    q, scale = quantize_fp8_per_channel(w)
    assert scale[2, 0].item() == torch.finfo(torch.float32).eps
    assert torch.all(q[2].float() == 0)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="2-D"):
        quantize_fp8_per_channel(torch.zeros(4))
    with pytest.raises(TypeError, match="floating"):
        quantize_fp8_per_channel(torch.zeros(2, 2, dtype=torch.int8))
    w = _weight()
    w[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        quantize_fp8_per_channel(w)


def test_equals_compressed_tensors_where_installed():
    ct = pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.lifecycle.forward import quantize
    from compressed_tensors.quantization.utils import calculate_qparams

    args = QuantizationArgs(num_bits=8, type="float", symmetric=True, strategy="channel")
    w = _weight(rows=32, cols=128, seed=7)
    w[1] = 0
    ref_scale, zp = calculate_qparams(w.float().amin(dim=1, keepdim=True), w.float().amax(dim=1, keepdim=True), args)
    ref_q = quantize(w.float(), ref_scale, zp, args, dtype=FP8_DTYPE)
    q, scale = quantize_fp8_per_channel(w)
    assert torch.equal(scale, ref_scale), ct.__version__
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
