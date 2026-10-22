"""Symmetric per-output-channel FP8 E4M3 weight quantization (S26 E6, tier 2).

Target format: compressed-tensors ``float-quantized`` with weights
``num_bits=8, type=float, symmetric=True, strategy=channel`` and dynamic
per-token FP8 activations (llm-compressor's ``FP8_DYNAMIC`` preset, W8A8).
vLLM and SGLang both serve it on Hopper (``CompressedTensorsW8A8Fp8``).

The rule reproduces compressed-tensors 0.18 (``calculate_qparams`` and
``_quantize``) for an FP32 scale:

* ``scale = amax(|w|, row) / 448`` in FP32 (448 is the E4M3 maximum; the
  symmetric range is ``(448 - (-448)) / 2``); a zero row gets the FP32
  machine epsilon, so dequantization stays finite;
* ``q = clamp(w / scale, -448, 448)`` cast to ``float8_e4m3fn`` (round to
  nearest even, done by the cast).

``tests/test_rl_quant_fp8.py`` compares this module with compressed-tensors
itself where it is installed (the ``sglang-next-venv`` has 0.18.0).

Each output row is quantized on its own, so any row slice of a weight gives
the same rows as the whole weight (per-rank quantization, :mod:`.shards`).
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def quantize_fp8_per_channel(
    weight: torch.Tensor, scale_dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight ``[out, in]``; returns ``(q float8_e4m3fn [out, in], scale [out, 1])``."""
    if weight.dim() != 2:
        raise ValueError(f"per-channel FP8 needs a 2-D weight, got shape {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError(f"per-channel FP8 needs a floating-point weight, got {weight.dtype}")
    w = weight.to(torch.float32)
    if not torch.isfinite(w).all():
        raise ValueError("weight holds NaN or infinity; refusing to quantize it")
    scale = w.abs().amax(dim=1, keepdim=True) / FP8_MAX
    scale = torch.where(scale == 0, torch.tensor(torch.finfo(torch.float32).eps), scale)
    scale = scale.to(scale_dtype)
    q = torch.clamp(w / scale.to(torch.float32), -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


def dequantize_fp8_per_channel(
    q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """``q * scale`` in ``dtype`` (the reference of what an engine computes with)."""
    if q.dtype != FP8_DTYPE:
        raise TypeError(f"expected {FP8_DTYPE}, got {q.dtype}")
    return (q.to(torch.float32) * scale.to(torch.float32)).to(dtype)
