"""Symmetric per-output-channel int4 weight quantization (compressed-tensors ``int-quantized``).

The rule reproduces llm-compressor's ``memoryless_minmax`` observer for
``num_bits=4, symmetric=True, strategy=channel``:

* ``scale = amax(|w|, row) / 7.5`` (half of the int4 range ``7 - (-8)``),
  rounded to the scale dtype (BF16 in the original artefact);
* ``q = clamp(round_half_even(w / scale), -8, 7)``, stored as int8.

Evidence: one routed expert (layer 30, expert 229, gate/up/down) of the
original artefact was recomputed from the BF16 source with the AWQ factors:
int values 100 % equal, scales bit-equal (CPU, 2026-10-16). The divisor 7.0
or 8.0 gives about 90 % equal values, so the divisor is not a free choice.

Each output row is quantized on its own, so any row slice of a weight gives
the same rows as the whole weight (the basis for per-rank quantization, D7).
"""

from __future__ import annotations

import torch

QMIN = -8
QMAX = 7
# compressed-tensors symmetric: scale = amax / ((QMAX - QMIN) / 2).
_HALF_RANGE = (QMAX - QMIN) / 2


def quantize_per_channel(
    weight: torch.Tensor, scale_dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight ``[out, in]``; returns ``(q int8 [out, in], scale [out, 1])``.

    The weight is read in its own precision (BF16 for the trainer output) and
    computed in FP32. A row of zeros gets the smallest positive scale of
    ``scale_dtype`` so that dequantization stays finite (zero rows: rule not
    compared with the original artefact).
    """
    if weight.dim() != 2:
        raise ValueError(f"per-channel int4 needs a 2-D weight, got shape {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    w = weight.float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / _HALF_RANGE).to(scale_dtype)
    scale = torch.where(scale == 0, torch.finfo(scale_dtype).tiny, scale)
    if not torch.isfinite(scale).all():
        raise ValueError("weight has non-finite values; refusing to quantize")
    q = torch.round(w / scale.float()).clamp_(QMIN, QMAX).to(torch.int8)
    return q, scale


def dequantize_per_channel(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of :func:`quantize_per_channel`: ``q * scale`` in ``dtype``."""
    if q.dtype != torch.int8 or q.dim() != 2 or scale.shape != (q.shape[0], 1):
        raise ValueError(f"expected int8 [out, in] and scale [out, 1], got {q.dtype} {tuple(q.shape)}, {tuple(scale.shape)}")
    return (q.float() * scale.float()).to(dtype)
