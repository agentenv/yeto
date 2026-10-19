"""MXFP4 weight quantization, group 32, RTN (compressed-tensors ``mxfp4-pack-quantized``).

The rules reproduce compressed-tensors 0.18 (the reference artefact
``yetalabs/apex-flash-1-mxfp4-experts@70e86a0`` declares 0.19.0) for
``num_bits=4, type=float, strategy=group, group_size=32, scale_dtype=uint8``:

* Groups are 32 consecutive input columns of one output row.
* Shared scale (E8M0, uint8, bias 127): take the group ``amax`` in BF16, round
  it to a power of two with the mantissa threshold 1.75 (``round_to_power_2``:
  add ``1 << 5`` to the BF16 bits, keep sign and exponent), then subtract 2
  (``floor(log2(6))``, the FP4 E2M1 maximum) from the exponent. The code is
  clamped below at 0, so codes lie in ``[0, 252]``; an all-zero group gets
  code 0. A group whose amax rounds to infinity (``>= 1.75 * 2**127``) is
  refused; compressed-tensors would store code 255 (E8M0 NaN) there.
* Element (FP4 E2M1): ``w / 2**(code - 127)`` rounded to the nearest value of
  ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}``, ties to even mantissa, saturating at 6. The
  sign bit is set for ``w < 0``, so a negative value that rounds to zero is
  stored as code ``0b1000`` (``-0``) while an input ``-0.0`` is stored as code 0,
  as ``cast_to_fp4`` + ``pack_fp4_to_uint8`` do.
* Packing: two codes per byte along the input dimension, the even column in the
  low nibble. ``weight_packed`` is ``uint8 [out, in / 2]``, ``weight_scale`` is
  ``uint8 [out, in / 32]``.

No calibration data and no smoothing: round to nearest only (RTN). Each output
row is quantized on its own, so any row slice of a weight gives the same rows
as the whole weight (the basis for per-rank quantization, D7).

Evidence (CPU, 2026-10-19): the BF16 layer-10 routed experts of
``apex-flash-1`` (288 ``down_proj``, 4 ``gate_proj`` / ``up_proj``; native
layout copy on Modal) quantized with this module equal the reference artefact
in every FP4 code (2.48e9, 1.2e8 of them ``-0``) and every E8M0 code (7.8e7).
``tests/test_rl_quant_mxfp4.py`` also cross-checks compressed-tensors itself
where it is installed.
"""

from __future__ import annotations

import torch

GROUP_SIZE = 32
E8M0_BIAS = 127
# floor(log2(FP4_E2M1_MAX)) with FP4_E2M1_MAX = 6.0.
_ELEM_EXP_OFFSET = 2
# BF16: 7 mantissa bits; FP4 E2M1: 1. Adding 1 << (7 - 1 - 1) carries every
# mantissa >= 1.75 into the next exponent before the mantissa is dropped.
_BF16_ROUND_ADD = 1 << 5
_BF16_EXP_MASK = 0x7F80

# Magnitude code k (0..7) decodes to FP4_VALUES[k]; bit 3 is the sign.
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Rounding boundaries between neighbouring magnitudes and whether the boundary
# value itself goes up (True) or down (False): ties to the even mantissa.
_BOUNDARIES = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
_SIGN_BIT = 0x8


def e8m0_codes(amax: torch.Tensor) -> torch.Tensor:
    """Group ``amax`` (non-negative BF16) -> E8M0 scale code (uint8), as described in the module doc."""
    if amax.dtype != torch.bfloat16:
        raise TypeError(f"amax must be bfloat16, got {amax.dtype}")
    if not torch.isfinite(amax).all():
        raise ValueError("weight has non-finite values; refusing to quantize")
    bits = amax.view(torch.int16).to(torch.int32) & 0xFFFF
    biased_exp = ((bits + _BF16_ROUND_ADD) & _BF16_EXP_MASK) >> 7
    if (biased_exp == 0xFF).any():
        # amax >= 1.75 * 2**127 rounds to infinity; compressed-tensors would store code 255 (E8M0 NaN).
        raise ValueError("group amax too large for an E8M0 scale; refusing to quantize")
    # biased_exp == 0 means amax rounded to zero (all-zero or tiny subnormal group):
    # compressed-tensors computes log2(0) = -inf there and clamps the code to 0.
    codes = torch.where(biased_exp == 0, 0, biased_exp - _ELEM_EXP_OFFSET)
    return codes.clamp_(min=0).to(torch.uint8)


def e8m0_to_scale(codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """E8M0 code -> power-of-two scale ``2 ** (code - 127)``."""
    if codes.dtype != torch.uint8:
        raise TypeError(f"E8M0 codes must be uint8, got {codes.dtype}")
    return torch.exp2(codes.to(torch.float32) - E8M0_BIAS).to(dtype)


def fp4_encode(x: torch.Tensor) -> torch.Tensor:
    """Round already-scaled values to FP4 E2M1 codes (uint8, 0..15), as described in the module doc."""
    mag = x.abs()
    k = torch.zeros(x.shape, dtype=torch.uint8, device=x.device)
    for bound, tie_up in _BOUNDARIES:
        k += (mag >= bound) if tie_up else (mag > bound)
    return k | ((x < 0).to(torch.uint8) << 3)


def fp4_decode(codes: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """FP4 E2M1 codes (uint8, 0..15) -> values."""
    table = torch.tensor(FP4_VALUES, dtype=torch.float32, device=codes.device)
    mag = table[(codes & 0x7).long()]
    return torch.where((codes & _SIGN_BIT).bool(), -mag, mag).to(dtype)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """``uint8 [out, in]`` codes -> ``uint8 [out, in / 2]``; the even column goes to the low nibble."""
    if codes.dim() != 2 or codes.shape[1] % 2:
        raise ValueError(f"need a 2-D code tensor with an even column count, got {tuple(codes.shape)}")
    if codes.dtype != torch.uint8 or (codes > 0xF).any():
        raise ValueError("FP4 codes must be uint8 in [0, 15]")
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_fp4`."""
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise ValueError(f"expected uint8 [out, in / 2], got {packed.dtype} {tuple(packed.shape)}")
    return torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(packed.shape[0], -1)


def quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 weight ``[out, in]``; returns ``(weight_packed [out, in/2], weight_scale [out, in/32])``.

    The division by a power of two is exact in FP32 for every BF16 input, so
    the result does not depend on the compute dtype.
    """
    if weight.dim() != 2:
        raise ValueError(f"MXFP4 needs a 2-D weight, got shape {tuple(weight.shape)}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"MXFP4 RTN reads the BF16 trainer weight, got {weight.dtype}")
    rows, cols = weight.shape
    if cols % GROUP_SIZE:
        raise ValueError(f"input dimension {cols} is not a multiple of the group size {GROUP_SIZE}")
    groups = weight.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    codes = e8m0_codes(groups.abs().amax(dim=-1))
    scaled = groups.float() / e8m0_to_scale(codes).unsqueeze(-1)
    return pack_fp4(fp4_encode(scaled).reshape(rows, cols)), codes


def dequantize_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of :func:`quantize_mxfp4`: FP4 value times its group scale, in ``dtype``."""
    values = fp4_decode(unpack_fp4(packed))
    rows, cols = values.shape
    if scale.dtype != torch.uint8 or scale.shape != (rows, cols // GROUP_SIZE) or cols % GROUP_SIZE:
        raise ValueError(f"scale {scale.dtype} {tuple(scale.shape)} does not fit packed {tuple(packed.shape)}")
    groups = values.reshape(rows, -1, GROUP_SIZE) * e8m0_to_scale(scale).unsqueeze(-1)
    return groups.reshape(rows, cols).to(dtype)
