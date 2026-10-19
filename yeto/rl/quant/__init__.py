"""Re-quantization of trained BF16 weights for W4A8 samplers (S25 N8, rl-hetero-sampler-engines D5-D7).

The trainer trains full parameters in BF16; the sampler serves a W4A8 checkpoint.
After each trainer release, this package turns the BF16 weights back into the
layout of the original W4A8 artefact, so the sampler loads it like the original.

Target format (read from the original artefact
``apex-w4a8-full:runs/apex-awq-w4a8-20261012-v10/checkpoint/``, 2026-10-16):
compressed-tensors ``int-quantized`` (llm-compressor), not msModelSlim.
Routed expert linears carry ``weight`` (I8 holding int4 values in [-8, 7]) and
``weight_scale`` (BF16, shape [out, 1], symmetric per output channel); all
other tensors stay high precision. AWQ smoothing was applied before
quantization; its per-layer factors are stored next to the checkpoint
(``awq-scales/<smooth_layer>.pt``), so option R2 reads them instead of
deriving them.

A second target (N8.5) is the MXFP4 experts artefact
``yetalabs/apex-flash-1-mxfp4-experts@70e86a0``: compressed-tensors
``mxfp4-pack-quantized``, routed experts only, ``weight_packed`` (U8, two FP4
E2M1 codes per byte) + ``weight_scale`` (U8 E8M0, one per 32 inputs), RTN
without AWQ; all other tensors BF16/F32 as in the source. The converter picks
the scheme from the template's ``quantization_config.format``.

Modules:

* :mod:`.int4`: per-channel symmetric int4 quantize / dequantize.
* :mod:`.mxfp4`: MXFP4 group-32 quantize / dequantize, E8M0 scales, FP4 packing.
* :mod:`.awq`: the fixed AWQ smoothing (R2) and the no-smoothing fallback (R1).
* :mod:`.requant`: the checkpoint layout and its scheme, the per-tensor
  transform, the shard-by-shard converter, and the tensor-by-tensor comparison.
"""
