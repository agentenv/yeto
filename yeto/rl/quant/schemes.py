"""Names of the weight schemes a policy release can carry (S25 N8.3).

This module is the one place for the scheme tags. The writer side
(:mod:`.requant`, :mod:`.release`) and the reader side (the sampler's
``PolicyFollower``) both import them from here. It does not import torch, so a
sampler process can use it without the quantization code.

* :data:`BF16`: trained weights as the trainer releases them.
* :data:`W4A8_INT4_CHANNEL`: routed experts int4 per output channel, BF16
  scales, 8-bit dynamic activations (the original W4A8 artefact, N8.1).
* :data:`MXFP4_G32_RTN`: routed experts MXFP4, one E8M0 scale per 32 inputs,
  round to nearest without AWQ (N8.5).
* :data:`FP8_FFN_CHANNEL`: FP8 E4M3 per output channel with dynamic per-token
  FP8 activations (W8A8) on the in-scope linears only (FFN of a dense model,
  routed experts of a MoE model; S26 E6 tier 2, :mod:`.scope`).

Release directories are named by scheme (:func:`release_dir_for`):
``policy-releases`` for BF16, ``policy-releases-<scheme>`` for a re-quantized
scheme. The W4A8 scheme keeps its older name ``policy-releases-w4a8``, so
stores written before N8.3 stay readable.
"""

from __future__ import annotations

BF16 = "bf16"
W4A8_INT4_CHANNEL = "w4a8-int4-channel"
MXFP4_G32_RTN = "mxfp4-g32-rtn"
FP8_FFN_CHANNEL = "fp8-w8a8-channel"
QUANTIZED_SCHEMES = (W4A8_INT4_CHANNEL, MXFP4_G32_RTN, FP8_FFN_CHANNEL)
ALL_SCHEMES = (BF16, *QUANTIZED_SCHEMES)

RELEASE_DIR = "policy-releases"
_LEGACY_DIRS = {W4A8_INT4_CHANNEL: "policy-releases-w4a8"}

# AWQ source tags of a release record (``derived_from.awq``).
AWQ_NONE = "none"  # option R1, and every MXFP4 and FP8 release
AWQ_FIXED_ARTEFACT = "fixed-artefact"  # option R2: fixed factors from the original artefact


def release_dir_for(scheme: str) -> str:
    """The store directory that holds the releases of ``scheme``."""
    if scheme == BF16:
        return RELEASE_DIR
    if scheme not in QUANTIZED_SCHEMES:
        raise ValueError(f"unknown weight scheme {scheme!r}; known: {', '.join(ALL_SCHEMES)}")
    return _LEGACY_DIRS.get(scheme, f"{RELEASE_DIR}-{scheme}")
