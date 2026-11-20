"""The Kimi-K3.1 trainer MFU switches of the next9 trainer image, on by default.

Miles ``k31/integ-next9`` (``MILES_K31_NEXT9_COMMIT``, the image
``MILES_K31_NEXT9_IMAGE``) carries the trainer MFU work behind environment switches
that each default off in Miles. With none set, next9 trains the math of the previous
full image at about 5% MFU; with the set below (the 16-layer and 33-layer slice
measurements) the full-model estimate is about 15.7%. The kimi_k31 recipe therefore
declares them as its trainer environment defaults (:func:`declaration`, served by the
Miles ``recipe_env`` role) for ``--tuning full`` when the trainer image is next9 (by
digest, any registry). An older image has no code behind these names, so it gets none.
The generic side (``yeto.launch.recipe_env``) applies them: a value the launch sets
itself wins, and ``--rl-recipe-env-defaults off`` turns them off.

:data:`MFU_SWITCHES` is the one list; ``docs/KIMI_K31_ASYNC_RL_GUIDE.md`` section 9.1.1
shows it (a test keeps the two equal). :func:`memory_terms_gib` is what the switches
change in the launch-time memory estimate (``yeto.launch.k31_checks``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yeto.rl.adapters.miles.pins import MILES_K31_NEXT9_IMAGE

# (variable, value): docs/KIMI_K31_ASYNC_RL_GUIDE.md section 9.1.1 (26) plus the
# attention-residual bank pad of the measured and formal configurations (0, the Miles
# default, written out as those runs did)
MFU_SWITCHES: tuple[tuple[str, str], ...] = (
    ("MILES_K31_INDEXER_TP_SPLIT", "1"),
    ("MILES_K31_INDEXER_RECOMPUTE_CACHE", "1"),
    ("MILES_K31_KDA_CP_RELAYOUT_V2", "1"),
    ("MILES_K31_MLA_CP_GATHER_V2", "1"),
    ("MILES_K31_KDA_SAVE_INTERMEDIATES", "1"),
    ("MILES_K31_MXFP4_LUT_DEQUANT", "1"),
    ("MILES_K31_MXFP4_REDEQUANT_BACKWARD", "1"),
    ("MILES_K31_MXFP4_BATCHED_DEQUANT", "1"),
    ("MILES_K31_MXFP4_TRITON_DEQUANT", "1"),
    ("MILES_K31_MXFP4_QAT_NO_SYNC", "1"),
    ("MILES_K3_SITU_TRITON", "1"),
    ("MILES_K3_SITU_PROBS_FUSED", "1"),
    ("MILES_K3_ATTN_RES_FUSED", "1"),
    ("MILES_K31_ROUTER_GEMM", "tf32"),
    ("MILES_K31_DSA_QUERY_SPLIT", "64"),
    ("MILES_K31_CHUNK_SORT_MAP_TORCH", "1"),
    ("MILES_K31_KDA_CP_RELAYOUT_A2A", "1"),
    ("MILES_K31_KDA_SP_RELAYOUT_FUSED", "1"),
    ("MILES_K31_KDA_CORE_RELAYOUT_OVERLAP", "1"),
    ("MILES_K31_KDA_OUT_PROJ_RS_CHUNKS", "4"),
    ("MILES_K31_KDA_FLA_CP_BLOCK", "32"),
    ("MILES_K3_MOE_A2A_OVERLAP", "1"),
    ("MILES_K3_MOE_A2A_OVERLAP_BWD", "1"),
    ("MILES_SC_TOPK_PREP", "1"),
    ("MILES_SC_LOGPROB_LEAN", "1"),
    ("MILES_CP_RESPONSE_VIEW", "1"),
    ("MILES_K3_ATTN_RES_BANK_PAD_ROWS", "0"),
)
NAMES = frozenset(name for name, _ in MFU_SWITCHES)
NEXT9_DIGEST = MILES_K31_NEXT9_IMAGE.rpartition("@sha256:")[2]


def image_digest(image: str | None) -> str:
    return str(image or "").rpartition("@sha256:")[2]


NAME = "kimi_k31 next9 trainer MFU switches"
DOC = "docs/KIMI_K31_ASYNC_RL_GUIDE.md section 9.1.1"


def declaration(args: Any) -> dict[str, Any] | None:
    """The recipe's trainer environment defaults for ``args`` (a launch, or one island's
    args): ``applies`` with ``env`` and the ``reason`` either way; None outside
    ``--rl-model-recipe kimi_k31 --tuning full``. The image is ``args.rl_image`` as
    resolved (``resolve_default_rl_image``)."""
    if getattr(args, "rl_model_recipe", None) != "kimi_k31" or getattr(args, "tuning", None) != "full":
        return None
    digest = image_digest(getattr(args, "rl_image", None))
    if digest != NEXT9_DIGEST:
        return {"name": NAME, "doc": DOC, "applies": False, "env": {},
                "reason": f"the trainer image ({digest[:12] or 'no digest'}) is not next9 "
                          f"({NEXT9_DIGEST[:12]}): it has no code behind these switches"}
    return {"name": NAME, "doc": DOC, "applies": True, "env": dict(MFU_SWITCHES),
            "reason": "kimi_k31 --tuning full on the next9 trainer image"}


# --- what the switches change in the launch-time memory estimate (GiB per GPU) ----------
# Reference: one GPU of 262k / CP2 / TP8 (16384 tokens per GPU after sequence parallel,
# 131072 per CP rank), EP16 with 128 trainable experts; each term is linear in tokens per GPU.
REF_TOKENS_PER_GPU = 16384
# INDEXER_RECOMPUTE_CACHE: this TP rank's rows of the [T_cp, 2048] int32 top-k, kept per DSA
# layer and microbatch in flight until its recompute (128 MiB at the reference)
INDEXER_TOPK = 2048
# KDA_SAVE_INTERMEDIATES: one KDA layer's chunk_kda intermediates kept for its backward
# (Miles f61eb35d linear_attn.kda_keeps_intermediates: ~3.6 GiB at 131k tokens x 12 heads)
KDA_SAVE_GIB = 3.6
# SITU_TRITON: autograd no longer saves six fp32 [rows, 3072] SiTU intermediates of the
# routed-expert layer being recomputed (Miles 6c33130e: ~18 GiB at the reference)
SITU_SAVED_GIB = 18.0


def memory_terms_gib(env: Mapping[str, str], *, tokens_per_gpu: float, in_flight: int,
                     dsa_layers: int, kda_layers: int, moe_layers: int,
                     frozen_dequant_gib: float) -> dict[str, float]:
    """Per-GPU allocated-peak change of one pipeline stage under the switches ``env`` sets.

    Added: the indexer top-k cache (checkpoint-like, per DSA layer and in-flight microbatch)
    and the KDA intermediates of the one layer in backward. Removed from the one-layer
    working set: the SiTU fp32 intermediates and, with MXFP4_REDEQUANT_BACKWARD, the frozen
    experts' BF16 copy TE saves for the backward (``frozen_dequant_gib``). The other switches
    are not modelled; the 16-layer measurement with every switch on peaked 25.3 GiB below the
    same layout without them, more than these terms remove, so the estimate stays above it."""
    scale = tokens_per_gpu / REF_TOKENS_PER_GPU
    on = lambda name: str(env.get(name, "0")) == "1"  # noqa: E731
    terms = {"indexer_topk_cache": 0.0, "kda_intermediates": 0.0, "situ_saved": 0.0,
             "frozen_redequant": 0.0}
    if on("MILES_K31_INDEXER_RECOMPUTE_CACHE") and dsa_layers:
        terms["indexer_topk_cache"] = in_flight * dsa_layers * tokens_per_gpu * INDEXER_TOPK * 4 / 1024 ** 3
    if on("MILES_K31_KDA_SAVE_INTERMEDIATES") and kda_layers:
        terms["kda_intermediates"] = KDA_SAVE_GIB * scale
    if on("MILES_K3_SITU_TRITON") and moe_layers:
        terms["situ_saved"] = -SITU_SAVED_GIB * scale
    if on("MILES_K31_MXFP4_REDEQUANT_BACKWARD") and moe_layers:
        terms["frozen_redequant"] = -frozen_dequant_gib
    return terms


__all__ = ["DOC", "MFU_SWITCHES", "NAME", "NAMES", "NEXT9_DIGEST", "declaration", "image_digest",
           "memory_terms_gib"]
