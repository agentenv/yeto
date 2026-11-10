"""Ports Miles argv of the Kimi-K3.1 partial-experts recipe.

The trainer reads the native MXFP4 checkpoint in raw bridge mode
(``--megatron-to-hf-mode raw``, Miles ``scripts/kimi_k31_w4a16_lora_common.py``),
permutes experts into EP slots from the selection file
(``--partial-experts-selection``, F2) and keeps the router and its bias
frozen (``--moe-router-bias-update-rate 0``). Export, routing replay and
checkpoints use original expert ids, so the engine serves the unmodified
checkpoint.

Constraints encoded here: EP must divide both 896 and 128 (power of two up
to 128); virtual pipeline stages are refused (the trainable / frozen expert
split is per layer and the F2 freeze hook runs once per model chunk); zigzag
context parallel needs the packed sequence length divisible by ``2 * CP``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from yeto.rl.adapters.miles.models.glm5_next import apply_recipe
from yeto.rl.adapters.miles.models.kimi_k31.model import (
    DENSE_FFN_HIDDEN_SIZE,
    EP_SIZE_DEFAULT,
    FIRST_K_DENSE_REPLACE,
    HIDDEN_SIZE,
    MODEL_CONFIG,
    MODEL_TYPE,
    MOE_FFN_HIDDEN_SIZE,
    MOE_LATENT_SIZE,
    NUM_ATTENTION_HEADS,
    NUM_EXPERTS,
    NUM_LAYERS,
    NUM_TRAINABLE,
    SHARED_EXPERT_INTERMEDIATE_SIZE,
    TOPK,
    VOCAB_SIZE,
)

MODEL_NAME = MODEL_TYPE  # Miles --model-name (mbridge plugin registered as kimi_k31, F1)


class KimiK31RecipeError(ValueError):
    """The requested layout violates a K3.1 recipe constraint."""

MEGATRON_TO_HF_MODE = "raw"
SELECTION_FLAG = "--partial-experts-selection"
ROUTED_EXPERT_QAT_FLAG = "--routed-expert-qat"
ROUTED_EXPERT_QAT_FORMATS = ("mxfp4",)
SPEC = ("miles_plugins.models.kimi_k31", "get_kimi_k31_spec")
# A Kimi-K3 checkpoint (HF model_type kimi_k3: MLA without the K3.1 DSA indexer, e.g.
# the Pinaster/Kimi-K3-4layer smoke model) takes the K3 spec and bridge: the K3.1
# ones read text_config.dsa_config, which a K3 config does not have (F2 2.1).
SPEC_K3 = ("miles_plugins.models.kimi_k3", "get_kimi_k3_spec")
MODEL_NAME_K3 = "kimi_k3"
HF_MODEL_TYPE_K3 = "kimi_k3"


def spec_for(hf_model_type: str = MODEL_TYPE) -> tuple[str, str]:
    """The Miles ``--spec`` pair of a checkpoint's HF ``model_type``."""
    return SPEC_K3 if hf_model_type == HF_MODEL_TYPE_K3 else SPEC


def model_name_for(hf_model_type: str = MODEL_TYPE) -> str:
    """The Miles ``--model-name`` (mbridge plugin) of a checkpoint's HF ``model_type``."""
    return MODEL_NAME_K3 if hf_model_type == HF_MODEL_TYPE_K3 else MODEL_NAME

# Flags of a virtual pipeline layout; any of them refuses the recipe.
VPP_FLAGS = ("--num-layers-per-virtual-pipeline-stage", "--num-virtual-stages-per-pipeline-rank")

# Miles scripts/models/kimi-k3.py with the K3.1 spec (F1 scripts/models/kimi-k31.py).
# The layer-dependent flags (--num-layers, --moe-layer-freq) are appended by
# :func:`model_argv`; a cut-down checkpoint (Kimi-K3-4layer) passes its own count.
_MODEL_ARGV: tuple[str, ...] = (
    "--disable-bias-linear",
    "--hidden-size", str(HIDDEN_SIZE),
    "--ffn-hidden-size", str(DENSE_FFN_HIDDEN_SIZE),
    "--num-attention-heads", str(NUM_ATTENTION_HEADS),
    "--num-query-groups", str(NUM_ATTENTION_HEADS),
    "--kv-channels", "256",
    "--normalization", "RMSNorm",
    "--position-embedding-type", "none",
    "--rotary-base", "10000",
    "--norm-epsilon", "1e-5",
    "--hidden-dropout", "0",
    "--attention-dropout", "0",
    "--disable-bf16-reduced-precision-matmul",
    "--swiglu",
    "--untie-embeddings-and-output-weights",
    "--vocab-size", str(VOCAB_SIZE),
    "--make-vocab-size-divisible-by", "128",
    "--multi-latent-attention",
    "--q-lora-rank", "1536",
    "--kv-lora-rank", "512",
    "--qk-head-dim", "128",
    "--qk-pos-emb-head-dim", "64",
    "--v-head-dim", "128",
    "--qk-layernorm",
    "--attention-softmax-in-fp32",
    "--moe-ffn-hidden-size", str(MOE_FFN_HIDDEN_SIZE),
    "--moe-latent-size", str(MOE_LATENT_SIZE),
    "--moe-shared-expert-intermediate-size", str(SHARED_EXPERT_INTERMEDIATE_SIZE),
    "--moe-router-topk", str(TOPK),
    "--moe-router-score-function", "sigmoid",
    "--moe-router-pre-softmax",
    "--moe-router-enable-expert-bias",
    "--moe-router-load-balancing-type", "none",
    "--moe-token-dispatcher-type", "alltoall",
    "--moe-router-bias-update-rate", "0",
    "--moe-aux-loss-coeff", "0",
    "--moe-router-topk-scaling-factor", "1.0",
    "--moe-router-dtype", "fp32",
    "--moe-grouped-gemm",
    "--moe-permute-fusion",
)


def moe_layer_freq(num_layers: int = NUM_LAYERS, first_k_dense_replace: int = FIRST_K_DENSE_REPLACE) -> str:
    """Miles ``--moe-layer-freq`` of ``first_k_dense_replace`` dense layers then MoE
    (``[0]+[1]*92`` for Kimi-K3.1-Pro, the form scripts/models/kimi-k3.py uses)."""
    layers, dense = int(num_layers), int(first_k_dense_replace)
    if layers < 1 or not 0 <= dense < layers:
        raise KimiK31RecipeError(f"first_k_dense_replace {dense} must be below num_layers {layers}")
    if dense == 0:
        return f"[1]*{layers}"
    return f"[0]+[1]*{layers - dense}" if dense == 1 else f"[0]*{dense}+[1]*{layers - dense}"


def model_argv(*, num_layers: int = NUM_LAYERS, first_k_dense_replace: int = FIRST_K_DENSE_REPLACE,
               hf_model_type: str = MODEL_TYPE, num_experts: int | None = None) -> list[str]:
    """The model flags of the recipe for a checkpoint of ``num_layers`` layers,
    ``num_experts`` routed experts (default Kimi-K3.1-Pro's 896; a cut-down
    checkpoint passes its own, e.g. 64) and HF ``hf_model_type`` (``kimi_k31``,
    or ``kimi_k3`` for a K3 checkpoint)."""
    experts = NUM_EXPERTS if num_experts is None else int(num_experts)
    if experts < 1:
        raise KimiK31RecipeError(f"num_experts {experts} must be positive")
    return ["--spec", *spec_for(hf_model_type), *_MODEL_ARGV, "--num-experts", str(experts),
            "--num-layers", str(int(num_layers)),
            "--moe-layer-freq", moe_layer_freq(num_layers, first_k_dense_replace)]


MODEL_ARGV: tuple[str, ...] = tuple(model_argv())  # Kimi-K3.1-Pro (93 layers)

# Recipe flags beyond the model: raw bridge, routing replay of the rollout's
# original ids (indexer replay stays off: ~196 KB of top-k per token).
RECIPE_ARGV: tuple[str, ...] = (
    "--megatron-to-hf-mode", MEGATRON_TO_HF_MODE,
    "--use-rollout-routing-replay",
    "--no-enable-sample-ownership-checker",
)



def check_expert_parallel(ep_size: int) -> int:
    try:
        ep = MODEL_CONFIG.check_ep_size(ep_size)
    except ValueError as error:
        raise KimiK31RecipeError(str(error)) from None
    if ep & (ep - 1):
        raise KimiK31RecipeError(f"expert parallel size {ep} must be a power of two")
    return ep


def check_zigzag_context_parallel(context_parallel_size: int, seq_length: int) -> int:
    """Megatron zigzag CP splits every packed sequence into ``2 * CP`` chunks."""

    cp = int(context_parallel_size)
    if cp <= 0 or cp & (cp - 1):
        raise KimiK31RecipeError(f"context parallel size {cp} must be a power of two")
    if cp > 1 and int(seq_length) % (2 * cp):
        raise KimiK31RecipeError(
            f"zigzag context parallel {cp} needs the sequence length ({seq_length}) divisible by {2 * cp}"
        )
    return cp


def refuse_virtual_pipeline(extra_argv: Sequence[str]) -> None:
    for token in extra_argv:
        flag = token.split("=", 1)[0]
        if flag in VPP_FLAGS:
            raise KimiK31RecipeError(f"{flag}: the Kimi-K3.1 partial-experts recipe does not support VPP")


def recipe_argv(
    *,
    selection_path: str | Path | None,
    ep_size: int = EP_SIZE_DEFAULT,
    context_parallel_size: int = 1,
    seq_length: int,
    extra_argv: Sequence[str] = (),
    include_model_argv: bool = True,
    num_layers: int = NUM_LAYERS,
    first_k_dense_replace: int = FIRST_K_DENSE_REPLACE,
    routed_expert_qat: str | None = None,
    hf_model_type: str = MODEL_TYPE,
    num_experts: int | None = None,
) -> list[str]:
    """Miles argv of the recipe for one layout.

    The selection path is passed as given (Miles opens it on every rank);
    ``ep_size`` / ``context_parallel_size`` are validated here but emitted by
    the parallel layout (``--expert-model-parallel-size``,
    ``--context-parallel-size``) as for every recipe. ``num_layers`` /
    ``first_k_dense_replace`` / ``num_experts``: the checkpoint's (a cut-down
    smoke model; ``num_experts`` None keeps Kimi-K3.1-Pro's 896).
    ``routed_expert_qat``: Miles ``--routed-expert-qat`` (needs Miles k31/qat);
    None leaves the argv as before. ``selection_path`` None: the LoRA form
    (``yeto.rl.engine.kimi_k31_lora``), no partial experts, so no selection flag;
    the full form always passes one.
    """

    check_expert_parallel(ep_size)
    check_zigzag_context_parallel(context_parallel_size, seq_length)
    refuse_virtual_pipeline(extra_argv)
    lora = selection_path is None
    path = "" if lora else str(selection_path)
    if not lora and not path:
        raise KimiK31RecipeError("the K3.1 recipe needs an expert selection file")
    if lora and routed_expert_qat is not None:
        raise KimiK31RecipeError("routed-expert QAT trains MXFP4 experts; the LoRA form keeps them frozen")
    if routed_expert_qat is not None and routed_expert_qat not in ROUTED_EXPERT_QAT_FORMATS:
        raise KimiK31RecipeError(f"routed-expert QAT {routed_expert_qat!r}: supported {ROUTED_EXPERT_QAT_FORMATS}")
    return [
        *(model_argv(num_layers=num_layers, first_k_dense_replace=first_k_dense_replace,
                     hf_model_type=hf_model_type, num_experts=num_experts)
          if include_model_argv else ()),
        *RECIPE_ARGV,
        *(() if lora else (SELECTION_FLAG, path)),
        *((ROUTED_EXPERT_QAT_FLAG, routed_expert_qat) if routed_expert_qat else ()),
    ]


__all__ = ["MODEL_ARGV", "MODEL_NAME", "MODEL_NAME_K3", "SPEC_K3", "spec_for", "model_name_for", "RECIPE_ARGV", "ROUTED_EXPERT_QAT_FLAG", "SELECTION_FLAG", "VPP_FLAGS", "KimiK31RecipeError",
           "apply_recipe", "check_expert_parallel", "check_zigzag_context_parallel", "model_argv",
           "moe_layer_freq", "recipe_argv", "refuse_virtual_pipeline", "trainable_summary"]


def trainable_summary(ep_size: int = EP_SIZE_DEFAULT) -> dict[str, int]:
    ep = check_expert_parallel(ep_size)
    return {
        "num_experts": NUM_EXPERTS,
        "num_trainable_per_layer": NUM_TRAINABLE,
        "ep_size": ep,
        "local_experts": NUM_EXPERTS // ep,
        "local_trainable": NUM_TRAINABLE // ep,
    }
