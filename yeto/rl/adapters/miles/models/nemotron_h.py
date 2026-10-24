"""Miles recipe for the NVIDIA Nemotron-H family (Mamba-2 + MoE + attention).

Miles trains Nemotron-H through Megatron-Bridge (``--megatron-to-hf-mode
bridge``): the model is built from the Bridge ``HybridModelProvider`` of the HF
checkpoint, and only the training knobs that
``miles/backends/megatron_utils/model_provider.py:_apply_bridge_runtime_config``
copies from args reach it. The Megatron argv still has to describe the same
model, because Megatron's argument checks and Miles' rollout/export code read
it. This module writes those flags from the provider, so one recipe covers
the family (pattern-driven, not one checkpoint):

* router (Bridge ``nemotron_h_bridge.py`` provider_bridge): sigmoid score,
  expert bias, ``seq_aux_loss``, fp32 router, grouped GEMM, the top-k scaling
  factor and group routing from the checkpoint;
* RL keeps the routing statistics fixed: ``--moe-router-bias-update-rate 0``
  and ``--moe-aux-loss-coeff 0`` (Miles ``scripts/models/nemotron-3-nano-30b-a3b.py``).
  Both are copied onto the provider by Miles, so the expert bias does not
  drift on an island and every island trains the same router;
* latent MoE (``moe_latent_size``, Nemotron-3 Super / 3.5 Super VL) is passed
  through.

Differences that matter for the 120B Super (VL) and are NOT handled here:

* MTP: the Super checkpoints carry one MTP layer; the Bridge provider builds it
  (``mtp_num_layers``) and Miles does not copy an MTP override in bridge mode.
  :func:`check_provider` refuses MTP in full-parameter RL until the island
  drops it (NeMo-RL's Super-VL recipe sets ``num_nextn_predict_layers=0``).
* 512 experts with top-22: covered by the same flags; EP must divide 512.
* VL: the Bridge provider of ``NemotronH_Omni_Reasoning_V3`` is not this one;
  the vision freeze rule is :mod:`yeto.rl.hybrid_params`.

Training defaults (:func:`train_flags`, S27 G-AN on 8 x H200, Nemotron-3 Nano,
real RL samples): selective recompute of the MoE and layernorm modules, a
dynamic micro batch of 8192 tokens per GPU at TP1, and the DeepEP flex
dispatcher with permute fusion when the image has DeepEP (else all-to-all with
permute fusion). Measured: 9.7k tokens/s/GPU, 19.4% actor MFU with TP1/EP8/DP8,
no optimizer offload and 512 samples per step. Each default steps aside for
the matching yeto option (``--rl-activation-recompute``,
``--rl-max-tokens-per-gpu``), and pass-through Megatron flags come later in the
argv, so they win. :func:`train_warnings` names the two settings that cost the
most there: few samples per optimizer step and a full optimizer offload on a
small trainer.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any


class NemotronHRecipeError(ValueError):
    pass


def _value(provider: Any, name: str, default: Any = None) -> Any:
    value = getattr(provider, name, default)
    return default if value is None else value


def check_provider(provider: Any, *, parameter_mode: str) -> None:
    pattern = str(_value(provider, "hybrid_layer_pattern", ""))
    if not pattern:
        raise NemotronHRecipeError("Nemotron-H provider has no hybrid_layer_pattern")
    if "E" in pattern and _value(provider, "num_moe_experts") is None:
        raise NemotronHRecipeError("hybrid pattern has MoE layers (E) but the provider has no experts")
    if int(_value(provider, "mtp_num_layers", 0) or 0) and parameter_mode == "full":
        raise NemotronHRecipeError(
            "Nemotron-H checkpoint has MTP layers; full-parameter RL would train and sync them, "
            "and Miles bridge mode cannot drop them yet (120B Super gap)")


def model_flags(provider: Any) -> tuple[str, ...]:
    """Miles/Megatron model flags of a Nemotron-H provider (module docstring)."""
    flags: list[str] = ["--make-vocab-size-divisible-by", "128"]
    if _value(provider, "num_moe_experts") is None:
        return tuple(flags)
    flags += [
        "--moe-router-score-function", str(_value(provider, "moe_router_score_function", "sigmoid")),
        "--moe-router-load-balancing-type", str(_value(provider, "moe_router_load_balancing_type", "seq_aux_loss")),
        "--moe-router-dtype", str(_value(provider, "moe_router_dtype", "fp32")),
        "--moe-router-bias-update-rate", "0",
        "--moe-aux-loss-coeff", "0",
    ]
    if _value(provider, "moe_router_enable_expert_bias", False):
        flags.append("--moe-router-enable-expert-bias")
    if _value(provider, "moe_grouped_gemm", False):
        flags.append("--moe-grouped-gemm")
    scaling = _value(provider, "moe_router_topk_scaling_factor")
    if scaling is not None:
        flags += ["--moe-router-topk-scaling-factor", repr(float(scaling))]
    groups = _value(provider, "moe_router_num_groups")
    if groups is not None:
        flags += ["--moe-router-num-groups", str(int(groups)),
                  "--moe-router-group-topk", str(int(_value(provider, "moe_router_group_topk", 1)))]
    if _value(provider, "moe_router_pre_softmax", False):
        flags.append("--moe-router-pre-softmax")
    latent = _value(provider, "moe_latent_size")
    if latent is not None:
        flags += ["--moe-latent-size", str(int(latent))]
    return tuple(flags)


# G-AN (SESSION27/MFU-INVESTIGATION.md §3.1, §5): 8192 tokens per GPU was the
# fastest micro batch at TP1 (16384: -5%); TP2 at 16384 matched it.
TOKENS_PER_GPU_PER_TP = 8192
SELECTIVE_RECOMPUTE_MODULES = ("moe", "layernorm")
# Below this many samples per optimizer step the fixed per-step costs
# (optimizer step, log-prob pass, weight update) dominate: 128 -> 512 samples
# raised training throughput by 22% at TP1/EP8/DP8 (E4 -> E4L).
MIN_SAMPLES_PER_STEP = 256
# A full optimizer offload on one node: 2 GPUs spent 16.3 s of a 24.6 s actor
# step in the CPU optimizer (Y1); 8 GPUs ran 4.6x slower than without (Xoff).
FEW_TRAINER_GPUS = 8


def _has_moe(model_flags: tuple[str, ...]) -> bool:
    """:func:`model_flags` writes the router flags only for an MoE provider."""

    return "--moe-router-score-function" in model_flags


def deepep_available() -> bool:
    """DeepEP is importable here (the island builds its argv inside the training image)."""

    return importlib.util.find_spec("deep_ep") is not None


def train_flags(
    model_flags: tuple[str, ...],
    *,
    activation_recompute: bool,
    max_tokens_per_gpu: int | None,
    tensor_parallel: int,
    context_parallel: int,
    seq_len: int,
    deepep: bool,
) -> tuple[str, ...]:
    """The Nemotron-H training defaults (module docstring) for one island."""

    flags: list[str] = []
    if not activation_recompute:  # --rl-activation-recompute keeps full recompute
        flags += ["--recompute-granularity", "selective", "--recompute-modules", *SELECTIVE_RECOMPUTE_MODULES]
    if max_tokens_per_gpu is None:  # --rl-max-tokens-per-gpu keeps its own budget
        # A packed micro batch holds max_tokens_per_gpu * CP tokens and must fit one sample.
        tokens = max(TOKENS_PER_GPU_PER_TP * tensor_parallel, math.ceil(seq_len / context_parallel))
        flags += ["--use-dynamic-batch-size", "--max-tokens-per-gpu", str(tokens)]
    if _has_moe(model_flags):
        if deepep:
            flags += ["--moe-token-dispatcher-type", "flex", "--moe-flex-dispatcher-backend", "deepep"]
        else:
            flags += ["--moe-token-dispatcher-type", "alltoall"]
        flags.append("--moe-permute-fusion")
    return tuple(flags)


def train_warnings(*, samples_per_step: int, optimizer_offload_fraction: float, trainer_gpus: int) -> tuple[str, ...]:
    """Launch-time warnings for the two settings that cost most in G-AN's measurements."""

    warnings: list[str] = []
    if samples_per_step < MIN_SAMPLES_PER_STEP:
        warnings.append(
            f"each optimizer step trains {samples_per_step} samples; below {MIN_SAMPLES_PER_STEP} the fixed "
            "per-step costs (optimizer step, log-prob pass, weight update) dominate (S27 G-AN: 128 -> 512 "
            "samples per step raised training throughput by 22%)")
    if optimizer_offload_fraction >= 1.0 and trainer_gpus <= FEW_TRAINER_GPUS:
        warnings.append(
            f"the whole optimizer state is offloaded to the host on {trainer_gpus} trainer GPU(s); the CPU "
            "optimizer step dominates (S27 G-AN: 16.3 s of a 24.6 s actor step on 2 GPUs; 4.6x slower than "
            "no offload on 8 GPUs): prefer more trainer GPUs without --rl-optimizer-offload-fraction")
    return tuple(warnings)
