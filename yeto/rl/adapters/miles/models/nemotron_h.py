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
"""

from __future__ import annotations

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
