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

For the 120B Super checkpoints the island builds its provider with the same
bridge as the Miles trainer: :func:`register_miles_bridges` imports Miles'
Megatron-Bridge plugins before ``AutoBridge`` (:func:`needs_miles_bridge`).
Two consequences:

* MTP: the Super checkpoints carry one MTP block. The Miles bridge drops it
  (``mtp_num_layers = None``) unless ``MILES_NEMOTRONH_KEEP_MTP`` is set, as
  NeMo-RL's Super-VL recipe does (``num_nextn_predict_layers=0``): RL has no
  MTP loss, so the block would only decay under weight decay and be synced
  for nothing. :func:`check_provider` refuses a provider that still has MTP.
  The rollout engine keeps the checkpoint's MTP weights (unused without
  speculative decoding).
* VL: a ``NemotronH_Omni_Reasoning_V3`` checkpoint (Nemotron-3.5 Super VL)
  goes through ``MilesNemotronHLanguageOnlyBridge``: the language model is
  built and trained, the vision encoder and projector are not built (frozen).
  Weight updates carry ``language_model.*`` names, which the SGLang VL model
  loads; the engine keeps its own vision weights.

512 experts with top-22 need EP dividing 512; EP = TP x CP at DP=1
(:mod:`yeto.rl.moe_layout`). The island's fragment layout can be computed
offline (:mod:`.nemotron_h_layout`).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from yeto.rl.hybrid_params import VL_ARCHITECTURE

MILES_BRIDGE_PLUGINS = "miles_plugins.megatron_bridge"
KEEP_MTP_ENV = "MILES_NEMOTRONH_KEEP_MTP"
# Miles commit that adds MilesNemotronHLanguageOnlyBridge (branch yeto/s27-120b-prep).
MILES_VL_BRIDGE_COMMIT = "b7868074c"


class NemotronHRecipeError(ValueError):
    pass


def _checkpoint_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def _is_vl(config: dict[str, Any]) -> bool:
    return VL_ARCHITECTURE in (config.get("architectures") or ())


def _needs_miles_bridge(config: dict[str, Any]) -> bool:
    if _is_vl(config):
        return True
    return ("NemotronHForCausalLM" in (config.get("architectures") or ())
            and bool(config.get("num_nextn_predict_layers")))


def needs_miles_bridge(model_path: str | Path) -> bool:
    """A Nemotron-H checkpoint that the upstream Megatron-Bridge reads differently
    from the Miles trainer: a VL checkpoint (upstream ``AutoBridge`` refuses
    ``NemotronH_Omni_Reasoning_V3``) or one with an MTP block (upstream builds it,
    Miles drops it). Other Nemotron-H checkpoints (Nemotron-3 Nano) keep the
    upstream provider their GPU runs were validated with."""
    return _needs_miles_bridge(_checkpoint_config(model_path))


def register_miles_bridges(model_path: str | Path) -> bool:
    """Import Miles' Megatron-Bridge plugins before the island's ``AutoBridge`` when
    :func:`needs_miles_bridge`, so the recipe checks the model Miles builds (MTP
    dropped, a VL checkpoint read as its language model). Returns whether it did.
    Refuses a VL checkpoint when the image's Miles has no language-only VL bridge."""
    config = _checkpoint_config(model_path)
    if not _needs_miles_bridge(config):
        return False
    importlib.import_module(MILES_BRIDGE_PLUGINS)
    if _is_vl(config):
        plugin = importlib.import_module(f"{MILES_BRIDGE_PLUGINS}.nemotron_h")
        if getattr(plugin, "NEMOTRON_H_OMNI_ARCHITECTURE", None) != VL_ARCHITECTURE:
            raise NemotronHRecipeError(
                f"a {VL_ARCHITECTURE} checkpoint needs Miles' language-only VL bridge "
                f"(michaellchung/miles {MILES_VL_BRIDGE_COMMIT} or later); this image's Miles has none")
    return True


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
            "the Nemotron-H provider still has its MTP block: full-parameter RL would train it "
            "without an MTP loss and sync it. Build the provider through Miles' bridge "
            f"({MILES_BRIDGE_PLUGINS}, see register_miles_bridges) and leave {KEEP_MTP_ENV} unset")


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
