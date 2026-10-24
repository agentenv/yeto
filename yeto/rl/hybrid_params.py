"""Parameter rules for hybrid Mamba-2 + MoE models (NVIDIA Nemotron-H family).

Covers the text models (``NemotronHForCausalLM``: Nemotron-3 Nano 30B-A3B,
Super 120B-A12B) and the VL model (``NemotronH_Omni_Reasoning_V3``:
Nemotron-3.5 Super VL 120B-A12B, a C-RADIO vision encoder plus an ``mlp1``
projector in front of a Nemotron-H language model).

Three rules, each a pure function on parameter names so that the trainer
side, the cut writer and the syncer layout apply the same rule:

* **Frozen vision.** With the vision part frozen, the vision encoder and the
  projector take no optimizer state and no syncer fragment. The rule is by
  name, independent of ``requires_grad``, so that a layout built before the
  trainer freezes anything is already correct. Names: HF ``vision_model.*``,
  ``mlp1.*``, ``vision_projector.*`` (Nemotron-3.5 Super VL weight index),
  ``sound*``; Megatron-Bridge Omni ``vision_model.*``, ``vision_projection.*``,
  ``sound_model.*``, ``sound_projection.*``. The Miles backend does not build
  these at all (``MilesNemotronHLanguageOnlyBridge``); the rule guards the
  layout.
* **f32 parameters.** Mamba-2 ``A_log``, ``D`` and ``dt_bias`` and the router
  score-correction bias must be f32 where a trainer keeps its *master*
  copy (the FSDP / verl path holds the master in the model).
  :func:`check_f32` refuses bf16 there. Megatron is different and passes by
  construction: every master is f32 (the full-parameter sync reads only FP32
  masters), while ``Float16Module`` casts the model copy of ``A_log``, ``D``,
  ``dt_bias`` to bf16 like every other weight (Megatron-LM miles-main
  ``transformer/module.py``: only ``mark_keep_in_fp32`` tensors stay f32) and
  the router ``expert_bias`` is an f32 buffer. The 120B checkpoint itself
  stores ``A_log``/``D``/``dt_bias`` in bf16 and the score-correction bias in
  f32 (safetensors headers).
* **Expert layout.** Routed experts must be complete per MoE layer: HF keeps
  them as one fused 3D tensor per projection (``mixer.experts.up_proj``,
  leading dim = number of experts); Megatron grouped GEMM names one tensor per
  expert (``decoder.layers.L.mlp.experts.linear_fc1.weightE``, the form
  :mod:`yeto.rl.moe_layout` reads). The MoE latent projections
  (``fc1_latent_proj`` / ``fc2_latent_proj``) and the shared expert are dense
  parameters, not routed experts.

Import-light: no torch import at module level.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .moe_layout import is_expert_parameter

# Wrapper prefixes in front of a model's own names (DDP / Float16Module /
# FSDP / HF composite models).
_WRAPPERS = re.compile(r"^(?:(?:module|_orig_mod|_fsdp_wrapped_module|model)\.)+")

VISION_PREFIXES = ("vision_model.", "vision_projection.", "vision_projector.", "mlp1.",
                   "sound_encoder.", "sound_model.", "sound_projection.")

# The Nemotron-H VL wrapper: the language model's config is ``llm_config``.
VL_ARCHITECTURE = "NemotronH_Omni_Reasoning_V3"

# HF layer kinds (current and legacy transformers names) -> pattern symbols.
_LAYER_SYMBOLS = {
    "mamba": "M", "linear_attention": "M",
    "attention": "*", "full_attention": "*",
    "moe": "E",
    "mlp": "-",
}

# Mamba-2 mixer vectors and the router bias that must stay f32.
_F32 = re.compile(
    r"(?:\.mixer\.(?:A_log|D|dt_bias)"           # Mamba-2 (HF and Megatron)
    r"|\.gate\.e_score_correction_bias"           # HF router bias
    r"|\.mlp\.router\.expert_bias)\Z"             # Megatron router bias
)

# HF fused experts: ``...layers.L.mixer.experts.{up_proj,down_proj}``.
_HF_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.mixer\.experts\.(up_proj|down_proj)\Z")
_MCORE_EXPERT = re.compile(r"decoder\.layers\.(\d+)\.mlp\.experts\.(linear_fc1|linear_fc2)\.weight(\d+)\Z")

# Vector-like hybrid tensors a caller may route to the avg fragment
# (``yeto.fragments.build_layout(avg_name_regex=...)``): Mamba vectors, conv
# kernels, norms and router biases. Whether RDA merging of these 1D tensors is
# worse than avg is not measured; this is the conservative choice.
HYBRID_AVG_NAME_REGEX = (
    r"(?:\.mixer\.(?:A_log|D|dt_bias|conv1d\.(?:weight|bias)|conv1d_weight|conv1d_bias|norm\.weight)"
    r"|e_score_correction_bias|router\.expert_bias|norm\.weight|layernorm\.weight)\Z"
)


def strip_wrappers(name: str) -> str:
    return _WRAPPERS.sub("", name)


def is_frozen_vision(name: str) -> bool:
    """A vision-encoder / projector / audio parameter of a Nemotron VL model."""
    return strip_wrappers(name).startswith(VISION_PREFIXES)


def requires_f32(name: str) -> bool:
    return _F32.search(strip_wrappers(name)) is not None


def trainable_names(names: Iterable[str], *, freeze_vision: bool) -> list[str]:
    """Names that take optimizer state and syncer fragments, input order kept."""
    return [n for n in names if not (freeze_vision and is_frozen_vision(n))]


def layout_named_numels(named_numels: Sequence[tuple[str, int]], *,
                        freeze_vision: bool) -> list[tuple[str, int]]:
    """``(name, numel)`` pairs for :func:`yeto.fragments.build_layout`, frozen
    vision parameters removed. Refuses an input with nothing left to train."""
    kept = [(n, s) for n, s in named_numels if not (freeze_vision and is_frozen_vision(n))]
    if not kept:
        raise ValueError("every parameter is frozen: no language-model parameters in the layout")
    return kept


class HybridParamError(ValueError):
    """The trainer's parameters break a hybrid-model rule."""


def check_f32(named_dtypes: Mapping[str, str]) -> None:
    """Refuse f32-required parameters held in another dtype.

    ``named_dtypes`` maps names to dtype strings (``"torch.float32"`` or
    ``"float32"``)."""
    bad = sorted(n for n, d in named_dtypes.items()
                 if requires_f32(n) and str(d).replace("torch.", "") != "float32")
    if bad:
        raise HybridParamError(f"must stay float32 (Mamba A_log/D/dt_bias, router bias): {bad[:8]}"
                               + (f" ... and {len(bad) - 8} more" if len(bad) > 8 else ""))


def check_frozen_excluded(trainable: Iterable[str]) -> None:
    """Refuse a trainable / optimizer parameter list that still holds vision parameters."""
    bad = sorted(n for n in trainable if is_frozen_vision(n))
    if bad:
        raise HybridParamError(f"frozen vision parameters in the trainable set: {bad[:8]}")


@dataclass(frozen=True)
class ExpertCoverage:
    num_experts: int
    moe_layers: tuple[int, ...]


def check_expert_coverage(named_shapes: Mapping[str, Sequence[int]], *, num_experts: int) -> ExpertCoverage:
    """Every MoE layer holds all ``num_experts`` routed experts, each once.

    Accepts HF fused names (shape ``[E, ...]``) and Megatron per-expert names
    (global expert index in the name, e.g. after ``moe_layout`` rewriting).
    A mix of both conventions is refused."""
    hf: dict[tuple[int, str], int] = {}
    mcore: dict[tuple[int, str], set[int]] = {}
    for raw, shape in named_shapes.items():
        name = strip_wrappers(raw)
        if (m := _HF_EXPERTS.search(name)) is not None:
            hf[(int(m[1]), m[2])] = int(shape[0])
        elif (m := _MCORE_EXPERT.search(name)) is not None:
            assert is_expert_parameter(name)
            key = (int(m[1]), m[2])
            expert = int(m[3])
            if expert in mcore.setdefault(key, set()):
                raise HybridParamError(f"expert {expert} of {key} appears twice")
            mcore[key].add(expert)
    if hf and mcore:
        raise HybridParamError("HF fused and Megatron per-expert names mixed in one layout")
    if not hf and not mcore:
        raise HybridParamError("no routed-expert parameters found")
    layers: set[int] = set()
    for (layer, proj), count in hf.items():
        if count != num_experts:
            raise HybridParamError(f"layer {layer} {proj}: {count} experts, expected {num_experts}")
        layers.add(layer)
    for (layer, proj), experts in mcore.items():
        if experts != set(range(num_experts)):
            missing = sorted(set(range(num_experts)) - experts)[:8]
            raise HybridParamError(f"layer {layer} {proj}: {len(experts)} of {num_experts} experts, missing {missing}")
        layers.add(layer)
    projections = {p for _, p in (hf or mcore)}
    for layer in layers:
        for proj in projections:
            if (layer, proj) not in (hf or mcore):
                raise HybridParamError(f"layer {layer} has no {proj} experts")
    return ExpertCoverage(num_experts=num_experts, moe_layers=tuple(sorted(layers)))


def language_model_config(config: Mapping) -> Mapping:
    """The NemotronH language-model part of an HF config dict: ``llm_config``
    of a VL checkpoint, the config itself for a text checkpoint."""
    if VL_ARCHITECTURE in (config.get("architectures") or ()) or "llm_config" in config:
        llm = config.get("llm_config")
        if not isinstance(llm, Mapping):
            raise HybridParamError(f"{VL_ARCHITECTURE} config has no llm_config")
        return llm
    return config


def is_vl_config(config: Mapping) -> bool:
    return language_model_config(config) is not config


def hybrid_pattern(config: Mapping) -> str:
    """The layer pattern (M mamba, E MoE, * attention, - MLP) of a NemotronH
    config: ``hybrid_override_pattern`` (Nemotron-3 Nano), else the
    ``layers_block_type`` list (Nemotron-3.5 Super VL ``llm_config``)."""
    llm = language_model_config(config)
    pattern = llm.get("hybrid_override_pattern")
    if pattern:
        return str(pattern)
    kinds = llm.get("layers_block_type")
    if not kinds:
        raise HybridParamError("config has neither hybrid_override_pattern nor layers_block_type")
    unknown = sorted(set(kinds) - set(_LAYER_SYMBOLS))
    if unknown:
        raise HybridParamError(f"unknown Nemotron-H layer kinds {unknown}")
    return "".join(_LAYER_SYMBOLS[kind] for kind in kinds)


def layer_counts(hybrid_override_pattern: str) -> dict[str, int]:
    """Layer kinds of a Nemotron-H ``hybrid_override_pattern`` (M mamba, E MoE,
    * attention, - dense MLP)."""
    kinds = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
    unknown = set(hybrid_override_pattern) - set(kinds)
    if unknown:
        raise ValueError(f"unknown layer symbols {sorted(unknown)} in hybrid_override_pattern")
    return {v: hybrid_override_pattern.count(k) for k, v in kinds.items()}
