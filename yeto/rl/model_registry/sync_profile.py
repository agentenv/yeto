"""The model contract of a fragment-synced (DiLoCo) run, derived from ``config.json``.

:func:`derive_model_sync_profile` reads an HF config (no weights) and returns a
:class:`ModelSyncProfile`: the MoE geometry (:mod:`yeto.rl.moe_geometry`, the one
config reader for it), the linear modules LoRA can target with their HF paths
and shapes, whether routed-expert LoRA is possible, the partial-expert contract
of a recipe, where the canonical layout comes from, and the constraints the
launcher and the upstream stack put on the parallel shape.

``--rl-model-recipe generic`` derives everything from the config. A recipe
(``kimi_k31``, ``deepseek-v4-flash``, ...) supplies what a config cannot say:
non-AutoBridge architectures, a partial-expert strategy, fixed EP/TP shapes.
A field that cannot be derived is a named missing contract item: the derive
raises :class:`ModelContractError` listing them, never a silent fallback;
:func:`profile_problems` returns them without raising.

Which source of LoRA shapes wins: for an architecture transformers can build,
the LoRA tensor contract is what PEFT discovers on empty weights
(``yeto.rl.export.derive_peft_lora_specs``); ``linear_modules`` is the offline
description for provider models transformers cannot build (Kimi-K3.1 and the
other recipe provider views) and for planning without torch. A test keeps the
two equal, names and shapes, for every fixture family transformers builds.
Routed experts are the exception: transformers 5 fuses them into one module
(``mlp.experts.gate_up_proj``), so their ``linear_modules`` carry the
checkpoint's per-expert names (``source="checkpoint"``), the names
Megatron-Bridge maps expert tensors to.

Import-light: stdlib, :mod:`yeto.rl.moe_geometry` and the model registry only;
the partial-expert registry is imported on demand.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from yeto.rl import moe_geometry

from . import upstream

LAYOUT_OFFLINE = "offline"          # the canonical layout follows from this profile
LAYOUT_FIRST_EXPORT = "first-export"  # learned from the island's first export (recipe provider views)
EXPERT_LORA_SUPPORTED = "supported"
RECIPE_GENERIC = "generic"
# recipe -> partial-expert model_type of yeto.rl.adapters.miles.models.partial_experts
PARTIAL_EXPERT_RECIPES = {"kimi_k31": "kimi_k31", "deepseek-v4-flash": "deepseek_v4"}
# recipes whose trainer reads a provider view instead of AutoBridge
PROVIDER_RECIPES = frozenset({"kimi_k31", "deepseek_v41", "glm5_next", "qwen3_8_next", "nemotron_h"})
# routed experts that transformers keeps in one fused module (no per-expert leaf to adapt)
FUSED_EXPERT_MODEL_TYPES = frozenset({"deepseek_v4", "gpt_oss", "llama4", "llama4_text"})
# routed-expert leaves (gate, up, down) when they are not gate_proj/up_proj/down_proj
_EXPERT_LEAVES = {"mixtral": ("w1", "w3", "w2")}
_EXPERT_BLOCK = {"mixtral": "block_sparse_moe"}
# dense MLPs whose gate and up projections are one module
_FUSED_GATE_UP = frozenset({"glm4", "phi3"})
_FUSED_QKV = frozenset({"phi3"})


class ModelContractError(ValueError):
    """A contract item of the model cannot be derived from its config and recipe."""

    def __init__(self, missing: Iterable[str]):
        self.missing = tuple(missing)
        super().__init__("model contract incomplete: " + "; ".join(self.missing))


@dataclass(frozen=True)
class LinearModule:
    """One linear projection family of the HF model.

    ``path`` is the module path with ``{layer}`` (and ``{expert}`` for routed
    experts) placeholders; ``layers`` are the decoder layers that have it."""

    kind: str  # attention | mlp | shared_experts | experts | mamba
    name: str  # HF leaf: q_proj, gate_proj, w1, ...
    in_features: int
    out_features: int
    path: str
    layers: tuple[int, ...]
    experts: int = 0  # routed experts per layer (kind == "experts")
    # "hf-module": an nn.Linear of the transformers model (PEFT finds it);
    # "checkpoint": a per-expert weight name of the checkpoint (transformers fuses it)
    source: str = "hf-module"


@dataclass(frozen=True)
class ModelSyncProfile:
    model_type: str
    architectures: tuple[str, ...]
    num_layers: int
    hidden_size: int
    moe_layers: tuple[int, ...]  # empty: dense
    num_routed_experts: int  # 0: dense
    experts_per_token: int | None
    # kind -> HF leaf names: attention, mlp, shared_experts, experts, all-linear
    lora_target_modules: Mapping[str, tuple[str, ...]]
    linear_modules: tuple[LinearModule, ...]
    expert_lora: str  # "supported" | "unsupported:<reason>"
    partial_experts: Any  # PartialExpertModelConfig | None
    layout_source: str  # offline | first-export
    recipe: str
    constraints: tuple[str, ...]
    missing: tuple[str, ...] = ()
    megatron_bridge: bool = field(default=False)
    miles_converter: str | None = None  # Megatron -> HF publication of full parameters (generic recipe)

    @property
    def moe(self) -> bool:
        return self.num_routed_experts > 0

    def expert_subset(self, trainable: Mapping[int, Iterable[int]]):
        """:class:`yeto.rl.moe_layout.ExpertSubset` of a partial-expert selection
        (MoE layer -> original expert ids), checked against this profile."""

        from yeto.rl.moe_layout import ExpertSubset

        subset = ExpertSubset.of(self.num_routed_experts, trainable)
        check_expert_subset(self, subset)
        return subset


def check_expert_subset(profile: ModelSyncProfile, subset: Any) -> None:
    """A selection fits the model: same router width, MoE layers only, and the
    recipe's per-layer trainable count when it declares one."""

    if not profile.moe:
        raise ModelContractError(["partial experts need a MoE model"])
    if subset.num_experts != profile.num_routed_experts:
        raise ModelContractError([f"selection has {subset.num_experts} experts, the model "
                                  f"{profile.num_routed_experts}"])
    stray = sorted(set(subset.trainable) - set(profile.moe_layers))
    if stray:
        raise ModelContractError([f"selection names non-MoE layers {stray[:8]}"])
    per_layer = getattr(profile.partial_experts, "num_trainable_per_layer", None)
    if per_layer is not None:
        uneven = sorted(layer for layer, ids in subset.trainable.items() if len(ids) != per_layer)
        if uneven:
            raise ModelContractError([f"layers {uneven[:8]} do not train {per_layer} experts each"])


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _int(text: Any, name: str, missing: list[str]) -> int:
    value = _get(text, name)
    if value is None:
        missing.append(f"config field {name}")
        return 0
    return int(value)


def _attention(text: Any) -> str:
    if _get(text, "index_topk") or _get(text, "index_n_heads"):
        return "dsa"
    if _get(text, "kv_lora_rank"):
        return "mla"
    if _get(text, "hybrid_override_pattern"):
        return "mamba-hybrid"
    if "linear_attention" in (_get(text, "layer_types") or ()) or _get(text, "linear_conv_kernel_dim"):
        return "gdn-hybrid"
    return "gqa"


def _attention_modules(text: Any, prefix: str, layers: tuple[int, ...], kind: str) -> list[LinearModule]:
    h = int(_get(text, "hidden_size"))
    heads = int(_get(text, "num_attention_heads"))
    kv = int(_get(text, "num_key_value_heads") or heads)
    head_dim = int(_get(text, "head_dim") or h // heads)
    path = f"{prefix}.{{layer}}.self_attn."
    if kind in ("mla", "dsa"):
        rope = int(_get(text, "qk_rope_head_dim") or 0)
        nope = int(_get(text, "qk_nope_head_dim") or head_dim)
        v_dim = int(_get(text, "v_head_dim") or nope)
        q_rank, kv_rank = _get(text, "q_lora_rank"), int(_get(text, "kv_lora_rank"))
        out = ([LinearModule("attention", "q_a_proj", h, int(q_rank), path + "q_a_proj", layers),
                LinearModule("attention", "q_b_proj", int(q_rank), heads * (nope + rope), path + "q_b_proj", layers)]
               if q_rank else [LinearModule("attention", "q_proj", h, heads * (nope + rope), path + "q_proj", layers)])
        if kind == "dsa":  # the sparse-attention indexer (DeepSeek V3.2); not an "attention" target
            n, d = int(_get(text, "index_n_heads")), int(_get(text, "index_head_dim"))
            out += [LinearModule("indexer", "wq_b", int(q_rank or h), n * d, path + "indexer.wq_b", layers),
                    LinearModule("indexer", "wk", h, d, path + "indexer.wk", layers),
                    LinearModule("indexer", "weights_proj", h, n, path + "indexer.weights_proj", layers)]
        return out + [
            LinearModule("attention", "kv_a_proj_with_mqa", h, kv_rank + rope, path + "kv_a_proj_with_mqa", layers),
            LinearModule("attention", "kv_b_proj", kv_rank, heads * (nope + v_dim), path + "kv_b_proj", layers),
            LinearModule("attention", "o_proj", heads * v_dim, h, path + "o_proj", layers),
        ]
    gate = 2 if _get(text, "attn_output_gate") else 1
    model_type = str(_get(text, "model_type") or "")
    if model_type in _FUSED_QKV:
        return [LinearModule("attention", "qkv_proj", h, (heads + 2 * kv) * head_dim, path + "qkv_proj", layers),
                LinearModule("attention", "o_proj", heads * head_dim, h, path + "o_proj", layers)]
    return [
        LinearModule("attention", "q_proj", h, gate * heads * head_dim, path + "q_proj", layers),
        LinearModule("attention", "k_proj", h, kv * head_dim, path + "k_proj", layers),
        LinearModule("attention", "v_proj", h, kv * head_dim, path + "v_proj", layers),
        LinearModule("attention", "o_proj", heads * head_dim, h, path + "o_proj", layers),
    ]


def _mlp_modules(kind: str, path: str, h_in: int, inner: int, layers: tuple[int, ...], *,
                 leaves: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
                 fused_gate_up: bool = False, experts: int = 0) -> list[LinearModule]:
    gate, up, down = leaves
    source = "checkpoint" if kind == "experts" else "hf-module"
    if fused_gate_up:
        return [LinearModule(kind, "gate_up_proj", h_in, 2 * inner, path + "gate_up_proj", layers, experts, source),
                LinearModule(kind, down, inner, h_in, path + down, layers, experts, source)]
    return [LinearModule(kind, gate, h_in, inner, path + gate, layers, experts, source),
            LinearModule(kind, up, h_in, inner, path + up, layers, experts, source),
            LinearModule(kind, down, inner, h_in, path + down, layers, experts, source)]


def _layer_prefix(config: Any, text: Any) -> str:
    """The HF decoder-layer prefix: ``model.layers``, or the language model's
    inside a multimodal ``...ForConditionalGeneration``."""

    if text is not config and any(str(a).endswith("ForConditionalGeneration")
                                  for a in (_get(config, "architectures") or ())):
        return "model.language_model.layers"
    return "model.layers"


def _targets(modules: tuple[LinearModule, ...], *, gated: bool) -> dict[str, tuple[str, ...]]:
    by_kind: dict[str, list[str]] = {}
    for module in modules:
        by_kind.setdefault(module.kind, [])
        if module.name not in by_kind[module.kind]:
            by_kind[module.kind].append(module.name)
    if gated:
        # yeto.rl.export.target_modules: Megatron cannot represent LoRA on a gated q_proj,
        # so "attention" is the output projection only.
        by_kind["attention"] = ["o_proj"]
    out = {kind: tuple(names) for kind, names in by_kind.items()}
    out["all-linear"] = tuple(dict.fromkeys(n for names in by_kind.values() for n in names))
    return out


def _partial_experts(recipe: str):
    model_type = PARTIAL_EXPERT_RECIPES.get(recipe)
    if model_type is None:
        return None
    from yeto.rl.adapters.miles.models.partial_experts.config import model_config

    return model_config(model_type)


def profile_problems(hf_config: Mapping[str, Any], *, recipe: str = RECIPE_GENERIC) -> tuple[str, ...]:
    """The missing contract items of a config under a recipe (empty: derivable)."""

    return _derive(hf_config, recipe=recipe).missing


def derive_model_sync_profile(hf_config: Mapping[str, Any], *, recipe: str = RECIPE_GENERIC) -> ModelSyncProfile:
    """The :class:`ModelSyncProfile` of an HF ``config.json``; raises
    :class:`ModelContractError` naming every item it cannot derive."""

    profile = _derive(hf_config, recipe=recipe)
    if profile.missing:
        raise ModelContractError(profile.missing)
    return profile


def _derive(config: Mapping[str, Any], *, recipe: str) -> ModelSyncProfile:
    text = moe_geometry.text_config(config)
    missing: list[str] = []
    constraints: list[str] = []
    model_type = str(_get(config, "model_type") or _get(text, "model_type") or "")
    text_type = str(_get(text, "model_type") or model_type)
    architectures = tuple(str(a) for a in (_get(config, "architectures") or _get(text, "architectures") or ()))
    if not architectures:
        missing.append("config field architectures")
    layers = _int(text, "num_hidden_layers", missing)
    hidden = _int(text, "hidden_size", missing)
    heads = _int(text, "num_attention_heads", missing)
    try:
        geometry = moe_geometry.from_hf_config(config)
    except moe_geometry.MoeGeometryError as error:
        missing.append(f"MoE layers: {error}")
        geometry = None
    moe_layers = geometry.moe_layers if geometry else ()
    experts = geometry.num_experts if geometry else 0
    attention = _attention(text)
    provider_recipe = recipe in PROVIDER_RECIPES
    if recipe == RECIPE_GENERIC:
        if text_type in ("deepseek_v4",) or model_type == "deepseek_v4":
            missing.append("DeepSeek V4 attention and fused clone experts: --rl-model-recipe deepseek-v4-flash")
        elif attention == "gdn-hybrid":
            missing.append("linear-attention (gated delta net) layers: a recipe such as qwen3_8_next "
                           "or the Qwen3.5 provider")
    modules: list[LinearModule] = []
    if not missing or (layers and hidden and heads):
        prefix = _layer_prefix(config, text)
        all_layers = tuple(range(layers))
        if attention == "mamba-hybrid":
            pattern = str(_get(text, "hybrid_override_pattern") or "")
            attn_layers = tuple(i for i, s in enumerate(pattern) if s == "*")
            modules += _attention_modules(text, prefix, attn_layers, "gqa")
            constraints.append("hybrid Mamba layers: the trainer reads the Nemotron-H provider; "
                               "the attention modules here cover the attention layers only")
        elif attention != "gdn-hybrid" and text_type != "deepseek_v4":
            modules += _attention_modules(text, prefix, all_layers, attention)
        inter = int(_get(text, "intermediate_size") or 4 * hidden)
        dense_layers = tuple(i for i in all_layers if i not in set(moe_layers))
        if dense_layers and attention != "mamba-hybrid":
            modules += _mlp_modules("mlp", f"{prefix}.{{layer}}.mlp.", hidden, inter, dense_layers,
                                    fused_gate_up=text_type in _FUSED_GATE_UP)
        if experts:
            moe_inter = int(_get(text, "moe_intermediate_size") or inter)
            latent = int(_get(text, "routed_expert_hidden_size") or hidden)
            shared = int(_get(text, "n_shared_experts") or _get(text, "num_shared_experts") or 0)
            shared_inter = int(_get(text, "shared_expert_intermediate_size") or 0) or shared * moe_inter
            block = _EXPERT_BLOCK.get(text_type, "mlp")
            if shared_inter:
                name = "shared_expert" if text_type == "qwen2_moe" else "shared_experts"
                modules += _mlp_modules("shared_experts", f"{prefix}.{{layer}}.{block}.{name}.", hidden,
                                        shared_inter, moe_layers)
            if text_type not in FUSED_EXPERT_MODEL_TYPES:
                modules += _mlp_modules("experts", f"{prefix}.{{layer}}.{block}.experts.{{expert}}.", latent,
                                        moe_inter, moe_layers, experts=experts,
                                        leaves=_EXPERT_LEAVES.get(text_type, ("gate_proj", "up_proj", "down_proj")))
    linear = tuple(modules)
    if heads:
        constraints.append(f"tp divides num_attention_heads={heads}")
    kv = int(_get(text, "num_key_value_heads") or heads or 0)
    if attention not in ("mla", "dsa") and kv and kv != heads:
        constraints.append(f"tp divides num_key_value_heads={kv} (Megatron num_query_groups)")
    if experts:
        constraints.append(f"ep divides num_routed_experts={experts}")
    if _get(text, "num_nextn_predict_layers"):
        constraints.append("MTP layers (num_nextn_predict_layers) are not trained")
    # routed-expert LoRA
    if not experts:
        expert_lora = "unsupported:dense model (no routed experts)"
    elif text_type in FUSED_EXPERT_MODEL_TYPES and recipe == RECIPE_GENERIC:
        expert_lora = "unsupported:transformers fuses the routed experts into one module (no per-expert leaf)"
    else:
        expert_lora = EXPERT_LORA_SUPPORTED
        if recipe in ("kimi_k31", "deepseek_v41"):
            constraints.append("vLLM samplers serve attention-only LoRA; routed-expert LoRA needs SGLang samplers")
        if recipe == "kimi_k31":
            constraints.append("synced LoRA trainers take --lora-targets attention only "
                               "(yeto/rl/engine/kimi_k31_lora.py check_tuning)")
        elif recipe == RECIPE_GENERIC:
            constraints.append("routed-expert LoRA (all-linear): one adapter per expert; under EP>1 grouped-GEMM "
                               "experts, BF16 experts, no capacity padding (yeto/rl/engine/run_config.py "
                               "check_generic_expert_lora)")
    # upstream trainer stack (generic recipe: AutoBridge and the converter of the config class)
    bridge = upstream.has_megatron_bridge(architectures)
    converter = next((c for c in (upstream.miles_full_parameter_converter(upstream.config_class_name(a))
                                  for a in architectures) if c), None)
    if recipe == RECIPE_GENERIC and architectures:
        if not bridge and attention != "mamba-hybrid":
            constraints.append(f"miles: Megatron-Bridge {upstream.MEGATRON_BRIDGE_COMMIT[:8]} has no bridge for "
                               f"{'/'.join(architectures)}; the Miles trainer cannot load it (verl FSDP2 can)")
        if converter is None:
            constraints.append(f"miles full-parameter publication: no Megatron->HF converter for "
                               f"{upstream.config_class_name(architectures[0])} (miles megatron_to_hf "
                               "_convert_to_hf_core); --tuning full and partial experts on Miles fail at the "
                               "first update_weights (verl, or LoRA)")
    partial = None
    try:
        partial = _partial_experts(recipe)
    except KeyError as error:
        missing.append(f"partial-expert contract: {error}")
    if partial is not None:
        if experts and partial.num_routed_experts != experts:
            missing.append(f"partial-expert contract has {partial.num_routed_experts} experts, "
                           f"the config {experts}")
        if partial.num_trainable_per_layer:
            constraints.append(f"ep divides num_trainable_per_layer={partial.num_trainable_per_layer}")
        if partial.needs_engine_patch:
            constraints.append("the rollout engine needs the recipe's routing patch (clone experts)")
    if recipe == "deepseek-v4-flash":
        constraints.append("tp==8, ep==8, 8-GPU rollout engines (run_config.py)")
    gated = bool(_get(text, "attn_output_gate"))
    layout = LAYOUT_FIRST_EXPORT if (provider_recipe or attention == "mamba-hybrid" or not bridge) else LAYOUT_OFFLINE
    return ModelSyncProfile(
        model_type=model_type, architectures=architectures, num_layers=layers, hidden_size=hidden,
        moe_layers=moe_layers, num_routed_experts=experts,
        experts_per_token=geometry.router_topk if geometry else None,
        lora_target_modules=MappingProxyType(_targets(linear, gated=gated)), linear_modules=linear,
        expert_lora=expert_lora, partial_experts=partial, layout_source=layout, recipe=recipe,
        constraints=tuple(constraints), missing=tuple(missing), megatron_bridge=bridge,
        miles_converter=converter)


def profile_dict(profile: ModelSyncProfile) -> dict[str, Any]:
    """A JSON-ready view (snapshots, ``yeto model check --json``)."""

    partial = profile.partial_experts
    return {
        "model_type": profile.model_type, "architectures": list(profile.architectures),
        "num_layers": profile.num_layers, "hidden_size": profile.hidden_size,
        "moe_layers": list(profile.moe_layers), "num_routed_experts": profile.num_routed_experts,
        "experts_per_token": profile.experts_per_token,
        "lora_target_modules": {k: list(v) for k, v in sorted(profile.lora_target_modules.items())},
        "linear_modules": [{"kind": m.kind, "name": m.name, "in": m.in_features, "out": m.out_features,
                            "path": m.path, "layers": len(m.layers), "experts": m.experts, "source": m.source}
                           for m in profile.linear_modules],
        "expert_lora": profile.expert_lora,
        "partial_experts": None if partial is None else {
            "model_type": partial.model_type, "strategy": partial.trainable_strategy,
            "num_routed_experts": partial.num_routed_experts,
            "num_trainable_per_layer": partial.num_trainable_per_layer,
            "ep_size_default": partial.ep_size_default},
        "layout_source": profile.layout_source, "recipe": profile.recipe,
        "constraints": list(profile.constraints), "missing": list(profile.missing),
        "megatron_bridge": profile.megatron_bridge, "miles_converter": profile.miles_converter,
    }
