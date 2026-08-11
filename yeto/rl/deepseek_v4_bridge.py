"""Pinned Megatron-Bridge registration for Miles' DeepSeek-V4 model.

The Miles image contains the V4 MCore implementation and its legacy ``mbridge``
checkpoint bridge, but the installed NVIDIA ``megatron.bridge`` release does
not register ``DeepseekV4ForCausalLM``.  Yeto needs the latter bridge API for
direct HF loading and collective canonical LoRA import/export.  This module
adapts the already-pinned Miles implementation without treating V4 as V3.

Imports of Megatron/Miles are deliberately lazy.  Controller-side provenance
and plan tooling must remain usable without the training image installed.
"""

from __future__ import annotations

import copy
import importlib.abc
import importlib.machinery
import sys
from dataclasses import fields
from functools import partial
from types import ModuleType, SimpleNamespace
from typing import Any


_BRIDGE_CLASS: type | None = None
_CLONE_ROUTER_CLASS: type | None = None
_ENSURING_BRIDGE = False
_MILES_HELPER_FINDER = None
_MILES_HELPER_TARGET = "miles.backends.megatron_utils.bridge_lora_helpers"
_LAYER_COMPRESSION = {
    "sliding_attention": 0,
    "compressed_sparse_attention": 4,
    "heavily_compressed_attention": 128,
}


def _compression_ratios(config: Any) -> list[int]:
    """Return the per-decoder-layer compression ratios fail-closed."""

    ratios = getattr(config, "compress_ratios", None)
    if ratios:
        values = [int(value) for value in ratios]
    else:
        layer_types = getattr(config, "layer_types", None)
        if not isinstance(layer_types, (list, tuple)):
            raise ValueError("DeepSeek V4 config has no layer compression contract")
        try:
            values = [_LAYER_COMPRESSION[layer_type] for layer_type in layer_types]
        except KeyError as exc:
            raise ValueError(
                f"unsupported DeepSeek V4 attention layer type {exc.args[0]!r}"
            ) from exc
    expected = int(getattr(config, "num_hidden_layers", 0))
    if expected <= 0 or len(values) < expected:
        raise ValueError(
            "DeepSeek V4 compression ratios do not cover every decoder layer"
        )
    return values[:expected]


def _rope_scaling_contract(config: Any) -> dict[str, float | int]:
    raw_scaling = dict(getattr(config, "rope_scaling", None) or {})
    # Transformers 5 normalizes V4's two RoPE lanes into ``main`` and
    # ``compress`` dictionaries.  Miles' V4 kernel has one set of YaRN scalar
    # fields and disables interpolation explicitly for main/sliding layers, so
    # those fields must come from the compressed lane.
    scaling = dict(raw_scaling.get("compress", raw_scaling) or {})
    rope_type = scaling.get("rope_type", scaling.get("type"))
    contract = {
        "rotary_scaling_factor": float(scaling.get("factor", 0)),
        "original_max_position_embeddings": int(
            scaling.get("original_max_position_embeddings", 0)
        ),
        "beta_fast": int(scaling.get("beta_fast", 0)),
        "beta_slow": int(scaling.get("beta_slow", 0)),
    }
    if rope_type != "yarn" or contract != {
        "rotary_scaling_factor": 16.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }:
        raise ValueError(f"unexpected DeepSeek V4 YaRN contract: {raw_scaling}")
    return contract


def _checkpoint_parameter_name(name: str) -> str:
    """Translate Transformers V4 names to the pinned SGLang checkpoint names.

    Atlas/SGLang checkpoints use DeepSeek's compact ``wq_a``/``wq_b`` naming,
    while PEFT discovers modules from the Transformers model class as
    ``q_a_proj``/``q_b_proj``.  Mappings are declared with the Transformers
    names so exported policy state is a standard PEFT adapter; this function is
    used only when reading the immutable base checkpoint.
    """

    replacements = (
        (
            ".self_attn.compressor.indexer.position_bias",
            ".self_attn.indexer.compressor.ape",
        ),
        (
            ".self_attn.compressor.indexer.kv_proj.",
            ".self_attn.indexer.compressor.wkv.",
        ),
        (
            ".self_attn.compressor.indexer.gate_proj.",
            ".self_attn.indexer.compressor.wgate.",
        ),
        (
            ".self_attn.compressor.indexer.kv_norm.",
            ".self_attn.indexer.compressor.norm.",
        ),
        (
            ".self_attn.compressor.indexer.q_b_proj.",
            ".self_attn.indexer.wq_b.",
        ),
        (
            ".self_attn.compressor.indexer.scorer.weights_proj.",
            ".self_attn.indexer.weights_proj.",
        ),
        (".self_attn.compressor.position_bias", ".self_attn.compressor.ape"),
        (".self_attn.compressor.kv_proj.", ".self_attn.compressor.wkv."),
        (".self_attn.compressor.gate_proj.", ".self_attn.compressor.wgate."),
        (".self_attn.compressor.kv_norm.", ".self_attn.compressor.norm."),
        (".self_attn.q_a_proj.", ".self_attn.wq_a."),
        (".self_attn.q_a_norm.", ".self_attn.q_norm."),
        (".self_attn.q_b_proj.", ".self_attn.wq_b."),
        (".self_attn.kv_proj.", ".self_attn.wkv."),
        (".self_attn.o_a_proj.", ".self_attn.wo_a."),
        (".self_attn.o_b_proj.", ".self_attn.wo_b."),
        (".self_attn.sinks", ".self_attn.attn_sink"),
        (".attn_hc.fn", ".hc_attn_fn"),
        (".attn_hc.base", ".hc_attn_base"),
        (".attn_hc.scale", ".hc_attn_scale"),
        (".ffn_hc.fn", ".hc_ffn_fn"),
        (".ffn_hc.base", ".hc_ffn_base"),
        (".ffn_hc.scale", ".hc_ffn_scale"),
        ("model.hc_head.hc_fn", "model.hc_head_fn"),
        ("model.hc_head.hc_base", "model.hc_head_base"),
        ("model.hc_head.hc_scale", "model.hc_head_scale"),
        (".mlp.gate.tid2eid", ".mlp.topk.tid2eid"),
    )
    for source, destination in replacements:
        name = name.replace(source, destination)
    return name


def _load_hf_parameter(
    hf_param,
    hf_state_dict,
    *,
    balanced_experts: bool,
):
    """Load a physical Bridge task from the logical checkpoint namespace."""

    from .deepseek_v4_expert_clone import training_to_logical_expert_name

    def load(name: str):
        if balanced_experts:
            name = training_to_logical_expert_name(name)
        return hf_state_dict[_checkpoint_parameter_name(name)]

    if isinstance(hf_param, str):
        return load(hf_param)
    return {component: load(name) for component, name in hf_param.items()}


def _remap_expert_weights(weights, remap) -> dict[str, Any]:
    remapped = {}
    for name, value in weights.items():
        destination = remap(name)
        if destination in remapped:
            raise RuntimeError(f"duplicate remapped expert weight {destination!r}")
        remapped[destination] = value
    return remapped


def _logical_expert_names(
    names: list[str],
    *,
    balanced_experts: bool,
) -> list[str]:
    if not balanced_experts:
        return list(names)
    from .deepseek_v4_expert_clone import training_to_logical_expert_name

    return [training_to_logical_expert_name(name) for name in names]


def _logical_expert_weights(
    weights,
    *,
    balanced_experts: bool,
) -> dict[str, Any]:
    if not balanced_experts:
        return dict(weights)
    from .deepseek_v4_expert_clone import training_to_logical_expert_name

    return _remap_expert_weights(weights, training_to_logical_expert_name)


def _training_expert_weights(
    weights,
    *,
    balanced_experts: bool,
) -> dict[str, Any]:
    if not balanced_experts:
        return dict(weights)
    from .deepseek_v4_expert_clone import logical_to_training_expert_name

    return _remap_expert_weights(weights, logical_to_training_expert_name)


def _balanced_experts_from_config(config: Any) -> bool:
    if config is None:
        return False
    from .deepseek_v4_expert_clone import contract_from_config

    return contract_from_config(config) is not None


def _normalized_config(config: Any) -> Any:
    normalized = copy.deepcopy(config)
    ratios = _compression_ratios(normalized)
    normalized.compress_ratios = ratios
    normalized.window_size = int(
        getattr(normalized, "sliding_window", None) or 128
    )
    normalized.kv_lora_rank = int(
        getattr(normalized, "kv_lora_rank", None) or normalized.head_dim
    )
    normalized.qk_nope_head_dim = int(
        getattr(normalized, "qk_nope_head_dim", None)
        or normalized.head_dim - normalized.qk_rope_head_dim
    )
    normalized.v_head_dim = int(
        getattr(normalized, "v_head_dim", None) or normalized.head_dim
    )
    if getattr(normalized, "yeto_routed_expert_clone", None) is not None:
        # Native DeepSeek-V4 and the rollout runtime treat this checkpoint as
        # all-MoE (the signed clone contract has one source row per decoder
        # layer).  Miles' legacy HF alias injects DeepSeek-V3's default of
        # three dense prefix layers when the V4 JSON omits this field; undo
        # that alias-only default so trainer and rollout instantiate the same
        # 43 routed-expert layers.
        normalized.first_k_dense_replace = 0
    else:
        normalized.first_k_dense_replace = int(
            getattr(normalized, "first_k_dense_replace", None) or 0
        )
    rope_scaling = dict(getattr(normalized, "rope_scaling", None) or {})
    main_scaling = dict(rope_scaling.get("main", {}) or {})
    compress_scaling = dict(rope_scaling.get("compress", {}) or {})
    main_rope_theta = getattr(normalized, "rope_theta", None)
    if main_rope_theta is None:
        main_rope_theta = main_scaling.get("rope_theta")
    if main_rope_theta is None and not main_scaling:
        # Miles registers a legacy HF config alias before trainer setup.  That
        # alias exposes one flat RoPE dictionary: its ``rope_theta`` is the
        # main/sliding-lane base, while the compressed base remains in the
        # explicit top-level ``compress_rope_theta`` field.
        main_rope_theta = rope_scaling.get("rope_theta")
    compress_rope_theta = getattr(normalized, "compress_rope_theta", None)
    if compress_rope_theta is None:
        compress_rope_theta = compress_scaling.get("rope_theta")
    if main_rope_theta is None or compress_rope_theta is None:
        raise ValueError(
            "DeepSeek V4 config is missing the main/compressed RoPE bases"
        )
    # AutoBridge may hand the provider a Transformers-normalized config whose
    # top-level main-lane value was moved under ``rope_scaling.main``.  The
    # pinned Miles compatibility bridge still consumes both top-level fields.
    normalized.rope_theta = float(main_rope_theta)
    normalized.compress_rope_theta = float(compress_rope_theta)
    # The pinned legacy Miles bridge reads this key while Transformers 5
    # normalizes it away.  Restore it only for construction; the final provider
    # below resets the main/sliding RoPE base to config.rope_theta.
    rope_scaling["rope_theta"] = normalized.compress_rope_theta
    normalized.rope_scaling = rope_scaling
    return normalized


def _clone_router_class() -> type:
    """Build the Megatron clone-split router only inside the training image."""

    global _CLONE_ROUTER_CLASS
    if _CLONE_ROUTER_CLASS is not None:
        return _CLONE_ROUTER_CLASS

    from megatron.core.transformer.moe.router import TopKRouter

    from .deepseek_v4_expert_clone import (
        NUM_LAYERS,
        ORIGINAL_EXPERTS,
        expand_routes_torch,
    )

    class DeepSeekV4CloneSplitRouter(TopKRouter):
        """A 256-way gate dispatching to 288 independently stored experts."""

        def __init__(
            self,
            *,
            config,
            clone_sources,
            pg_collection=None,
            layer_number=None,
            is_mtp_layer=False,
        ) -> None:
            sources = tuple(tuple(int(expert) for expert in row) for row in clone_sources)
            if len(sources) != NUM_LAYERS:
                raise ValueError("DeepSeek V4 clone router requires 43 source rows")
            # Router weights, correction bias, hash tables, and replay remain
            # in the original 256-category space.  The enclosing MoELayer keeps
            # the unmodified 288-expert config and consumes the expanded map.
            router_config = copy.copy(config)
            router_config.num_moe_experts = ORIGINAL_EXPERTS
            super().__init__(
                config=router_config,
                pg_collection=pg_collection,
                layer_number=layer_number,
                is_mtp_layer=is_mtp_layer,
            )
            self._yeto_clone_sources = sources
            self._yeto_total_experts = int(config.num_moe_experts)

        def forward(self, input, padding_mask=None, input_ids=None):
            if self.is_mtp_layer:
                raise RuntimeError("Yeto's expanded checkpoint disables MTP")
            if input_ids is None:
                raise RuntimeError("clone-split routing requires input token IDs")
            if self.layer_number is None or not 1 <= int(self.layer_number) <= NUM_LAYERS:
                raise RuntimeError("clone-split router has no valid decoder layer number")
            base_probs, base_map = super().forward(
                input,
                padding_mask=padding_mask,
                input_ids=input_ids,
            )
            return expand_routes_torch(
                base_probs,
                base_map,
                input_ids,
                layer_id=int(self.layer_number) - 1,
                source_experts=self._yeto_clone_sources[int(self.layer_number) - 1],
            )

    _CLONE_ROUTER_CLASS = DeepSeekV4CloneSplitRouter
    return DeepSeekV4CloneSplitRouter


def _build_clone_router(*, clone_sources, **kwargs):
    """Pickle-safe MoESubmodules router builder used by the provider spec."""

    return _clone_router_class()(clone_sources=clone_sources, **kwargs)


def _layer_spec(config, vp_stage=None, *, clone_sources=None):
    from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec

    # This selects the small top-k operation used after the TileLang indexer;
    # it is not the V4 indexer implementation itself.  The pinned Miles image
    # accepts only ``torch`` or ``flashinfer`` here, and defaults to ``torch``.
    spec = get_dsv4_spec(
        SimpleNamespace(miles_dsa_topk_backend="torch"),
        config,
        vp_stage,
    )
    if clone_sources is None:
        return spec

    layer_specs = getattr(spec, "layer_specs", None)
    if not isinstance(layer_specs, list) or not layer_specs:
        raise RuntimeError("DeepSeek V4 provider returned no patchable layer specs")
    patched = 0
    for decoder_layer in layer_specs:
        layer_submodules = getattr(decoder_layer, "submodules", None)
        mlp_spec = getattr(layer_submodules, "mlp", None)
        moe_submodules = getattr(mlp_spec, "submodules", None)
        if moe_submodules is None or not hasattr(moe_submodules, "router"):
            raise RuntimeError("DeepSeek V4 decoder layer is not an MoE layer")
        moe_submodules.router = partial(
            _build_clone_router,
            clone_sources=clone_sources,
        )
        patched += 1
    if patched != len(layer_specs):
        raise RuntimeError("not every DeepSeek V4 layer received clone routing")
    return spec


def ensure_deepseek_v4_bridge() -> type:
    """Register and return the V4 bridge in the current training process."""

    global _BRIDGE_CLASS, _ENSURING_BRIDGE
    if _BRIDGE_CLASS is not None:
        return _BRIDGE_CLASS

    # The import hook is cheap and contract-gated: ordinary 256-expert V4
    # configs retain SGLang's original behavior.  Installing it here also
    # covers callers that add Yeto to PYTHONPATH after interpreter startup and
    # therefore missed sitecustomize.
    from .sglang_deepseek_v4_clone import install as install_sglang_clone_split

    # ``install_sglang_clone_split`` also arms the Ray-trainer import hook
    # below.  Guard this call so an already-imported Miles helper cannot
    # recursively re-enter bridge construction in the main learner process.
    _ENSURING_BRIDGE = True
    try:
        install_sglang_clone_split()
    finally:
        _ENSURING_BRIDGE = False

    import torch
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.conversion.param_mapping import (
        AutoMapping,
        ColumnParallelMapping,
        GatedMLPMapping,
        ReplicatedMapping,
    )
    from megatron.bridge.models.mla_provider import MLAModelProvider
    from megatron.core.models.gpt.gpt_model import GPTModel
    from mbridge.core.parallel_states import ParallelStates

    # Importing this package registers Miles' legacy V4 bridge, whose config
    # owns the pinned MCore extension fields used by the image.
    import miles_plugins.mbridge  # noqa: F401
    from miles_plugins.mbridge.deepseekv4 import DeepseekV4Bridge as LegacyV4

    class _AliasedAutoMapping(AutoMapping):
        def __init__(
            self,
            megatron_param: str,
            hf_param: str,
            permute_dims: tuple[int, ...] | None = None,
        ) -> None:
            super().__init__(megatron_param, hf_param, permute_dims)
            self.allow_hf_name_mismatch = True

    class _AliasedReplicatedMapping(ReplicatedMapping):
        def __init__(self, megatron_param: str, hf_param: str) -> None:
            super().__init__(megatron_param, hf_param)
            self.allow_hf_name_mismatch = True

    class _AliasedColumnMapping(ColumnParallelMapping):
        def __init__(self, megatron_param: str, hf_param: str) -> None:
            super().__init__(megatron_param, hf_param)
            self.allow_hf_name_mismatch = True

    @MegatronModelBridge.register_bridge(
        source="DeepseekV4ForCausalLM",
        target=GPTModel,
        provider=MLAModelProvider,
        model_type="deepseek_v4",
    )
    class DeepSeekV4Bridge(MegatronModelBridge):
        """NVIDIA Bridge facade for the pinned Miles V4 implementation."""

        def _uses_balanced_experts(self) -> bool:
            configured = getattr(self, "_yeto_balanced_experts", None)
            if configured is not None:
                return bool(configured)
            config = getattr(self, "hf_config", None)
            return _balanced_experts_from_config(config)

        def provider_bridge(self, hf_pretrained):
            from .deepseek_v4_expert_clone import contract_from_config

            hf_config = _normalized_config(hf_pretrained.config)
            # Pinned Miles' legacy V4 bridge rewrites ``hf_config.rope_theta``
            # to the compressed-lane base while constructing its config.  The
            # main/sliding lane must retain the checkpoint's top-level base;
            # capture it before invoking that mutating compatibility bridge.
            main_rope_theta = float(hf_config.rope_theta)
            clone_contract = contract_from_config(hf_config)
            self._yeto_balanced_experts = clone_contract is not None
            rope_scaling = _rope_scaling_contract(hf_config)
            legacy = LegacyV4(
                hf_config,
                parallel_states=ParallelStates(),
            )
            legacy_config = legacy.config
            valid = MLAModelProvider.__dataclass_fields__
            kwargs = {
                field.name: getattr(legacy_config, field.name)
                for field in fields(legacy_config)
                if field.name in valid
            }
            # The rollout does not enable speculative decoding.  Excluding MTP
            # avoids training and synchronizing an unused extra layer; expanded
            # checkpoints also attest that their unmodified MTP weights are
            # disabled in config.
            kwargs.update(
                vocab_size=int(hf_config.vocab_size),
                seq_length=min(int(hf_config.max_position_embeddings), 32768),
                position_embedding_type="rope",
                rotary_base=main_rope_theta,
                **rope_scaling,
                share_embeddings_and_output_weights=bool(
                    hf_config.tie_word_embeddings
                ),
                make_vocab_size_divisible_by=1280,
                transformer_layer_spec=partial(
                    _layer_spec,
                    clone_sources=(
                        None
                        if clone_contract is None
                        else clone_contract.source_experts_by_layer
                    ),
                ),
                virtual_pipeline_model_parallel_size=None,
                mtp_num_layers=None,
                mtp_enabled=False,
                dsv4_compress_ratios=_compression_ratios(hf_config),
                dsv4_compress_rope_theta=float(hf_config.compress_rope_theta),
                dsv4_window_size=int(hf_config.sliding_window),
                bf16=True,
                fp16=False,
                params_dtype=torch.bfloat16,
            )
            return MLAModelProvider(**kwargs)

        def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
            return _load_hf_parameter(
                hf_param,
                hf_state_dict,
                balanced_experts=self._uses_balanced_experts(),
            )

        def maybe_modify_converted_hf_weight(
            self,
            task,
            converted_weights_dict,
            hf_state_dict,
        ):
            converted = super().maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )
            return _logical_expert_weights(
                converted,
                balanced_experts=self._uses_balanced_experts(),
            )

        def _get_base_hf_param_names_for_adapter(
            self,
            mapping_registry,
            global_base_prefix,
            adapter_key,
            base_suffix,
        ):
            names = super()._get_base_hf_param_names_for_adapter(
                mapping_registry,
                global_base_prefix,
                adapter_key,
                base_suffix,
            )
            return _logical_expert_names(
                names,
                balanced_experts=self._uses_balanced_experts(),
            )

        def _merge_lora_adapter_weights(
            self,
            megatron_model,
            converted_weights_dict,
            adapter_weights,
        ):
            balanced = self._uses_balanced_experts()
            training_weights = _training_expert_weights(
                converted_weights_dict,
                balanced_experts=balanced,
            )
            merged = super()._merge_lora_adapter_weights(
                megatron_model,
                training_weights,
                adapter_weights,
            )
            return _logical_expert_weights(
                merged,
                balanced_experts=balanced,
            )

        def mapping_registry(self):
            mappings = [
                AutoMapping(
                    "embedding.word_embeddings.weight",
                    "model.embed_tokens.weight",
                ),
                AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
                AutoMapping("output_layer.weight", "lm_head.weight"),
                AutoMapping(
                    "decoder.layers.*.input_layernorm.weight",
                    "model.layers.*.input_layernorm.weight",
                ),
                AutoMapping(
                    "decoder.layers.*.pre_mlp_layernorm.weight",
                    "model.layers.*.post_attention_layernorm.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.wq_a.weight",
                    "model.layers.*.self_attn.q_a_proj.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.q_norm.weight",
                    "model.layers.*.self_attn.q_a_norm.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.wq_b.weight",
                    "model.layers.*.self_attn.q_b_proj.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.wkv.weight",
                    "model.layers.*.self_attn.kv_proj.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.kv_norm.weight",
                    "model.layers.*.self_attn.kv_norm.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.wo_a.weight",
                    "model.layers.*.self_attn.o_a_proj.weight",
                ),
                _AliasedAutoMapping(
                    "decoder.layers.*.self_attention.wo_b.weight",
                    "model.layers.*.self_attn.o_b_proj.weight",
                ),
                _AliasedColumnMapping(
                    "decoder.layers.*.self_attention.attn_sink",
                    "model.layers.*.self_attn.sinks",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.compressor.ape",
                    "model.layers.*.self_attn.compressor.position_bias",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.compressor.wkv.weight",
                    "model.layers.*.self_attn.compressor.kv_proj.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.compressor.wgate.weight",
                    "model.layers.*.self_attn.compressor.gate_proj.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.compressor.norm.weight",
                    "model.layers.*.self_attn.compressor.kv_norm.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.indexer.compressor.ape",
                    "model.layers.*.self_attn.compressor.indexer.position_bias",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.indexer.compressor.wkv.weight",
                    "model.layers.*.self_attn.compressor.indexer.kv_proj.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.indexer.compressor.wgate.weight",
                    "model.layers.*.self_attn.compressor.indexer.gate_proj.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.indexer.compressor.norm.weight",
                    "model.layers.*.self_attn.compressor.indexer.kv_norm.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.indexer.linear_wq_b.weight",
                    "model.layers.*.self_attn.compressor.indexer.q_b_proj.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.self_attention.indexer.linear_weights_proj.weight",
                    "model.layers.*.self_attn.compressor.indexer.scorer.weights_proj.weight",
                ),
                ReplicatedMapping(
                    "decoder.layers.*.mlp.router.weight",
                    "model.layers.*.mlp.gate.weight",
                ),
                ReplicatedMapping(
                    "decoder.layers.*.mlp.router.expert_bias",
                    "model.layers.*.mlp.gate.e_score_correction_bias",
                ),
                ReplicatedMapping(
                    "decoder.layers.*.mlp.router.tid2eid",
                    "model.layers.*.mlp.topk.tid2eid",
                ),
                GatedMLPMapping(
                    megatron_param=(
                        "decoder.layers.*.mlp.experts.linear_fc1.weight*"
                    ),
                    gate=(
                        "model.layers.*.mlp.experts.*.gate_proj.weight"
                    ),
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    "model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=(
                        "decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight"
                    ),
                    gate=(
                        "model.layers.*.mlp.experts.*.gate_proj.weight"
                    ),
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    "decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    "model.layers.*.mlp.experts.*.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=(
                        "decoder.layers.*.mlp.shared_experts.linear_fc1.weight"
                    ),
                    gate=(
                        "model.layers.*.mlp.shared_experts.gate_proj.weight"
                    ),
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                AutoMapping(
                    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    "model.layers.*.mlp.shared_experts.down_proj.weight",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.hc_attn_fn",
                    "model.layers.*.attn_hc.fn",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.hc_attn_base",
                    "model.layers.*.attn_hc.base",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.hc_attn_scale",
                    "model.layers.*.attn_hc.scale",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.hc_ffn_fn",
                    "model.layers.*.ffn_hc.fn",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.hc_ffn_base",
                    "model.layers.*.ffn_hc.base",
                ),
                _AliasedReplicatedMapping(
                    "decoder.layers.*.hc_ffn_scale",
                    "model.layers.*.ffn_hc.scale",
                ),
                _AliasedReplicatedMapping(
                    "decoder.hc_head_params.hc_head_fn",
                    "model.hc_head.hc_fn",
                ),
                _AliasedReplicatedMapping(
                    "decoder.hc_head_params.hc_head_base",
                    "model.hc_head.hc_base",
                ),
                _AliasedReplicatedMapping(
                    "decoder.hc_head_params.hc_head_scale",
                    "model.hc_head.hc_scale",
                ),
            ]
            return MegatronMappingRegistry(*mappings)

    _BRIDGE_CLASS = DeepSeekV4Bridge
    return DeepSeekV4Bridge


def install_on_miles_bridge_helpers(module: ModuleType) -> None:
    """Register V4 before a Ray trainer asks Miles to construct its model.

    Megatron-Bridge registrations are process-local.  The main learner's
    registration therefore does not reach Ray's freshly spawned trainer
    workers.  This installer runs after Miles' bridge helper is imported in
    each worker and registers the same audited bridge before its setup
    function can call ``AutoBridge.from_hf_pretrained``.
    """

    if getattr(module, "_yeto_deepseek_v4_bridge_installed", False):
        return
    if _ENSURING_BRIDGE:
        # The outer ensure call will finish registration in this process.
        return
    ensure_deepseek_v4_bridge()
    module._yeto_deepseek_v4_bridge_installed = True


class _MilesHelperLoader(importlib.abc.Loader):
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module) -> None:
        self.wrapped.exec_module(module)
        install_on_miles_bridge_helpers(module)


class _MilesHelperFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MILES_HELPER_TARGET:
            return None
        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot locate required module {fullname}")
        spec.loader = _MilesHelperLoader(spec.loader)
        return spec


def install_deepseek_v4_actor_bridge_hook() -> None:
    """Patch an imported Miles helper or arm its process-wide import hook."""

    global _MILES_HELPER_FINDER
    loaded = sys.modules.get(_MILES_HELPER_TARGET)
    if loaded is not None:
        install_on_miles_bridge_helpers(loaded)
    elif _MILES_HELPER_FINDER is None:
        _MILES_HELPER_FINDER = _MilesHelperFinder()
        sys.meta_path.insert(0, _MILES_HELPER_FINDER)
