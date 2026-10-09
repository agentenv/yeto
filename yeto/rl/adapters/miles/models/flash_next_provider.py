"""Qwen3.8-Flash-Next (``qwen4_exp``) provider view without Megatron-Bridge.

try22 (s11-h200-20261005v-fnboot) died in ``run_miles`` at
``megatron.bridge.AutoBridge.from_hf_pretrained``: transformers 5.12.1 (the
pin of both our ports image and radixark/miles:qwen38next) has no
``qwen4_exp`` model type.  Miles registers it at runtime
(``miles/utils/hf_utils/config.py::register_hf_config_aliases``, c35702e) and
trains Flash-Next through its own plugin (``--custom-model-provider-path``,
``miles_plugins/models/qwen3_8_next``), never through AutoBridge.

So for Flash-Next the learner does not touch ``megatron.bridge``: it registers
the Miles aliases, loads the HF config with ``AutoConfig`` and builds the
read-only provider view ``resolve_rl_run_config`` needs.  The model / LoRA
argv itself is replaced by ``profiles/qwen3_8_next.ports_recipe_argv``; the
view must still agree with it (fail-closed checks below, cross-checked in
``tests/test_rl_fn_provider_view.py``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

FLASH_NEXT_MODEL_TYPES = frozenset({"qwen4_exp", "qwen4_exp_text"})
GATED_DELTA_NET = "gated_delta_net"
# Megatron's full-attention cadence of the Miles plugin: 3 GDN + 1 QSA.
FULL_ATTENTION_INTERVAL = 4


@dataclasses.dataclass(frozen=True)
class Qwen4ExpModelProvider:
    """Read-only stand-in for a Megatron-Bridge provider (same class name as
    ``tests/multinode_gpu/fp_fn.fn_provider`` so ``provider_class`` matches)."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    ffn_hidden_size: int
    num_moe_experts: int
    moe_ffn_hidden_size: int
    moe_router_topk: int
    moe_layer_freq: int
    moe_shared_expert_intermediate_size: int
    vocab_size: int
    seq_length: int
    max_position_embeddings: int
    layernorm_epsilon: float
    rotary_base: int
    rotary_percent: float
    share_embeddings_and_output_weights: bool
    layer_types: tuple[str, ...]
    experimental_attention_variant: str = GATED_DELTA_NET
    attention_output_gate: bool = True
    qk_layernorm: bool = True
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    multi_latent_attention: bool = False
    normalization: str = "RMSNorm"
    position_embedding_type: str = "rope"

    def finalize(self) -> None:  # provider API parity; nothing to derive
        return None


def _config_model_type(model_path) -> str | None:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return None


def is_flash_next(args, model_path) -> bool:
    """Decided before any transformers / Bridge call: the HF repo basename or
    the raw ``config.json`` model_type."""
    from yeto.rl.engine.run_config import qwen3_8_next_variant

    if qwen3_8_next_variant(args, None) is not None:
        return True
    return _config_model_type(model_path) in FLASH_NEXT_MODEL_TYPES


def register_miles_hf_aliases() -> None:
    try:
        from miles.utils.hf_utils.config import register_hf_config_aliases
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3.8-Flash-Next (qwen4_exp) needs Miles' "
            "miles.utils.hf_utils.config.register_hf_config_aliases (Miles >= c35702e) "
            "to load its HF config; refusing to fall back to megatron.bridge AutoBridge, "
            "which cannot parse qwen4_exp"
        ) from exc
    register_hf_config_aliases()


def load_flash_next_config(model_path, *, trust_remote_code: bool = False):
    register_miles_hf_aliases()
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)


def _rope(text, name: str):
    params = getattr(text, "rope_parameters", None) or {}
    value = params.get(name) if isinstance(params, dict) else None
    if value is None:
        value = getattr(text, name, None)
    if value is None:
        raise ValueError(f"Flash-Next config has no {name}")
    return value


def provider_view(config) -> Qwen4ExpModelProvider:
    from yeto.rl.profiles import qwen3_8_next as q

    text = getattr(config, "text_config", None) or config
    model_type = getattr(config, "model_type", None)
    if model_type not in FLASH_NEXT_MODEL_TYPES:
        raise ValueError(f"not a Flash-Next config: model_type={model_type!r}")
    layers = int(text.num_hidden_layers)
    if layers not in q.NUM_LAYERS.values():
        raise ValueError(f"no Flash-Next variant has {layers} layers")
    layer_types = tuple(text.layer_types)
    expected = tuple(
        "full_attention" if (i + 1) % FULL_ATTENTION_INTERVAL == 0 else "linear_attention"
        for i in range(layers)
    )
    if layer_types != expected:
        raise ValueError(f"unexpected Flash-Next layer_types {layer_types}")
    if text.hidden_act != "silu":
        raise ValueError(f"Flash-Next expects SwiGLU (silu), got {text.hidden_act!r}")
    if getattr(text, "output_gate_type", None) != "sigmoid":
        raise ValueError("Flash-Next expects a sigmoid attention output gate")
    if bool(getattr(text, "attention_bias", False)):
        raise ValueError("Flash-Next expects bias-free linears")
    tied = bool(getattr(config, "tie_word_embeddings", False)) or bool(
        getattr(text, "tie_word_embeddings", False)
    )
    max_positions = int(text.max_position_embeddings)
    return Qwen4ExpModelProvider(
        num_layers=layers,
        hidden_size=int(text.hidden_size),
        num_attention_heads=int(text.num_attention_heads),
        num_query_groups=int(text.num_key_value_heads),
        kv_channels=int(text.head_dim),
        # every layer is MoE; Miles' model_args set --ffn-hidden-size to the
        # expert width
        ffn_hidden_size=int(text.moe_intermediate_size),
        num_moe_experts=int(text.num_experts),
        moe_ffn_hidden_size=int(text.moe_intermediate_size),
        moe_router_topk=int(text.num_experts_per_tok),
        moe_layer_freq=1,
        moe_shared_expert_intermediate_size=int(text.shared_expert_intermediate_size),
        vocab_size=int(text.vocab_size),
        seq_length=max_positions,
        max_position_embeddings=max_positions,
        layernorm_epsilon=float(text.rms_norm_eps),
        rotary_base=int(_rope(text, "rope_theta")),
        rotary_percent=float(_rope(text, "partial_rotary_factor")),
        share_embeddings_and_output_weights=tied,
        layer_types=layer_types,
    )


def flash_next_provider(args, model_path, *, trust_remote_code: bool = False):
    from yeto.rl.engine.run_config import qwen3_8_next_variant

    provider = provider_view(
        load_flash_next_config(model_path, trust_remote_code=trust_remote_code)
    )
    by_name = qwen3_8_next_variant(args, None)
    by_shape = qwen3_8_next_variant(_shape_only_args(), provider)
    if by_shape is None or (by_name is not None and by_name != by_shape):
        raise ValueError(
            f"Flash-Next config shape ({provider.num_layers} layers) does not match "
            f"the model name variant {by_name!r}"
        )
    return provider


def _shape_only_args():
    """Args stand-in with no model name, so the variant comes from the shape."""
    from types import SimpleNamespace

    return SimpleNamespace(model=None)


def target_modules() -> list[str]:
    """HF leaf names of the native plugin's LoRA targets (the argv itself is
    overridden by ``ports_recipe_argv``; same set as ``fp_fn``)."""
    from yeto.rl.profiles import qwen3_8_next as q

    return sorted({m.rsplit(".", 1)[-1] for m in q.LORA_TARGET_MODULES})
