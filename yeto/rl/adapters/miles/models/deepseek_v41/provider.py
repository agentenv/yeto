"""Read-only provider view of a DeepSeek-V4.1-Flash checkpoint's ``config.json``.

Megatron-Bridge's AutoBridge cannot read ``deepseek_v41`` (transformers has no such
model type; Miles registers it as a DeepseekV3Config alias whose decoder fields may
sit under ``text_config``) and Miles trains the model through its plugin spec
(``miles_plugins.models.deepseek_v4_1``), so the island reads the geometry
``run_config.resolve_rl_run_config`` needs from the checkpoint itself (the same
approach as the kimi_k31 and glm5_next views). Missing fields take Miles'
defaults (``miles_plugins/mbridge/deepseek_v4_1.py``: ``kv_lora_rank`` and
``v_head_dim`` default to ``head_dim``; every layer is MoE).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

MODEL_TYPE = "deepseek_v41"
HF_MODEL_TYPES = frozenset({MODEL_TYPE})
ARCHITECTURES = frozenset({"DeepseekV41ForCausalLM"})


@dataclasses.dataclass(frozen=True)
class DeepseekV41ModelProvider:
    """The fields ``run_config`` reads off a Megatron-Bridge provider."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    ffn_hidden_size: int
    num_moe_experts: int
    moe_ffn_hidden_size: int
    moe_router_topk: int
    moe_layer_freq: tuple[int, ...]
    moe_shared_expert_intermediate_size: int
    vocab_size: int
    seq_length: int
    max_position_embeddings: int
    layernorm_epsilon: float
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
    compress_ratios: tuple[int, ...] | None = None
    first_k_dense_replace: int = 0
    rotary_base: int = 10000
    rotary_percent: float = 1.0
    share_embeddings_and_output_weights: bool = False
    multi_latent_attention: bool = True
    qk_layernorm: bool = True
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    normalization: str = "RMSNorm"
    position_embedding_type: str = "rope"

    def finalize(self) -> None:  # provider API parity; nothing to derive
        return None


def flat_config(config: dict[str, Any]) -> dict[str, Any]:
    """The decoder config: ``text_config`` laid over the outer keys (Miles
    ``_normalize_deepseek_v41_kwargs``)."""
    text = config.get("text_config")
    if isinstance(text, dict):
        text = {k: v for k, v in text.items() if k != "model_type"}
        return {**text, **config}
    return dict(config)


def is_deepseek_v41_config(config: dict[str, Any]) -> bool:
    return (str(config.get("model_type")) in HF_MODEL_TYPES
            or bool(set(config.get("architectures") or ()) & ARCHITECTURES))


def is_deepseek_v41(model_path: str | Path) -> bool:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return False
    try:
        return is_deepseek_v41_config(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return False


def provider_view(config: dict[str, Any]) -> DeepseekV41ModelProvider:
    """The provider view of a DeepSeek-V4.1 ``config.json`` (module doc)."""
    if not is_deepseek_v41_config(config):
        raise ValueError(f"not a DeepSeek-V4.1 config: model_type={config.get('model_type')!r}")
    c = flat_config(config)
    layers = int(c["num_hidden_layers"])
    dense = int(c.get("first_k_dense_replace", 0) or 0)
    if dense:
        raise ValueError(f"first_k_dense_replace={dense}: every DeepSeek-V4.1 layer is MoE")
    heads = int(c["num_attention_heads"])
    head_dim = int(c.get("head_dim", 512))
    rope_dim = int(c.get("qk_rope_head_dim", 64))
    moe_ffn = int(c["moe_intermediate_size"])
    shared = int(c.get("n_shared_experts", 1) or 0)
    max_positions = int(c.get("max_position_embeddings", 1048576))
    # informational only (the recipe passes Miles' --csa-compress-ratios); an MTP layer may add entries
    ratios = c.get("compress_ratios")
    ratios = tuple(int(r) for r in ratios) if isinstance(ratios, (list, tuple)) else None
    return DeepseekV41ModelProvider(
        num_layers=layers,
        hidden_size=int(c["hidden_size"]),
        num_attention_heads=heads,
        num_query_groups=heads,
        kv_channels=head_dim,
        ffn_hidden_size=int(c.get("intermediate_size", moe_ffn)),
        num_moe_experts=int(c["n_routed_experts"]),
        moe_ffn_hidden_size=moe_ffn,
        moe_router_topk=int(c["num_experts_per_tok"]),
        moe_layer_freq=tuple([1] * layers),
        moe_shared_expert_intermediate_size=moe_ffn * shared,
        vocab_size=int(c["vocab_size"]),
        seq_length=max_positions,
        max_position_embeddings=max_positions,
        layernorm_epsilon=float(c.get("rms_norm_eps", 1e-6)),
        q_lora_rank=int(c["q_lora_rank"]),
        kv_lora_rank=int(c.get("kv_lora_rank", head_dim)),
        qk_head_dim=head_dim,  # Miles --qk-head-dim 512 (scripts/models/deepseek-v4.1.py)
        qk_pos_emb_head_dim=rope_dim,
        v_head_dim=int(c.get("v_head_dim", head_dim)),
        compress_ratios=ratios,
        rotary_base=int(c.get("rope_theta", 10000)),
        share_embeddings_and_output_weights=bool(c.get("tie_word_embeddings", False)),
    )


def deepseek_v41_provider(model_path: str | Path) -> DeepseekV41ModelProvider:
    return provider_view(json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8")))
