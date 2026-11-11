"""Read-only provider view of a Kimi-K3 / K3.1 checkpoint's ``config.json``.

Megatron-Bridge's AutoBridge cannot read ``kimi_k3`` / ``kimi_k31`` (transformers
has no such model type) and Miles trains the model through its plugin spec
(``miles_plugins.models.kimi_k31``), so the island reads the geometry
``run_config.resolve_rl_run_config`` needs from the checkpoint's ``text_config``
(the same approach as the glm5_next and Flash-Next views). A cut-down
checkpoint (``Pinaster/Kimi-K3-4layer``: 4 layers, the dense layer 0 and three
MoE layers) differs from Kimi-K3.1-Pro only in ``num_hidden_layers`` and the
attention-layer tables, so the recipe's layer count follows the view.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from yeto.rl.adapters.miles.models.kimi_k31.model import HF_MODEL_TYPES, text_config

ARCHITECTURES = frozenset({"KimiK3ForConditionalGeneration", "KimiK31ForConditionalGeneration",
                           "KimiK3LinearForCausalLM", "KimiLinearForCausalLM"})


@dataclasses.dataclass(frozen=True)
class KimiK31ModelProvider:
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
    first_k_dense_replace: int
    moe_latent_size: int
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

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers - self.first_k_dense_replace

    def finalize(self) -> None:  # provider API parity; nothing to derive
        return None


def is_kimi_k31(model_path: str | Path) -> bool:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return False
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (str(config.get("model_type")) in HF_MODEL_TYPES
            or bool(set(config.get("architectures") or ()) & ARCHITECTURES))


def provider_view(config: dict) -> KimiK31ModelProvider:
    """The provider view of a Kimi-K3 / K3.1 ``config.json`` (module doc)."""
    model_type = str(config.get("model_type") or "")
    if model_type not in HF_MODEL_TYPES and not (set(config.get("architectures") or ()) & ARCHITECTURES):
        raise ValueError(f"not a Kimi-K3 / K3.1 config: model_type={model_type!r}")
    text = text_config(config)
    layers = int(text["num_hidden_layers"])
    dense = int(text.get("first_k_dense_replace", 1))
    if not 0 <= dense < layers:
        raise ValueError(f"first_k_dense_replace {dense} is outside 0..{layers - 1}")
    experts = text.get("num_experts", text.get("n_routed_experts"))
    if experts is None:
        raise ValueError("Kimi-K3.1 config lacks num_experts")
    for name in ("num_expert_group", "topk_group"):
        if text.get(name) not in (None, 1):
            raise ValueError(f"{name}={text[name]}: grouped top-k breaks the slot-permutation equivalence")
    max_positions = int(text["max_position_embeddings"])
    heads = int(text["num_attention_heads"])
    return KimiK31ModelProvider(
        num_layers=layers,
        hidden_size=int(text["hidden_size"]),
        num_attention_heads=heads,
        num_query_groups=int(text.get("num_key_value_heads", heads)),
        kv_channels=int(text.get("qk_nope_head_dim", text.get("v_head_dim", 128)))
        + int(text.get("qk_rope_head_dim", 0)),
        ffn_hidden_size=int(text["intermediate_size"]),
        num_moe_experts=int(experts),
        moe_ffn_hidden_size=int(text["moe_intermediate_size"]),
        moe_router_topk=int(text.get("num_experts_per_token", text.get("num_experts_per_tok"))),
        moe_layer_freq=tuple([0] * dense + [1] * (layers - dense)),
        moe_shared_expert_intermediate_size=int(text["moe_intermediate_size"]) * int(text.get("num_shared_experts", 1)),
        vocab_size=int(text["vocab_size"]),
        seq_length=max_positions,
        max_position_embeddings=max_positions,
        layernorm_epsilon=float(text.get("rms_norm_eps", 1e-5)),
        q_lora_rank=int(text["q_lora_rank"]),
        kv_lora_rank=int(text["kv_lora_rank"]),
        qk_head_dim=int(text.get("qk_nope_head_dim", 128)),
        qk_pos_emb_head_dim=int(text.get("qk_rope_head_dim", 64)),
        v_head_dim=int(text["v_head_dim"]),
        first_k_dense_replace=dense,
        moe_latent_size=int(text.get("routed_expert_hidden_size", text["moe_intermediate_size"])),
        share_embeddings_and_output_weights=bool(config.get("tie_word_embeddings", False)
                                                 or text.get("tie_word_embeddings", False)),
    )


def kimi_k31_provider(model_path: str | Path) -> KimiK31ModelProvider:
    return provider_view(json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8")))
