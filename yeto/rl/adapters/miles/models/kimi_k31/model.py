"""Kimi-K3.1-Pro constants and the partial-experts contract.

Numbers from the HF ``config.json`` (``text_config``), cross-checked against
Miles ``scripts/models/kimi-k3.py``: 93 layers of which 92 are MoE (layer 0
is dense), 896 routed experts, top-16 sigmoid routing with a correction bias
and no expert groups, latent MoE (3584) with 3072-wide experts, hidden 7168.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yeto.rl.adapters.miles.models.partial_experts.config import (
    IN_PLACE_PERMUTED,
    PartialExpertModelConfig,
    register_model,
)

HF_MODEL_ID = "moonshotai/Kimi-K3.1-Pro"
MODEL_TYPE = "kimi_k31"  # selection "model" field and the Miles plugin name
HF_MODEL_TYPES = ("kimi_k3", "kimi_k31")

NUM_LAYERS = 93
FIRST_K_DENSE_REPLACE = 1
NUM_MOE_LAYERS = NUM_LAYERS - FIRST_K_DENSE_REPLACE  # 92
NUM_EXPERTS = 896
TOPK = 16
NUM_TRAINABLE = 128
MOE_FFN_HIDDEN_SIZE = 3072
MOE_LATENT_SIZE = 3584
HIDDEN_SIZE = 7168
DENSE_FFN_HIDDEN_SIZE = 33792
SHARED_EXPERT_INTERMEDIATE_SIZE = 6144
NUM_ATTENTION_HEADS = 96
VOCAB_SIZE = 163840
EP_SIZE_DEFAULT = 8
EXPERT_PARAMETERS = 3 * MOE_FFN_HIDDEN_SIZE * MOE_LATENT_SIZE  # 33,030,144 per expert

# Megatron parameter-name fragment of a trainable expert (F2 split layout).
TRAINABLE_PARAM_FRAGMENT = ".mlp.experts.trainable."
TRAINABLE_PARAM_ATTR = "partial_expert"
HF_EXPERT_WEIGHT = "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.w{w}.weight"

MODEL_CONFIG = register_model(PartialExpertModelConfig(
    model_type=MODEL_TYPE,
    num_routed_experts=NUM_EXPERTS,
    num_moe_layers=NUM_MOE_LAYERS,
    trainable_strategy=IN_PLACE_PERMUTED,
    ep_size_default=EP_SIZE_DEFAULT,
    # GRPO: no master streaming; EDP=1 keeps the expert masters whole anyway,
    # and precision-aware / offloaded optimizers need the distributed one.
    allow_distributed_optimizer=True,
    requires_complete_master=False,
    trainable_param_attr=TRAINABLE_PARAM_ATTR,
    needs_engine_patch=False,
    num_trainable_per_layer=NUM_TRAINABLE,
    trainable_param_fragment=TRAINABLE_PARAM_FRAGMENT,
    routed_experts_dtype="uint16",
    notes=(
        "router and expert_bias frozen (--moe-router-bias-update-rate 0)",
        "experts permuted into EP slots from expert_selection.json at load",
        "export, R3 and checkpoints use original expert ids",
    ),
))


def moe_layers() -> tuple[int, ...]:
    """HF indices of the MoE layers (1..92)."""

    return tuple(range(FIRST_K_DENSE_REPLACE, NUM_LAYERS))


def hf_expert_weight_name(layer: int, expert: int, projection: int) -> str:
    """``w1`` / ``w3`` / ``w2`` of one original expert in the HF checkpoint."""

    if projection not in (1, 2, 3):
        raise ValueError("projection must be 1 (gate), 3 (up) or 2 (down)")
    if not FIRST_K_DENSE_REPLACE <= layer < NUM_LAYERS or not 0 <= expert < NUM_EXPERTS:
        raise ValueError(f"layer {layer} / expert {expert} outside the K3.1 MoE")
    return HF_EXPERT_WEIGHT.format(layer=layer, expert=expert, w=projection)


def _field(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def text_config(config: Any) -> Any:
    """The language-model part of a (possibly multimodal) HF config."""

    inner = _field(config, "text_config")
    return inner if inner is not None else config


def check_hf_config(config: Any) -> dict[str, int]:
    """Assert the checkpoint matches the contract the permutation proof relies on.

    Returns the checked geometry. The equivalence of the slot permutation
    needs ungrouped top-k (``num_expert_group`` / ``topk_group`` absent or 1)
    and the frozen correction bias; every size enters the export / release
    accounting.
    """

    text = text_config(config)
    model_type = str(_field(config, "model_type") or _field(text, "model_type") or "")
    if model_type not in HF_MODEL_TYPES:
        raise ValueError(f"model_type {model_type!r} is not Kimi-K3.1 ({HF_MODEL_TYPES})")
    expected = {
        "num_hidden_layers": NUM_LAYERS,
        "first_k_dense_replace": FIRST_K_DENSE_REPLACE,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_token": TOPK,
        "moe_intermediate_size": MOE_FFN_HIDDEN_SIZE,
        "routed_expert_hidden_size": MOE_LATENT_SIZE,
        "hidden_size": HIDDEN_SIZE,
    }
    aliases = {"num_experts": ("n_routed_experts",), "num_experts_per_token": ("num_experts_per_tok",)}
    got: dict[str, int] = {}
    for name, want in expected.items():
        value = _field(text, name)
        for alias in aliases.get(name, ()):
            if value is None:
                value = _field(text, alias)
        if value is None:
            raise ValueError(f"Kimi-K3.1 config lacks {name}")
        if int(value) != want:
            raise ValueError(f"Kimi-K3.1 config {name}={value}, expected {want}")
        got[name] = int(value)
    for name in ("num_expert_group", "n_group", "topk_group"):
        value = _field(text, name)
        if value not in (None, 1):
            raise ValueError(
                f"{name}={value}: grouped top-k breaks the slot-permutation equivalence; "
                "Kimi-K3.1 must route without expert groups"
            )
    return got


def moe_layer_count(config: Any) -> int:
    """``L_moe`` of the R3 array: ``num_hidden_layers - first_k_dense_replace`` of the text config.

    Kimi-K3.1-Pro: 93 - 1 = 92. The sampler (S1) reshapes ``meta_info.routed_experts``
    with this count; the selection generator's layer offset is ``first_k_dense_replace``.
    """

    text = text_config(config)
    layers = _field(text, "num_hidden_layers")
    dense = _field(text, "first_k_dense_replace", 0)
    if layers is None:
        raise ValueError("config lacks num_hidden_layers")
    count = int(layers) - int(dense or 0)
    if count <= 0:
        raise ValueError(f"no MoE layers: num_hidden_layers={layers}, first_k_dense_replace={dense}")
    return count
