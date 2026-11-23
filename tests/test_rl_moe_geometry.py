"""Routed-expert geometry from HF configs and Megatron-Bridge providers.

Each config carries only the fields its transformers decoder layer reads;
the expected MoE layers are what that decoder layer builds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.moe_geometry import (
    MoeGeometryError,
    RoutedExpertGeometry,
    from_bridge_provider,
    from_hf_config,
    hf_expert_of,
    megatron_moe_layer_mask,
)

CASES = {
    # Qwen3MoeDecoderLayer: not in mlp_only_layers and (i + 1) % decoder_sparse_step == 0.
    "qwen3_moe": ({"model_type": "qwen3_moe", "num_hidden_layers": 4, "num_experts": 8,
                   "num_experts_per_tok": 2, "decoder_sparse_step": 1, "mlp_only_layers": []},
                  (0, 1, 2, 3), 8, 2),
    "qwen2_moe_sparse_step": ({"model_type": "qwen2_moe", "num_hidden_layers": 6, "num_experts": 4,
                               "num_experts_per_tok": 2, "decoder_sparse_step": 2, "mlp_only_layers": [3]},
                              (1, 5), 4, 2),
    # MixtralDecoderLayer / OlmoeDecoderLayer: every layer routes.
    "mixtral": ({"model_type": "mixtral", "num_hidden_layers": 3, "num_local_experts": 8,
                 "num_experts_per_tok": 2}, (0, 1, 2), 8, 2),
    "olmoe": ({"model_type": "olmoe", "num_hidden_layers": 2, "num_experts": 64,
               "num_experts_per_tok": 8}, (0, 1), 64, 8),
    # DeepseekV3DecoderLayer / Glm4MoeDecoderLayer: i >= first_k_dense_replace (and i % freq == 0).
    "deepseek_v3": ({"model_type": "deepseek_v3", "num_hidden_layers": 6, "n_routed_experts": 16,
                     "num_experts_per_tok": 4, "first_k_dense_replace": 2, "moe_layer_freq": 1},
                    (2, 3, 4, 5), 16, 4),
    "deepseek_v2_freq2": ({"model_type": "deepseek_v2", "num_hidden_layers": 7, "n_routed_experts": 8,
                           "num_experts_per_tok": 2, "first_k_dense_replace": 1, "moe_layer_freq": 2},
                          (2, 4, 6), 8, 2),
    "glm4_moe": ({"model_type": "glm4_moe", "num_hidden_layers": 5, "n_routed_experts": 8,
                  "num_experts_per_tok": 2, "first_k_dense_replace": 1}, (1, 2, 3, 4), 8, 2),
    # Kimi-K3 nests the language model in text_config.
    "kimi_k31": ({"model_type": "kimi_k31", "text_config": {
        "num_hidden_layers": 4, "num_experts": 32, "num_experts_per_token": 4,
        "first_k_dense_replace": 1}}, (1, 2, 3), 32, 4),
    # Nemotron-H: "E" symbols of hybrid_override_pattern.
    "nemotron_h": ({"model_type": "nemotron_h", "num_hidden_layers": 6, "n_routed_experts": 8,
                    "num_experts_per_tok": 2, "hybrid_override_pattern": "ME*EME"}, (1, 3, 5), 8, 2),
    # Llama4TextDecoderLayer: (i + 1) % interleave_moe_layer_step == 0.
    "llama4": ({"model_type": "llama4", "text_config": {
        "num_hidden_layers": 4, "num_local_experts": 16, "num_experts_per_tok": 1,
        "interleave_moe_layer_step": 2}}, (1, 3), 16, 1),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_hf_config_moe_layers(name):
    config, layers, experts, topk = CASES[name]
    geometry = from_hf_config(config)
    assert geometry is not None
    assert geometry.moe_layers == layers
    assert geometry.num_experts == experts
    assert geometry.router_topk == topk
    assert geometry.num_layers == config.get("text_config", config)["num_hidden_layers"]
    assert geometry.model_type == config["model_type"]


def test_dense_config_has_no_geometry():
    assert from_hf_config({"model_type": "qwen3", "num_hidden_layers": 28}) is None
    assert from_hf_config({"model_type": "qwen2_moe", "num_hidden_layers": 2, "num_experts": 0}) is None


def test_megatron_mask_matches_megatron_semantics():
    assert megatron_moe_layer_mask(5, 1) == (1, 1, 1, 1, 1)
    assert megatron_moe_layer_mask(5, 2) == (1, 0, 1, 0, 1)
    assert megatron_moe_layer_mask(3, [0, 1, 1]) == (0, 1, 1)
    with pytest.raises(MoeGeometryError):
        megatron_moe_layer_mask(3, [0, 1])
    with pytest.raises(MoeGeometryError):
        megatron_moe_layer_mask(3, 0)


def test_bridge_provider_agrees_with_the_hf_config():
    hf = from_hf_config(CASES["deepseek_v3"][0])
    # AutoBridge's DeepSeek provider: moe_layer_freq as the per-layer mask.
    provider = SimpleNamespace(num_layers=6, num_moe_experts=16, moe_router_topk=4,
                               moe_layer_freq=[0, 0, 1, 1, 1, 1])
    bridge = from_bridge_provider(provider)
    hf.same_as(bridge)
    other = SimpleNamespace(num_layers=6, num_moe_experts=16, moe_router_topk=4, moe_layer_freq=1)
    with pytest.raises(MoeGeometryError, match="differs"):
        hf.same_as(from_bridge_provider(other))
    assert from_bridge_provider(SimpleNamespace(num_layers=4, num_moe_experts=None)) is None


def test_geometry_rejects_bad_values_and_checks_ep():
    with pytest.raises(MoeGeometryError):
        RoutedExpertGeometry(num_layers=2, moe_layers=(), num_experts=4)
    with pytest.raises(MoeGeometryError):
        RoutedExpertGeometry(num_layers=2, moe_layers=(2,), num_experts=4)
    geometry = RoutedExpertGeometry(num_layers=2, moe_layers=(0, 1), num_experts=6)
    assert geometry.check_ep_size(3) == 3
    with pytest.raises(MoeGeometryError):
        geometry.check_ep_size(4)
    with pytest.raises(MoeGeometryError, match="num_hidden_layers"):
        from_hf_config({"num_experts": 4})


@pytest.mark.parametrize("name, expected", [
    ("model.layers.3.mlp.experts.17.gate_proj.weight", (3, 17)),
    ("language_model.model.layers.1.block_sparse_moe.experts.5.w2.weight", (1, 5)),
    ("model.layers.2.mlp.shared_experts.gate_proj.weight", None),
    ("model.layers.2.mlp.gate.weight", None),
])
def test_hf_expert_names(name, expected):
    assert hf_expert_of(name) == expected
