from __future__ import annotations

from types import SimpleNamespace

import pytest

import yeto.rl.deepseek_v4_bridge as bridge_module
from yeto.rl.deepseek_v4_bridge import (
    _checkpoint_parameter_name,
    _compression_ratios,
    _normalized_config,
    _rope_scaling_contract,
)


def test_miles_trainer_helper_registers_v4_bridge_once(monkeypatch):
    calls = []
    helper = SimpleNamespace()
    monkeypatch.setattr(
        bridge_module,
        "ensure_deepseek_v4_bridge",
        lambda: calls.append("registered"),
    )

    bridge_module.install_on_miles_bridge_helpers(helper)
    bridge_module.install_on_miles_bridge_helpers(helper)

    assert calls == ["registered"]
    assert helper._yeto_deepseek_v4_bridge_installed is True


def test_v4_compression_ratios_accept_legacy_and_transformers_5_configs():
    legacy = SimpleNamespace(num_hidden_layers=4, compress_ratios=[0, 4, 128, 4])
    native = SimpleNamespace(
        num_hidden_layers=4,
        compress_ratios=None,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
    )

    assert _compression_ratios(legacy) == [0, 4, 128, 4]
    assert _compression_ratios(native) == [0, 4, 128, 4]


def test_v4_compression_ratios_reject_unknown_or_incomplete_layers():
    with pytest.raises(ValueError, match="unsupported"):
        _compression_ratios(
            SimpleNamespace(
                num_hidden_layers=1,
                compress_ratios=None,
                layer_types=["mystery_attention"],
            )
        )
    with pytest.raises(ValueError, match="cover every"):
        _compression_ratios(
            SimpleNamespace(num_hidden_layers=2, compress_ratios=[4])
        )


def test_v4_rope_scaling_contract_is_exact():
    expected = {
        "rotary_scaling_factor": 16.0,
        "original_max_position_embeddings": 65536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    assert _rope_scaling_contract(
        SimpleNamespace(
            rope_scaling={
                "rope_type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            }
        )
    ) == expected
    assert _rope_scaling_contract(
        SimpleNamespace(
            rope_scaling={
                "main": {"rope_type": "default", "rope_theta": 10000},
                "compress": {
                    "rope_type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32,
                    "beta_slow": 1,
                },
            }
        )
    ) == expected
    with pytest.raises(ValueError, match="YaRN"):
        _rope_scaling_contract(
            SimpleNamespace(
                rope_scaling={
                    "type": "yarn",
                    "factor": 1,
                    "original_max_position_embeddings": 65536,
                    "beta_fast": 32,
                    "beta_slow": 1,
                }
            )
        )


def test_v4_normalized_config_restores_top_level_rope_bases():
    normalized = _normalized_config(
        SimpleNamespace(
            num_hidden_layers=2,
            compress_ratios=[0, 4],
            sliding_window=128,
            kv_lora_rank=None,
            head_dim=512,
            qk_nope_head_dim=None,
            qk_rope_head_dim=64,
            v_head_dim=None,
            first_k_dense_replace=3,
            yeto_routed_expert_clone={"schema": 1},
            compress_rope_theta=None,
            rope_scaling={
                "main": {"rope_theta": 10_000},
                "compress": {"rope_theta": 160_000},
            },
        )
    )

    assert normalized.rope_theta == 10_000
    assert normalized.compress_rope_theta == 160_000
    assert normalized.first_k_dense_replace == 0
    assert normalized.rope_scaling["rope_theta"] == 160_000


def test_v4_normalized_config_accepts_miles_flat_alias_contract():
    normalized = _normalized_config(
        SimpleNamespace(
            num_hidden_layers=2,
            compress_ratios=[0, 4],
            sliding_window=128,
            kv_lora_rank=None,
            head_dim=512,
            qk_nope_head_dim=None,
            qk_rope_head_dim=64,
            v_head_dim=None,
            first_k_dense_replace=3,
            yeto_routed_expert_clone={"schema": 1},
            compress_rope_theta=160_000,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65_536,
                "beta_fast": 32,
                "beta_slow": 1,
                "rope_theta": 10_000,
            },
        )
    )

    assert normalized.rope_theta == 10_000
    assert normalized.compress_rope_theta == 160_000
    assert normalized.first_k_dense_replace == 0
    # The pinned legacy V4 bridge consumes this flat key for the compressed
    # lane after the main base has been captured above.
    assert normalized.rope_scaling["rope_theta"] == 160_000


@pytest.mark.parametrize(
    ("canonical", "checkpoint"),
    [
        (
            "model.layers.2.self_attn.q_a_proj.weight",
            "model.layers.2.self_attn.wq_a.weight",
        ),
        (
            "model.layers.2.self_attn.q_b_proj.weight",
            "model.layers.2.self_attn.wq_b.weight",
        ),
        (
            "model.layers.2.self_attn.compressor.indexer.q_b_proj.weight",
            "model.layers.2.self_attn.indexer.wq_b.weight",
        ),
        (
            "model.layers.2.self_attn.compressor.indexer.position_bias",
            "model.layers.2.self_attn.indexer.compressor.ape",
        ),
        (
            "model.layers.2.attn_hc.scale",
            "model.layers.2.hc_attn_scale",
        ),
        (
            "model.hc_head.hc_fn",
            "model.hc_head_fn",
        ),
        (
            "model.layers.0.mlp.gate.tid2eid",
            "model.layers.0.mlp.topk.tid2eid",
        ),
    ],
)
def test_v4_canonical_peft_names_resolve_to_pinned_checkpoint_names(
    canonical, checkpoint
):
    assert _checkpoint_parameter_name(canonical) == checkpoint
