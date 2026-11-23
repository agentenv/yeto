"""ModelSyncProfile (yeto/rl/model_registry/sync_profile.py): golden profiles of Qwen3
dense, Qwen3-MoE, DeepSeek-V3 and Kimi-K3.1 configs, the named missing contract items,
the upstream converter/bridge constraints, and the expert-subset round trip."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.rl import moe_geometry
from yeto.rl.model_registry import upstream
from yeto.rl.model_registry.sync_profile import (
    EXPERT_LORA_SUPPORTED,
    LAYOUT_FIRST_EXPORT,
    LAYOUT_OFFLINE,
    ModelContractError,
    derive_model_sync_profile,
    profile_dict,
    profile_problems,
)

DATA = Path(__file__).parent / "data" / "model_registry"
CONFIGS = DATA / "configs"
GOLDEN = DATA / "sync_profiles.json"
SNAPSHOT = {"qwen3_8b": "generic", "qwen3_30b_a3b": "generic", "deepseek_v3": "generic", "kimi_k31": "kimi_k31"}


def _cfg(name: str) -> dict:
    path = CONFIGS / f"{name}.json"
    return json.loads((path if path.exists() else DATA / f"{name}.json").read_text())


def test_golden_profiles():
    """Edit the derivation -> regenerate with YETO_UPDATE_GOLDEN=1 and review the diff."""
    snap = {name: profile_dict(derive_model_sync_profile(_cfg(name), recipe=recipe))
            for name, recipe in SNAPSHOT.items()}
    snap = json.loads(json.dumps(snap))
    if os.environ.get("YETO_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    assert snap == json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", sorted(p.stem for p in CONFIGS.glob("*.json")) + ["kimi_k31"])
def test_moe_geometry_comes_from_the_one_config_reader(name):
    cfg = _cfg(name)
    profile = derive_model_sync_profile(cfg, recipe="kimi_k31" if name == "kimi_k31" else "generic") \
        if not profile_problems(cfg) or name == "kimi_k31" else None
    geometry = moe_geometry.from_hf_config(cfg)
    if profile is None:
        return
    assert profile.moe_layers == (geometry.moe_layers if geometry else ())
    assert profile.num_routed_experts == (geometry.num_experts if geometry else 0)
    assert profile.experts_per_token == (geometry.router_topk if geometry else None)


def test_dense_and_moe_targets_and_shapes():
    dense = derive_model_sync_profile(_cfg("qwen3_8b"))
    assert dense.lora_target_modules["attention"] == ("q_proj", "k_proj", "v_proj", "o_proj")
    assert dense.lora_target_modules["mlp"] == ("gate_proj", "up_proj", "down_proj")
    assert "experts" not in dense.lora_target_modules and dense.expert_lora.startswith("unsupported:")
    q = next(m for m in dense.linear_modules if m.name == "q_proj")
    assert (q.in_features, q.out_features, q.path, len(q.layers)) == (4096, 32 * 128,
                                                                       "model.layers.{layer}.self_attn.q_proj", 36)
    moe = derive_model_sync_profile(_cfg("qwen3_30b_a3b"))
    experts = [m for m in moe.linear_modules if m.kind == "experts"]
    assert [(m.name, m.in_features, m.out_features, m.experts) for m in experts] == [
        ("gate_proj", 2048, 768, 128), ("up_proj", 2048, 768, 128), ("down_proj", 768, 2048, 128)]
    assert experts[0].path == "model.layers.{layer}.mlp.experts.{expert}.gate_proj"
    assert moe.expert_lora == EXPERT_LORA_SUPPORTED and moe.layout_source == LAYOUT_OFFLINE
    assert "ep divides num_routed_experts=128" in moe.constraints


def test_mixtral_names_its_experts_w1_w3_w2_and_has_no_bridge():
    profile = derive_model_sync_profile(_cfg("mixtral_8x7b"))
    assert profile.num_routed_experts == 8 and len(profile.moe_layers) == 32
    assert profile.lora_target_modules["experts"] == ("w1", "w3", "w2")
    assert next(m for m in profile.linear_modules if m.name == "w1").path == \
        "model.layers.{layer}.block_sparse_moe.experts.{expert}.w1"
    assert not profile.megatron_bridge and profile.layout_source == LAYOUT_FIRST_EXPORT
    assert any("no bridge for MixtralForCausalLM" in c for c in profile.constraints)


@pytest.mark.parametrize("name, config_class", [
    ("mistral_7b", "MistralConfig"), ("mixtral_8x7b", "MixtralConfig"), ("olmoe_1b_7b", "OlmoeConfig"),
    ("gemma2_9b", "Gemma2Config"), ("gemma3_12b", "Gemma3Config")])
def test_no_miles_full_parameter_converter_is_a_constraint(name, config_class):
    profile = derive_model_sync_profile(_cfg(name))
    assert profile.miles_converter is None
    assert any(f"no Megatron->HF converter for {config_class}" in c for c in profile.constraints)


@pytest.mark.parametrize("name, converter", [
    ("qwen3_8b", "qwen2"), ("qwen3_30b_a3b", "qwen3moe"), ("llama3_1_8b", "llama"), ("deepseek_v3", "deepseekv3"),
    ("glm4_9b", "glm4"), ("glm4_5_air", "glm4moe"), ("kimi_k2", "deepseekv3")])
def test_the_converter_follows_miles_dispatch_order(name, converter):
    assert derive_model_sync_profile(_cfg(name)).miles_converter == converter


def test_converter_dispatch_matches_miles_substring_order():
    # glm4moe is matched before glm4, qwen3moe before qwen3 (miles _convert_to_hf_core)
    assert upstream.miles_full_parameter_converter("Glm4MoeConfig") == "glm4moe"
    assert upstream.miles_full_parameter_converter("Glm4MoeLiteConfig") == "deepseekv3"
    assert upstream.miles_full_parameter_converter("Qwen3MoeConfig") == "qwen3moe"
    assert upstream.miles_full_parameter_converter("MistralConfig") is None


def test_missing_contract_items_are_named_never_a_silent_fallback():
    with pytest.raises(ModelContractError, match="deepseek-v4-flash") as error:
        derive_model_sync_profile(_cfg("deepseek_v4_flash"))
    assert error.value.missing == profile_problems(_cfg("deepseek_v4_flash"))
    broken = {k: v for k, v in _cfg("qwen3_8b").items() if k not in ("num_hidden_layers", "architectures")}
    with pytest.raises(ModelContractError) as error:
        derive_model_sync_profile(broken)
    assert set(error.value.missing) >= {"config field num_hidden_layers", "config field architectures"}


def test_mla_and_gated_attention_targets():
    mla = derive_model_sync_profile(_cfg("deepseek_v3"))
    assert mla.lora_target_modules["attention"] == ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
                                                    "o_proj")
    assert not any("num_key_value_heads" in c for c in mla.constraints)  # MLA: no GQA split
    gated = derive_model_sync_profile({**_cfg("qwen3_8b"), "attn_output_gate": True})
    assert gated.lora_target_modules["attention"] == ("o_proj",)  # yeto.rl.export.target_modules
    assert next(m for m in gated.linear_modules if m.name == "q_proj").out_features == 2 * 32 * 128


def test_k31_recipe_carries_the_partial_expert_contract_and_subsets_round_trip():
    profile = derive_model_sync_profile(_cfg("kimi_k31"), recipe="kimi_k31")
    assert profile.partial_experts.num_trainable_per_layer == 128
    assert profile.layout_source == LAYOUT_FIRST_EXPORT
    assert "ep divides num_trainable_per_layer=128" in profile.constraints
    subset = profile.expert_subset({layer: range(128) for layer in profile.moe_layers})
    assert subset.num_experts == 896 and set(subset.trainable) == set(profile.moe_layers)
    with pytest.raises(ModelContractError, match="non-MoE layers"):
        profile.expert_subset({0: range(128)})  # layer 0 is dense (first_k_dense_replace)
    with pytest.raises(ModelContractError, match="128 experts each"):
        profile.expert_subset({profile.moe_layers[0]: range(64)})


def test_the_module_is_import_light():
    code = ("import sys; import yeto.rl.model_registry.sync_profile; "
            "print(sorted(m for m in ('torch', 'transformers', 'megatron', 'miles') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parent.parent)
    assert out.stdout.strip() == "[]"
