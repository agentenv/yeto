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


def test_the_converter_table_names_the_miles_pin():
    """The converter table was read at the fork pin: a pin move must re-read it."""
    from yeto.rl.adapters.miles import pins

    assert upstream.MILES_COMMIT == pins.MILES_NEXT_COMMIT, (
        "MILES_NEXT_COMMIT moved: re-read megatron_to_hf/_convert_to_hf_core and the Megatron-Bridge pin "
        "of the new fork commit, then update yeto/rl/model_registry/upstream.py")


# Fixtures transformers builds without remote code (kimi_k2 needs it; DeepSeek V4 needs its recipe).
PEFT_FAMILIES = ("deepseek_v3", "deepseek_v32", "gemma2_9b", "gemma3_12b", "glm4_5_air", "glm4_9b", "llama3_1_8b",
                 "mistral_7b", "mixtral_8x7b", "olmoe_1b_7b", "qwen3_0p6b", "qwen3_30b_a3b", "qwen3_8b")


PEFT_MOVES_MLP_TO_FUSED_EXPERTS = frozenset({"deepseek_v3", "deepseek_v32", "glm4_5_air"})


def _small(cfg: dict) -> dict:
    """The same per-layer shapes with few layers and experts (meta-device build stays fast)."""
    cfg = json.loads(json.dumps(cfg))
    text = cfg.get("text_config", cfg)
    text["num_hidden_layers"] = max(2, int(text.get("first_k_dense_replace") or 0) + 2)
    for key in ("n_routed_experts", "num_experts", "num_local_experts"):
        if text.get(key):
            text[key] = 8
    if text.get("num_experts_per_tok"):
        text["num_experts_per_tok"] = min(text["num_experts_per_tok"], 2)
    if isinstance(text.get("layer_types"), list):
        text["layer_types"] = text["layer_types"][: text["num_hidden_layers"]]
    return cfg


@pytest.mark.parametrize("name", PEFT_FAMILIES)
def test_linear_modules_equal_what_peft_finds_on_empty_weights(name, tmp_path):
    """Names and shapes of every non-expert linear module equal PEFT's LoRA
    contract (``derive_peft_lora_specs``, all-linear) on the transformers model;
    the routed experts are fused there and keep their checkpoint names here."""
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")
    import re

    from yeto.rl.export import derive_peft_lora_specs

    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    from yeto.rl.export import _rl_model_from_config

    cfg = _small(_cfg(name))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    rank = 4
    # Every nn.Linear but the LM head, named explicitly: PEFT's "all-linear"
    # shorthand finds no target on a transformers-5 MoE (fused experts).
    with init_empty_weights():
        model = _rl_model_from_config(AutoConfig.from_pretrained(str(tmp_path)), trust_remote_code=False)
    linear = sorted(n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != "lm_head")
    specs = derive_peft_lora_specs(str(tmp_path), None, rank=rank, targets=linear)
    profile = derive_model_sync_profile(cfg)
    prefix = profile.linear_modules[0].path.split("{layer}")[0]
    peft: set[tuple[str, int, int]] = set()
    shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    for spec in specs:
        module, side = re.match(r"(?:base_model\.model\.)?(.+)\.(lora_[AB])\.weight$", spec.name).groups()
        shapes.setdefault(module, {})[side] = tuple(spec.shape)
    for module, sides in shapes.items():
        if not module.startswith(prefix) or re.search(r"\.mlp\.experts(\.base_layer)?$", module):
            # a vision tower; the fused routed experts (PEFT 0.19 also wraps the fused
            # experts parameters of DeepSeek-V3 / GLM-4.5 layouts: not a per-expert contract)
            continue
        layer = int(module[len(prefix):].split(".", 1)[0])
        template = module.replace(f"{prefix}{layer}.", prefix + "{layer}.", 1)
        assert sides["lora_A"][0] == sides["lora_B"][1] == rank
        peft.add((template, layer, sides["lora_A"][1], sides["lora_B"][0]))
    ours = {(m.path, layer, m.in_features, m.out_features)
            for m in profile.linear_modules if m.source == "hf-module" for layer in m.layers}
    if name in PEFT_MOVES_MLP_TO_FUSED_EXPERTS:
        # PEFT 0.19 on these transformers-5 layouts puts the MLP-named targets on the
        # fused routed experts and drops the dense and shared-expert MLP LoRA: only
        # attention is a PEFT contract there (pinned here so a PEFT change shows up).
        attention = {x for x in ours if ".self_attn." in x[0]}
        assert {x for x in peft if ".self_attn." in x[0]} == attention
        assert not {x for x in peft if ".mlp." in x[0]}
        return
    assert ours == peft
    experts = [m for m in profile.linear_modules if m.kind == "experts"]
    assert all(m.source == "checkpoint" for m in experts)
    assert bool(experts) == profile.moe
