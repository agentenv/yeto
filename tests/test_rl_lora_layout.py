"""Offline canonical LoRA layout (``yeto.rl.lora_layout``), routed experts included."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl import lora_layout
from yeto.rl.engine.run_config import check_generic_expert_lora
from yeto.rl.export import adapter_targets, derive_peft_lora_specs

TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16)
MOE = dict(moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2)


def _expert(layer: int, expert: int, proj: str, side: str) -> str:
    return f"base_model.model.model.layers.{layer}.mlp.experts.{expert}.{proj}.lora_{side}.weight"


@pytest.mark.parametrize("transposed", [False, True])
def test_fused_experts_get_one_adapter_per_expert(transposed):
    # [E, 2I, H] / [E, H, I] (Qwen3-MoE, DeepSeek-V3) or the transposed storage
    gate_up, down = ((3, 64, 20), (3, 10, 64)) if transposed else ((3, 20, 64), (3, 64, 10))
    shapes = [("model.layers.0.mlp.experts.gate_up_proj", gate_up), ("model.layers.0.mlp.experts.down_proj", down),
              ("model.layers.0.self_attn.q_proj.weight", (64, 64))]

    specs = {s.name: s.shape for s in lora_layout.fused_expert_lora_specs(shapes, rank=4, hidden_size=64)}

    assert len(specs) == 3 * 6
    assert specs[_expert(0, 2, "gate_proj", "A")] == (4, 64) and specs[_expert(0, 2, "gate_proj", "B")] == (10, 4)
    assert specs[_expert(0, 0, "up_proj", "B")] == (10, 4)
    assert specs[_expert(0, 1, "down_proj", "A")] == (4, 10) and specs[_expert(0, 1, "down_proj", "B")] == (64, 4)


def test_fused_experts_that_are_not_a_gated_mlp_are_refused():
    with pytest.raises(lora_layout.LoraLayoutError, match="without both"):
        lora_layout.fused_expert_lora_specs([("m.experts.gate_up_proj", (2, 8, 4))], rank=2, hidden_size=4)
    with pytest.raises(lora_layout.LoraLayoutError, match="one gated MLP"):
        lora_layout.fused_expert_lora_specs([("m.experts.gate_up_proj", (2, 6, 4)), ("m.experts.down_proj", (2, 4, 4))],
                                            rank=2, hidden_size=4)
    with pytest.raises(lora_layout.LoraLayoutError, match="no hidden-size"):
        lora_layout.fused_expert_lora_specs([("m.experts.gate_up_proj", (2, 6, 5)), ("m.experts.down_proj", (2, 4, 3))],
                                            rank=2, hidden_size=4)


def test_all_linear_skips_the_router_and_the_head():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = torch.nn.Linear(8, 4)  # router, next to the experts
            self.experts = torch.nn.Module()
            self.shared_expert = torch.nn.Module()
            self.shared_expert.gate_proj = torch.nn.Linear(8, 8)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.mlp, self.lm_head = torch.nn.Linear(8, 8), Block(), torch.nn.Linear(8, 16)

        def get_output_embeddings(self):
            return self.lm_head

    assert lora_layout.all_linear_modules(Model()) == ["mlp.shared_expert.gate_proj", "q_proj"]


def test_canonical_size_gate():
    spec = lora_layout.CanonicalTensorSpec("base_model.model.a.lora_A.weight", (4, 1024), "float32", 4096)
    assert lora_layout.check_canonical_size([spec], limit=4 * 4096) == 4 * 4096
    with pytest.raises(lora_layout.LoraLayoutError, match="full-parameter stack"):
        lora_layout.check_canonical_size([spec], limit=4 * 4096 - 1)


def _config(tmp_path, moe: bool):
    from transformers import Qwen3Config, Qwen3MoeConfig

    path = tmp_path / ("moe" if moe else "dense")
    config = (Qwen3MoeConfig(**TINY, **MOE, architectures=["Qwen3MoeForCausalLM"]) if moe
              else Qwen3Config(**TINY, architectures=["Qwen3ForCausalLM"]))
    config.save_pretrained(path)
    return str(path)


def test_offline_specs_adapt_every_routed_expert_and_no_router(tmp_path):
    specs = lora_layout.offline_canonical_specs(None, rank=4, targets="all-linear",
                                                model_path=_config(tmp_path, moe=True))
    names = {s.name: s.shape for s in specs}

    assert len(names) == 3 * (4 * 2 + 4 * 6)  # layers x (q/k/v/o + experts x gate/up/down) x (A, B)
    assert names[_expert(2, 3, "down_proj", "A")] == (4, 32)
    assert not [n for n in names if ".mlp.gate." in n or ".experts.base_layer." in n or ".experts.lora_" in n]
    assert lora_layout.has_expert_specs(specs)
    # Miles --target-modules: PEFT's own all-linear gave base_layer/experts/gate here,
    # which Miles' target resolution refuses ("LoRA targets have no Bridge mapping")
    assert adapter_targets(specs) == ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]


def test_routed_experts_leave_dense_models_and_attention_alone(tmp_path):
    dense = _config(tmp_path, moe=False)
    moe = _config(tmp_path, moe=True)
    assert (derive_peft_lora_specs(dense, None, rank=2, targets="all-linear", routed_experts=True)
            == derive_peft_lora_specs(dense, None, rank=2, targets="all-linear"))
    attention = derive_peft_lora_specs(moe, None, rank=2, targets="attention", routed_experts=True)
    assert attention == derive_peft_lora_specs(moe, None, rank=2, targets="attention")
    assert not lora_layout.has_expert_specs(attention)


def test_a_first_export_recipe_has_no_offline_layout():
    profile = SimpleNamespace(recipe="kimi_k31", layout_source="first-export")
    with pytest.raises(lora_layout.LoraLayoutError, match="learned from the first export"):
        lora_layout.offline_canonical_specs(profile, rank=8, targets="attention", model_path="unused")


def test_generic_expert_lora_model_limits():
    grouped = SimpleNamespace(moe_grouped_gemm=True)
    check_generic_expert_lora(grouped, expert_parallel=2)
    check_generic_expert_lora(SimpleNamespace(moe_grouped_gemm=False), expert_parallel=1)
    with pytest.raises(ValueError, match="grouped-GEMM experts"):
        check_generic_expert_lora(SimpleNamespace(moe_grouped_gemm=False), expert_parallel=2)
    with pytest.raises(ValueError, match="no fp8/fp4"):
        check_generic_expert_lora(SimpleNamespace(moe_grouped_gemm=True, fp8="e4m3"), expert_parallel=2)


def _family(tmp_path, name: str) -> str:
    from transformers import DeepseekV3Config, Glm4MoeConfig

    base = dict(vocab_size=128, hidden_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=3,
                intermediate_size=96, moe_intermediate_size=32, n_routed_experts=4, num_experts_per_tok=2,
                n_shared_experts=1, first_k_dense_replace=1, n_group=1, topk_group=1)
    config = {
        "deepseek_v3": lambda: DeepseekV3Config(**base, q_lora_rank=32, kv_lora_rank=16, qk_rope_head_dim=8,
                                                qk_nope_head_dim=8, v_head_dim=16,
                                                architectures=["DeepseekV3ForCausalLM"]),
        "glm4_moe": lambda: Glm4MoeConfig(**base, head_dim=16, architectures=["Glm4MoeForCausalLM"]),
    }[name]()
    path = tmp_path / name
    config.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("family", ["deepseek_v3", "glm4_moe"])
def test_dense_first_layer_and_shared_experts_keep_their_adapters(tmp_path, family):
    """On these layouts PEFT's Transformers-5 target conversion wraps the fused
    experts as one stacked parameter and drops the dense and shared MLPs; the
    offline layout keeps all three, one adapter per routed expert."""

    specs = lora_layout.offline_canonical_specs(None, rank=4, targets="all-linear",
                                                model_path=_family(tmp_path, family))
    names = {s.name: s.shape for s in specs}
    prefix = "base_model.model.model.layers"

    assert names[f"{prefix}.0.mlp.gate_proj.lora_B.weight"] == (96, 4)  # dense first layer
    assert names[f"{prefix}.2.mlp.shared_experts.down_proj.lora_A.weight"] == (4, 32)
    assert names[f"{prefix}.1.mlp.experts.3.up_proj.lora_B.weight"] == (32, 4)
    assert not [n for n in names if ".experts.base_layer." in n or ".experts.lora_" in n or ".layers.0.mlp.experts" in n]
    assert sum(".experts." in n and "shared" not in n for n in names) == 2 * 4 * 6


def test_offline_linear_pairs_equal_peft_where_peft_keeps_the_targets(tmp_path):
    """The pairs computed for nn.Linear modules are PEFT's (Qwen3-MoE: PEFT's
    target conversion leaves an explicit attention list alone)."""

    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(_config(tmp_path, moe=True))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    modules = lora_layout.all_linear_modules(model)
    ours = {s.name: s.shape for s in lora_layout.linear_lora_specs(model, modules, rank=4)}
    with init_empty_weights():
        peft = get_peft_model(AutoModelForCausalLM.from_config(config),
                              LoraConfig(r=4, lora_alpha=4, target_modules=modules, task_type="CAUSAL_LM"))
    theirs = {name: tuple(t.shape) for name, t in get_peft_model_state_dict(peft).items()}
    assert ours == theirs
