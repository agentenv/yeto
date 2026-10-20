"""Scope rule of quantized rollout (S26 E6): which linears may be quantized, and the
tier 1 engine ignore lists that follow from it."""

from __future__ import annotations

import pytest

from yeto.rl.quant import scope
from yeto.rl.quant.scope import DENSE, MOE, QuantScopeError

H, I = 8, 16


def dense_shapes(layers: int = 2) -> dict[str, list[int]]:
    """Qwen3 dense names (Qwen3-4B layout, tied lm_head omitted like the HF checkpoint)."""
    out = {"model.embed_tokens.weight": [32, H], "model.norm.weight": [H]}
    for i in range(layers):
        p = f"model.layers.{i}"
        out.update({f"{p}.input_layernorm.weight": [H], f"{p}.post_attention_layernorm.weight": [H],
                    f"{p}.self_attn.q_proj.weight": [H, H], f"{p}.self_attn.k_proj.weight": [H, H],
                    f"{p}.self_attn.v_proj.weight": [H, H], f"{p}.self_attn.o_proj.weight": [H, H],
                    f"{p}.self_attn.q_norm.weight": [H], f"{p}.self_attn.k_norm.weight": [H],
                    f"{p}.mlp.gate_proj.weight": [I, H], f"{p}.mlp.up_proj.weight": [I, H],
                    f"{p}.mlp.down_proj.weight": [H, I]})
    return out


def moe_shapes(experts: int = 2) -> dict[str, list[int]]:
    """Qwen3-MoE / Qwen2-MoE names: router, routed experts, shared expert + its gate,
    one dense FFN layer (layer 0, as DeepSeek's first dense layers) and lm_head."""
    out = {"model.embed_tokens.weight": [32, H], "model.norm.weight": [H], "lm_head.weight": [32, H]}
    out.update({"model.layers.0.self_attn.q_proj.weight": [H, H], "model.layers.0.self_attn.o_proj.weight": [H, H],
                "model.layers.0.mlp.gate_proj.weight": [I, H], "model.layers.0.mlp.up_proj.weight": [I, H],
                "model.layers.0.mlp.down_proj.weight": [H, I]})
    p = "model.layers.1"
    out.update({f"{p}.self_attn.q_proj.weight": [H, H], f"{p}.self_attn.o_proj.weight": [H, H],
                f"{p}.mlp.gate.weight": [experts, H], f"{p}.mlp.shared_expert_gate.weight": [1, H],
                f"{p}.mlp.shared_expert.gate_proj.weight": [I, H], f"{p}.mlp.shared_expert.up_proj.weight": [I, H],
                f"{p}.mlp.shared_expert.down_proj.weight": [H, I]})
    for e in range(experts):
        out.update({f"{p}.mlp.experts.{e}.gate_proj.weight": [I, H], f"{p}.mlp.experts.{e}.up_proj.weight": [I, H],
                    f"{p}.mlp.experts.{e}.down_proj.weight": [H, I]})
    return out


def test_model_kind_reads_the_expert_count():
    assert scope.model_kind({"num_hidden_layers": 36}) == DENSE
    assert scope.model_kind({"num_experts": 128}) == MOE
    assert scope.model_kind({"n_routed_experts": 256}) == MOE
    assert scope.model_kind({"text_config": {"num_local_experts": 8}}) == MOE
    assert scope.model_kind({"num_experts": 0}) == DENSE


def test_dense_scope_is_the_ffn_only():
    q = [m for m in scope.linear_modules(dense_shapes()) if scope.in_scope(m, DENSE)]
    assert q == sorted(f"model.layers.{i}.mlp.{p}" for i in range(2) for p in ("down_proj", "gate_proj", "up_proj"))
    assert scope.in_scope("model.layers.0.mlp.gate_up_proj", DENSE)  # fused engine name
    for name in ("model.layers.0.self_attn.q_proj", "lm_head", "model.embed_tokens", "model.norm",
                 "model.layers.0.mlp.shared_expert.up_proj", "visual.blocks.0.mlp.fc1"):
        assert not scope.in_scope(name, DENSE), name


def test_moe_scope_is_the_routed_experts_only():
    q = [m for m in scope.linear_modules(moe_shapes()) if scope.in_scope(m, MOE)]
    assert q == sorted(f"model.layers.1.mlp.experts.{e}.{p}" for e in range(2)
                       for p in ("down_proj", "gate_proj", "up_proj"))
    assert scope.in_scope("model.layers.3.mlp.experts.gate_up_proj", MOE)  # stacked trainer layout
    assert scope.in_scope("model.language_model.layers.3.mlp.experts.7.w2", MOE)
    reasons = scope.violations(scope.linear_modules(moe_shapes()), MOE)
    assert reasons["model.layers.1.mlp.gate"] == "router"
    assert reasons["model.layers.1.mlp.shared_expert_gate"] == "shared_expert"
    assert reasons["model.layers.1.mlp.shared_expert.up_proj"] == "shared_expert"
    assert reasons["model.layers.0.mlp.up_proj"] == "not_routed_expert"  # dense layer of a MoE model
    assert reasons["model.layers.0.self_attn.q_proj"] == "attention"
    assert reasons["lm_head"] == "lm_head"


def test_check_quantized_refuses_out_of_scope_modules():
    scope.check_quantized(["model.layers.0.mlp.up_proj"], DENSE, what="t")
    with pytest.raises(QuantScopeError, match=r"1 module\(s\) outside the dense scope.*self_attn.o_proj \(attention\)"):
        scope.check_quantized(["model.layers.0.mlp.up_proj", "model.layers.0.self_attn.o_proj"], DENSE, what="t")
    with pytest.raises(QuantScopeError, match="router"):
        scope.check_quantized(["model.layers.1.mlp.gate"], MOE, what="t")
    with pytest.raises(ValueError, match="unknown model kind"):
        scope.in_scope("x", "hybrid")


@pytest.mark.parametrize("kind,shapes", [(DENSE, dense_shapes()), (MOE, moe_shapes())])
def test_engine_ignore_list_quantizes_exactly_the_scope(kind, shapes):
    ignore = scope.engine_ignore_list(shapes, kind)
    quantized = scope.check_online_plan(shapes, kind, ignore)
    assert quantized == [m for m in scope.linear_modules(shapes) if scope.in_scope(m, kind)]
    # the path segments alone must never cover an in-scope module (no overmatch)
    segments = sorted(set(ignore) - set(scope.ignored_modules(shapes, kind)))
    assert all(not scope.path_match(s, m) for s in segments for m in quantized)


def test_segments_also_cover_renamed_prefixes():
    """A module whose engine prefix differs from the checkpoint name (composite models)
    is still kept BF16 by the segment entries."""
    ignore = scope.engine_ignore_list(dense_shapes(), DENSE)
    renamed = ["language_model.model.layers.0.self_attn.qkv_proj", "language_model.model.layers.0.mlp.gate_up_proj"]
    assert scope.online_quantized(renamed, ignore) == ["language_model.model.layers.0.mlp.gate_up_proj"]


def test_online_plan_refuses_an_ignore_list_that_misses_a_module():
    shapes = dense_shapes()
    with pytest.raises(QuantScopeError, match="attention"):
        scope.check_online_plan(shapes, DENSE, ["lm_head"])
    with pytest.raises(QuantScopeError, match="quantizes nothing"):
        scope.check_online_plan(shapes, DENSE, scope.linear_modules(shapes))


def test_path_match_follows_sglang_segments():
    assert scope.path_match("self_attn", "model.layers.0.self_attn.qkv_proj")
    assert scope.path_match("mlp.gate", "model.layers.3.mlp.gate")
    assert not scope.path_match("mlp.gate", "model.layers.3.mlp.gate_proj")
    assert not scope.path_match("mlp.gate", "model.layers.3.mlp.gate_up_proj")
    assert scope.path_match("model.layers.0", "model.layers.0.mlp.up_proj")
    assert not scope.path_match("model.layers.1", "model.layers.10.mlp.up_proj")


def test_path_match_equals_sglang_where_installed():
    utils = pytest.importorskip("sglang.srt.layers.quantization.utils")
    cases = [("self_attn", "model.layers.0.self_attn.qkv_proj"), ("mlp.gate", "model.layers.3.mlp.gate_proj"),
             ("mlp.gate", "model.layers.3.mlp.gate"), ("model.layers.1", "model.layers.10.mlp.up_proj"),
             ("mlp.shared_expert", "model.layers.1.mlp.shared_expert_gate")]
    for ignored, prefix in cases:
        assert scope.path_match(ignored, prefix) == utils._module_path_match(ignored, prefix), (ignored, prefix)


@pytest.mark.parametrize("kind,shapes,engine_modules", [
    (DENSE, dense_shapes(), {"model.layers.0.self_attn.qkv_proj": True, "model.layers.0.self_attn.o_proj": True,
                             "model.layers.0.mlp.gate_up_proj": False, "model.layers.0.mlp.down_proj": False}),
    (MOE, moe_shapes(), {"model.layers.1.self_attn.qkv_proj": True, "model.layers.1.mlp.gate": True,
                         "model.layers.1.mlp.shared_expert.gate_up_proj": True,
                         "model.layers.1.mlp.shared_expert_gate": True, "model.layers.0.mlp.gate_up_proj": True,
                         "model.layers.0.mlp.down_proj": True, "model.layers.1.mlp.experts": False}),
])
def test_sglang_is_layer_skipped_keeps_out_of_scope_engine_modules(kind, shapes, engine_modules):
    """SGLang's own skip rule (fused qkv_proj / gate_up_proj shards, FusedMoE prefix) on the
    tier 1 ignore list: True = stays BF16."""
    utils = pytest.importorskip("sglang.srt.layers.quantization.utils")
    ignore = scope.engine_ignore_list(shapes, kind)
    got = {prefix: bool(utils.is_layer_skipped(prefix, ignore)) for prefix in engine_modules}
    assert got == engine_modules
