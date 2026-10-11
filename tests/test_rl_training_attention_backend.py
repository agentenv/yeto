"""--rl-training-attention-backend override (rlf-codex-rl 5.8).

The recipe picks the Megatron --attention-backend (qwen3_5 -> flash,
generic -> unfused). The 5.8 gate run hit a flash-attn cute JIT compile
failure on L40S, so the CLI gained an explicit override; flash-only
recipes refuse a non-flash override.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from yeto.rl.adapters.miles.island_entry import build_miles_argv


def _args(**extra):
    base = dict(
        seq_len=128,
        groups_per_round=4,
        samples_per_group=2,
        optimizer_steps=1,
        lora_targets="auto",
        lora_r=4,
        seed=1,
        learner_id=0,
        global_rounds=3,
        inner_lr=1e-4,
        reward_function="pkg.reward:score",
        over_sampling_batch_size=6,
        rollout_max_response_len=64,
        custom_generate_function_path="pkg.agent.generate",
        custom_agent_function_path="pkg.agent.run",
        agent_max_seq_len=96,
        use_session_server=True,
        session_server_ip="127.0.0.1",
        session_server_port=[31000, 31002],
        tito_model="qwen3",
        tito_allowed_append_roles=["tool", "user"],
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        expert_parallel=None,
        apply_chat_template_kwargs={"enable_thinking": False},
    )
    base.update(extra)
    return argparse.Namespace(**base)


def _provider(**extra):
    base = dict(
        hidden_size=16,
        num_attention_heads=4,
        num_layers=2,
        ffn_hidden_size=32,
        num_query_groups=2,
        kv_channels=4,
        seq_length=256,
        vocab_size=64,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        position_embedding_type="yarn",
        rotary_base=10000,
        gated_linear_unit=True,
        share_embeddings_and_output_weights=False,
        add_bias_linear=False,
        add_qkv_bias=True,
        qk_layernorm=True,
        num_moe_experts=8,
        moe_ffn_hidden_size=24,
        moe_router_topk=2,
        moe_layer_freq=[0, 1],
        moe_shared_expert_intermediate_size=12,
        multi_latent_attention=True,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_head_dim=4,
        qk_pos_emb_head_dim=2,
        v_head_dim=4,
    )
    base.update(extra)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _chat_template_stub(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "miles.utils.chat_template_utils",
        types.SimpleNamespace(
            resolve_reasoning_and_tool_call_parser=lambda model: ("qwen3", "qwen25")
        ),
    )


def _backend_of(argv):
    return argv[argv.index("--attention-backend") + 1]


def test_gdn_recipe_default_stays_flash():
    argv = build_miles_argv(
        _args(),
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=_provider(experimental_attention_variant="gated_delta_net"),
        target_modules=["qkv_proj", "out_proj"],
    )
    assert _backend_of(argv) == "flash"


def test_override_unfused_on_gdn_recipe():
    argv = build_miles_argv(
        _args(rl_training_attention_backend="unfused"),
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=_provider(experimental_attention_variant="gated_delta_net"),
        target_modules=["qkv_proj", "out_proj"],
    )
    assert _backend_of(argv) == "unfused"


def test_override_fused_on_generic_recipe():
    argv = build_miles_argv(
        _args(rl_training_attention_backend="fused"),
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=_provider(),
        target_modules=["qkv_proj", "out_proj"],
    )
    assert _backend_of(argv) == "fused"


def test_auto_default_when_attr_missing():
    # Old launch-side namespaces (and most tests) never set the attribute.
    argv = build_miles_argv(
        _args(),
        model_path="/model",
        prompt_path="/prompts.jsonl",
        provider=_provider(),
        target_modules=["qkv_proj", "out_proj"],
    )
    assert _backend_of(argv) == "unfused"  # generic recipe default
