"""Shared learner fixture for the ports run-config tests (rl-engine-ports task 2.4).

The legacy argv snapshot guard that used to live here went with the legacy
Miles translation (rl-code-quality-cleanup 1.10); ``_captured_args`` stays as
the fixture other tests import.
"""

from __future__ import annotations

import argparse

import pytest

from yeto.rl.engine import run_config as rc


def _captured_args():
    """A minimal learner namespace and the resolve_rl_run_config keywords of an
    mrope provider (the launcher tests' former mrope fixture)."""

    args = argparse.Namespace(
        seq_len=128, groups_per_round=4, samples_per_group=2, optimizer_steps=1,
        lora_targets="attention", lora_r=4, seed=1, learner_id=0, global_rounds=1,
        inner_lr=1e-5, reward_function="pkg.reward:score", over_sampling_batch_size=4,
        rollout_max_response_len=64, custom_generate_function_path=None,
        use_session_server=False, actor_num_nodes=1, actor_num_gpus_per_node=2,
        expert_parallel=None,
    )
    provider = argparse.Namespace(
        hidden_size=16, num_attention_heads=4, num_layers=2, ffn_hidden_size=32,
        num_query_groups=2, kv_channels=4, seq_length=256, vocab_size=64,
        layernorm_epsilon=1e-6, position_embedding_type="mrope", rotary_base=1000000,
        rotary_percent=0.25, mrope_section=[11, 11, 10],
    )
    return (args,), {"model_path": "/model", "prompt_path": "/prompts.jsonl",
                     "provider": provider, "target_modules": ["q_proj"]}


def test_standby_gpus_need_a_lora_fixed_partition():
    """rl-infra-spec 2.1 / review F7: standby is never silently dropped."""

    import argparse

    (args,), kwargs = _captured_args()
    base = vars(args)
    part = argparse.Namespace(**{**base, "rl_placement": "fixed-partition",
                                 "rollout_num_gpus": 2, "rl_standby_gpus": 2})
    config = rc.resolve_rl_run_config(part, **kwargs)
    assert config.parallel.standby_gpus == 2
    assert config.parallel.dedicated_rollout_gpus == 2
    colo = argparse.Namespace(**{**base, "rl_standby_gpus": 1})
    with pytest.raises(ValueError, match="rl-placement fixed-partition"):
        rc.resolve_rl_run_config(colo, **kwargs)
    full = argparse.Namespace(**{**base, "parameter_mode": "full", "rl_standby_gpus": 1})
    with pytest.raises(ValueError, match="not supported for full-parameter"):
        rc.resolve_rl_run_config(full, **kwargs)
