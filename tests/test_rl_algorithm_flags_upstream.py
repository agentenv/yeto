"""Upstream Miles parser checks (rl-algorithm-capabilities 2.2 / 2.6).

Skips unless upstream Miles imports. Run in the pinned upstream venv:

    PYTHONPATH=/home/michael/work/miles-next:$PWD \\
      /home/michael/work/miles-next-venv/bin/python -m pytest -q \\
      tests/test_rl_algorithm_flags_upstream.py

2.6 runs upstream ``miles.utils.arguments.parse_args`` (which also runs
``miles_validate_args`` and ``sglang_validate_args``) on the algorithm argv.
That venv has no Megatron-LM, so the run uses ``--train-backend fsdp``; every
algorithm flag is a Miles (backend-independent) argument, and the Megatron
argv half of the ports translation is covered by
``test_rl_miles_adapter_config.py::test_upstream_parse_args_accepts_translation``
where Megatron is installed.
"""

from __future__ import annotations

import argparse
import json
import sys

import pytest

arguments = pytest.importorskip("miles.utils.arguments")

from yeto.rl.engine.algorithm import (  # noqa: E402
    STOCK_NONZERO_STD_FILTER,
    AdvantageSpec,
    AlgorithmSpec,
    CorrectionSpec,
    KlSpec,
    LossSpec,
    PluginRef,
    SamplingSpec,
)
from yeto.rl.engine.miles_adapter import algorithm_flags as af  # noqa: E402

ICEPOP = "miles.backends.training_utils.loss_hub.corrections.icepop_function"
REF = PluginRef.from_path("yeto.rl.engine.algorithm.plugin_source_sha256")
TINY_QWEN3 = {
    "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 64,
    "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 4096,
    "vocab_size": 1000, "rms_norm_eps": 1e-6, "rope_theta": 10000,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "hidden_act": "silu",
    "attention_bias": False,
}


def _upstream_option_strings() -> set[str]:
    parser = argparse.ArgumentParser()
    arguments.get_miles_extra_args_provider()(parser)
    return {option for action in parser._actions for option in action.option_strings}


def test_objective_and_mapped_flags_exist_upstream():
    options = _upstream_option_strings()
    # rl-algo-loss-variants: the fork flags exist only once the pin moves to
    # a fork commit that implements them (loss_variants.FORK_COMMITS).
    from yeto.rl.algos import loss_variants

    pending = frozenset() if loss_variants.fork_supports_variants() else loss_variants.FORK_FLAGS
    from yeto.rl.algos import critic  # rl-algo-critic-family 7.2: fork-only GAE / VAPO flags

    pending |= critic.FORK_FLAGS
    missing = sorted(af.objective_flags() - options - pending)
    assert not missing, f"not in the upstream Miles parser: {missing}"


def _r0_positioned(spec: AlgorithmSpec) -> list[str]:
    argv = ["--advantage-estimator", spec.advantage_estimator]
    if spec.kl_coef is not None:
        argv += ["--kl-coef", str(spec.kl_coef)]
    if spec.dynamic_sampling_filter is not None:
        argv += ["--dynamic-sampling-filter-path", spec.dynamic_sampling_filter]
    if spec.sampling.over_sampling_batch_size is not None:
        argv += ["--over-sampling-batch-size", str(spec.sampling.over_sampling_batch_size)]
    return argv


CASES = [
    (dict(loss=LossSpec(eps_clip=0.25)), dict(eps_clip=0.25)),
    (dict(loss=LossSpec(eps_clip_high=0.28)), dict(eps_clip_high=0.28)),
    (dict(loss=LossSpec(eps_clip_c=3)), dict(eps_clip_c=3.0)),
    (dict(loss=LossSpec(aggregation="token")), dict(calculate_per_token_loss=True)),
    (dict(loss=LossSpec(reducer=REF)), dict(custom_pg_loss_reducer_function_path=REF.path)),
    (dict(loss=LossSpec(variant="custom_loss", custom_loss=REF)),
     dict(loss_type="custom_loss", custom_loss_function_path=REF.path)),
    (dict(advantage=AdvantageSpec(std_normalization=False)),
     dict(grpo_std_normalization=False)),
    (dict(advantage=AdvantageSpec(rewards_normalization=False)),
     dict(rewards_normalization=False)),
    (dict(advantage=AdvantageSpec(whiten=True)), dict(normalize_advantages=True)),
    (dict(advantage=AdvantageSpec(reward_postprocess=REF)),
     dict(custom_reward_post_process_path=REF.path)),
    (dict(advantage=AdvantageSpec(estimator="gspo"),
          loss=LossSpec(eps_clip=3e-4, eps_clip_high=4e-4)),
     dict(advantage_estimator="gspo", eps_clip=3e-4, eps_clip_high=4e-4)),
    (dict(advantage=AdvantageSpec(estimator="reinforce_plus_plus", whiten=True),
          kl=KlSpec(placement="reward", coef=0.05)),
     dict(advantage_estimator="reinforce_plus_plus", kl_coef=0.05, normalize_advantages=True)),
    (dict(kl=KlSpec(placement="loss", coef=0.01, estimator="k3", unbiased=True)),
     dict(use_kl_loss=True, kl_loss_coef=0.01, kl_loss_type="k3", use_unbiased_kl=True,
          kl_coef=0.0)),
    (dict(entropy_coef=0.001), dict(entropy_coef=0.001)),
    (dict(correction=CorrectionSpec(method="tis", tis_clip=2, tis_clip_low=0.5)),
     dict(use_tis=True, tis_clip=2.0, tis_clip_low=0.5)),
    (dict(correction=CorrectionSpec(method="custom", function=PluginRef(ICEPOP, "0" * 64),
                                    tis_clip=5, tis_clip_low=0.5, mismatch_metrics=True)),
     dict(use_tis=True, custom_tis_function_path=ICEPOP, get_mismatch_metrics=True,
          tis_clip=5.0, tis_clip_low=0.5)),
    (dict(correction=CorrectionSpec(use_rollout_logprobs=True)), dict(use_rollout_logprobs=True)),
    (dict(correction=CorrectionSpec(method="opsm", opsm_delta=1e-4)),
     dict(use_opsm=True, opsm_delta=1e-4)),
    (dict(sampling=SamplingSpec(filter=STOCK_NONZERO_STD_FILTER, over_sampling_batch_size=4)),
     dict(dynamic_sampling_filter_path=STOCK_NONZERO_STD_FILTER, over_sampling_batch_size=4)),
]


@pytest.mark.parametrize("change, expected", CASES)
def test_upstream_parse_args_accepts_non_default_mapping(tmp_path, monkeypatch, change, expected):
    (tmp_path / "config.json").write_text(json.dumps(TINY_QWEN3))
    spec = AlgorithmSpec(**change)
    argv = [
        "train.py", "--train-backend", "fsdp",
        "--hf-checkpoint", str(tmp_path), "--ref-load", str(tmp_path),
        "--rollout-batch-size", "2", "--n-samples-per-prompt", "2",
        "--global-batch-size", "4", "--num-rollout", "1",
        "--colocate", "--actor-num-gpus-per-node", "1",
        *_r0_positioned(spec), *af.algorithm_argv(spec),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = arguments.parse_args()
    for name, value in expected.items():
        assert getattr(args, name) == pytest.approx(value) if isinstance(value, float) \
            else getattr(args, name) == value, name
