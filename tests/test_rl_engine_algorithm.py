import inspect
from types import SimpleNamespace

import pytest

from yeto.rl import learner
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    STOCK_NONZERO_STD_FILTER,
    AlgorithmSpec,
    AlgorithmSpecError,
)

BOUNDED = AlgorithmSpec(
    dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
    dynamic_sampling_max_replacements=4,
)


def test_hash_stable_and_canonical():
    assert BOUNDED.sha256() == AlgorithmSpec.from_dict(BOUNDED.to_dict()).sha256()
    # Key order in the input does not matter.
    reordered = dict(reversed(list(BOUNDED.to_dict().items())))
    assert AlgorithmSpec.from_dict(reordered).sha256() == BOUNDED.sha256()
    assert AlgorithmSpec().sha256() != BOUNDED.sha256()
    assert AlgorithmSpec(kl_coef=0).sha256() == AlgorithmSpec(kl_coef=0.0).sha256()
    # Pinned: changing canonicalization is an identity break and must be deliberate.
    assert BOUNDED.canonical_json() == (
        '{"advantage_estimator":"grpo","dynamic_sampling_filter":'
        '"yeto.rl.filters.bounded_nonzero_reward_std",'
        '"dynamic_sampling_max_replacements":4,"kl_coef":null,'
        '"loss":"policy_loss","schema":"yeto-rl-algorithm-spec-v1"}'
    )


def test_unknown_and_invalid_rejected():
    with pytest.raises(AlgorithmSpecError, match="unknown algorithm spec fields"):
        AlgorithmSpec.from_dict({"advantage_estimator": "grpo", "eps_clip": 0.2})
    with pytest.raises(AlgorithmSpecError, match="schema"):
        AlgorithmSpec.from_dict({"schema": "v0"})
    with pytest.raises(AlgorithmSpecError, match="grpo"):
        AlgorithmSpec(advantage_estimator="gspo")  # ppo: v1 since rl-algo-critic-family
    with pytest.raises(AlgorithmSpecError):
        AlgorithmSpec(dynamic_sampling_filter="x.y")
    with pytest.raises(AlgorithmSpecError, match="requires"):
        AlgorithmSpec(dynamic_sampling_max_replacements=2)


def test_legacy_mapping():
    args = SimpleNamespace(
        dynamic_sampling_filter_path=STOCK_NONZERO_STD_FILTER,
        dynamic_sampling_max_replacements=4,
    )
    spec = AlgorithmSpec.from_legacy_args(args)
    assert spec == BOUNDED
    assert spec.to_legacy_argv() == [
        "--advantage-estimator", "grpo",
        "--dynamic-sampling-filter-path", BOUNDED_NONZERO_STD_FILTER,
    ]
    assert spec.to_legacy_runtime_attrs() == {
        "yeto_rl_dynamic_sampling_max_replacements": 4
    }
    assert AlgorithmSpec.from_legacy_args(SimpleNamespace()).to_legacy_argv() == [
        "--advantage-estimator", "grpo",
    ]
    # Flags must match what legacy build_miles_argv actually emits.
    source = inspect.getsource(learner)
    for token in ('"--advantage-estimator", "grpo"', '"--dynamic-sampling-filter-path"',
                  '"--kl-coef"', "yeto_rl_dynamic_sampling_max_replacements"):
        assert token in source
