
import pytest

from yeto.rl.engine.algorithm import BOUNDED_NONZERO_STD_FILTER, AlgorithmSpec, AlgorithmSpecError

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
        '"nonzero_reward_std_bounded",'  # neutral name since decoupling 4.4 (hash-migration.md)
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


