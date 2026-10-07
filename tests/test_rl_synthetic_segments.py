"""Task 6.4 G1 helper: synthetic two-segment reward + its launcher gate."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from yeto import launcher as L
from yeto.rl import synthetic_segments as ss


def _args(variant, reward=ss.SYNTHETIC_SEGMENTS_REWARD):
    spec = {"advantage": {"gae_variant": variant}} if variant else {}
    return SimpleNamespace(training_mode="rl", rl_algorithm_spec_json=json.dumps(spec),
                           reward_function=reward, custom_agent_function_path=None)


def test_labels_alternate_and_keep_gsm8k_reward():
    s0 = SimpleNamespace(response="so \\boxed{4}", label="#### 4", metadata=None, index=6)
    s1 = SimpleNamespace(response="5", label="#### 4", metadata={}, index=7)
    assert asyncio.run(ss.score(None, s0)) == 1.0 and asyncio.run(ss.score(None, s1)) == 0.0
    assert s0.metadata["tokens_after"] == ss.SYNTHETIC_TOKENS_AFTER and s0.metadata["success"] is True
    assert s1.metadata["tokens_after"] == 0 and s1.metadata["success"] is False


def test_launcher_accepts_synthetic_only_with_per_sample_variant():
    assert L.codex_harness_launch(_args("cross_segment_per_sample"), environ={}) is None
    with pytest.raises(ValueError, match="only valid"):
        L.codex_harness_launch(_args("length_adaptive"), environ={})
    with pytest.raises(ValueError, match="needs compacted rollouts"):
        L.codex_harness_launch(_args("cross_segment_per_sample", "yeto.rl.gsm8k_reward:score"), environ={})
