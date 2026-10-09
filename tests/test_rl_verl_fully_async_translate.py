"""agentic-rollout-utilization 6.4a: verl fully_async <-> yeto translation (pure, CPU)."""

from __future__ import annotations

import math

import pytest

from yeto.rl.adapters.verl import fully_async_translate as t
from yeto.rl.engine.policy_age import PolicyAgeError


def _vmap(pairs=((0, 10), (1, 11), (2, 12), (3, 13))):
    vm = t.VersionMap()
    for p, o in pairs:
        vm.record(p, o)
    return vm


def test_limit_to_staleness_threshold_and_queue_lag():
    """(c) staleness only throttles by samples: s = N - 1, queued lag floor(s) + 1."""
    assert t.staleness_threshold_for(1) == 0.0 and t.staleness_threshold_for(2) == 1.0
    assert t.queue_version_lag(t.staleness_threshold_for(1)) == 1
    assert t.queue_version_lag(0.1) == 1 and t.queue_version_lag(1.0) == 2
    with pytest.raises(PolicyAgeError):
        t.staleness_threshold_for(0)
    with pytest.raises(t.VerlTranslationError, match="trigger_parameter_sync_step"):
        t.queue_version_lag(0.0, 4)
    assert t.fully_async_overrides(1, samples_per_round=128, ppo_mini_batch_size=32) == {
        "async_training.staleness_threshold": 0.0, "async_training.partial_rollout": True,
        "async_training.trigger_parameter_sync_step": 1, "async_training.require_batches": 4}
    with pytest.raises(t.VerlTranslationError, match="multiple"):
        t.fully_async_overrides(1, samples_per_round=100, ppo_mini_batch_size=32)


def test_version_map_translates_and_refuses_contradictions():
    """(d) trainer-local current_param_version <-> outer version; checkpoints keyed by outer."""
    vm = _vmap()
    assert vm.outer(2) == 12 and vm.local(13) == 3 and vm.checkpoint_step(1) == 11
    vm.record(3, 13)  # the same pairing again (restart): accepted
    with pytest.raises(t.VerlTranslationError, match="contradicts"):
        vm.record(3, 14)
    with pytest.raises(t.VerlTranslationError, match="increase"):
        vm.record(4, 9)  # outer goes backwards
    with pytest.raises(t.VerlTranslationError, match="never published"):
        vm.outer(9)
    assert t.VersionMap.from_dict(vm.to_dict()).to_dict() == vm.to_dict()


def test_min_max_global_steps_become_version_segments_judged_by_the_oldest():
    """(b) per-trajectory min/max -> every outer version in between; (c) yeto discards."""
    vm = _vmap()
    assert t.versions_from_global_steps(1, 2, vm) == (11, 12)
    assert t.versions_from_global_steps(1, 3, vm) == (11, 12, 13)
    assert t.judge_trajectory((12, 13), 13, 1) is None
    assert "exceeds max_policy_age=1" in t.judge_trajectory((11, 12, 13), 13, 1)
    assert t.judge_trajectory((11, 12, 13), 13, 2) is None
    assert t.judge_trajectory((), 13, 1) == "unknown version"
    assert "future" in t.judge_trajectory((14,), 13, 1)
    with pytest.raises(t.VerlTranslationError, match="unknown"):
        t.versions_from_global_steps(None, 2, vm)
    with pytest.raises(t.VerlTranslationError, match="unpublished"):
        t.versions_from_global_steps(2, 5, vm)


def test_resumed_calls_give_per_token_versions_and_logprobs():
    """(a) the prefix continuation is invisible to the agent loop: the adapter's
    per-call records rebuild the token-level segments."""
    vm = _vmap()
    prov = t.provenance_from_calls(
        [t.ResumedCall(1, (-0.1, -0.2)), t.ResumedCall(2, (-0.3,)), t.ResumedCall(2, (1e-9,))], vm)
    assert prov.versions == (11, 11, 12, 12) and prov.segments() == [(11, 0, 2), (12, 2, 4)]
    assert prov.logprobs[3] == 0.0 and math.isclose(prov.logprobs[0], -0.1)


def test_verl_still_declares_stage_one():
    from yeto.rl.adapters.verl.policy_age import SUPPORT

    assert (SUPPORT.stage, SUPPORT.max_policy_age) == (1, 0)
