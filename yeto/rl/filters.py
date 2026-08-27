"""Bounded rollout filters for difficult external-reward environments."""

from __future__ import annotations

import math
import statistics
from types import SimpleNamespace
from typing import Any


def _group_key(samples: list[Any]) -> tuple[Any, ...]:
    values = []
    for sample in samples:
        index = getattr(sample, "index", None)
        values.append(index if index is not None else id(sample))
    return tuple(values)


def _reward(args, sample: Any) -> float:
    getter = getattr(sample, "get_reward_value", None)
    if callable(getter):
        return float(getter(args))
    return float(getattr(sample, "reward"))


def _output(keep: bool, reason: str | None):
    # Keep the module importable in the controller/test environment; the Miles
    # image supplies the native result type at runtime.
    try:
        from miles.rollout.filter_hub.base_types import DynamicFilterOutput
    except ImportError:
        return SimpleNamespace(keep=keep, reason=reason)
    return DynamicFilterOutput(keep=keep, reason=reason)


def bounded_nonzero_reward_std(args, samples: list[Any], **kwargs):
    """Prefer non-zero reward variance, then accept a group after a bound.

    ``--dynamic-sampling-max-replacements N`` rejects the first ``N``
    zero-variance groups in a rollout and accepts the next one.  Decisions are
    memoized because Yeto's all-samples callback sees each group a second time.
    """

    rollout_id = getattr(args, "yeto_rl_policy_version", None)
    state = getattr(args, "_yeto_bounded_filter_state", None)
    if state is None or state.get("rollout_id") != rollout_id:
        state = {"rollout_id": rollout_id, "rejections": 0, "forced": 0, "decisions": {}}
        args._yeto_bounded_filter_state = state

    key = _group_key(samples)
    previous = state["decisions"].get(key)
    if previous is not None:
        return _output(*previous)

    rewards = [_reward(args, sample) for sample in samples]
    std = statistics.pstdev(rewards) if rewards else 0.0
    if not math.isfinite(std):
        std = 0.0
    if std > 1e-8:
        decision = (True, None)
    else:
        limit = getattr(args, "yeto_rl_dynamic_sampling_max_replacements", None)
        if limit is None or state["rejections"] < int(limit):
            state["rejections"] += 1
            value = round(rewards[0], 1) if rewards else 0.0
            decision = (False, f"zero_std_{value}")
        else:
            state["forced"] += 1
            decision = (True, f"bounded_fallback_after_{int(limit)}_replacements")
    state["decisions"][key] = decision
    return _output(*decision)
