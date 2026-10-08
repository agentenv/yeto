"""Bounded rollout filters for difficult external-reward environments."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from yeto.rl.rewards.builtin import BoundedNonzeroStdFilter
from yeto.rl.rewards.types import Trajectory


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

    Miles entry point; the decision and its state live in the neutral
    :class:`yeto.rl.rewards.builtin.BoundedNonzeroStdFilter` (decoupling 3.2),
    kept on ``args`` with its state dict exposed as before.
    """

    flt = getattr(args, "_yeto_bounded_filter", None)
    if flt is None:
        flt = BoundedNonzeroStdFilter()
        args._yeto_bounded_filter = flt
    # The Miles round hook resets ``args._yeto_bounded_filter_state`` to None
    # per rollout; the attribute stays the source of truth for the state.
    shared = getattr(args, "_yeto_bounded_filter_state", None)
    flt.state = shared if isinstance(shared, dict) else {}
    round_id = getattr(args, "yeto_rl_policy_version", None)
    key = _group_key(samples)
    decision = flt.cached(round_id=round_id, key=key)
    if decision is None:
        group = [Trajectory(index=getattr(s, "index", None), reward=_reward(args, s))
                 for s in samples]
        decision = flt(group, round_id=round_id, key=key,
                       max_replacements=getattr(args, "yeto_rl_dynamic_sampling_max_replacements",
                                                None))
    args._yeto_bounded_filter_state = flt.state
    return _output(decision.keep, decision.reason)
