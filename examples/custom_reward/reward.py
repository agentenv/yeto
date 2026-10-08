"""Minimal user-defined reward (decoupling 3.10): no training framework needed.

A neutral reward reads a ``Trajectory`` and returns a ``RewardResult``.  Here:
1.0 when the response ends with the label (after stripping), else 0.0, and the
response length goes into ``metadata`` for inspection.

Use it from Miles either by registered name (import this module first) or by
reference::

    from yeto.rl.engine.miles_adapter.rewards import miles_custom_rm_path
    miles_custom_rm_path("examples.custom_reward.reward:ends_with_label")
    # -> value for --custom-rm-path
"""

from __future__ import annotations

from yeto.rl.rewards.registry import register_reward
from yeto.rl.rewards.types import RewardResult, Trajectory


@register_reward("ends_with_label")
def ends_with_label(trajectory: Trajectory) -> RewardResult:
    label = "" if trajectory.label is None else str(trajectory.label).strip()
    hit = bool(label) and trajectory.response.strip().endswith(label)
    return RewardResult(value=1.0 if hit else 0.0,
                        metadata={"success": hit, "response_chars": len(trajectory.response)})
