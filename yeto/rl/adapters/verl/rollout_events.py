"""verl -> yeto unified rollout event fields (agentic-rollout-utilization 6.2).

Miles and verl report the same fields with the same meaning:

* per trajectory (``rl_trajectory_reward`` optional keys): ``generation_seconds``
  (model generation), ``tool_seconds`` (tool execution), ``evaluate_time``
  (judging) -- from verl's agent-loop per-sample timers ``generate_sequences``
  / ``tool_calls`` / ``compute_score`` (verl ``agent_loop.py:82-83, 1135``);
  verl records one policy-version interval per trajectory
  (``min_global_steps``/``max_global_steps``), translated to the version
  segments' bounds ``policy_versions`` (oldest, newest);
* per round (``rl_rollout_cutoff``): the discard tally
  ``{groups, samples, response_tokens, unknown_groups}`` through the same
  :func:`yeto.rl.engine.rollout_cutoff.discard_stats_fields` as Miles.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yeto.rl.engine.rollout_cutoff import discard_stats_fields

_TIMERS = {"generate_sequences": "generation_seconds", "tool_calls": "tool_seconds",
           "compute_score": "evaluate_time"}


def trajectory_fields(metrics: Mapping[str, Any] | None,
                      non_tensor: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One verl sample's agent-loop metrics (+ non-tensor fields) -> unified keys.
    Missing or non-numeric values are left out (unknown, never guessed)."""
    out: dict[str, Any] = {}
    for src, dst in _TIMERS.items():
        value = (metrics or {}).get(src)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            out[dst] = round(float(value), 3)
    lo = (non_tensor or {}).get("min_global_steps")
    hi = (non_tensor or {}).get("max_global_steps")
    if type(lo) is int and type(hi) is int and 0 <= lo <= hi:
        out["policy_versions"] = [lo] if lo == hi else [lo, hi]
    return out


def cutoff_fields(discard: Mapping[str, Any] | None) -> dict[str, int]:
    """verl's discard tally -> the rollout-metadata fields Miles reports."""
    return discard_stats_fields(discard)
