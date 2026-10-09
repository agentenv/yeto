"""agentic-rollout-utilization 1.3: per-trajectory phase timing (backend-neutral).

A trajectory alternates model generation and tool execution, then is judged.
:class:`PhaseClock` turns the harness's tool enter/exit edges into per-turn
durations:

* ``turn_generation_seconds[i]``: from the previous tool exit (or the worker
  start) to the next tool enter (or the result) -- model generation, including
  the agent process's own overhead and engine queueing;
* ``turn_tool_seconds[i]``: one tool execution (sandbox command / submit);
* judging is the verifier's ``evaluate_time`` (recorded by the trusted layer).

``len(turn_generation_seconds) == len(turn_tool_seconds) + 1`` and their sum is
the worker's wall time (``worker_seconds``), so a tape can split each
trajectory's generation segment into the three phases. Wall-clock
``trajectory_started_at``/``trajectory_ended_at`` and the sandbox cold start
(``sandbox_start_seconds``: environment acquire) are recorded by the harness.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

# Optional keys a trajectory record may carry (rl_trajectory_reward).
TIMING_FIELDS: dict[str, tuple[type, ...]] = {
    "trajectory_started_at": (float,),
    "trajectory_ended_at": (float,),
    "sandbox_start_seconds": (float,),
    "worker_seconds": (float,),
    "turn_generation_seconds": (list,),
    "turn_tool_seconds": (list,),
}
MAX_TURNS_RECORDED = 256


class PhaseClock:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = self._mark = clock()
        self._in_tool = False
        self.generation: list[float] = []
        self.tool: list[float] = []

    def enter_tool(self) -> None:
        if self._in_tool:
            return
        now = self._clock()
        self.generation.append(now - self._mark)
        self._mark, self._in_tool = now, True

    def exit_tool(self) -> None:
        if not self._in_tool:
            return
        now = self._clock()
        self.tool.append(now - self._mark)
        self._mark, self._in_tool = now, False

    def finish(self) -> dict[str, Any]:
        """Close the open phase and return the metrics fields."""
        now = self._clock()
        (self.tool if self._in_tool else self.generation).append(now - self._mark)
        self._mark, self._in_tool = now, False
        return {
            "worker_seconds": round(now - self._start, 3),
            "turn_generation_seconds": [round(v, 3) for v in self.generation[:MAX_TURNS_RECORDED]],
            "turn_tool_seconds": [round(v, 3) for v in self.tool[:MAX_TURNS_RECORDED]],
        }


def phase_totals(record: Mapping[str, Any]) -> dict[str, float | None]:
    """The three phase totals of one trajectory record (None = not reported)."""

    def total(key: str) -> float | None:
        values = record.get(key)
        if not isinstance(values, list) or any(
                not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
            return None
        return float(sum(values))

    judge = record.get("evaluate_time")
    return {
        "generation_seconds": total("turn_generation_seconds"),
        "tool_seconds": total("turn_tool_seconds"),
        "judge_seconds": float(judge) if isinstance(judge, (int, float))
        and not isinstance(judge, bool) else None,
    }


def timing_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the timing keys (top-level or under ``agent_metrics``) that are well-typed."""
    sources = [metadata]
    metrics = metadata.get("agent_metrics")
    if isinstance(metrics, Mapping):
        sources.append(metrics)
    out: dict[str, Any] = {}
    for key, types in TIMING_FIELDS.items():
        for source in sources:
            value = source.get(key)
            if value is None:
                continue
            if list in types:
                if isinstance(value, list) and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
                    out[key] = [float(v) for v in value[:MAX_TURNS_RECORDED]]
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
            break
    return out
