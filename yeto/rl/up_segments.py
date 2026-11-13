"""The ``yeto run up`` periods of a run (``state.json`` ``ups``) and what they add up to.

Each ``yeto run up`` process appends one period ``{"pid", "start", "seen", "end"}``
(unix seconds) when it starts driving the pools; its monitor refreshes ``seen`` every
poll and ``yeto run down`` (or the up's own exit) sets ``end``. The per-up clocks
(``budget.wall_h``, starvation, first-group wait) count from ``t0``, the start of the
current up; the run's cumulative runtime is the sum of the periods (the stopped gaps
between a ``down`` and the next ``up`` excluded) and the spend stays cumulative over
the whole run (``run_t0``)."""

from __future__ import annotations

import os
from typing import Any


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def period_end(period: dict[str, Any], now: float, *, alive: Any = _pid_alive) -> float:
    """Where a period ends: its ``end``; for an open one, now while its up process runs,
    else the last time its monitor was seen (a crashed up: the gap after is not counted)."""
    start = float(period.get("start") or 0.0)
    if period.get("end") is not None:
        return max(start, float(period["end"]))
    if period.get("pid") is not None and alive(period["pid"]):
        return max(start, now)
    return max(start, float(period.get("seen") or start))


def uptime_summary(state: dict[str, Any], now: float, *, alive: Any = _pid_alive) -> dict[str, Any]:
    """Cumulative runtime of the run (every up period, gaps excluded), the current (last)
    up's runtime, the number of ups and the last spend estimate the monitor recorded."""
    ups = [p for p in state.get("ups") or () if isinstance(p, dict) and p.get("start") is not None]
    spans = [period_end(p, now, alive=alive) - float(p["start"]) for p in ups]
    last = ups[-1] if ups else None
    return {
        "ups": len(ups),
        "cumulative_runtime_h": round(sum(spans) / 3600, 4),
        "current_up_runtime_h": round(spans[-1] / 3600, 4) if spans else None,
        "current_up_open": bool(last is not None and last.get("end") is None
                                and last.get("pid") is not None and alive(last["pid"])),
        "current_up_start": last.get("start") if last else None,
        "run_t0": state.get("run_t0", state.get("t0")),
        "spend_usd_estimate": state.get("spend_usd_estimate"),
    }
