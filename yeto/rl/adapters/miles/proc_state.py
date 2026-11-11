"""Process-local runtime state for yeto code running inside Miles processes.

The registry is engine-neutral (:mod:`yeto.rl.engine.proc_state`). Importing
this module installs Miles' rollout statistics channel
(``miles.rollout.rollout_runtime_stats.take(kind)``, k31 F1/B5) as the source
:func:`take_runtime_stats` reads; only a Miles without the module (the
ports-v1 fork) leaves the legacy ``args.rollout_<kind>_stats`` fallback.
"""

from __future__ import annotations

from typing import Any

from yeto.rl.engine.proc_state import (
    RUNTIME_STATS_UNAVAILABLE,
    STORE_ATTR,
    has_proc_state,
    pop_proc_state,
    proc_state,
    register_runtime_stats_source,
    reset_proc_state,
    set_proc_state,
    take_runtime_stats,
)


def miles_runtime_stats(kind: str) -> Any:
    """``miles.rollout.rollout_runtime_stats.take(kind)``;
    :data:`RUNTIME_STATS_UNAVAILABLE` on a Miles without the module."""
    try:
        from miles.rollout import rollout_runtime_stats
    except ImportError:
        return RUNTIME_STATS_UNAVAILABLE
    return rollout_runtime_stats.take(kind)


register_runtime_stats_source(miles_runtime_stats)

__all__ = [
    "RUNTIME_STATS_UNAVAILABLE",
    "STORE_ATTR",
    "has_proc_state",
    "miles_runtime_stats",
    "pop_proc_state",
    "proc_state",
    "register_runtime_stats_source",
    "reset_proc_state",
    "set_proc_state",
    "take_runtime_stats",
]
