"""Launch-layer breaker trips: one ``rl_breaker`` line on the launch tape.

The launcher's own breakers (``launch.stall``, ``launch.trainer_failures``,
``launch.teardown_verify``) report through :func:`yeto.breakers.trip`, like the cost
guard does: the event goes to the run's ``cost-events.jsonl`` (the launch layer's tape) and
a stop records the process's ``stop_reason`` (``breaker:<name>``), which the worker writes to
the run record. Exit codes do not change.
"""

from __future__ import annotations

import sys
from typing import Any


def trip(run: str, name: str, value: float | None, limit: float | None, **fields: Any) -> dict | None:
    """Report a launch-layer trip; never raises (the caller's exit path must not change)."""
    from .. import breakers, runs
    from ..cost.guard import EVENTS_FILE, append_event

    path = runs.run_dir(run) / EVENTS_FILE
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    try:
        return breakers.trip(name, value, limit, run=run, emit=lambda rec: append_event(path, rec, log),
                             **fields)
    except Exception as exc:  # noqa: BLE001
        log(f"[launcher] rl_breaker {name} not recorded: {type(exc).__name__}: {exc}")
        return None
