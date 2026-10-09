"""Core RL event writer (yeto-framework-decoupling 2.3). Import-light.

The engine core and the neutral algorithm code write tape events through
:func:`write_event`; the backend injects its implementation with
:func:`set_event_writer`. The default, :func:`append_tape_event`, is the
island tape format every backend shares today (moved unchanged from
``yeto.rl.adapters.miles.legacy.engine._append_rl_event``, which now re-exports it): one JSONL
record ``{"island_id", "time_unix", **event}`` appended to
``args.yeto_rl_event_tape`` (echoed when ``YETO_RL_ECHO_EVENTS=1``), then
teed to W&B. Event names and fields are never rewritten here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

EventWriter = Callable[[Any, dict[str, Any]], None]


def append_tape_event(args, event: dict[str, Any]) -> None:
    path = Path(args.yeto_rl_event_tape).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "island_id": int(args.yeto_rl_learner_id),
        "time_unix": time.time(),
        **event,
    }
    from yeto.rl.event_echo import append_record

    append_record(path, event)  # echoed when YETO_RL_ECHO_EVENTS=1
    # The tape is the island's telemetry; W&B is a second reader of it, not
    # a second instrumentation pass. Writing the file first keeps the tape
    # authoritative when the network is not.
    from yeto.rl.wandb_rl import tee as _wandb_tee

    _wandb_tee(args, event)


_writer: EventWriter | None = None


def set_event_writer(writer: EventWriter | None) -> EventWriter | None:
    """Install the backend's event writer (``None`` restores the default); returns the previous one."""
    global _writer
    previous, _writer = _writer, writer
    return previous


def event_writer() -> EventWriter:
    return _writer if _writer is not None else append_tape_event


def write_event(args, event: dict[str, Any]) -> None:
    """Write one event to the configured tape through the injected writer."""
    event_writer()(args, event)
