"""Scheduler S1: start on backlog, stop by ski rental (rl-trainer-start-stop D6, D8).

* Start when the backlog ``Q`` (in batches) reaches ``Q_on``. Default
  ``Q_on = A_hard + 1``. With the optional ``alpha`` (allowed start-cost share)
  ``Q_on = min(Q_s1, A_hard + 1)``, ``Q_s1 = ceil(T_s * (1 - alpha) / (alpha * T_b))``.
  ``alpha`` has no default.
* Stop (``STOPPED``) when the backlog is empty and the trainer has been idle for
  ``T_idle = T_s``; stop (``FINISHED``) when the data ended and the backlog is empty.
* ``T_s`` is the last measured start (``trainer_start_summary.total_s``, the event of
  rl-trainer-cold-start-cost L0.1) or, without a measurement, the configured value.
* Prices are accepted and ignored (S3 is not in this change).

The idle start time is the only state. It is a wall-clock time, so a scheduler
restored from the controller journal after a head restart continues with the idle
time already spent (D8). The journal record has kind ``trainer_scheduler_state``
and holds the inputs and the decision, in the same way as ``auto_state``
(``elastic.py``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from yeto.rl.elastic_client import LeaveReason

from .trainer_lifecycle import BacklogStats, Decision, Hold, LifecycleState, Start, Stop

START_SUMMARY_EVENT = "trainer_start_summary"  # rl-trainer-cold-start-cost L0.1
SCHEDULER_STATE_KIND = "trainer_scheduler_state"

# A cold trainer starts from these states; a running trainer is judged for stop
# only while TRAINING. The states in between are transitions: hold.
_STARTABLE = (LifecycleState.COLD, LifecycleState.STOPPED)

Record = Callable[..., None]  # controller._record(kind, tx_id=None, **fields)


def last_start_seconds(events: Iterable[Mapping[str, Any]]) -> float | None:
    """``total_s`` of the last complete ``trainer_start_summary``, or None."""
    last = None
    for e in events:
        if e.get("event") == START_SUMMARY_EVENT and e.get("total_s") is not None:
            last = float(e["total_s"])
    return last


@dataclass(frozen=True)
class S1Config:
    """``start_seconds``: configured ``T_s``, used when no start was measured.
    ``batch_seconds`` (``T_b``) is needed only with ``alpha``."""

    a_hard: int
    start_seconds: float
    alpha: float | None = None
    batch_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.a_hard < 0:
            raise ValueError(f"a_hard must be >= 0, got {self.a_hard}")
        if self.start_seconds < 0:
            raise ValueError(f"start_seconds must be >= 0, got {self.start_seconds}")
        if self.alpha is not None:
            if not 0.0 < self.alpha <= 1.0:
                raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
            if not self.batch_seconds or self.batch_seconds <= 0:
                raise ValueError("alpha needs batch_seconds > 0 (T_b)")


class S1Scheduler:
    """:class:`~.trainer_lifecycle.TrainerScheduler` S1.

    ``start_events`` returns the start-phase events read so far (any iterable of
    event dicts); the newest summary gives ``T_s``. ``record`` is the controller
    journal writer; without it the state is not journaled."""

    def __init__(self, config: S1Config, *,
                 start_events: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
                 record: Record | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.config = config
        self._start_events = start_events
        self._record = record
        self._clock = clock
        self.idle_since: float | None = None

    # ------------------------------------------------------------ inputs
    def start_seconds(self) -> float:
        measured = last_start_seconds(self._start_events()) if self._start_events else None
        return self.config.start_seconds if measured is None else measured

    def q_on(self, start_seconds: float) -> float:
        cap = self.config.a_hard + 1
        if self.config.alpha is None:
            return cap
        a, t_b = self.config.alpha, self.config.batch_seconds
        q_s1 = math.ceil(start_seconds * (1.0 - a) / (a * t_b))
        return min(q_s1, cap)

    # ------------------------------------------------------------ decision
    def decide(self, backlog: BacklogStats, lifecycle_state: LifecycleState,
               prices: Mapping[str, float] | None = None) -> Decision:
        del prices  # S1 ignores prices (D6)
        now = self._clock()
        t_s = self.start_seconds()
        q_on = self.q_on(t_s)
        before = self.idle_since
        decision = self._decide(backlog, lifecycle_state, now, t_s, q_on)
        self._journal(backlog, lifecycle_state, now, t_s, q_on, decision, before)
        return decision

    def _decide(self, backlog: BacklogStats, state: LifecycleState, now: float,
                t_s: float, q_on: float) -> Decision:
        empty = backlog.samples == 0
        if state in _STARTABLE:
            self.idle_since = None
            if backlog.batches >= q_on or (backlog.data_ended and not empty):
                return Start()  # at the data end, train the remainder below Q_on
            return Hold()
        if state is not LifecycleState.TRAINING:
            return Hold()
        if not empty:
            self.idle_since = None
            return Hold()
        if backlog.data_ended:
            return Stop(LeaveReason.FINISHED)
        if self.idle_since is None:
            self.idle_since = now
        if now - self.idle_since >= t_s:
            self.idle_since = None
            return Stop(LeaveReason.STOPPED)
        return Hold()

    # ------------------------------------------------------------ journal (D8)
    def _journal(self, backlog: BacklogStats, state: LifecycleState, now: float,
                 t_s: float, q_on: float, decision: Decision, before: float | None) -> None:
        # Write when the state changes or the decision acts; a steady Hold is not news.
        if self._record is None or (self.idle_since == before and isinstance(decision, Hold)):
            return
        self._record(SCHEDULER_STATE_KIND, tx_id=None,
                     state={"idle_since": self.idle_since},
                     inputs={"at": now, "lifecycle_state": state.value,
                             "batches": backlog.batches, "samples": backlog.samples,
                             "data_ended": backlog.data_ended,
                             "start_seconds": t_s, "q_on": q_on},
                     decision=_decision_dict(decision))

    def restore(self, records: Iterable[Mapping[str, Any]]) -> bool:
        """Apply the last journaled state; False if there is none."""
        last = None
        for r in records:
            if r.get("kind") == SCHEDULER_STATE_KIND:
                last = r
        if last is None:
            return False
        self.idle_since = last["state"].get("idle_since")
        return True


def _decision_dict(decision: Decision) -> dict[str, Any]:
    if isinstance(decision, Stop):
        return {"action": "stop", "reason": decision.reason.name}
    return {"action": "start" if isinstance(decision, Start) else "hold"}
