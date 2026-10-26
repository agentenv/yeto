"""Admission gate and harness metrics of the agentic outer layer (S25 N10.2; was L10).

One trajectory holds one sandbox for its whole life, so the number of
trajectories in flight is the number of live sandboxes.  The gate bounds it
(provider quota, cost) and supports drain: after :meth:`AdmissionGate.drain`
no new trajectory starts, running ones finish.  The policy version is read
at admission, so every trajectory carries the version it started under.

Metric names reuse the engine timeline gauges (``harness_in_flight``,
``env_live``) so dashboards read the Harbor path like the Codex path.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from .types import Outcome


class AdmissionClosed(RuntimeError):
    """The gate is draining; the sample must be aborted, not started."""


@dataclass
class Admission:
    """Ticket for one admitted trajectory."""

    policy_version: int | None


class HarnessMetrics:
    """Counters and gauges of the agentic path, per process."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.admitted = 0
        self.refused = 0
        self.outcomes: Counter[str] = Counter()

    def record_outcome(self, outcome: Outcome) -> None:
        self.outcomes[outcome.value] += 1

    def reclassify(self, old: Outcome, new: Outcome) -> None:
        """A backend overruled the outcome after the trial (e.g. token alignment failed)."""
        self.outcomes[old.value] -= 1
        self.outcomes[new.value] += 1

    def snapshot(self) -> dict[str, int]:
        return {
            "harness_in_flight": self.in_flight,
            # Harbor owns the sandbox for exactly the life of the trial.
            "env_live": self.in_flight,
            "harness_admitted_total": self.admitted,
            "harness_refused_total": self.refused,
            **{f"harness_outcome_{o.value}_total": self.outcomes[o.value] for o in Outcome},
        }


class AdmissionGate:
    """Bound the trajectories (= sandboxes) in flight; refuse new ones while draining."""

    def __init__(self, max_in_flight: int, *, policy_version: Callable[[], int | None] = lambda: None,
                 metrics: HarnessMetrics | None = None) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self._slots = asyncio.Semaphore(max_in_flight)
        self._policy_version = policy_version
        self._draining = False
        self.metrics = metrics or HarnessMetrics()

    def drain(self) -> None:
        self._draining = True

    def reopen(self) -> None:
        self._draining = False

    @contextlib.asynccontextmanager
    async def admit(self) -> AsyncIterator[Admission]:
        if self._draining:
            self.metrics.refused += 1
            raise AdmissionClosed("admission is draining")
        async with self._slots:
            if self._draining:  # drain started while this one waited for a slot
                self.metrics.refused += 1
                raise AdmissionClosed("admission is draining")
            self.metrics.admitted += 1
            self.metrics.in_flight += 1
            try:
                yield Admission(policy_version=self._policy_version())
            finally:
                self.metrics.in_flight -= 1
