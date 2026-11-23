"""Infrastructure streak of an agentic sampler (breaker ``sampler.infra_streak``).

A gateway or sandbox provider outage fails every trial within seconds of its
start: each group ends ``below_min_samples`` (its task goes out again, counted
toward the dispenser's redo limit) and the sampler burns through its tasks in
minutes. :class:`InfraStreak` counts the agentic groups in a row that ended that
way; a written group ends the streak. A group that also lost a trial to a drain
cut or to a weight switch (``drain_timeout``, ``served_by_other_version``) is not
the infrastructure's: a release cuts several groups at once on a healthy run, so
such a group neither counts nor ends the streak. At ``max_n`` it trips once per
streak:

* action ``warn`` (default): one ``rl_breaker`` record, nothing else changes;
* action ``pause``: the store dispenser claims no new task
  (``StoreDispensedPrompts.paused``) while the trials in flight go on. It claims
  again once a group is written or ``cooldown_s`` after the trip, whichever comes
  first; ``rl_breaker`` marks both edges (``trip`` / ``clear``). Without a store
  dispenser the pause has nothing to hold and the trip only warns.

Configured in the spec's ``agentic.infra_streak`` (``enabled``, ``max_n``,
``cooldown_s``, ``action``) or the breakers section (``breakers.sampler.infra_streak``).
The sampler prints its setting at start (:func:`startup_line`). To hold or release
a pool member by hand, ``yeto run pause`` / ``yeto run resume``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from .sampler_breakers import trip

NAME = "sampler.infra_streak"
DEFAULT_MAX_N = 5
DEFAULT_COOLDOWN_S = 300.0
INFRA_REASON = "below_min_samples"  # a group lost to infrastructure (agentic_round._settle)
# trial losses that are not the infrastructure's (agentic_round.LOST_DRAIN / LOST_VERSION)
NOT_INFRA_LOSSES = frozenset({"drain_timeout", "served_by_other_version"})


class InfraStreak:
    """Groups in a row lost to infrastructure (module doc). Called on the agentic
    round's calling thread only (``group`` from the settle, ``paused`` from the
    dispenser's claim)."""

    def __init__(self, *, max_n: int = DEFAULT_MAX_N, cooldown_s: float = DEFAULT_COOLDOWN_S,
                 action: str = "warn", clock: Callable[[], float] = time.monotonic) -> None:
        if int(max_n) < 1:
            raise ValueError(f"infra_streak max_n must be >= 1, got {max_n}")
        if action not in ("warn", "pause"):
            raise ValueError(f"infra_streak action must be warn or pause, got {action!r}")
        self.max_n, self.cooldown_s, self.action = int(max_n), float(cooldown_s), action
        self.clock = clock
        self.emit: Callable[..., None] | None = None  # the sampler's tape (AgenticRound sets it)
        self.streak = 0
        self.trips = 0
        self.paused_at: float | None = None
        self._tripped = False  # this streak has tripped

    @classmethod
    def of(cls, resolved: Any) -> "InfraStreak | None":
        """The streak of the resolved ``sampler.infra_streak`` entry; None when it is off."""
        entry = resolved.entry(NAME)
        if not entry.enabled:
            return None
        return cls(max_n=entry.params["max_n"], cooldown_s=entry.params["cooldown_s"],
                   action=entry.action or "warn")

    def group(self, reason: str | None, lost: Mapping[str, int] | None = None) -> None:
        """One settled group: ``reason`` None (written) ends the streak, ``below_min_samples``
        adds to it unless ``lost`` (its trial losses by reason) has a drain or version loss
        (module doc); any other reason (fenced, dispense_lost, ...) is not the infrastructure's."""
        if reason is None:
            if self.paused_at is not None:
                self._resume("a group was written")
            self.streak, self._tripped = 0, False
            return
        if reason != INFRA_REASON or any((lost or {}).get(k) for k in NOT_INFRA_LOSSES):
            return
        self.streak += 1
        if self.streak < self.max_n or self._tripped:
            return
        self._tripped = True
        self.trips += 1
        pausing = self.action == "pause"
        if pausing:
            self.paused_at = self.clock()
        trip(self.emit, NAME, self.streak, self.max_n, action=self.action,
             detail=f"{self.streak} groups in a row lost to infrastructure"
                    + (f"; no new task for up to {self.cooldown_s:g} s" if pausing else ""))

    def paused(self) -> bool:
        """True while a ``pause`` trip holds new claims (the store dispenser asks)."""
        if self.paused_at is None:
            return False
        if self.clock() - self.paused_at >= self.cooldown_s:
            self._resume(f"cool-down of {self.cooldown_s:g} s")
            return False
        return True

    def _resume(self, why: str) -> None:
        held = self.clock() - (self.paused_at or self.clock())
        self.paused_at = None
        self.streak, self._tripped = 0, False
        trip(self.emit, NAME, round(held, 1), self.cooldown_s, state="clear", action="pause", unit="s",
             detail=f"claiming tasks again: {why}")


def startup_line(streak: InfraStreak | None) -> str:
    """The line the sampler prints at start: on or off, the limit, how to change it."""
    off = "off: agentic.infra_streak.enabled false (or breakers.sampler.infra_streak.enabled false)"
    if streak is None:
        return f"[agentic] {NAME} off (on: agentic.infra_streak.enabled true)"
    what = (f"no new task claim for up to {streak.cooldown_s:g} s" if streak.action == "pause"
            else "rl_breaker warning only (pause new claims: agentic.infra_streak.action pause)")
    return (f"[agentic] {NAME} on: after {streak.max_n} groups in a row lost to infrastructure, {what}; "
            f"limit: agentic.infra_streak.max_n; {off}")
