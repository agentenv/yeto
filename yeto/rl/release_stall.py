"""Release stall: the trainer runs, but no new policy release reaches the store.

A trainer that still writes its log (so the silence backstop holds) and whose samplers
still produce (so the starvation watch holds) can still stop publishing: a hung weight
sync, a release upload that retries forever, a round that never ends. The samplers then
serve an ever older policy while every machine bills. :class:`ReleaseStallWatch` is the
head-side verdict ``yeto run up`` takes from the store's ``LATEST`` release pointer:

* the clock starts at the first release seen (none yet: no verdict), and starts again
  at a new release and at a trainer restart (a new launcher: it loads before it publishes);
* the limit is ``limit_s`` when set, else :data:`DEFAULT_LIMIT_S` until
  :data:`LEARN_RELEASES` releases were seen in one trainer launch, then the larger of
  :data:`MIN_LIMIT_S` and :data:`INTERVAL_FACTOR` x the median interval between them;
* past the limit: ``alert`` once per stalled stretch, or ``stop`` with ``action: stop``.
  A pointer that cannot be read is no evidence: no stop.

Neutral: no store, no clock of its own.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Hashable

DEFAULT_LIMIT_S = 3600.0  # the limit until the release interval is known
MIN_LIMIT_S = 1800.0  # the learned limit is never below this
INTERVAL_FACTOR = 5  # the learned limit: this many median release intervals
LEARN_RELEASES = 3  # releases seen in one trainer launch before the interval is learned
DEFAULT_CHECK_S = 300.0  # read the LATEST pointer at most this often
_INTERVALS_KEPT = 20

LATEST = "LATEST.json"  # yeto.rl.engine.policy_release.LATEST (that module imports numpy)

ALERT_EVENT = "trainer_release_stall_alert"
RECOVERED_EVENT = "trainer_release_stall_recovered"
STOP_EVENT = "trainer_release_stall_stop"

OK, ALERT, STOP = "ok", "alert", "stop"


@dataclass(frozen=True)
class ReleaseStallConfig:
    limit_s: float | None = None  # None: learned from the release interval
    check_s: float = DEFAULT_CHECK_S
    stop: bool = False  # action stop (default: an alert only)


@dataclass(frozen=True)
class Verdict:
    state: str  # ok | alert | stop
    stalled_s: float = 0.0
    limit_s: float | None = None
    reason: str = ""
    new_alert: bool = False  # the first alert of this stalled stretch
    recovered: bool = False  # a new release after an alert


@dataclass
class ReleaseStallWatch:
    config: ReleaseStallConfig
    release: Hashable | None = None  # the newest release seen (its LATEST identity)
    since: float | None = None  # the clock's start: the newest release, or the trainer's restart
    incarnation: Hashable | None = None  # the trainer launcher the clock belongs to
    released_at: float | None = None  # when this launcher's newest release was seen
    intervals: list[float] = field(default_factory=list)
    alerted: bool = False

    def limit_s(self) -> float:
        if self.config.limit_s is not None:
            return float(self.config.limit_s)
        if len(self.intervals) >= LEARN_RELEASES - 1:
            return max(MIN_LIMIT_S, INTERVAL_FACTOR * statistics.median(self.intervals))
        return DEFAULT_LIMIT_S

    def observe(self, now: float, *, release: Hashable | None, incarnation: Hashable, read: bool = True) -> Verdict:
        """One check. ``release``: the identity of the store's LATEST release (None: none
        yet); ``read``: the pointer was read this check (False: no evidence, no stop).
        ``incarnation``: the trainer launcher (a new one restarts the clock)."""
        recovered = False
        if incarnation != self.incarnation:
            self.incarnation = incarnation
            self.released_at = None  # an interval across a restart counts the restart
            self.since = now if self.release is not None else None
            self.alerted = False
        if read and release is not None and release != self.release:
            if self.release is not None and self.released_at is not None:
                self.intervals = (self.intervals + [now - self.released_at])[-_INTERVALS_KEPT:]
            self.release, self.since, self.released_at = release, now, now
            recovered, self.alerted = self.alerted, False
        if self.since is None:
            return Verdict(OK, recovered=recovered)
        stalled, limit = max(0.0, now - self.since), self.limit_s()
        reason = (f"no new policy release for {stalled / 60:.0f} min (limit {limit / 60:.0f} min"
                  + ("" if self.config.limit_s is not None else
                     f", {len(self.intervals)} release interval(s) learned") + ")")
        if not read:
            reason += "; the latest read of the release pointer failed"
        if stalled < limit:
            return Verdict(OK, stalled, limit, reason, recovered=recovered)
        new, self.alerted = not self.alerted, True
        if self.config.stop and read:
            return Verdict(STOP, stalled, limit, reason, new_alert=new, recovered=recovered)
        return Verdict(ALERT, stalled, limit, reason, new_alert=new, recovered=recovered)


def release_identity(record: Any) -> Hashable | None:
    """The identity of a LATEST record (its ``seq`` and ``path``); None when it is not one."""
    if not isinstance(record, dict) or record.get("path") is None:
        return None
    return (record.get("seq"), str(record["path"]))
