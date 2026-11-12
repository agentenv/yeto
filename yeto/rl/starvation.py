"""Trainer starvation: the trainer is up, but no new group reaches the buffer.

When every sampler is gone (all died, the pool cannot replace them, every account is
halted) the trainer keeps waiting for data: its buffer reader (or, with the trainer
scheduler, the head waiting for the RUN ack) waits while the trainer's GPUs bill.
:class:`StarvationWatch` is the head-side verdict ``yeto run up`` takes each monitor poll
from two signals it already has:

* the object keys under the run store's ``sample-buffer/shards/`` (a key not seen in the
  previous listing is a new group, or a new index object written with one);
* the sampler pools' own state (members in ``ready``, the pool loop alive).

Verdicts, both measured from the last new group (or the start of the watch):

* ``alert`` after ``alert_s`` without a new group, once per starved stretch (an
  alert only; agentic trials may run for a long time, so the default is wide);
* ``stop`` after ``stop_s`` without a new group **and** no healthy sampler for at
  least ``unhealthy_s`` (no member ready, or no pool loop driving one). A ready but
  paused member (``yeto run pause``, in any state) is an operator's choice: no stop while
  one is paused. A replacement under way (unpaused provisioning / starting members within
  their pool's ``elastic.start_timeout_s``, for at most that long without a healthy
  sampler) holds the stop too. Pool state that cannot be read (any pool's) and a failed buffer listing
  are no evidence either way: no stop.

``0`` turns a verdict off. Neutral: no store, no pool, no clock of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_ALERT_MIN = 30.0
DEFAULT_STOP_MIN = 120.0
DEFAULT_UNHEALTHY_MIN = 10.0
DEFAULT_CHECK_S = 300.0  # store listings of the buffer at most this often
DEFAULT_STOP_GRACE_MIN = 20.0  # after the STOP flag: wait this long for the launcher, then ``yeto down``

ALERT_EVENT = "trainer_starvation_alert"
RECOVERED_EVENT = "trainer_starvation_recovered"
STOP_EVENT = "trainer_starvation_stop"

OK, ALERT, STOP = "ok", "alert", "stop"


@dataclass(frozen=True)
class StarvationConfig:
    alert_s: float = DEFAULT_ALERT_MIN * 60
    stop_s: float = DEFAULT_STOP_MIN * 60
    unhealthy_s: float = DEFAULT_UNHEALTHY_MIN * 60
    check_s: float = DEFAULT_CHECK_S
    stop_grace_s: float = DEFAULT_STOP_GRACE_MIN * 60

    @property
    def enabled(self) -> bool:
        return self.alert_s > 0 or self.stop_s > 0

    @classmethod
    def from_section(cls, section: Mapping[str, Any] | None) -> StarvationConfig:
        """The run config's ``starvation`` section (minutes, ``check_s`` in seconds).
        Raises ValueError on a bad value."""
        s = dict(section or {})
        unknown = sorted(set(s) - {"alert_min", "stop_min", "unhealthy_min", "check_s", "stop_grace_min"})
        if unknown:
            raise ValueError(f"starvation: unknown key(s) {unknown}")

        def num(key: str, default: float) -> float:
            value = s.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"starvation.{key} must be a number >= 0 (got {value!r})")
            return float(value)

        cfg = cls(alert_s=num("alert_min", DEFAULT_ALERT_MIN) * 60, stop_s=num("stop_min", DEFAULT_STOP_MIN) * 60,
                  unhealthy_s=num("unhealthy_min", DEFAULT_UNHEALTHY_MIN) * 60,
                  check_s=num("check_s", DEFAULT_CHECK_S),
                  stop_grace_s=num("stop_grace_min", DEFAULT_STOP_GRACE_MIN) * 60)
        if cfg.check_s <= 0 and cfg.enabled:
            raise ValueError("starvation.check_s must be > 0")
        if cfg.alert_s > 0 and cfg.stop_s > 0 and cfg.stop_s < cfg.alert_s:
            raise ValueError(f"starvation.stop_min ({cfg.stop_s / 60:g}) must be >= alert_min "
                             f"({cfg.alert_s / 60:g}) (or 0: no stop)")
        return cfg


@dataclass(frozen=True)
class SamplerHealth:
    """The sampler side as the pools report it. ``healthy``: ready, unpaused members of
    pools whose loop runs (None: no pool state readable)."""
    healthy: int | None
    paused: int = 0
    live: int = 0
    detail: str = ""
    # unpaused provisioning / starting members of running pools still within their pool's
    # ``elastic.start_timeout_s`` (a replacement under way), and the longest such timeout
    recovering: int = 0
    recover_s: float = 0.0

    def phrase(self) -> str:
        if self.healthy is None:
            return f"sampler pools: state unreadable{'; ' + self.detail if self.detail else ''}"
        text = f"{self.healthy} healthy sampler(s), {self.live} live member(s), {self.paused} paused"
        if self.recovering:
            text += f", recovering: {self.recovering} member(s) starting"
        return text + (f" ({self.detail})" if self.detail else "")


@dataclass(frozen=True)
class Verdict:
    state: str  # ok | alert | stop
    starved_s: float = 0.0
    reason: str = ""
    new_alert: bool = False  # the first alert of this starved stretch
    recovered: bool = False  # a new group arrived after an alert


class StarvationWatch:
    def __init__(self, config: StarvationConfig, now: float) -> None:
        self.config = config
        self.since = float(now)  # last new group (or the watch start)
        self.unhealthy_since: float | None = None
        self.recovering_since: float | None = None  # replacements seen through this unhealthy stretch
        self.alerted = False

    def observe(self, now: float, *, new_groups: int | None, health: SamplerHealth,
                cause: str | None = None, listed: bool = True) -> Verdict:
        """One poll. ``new_groups``: buffer objects the listing found that the previous
        listing did not (None: not listed this poll, the listing failed, or the first
        listing). ``cause``: what the trainer side reports (e.g. the incarnation gate's
        fail-closed pause), named in the reason. ``listed``: the latest listing of the
        buffer succeeded; after a failed one there is no stop (no evidence of no group)."""
        recovered = False
        if new_groups:
            self.since = now
            recovered = self.alerted
            self.alerted = False
        if health.healthy == 0:
            if self.unhealthy_since is None:
                self.unhealthy_since = now
        else:
            self.unhealthy_since = None
        starved = max(0.0, now - self.since)
        cfg = self.config
        reason = f"no new group in the buffer for {starved / 60:.0f} min; {health.phrase()}"
        if cause:
            reason += f"; trainer side: {cause}"
        if not listed:
            reason += "; the latest buffer listing failed"
        # a replacement under way holds the stop for at most the pool's start timeout from the
        # first replacement of this stretch (a pool that orders again and again is stuck)
        if health.healthy == 0 and health.recovering:
            if self.recovering_since is None:
                self.recovering_since = now
        elif health.healthy != 0:
            self.recovering_since = None
        recovering = bool(health.recovering and self.recovering_since is not None
                          and now - self.recovering_since < health.recover_s)
        if (listed and not recovering and cfg.stop_s > 0 and starved >= cfg.stop_s and health.healthy == 0
                and health.paused == 0
                and self.unhealthy_since is not None and now - self.unhealthy_since >= cfg.unhealthy_s):
            return Verdict(STOP, starved, reason + f" (stop after {cfg.stop_s / 60:.0f} min with no "
                           f"healthy sampler for {(now - self.unhealthy_since) / 60:.0f} min)",
                           new_alert=not self.alerted, recovered=recovered)
        if cfg.alert_s > 0 and starved >= cfg.alert_s:
            new = not self.alerted
            self.alerted = True
            why = ""
            if cfg.stop_s > 0 and listed and not recovering and health.healthy == 0 and health.paused == 0:
                why = f" (the trainer is stopped at {cfg.stop_s / 60:.0f} min if no sampler recovers)"
            elif cfg.stop_s > 0:
                why = " (no stop: a sampler is healthy, paused or starting, or the pool state or the buffer listing is unreadable)"
            return Verdict(ALERT, starved, reason + why, new_alert=new, recovered=recovered)
        return Verdict(OK, starved, reason, recovered=recovered)
