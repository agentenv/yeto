"""When a round cut is due: every N rounds, or by wall time (S27 R-2).

:class:`FixedCutInterval` is ``--rl-cut-every N`` (the default, N = 1).

:class:`AutoCutInterval` (``--rl-cut-auto-mtbf-s``) is a timed
interval: with ``C`` the
seconds one cut costs the round's critical path and ``MTBF`` the mean seconds
between interruptions, a cut is due once ``T = sqrt(2 * C * MTBF)`` seconds have
passed since the previous one, with ``T`` clamped to
``[min_interval_s, max_interval_s]`` (the upper bound caps the loss when the MTBF
prior is wrong). ``C`` is the mean of the last few measured cuts; before the first
measurement the first safe point cuts (and measures). Rounds do not need a fixed
length: the decision is by elapsed time, taken once per safe point.

The write time of a background cut is not part of ``T``: a periodic cut is
skipped while the previous one still commits (``RoundCutCheckpoint._commit_busy``),
which is the plan's lower bound ``k >= write time / round time``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable

DEFAULT_MAX_INTERVAL_S = 3600.0  # plan C: at most one hour between cuts
COST_WINDOW = 4  # cuts whose measured cost is averaged


def check_every(every: int) -> int:
    if int(every) < 1:
        raise ValueError("--rl-cut-every must be >= 1")
    return int(every)


class FixedCutInterval:
    """A cut every ``every`` rounds (and always at the end)."""

    def __init__(self, every: int) -> None:
        self.every = check_every(every)

    def due(self, rollout_id: int, *, final: bool = False) -> bool:
        from yeto.rl.engine.resume import should_cut

        return should_cut(int(rollout_id), every=self.every, final=final)

    def record_cut(self, rollout_id: int, *, cost_s: float) -> None:
        pass

    def describe(self) -> dict[str, object]:
        return {"mode": "rounds", "every": self.every}


class AutoCutInterval:
    """A cut once ``sqrt(2 * C * MTBF)`` seconds have passed since the last one (module doc)."""

    def __init__(self, *, mtbf_s: float, max_interval_s: float = DEFAULT_MAX_INTERVAL_S,
                 min_interval_s: float = 0.0, clock: Callable[[], float] = time.monotonic) -> None:
        if not mtbf_s > 0:
            raise ValueError("--rl-cut-auto-mtbf-s must be positive")
        if not max_interval_s > 0:
            raise ValueError("--rl-cut-max-interval-s must be positive")
        if not 0 <= min_interval_s <= max_interval_s:
            raise ValueError("the minimum cut interval must lie in [0, max interval]")
        self.mtbf_s = float(mtbf_s)
        self.max_interval_s = float(max_interval_s)
        self.min_interval_s = float(min_interval_s)
        self.clock = clock
        self._costs: deque[float] = deque(maxlen=COST_WINDOW)
        self._last_cut: tuple[int, float] | None = None  # (rollout id, clock) of the last cut
        self._decided: tuple[int, bool] | None = None  # the answer for the current safe point
        self.every = 1  # rounds between the last two cuts (publication checks follow it)

    @property
    def cost_s(self) -> float | None:
        return sum(self._costs) / len(self._costs) if self._costs else None

    def interval_s(self) -> float | None:
        """``T`` (None before the first measured cut)."""
        cost = self.cost_s
        if cost is None:
            return None
        optimum = math.sqrt(2.0 * max(cost, 0.0) * self.mtbf_s)
        return min(max(optimum, self.min_interval_s), self.max_interval_s)

    def due(self, rollout_id: int, *, final: bool = False) -> bool:
        """Is a cut due at the safe point before ``rollout_id``? Decided once per
        rollout id, so the prepare before the trainer offloads, the publication
        check and the safe point itself agree."""
        rollout_id = int(rollout_id)
        if rollout_id <= 0:
            return False
        if final:
            return True
        if self._decided is not None and self._decided[0] == rollout_id:
            return self._decided[1]
        interval = self.interval_s()
        due = (interval is None or self._last_cut is None
               or self.clock() - self._last_cut[1] >= interval)
        self._decided = (rollout_id, due)
        return due

    def record_cut(self, rollout_id: int, *, cost_s: float) -> None:
        """A cut was taken at the safe point before ``rollout_id``; it cost the round
        ``cost_s`` seconds of critical path."""
        rollout_id = int(rollout_id)
        if self._last_cut is not None and rollout_id > self._last_cut[0]:
            self.every = rollout_id - self._last_cut[0]
        self._last_cut = (rollout_id, self.clock())
        self._costs.append(max(float(cost_s), 0.0))

    def describe(self) -> dict[str, object]:
        interval = self.interval_s()
        return {"mode": "auto", "mtbf_s": self.mtbf_s, "max_interval_s": self.max_interval_s,
                "cost_s": None if self.cost_s is None else round(self.cost_s, 3),
                "interval_s": None if interval is None else round(interval, 1)}


def cut_interval_for(*, every: int | None, auto_mtbf_s: float | None,
                     max_interval_s: float | None) -> FixedCutInterval | AutoCutInterval:
    """The interval of the cut flags (``--rl-cut-every`` / ``--rl-cut-auto-mtbf-s`` /
    ``--rl-cut-max-interval-s``); the two modes exclude each other."""
    from yeto.rl.engine.resume import DEFAULT_CUT_EVERY

    if auto_mtbf_s is None:
        if max_interval_s is not None:
            raise ValueError("--rl-cut-max-interval-s needs --rl-cut-auto-mtbf-s")
        return FixedCutInterval(DEFAULT_CUT_EVERY if every is None else every)
    if every is not None:
        raise ValueError("--rl-cut-every and --rl-cut-auto-mtbf-s exclude each other")
    return AutoCutInterval(mtbf_s=float(auto_mtbf_s),
                           max_interval_s=(DEFAULT_MAX_INTERVAL_S if max_interval_s is None
                                           else float(max_interval_s)))
