"""When a round cut is due: every N rounds, or by wall time (S27 R-2).

:class:`FixedCutInterval` is ``--rl-cut-every N``.

The default (neither flag; S27 G-AT) is a timed interval with a prior MTBF of
:data:`DEFAULT_MTBF_S` and a cost cap: a cut costs at most
:data:`DEFAULT_MAX_COST_SHARE` of the wall time, so the gap is at least
``C / share`` (a 30B trainer whose cut costs about 400 s cuts about every
8000 s, not every round). Its first cut waits :data:`DEFAULT_FIRST_CUT_S` after
the first safe point (S27 G-CD: a cut at the first safe point saves one round;
on s27-bb-b1 it cost a Nano DP8 trainer 144 s on the round, then 529 s of
store sync and a store commit that timed out after 600 s). ``--rl-cut-every`` and ``--rl-cut-auto-mtbf-s`` replace it.

:class:`AutoCutInterval` (``--rl-cut-auto-mtbf-s``) is a timed
interval: with ``C`` the
seconds one cut costs and ``MTBF`` the mean seconds
between interruptions, a cut is due once ``T = sqrt(2 * C * MTBF)`` seconds have
passed since the previous one, with ``T`` clamped to
``[min_interval_s, max_interval_s]`` (the upper bound caps the loss when the MTBF
prior is wrong). ``C`` is the mean of the last few measured cuts; before the first
measurement the first safe point cuts (and measures), or the first one
``first_cut_s`` after the first safe point. Rounds do not need a fixed
length: the decision is by elapsed time, taken once per safe point.

``C`` of one cut is its critical-path seconds (:meth:`AutoCutInterval.record_cut`)
plus, once its background part is done, the seconds of that part: collecting
the trainer's shards, the store sync and the store commit
(:meth:`AutoCutInterval.add_cost`, S27 G-CD; s27-bb-b1, a 473 GB cut: 144 s on
the round, 529 s of store sync, then a commit that timed out after 600 s). A periodic cut is
also skipped while the previous one still commits (``RoundCutCheckpoint._commit_busy``),
which is the plan's lower bound ``k >= write time / round time``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable

DEFAULT_MAX_INTERVAL_S = 3600.0  # plan C: at most one hour between cuts
DEFAULT_MTBF_S = 21600.0  # the default interval's prior: six hours between interruptions (spot)
DEFAULT_MAX_COST_SHARE = 0.05  # the default interval: cuts cost at most 5% of the wall time
# the default interval's first cut: Young-Daly with the MTBF prior for a cut of 75 s, and
# the 5% cap holds for a first cut of up to 90 s (a larger one spaces the next cuts)
DEFAULT_FIRST_CUT_S = 1800.0
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

    def add_cost(self, rollout_id: int, *, cost_s: float) -> None:
        pass

    def describe(self) -> dict[str, object]:
        return {"mode": "rounds", "every": self.every}


class AutoCutInterval:
    """A cut once ``sqrt(2 * C * MTBF)`` seconds have passed since the last one (module doc).

    ``max_cost_share``: the gap is also at least ``C / max_cost_share``, above
    ``max_interval_s`` when needed (the default interval); None: no such floor.
    ``first_cut_s``: before any measured cut, a cut is due once this many seconds
    passed since the first safe point (0: at the first safe point)."""

    def __init__(self, *, mtbf_s: float, max_interval_s: float = DEFAULT_MAX_INTERVAL_S,
                 min_interval_s: float = 0.0, max_cost_share: float | None = None,
                 first_cut_s: float = 0.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if not first_cut_s >= 0:
            raise ValueError("the first cut delay must be >= 0")
        if not mtbf_s > 0:
            raise ValueError("--rl-cut-auto-mtbf-s must be positive")
        if not max_interval_s > 0:
            raise ValueError("--rl-cut-max-interval-s must be positive")
        if not 0 <= min_interval_s <= max_interval_s:
            raise ValueError("the minimum cut interval must lie in [0, max interval]")
        if max_cost_share is not None and not 0 < max_cost_share <= 1:
            raise ValueError("the cut cost share must lie in (0, 1]")
        self.mtbf_s = float(mtbf_s)
        self.max_interval_s = float(max_interval_s)
        self.min_interval_s = float(min_interval_s)
        self.max_cost_share = None if max_cost_share is None else float(max_cost_share)
        self.first_cut_s = float(first_cut_s)
        self.clock = clock
        # [rollout id, seconds] of the last measured cuts (add_cost grows an entry)
        self._costs: deque[list[float]] = deque(maxlen=COST_WINDOW)
        self._started: float | None = None  # clock at the first safe point
        self._last_cut: tuple[int, float] | None = None  # (rollout id, clock) of the last cut
        self._decided: tuple[int, bool] | None = None  # the answer for the current safe point
        self.every = 1  # rounds between the last two cuts (publication checks follow it)

    @property
    def cost_s(self) -> float | None:
        return sum(cost for _, cost in self._costs) / len(self._costs) if self._costs else None

    def interval_s(self) -> float | None:
        """``T`` (None before the first measured cut)."""
        cost = self.cost_s
        if cost is None:
            return None
        cost = max(cost, 0.0)
        optimum = math.sqrt(2.0 * cost * self.mtbf_s)
        interval = min(max(optimum, self.min_interval_s), self.max_interval_s)
        if self.max_cost_share is not None:
            interval = max(interval, cost / self.max_cost_share)
        return interval

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
        now = self.clock()
        if self._started is None:
            self._started = now
        interval = self.interval_s()
        if interval is None or self._last_cut is None:
            due = now - self._started >= self.first_cut_s
        else:
            due = now - self._last_cut[1] >= interval
        self._decided = (rollout_id, due)
        return due

    def record_cut(self, rollout_id: int, *, cost_s: float) -> None:
        """A cut was taken at the safe point before ``rollout_id``; it cost the round
        ``cost_s`` seconds of critical path."""
        rollout_id = int(rollout_id)
        if self._last_cut is not None and rollout_id > self._last_cut[0]:
            self.every = rollout_id - self._last_cut[0]
        self._last_cut = (rollout_id, self.clock())
        self._costs.append([rollout_id, max(float(cost_s), 0.0)])

    def add_cost(self, rollout_id: int, *, cost_s: float) -> None:
        """The background part of the cut taken before ``rollout_id`` (shard
        collection, store sync and commit) took ``cost_s`` more seconds."""
        for entry in self._costs:
            if entry[0] == int(rollout_id):
                entry[1] += max(float(cost_s), 0.0)

    def describe(self) -> dict[str, object]:
        interval = self.interval_s()
        return {"mode": "auto", "mtbf_s": self.mtbf_s, "max_interval_s": self.max_interval_s,
                **({} if self.max_cost_share is None else {"max_cost_share": self.max_cost_share}),
                **({"first_cut_s": self.first_cut_s} if self.first_cut_s else {}),
                "cost_s": None if self.cost_s is None else round(self.cost_s, 3),
                "interval_s": None if interval is None else round(interval, 1)}


def cut_interval_for(*, every: int | None, auto_mtbf_s: float | None,
                     max_interval_s: float | None) -> FixedCutInterval | AutoCutInterval:
    """The interval of the cut flags (``--rl-cut-every`` / ``--rl-cut-auto-mtbf-s`` /
    ``--rl-cut-max-interval-s``); the two modes exclude each other. Neither:
    the default timed interval (module doc)."""
    if auto_mtbf_s is None:
        if max_interval_s is not None:
            raise ValueError("--rl-cut-max-interval-s needs --rl-cut-auto-mtbf-s")
        if every is None:
            return AutoCutInterval(mtbf_s=DEFAULT_MTBF_S, max_cost_share=DEFAULT_MAX_COST_SHARE,
                                   first_cut_s=DEFAULT_FIRST_CUT_S)
        return FixedCutInterval(every)
    if every is not None:
        raise ValueError("--rl-cut-every and --rl-cut-auto-mtbf-s exclude each other")
    return AutoCutInterval(mtbf_s=float(auto_mtbf_s),
                           max_interval_s=(DEFAULT_MAX_INTERVAL_S if max_interval_s is None
                                           else float(max_interval_s)))
