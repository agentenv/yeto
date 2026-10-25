"""Which island reads which rows of ``--data`` (S27 task assignment, P0/P1).

Plain rules (row ``i`` = the 0-based number of a row in ``--data``, in the
order the rows are read, before any split):

* A *data slot* ``I/N`` is the rows with ``i % N == I``. ``N`` islands with
  the slots ``0/N`` ... ``N-1/N`` read disjoint rows, and together they read
  every row. Each island reads the rows of its slot in data order, so in round
  ``r`` the ``N`` islands together read the same rows that ``N`` rounds of one
  island would read.
* Same-card training islands (sample role ``both``: they generate and train)
  of a launch with two or more of them: by default (``--rl-data-split auto``)
  they claim their rows from the dispenser (below) when the launch can use it;
  otherwise, and with ``--rl-data-split stride``, the island with learner id
  ``k`` reads the slot ``k/N``, where ``N`` is the number of those islands at
  launch (the launch log names why the dispenser does not apply). ``--rl-island-tasks`` replaces the slots
  with the user's task lists. ``--rl-data-split shared`` gives every island
  every row, in the same order (the behaviour before S27 G-AV); it is for
  controlled experiments only and the launch warns.
* Sampler islands keep their own slots: sampler ``k`` of ``S`` samplers reads
  the slot ``k/S`` (``sample_roles.island_args``). Trainers that read the
  sample buffer read buffer lanes (``merged_release.BufferLane``), not data.
* Passes: when an island has read every row of its slot, it starts pass 1,
  then pass 2, and so on, and continues. Pass 0 is the slot in data order. Pass
  ``p >= 1`` is the slot reshuffled with a seed made from the run seed and
  ``p`` (:func:`pass_order`), so a later pass does not repeat the order of the
  pass before it. The Miles adapter applies this order
  (``adapters.miles.data_passes``). verl keeps its own sampler, which already
  shuffles every pass with ``data.seed`` (pass 0 included).
* Elastic membership: the slots are fixed at launch; ``N`` never changes. A
  replacement, relaunched or re-joined island keeps its learner id and so its
  slot, and its data position counts rows of that slot only. Known limits
  until a central dispenser exists (P2): the rows of an island that left are
  read by no island, a fast island starts its next pass while a slow island is
  still in its first, and an island added after the launch gets no slot.
* Dispenser (``--rl-data-split dispense``, S27 P2, :mod:`.task_dispense`):
  the actor syncer hands out the rows. Every training island reads every row
  of ``--data`` but trains only the rows it claims, a block per round; a fast
  island claims more often, the rows of an island that leaves go to the
  others, a relaunched or added island just claims, and passes after the
  first are reshuffled (:func:`pass_order` over the whole data). The
  default (``auto``) since its GPU checks (S27 G-DH: 0.6B and Nano, two
  islands each); ``--rl-data-split stride`` keeps the slots.
* Resume and relaunch: every data cursor counts rows of the island's own slot,
  so a resumed or relaunched island with the same slot continues at the same
  row. A full-parameter island keeps its position in the run store
  (:mod:`.data_position`), because its round counter restarts at 0 in every
  process.

This module is backend-neutral: it knows row numbers, slots and the launch
shape only.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

AUTO, STRIDE, SHARED, DISPENSE = "auto", "stride", "shared", "dispense"
DATA_SPLITS = (AUTO, STRIDE, DISPENSE, SHARED)
# the mode of a launch whose islands do not split by slot
TASKS, SINGLE, ROLES = "tasks", "single", "roles"

# a row's number in the unsplit data, kept in the row's metadata after a split
DATA_ROW_KEY = "yeto_data_row"

DATA_SPLIT_FLAG = "--rl-data-split"


@dataclass(frozen=True)
class DataSlot:
    """The part of the data one island reads: rows ``i`` with ``i % count == index``.

    ``DataSlot(0, 1)`` is the whole data. A replacement incarnation of an island
    keeps its slot, so a data cursor in slot rows stays valid for it."""

    index: int = 0
    count: int = 1

    def __post_init__(self) -> None:
        if not (int(self.count) >= 1 and 0 <= int(self.index) < int(self.count)):
            raise ValueError(f"data slot {self.index}/{self.count}: need 0 <= index < count")

    @classmethod
    def parse(cls, text: str | None) -> "DataSlot":
        """``"index/count"`` (``None`` or empty: the whole data)."""
        if not text:
            return cls()
        index, sep, count = str(text).partition("/")
        if not sep:
            raise ValueError(f"data slot {text!r}: expected 'index/count'")
        return cls(int(index), int(count))

    def env(self) -> str:
        return f"{self.index}/{self.count}"

    @property
    def whole(self) -> bool:
        return self.count == 1

    def owns(self, row: int) -> bool:
        return int(row) % self.count == self.index

    def rows(self, rows: list) -> list:
        """The rows of this slot, in their original order."""
        out = list(rows[self.index::self.count])
        if not out:
            raise ValueError(f"data slot {self.env()} of {len(rows)} rows is empty")
        return out

    def numbered(self, rows: Iterable[Any]) -> list[tuple[int, Any]]:
        """``(row number in the unsplit data, row)`` of this slot, in data order."""
        out = [(i, row) for i, row in enumerate(rows) if self.owns(i)]
        if not out:
            raise ValueError(f"data slot {self.env()} has no rows (the data has fewer than "
                             f"{self.index + 1} rows)")
        return out


def pass_order(count: int, *, seed: int, data_pass: int) -> list[int]:
    """The row order (indexes into the slot's rows) of pass ``data_pass``.

    Pass 0 is data order. Pass ``p >= 1`` is a shuffle seeded by ``seed`` and
    ``p``; the seed goes through SHA-256, so every host and Python version gets
    the same order."""
    order = list(range(int(count)))
    if int(data_pass) <= 0:
        return order
    digest = hashlib.sha256(f"yeto-data-pass:{int(seed)}:{int(data_pass)}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    return order


@dataclass(frozen=True)
class DataSplitPlan:
    """Which rows each same-card training island of one launch reads.

    ``mode``: :data:`STRIDE` (each island its slot), :data:`SHARED` (every island
    every row), :data:`TASKS` (``--rl-island-tasks`` lists), :data:`SINGLE` (one
    island reads every row) or :data:`ROLES` (sampler islands read their own
    slots, buffer trainers read lanes: no same-card island). ``islands``: the
    learner ids of the same-card islands that read ``--data``."""

    mode: str
    requested: str
    islands: tuple[int, ...] = ()
    reason: str = ""

    @property
    def several_islands(self) -> bool:
        """Whether several islands read the one ``--data`` file (split or shared)."""
        return self.mode in (STRIDE, SHARED, DISPENSE)

    @property
    def dispense(self) -> bool:
        """The islands claim their rows from the actor syncer's dispenser."""
        return self.mode == DISPENSE

    def slot(self, learner_id: int) -> DataSlot | None:
        """The island's slot under :data:`STRIDE`, else None (no ``--rl-data-slot``)."""
        if self.mode != STRIDE or int(learner_id) not in self.islands:
            return None
        return DataSlot(self.islands.index(int(learner_id)), len(self.islands))

    def describe(self) -> str:
        """One plain sentence for the launch log and the plan output."""
        n = len(self.islands)
        if self.mode == STRIDE:
            return (f"{n} training islands split --data by row number: island k reads the "
                    f"rows i with i % {n} == k, in data order; after its last row an island "
                    "starts the next pass over its own rows, reshuffled"
                    + (f" ({self.reason})" if self.reason else ""))
        if self.mode == SHARED:
            return (f"{n} training islands read every row of --data in the same order "
                    f"({self.reason})")
        if self.mode == DISPENSE:
            return (f"{n} training island(s) claim the rows of --data from the syncer's "
                    "dispenser, one block per round: no row goes to two islands, a fast island "
                    "claims more, the rows of an island that leaves go to the others, passes "
                    "after the first are reshuffled")
        if self.mode == TASKS:
            return "each training island reads the rows of its --rl-island-tasks list"
        if self.mode == ROLES:
            return ("sampler islands read disjoint row slots (sampler k of S: rows i with "
                    "i % S == k); buffer trainers read buffer lanes, not --data")
        return "one training island reads every row of --data"

    def warning(self) -> str | None:
        """The loud launch warning when several islands read the same rows."""
        if self.mode != SHARED:
            return None
        remedy = (f"drop {DATA_SPLIT_FLAG} shared (default auto) to split the rows by island"
                  if self.requested == SHARED else
                  "an external learner reads the data of its own launch: give it other rows there")
        return (f"{len(self.islands)} training islands read the same --data rows in the same "
                f"order ({self.reason}): in every round each island trains the same prompts, "
                "so the merged update averages repeated samples of one batch instead of "
                f"covering more prompts; {remedy}")

    def record(self) -> dict[str, Any]:
        """What the run manifest and the dry-run plan carry for the launch."""
        out: dict[str, Any] = {"mode": self.mode, "requested": self.requested,
                               "islands": list(self.islands), "rule": self.describe()}
        if self.reason:
            out["reason"] = self.reason
        return out

    def island_record(self, learner_id: int) -> dict[str, Any]:
        """What the dry-run plan carries for one island."""
        slot = self.slot(learner_id)
        return {"mode": self.mode, "data_slot": None if slot is None else slot.env()}


def plan_data_split(requested: str | None, *, islands: Sequence[int], tasks: bool = False,
                    sample_roles: bool = False, external_learners: int = 0,
                    dispense_blockers: Sequence[str] | None = None) -> DataSplitPlan:
    """The data split of a launch (module doc). Raises ``ValueError`` on a request
    the launch shape cannot honour.

    ``islands``: learner ids of the same-card training islands the launch builds;
    ``tasks``: ``--rl-island-tasks`` is given; ``sample_roles``: the islands have
    sampler/trainer roles; ``external_learners``: islands of other launches that
    join the same syncer (their data is not this launch's to split);
    ``dispense_blockers``: why this launch cannot use the dispenser (empty: it
    can; None: not checked, so ``auto`` keeps the slots)."""
    requested = AUTO if requested is None else str(requested)
    if requested not in DATA_SPLITS:
        raise ValueError(f"{DATA_SPLIT_FLAG} {requested!r}: expected one of {DATA_SPLITS}")
    islands = tuple(int(i) for i in islands)
    if requested == DISPENSE:
        if sample_roles:
            raise ValueError(f"{DATA_SPLIT_FLAG} dispense covers training islands that generate and "
                             "train; sampler islands keep their row slots (P4)")
        if tasks:
            raise ValueError(f"{DATA_SPLIT_FLAG} dispense and --rl-island-tasks exclude each other")
        if external_learners:
            raise ValueError(f"{DATA_SPLIT_FLAG} dispense does not cover --external-learners")
        if not islands:
            raise ValueError(f"{DATA_SPLIT_FLAG} dispense needs at least one training island")
        return DataSplitPlan(DISPENSE, requested, islands)
    if sample_roles:
        if requested != AUTO:
            raise ValueError(f"{DATA_SPLIT_FLAG} {requested}: sampler islands always read "
                             "their own row slots and buffer trainers read lanes; drop the flag")
        return DataSplitPlan(ROLES, requested)
    if tasks:
        if requested != AUTO:
            raise ValueError(f"{DATA_SPLIT_FLAG} {requested} and --rl-island-tasks exclude each "
                             "other: the task lists already say which island reads which rows")
        return DataSplitPlan(TASKS, requested, islands)
    if len(islands) < 2:
        return DataSplitPlan(SINGLE, requested, islands)
    if requested == SHARED:
        return DataSplitPlan(SHARED, requested, islands, reason=f"{DATA_SPLIT_FLAG} shared")
    if external_learners:
        if requested == STRIDE:
            raise ValueError(f"{DATA_SPLIT_FLAG} stride does not cover --external-learners "
                             "(they read their own data)")
        return DataSplitPlan(SHARED, requested, islands,
                             reason="--external-learners: their rows are not this launch's to split")
    if requested == AUTO:
        if dispense_blockers is not None and not dispense_blockers:
            return DataSplitPlan(DISPENSE, requested, islands)
        reason = ("; ".join(dispense_blockers) if dispense_blockers
                  else "the launch was not checked for the dispenser")
        return DataSplitPlan(STRIDE, requested, islands, reason=f"no dispenser: {reason}")
    return DataSplitPlan(STRIDE, requested, islands)


def overlapping_tasks(task_lists: Mapping[int, Sequence[str]]) -> dict[tuple[int, int], int]:
    """``(island a, island b) -> number of task ids both lists name``, for a < b,
    only for pairs that share at least one task."""
    sets = {island: set(tasks) for island, tasks in task_lists.items()}
    keys = sorted(sets)
    out = {}
    for n, a in enumerate(keys):
        for b in keys[n + 1:]:
            common = len(sets[a] & sets[b])
            if common:
                out[(a, b)] = common
    return out


def row_runs(rows: Iterable[int]) -> list[list[int]]:
    """The distinct ``rows`` as arithmetic runs ``[first, last, step]``, ascending.

    Used by ``rl_round_data_rows`` so that one round's record stays small. A
    round of pass 0 of a slot (rows ``k, k+N, k+2N, ...`` in data order) is one
    run. A reshuffled pass or filtered groups give more runs, at most one per
    two rows. A run of one row is ``[row, row, 1]``."""
    ordered = sorted(set(rows))
    runs: list[list[int]] = []
    index = 0
    while index < len(ordered):
        first = ordered[index]
        if index + 1 == len(ordered):
            runs.append([first, first, 1])
            break
        step = ordered[index + 1] - first
        last = index + 1
        while last + 1 < len(ordered) and ordered[last + 1] - ordered[last] == step:
            last += 1
        runs.append([first, ordered[last], step])
        index = last + 1
    return runs


def _run_length(run: Sequence[int]) -> int:
    first, last, step = run
    if step < 1 or last < first or (last - first) % step:
        raise ValueError(f"malformed row run {list(run)}")
    return (last - first) // step + 1


def run_row_count(runs: Iterable[Sequence[int]]) -> int:
    """The number of rows :func:`row_runs` encoded (no expansion)."""
    return sum(_run_length(run) for run in runs)


def rows_of_runs(runs: Iterable[Sequence[int]]) -> list[int]:
    """The rows :func:`row_runs` encoded, ascending."""
    rows: list[int] = []
    for run in runs:
        _run_length(run)
        first, last, step = run
        rows.extend(range(first, last + 1, step))
    return rows
