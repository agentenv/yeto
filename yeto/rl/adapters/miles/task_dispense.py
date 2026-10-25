"""Miles: an island takes its prompt rows from the task dispenser (S27 P2).

With ``--rl-data-split dispense`` the island's ``prompts.jsonl`` holds every
row of ``--data`` (each with ``metadata["yeto_data_row"]``), and the rollout
executor's data source draws rows by dispenser position instead of by its
own cursor (``yeto.rl.engine.task_dispense`` module doc):

* ``get_samples(n)``: groups Miles put back in its buffer (partial rollouts)
  come first, as Miles does; the rest are ``n`` positions taken from the
  island's :class:`~yeto.rl.engine.task_dispense.TaskSource`, each mapped to
  its row (``PositionMap``) and built into a group exactly like Miles'
  ``RolloutDataSource.get_samples`` (deep copies, group and sample indexes).
  Each sample carries its position in ``metadata["yeto_dispense_position"]``.
  A row Miles filtered out at load (e.g. longer than the prompt limit) is
  reported dropped and another position is taken.
* :func:`on_trained` runs in the sample filter hook at the end of every
  training rollout (``rollout_meta_hook``): the positions of the kept groups
  are reported trained. Miles puts the aborted groups of a partial rollout
  back into its buffer only after the hook (``sglang_rollout.generate_rollout``),
  so the other positions taken so far are reported dropped (filtered or
  aborted for good) at the next draw or hook, unless they are in the buffer
  by then. So a position is reported when its group enters a training batch;
  the rows of a round the island never finishes are lost to the run (at most
  once, never twice).
* Miles' own cursor (``sample_offset``, ``epoch_id``) does not move; the
  dispenser keeps the data position, so a relaunched island needs none.

Runs inside the rollout executor actor (``__ray_call__``), installed once
before the first draw by ``MilesRolloutPool``. Import-light.
"""

from __future__ import annotations

import atexit
import copy
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from yeto.rl.engine.data_split import DATA_ROW_KEY
from yeto.rl.engine.task_dispense import (
    DISPENSE_POSITION_KEY,
    PositionMap,
    TaskSource,
    tcp_connector,
)

logger = logging.getLogger(__name__)

_ATTR = "_yeto_task_dispense"
_CURRENT: "DispensedSource | None" = None  # the one dispensed source of this process


class DispenseInstallError(RuntimeError):
    """The data source cannot take its rows from the dispenser."""


def _metadata(sample: Any) -> dict[str, Any]:
    meta = getattr(sample, "metadata", None)
    if meta is None:
        meta = {}
        sample.metadata = meta
    return meta


def _position(group: Sequence[Any]) -> int | None:
    first = group[0]
    while isinstance(first, list):
        first = first[0]
    value = (getattr(first, "metadata", None) or {}).get(DISPENSE_POSITION_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class DispensedSource:
    """Wraps one Miles data source (module doc)."""

    def __init__(self, data_source: Any, tasks: TaskSource, positions: PositionMap) -> None:
        dataset = getattr(data_source, "dataset", None)
        if dataset is None:
            raise DispenseInstallError("the rollout data source has no dataset "
                                       "(--disable-rollout-global-dataset?)")
        self.source = data_source
        self.tasks = tasks
        self.positions = positions
        self.by_row: dict[int, Any] = {}
        for sample in getattr(dataset, "origin_samples", None) or dataset.samples:
            row = (getattr(sample, "metadata", None) or {}).get(DATA_ROW_KEY)
            if isinstance(row, int):
                self.by_row[row] = sample
        if not self.by_row:
            raise DispenseInstallError(f"no prompt row carries metadata[{DATA_ROW_KEY!r}]")
        if max(self.by_row) >= positions.rows:
            raise DispenseInstallError(f"row {max(self.by_row)} is outside the {positions.rows} "
                                       "rows the dispenser was opened with")
        self.filtered_rows = positions.rows - len(self.by_row)
        self._buffer_draw = getattr(data_source, "_get_samples_from_buffer", None)
        self.drawn = 0
        # positions the last rollout took and did not keep: dropped at the next
        # draw or hook unless Miles put their group back into its buffer
        self._candidates: set[int] = set()
        data_source.get_samples = self.get_samples

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        self._settle()
        wanted = int(num_samples)
        groups: list[list[Any]] = []
        if callable(self._buffer_draw):
            groups.extend(self._buffer_draw(wanted))
        args = self.source.args
        while len(groups) < wanted:
            for position in self.tasks.take(wanted - len(groups)):
                prompt = self.by_row.get(self.positions.row(position))
                if prompt is None:  # Miles filtered this row at load
                    self.tasks.report(dropped=[position])
                    continue
                group = []
                for _ in range(int(args.n_samples_per_prompt)):
                    sample = copy.deepcopy(prompt)
                    sample.group_index = self.source.sample_group_index
                    sample.index = self.source.sample_index
                    self.source.sample_index += 1
                    _metadata(sample)[DISPENSE_POSITION_KEY] = int(position)
                    group.append(sample)
                self.source.sample_group_index += 1
                groups.append(group)
                self.drawn += 1
        return groups

    def buffered_positions(self) -> set[int]:
        out = set()
        for group in getattr(self.source, "buffer", None) or ():
            position = _position(group)
            if position is not None:
                out.add(position)
        return out

    def _settle(self) -> int:
        """Report the previous rollout's taken, unkept positions dropped unless
        their group is back in Miles' buffer; returns how many."""
        if not self._candidates:
            return 0
        dropped = (self._candidates & self.tasks.taken()) - self.buffered_positions()
        self._candidates = set()
        if dropped:
            self.tasks.report(dropped=sorted(dropped))
        return len(dropped)

    def on_trained(self, kept: Iterable[Sequence[Any]]) -> dict[str, int]:
        """Report the kept groups trained; the other taken positions become
        drop candidates (module doc). Returns the counts reported now."""
        dropped = self._settle()
        trained = {p for p in (_position(g) for g in kept) if p is not None}
        taken = self.tasks.taken()
        self.tasks.report(trained=sorted(trained & taken))
        self._candidates = taken - trained
        return {"trained": len(trained & taken), "dropped": dropped}


def install(executor: Any, config: Mapping[str, Any], key: bytes | None) -> dict[str, Any]:
    """Runs INSIDE the rollout executor actor: wrap its data source once and open
    the island's connection to the dispenser. ``config`` is
    ``miles_args.yeto_rl_data_dispense`` (island entry)."""
    global _CURRENT
    source = getattr(executor, "data_source", None)
    source = getattr(source, "__self__", source)
    if source is None:
        raise DispenseInstallError("the rollout executor has no data source")
    current = getattr(source, _ATTR, None)
    if current is not None:
        _CURRENT = current
        return {"installed": False, "incarnation": f"{current.tasks.incarnation:016x}"}
    if getattr(getattr(source, "args", None), "fully_async", False):
        raise DispenseInstallError("the dispenser does not cover fully_async rollouts yet")
    rows = int(config["rows"])
    connect = tcp_connector(str(config["address"]), learner_id=int(config["learner_id"]), rows=rows,
                            fingerprint=bytes.fromhex(str(config["fingerprint"])), key=key)
    tasks = TaskSource(connect=connect, block=int(config["block"]),
                       prefetch_blocks=int(config.get("prefetch_blocks", 2)),
                       emit=lambda kind, **fields: logger.info("%s %s", kind, fields))
    tasks.start()
    try:
        wrapped = DispensedSource(source, tasks, PositionMap(rows, seed=int(config.get("seed", 0))))
    except BaseException:
        tasks.close(release=True)
        raise
    setattr(source, _ATTR, wrapped)
    _CURRENT = wrapped
    atexit.register(tasks.close)
    logger.info("task dispenser: island %s takes its rows from %s (incarnation %016x, %d rows, "
                "%d filtered at load)", config["learner_id"], config["address"], tasks.incarnation,
                rows, wrapped.filtered_rows)
    return {"installed": True, "incarnation": f"{tasks.incarnation:016x}",
            "filtered_rows": wrapped.filtered_rows}


def current() -> DispensedSource | None:
    return _CURRENT


def on_trained(kept: Iterable[Sequence[Any]]) -> dict[str, int] | None:
    """The sample filter hook's call (``rollout_meta_hook``); None without a dispenser."""
    source = _CURRENT
    return None if source is None else source.on_trained(list(kept))
