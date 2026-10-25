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
  wait for that round's commit. Miles puts the aborted groups of a partial
  rollout back into its buffer only after the hook
  (``sglang_rollout.generate_rollout``), so the other positions taken so far
  are reported dropped (filtered or aborted for good) at the next draw or
  hook, unless they are in the buffer by then.
* :func:`commit_round` runs when the island driver passed the round's sync
  boundary (``IslandDriver._commit_round_data``, through ``__ray_call__``):
  the kept positions of that round (and of earlier rounds) are reported
  trained, and the call waits for the dispenser's answer (bounded). So a
  position is reported trained only after its round trained and passed the
  sync boundary. An island process that dies before that point leaves the
  round's positions unreported; the dispenser gives them out again (first in
  line) when the relaunched process opens (``superseded``) or when the lease
  of the dead process ends. Semantics: a row may be generated more than once,
  and trained more than once only when the process dies between the sync
  boundary and the dispenser's answer to the commit (a short window).
* The end of the data (``--rl-data-passes``): a draw the dispenser cannot fill
  any more (WAIT: nothing fresh is left, other holders may still give rows
  back) while this island holds rows of the rollout it is drawing gives up:
  those rows and the prefetched ones are released (other islands may still
  fill a batch with them) and :class:`DispenseEnded` is raised. An island that
  holds no rows of the rollout waits instead, until rows come back or the
  dispenser answers END (also :class:`DispenseEnded`). The island driver ends
  the run FINISHED on it (``IslandDriver._ends_on_data_end``). So the rows of
  the last, incomplete batches of a limited run are not trained.
* :func:`close_source` (the driver, at the end of a finished run): every
  position not trained yet goes back to the dispenser (release).
* Miles' own cursor (``sample_offset``, ``epoch_id``) does not move; the
  dispenser keeps the data position, so a relaunched island needs none.
* Sampler islands (S27 P4, ``sampler_island.run_sampler_island``): the same
  wrapped data source; a position is reported trained when the buffer writer
  has written its group into the sample buffer (:func:`written`), so a
  sampler that dies before the write leaves its rows unreported and the
  dispenser gives them to the next claim (``superseded`` when the same
  sampler id comes back, else at the end of the lease). Every other row the
  round took is dropped unless Miles put its group back into its buffer.

Runs inside the rollout executor actor (``__ray_call__``), installed once
before the first draw by ``MilesRolloutPool``. Import-light.
"""

from __future__ import annotations

import atexit
import copy
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from yeto.rl.engine.data_split import DATA_ROW_KEY
from yeto.rl.engine.task_dispense import (
    DISPENSE_POSITION_KEY,
    DispenseEnded,
    PositionMap,
    StreamTable,
    TaskSource,
    tcp_connector,
)

logger = logging.getLogger(__name__)

_ATTR = "_yeto_task_dispense"
_CURRENT: "DispensedSource | None" = None  # the one dispensed source of this process

# how long a round's commit waits for the dispenser's answer; on timeout the
# report stays queued (sent when the head answers again) and the round goes on
COMMIT_ACK_TIMEOUT_S = 30.0


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

    def __init__(self, data_source: Any, tasks: TaskSource, positions: PositionMap,
                 executor: Any = None) -> None:
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
        # rollout id -> kept positions waiting for that round's commit (module doc)
        self._pending: dict[int, set[int]] = {}
        self._hooks = 0
        self._executor = executor
        self.gave_up: str | None = None
        # the generation thread draws and runs the hook; the driver's commit and
        # close arrive on the actor's own thread (``__ray_call__``)
        self._lock = threading.RLock()
        data_source.get_samples = self.get_samples

    def get_samples(self, num_samples: int) -> list[list[Any]]:
        with self._lock:
            self._settle()
        wanted = int(num_samples)
        groups: list[list[Any]] = []
        if callable(self._buffer_draw):
            groups.extend(self._buffer_draw(wanted))
        args = self.source.args
        while len(groups) < wanted:
            need = wanted - len(groups)
            holding = self._holds_rows()
            # holding rows of this rollout: a draw the dispenser cannot fill returns short
            taken = self.tasks.take(need, drained_ok=holding)
            if not taken:
                self._give_up(need)
            for position in taken:
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

    def pending_positions(self) -> set[int]:
        """Kept positions waiting for their round's commit."""
        with self._lock:
            return set().union(*self._pending.values()) if self._pending else set()

    def _holds_rows(self) -> bool:
        """Whether this island holds rows besides those waiting for a commit: rows
        of the rollout being drawn, or groups back in Miles' buffer."""
        return bool(self.tasks.taken() - self.pending_positions())

    def _give_up(self, need: int) -> None:
        """The data ended for this rollout (module doc): release what it holds."""
        with self._lock:
            pending = self.pending_positions()
            prefetched = self.tasks.release_ready(stop_claims=True)
            held = sorted(self.tasks.taken() - pending)
            self.tasks.report(release=held)
            self._candidates = set()
            self.gave_up = (f"the dispenser has no rows left to fill this rollout ({need} more "
                            f"wanted): {len(held)} drawn and {prefetched} prefetched rows released")
        logger.info("task dispenser: %s", self.gave_up)
        raise DispenseEnded(self.gave_up)

    def buffered_positions(self) -> set[int]:
        out = set()
        for group in getattr(self.source, "buffer", None) or ():
            position = _position(group)
            if position is not None:
                out.add(position)
        return out

    def _settle(self) -> int:
        """Report the previous rollout's taken, unkept positions dropped unless
        their group is back in Miles' buffer; returns how many. Under the lock."""
        if not self._candidates:
            return 0
        dropped = ((self._candidates & self.tasks.taken()) - self.buffered_positions()
                   - self.pending_positions())
        self._candidates = set()
        if dropped:
            self.tasks.report(dropped=sorted(dropped))
        return len(dropped)

    def _hook_round(self) -> int:
        """The rollout id of the batch the hook sees: the executor's (Miles
        ``RolloutExecutor.get`` sets ``rollout_id`` before it generates), else
        the number of hooks this process ran (rounds count from 0 in every
        process of a full-parameter island)."""
        value = getattr(self._executor, "rollout_id", None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return self._hooks

    def on_trained(self, kept: Iterable[Sequence[Any]], rollout_id: int | None = None) -> dict[str, int]:
        """The kept groups wait for their round's commit; the other taken
        positions become drop candidates (module doc). Returns the counts."""
        with self._lock:
            dropped = self._settle()
            round_id = self._hook_round() if rollout_id is None else int(rollout_id)
            self._hooks += 1
            pending = self.pending_positions()
            taken = self.tasks.taken()
            kept_now = {p for p in (_position(g) for g in kept) if p is not None} & taken
            kept_now -= pending
            if kept_now:
                self._pending.setdefault(round_id, set()).update(kept_now)
            self._candidates = taken - kept_now - pending
            return {"kept": len(kept_now), "dropped": dropped, "rollout_id": round_id}

    def commit(self, rollout_id: int, *, timeout_s: float = COMMIT_ACK_TIMEOUT_S) -> dict[str, Any]:
        """Report the kept positions of rounds <= ``rollout_id`` trained and wait
        (``timeout_s``) for the dispenser's answer (module doc)."""
        with self._lock:
            due = sorted(r for r in self._pending if r <= int(rollout_id))
            positions: set[int] = set()
            for round_id in due:
                positions |= self._pending.pop(round_id)
            taken = self.tasks.taken()
            trained = sorted(positions & taken)
            self.tasks.report(trained=trained)
        acked = self.tasks.flush(timeout_s) if trained else True
        return {"rounds": due, "trained": len(trained), "acked": bool(acked),
                # fenced while they waited: the dispenser gave them out again
                "lost": len(positions) - len(trained)}

    def written(self, rollout_id: int, positions: Iterable[int], *,
                timeout_s: float = COMMIT_ACK_TIMEOUT_S) -> dict[str, Any]:
        """Sampler island (S27 P4): the groups of ``positions`` are in the sample
        buffer, so their rows are done: report them trained (the sampler's
        "trained" is "written for the trainers") and wait (``timeout_s``) for
        the dispenser's answer. Every other position of the round that is still
        taken becomes a drop candidate (dropped at the next draw unless Miles put
        its group back into its buffer, as for a training island)."""
        with self._lock:
            done = {int(p) for p in positions} & self.tasks.taken()
            self._pending = {r: kept - done for r, kept in self._pending.items()
                             if r > int(rollout_id) and kept - done}
            self.tasks.report(trained=sorted(done))
            self._candidates = self.tasks.taken() - done - self.pending_positions()
        acked = self.tasks.flush(timeout_s) if done else True
        return {"rollout_id": int(rollout_id), "trained": len(done), "acked": bool(acked)}

    def close(self) -> dict[str, Any]:
        """Release every position not trained yet and stop (module doc)."""
        with self._lock:
            pending = self.pending_positions()
            self._pending.clear()
            self._candidates = set()
            held = len(self.tasks.taken()) + self.tasks.ready_count()
        self.tasks.close(release=True)
        return {"released": held, "uncommitted": len(pending),
                **{k: self.tasks.stats[k] for k in ("granted", "trained", "dropped", "release",
                                                     "fenced", "lost_reports")}}


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
    # S27 P3: the streams of --data-mix / task labels (None: the one P2 stream)
    table = StreamTable.from_config(config["streams"]) if config.get("streams") else None
    accept = config.get("accept")
    connect = tcp_connector(str(config["address"]), learner_id=int(config["learner_id"]), rows=rows,
                            fingerprint=bytes.fromhex(str(config["fingerprint"])), key=key,
                            streams=None if table is None else table.decl(),
                            accept=None if accept is None else [int(a) for a in accept])
    tasks = TaskSource(connect=connect, block=int(config["block"]),
                       prefetch_blocks=int(config.get("prefetch_blocks", 2)),
                       emit=lambda kind, **fields: logger.info("%s %s", kind, fields))
    tasks.start()
    try:
        positions = (PositionMap(rows, seed=int(config.get("seed", 0))) if table is None
                     else PositionMap(seed=int(config.get("seed", 0)), table=table))
        wrapped = DispensedSource(source, tasks, positions, executor=executor)
    except BaseException:
        tasks.close(release=True)
        raise
    setattr(source, _ATTR, wrapped)
    _CURRENT = wrapped
    atexit.register(tasks.close)
    logger.info("task dispenser: island %s takes its rows from %s (incarnation %016x, %d rows, "
                "%d filtered at load, %s)", config["learner_id"], config["address"], tasks.incarnation,
                rows, wrapped.filtered_rows,
                "one stream" if table is None else
                f"{len(table.streams)} streams, accepts {'all' if accept is None else accept}")
    return {"installed": True, "incarnation": f"{tasks.incarnation:016x}",
            "filtered_rows": wrapped.filtered_rows}


def current() -> DispensedSource | None:
    return _CURRENT


def on_trained(kept: Iterable[Sequence[Any]]) -> dict[str, int] | None:
    """The sample filter hook's call (``rollout_meta_hook``); None without a dispenser."""
    source = _CURRENT
    return None if source is None else source.on_trained(list(kept))


def commit_round(executor: Any, rollout_id: int) -> dict[str, Any] | None:
    """Runs INSIDE the rollout executor actor (``MilesRolloutPool.commit_dispensed``):
    the round passed its sync boundary (module doc). None without a dispenser."""
    source = _CURRENT
    if source is None:
        return None
    try:
        return source.commit(int(rollout_id))
    except Exception as error:  # noqa: BLE001 - bookkeeping must not fail the round
        logger.warning("task dispenser: commit of round %s failed: %r", rollout_id, error)
        return {"acked": False, "error": f"{type(error).__name__}: {error}"}


def written(rollout_id: int, positions: Sequence[int]) -> dict[str, Any] | None:
    """The sampler's buffer writer (``BufferWriterRolloutFn._write``, in the rollout
    executor): the groups of ``positions`` are in the buffer (S27 P4). None without
    a dispenser; never raises (the buffer write already happened)."""
    source = _CURRENT
    if source is None:
        return None
    try:
        return source.written(int(rollout_id), positions)
    except Exception as error:  # noqa: BLE001 - bookkeeping must not fail the round
        logger.warning("task dispenser: report of round %s's written groups failed: %r", rollout_id, error)
        return {"acked": False, "error": f"{type(error).__name__}: {error}"}


def close_source(executor: Any) -> dict[str, Any] | None:
    """Runs INSIDE the rollout executor actor (``MilesRolloutPool.close_dispensed``)."""
    global _CURRENT
    source = _CURRENT
    if source is None:
        return None
    _CURRENT = None
    return source.close()
