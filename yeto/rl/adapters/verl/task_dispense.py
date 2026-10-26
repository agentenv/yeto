"""verl: a training island takes its prompt rows from the task dispenser
(S27 P2 on verl, VERL-PARITY #31).

With ``--rl-data-split dispense`` the launcher gives every verl training island
``--rl-data-dispense`` instead of a row slot, and the actor syncer runs the
dispenser (``syncer/src/dispense.rs``). The rules are the Miles port's
(``adapters.miles.task_dispense`` module doc); the bookkeeping is the
backend-neutral :class:`~yeto.rl.engine.task_dispense.DispensedRows`. What is
verl-specific:

* Rows: the island's train parquet keeps every row of ``--data``, each with its
  row number as ``extra_info["yeto_data_row"]`` (``island_entry.prepare_data``
  with ``number_rows``). The row count and the fingerprint the dispenser checks
  come from ``train-dispense.jsonl`` (:func:`write_rows_file`): one canonical
  JSON line per row (messages, label, task label, ``--data-mix`` source), so
  every island of a run that reads the same data sends the same fingerprint.
* Draws: verl's sync trainer draws its prompts only through
  ``trainer._fetch_one_gen_batch`` (one ``gen_batch_size`` chunk per call;
  the data cursor ``data_cursor.VerlDataCursor`` wraps the same method).
  :class:`VerlDispensedBatches` replaces it: it takes positions, maps each to
  its row and to that row's index in verl's ``train_dataset`` (a row verl
  filtered out at load, e.g. a prompt over ``max_prompt_length``, is reported
  dropped and another position is taken), and builds the chunk as verl does
  (``collate_fn``, one new ``uid`` per prompt). It remembers uid -> position.
* Kept: ``VerlRolloutPool.generate`` reads the uids of the batch it hands to the
  trainer; their positions wait for the round's commit (:meth:`kept_round`).
  The other positions the round took (none on the sync path, which trains
  every prompt it drew) are reported dropped at the next draw.
* Commit and close: the island driver calls ``commit_dispensed`` after the
  round's sync boundary and ``close_dispensed`` at the end of a finished run
  (``IslandDriver._commit_round_data`` / ``_close_rollout_data``), as on Miles.
* No data cursor: the dispenser keeps the data position, so a relaunched island
  needs none; the pool offers no seek (``VerlRolloutPool``).
* The end of the data (``--rl-data-passes``): :class:`DispenseEnded` is raised
  inside ``_fetch_one_gen_batch``, before verl submits the round's prompts, and
  the driver ends the run FINISHED on it.

Not covered (the island refuses ``--rl-data-dispense`` there): the Megatron
island (one island, no syncer), the fully_async path and the buffer trainer
(their rows come from verl's rollouter or the sample buffer), ``--rl-data-slot``.
A verl *sampler* island claims through ``DispensedPrompts`` (S27 P4,
``sampler_entry.dispense_spec``), not through this module.

Runs inside verl's task-runner Ray actor (``verl_main.YetoTaskRunner``), next
to the trainer object; imports verl and numpy lazily (CPU-testable with a fake
trainer).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from yeto.rl.engine.data_split import DATA_ROW_KEY
from yeto.rl.engine.task_dispense import (DATA_SOURCE_KEY, TASK_LABEL_KEY, DispensedRows,
                                          PositionMap, StreamTable, TaskSource, file_fingerprint,
                                          open_task_source)

logger = logging.getLogger(__name__)

ROWS_FILE = "train-dispense.jsonl"
PLAN_KEY = "dispense"


class DispenseIslandError(ValueError):
    """The island's flags do not allow it to take rows from the dispenser."""


# ----------------------------------------------------------------- island entry


def island_dispense(args: Any, *, engine: str = "fsdp2", buffer_trainer: bool = False) -> bool:
    """Whether this verl island claims its rows from the dispenser
    (``--rl-data-dispense``); raises :class:`DispenseIslandError` where it does
    not apply (module doc). The fully_async path is ``--rl-max-policy-age > 0``
    without ``--rl-rollout-ahead`` (the launcher's rule; with rollout ahead the
    synchronous trainer generates one batch ahead and draws through the same
    ``_fetch_one_gen_batch``). A sampler island never reaches here (it runs the
    engine sampler, ``sampler_entry``)."""
    fully_async = (int(getattr(args, "rl_max_policy_age", 0) or 0) >= 1
                   and not int(getattr(args, "rl_rollout_ahead", 0) or 0))
    if not getattr(args, "rl_data_dispense", False):
        return False
    problems = []
    if getattr(args, "rl_data_slot", None):
        problems.append("--rl-data-slot (a fixed row slot)")
    if getattr(args, "rl_data_position_store", None):
        problems.append("--rl-data-position-store (the dispenser keeps the data position)")
    if getattr(args, "rl_single_island_no_sync", False) or not getattr(args, "syncer", None):
        problems.append("no syncer (--rl-single-island-no-sync): the dispenser runs in the actor syncer")
    if engine not in ("fsdp2", "megatron"):  # S27 G-FG: a synced Megatron island draws through the same loader
        problems.append(f"the {engine} island")
    if fully_async:
        problems.append("the fully_async path (--rl-max-policy-age > 0: verl's rollouter draws the rows)")
    if buffer_trainer:
        problems.append("a buffer trainer (--rl-sample-role trainer: the sampler islands claim the rows)")
    if problems:
        raise DispenseIslandError("--rl-data-dispense on a verl training island does not cover: "
                                  + "; ".join(problems))
    return True


def _messages(value: Any) -> list[dict[str, str]]:
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in value]


def rows_file_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """One canonical line of :data:`ROWS_FILE` for a ``--data`` row (``messages``,
    ``label``; the task label and the ``--data-mix`` source when the row has them,
    so the streams of S27 P3 read the same file)."""
    record: dict[str, Any] = {"messages": _messages(row["messages"]), "label": str(row["label"])}
    raw_meta = row.get("metadata")
    metadata = raw_meta if isinstance(raw_meta, Mapping) else {}
    label = row.get(TASK_LABEL_KEY)
    if label is None:
        label = metadata.get(TASK_LABEL_KEY)
    if label is not None and not (isinstance(label, float) and label != label):  # pandas NaN: no label
        record[TASK_LABEL_KEY] = label if isinstance(label, str) else list(label)
    if metadata.get(DATA_SOURCE_KEY) is not None:
        record["metadata"] = {DATA_SOURCE_KEY: int(metadata[DATA_SOURCE_KEY])}
    return record


def write_rows_file(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    """Write :data:`ROWS_FILE` (module doc); returns ``(rows, sha256 hex)``."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(rows_file_record(row), ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    os.replace(temporary, path)
    count, digest = file_fingerprint(path)
    return count, digest.hex()


def dispense_section(args: Any, rows_file: str | os.PathLike[str], *, block: int) -> dict[str, Any]:
    """The plan's ``dispense`` section (what the task runner needs to open the
    island's connection, :func:`install`). The actor syncer (shard 0) dispenses;
    ``block`` is the island's round (``--groups-per-round`` prompts)."""
    from yeto.syncer_shards import parse_syncer_addrs

    host, port = parse_syncer_addrs(str(args.syncer))[0]
    count, digest = file_fingerprint(rows_file)
    mix = getattr(args, "rl_dispense_mix", None)
    labels = getattr(args, "rl_dispense_labels", None)
    by_label = bool(getattr(args, "rl_dispense_by_label", False)) or bool(labels)
    section: dict[str, Any] = {
        "address": f"{host}:{port}", "learner_id": int(args.learner_id), "rows": count,
        "fingerprint": digest.hex(), "seed": int(args.seed), "block": int(block),
        "prefetch_blocks": int(getattr(args, "rl_dispense_prefetch_blocks", 2) or 2),
        "rows_file": str(rows_file),
        "mix": mix.split(",") if mix else None, "by_label": by_label,
        "labels": labels.split(",") if labels else None,
    }
    if mix or by_label:  # S27 P3: the island's streams, checked here before any GPU work
        try:
            table = StreamTable.from_prompts(rows_file, mix=section["mix"], by_label=by_label)
            accept = table.accept(section["labels"]) if section["labels"] else None
        except ValueError as error:
            raise DispenseIslandError(str(error)) from None
        if not (table.single and accept is None):
            section["data_streams"] = {"streams": table.record(), "accept": accept}
    return section


# ---------------------------------------------------------------- task runner


def _default_collate(items: list[dict]) -> dict:
    from verl.utils.dataset.rl_dataset import collate_fn

    return collate_fn(items)


def _default_finish(batch_dict: dict, uids: list[str]) -> Any:
    """verl's own end of ``_fetch_one_gen_batch``: the uid array and the TensorDict."""
    import numpy as np
    from verl.utils import tensordict_utils as tu

    batch_dict["uid"] = np.array(uids, dtype=object)
    return tu.get_tensordict(batch_dict)


def row_index(dataset: Any) -> dict[int, int]:
    """``--data`` row number -> index in verl's (filtered) train dataset, read from
    every row's ``extra_info["yeto_data_row"]``."""
    frame = getattr(dataset, "dataframe", None)
    if frame is None:
        raise DispenseIslandError("verl's train dataset has no dataframe")
    out: dict[int, int] = {}
    for index, info in enumerate(frame["extra_info"]):
        row = (info or {}).get(DATA_ROW_KEY) if isinstance(info, Mapping) else None
        if row is None:
            raise DispenseIslandError(f"train row {index} carries no extra_info[{DATA_ROW_KEY!r}] "
                                      "(the island wrote its parquet without row numbers)")
        out[int(row)] = index
    return out


class VerlDispensedBatches(DispensedRows):
    """The prompts of one verl training island from the task dispenser (module doc)."""

    def __init__(self, trainer: Any, tasks: TaskSource, positions: PositionMap, *,
                 collate: Callable[[list[dict]], dict] | None = None,
                 finish: Callable[[dict, list[str]], Any] | None = None) -> None:
        super().__init__(tasks, positions)
        self.trainer = trainer
        self.dataset = trainer.train_dataset
        self.by_row = row_index(self.dataset)
        if not self.by_row:
            raise DispenseIslandError("verl filtered out every train row")
        if max(self.by_row) >= positions.rows:
            raise DispenseIslandError(f"row {max(self.by_row)} is outside the {positions.rows} rows the "
                                      "dispenser was opened with")
        self.filtered_rows = positions.rows - len(self.by_row)
        self._collate = collate or _default_collate
        self._finish = finish or _default_finish
        self._uid_position: dict[str, int] = {}
        self.drawn = 0
        original = getattr(trainer, "_fetch_one_gen_batch", None)
        if getattr(original, "_yeto_cursor", None) is not None or getattr(original, "_yeto_dispense", None):
            raise DispenseIslandError("verl trainer: a data cursor or a dispenser is installed already")

        def fetch() -> Any:
            return self.fetch()

        fetch._yeto_dispense = self  # type: ignore[attr-defined]
        trainer._fetch_one_gen_batch = fetch

    def _chunk(self) -> int:
        data = self.trainer.config.data
        get = data.get if hasattr(data, "get") else (lambda k, d=None: getattr(data, k, d))
        return int(get("gen_batch_size", None) or get("train_batch_size"))

    def fetch(self) -> Any:
        """verl's ``_fetch_one_gen_batch`` (module doc): one chunk of dispensed rows."""
        with self._lock:
            self._settle()
        need = self._chunk()
        items: list[dict] = []
        taken: list[int] = []
        while len(items) < need:
            for position in self.take_rows(need - len(items)):
                index = self.by_row.get(self.positions.row(position))
                if index is None:  # verl filtered this row at load
                    self.tasks.report(dropped=[position])
                    continue
                items.append(self.dataset[index])
                taken.append(int(position))
        uids = [str(uuid.uuid4()) for _ in items]
        with self._lock:
            self._uid_position.update(zip(uids, taken))
        self.drawn += len(items)
        return self._finish(self._collate(items), uids)

    def kept_round(self, rollout_id: int, samples_by_uid: Mapping[str, int]) -> dict[str, Any] | None:
        """The trainer gets the groups of ``samples_by_uid`` (uid -> samples) in
        round ``rollout_id``: their positions wait for the round's commit. Returns
        the round's ``round_data_rows`` (``{"samples", "rows"}``, the driver's
        ``rl_round_data_rows``), None when no group came from the dispenser."""
        from yeto.rl.engine.data_split import row_runs

        with self._lock:
            kept = {u: self._uid_position[u] for u in samples_by_uid if u in self._uid_position}
        self.kept(sorted(set(kept.values())), int(rollout_id))
        with self._lock:
            held = self.tasks.taken()
            self._uid_position = {u: p for u, p in self._uid_position.items()
                                  if p in held and u not in kept}
        if not kept:
            return None
        rows = [self.positions.row(p) for p in kept.values()]
        return {"samples": int(sum(int(samples_by_uid[u]) for u in kept)), "rows": row_runs(rows)}

    def cursor(self) -> dict[str, int]:
        """The rows drawn so far (recorded with each batch; the dispenser keeps
        the data position, nothing seeks to it)."""
        return {"dispensed": int(self.drawn)}


def install(trainer: Any, section: Mapping[str, Any], *, key: bytes | None = None,
            connect: Any = None, emit: Callable[..., None] | None = None,
            collate: Callable[[list[dict]], dict] | None = None,
            finish: Callable[[dict, list[str]], Any] | None = None) -> VerlDispensedBatches:
    """Open the island's connection and replace the trainer's prompt draw
    (module doc). ``section``: the plan's ``dispense`` section. ``key``: the
    island HMAC key (None: ``YETO_ISLAND_HMAC_KEY`` of this process)."""
    if key is None:
        from yeto.protocol import island_hmac_key_from_env

        key = island_hmac_key_from_env()
    count, digest = file_fingerprint(section["rows_file"])
    if count != int(section["rows"]) or digest.hex() != str(section["fingerprint"]):
        raise DispenseIslandError(f"{section['rows_file']} changed since the island entry read it")
    if getattr(getattr(trainer, "_fetch_one_gen_batch", None), "_yeto_dispense", None) is not None:
        # before any connection: a second holder of this learner id would supersede the first
        raise DispenseIslandError("verl trainer: a dispenser is installed already")
    tasks, positions = open_task_source(
        section["rows_file"], section, block=int(section["block"]), key=key, connect=connect,
        emit=lambda kind, **fields: logger.info("%s %s", kind, fields))
    try:
        wrapped = VerlDispensedBatches(trainer, tasks, positions, collate=collate, finish=finish)
    except BaseException:
        tasks.close(release=True)
        raise
    atexit.register(tasks.close)
    record = {"address": section["address"], "learner_id": int(section["learner_id"]),
              "incarnation": f"{tasks.incarnation:016x}", "rows": positions.rows,
              "filtered_rows": wrapped.filtered_rows, "block": int(section["block"]),
              "streams": (section.get("data_streams") or {}).get("streams")}
    logger.info("task dispenser installed on the verl trainer: %s", record)
    print(f"[yeto-verl] task dispenser installed: {json.dumps(record)}", flush=True)
    if emit is not None:
        emit("rl_dispense_install", **record)
    return wrapped
