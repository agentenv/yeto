"""Miles rollout through the durable sample buffer (rl-trainer-start-stop L4.5).

Two Miles rollout functions (``--rollout-function-path``, classes in
:mod:`.buffer_rollout_miles`) connect a sampler island and a trainer island
through a :class:`~yeto.rl.engine.sample_buffer.FileSampleBuffer`:

* sampler (``BufferWriterRolloutFn``): Miles generates the round as usual; every
  group is written to the buffer as one payload plus one index entry
  (:class:`GroupWriter`). The payload keeps each sample whole, so the per-token
  behavior log-probs (``rollout_log_probs``) and the per-token engine versions
  (``weight_versions``) reach the trainer for the D2 version gate and decoupled
  PPO. The entry's ``behavior_version`` is the OLDEST engine version in the group
  (the start version; P0-S schema 4).
* trainer (``BufferReaderRolloutFn``): the round is selected from the buffer, not
  generated (:class:`BatchReader`). Groups older than ``max_policy_age`` versions
  are passed over (consumed, not trained) and counted.

The consumed watermark: the reader remembers the buffer position after each
rollout id. The round cut before rollout ``N`` (all rollouts below ``N`` are
trained) commits the position after rollout ``N - 1`` and writes it into the cut
record (``RoundCutCheckpoint.buffer_watermark``); a restore resets the reader to
the watermark of the restored cut (``RoundCutCheckpoint.buffer_resume``). The
driver reaches the reader in the rollout executor's process with
:func:`executor_watermark_before` / :func:`executor_reset` (``__ray_call__``,
``MilesRolloutPool._in_executor``); :func:`wire_sample_buffer` connects both.

This module does not import Miles (it is tested without it).

Limits (first implementation):

* one trainer reads a buffer, or synced trainers read one lane each
  (:mod:`yeto.rl.engine.merged_release`); on a shared directory (``index.jsonl``) one
  sampler island writes it (``flock`` holds only on one host);
* the buffer lives in the checkpoint store (:func:`~yeto.rl.engine.resume.store_for`
  at ``BUFFER_ROOT_ENV``): a Modal Volume or a shared directory that both islands
  mount (visibility is the store's ``commit``/``reload``), or an ``s3://`` store
  read and written by object key (``CheckpointStore.objects``), not through its
  bucket mount;
* the trainer does no evaluation through the buffer (eval rounds are refused);
* ``--fully-async`` is not supported on either side (it selects its own rollout
  function).
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

# The payload format and the group writer are engine-neutral (S25 N2.3: the
# engine sampler island writes the same payloads); re-exported here.
from yeto.rl.engine.buffer_payload import (  # noqa: F401
    PAYLOAD_SCHEMA,
    GroupWriter,
    SampleBufferExhausted,
    SampleBufferRolloutError,
    decode_group,
    encode_group,
    group_entry_fields,
)
from yeto.rl.engine.merged_release import (
    LANE_ENV,
    BufferLane,
    LaneCursorLog,
    ObjectLaneCursorLog,
    gate_version,
    lane_cursor_log,
    merges_shards,
)
from yeto.rl.engine.data_split import DATA_ROW_KEY, DataSlot  # noqa: F401 - DATA_ROW_KEY re-exported

SAMPLER, TRAINER, CONVERTIBLE = "sampler", "trainer", "convertible"
ROLES = (SAMPLER, TRAINER, CONVERTIBLE)

BUFFER_ROOT_ENV = "YETO_SAMPLE_BUFFER_ROOT"  # checkpoint store path holding sample-buffer/
SAMPLER_ID_ENV = "YETO_SAMPLE_BUFFER_SAMPLER_ID"  # prefix of the sample ids a sampler writes
WAIT_ENV = "YETO_SAMPLE_BUFFER_WAIT_S"  # reader: max seconds to wait for a full round
POLL_ENV = "YETO_SAMPLE_BUFFER_POLL_S"  # reader: seconds between index reads while waiting
MAX_POLICY_AGE_ENV = "YETO_SAMPLE_BUFFER_MAX_POLICY_AGE"
SHARD_ENV = "YETO_SAMPLE_BUFFER_SHARD"  # convertible node: its own shard of the index
# S25 C1: several sampler islands on one shared directory (a Modal Volume): each
# sampler writes its own shard, the trainer merges these shards (comma list)
SHARDS_ENV = "YETO_SAMPLE_BUFFER_SHARDS"
DATA_SLOT_ENV = "YETO_SAMPLE_BUFFER_DATA_SLOT"  # sampler: "index/count" of the prompt data (S26 E9.3)
INCARNATION_ENV = "YETO_SAMPLER_INCARNATION"  # set by the sampler island before Ray starts
# the rollout functions run in Ray workers, which inherit the raylet's environment:
# connect_island_ray forwards these to every worker
ISLAND_ENVS = (BUFFER_ROOT_ENV, SAMPLER_ID_ENV, WAIT_ENV, POLL_ENV, MAX_POLICY_AGE_ENV, SHARD_ENV,
               SHARDS_ENV, INCARNATION_ENV, LANE_ENV, DATA_SLOT_ENV)

DEFAULT_WAIT_S = 3600.0
DEFAULT_POLL_S = 2.0
TRAINER_CONSUMER = "trainer"

_MODULE = "yeto.rl.adapters.miles.buffer_rollout_miles"
ROLLOUT_FUNCTION_PATHS = {
    SAMPLER: f"{_MODULE}.BufferWriterRolloutFn",
    TRAINER: f"{_MODULE}.BufferReaderRolloutFn",
    CONVERTIBLE: f"{_MODULE}.ConvertibleRolloutFn",
}
# the rollout functions that read the buffer (the trainer keeps a watermark)
READER_FUNCTION_PATHS = (ROLLOUT_FUNCTION_PATHS[TRAINER], ROLLOUT_FUNCTION_PATHS[CONVERTIBLE])


# ---------------------------------------------------------------------- argv
def rollout_argv(role: str | None, *, max_policy_age: int, fully_async: bool = False,
                 rollout_function_path: str | None = None) -> tuple[str, ...]:
    """The Miles flags of an island that writes (``sampler``) or reads (``trainer``)
    the sample buffer; ``()`` when ``role`` is None. Raises ``ValueError`` on a
    combination the buffer rollout does not support."""
    if role is None:
        return ()
    if role not in ROLES:
        raise ValueError(f"sample buffer role {role!r}: expected one of {ROLES}")
    if fully_async:
        raise ValueError("the sample buffer rollout does not support --fully-async "
                         "(it selects its own rollout function)")
    if rollout_function_path is not None:
        raise ValueError(f"the sample buffer rollout replaces the rollout function; "
                         f"{rollout_function_path!r} was also given")
    if role in (TRAINER, CONVERTIBLE) and int(max_policy_age) < 1:
        raise ValueError("a trainer that reads the sample buffer trains on samples of earlier "
                         f"versions: it needs max_policy_age >= 1 (got {max_policy_age})")
    return ("--rollout-function-path", ROLLOUT_FUNCTION_PATHS[role])


def island_env(role: str, *, store_root: str, sampler_id: str | None = None,
               max_policy_age: int | None = None, wait_s: float | None = None,
               shard: str | None = None, shards: Sequence[str] | None = None,
               lane: BufferLane | None = None, data_slot: str | None = None) -> dict[str, str]:
    """The environment the launch layer exports on the island (read in the rollout
    process). A convertible node writes like a sampler, into its own ``shard`` of
    the index, and reads like a trainer. ``shards``: the sampler shards of a
    launch with several sampler islands (a sampler writes the one named by its
    sampler id, the trainer merges them all). ``lane``: a synced trainer's
    buffer lane. ``data_slot``: a sampler's slot of the prompt data
    (``sample_roles.DataSlot``; several sampler islands draw disjoint prompts)."""
    if role not in ROLES:
        raise ValueError(f"sample buffer role {role!r}: expected one of {ROLES}")
    env = {BUFFER_ROOT_ENV: str(store_root)}
    if role in (SAMPLER, CONVERTIBLE):
        if not sampler_id:
            raise ValueError("a sampler needs a sampler id (sample id prefix)")
        env[SAMPLER_ID_ENV] = str(sampler_id)
    if role == CONVERTIBLE:
        if not shard:
            raise ValueError("a convertible node needs its buffer shard name")
        env[SHARD_ENV] = str(shard)
    if role in (TRAINER, CONVERTIBLE):
        if max_policy_age is None:
            raise ValueError("a trainer needs max_policy_age")
        env[MAX_POLICY_AGE_ENV] = str(int(max_policy_age))
        if wait_s is not None:
            env[WAIT_ENV] = f"{float(wait_s):g}"
    if shards:
        if role == CONVERTIBLE:
            raise ValueError("a convertible pool names its shards per node, not with shards")
        if role == SAMPLER and sampler_id not in shards:
            raise ValueError(f"sampler {sampler_id!r} is not one of the buffer shards {list(shards)}")
        env[SHARDS_ENV] = ",".join(shards)
    if lane is not None:
        if role != TRAINER:
            raise ValueError("only a trainer reads a buffer lane")
        env[LANE_ENV] = lane.env()
    if data_slot:
        if role != SAMPLER:
            raise ValueError("only a sampler island draws a data slot")
        env[DATA_SLOT_ENV] = DataSlot.parse(data_slot).env()
    return env


def buffer_shards(environ: Mapping[str, str]) -> tuple[str, ...]:
    """The sampler shards of :data:`SHARDS_ENV` (() when unset)."""
    return tuple(s.strip() for s in (environ.get(SHARDS_ENV) or "").split(",") if s.strip())


def open_buffer(environ: Mapping[str, str] | None = None, *, batch_size: int = 1,
                shard: str | None = None, create_index: bool = True, sampler: bool = False,
                reader: bool = False) -> Any:
    """The island's buffer, from :data:`BUFFER_ROOT_ENV` (store chosen by ``store_for``);
    with ``shard``, the writer of that shard of the index; ``create_index=False``
    for a reader whose index another host writes.

    On a store that needs one shard per sampler island (``sample_buffer.needs_shards``),
    or with :data:`SHARDS_ENV` (several sampler islands on one shared directory):
    ``sampler`` opens the shard named by :data:`SAMPLER_ID_ENV`, and ``reader``
    (the trainer) merges the shards into the index (with :data:`SHARDS_ENV`, the
    data ends once every named shard has ended)."""
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.sample_buffer import INDEX, FileSampleBuffer, needs_shards, shard_index

    env = os.environ if environ is None else environ
    root = env.get(BUFFER_ROOT_ENV)
    if not root:
        raise SampleBufferRolloutError(f"{BUFFER_ROOT_ENV} is not set")
    store = store_for(root, environ=env)
    merge = False
    named = buffer_shards(env)
    if (needs_shards(store) or named) and shard is None:
        if sampler:
            shard = env.get(SAMPLER_ID_ENV)
            if not shard:
                raise SampleBufferRolloutError(f"{SAMPLER_ID_ENV} is not set (the sampler's buffer shard)")
        merge = reader
    lane = BufferLane.parse(env.get(LANE_ENV)) if reader else None
    merge = merge and merges_shards(lane)
    return FileSampleBuffer(store, batch_size=batch_size,
                            index=INDEX if shard is None else shard_index(shard),
                            create_index=create_index, merge_shards=merge,
                            expected_shards=named if merge and named else None,
                            lane=lane)


def open_lane_log(buffer: Any) -> LaneCursorLog | ObjectLaneCursorLog | None:
    """The lane cursor log of a lane reader (None for a single reader)."""
    return lane_cursor_log(buffer)


def renumber_group(samples: Sequence[dict[str, Any]], *, position: int,
                   samples_per_group: int) -> list[dict[str, Any]]:
    """Give a group read from the buffer an identity unique in the buffer.

    Miles numbers groups and samples per sampler process from 0, so groups from two
    sampler rounds (or a restarted sampler) repeat them; the trainer's batch checks
    and trajectory ids need them unique. The buffer position is unique for the
    buffer's life and the same after a restore re-reads it, so ``group_index`` is
    the position and ``index`` is ``position * samples_per_group + j``. The
    sampler's numbers stay in ``metadata["yeto_sampler_identity"]``."""
    if len(samples) > int(samples_per_group):
        raise SampleBufferRolloutError(
            f"group at buffer position {position} has {len(samples)} samples > "
            f"samples per group {samples_per_group}")
    out = []
    for j, sample in enumerate(samples):
        sample = dict(sample)
        metadata = dict(sample.get("metadata") or {})
        metadata["yeto_sampler_identity"] = {"group_index": sample.get("group_index"),
                                             "index": sample.get("index")}
        sample.update(metadata=metadata, group_index=int(position),
                      index=int(position) * int(samples_per_group) + j)
        out.append(sample)
    return out


# ---------------------------------------------------------------------- sampler side
def apply_data_slot(data_source: Any, slot: DataSlot, *, shuffle: bool) -> int:
    """Keep only ``slot``'s rows in a Miles ``RolloutDataSource`` (S26 E9.3).

    Called once, before the first round and before a resume seek: the dataset's
    original rows become the slot's rows, and the current epoch's order is made
    again from them (the same seeded shuffle, over the slot). The data cursor of
    a later round then counts in slot rows, so a replacement incarnation with
    the same slot seeks to the same place. Each kept row records its index in
    the unsplit data as ``metadata["yeto_data_row"]`` (it reaches the buffer with the
    sample), so a run shows its samplers drew disjoint rows. Returns the slot's
    row count."""
    source = getattr(data_source, "__self__", data_source)
    dataset = getattr(source, "dataset", None)
    if slot.whole or dataset is None:
        return len(dataset) if dataset is not None else 0
    origin = getattr(dataset, "origin_samples", None)
    if origin is None:
        raise SampleBufferRolloutError(
            f"cannot split data source {type(source).__name__} into slot {slot.env()} "
            "(its dataset has no origin_samples)")
    if int(getattr(source, "sample_offset", 0) or 0) or int(getattr(source, "epoch_id", 0) or 0):
        raise SampleBufferRolloutError("the data slot is applied before the first draw")
    for row, sample in enumerate(origin):
        if isinstance(getattr(sample, "metadata", None), dict):
            sample.metadata[DATA_ROW_KEY] = row
    dataset.origin_samples = slot.rows(origin)
    dataset.samples = list(dataset.origin_samples)
    if shuffle:
        dataset.epoch_id = None  # force the reshuffle of the current epoch over the slot
        dataset.shuffle(int(source.epoch_id))
    return len(dataset.samples)


_DATA_CURSOR_FIELDS = ("sample_offset", "epoch_id", "sample_group_index", "sample_index")


def seek_data_source(data_source: Any, cursor: Mapping[str, int], *, shuffle: bool) -> None:
    """Put a Miles ``RolloutDataSource`` at ``cursor`` the way its ``load`` does
    (the four position fields, then the epoch's shuffle). Raises when the source
    does not have the fields (a source this resume does not know)."""
    source = getattr(data_source, "__self__", data_source)
    missing = [f for f in _DATA_CURSOR_FIELDS if not isinstance(getattr(source, f, None), int)]
    if missing or set(cursor) != set(_DATA_CURSOR_FIELDS):
        raise SampleBufferRolloutError(
            f"cannot seek data source {type(source).__name__} to {dict(cursor)} "
            f"(missing fields {missing})")
    for name in _DATA_CURSOR_FIELDS:
        setattr(source, name, int(cursor[name]))
    if shuffle and getattr(source, "dataset", None) is not None:
        source.dataset.shuffle(int(cursor["epoch_id"]))


class SamplerRoundWriter:
    """The sampler's buffer writes, resumable across incarnations (S25 C1).

    Opened in the rollout process. On open it reads where this sampler's earlier
    incarnations got to (:func:`~yeto.rl.engine.sampler_resume.resume_point`);
    :meth:`seek` puts the data source at the resume cursor before the first
    round; :meth:`write_round` writes one round's finished groups, each with its
    :class:`~yeto.rl.engine.sampler_resume.SamplerRound` record. After the
    reclaim marker appears (the island is leaving) nothing more is written: an
    old incarnation must not write while its replacement writes the same shard."""

    def __init__(self, buffer: Any, sampler_id: str, *, incarnation: str, resume: bool = True,
                 reclaimed: Callable[[], bool] = lambda: False) -> None:
        """``resume=False``: write as before (a convertible node: no resume
        point, no round check, no seek)."""
        from yeto.rl.engine.sampler_resume import RoundGuard, resume_point

        self.buffer = buffer
        self.sampler_id = str(sampler_id)
        self.incarnation = str(incarnation)
        self.reclaimed = reclaimed
        self.resume = resume_point(buffer.entries() if resume else (), self.sampler_id)
        self.guard = RoundGuard(self.resume) if resume else None
        self._writer = GroupWriter(buffer, f"{self.sampler_id}-{self.incarnation}")
        self._sought = False
        self.dropped_groups = 0  # groups not written because the island is leaving

    def seek(self, data_source: Any, *, shuffle: bool) -> dict[str, int] | None:
        """Before the first round only: the resume cursor applied (None: none)."""
        if self._sought or self.guard is None:
            return None
        self._sought = True
        cursor = self.resume.data_cursor
        if cursor is not None:
            seek_data_source(data_source, cursor, shuffle=shuffle)
        return cursor

    def write_round(self, rollout_id: int, groups: Sequence[tuple[str, list[dict[str, Any]]]],
                    data_cursor: Mapping[str, int] | None,
                    positions: Sequence[int | None] | None = None) -> int:
        """Write ``groups`` (``(group key, samples)``, aborted groups already left
        out) of round ``rollout_id``, with one store commit at the end; returns the
        groups written (the first ones, in order). ``positions`` (S27 P4): each
        group's task-dispenser position, kept in its round record."""
        from yeto.rl.engine.sampler_resume import SamplerRound

        if self.guard is not None:
            self.guard.check(rollout_id)
        written = 0
        # one store commit for the round, not one per group (S27 G-AO: 64 groups of a
        # 30B round took ~170 s of per-group Modal Volume commits)
        hold = getattr(self.buffer, "one_commit", None)
        with hold() if callable(hold) else nullcontext():
            for i, (key, samples) in enumerate(groups):
                if self.reclaimed():
                    self.dropped_groups += len(groups) - i
                    break
                record = SamplerRound(self.sampler_id, self.incarnation, int(rollout_id),
                                      data_cursor=data_cursor, last=i == len(groups) - 1,
                                      dispense_position=None if positions is None else positions[i])
                self._writer.write(samples, group_key=key, sampler_round=record.record())
                written += 1
        if written and self.guard is not None:
            self.guard.written(rollout_id)
        return written


# ---------------------------------------------------------------------- trainer side
@dataclass
class TakeReport:
    rollout_id: int
    groups: int
    stale_groups: int  # passed over: older than max_policy_age (consumed, not trained)
    start: int  # buffer position of the first selected entry
    end: int  # buffer position after the last selected entry
    behavior_versions: dict[int, int] = field(default_factory=dict)
    wait_s: float = 0.0
    # seconds of the take's phases: index refresh and selection (a store reload, the
    # shard merge), waiting for the payload reads, decoding; and the groups whose
    # payload was read ahead of the take (BatchReader prefetch)
    select_s: float = 0.0
    read_s: float = 0.0
    decode_s: float = 0.0
    prefetched: int = 0
    # S27 P4: task-dispenser positions of the stale groups (their sampler had a dispenser)
    stale_positions: list[int] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {"sample_buffer/groups": float(self.groups),
                "sample_buffer/stale_groups": float(self.stale_groups),
                "sample_buffer/start": float(self.start), "sample_buffer/end": float(self.end),
                "sample_buffer/wait_s": float(self.wait_s),
                "sample_buffer/select_s": float(self.select_s),
                "sample_buffer/read_s": float(self.read_s),
                "sample_buffer/decode_s": float(self.decode_s),
                "sample_buffer/prefetched_groups": float(self.prefetched)}


DEFAULT_READ_WORKERS = 8  # payload reads in flight (one Modal Volume file each)


class BatchReader:
    """Selects the trainer's rounds from the buffer (single consumer).

    Payload reads: the payloads of a take are read by ``read_workers`` threads at
    once (decoded in buffer order on the caller's thread). With ``prefetch``, after
    a take returns a background thread watches the index for the next round's
    entries and reads their payloads while the trainer trains; the next take uses
    the bytes read ahead (``TakeReport.prefetched``). The read cursor only moves in
    :meth:`take`: prefetching selects nothing. No store refresh (a Modal Volume
    reload) runs while a payload read of this reader is open: the reload of a
    Volume with open files fails."""

    def __init__(self, buffer: Any, *, max_policy_age: int, wait_s: float = DEFAULT_WAIT_S,
                 poll_s: float = DEFAULT_POLL_S, consumer: str = TRAINER_CONSUMER,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 cursor_log: LaneCursorLog | ObjectLaneCursorLog | None = None,
                 read_workers: int = DEFAULT_READ_WORKERS, prefetch: bool = True) -> None:
        """``cursor_log``: a synced trainer's lane cursor (:mod:`yeto.rl.engine.merged_release`):
        the reader continues after its last line and appends one per round.
        ``read_workers`` < 2 reads the payloads one by one; ``prefetch`` needs a
        buffer with ``peek`` (the class doc)."""
        if int(max_policy_age) < 1:
            raise ValueError("max_policy_age must be >= 1")
        self.buffer = buffer
        self.max_policy_age = int(max_policy_age)
        self.wait_s = float(wait_s)
        self.poll_s = float(poll_s)
        self.consumer = consumer
        self.clock = clock
        self.sleep = sleep
        self._ends: dict[int, int] = {}  # rollout id -> buffer position after its round
        # a synced trainer gates on the merged version it last published (set_gate_version)
        self.gate_version: int | None = None
        self.cursor_log = cursor_log
        self._pool = (ThreadPoolExecutor(int(read_workers), thread_name_prefix="buffer-read")
                      if int(read_workers) > 1 else None)
        self.prefetch = bool(prefetch) and callable(getattr(buffer, "peek", None))
        self._reads: dict[str, Future] = {}  # payload ref -> its read (ahead of a take)
        self._prefetcher: threading.Thread | None = None
        self._stop_prefetch = threading.Event()
        if cursor_log is not None:
            self.buffer.reset_to(cursor_log.last_cursor())

    def _read(self, ref: str) -> Future:
        if self._pool is None:
            done: Future = Future()
            try:
                done.set_result(self.buffer.read_payload(ref))
            except BaseException as error:  # noqa: BLE001 - raised by result()
                done.set_exception(error)
            return done
        return self._pool.submit(self.buffer.read_payload, ref)

    def take(self, rollout_id: int, groups: int, *, current_version: int | None
             ) -> tuple[list[tuple[int, list[dict[str, Any]]]], TakeReport]:
        """``groups`` ``(buffer position, decoded group)`` pairs for ``rollout_id``.
        Waits up to ``wait_s`` for them; after the data end returns what is left
        (raises when nothing is)."""
        if int(groups) < 1:
            raise ValueError("groups must be positive")
        rollout_id = int(rollout_id)
        started = self.clock()
        self._stop_prefetching()
        ahead = dict(self._reads)
        self._reads.clear()
        start = self.buffer.read_cursor
        chosen: list[tuple[int, Future, bool]] = []  # (position, read, read ahead)
        versions: dict[int, int] = {}
        stale = 0
        stale_positions: list[int] = []
        select_s = read_s = 0.0
        current_version = gate_version(current_version, self.gate_version)
        floor = None if current_version is None else int(current_version) - self.max_policy_age
        while len(chosen) < groups:
            # the reads of the last selection end before the next refresh (a reload)
            read_s += self._settle([read for _, read, _ in chosen])
            begun = self.clock()
            selected = self.buffer.select(self.consumer, groups - len(chosen))
            select_s += self.clock() - begun
            for item in selected:
                if floor is not None and item.entry.behavior_version < floor:
                    stale += 1
                    position = (item.entry.sampler_round or {}).get("dispense_position")
                    if isinstance(position, int) and not isinstance(position, bool):
                        stale_positions.append(position)
                    continue
                ref = item.entry.payload_ref
                read = ahead.pop(ref, None)
                chosen.append((item.position, read or self._read(ref), read is not None))
                versions[item.entry.behavior_version] = versions.get(item.entry.behavior_version, 0) + 1
            if len(chosen) >= groups or selected:
                continue
            if self.buffer.data_ended:
                break
            if self.clock() - started > self.wait_s:
                raise SampleBufferRolloutError(
                    f"rollout {rollout_id}: {len(chosen)}/{groups} groups after {self.wait_s:g} s "
                    f"(buffer position {self.buffer.read_cursor})")
            self.sleep(self.poll_s)
        read_s += self._settle([read for _, read, _ in chosen])
        begun = self.clock()
        out = [(position, decode_group(read.result())) for position, read, _ in chosen]
        decode_s = self.clock() - begun
        end = self.buffer.read_cursor
        if not out:
            raise SampleBufferExhausted(f"rollout {rollout_id}: the sample buffer ended "
                                        f"with no trainable group (position {end}, {stale} passed "
                                        "over as too old)")
        self._ends[rollout_id] = end
        if self.cursor_log is not None:
            self.cursor_log.append(rollout_id, [position for position, _ in out], end)
        if self.prefetch and not self.buffer.data_ended:
            next_floor = None if current_version is None else floor + 1  # the next take's
            self._start_prefetching(int(groups), next_floor)
        return out, TakeReport(rollout_id=rollout_id, groups=len(out), stale_groups=stale,
                               start=start, end=end, behavior_versions=dict(sorted(versions.items())),
                               wait_s=round(self.clock() - started, 3), select_s=round(select_s, 3),
                               read_s=round(read_s, 3), decode_s=round(decode_s, 3),
                               prefetched=sum(1 for _, _, early in chosen if early),
                               stale_positions=stale_positions)

    def _settle(self, reads: list[Future]) -> float:
        """Wait for ``reads`` (their errors surface at ``result()``); the seconds waited."""
        pending = [read for read in reads if not read.done()]
        if not pending:
            return 0.0
        begun = self.clock()
        wait(pending)
        return self.clock() - begun

    # ------------------------------------------------------------ prefetch
    def _start_prefetching(self, groups: int, floor: int | None) -> None:
        self._stop_prefetch.clear()
        self._prefetcher = threading.Thread(target=self._prefetch_loop, args=(groups, floor),
                                            name="buffer-prefetch", daemon=True)
        self._prefetcher.start()

    def _stop_prefetching(self) -> None:
        """Stop the prefetch thread and wait for its reads (so a refresh may reload)."""
        if self._prefetcher is not None:
            self._stop_prefetch.set()
            self._prefetcher.join()
            self._prefetcher = None
        self._settle(list(self._reads.values()))

    def _prefetch_loop(self, groups: int, floor: int | None) -> None:
        """Read the payloads of the next ``groups`` trainable entries as they appear
        (the floor is the next take's stale bound). Best effort: an error here only
        ends the prefetch; the take reads what is missing and raises its own errors."""
        deadline = self.clock() + self.wait_s
        try:
            while not self._stop_prefetch.is_set() and self.clock() < deadline:
                self._settle(list(self._reads.values()))  # no open reads during the refresh
                if self._stop_prefetch.is_set():
                    return
                wanted = [item for item in self.buffer.peek(groups)
                          if floor is None or item.entry.behavior_version >= floor]
                for item in wanted:
                    ref = item.entry.payload_ref
                    if ref not in self._reads:
                        self._reads[ref] = self._read(ref)
                if len(wanted) >= groups or self.buffer.data_ended:
                    return
                self._stop_prefetch.wait(self.poll_s)
        except Exception as error:  # noqa: BLE001 - the take reads (and reports) on its own
            print(f"[sample-buffer] prefetch stopped: {type(error).__name__}: {error}", flush=True)

    def watermark_before(self, rollout_id: int) -> int:
        """Commit and return the position after the last round below ``rollout_id``
        (every round below it is trained when the cut before ``rollout_id`` runs)."""
        done = [rid for rid in self._ends if rid < int(rollout_id)]
        if done:
            self.buffer.commit_consumed(self._ends[max(done)])
            for rid in done:
                del self._ends[rid]
        return int(self.buffer.watermark)

    def reset(self, watermark: int) -> int:
        """Continue at the restored cut's watermark (rounds after it are selected again)."""
        self.buffer.reset_to(int(watermark))
        self._ends.clear()
        return int(self.buffer.watermark)


# ---------------------------------------------------------------------- executor link
_ACTIVE: dict[str, BatchReader] = {}
# the mode of a convertible node's rollout function (``ConvertibleRolloutFn``):
# SAMPLER generates into the node's shard, TRAINER reads the buffer
_MODE: dict[str, str] = {CONVERTIBLE: SAMPLER}


def convertible_mode() -> str:
    return _MODE[CONVERTIBLE]


def executor_set_mode(_executor: Any, mode: str) -> str:
    """Runs INSIDE the rollout executor actor (``__ray_call__``): switch a
    convertible node's rollout function between generating (``sampler``) and
    reading the buffer (``trainer``). Returns the previous mode."""
    if mode not in (SAMPLER, TRAINER):
        raise SampleBufferRolloutError(f"convertible rollout mode {mode!r}: expected "
                                       f"{SAMPLER} or {TRAINER}")
    previous, _MODE[CONVERTIBLE] = _MODE[CONVERTIBLE], mode
    return previous


def register_reader(reader: BatchReader) -> None:
    """Called by the reader rollout function in the rollout executor's process."""
    _ACTIVE[TRAINER] = reader


def _active_reader() -> BatchReader:
    reader = _ACTIVE.get(TRAINER)
    if reader is None:
        raise SampleBufferRolloutError("no sample buffer reader in this rollout process")
    return reader


def executor_watermark_before(_executor: Any, rollout_id: int) -> int:
    """Runs INSIDE the rollout executor actor (``__ray_call__``)."""
    return _active_reader().watermark_before(int(rollout_id))


def executor_read_cursor(_executor: Any) -> int | None:
    """Runs INSIDE the rollout executor actor: the buffer position after the last
    selected group (None before the reader exists)."""
    reader = _ACTIVE.get(TRAINER)
    return None if reader is None else int(reader.buffer.read_cursor)


def executor_read_shards(_executor: Any) -> dict[str, int] | None:
    """Runs INSIDE the rollout executor actor: the read cursor per sampler shard
    (None before the reader exists or when the buffer has no merged shards)."""
    reader = _ACTIVE.get(TRAINER)
    if reader is None:
        return None
    return reader.buffer.shard_positions(reader.buffer.read_cursor) or None


def executor_set_gate_version(_executor: Any, version: int) -> int | None:
    """Runs INSIDE the rollout executor actor: a synced trainer's merged version,
    the version its ``max_policy_age`` gate compares against. Returns the previous
    one (None before the reader exists: nothing to set)."""
    reader = _ACTIVE.get(TRAINER)
    if reader is None:
        return None
    previous, reader.gate_version = reader.gate_version, int(version)
    return previous


def executor_reset(_executor: Any, watermark: int) -> int:
    """Runs INSIDE the rollout executor actor (``__ray_call__``)."""
    return _active_reader().reset(int(watermark))


class BufferCutLink:
    """Driver side: the round cut's watermark and restore hooks over ``in_executor``
    (``MilesRolloutPool._in_executor``: runs ``fn(executor, *args)`` in the actor)."""

    def __init__(self, in_executor: Callable[..., Any]) -> None:
        self.in_executor = in_executor

    def watermark(self, rollout_id: int) -> int:
        return int(self.in_executor(executor_watermark_before, int(rollout_id)))

    def resume(self, watermark: int) -> int:
        return int(self.in_executor(executor_reset, int(watermark)))


def wire_sample_buffer(checkpoint: Any, rollout: Any) -> BufferCutLink:
    """Give a ``RoundCutCheckpoint`` the trainer's buffer watermark and restore hook."""
    in_executor = getattr(rollout, "_in_executor", None)
    if not callable(in_executor):
        raise SampleBufferRolloutError(f"rollout pool {type(rollout).__name__} cannot run "
                                       "code in the rollout executor")
    link = BufferCutLink(in_executor)
    checkpoint.buffer_watermark = link.watermark
    checkpoint.buffer_resume = link.resume
    return link
