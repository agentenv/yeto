"""Miles rollout through the durable sample buffer.

Two Miles rollout functions (``--rollout-function-path``, classes in
:mod:`.buffer_rollout_miles`) connect a sampler island and a trainer island
through a :class:`~yeto.rl.engine.sample_buffer.FileSampleBuffer`:

* sampler (``BufferWriterRolloutFn``): Miles generates the round as usual; every
  group is written to the buffer as one payload plus one index entry
  (:class:`GroupWriter`). The payload keeps each sample whole, so the per-token
  behavior log-probs (``rollout_log_probs``) and the per-token engine versions
  (``weight_versions``) reach the trainer for the version gate and decoupled
  PPO. The entry's ``behavior_version`` is the OLDEST engine version in the group
  (the start version; schema 4).
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
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

# The payload format and the group writer are engine-neutral (the
# engine sampler island writes the same payloads); re-exported here.
from yeto.rl.engine.buffer_payload import (  # noqa: F401
    finite_rewards,
    PAYLOAD_SCHEMA,
    DamagedPayload,
    DamagedPayloadLimit,
    GroupWriter,
    SampleBufferExhausted,
    SampleBufferRolloutError,
    decode_group,
    encode_group,
    group_entry_fields,
    nonfinite_logprob,
    passed_over_text,
    payload_is_missing,
    payload_oldest_version,
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
from yeto.rl.engine.incarnation_lease import LeaseUnknown
from yeto.rl.engine.policy_age import age_floor
from yeto.rl.engine.trainer_lifecycle import Selected
from yeto.rl.engine.xhw_guard import downweight_group, rollout_stats, source_key

SAMPLER, TRAINER, CONVERTIBLE = "sampler", "trainer", "convertible"
ROLES = (SAMPLER, TRAINER, CONVERTIBLE)

BUFFER_ROOT_ENV = "YETO_SAMPLE_BUFFER_ROOT"  # checkpoint store path holding sample-buffer/
SAMPLER_ID_ENV = "YETO_SAMPLE_BUFFER_SAMPLER_ID"  # prefix of the sample ids a sampler writes
WAIT_ENV = "YETO_SAMPLE_BUFFER_WAIT_S"  # reader: max seconds without a new group while a round fills
POLL_ENV = "YETO_SAMPLE_BUFFER_POLL_S"  # reader: seconds between index reads while waiting
MAX_POLICY_AGE_ENV = "YETO_SAMPLE_BUFFER_MAX_POLICY_AGE"
# reader: 0 turns off "one trained group per store-dispenser task" (TaskSlots); default on
ONE_GROUP_PER_TASK_ENV = "YETO_SAMPLE_BUFFER_ONE_GROUP_PER_TASK"
# reader: damaged payloads in a row (or, at least that many, over half of the payloads
# a take decoded) at which the take raises instead of passing them over; 0: no limit
MAX_DAMAGED_RUN_ENV = "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN"
DEFAULT_MAX_DAMAGED_RUN = 8
SHARD_ENV = "YETO_SAMPLE_BUFFER_SHARD"  # convertible node: its own shard of the index
# Several sampler islands on one shared directory (a Modal Volume): each
# sampler writes its own shard, the trainer merges these shards (comma list)
SHARDS_ENV = "YETO_SAMPLE_BUFFER_SHARDS"
DATA_SLOT_ENV = "YETO_SAMPLE_BUFFER_DATA_SLOT"  # sampler: "index/count" of the prompt data
INCARNATION_ENV = "YETO_SAMPLER_INCARNATION"  # set by the sampler island before Ray starts
# the rollout functions run in Ray workers, which inherit the raylet's environment:
# connect_island_ray forwards these to every worker
ISLAND_ENVS = (BUFFER_ROOT_ENV, SAMPLER_ID_ENV, WAIT_ENV, POLL_ENV, MAX_POLICY_AGE_ENV, SHARD_ENV,
               SHARDS_ENV, INCARNATION_ENV, LANE_ENV, DATA_SLOT_ENV,
               # the trainer's relay pump (yeto.rl.spot_rollout.relay_buffer.RELAY_ENVS)
               "YETO_SAMPLE_RELAY_ADDR", "YETO_SAMPLE_RELAY_CHANNEL", "YETO_SAMPLE_RELAY_PREFETCH",
               # the pump's boat orphan watchdog (yeto.rl.harness.harbor.boat_labels)
               "YETO_BOAT_RUN_ID", "YETO_BOAT_ORPHAN_LEASE_S", "BOAT_API_KEY")

DEFAULT_WAIT_S = 3600.0
DEFAULT_POLL_S = 2.0
TRAINER_CONSUMER = "trainer"

_MODULE = "yeto.rl.adapters.miles.buffer_rollout_miles"
_DEFER = object()  # BatchReader._gate_verdict: the lease could not be read
ROLLOUT_FUNCTION_PATHS = {
    SAMPLER: f"{_MODULE}.BufferWriterRolloutFn",
    TRAINER: f"{_MODULE}.BufferReaderRolloutFn",
    CONVERTIBLE: f"{_MODULE}.ConvertibleRolloutFn",
}
# a reader's rollout log: Miles' rollout metrics over the real samples (padding left out)
ROLLOUT_LOG_FUNCTION = f"{_MODULE}.log_rollout_data"
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
    buffer lane. ``data_slot``: a sampler's or a convertible node's slot of the
    prompt data (``sample_roles.DataSlot``; several sampler islands, or the nodes of
    a convertible pool, draw disjoint prompts)."""
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
        if role not in (SAMPLER, CONVERTIBLE):
            raise ValueError("only a sampler island or a convertible node draws a data slot")
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
    if sampler:
        # a spot sampler writes its groups to the head's relay (env unset: no change)
        from yeto.rl.spot_rollout.relay_buffer import relay_sampler_buffer

        relay = relay_sampler_buffer(env)
        if relay is not None:
            return relay
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
    buffer = FileSampleBuffer(store, batch_size=batch_size,
                              index=INDEX if shard is None else shard_index(shard),
                              create_index=create_index, merge_shards=merge,
                              expected_shards=named if merge and named else None,
                              lane=lane)
    if reader and shard is None and lane is None and not merge:
        check_unmerged_shards(buffer)
    return buffer


def check_unmerged_shards(buffer: Any) -> None:
    """A single trainer that does not merge shards reads only the merged index: when
    that index is empty but sampler islands wrote shards (each sampler its own
    ``shards/<name>``, as on a store with ``buffer_shards`` or a multi-sampler run),
    the trainer would poll an index nobody writes until its wait runs out. Refuse at
    open and name the flag that makes it merge them."""
    names = list(buffer.index.shard_names())
    if names and len(buffer) == 0:
        raise SampleBufferRolloutError(
            f"the sample buffer holds sampler shards {names} but no merged index, and this "
            f"trainer does not merge shards: launch it with --rl-buffer-shards {','.join(names)} "
            "(every sampler shard of the run)")


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


def miles_groups(groups: Sequence[tuple[int, Sequence[dict[str, Any]]]], *, samples_per_group: int,
                 r3_layers: tuple[int, int] | None = None, r3_num_experts: int | None = None
                 ) -> tuple[list[tuple[int, list[dict[str, Any]]]], list[str]]:
    """The taken groups renumbered (:func:`renumber_group`) and in Miles' sample form
    (``buffer_payload.miles_sample_dict``), and why each group that cannot be converted
    was left out. One malformed sample (an R3 routing row that mixes pads and ids, an
    expert id past the model's, a layer count of another model) leaves its group out
    of the round, which is padded instead, rather than stopping the trainer (the caller
    takes again when no group of a take converts, and fails after a few such takes)."""
    from yeto.rl.engine.buffer_payload import miles_sample_dict  # noqa: PLC0415

    kept, problems = [], []
    for position, group in groups:
        try:
            kept.append((position, [miles_sample_dict(d, r3_layers=r3_layers, r3_num_experts=r3_num_experts)
                                    for d in renumber_group(group, position=position,
                                                            samples_per_group=samples_per_group)]))
        except SampleBufferRolloutError as error:
            problems.append(f"buffer position {position}: {error}")
    for problem in problems:
        print(f"[sample-buffer] WARNING: group left out of the round: {problem}", flush=True)
    return kept, problems


PADDING_GROUP_BASE = 1 << 40  # group_index of padding samples: past any buffer position
PADDING_KEY = "yeto_padding"


def padding_samples(groups: Sequence[Sequence[dict[str, Any]]], count: int, *,
                    samples_per_group: int) -> list[dict[str, Any]]:
    """``count`` samples that fill a round to Miles' global batch without training.

    Each is a copy of the round's shortest real sample with a response (the cheapest
    forward pass)
    with ``loss_mask`` all 0 and ``remove_sample`` set (no loss, no gradient), and a
    group of its own (``group_index`` from ``PADDING_GROUP_BASE``): a one-sample group
    centers to advantage 0 and takes no part in a real group's mean or spread.
    ``metadata["yeto_padding"]`` marks it for the statistics."""
    if int(count) <= 0:
        return []
    real = [d for group in groups for d in group]
    if not real:
        raise SampleBufferRolloutError("no real sample to build padding from")
    # a sample with a response (an empty one, e.g. an engine that stopped at once, gives
    # Miles a zero-length response row: ``x[-0:]`` is the whole sequence, not nothing)
    responded = [d for d in real if int(d.get("response_length") or 0) > 0] or real
    shortest = min(responded, key=lambda d: len(d.get("tokens") or ()))
    out = []
    for k in range(int(count)):
        pad = dict(shortest)
        pad["loss_mask"] = [0] * int(pad.get("response_length") or 0)
        pad["remove_sample"] = True
        pad["rollout_id"] = None
        pad["metadata"] = {**dict(pad.get("metadata") or {}), PADDING_KEY: True}
        pad["group_index"] = PADDING_GROUP_BASE + k
        pad["index"] = (PADDING_GROUP_BASE + k) * int(samples_per_group)
        out.append(pad)
    return out


def is_padding(sample: Any) -> bool:
    """A padding sample of :func:`padding_samples` (a dict or a Miles ``Sample``)."""
    meta = sample.get("metadata") if isinstance(sample, dict) else getattr(sample, "metadata", None)
    return bool((meta or {}).get(PADDING_KEY))


# ---------------------------------------------------------------------- sampler side
def apply_data_slot(data_source: Any, slot: DataSlot, *, shuffle: bool) -> int:
    """Keep only ``slot``'s rows in a Miles ``RolloutDataSource``.

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
    """The sampler's buffer writes, resumable across incarnations.

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
        groups written (the first ones, in order). ``positions``: each
        group's task-dispenser position, kept in its round record."""
        from yeto.rl.engine.sampler_resume import SamplerRound

        if self.guard is not None:
            self.guard.check(rollout_id)
        written = 0
        # one store commit for the round, not one per group (64 groups of a
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
    # task-dispenser positions of the stale groups (their sampler had a dispenser)
    stale_positions: list[int] = field(default_factory=list)
    # the version the max_policy_age gate compared against (not a metric)
    gate_version: int | None = None
    # per-source (xhw_guard.source_key) groups and tokens of the trained groups, the
    # source of each chosen buffer position, and the groups a quarantine passed over
    source_groups: dict[str, int] = field(default_factory=dict)
    source_tokens: dict[str, int] = field(default_factory=dict)
    position_sources: dict[int, str] = field(default_factory=dict)
    quarantined_groups: int = 0
    # groups of a fenced (superseded) sampler incarnation written after its successor took
    # the shard: never read. On a shard with ownership records the merge drops them
    # (sample_buffer module doc; ``stale_incarnation_shards``: shard, owner epoch, owner,
    # writer, groups); in a shard's legacy part the lease gate does (IncarnationGate)
    stale_incarnation_groups: int = 0
    stale_incarnation_shards: list[dict[str, Any]] = field(default_factory=list)
    # what the buffer's merge found so far (cumulative, FileSampleBuffer.merge_counters):
    # merged records that repeat an earlier one (passed over), shard records an earlier
    # merger skipped for good, damaged shard records passed over
    duplicate_merged_records: int = 0
    unmerged_shard_gaps: int = 0
    bad_shard_records: int = 0
    # groups whose sampler's lease could not be read (IncarnationGate fails closed):
    # neither trained nor consumed; they wait (judged again) for a later selection
    # (BatchReader._deferred). Counted once, by the take that first deferred them
    lease_deferred_groups: int = 0
    # groups with fewer samples than the group size (a sampler lost trials) that are
    # trained anyway (their advantages use their own samples only); groups with fewer
    # than MIN_GROUP_SAMPLES (no in-group spread) or more than the group size are
    # consumed, never trained
    incomplete_groups: int = 0
    dropped_small_groups: int = 0
    # the round's samples fall short of groups * group_size: the whole group that
    # would overflow it waits for the next take (carried over, never split), and the
    # rest is filled with padding samples (loss mask 0, a group of their own)
    carried_over_groups: int = 0
    padded_samples: int = 0
    # padded_samples over the round's global batch: Miles divides the round's loss by the
    # full batch, so the real samples' gradient shrinks by this fraction (not rescaled)
    padded_fraction: float = 0.0
    # groups with a sample whose reward is missing or not finite (a failed reward call):
    # consumed, never trained (one NaN reward makes the whole group's advantages NaN)
    nonfinite_reward_groups: int = 0
    # groups with a sample whose behavior log-probs (rollout_log_probs) hold a NaN/inf:
    # consumed, never trained (TIS / off-policy correction would make the loss NaN)
    nonfinite_logprob_groups: int = 0
    # groups whose payload is missing or is not a group (cut short, not JSON, another
    # schema): consumed, never trained; one record each in ``damaged_payloads``
    # ({"position", "sample_id", "payload_ref", "reason"[, "dispense_position"]}). A
    # payload the store failed to return is not one of them: that take raises
    damaged_payload_groups: int = 0
    damaged_payloads: list[dict[str, Any]] = field(default_factory=list)
    # of ``stale_groups``: groups whose index record passed the age gate but whose
    # payload holds an older version, below the floor (the record and the payload
    # disagree; the tokens trained are the payload's, so the payload decides)
    stale_payload_groups: int = 0
    # groups of a store-dispenser task that an earlier group of the buffer already
    # holds (the task was given out and written twice): consumed, never trained; one
    # record each in ``duplicate_tasks`` ({"slot", "position", "sample_id",
    # "first_position", "first_sample_id", "first_trained_here"[, "plan", "attempt"]}).
    # ``duplicate_task_unverified_groups``: those whose first group this process did
    # not judge itself (below a restored watermark, or of another lane: TaskSlots)
    duplicate_task_groups: int = 0
    duplicate_task_unverified_groups: int = 0
    duplicate_tasks: list[dict[str, Any]] = field(default_factory=list)
    # groups inside the policy-age window whose version no record names for a trainer
    # that resumed behind its releases (BatchReader.set_unidentified_gate): consumed,
    # never trained; {version: groups} and their task-dispenser positions
    unidentified_policy_groups: int = 0
    unidentified_policy_versions: dict[int, int] = field(default_factory=dict)
    unidentified_policy_positions: list[int] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        return {"sample_buffer/groups": float(self.groups),
                "sample_buffer/stale_groups": float(self.stale_groups),
                "sample_buffer/start": float(self.start), "sample_buffer/end": float(self.end),
                "sample_buffer/wait_s": float(self.wait_s),
                "sample_buffer/select_s": float(self.select_s),
                "sample_buffer/read_s": float(self.read_s),
                "sample_buffer/decode_s": float(self.decode_s),
                "sample_buffer/prefetched_groups": float(self.prefetched),
                "sample_buffer/xhw_quarantined_groups": float(self.quarantined_groups),
                "sample_buffer/stale_incarnation_groups": float(self.stale_incarnation_groups),
                "sample_buffer/duplicate_merged_records": float(self.duplicate_merged_records),
                "sample_buffer/unmerged_shard_gaps": float(self.unmerged_shard_gaps),
                "sample_buffer/bad_shard_records": float(self.bad_shard_records),
                "sample_buffer/lease_deferred_groups": float(self.lease_deferred_groups),
                "sample_buffer/incomplete_groups": float(self.incomplete_groups),
                "sample_buffer/dropped_small_groups": float(self.dropped_small_groups),
                "sample_buffer/carried_over_groups": float(self.carried_over_groups),
                "sample_buffer/padded_samples": float(self.padded_samples),
                "sample_buffer/padded_fraction": float(self.padded_fraction),
                "sample_buffer/nonfinite_reward_groups": float(self.nonfinite_reward_groups),
                "sample_buffer/nonfinite_logprob_groups": float(self.nonfinite_logprob_groups),
                "sample_buffer/damaged_payload_groups": float(self.damaged_payload_groups),
                "sample_buffer/stale_payload_groups": float(self.stale_payload_groups),
                "sample_buffer/duplicate_task_groups": float(self.duplicate_task_groups),
                "sample_buffer/duplicate_task_unverified_groups": float(self.duplicate_task_unverified_groups),
                "sample_buffer/unidentified_policy_groups": float(self.unidentified_policy_groups)}


def _dispense_position(entry: Any) -> int | None:
    """The task-dispenser position of a buffer entry (its sampler had a dispenser)."""
    position = (getattr(entry, "sampler_round", None) or {}).get("dispense_position")
    return position if isinstance(position, int) and not isinstance(position, bool) else None


def task_key(entry: Any) -> tuple[str | None, int] | None:
    """``(dispenser plan, slot)`` of a store-dispenser buffer entry: its
    ``dispense_position`` when the round record also names the dispenser's attempt
    (``dispense_attempt``, which only ``store_dispense`` hands out), with the
    dispenser's identity (``dispense_plan``: ``StoreDispensedPrompts.plan_id``, the
    namespace and a digest of its plan).
    A slot is one task of that dispenser (a row in one pass: ``SlotMap``); a dispenser
    opened under another namespace on the same buffer (more data, more passes, another
    seed) numbers its slots from 0 again, and they are other tasks. A record with an
    attempt and no plan (written before the plan was recorded) is keyed by its slot
    alone. None for every other entry: the syncer's dispenser records a position
    without an attempt, and its positions are not judged here."""
    record = getattr(entry, "sampler_round", None) or {}
    attempt = record.get("dispense_attempt")
    slot = _dispense_position(entry)
    if slot is None or not isinstance(attempt, int) or isinstance(attempt, bool):
        return None
    plan = record.get("dispense_plan")
    return (str(plan) if plan else None, slot)


class TaskSlots:
    """One trained group per store-dispenser task (the reader's last line of defense).

    The dispenser gives a task to one sampler at a time, but two groups of one slot
    can reach the buffer without any fence seeing it: a holder that stalls longer
    than its lease between ``still_holds`` and the write, a scan that expires a claim
    from a listing older than its settlement, a group written but not yet settled
    when its sampler is retired. The reader trains the first group of a task
    (:func:`task_key`) in buffer order and passes over every later one
    (``TakeReport.duplicate_task_groups``).

    A task is held by the first group that was *accepted for training*. A group the
    reader passed over (too old, a fenced incarnation's, quarantined, too small,
    without a finite reward, a damaged payload), selected by a take that raised, or
    selected in a round that was rolled back (:meth:`reset`) holds nothing, so the
    group written in its place is trained.

    What this process did not judge. After a restart (or a reset to a watermark
    outside the positions this reader judged) the reader does not know what became
    of the entries below the watermark; the index does not say. Those entries (and,
    for a synced trainer, the entries of the other lanes) are taken from the index:
    each holds its task unless the incarnation gate fences it (a verdict the index
    and the lease decide, the same in every process). So a judgement does not depend
    on a restart when the first group was trained or fenced, which is every case in
    which the task would otherwise be trained twice. The one case it changes: a first
    group below the watermark that the earlier process passed over as too old, too
    small, damaged or without a finite reward is counted as trained, and the group
    written in its place is passed over. That task is then trained zero times, never
    twice (accepted: the cut record holds no list of trained tasks). Such a verdict
    is marked (``first_trained_here`` False in the record) and counted
    (``TakeReport.duplicate_task_unverified_groups``)."""

    def __init__(self, buffer: Any, gate: Any = None) -> None:
        self.buffer = buffer
        self.gate = gate
        self.owner: dict[tuple[str | None, int], int] = {}  # task -> position this reader accepted
        self.history: dict[tuple[str | None, int], list[int]] = {}  # task -> positions it did not judge
        # this reader judged the positions judged_from <= p < judged_to (of its lane)
        self.judged_from: int | None = None
        self.judged_to = 0
        self._absorbed = 0  # index positions below this are in ``history``
        self._pending: dict[tuple[str | None, int], int] = {}  # accepted by the take under way

    def begin(self, cursor: int) -> None:
        """A take starts at ``cursor``; what a take that raised accepted is forgotten."""
        if self.judged_from is None:
            self.judged_from = self.judged_to = int(cursor)
        self._pending = {}

    def _absorb(self, upto: int) -> None:
        lane = getattr(self.buffer, "lane", None)
        start = int(self.judged_from or 0)
        if self._absorbed >= (int(upto) if lane is not None else start):
            return
        entries = self.buffer.entries()
        end = min(int(upto) if lane is not None else start, len(entries))
        for position in range(self._absorbed, end):
            if position >= start and lane.owns(position):
                continue  # judged by this reader
            key = task_key(entries[position])
            if key is not None:
                self.history.setdefault(key, []).append(position)
        self._absorbed = max(self._absorbed, end)

    def _fenced(self, position: int) -> bool:
        if self.gate is None:
            return False
        return self.gate.verdict(self.buffer.entries()[position], int(position)) is not None

    def first(self, entry: Any, position: int) -> tuple[int, bool] | None:
        """``(position, trained by this reader)`` of the earlier group that holds this
        entry's task; None when the entry may be trained (no store-dispenser task, or
        its task is free)."""
        key = task_key(entry)
        if key is None:
            return None
        self._absorb(int(position))
        for earlier in self.history.get(key, ()):
            if earlier < int(position) and not self._fenced(earlier):
                return earlier, False
        owner = self.owner.get(key, self._pending.get(key))
        return (owner, True) if owner is not None and owner < int(position) else None

    def accept(self, key: tuple[str | None, int] | None, position: int) -> None:
        """The group at ``position`` is in the round: it holds its task once the take returns."""
        if key is not None:
            self._pending.setdefault(key, int(position))

    def release(self, position: int) -> None:
        """The group at ``position`` left the take under way: it holds no task."""
        self._pending = {key: p for key, p in self._pending.items() if p != int(position)}

    def commit(self, end: int) -> None:
        """The take returns its round (the positions below ``end``): its groups hold their tasks."""
        for key, position in self._pending.items():
            self.owner.setdefault(key, position)
        self._pending = {}
        self.judged_to = max(self.judged_to, int(end))

    def reset(self, watermark: int) -> None:
        """The reader continues at ``watermark``. Inside the positions it judged: the
        groups it accepted from there on are selected (and judged) again. Outside them
        (below, or past them: a cut another process wrote): as after a restart, what
        lies below the watermark is taken from the index."""
        watermark = int(watermark)
        self._pending = {}
        if self.judged_from is None:
            return
        if not self.judged_from <= watermark <= self.judged_to:
            self.owner, self.history, self._absorbed = {}, {}, 0
            self.judged_from = self.judged_to = watermark
            return
        self.owner = {key: position for key, position in self.owner.items() if position < watermark}
        self.history = {key: kept for key, positions in self.history.items()
                        if (kept := [p for p in positions if p < watermark])}
        self._absorbed = min(self._absorbed, watermark)
        self.judged_to = watermark


def unidentified_metadata(report: Any) -> dict[str, Any]:
    """The round metadata of the groups a reader passed over as of a version no record
    names (``TakeReport.unidentified_policy_*``); empty when there were none."""
    count = int(getattr(report, "unidentified_policy_groups", 0) or 0)
    if not count:
        return {}
    from yeto.rl.engine.data_split import row_runs  # noqa: PLC0415

    out: dict[str, Any] = {"buffer_unidentified_groups": count,
                           "buffer_unidentified_versions": [[int(v), int(n)] for v, n in sorted(
                               report.unidentified_policy_versions.items())]}
    if report.unidentified_policy_positions:
        out["buffer_unidentified_positions"] = row_runs(report.unidentified_policy_positions)
    return out


MIN_GROUP_SAMPLES = 2  # fewer: no in-group spread, no group-relative advantage
# the reader's warning when it reads groups and keeps none (BatchReader._note_selection)
UNIDENTIFIED_STARVED_EVENT = "rl_buffer_unidentified_policy_starved"
UNIDENTIFIED_STARVED_SELECTS_ENV = "YETO_BUFFER_UNIDENTIFIED_STARVED_SELECTS"  # 0: never warn
DEFAULT_UNIDENTIFIED_STARVED_SELECTS = 3


SETTLE_WARN_S = 300.0  # _settle warns every this many seconds a read stays open


def _starved_selects_limit(environ: Mapping[str, str] | None = None) -> int:
    raw = (os.environ if environ is None else environ).get(UNIDENTIFIED_STARVED_SELECTS_ENV, "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_UNIDENTIFIED_STARVED_SELECTS
    except ValueError:
        return DEFAULT_UNIDENTIFIED_STARVED_SELECTS


# the reader's warning when one source's groups are all passed over as bad (damaged
# payload, non-finite reward or log-prob, a size outside the group) many in a row while
# other sources keep the run going (BatchReader._note_source)
SOURCE_BAD_STREAK_EVENT = "rl_buffer_source_bad_streak"
SOURCE_BAD_STREAK_ENV = "YETO_BUFFER_SOURCE_BAD_STREAK"  # 0: never warn
DEFAULT_SOURCE_BAD_STREAK = 16


def _source_bad_streak_limit(environ: Mapping[str, str] | None = None) -> int:
    raw = (os.environ if environ is None else environ).get(SOURCE_BAD_STREAK_ENV, "").strip()
    try:
        return max(0, int(raw)) if raw else DEFAULT_SOURCE_BAD_STREAK
    except ValueError:
        return DEFAULT_SOURCE_BAD_STREAK


def _echo_event(event: str, **fields: Any) -> None:
    """One ``YETO_RL_EVENT`` stdout record from the rollout executor's process (it has
    no event tape of its own; the island's log stream carries the record)."""
    from yeto.rl.event_echo import format_record

    print(format_record({"island_id": fields.pop("island_id", None), "time_unix": time.time(),
                         "event": str(event), **fields}), flush=True)


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
    Volume with open files fails. Each payload read holds the store's reader
    guard (``CheckpointStore.hold_reader``), so a reload by another process of
    the container (the driver reading its policy releases) is skipped while a
    read is open.

    Faults of a payload. A payload that does not exist or is not a group (cut
    short, not JSON, another schema; :class:`DamagedPayload`,
    :func:`payload_is_missing`) is a fault of the data: reading it again gives
    the same answer, so the group is passed over like one without a finite reward
    (consumed, counted in ``TakeReport.damaged_payload_groups``, named in
    ``damaged_payloads``) and the round is filled from the groups after it. A
    read the store did not answer (a throttle, a 5xx, a timeout) is not: the take
    raises it. Damage is passed over one object at a time only: ``max_damaged_run``
    damaged payloads in a row, or that many and more than half of the payloads the
    take decoded, is a systematic fault (a sampler build that writes another
    schema, a reader that cannot decode), and the take raises
    :class:`DamagedPayloadLimit` rather than consume the backlog (the tasks are
    already settled in the dispenser and would never be generated again). A take
    that ends without a round (the data ended, no group arrived in time) names what
    it passed over in its error (``error.passed_over``, and the counts at the end
    of its text, ``buffer_payload.passed_over_of``) and logs the records.
    A take that raises returned nothing, so the groups it selected are
    kept (``_retry``) and the next take on this reader starts with them: taking
    again in place loses no group.

    The age gate reads the index record's ``behavior_version`` before the payload
    is read, and again the payload's own oldest version once it is decoded
    (:func:`payload_oldest_version`): a record that names a newer version than its
    payload holds cannot pass an over-age group (it is counted in ``stale_groups``
    and ``stale_payload_groups``). The second look only ever passes over more."""

    def __init__(self, buffer: Any, *, max_policy_age: int, wait_s: float = DEFAULT_WAIT_S,
                 poll_s: float = DEFAULT_POLL_S, consumer: str = TRAINER_CONSUMER,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 cursor_log: LaneCursorLog | ObjectLaneCursorLog | None = None,
                 read_workers: int = DEFAULT_READ_WORKERS, prefetch: bool = True,
                 incarnation_gate: Any = "auto", group_size: int | None = None,
                 max_damaged_run: int = DEFAULT_MAX_DAMAGED_RUN, one_group_per_task: bool = True) -> None:
        """``cursor_log``: a synced trainer's lane cursor (:mod:`yeto.rl.engine.merged_release`):
        the reader continues after its last line and appends one per round.
        ``read_workers`` < 2 reads the payloads one by one; ``prefetch`` needs a
        buffer with ``peek`` (the class doc). ``incarnation_gate``: an
        ``incarnation_lease.IncarnationGate``; a group a superseded sampler
        incarnation wrote after its successor claimed the lease (an NC node the pool
        fenced and replaced, still writing) is passed over unread
        (``TakeReport.stale_incarnation_groups``). ``"auto"`` builds it over the
        buffer's store (:func:`~yeto.rl.engine.incarnation_lease.default_incarnation_gate`,
        ``YETO_BUFFER_INCARNATION_FENCE=0`` turns it off); None: no gate.
        ``group_size``: the round must hold exactly ``groups * group_size`` samples
        (Miles' global batch). A short group (a sampler lost trials) is trained with
        its own samples (``TakeReport.incomplete_groups``); one with fewer than
        ``MIN_GROUP_SAMPLES`` or more than ``group_size`` is consumed, not trained.
        A round that falls short selects more whole groups while they fit; the first
        that does not is carried over to the next take (the round's end, hence the
        watermark, stays before it); what is still missing is reported as
        ``padded_samples`` (:func:`padding_samples` builds them). None: no check.
        ``max_damaged_run``: the damaged payloads at which a take raises (the class
        doc); 0: no limit. ``one_group_per_task``: a second group of a store-dispenser
        task is passed over (:class:`TaskSlots`, ``TakeReport.duplicate_task_groups``)."""
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
        self._stop_lock = threading.Lock()  # executor_set_mode vs reset/take
        # a convertible node's sampler stint (executor_set_mode): no read-ahead starts
        self._sampler_stint = False
        self._stop_prefetch = threading.Event()
        # xhw_guard actions {source: {"action", ...}} set by the driver after each step
        self.xhw_actions: dict[str, dict[str, Any]] = {}
        if cursor_log is not None:
            self.buffer.reset_to(cursor_log.last_cursor())
        if incarnation_gate == "auto":
            from yeto.rl.engine.incarnation_lease import default_incarnation_gate  # noqa: PLC0415

            incarnation_gate = default_incarnation_gate(buffer)
        self.gate = incarnation_gate
        self.group_size = None if group_size is None else int(group_size)
        # a whole group selected by a take that did not fit its round:
        # (position, decoded group, behavior version, tokens, source, dispense position,
        # store-dispenser task)
        self._carry: tuple[int, list[dict[str, Any]], int, int, str, int | None, Any] | None = None
        self.tasks = TaskSlots(buffer, self.gate) if one_group_per_task else None
        self._duplicates: list[dict[str, Any]] = []  # the take's duplicate task groups
        # selected entries the incarnation gate could not judge (their lease was unreadable):
        # (position, selected item), judged again by the next selections. Strict buffer
        # order: every entry behind the first of them waits too (any shard), so a round's
        # end (its committed watermark) is exactly that first position and a restore
        # trains nothing twice; an unreadable lease pauses the training
        self._deferred: list[tuple[int, Any]] = []
        self._nonfinite = 0
        self._nonfinite_lp = 0
        # the items the take under way selected, and those of a take that raised (they
        # come first in the next take: the buffer's read cursor is already past them)
        self._taken: list[Any] = []
        self._retry: list[Any] = []
        self._entry_of: dict[int, Any] = {}  # position -> index entry of the take's reads
        self._damaged: list[dict[str, Any]] = []  # the take's damaged payloads
        self._floor: int | None = None  # the take's age floor
        self.max_damaged_run = int(max_damaged_run)
        self._damaged_run = 0  # damaged payloads in a row, and the payloads decoded, of the take
        self._decoded_payloads = 0
        # dispense positions (None without one) of the take's groups whose payload is
        # older than the floor although their index record is not
        self._stale_payloads: list[int | None] = []
        # set_unidentified_gate: (start version K, the versions below K a record names)
        self._unidentified_gate: tuple[int, frozenset[int]] | None = None
        self._unidentified: dict[str, Any] = {"versions": {}, "positions": []}
        # selections in a row that yielded groups and kept none, at least one of them
        # unidentified: after UNIDENTIFIED_STARVED_SELECTS of them the reader warns
        # (the take goes on waiting: the samplers may still serve an older release)
        self.starved_selects_limit = _starved_selects_limit()
        self._starved_selects = 0
        self.unidentified_total = 0  # groups passed over as unidentified by this reader
        # per source: the bad groups in a row since its last good one, by reason
        self.source_bad_streak_limit = _source_bad_streak_limit()
        self._source_bad: dict[str, dict[str, int]] = {}
        self._rollout_id = 0
        self.emit: Callable[..., None] = _echo_event

    def set_unidentified_gate(self, start_version: int, known_versions: Any) -> None:
        """A relaunched trainer starts at vK and reads the buffer on from an earlier
        position. A group of a version below K that no record names any more (its release
        was pruned) cannot be given a policy identity: it is consumed and counted
        (``TakeReport.unidentified_policy_groups``), never trained, instead of
        stopping the trainer at every start. Versions from K on are this process's
        own publications; a version that is unknown there is still an error."""
        self._unidentified_gate = (int(start_version), frozenset(int(v) for v in known_versions))

    def _is_unidentified(self, entry: Any) -> bool:
        gate = self._unidentified_gate
        version = int(entry.behavior_version)
        if gate is None or version >= gate[0] or version in gate[1]:
            return False
        seen = self._unidentified
        seen["versions"][version] = seen["versions"].get(version, 0) + 1
        self.unidentified_total += 1
        position = _dispense_position(entry)
        if position is not None:
            seen["positions"].append(position)
        return True

    def set_xhw_actions(self, actions: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        self.xhw_actions = {str(k): dict(v) for k, v in (actions or {}).items()}
        return self.xhw_actions

    def _note_selection(self, rollout_id: int, selected: int, kept: int, unidentified: int) -> None:
        """Count selections that kept nothing while groups were passed over as
        unidentified; at ``starved_selects_limit`` in a row (and every further
        multiple of it) warn: the trainer is reading groups and can train none."""
        if not selected:
            return
        if kept or not unidentified:
            self._starved_selects = 0
            return
        self._starved_selects += 1
        limit = self.starved_selects_limit
        if limit > 0 and self._starved_selects % limit == 0:
            gate = self._unidentified_gate
            fields = {"rollout_id": int(rollout_id), "selections": self._starved_selects,
                      "unidentified_policy_groups_total": self.unidentified_total,
                      "start_version": None if gate is None else gate[0],
                      "known_versions": None if gate is None else sorted(gate[1]),
                      "buffer_position": int(self.buffer.read_cursor)}
            logging.getLogger(__name__).warning(
                "sample buffer: %d selections in a row kept no group (groups of versions no record "
                "names are passed over): %s", self._starved_selects, fields)
            self.emit(UNIDENTIFIED_STARVED_EVENT, **fields)

    def _note_source(self, source: str | None, reason: str | None) -> None:
        """One decoded group of ``source``: good (``reason`` None) or passed over for a
        fault of its data. The take's own bound (``max_damaged_run``) does not see one
        bad source among good ones (their groups reset its run); here each source
        counts its bad groups in a row, and at ``source_bad_streak_limit`` (and every
        further multiple) the reader warns, naming it. The run goes on."""
        if source is None:
            return
        if reason is None:
            self._source_bad.pop(source, None)
            return
        reasons = self._source_bad.setdefault(source, {})
        reasons[reason] = reasons.get(reason, 0) + 1
        bad = sum(reasons.values())
        limit = self.source_bad_streak_limit
        if limit > 0 and bad % limit == 0:
            fields = {"rollout_id": int(self._rollout_id), "source": source, "bad_groups": bad,
                      "reasons": dict(sorted(reasons.items()))}
            logging.getLogger(__name__).warning(
                "sample buffer: source %s wrote %d bad groups in a row (none trained): %s", source, bad, fields)
            self.emit(SOURCE_BAD_STREAK_EVENT, **fields)

    def _quarantined(self, source: str, rollout_id: int) -> bool:
        action = self.xhw_actions.get(source) or {}
        until = action.get("until_round")
        return action.get("action") == "quarantine" and (until is None or rollout_id < int(until))

    def _duplicate(self, entry: Any, position: int, *, unknown: Any = None) -> Any:
        """Whether an earlier group holds this entry's store-dispenser task; if so the
        group is recorded as passed over. An earlier group's lease that cannot be read
        (:class:`LeaseUnknown`, asking whether it was fenced) answers ``unknown``, or
        raises when ``unknown`` is None."""
        try:
            first = None if self.tasks is None else self.tasks.first(entry, int(position))
        except LeaseUnknown:
            if unknown is None:
                raise
            return unknown
        if first is None:
            return False
        earlier, here = first
        plan, slot = task_key(entry)
        attempt = (getattr(entry, "sampler_round", None) or {}).get("dispense_attempt")
        self._duplicates.append({
            "slot": slot, "position": int(position), "sample_id": entry.sample_id,
            "first_position": int(earlier), "first_sample_id": self.buffer.entries()[earlier].sample_id,
            "first_trained_here": bool(here), **({} if plan is None else {"plan": plan}),
            **({} if attempt is None else {"attempt": int(attempt)})})
        return True

    def _accept(self, entry: Any, position: int) -> None:
        if self.tasks is not None:
            self.tasks.accept(task_key(entry), int(position))

    def _select(self, limit: int) -> list[Any]:
        """The next ``limit`` items: first those a take that raised had selected."""
        out = self._retry[:max(0, int(limit))]
        del self._retry[:len(out)]
        if len(out) < int(limit) and not self._retry:
            out += self.buffer.select(self.consumer, int(limit) - len(out))
        self._taken += out
        return out

    def _cursor(self) -> int:
        """The position of the next item a take selects."""
        return int(self._retry[0].position) if self._retry else self.buffer.read_cursor

    def _group_at(self, position: int, read: Future) -> list[dict[str, Any]] | None:
        """The decoded group of a payload read; None for a payload that is missing or
        not a group (recorded in ``self._damaged``). A read the store failed raises."""
        entry = self._entry_of.get(int(position))
        ref = getattr(entry, "payload_ref", None)
        try:
            group = decode_group(read.result())
            self._damaged_run = 0
            self._decoded_payloads += 1
            return group
        except DamagedPayload as error:
            reason = str(error)
        except Exception as error:  # noqa: BLE001 - only a payload that is gone is passed over
            if ref is None or not payload_is_missing(self.buffer, ref, error):
                raise
            reason = "payload is missing"
        dispensed = _dispense_position(entry)
        self._damaged.append({"position": int(position), "sample_id": getattr(entry, "sample_id", None),
                              "payload_ref": ref, "reason": reason,
                              **({} if dispensed is None else {"dispense_position": dispensed})})
        self._damaged_run += 1
        limit, damaged = self.max_damaged_run, len(self._damaged)
        if limit > 0 and (self._damaged_run >= limit
                          or (damaged >= limit and damaged > self._decoded_payloads)):
            raise DamagedPayloadLimit(
                f"{damaged} damaged payloads in this take ({self._damaged_run} in a row, "
                f"{self._decoded_payloads} decoded), the last at buffer position {int(position)} "
                f"({getattr(entry, 'sample_id', None)}: {reason}); this is not one damaged object: check the "
                f"samplers' payload schema against this reader's ({PAYLOAD_SCHEMA}). Nothing was consumed; "
                f"{MAX_DAMAGED_RUN_ENV} (now {limit}, 0: no limit) sets the bound")
        return None

    def _ended(self, error: SampleBufferRolloutError, **counts: int) -> SampleBufferRolloutError:
        """``error`` of a take that ends without a round, with what the take passed over:
        the counts at the end of its text and in ``error.passed_over`` (with the
        records), and one log line."""
        counts = {**counts, "stale": counts.get("stale", 0) + len(self._stale_payloads),
                  "stale_payload": len(self._stale_payloads), "nonfinite_reward": self._nonfinite,
                  "nonfinite_logprob": self._nonfinite_lp,
                  "damaged_payload": len(self._damaged), "duplicate_task": len(self._duplicates)}
        counts = {key: int(value) for key, value in counts.items() if value}
        out = type(error)(f"{error}{passed_over_text(counts)}")
        out.passed_over = {**counts, "damaged_payloads": list(self._damaged),
                           "duplicate_tasks": list(self._duplicates)}
        if counts:
            print(f"[sample-buffer] take ended without a round: {json.dumps(out.passed_over, sort_keys=True)}",
                  flush=True)
        return out

    def _stale_payload(self, position: int, group: Sequence[Mapping[str, Any]]) -> bool:
        """Whether the decoded payload is older than the take's age floor (its index
        record was not: the two disagree); if so the group is recorded as stale."""
        if self._floor is None:
            return False
        oldest = payload_oldest_version(group)
        if oldest is None or oldest >= self._floor:
            return False
        self._stale_payloads.append(_dispense_position(self._entry_of.get(int(position))))
        return True

    def _read(self, ref: str, *, inline: bool = False) -> Future:
        if self._pool is None or inline:
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
        Waits for them while the samplers write: it raises only after ``wait_s``
        seconds in which no new group reached the buffer (a slow round that keeps
        arriving is waited for); after the data end returns what is left (raises
        when nothing is). When it raises, the groups it selected are selected again
        by the next take on this reader (the class doc)."""
        if int(groups) < 1:
            raise ValueError("groups must be positive")
        carried_in, deferred_in = self._carry, list(self._deferred)
        self._taken, self._entry_of, self._damaged, self._stale_payloads = [], {}, [], []
        self._damaged_run = self._decoded_payloads = 0
        self._duplicates = []
        try:
            return self._take(int(rollout_id), int(groups), current_version)
        except BaseException:
            self._retry = self._taken + self._retry
            self._carry = carried_in
            # the groups deferred by an earlier take wait again (those this take
            # selected and deferred are selected again from ``_retry``)
            self._deferred = deferred_in
            raise
        finally:
            self._taken = []

    def _take(self, rollout_id: int, groups: int, current_version: int | None
              ) -> tuple[list[tuple[int, list[dict[str, Any]]]], TakeReport]:
        started = progress = self.clock()
        self._rollout_id = rollout_id
        self._stop_prefetching()
        ahead = dict(self._reads)
        self._reads.clear()
        start = self._cursor()
        chosen: list[tuple[int, Future, bool]] = []  # (position, read, read ahead)
        versions: dict[int, int] = {}
        stale = 0
        stale_positions: list[int] = []
        quarantined = 0
        fenced = 0
        incomplete = 0
        decoded: dict[int, list[dict[str, Any]]] = {}
        accounted: dict[int, tuple[int, int]] = {}  # position -> (behavior version, tokens)
        source_groups: dict[str, int] = {}
        source_tokens: dict[str, int] = {}
        position_sources: dict[int, str] = {}
        select_s = read_s = decode_s = 0.0
        current_version = gate_version(current_version, self.gate_version)
        floor = None if current_version is None else age_floor(int(current_version), self.max_policy_age)
        self._floor = floor
        dropped_small = 0
        self._nonfinite = 0  # groups with a missing or non-finite reward (counted by the helpers)
        self._nonfinite_lp = 0  # groups with a non-finite behavior log-prob (likewise)
        self._unidentified = {"versions": {}, "positions": []}
        carried_in, self._carry = self._carry, None
        if self.tasks is not None:
            self.tasks.begin(start if carried_in is None else min(start, carried_in[0]))
        if carried_in is not None:  # selected by the last take; its position is that round's end
            position, group, version, tokens, source, dispensed, task = carried_in
            start = min(start, position)
            if floor is not None and version < floor:  # aged out while it waited: stale
                stale += 1
                if dispensed is not None:
                    stale_positions.append(dispensed)
            else:
                self._choose(position, group, version, tokens, source, chosen, decoded, accounted,
                             versions, source_groups, source_tokens, position_sources)
                if self.tasks is not None:  # judged when it was carried over: it holds its task now
                    self.tasks.accept(task, position)
        if self._deferred:
            start = min(start, self._deferred[0][0])
        deferred_now = 0
        lease_reads: dict[str, float] = {}  # shard -> when the recheck read its lease
        accounts = (decoded, accounted, versions, source_groups, source_tokens, position_sources)
        while True:
            if len(chosen) >= groups:
                # judged again right before the round is handed over: a fence that came
                # while the take waited for later groups keeps an earlier one out
                rejected, again_deferred = self._recheck(chosen, accounts, lease_reads)
                fenced += rejected
                deferred_now += again_deferred
                if not rejected and not again_deferred:
                    break
                continue
            # the reads of the last selection end before the next refresh (a reload)
            read_s += self._settle([read for _, read, _ in chosen])
            waiting, self._deferred = self._deferred, []
            again = [item for _, item in waiting]  # judged again first, in buffer order
            judged_before = {id(item) for item in again}
            kept_before, unidentified_before = len(chosen), self.unidentified_total
            selection: list[list[Any]] = [[]]

            def candidates() -> Any:
                """The deferred items, then (none still deferred) a selection of the
                groups still missing. Strict buffer order: once one is deferred (at p),
                every later one waits too, unjudged, so the round ends exactly at p."""
                nonlocal select_s
                for pending in again:
                    if len(chosen) >= groups:  # full: it waits for the next take
                        self._deferred.append((int(pending.position), pending))
                    else:
                        yield pending
                if len(chosen) < groups and not self._deferred:
                    begun = self.clock()
                    selection[0] = self._select(groups - len(chosen))
                    select_s += self.clock() - begun
                    yield from selection[0]

            for item in candidates():
                if self._deferred:  # behind a group whose lease is unreadable: it waits, unjudged
                    self._deferred.append((int(item.position), item))
                    deferred_now += 0 if id(item) in judged_before else 1
                    continue
                if floor is not None and item.entry.behavior_version < floor:
                    stale += 1
                    position = _dispense_position(item.entry)
                    if position is not None:
                        stale_positions.append(position)
                    continue
                # judged again too: an item deferred behind another was never judged (and the
                # check has no effect on an item it once let through)
                if self._is_unidentified(item.entry):
                    continue
                verdict = self._gate_verdict(item)
                if verdict is _DEFER:
                    deferred_now += 0 if id(item) in judged_before else 1
                    continue
                if verdict is not None:
                    fenced += 1  # a fenced incarnation's late write: never trained, never read
                    continue
                source = source_key(item.entry.provenance, self._shard_at(item.position))
                if self._quarantined(source, rollout_id):
                    quarantined += 1
                    continue
                duplicate = self._duplicate(item.entry, item.position, unknown=_DEFER)
                if duplicate is _DEFER:  # an earlier group's lease is unreadable: it waits (strict order)
                    self._deferred.append((int(item.position), item))
                    deferred_now += 0 if id(item) in judged_before else 1
                    continue
                if duplicate:
                    continue  # its task is trained from an earlier group: never read
                source_groups[source] = source_groups.get(source, 0) + 1
                source_tokens[source] = source_tokens.get(source, 0) + int(item.entry.tokens or 0)
                position_sources[int(item.position)] = source
                ref = item.entry.payload_ref
                read = ahead.pop(ref, None)
                if read is not None and read.exception() is not None:
                    read = None  # the read ahead failed: this take reads (and reports) on its own
                self._entry_of[int(item.position)] = item.entry
                chosen.append((item.position, read or self._read(ref), read is not None))
                versions[item.entry.behavior_version] = versions.get(item.entry.behavior_version, 0) + 1
                accounted[int(item.position)] = (item.entry.behavior_version, int(item.entry.tokens or 0))
            selected = selection[0]
            self._note_selection(rollout_id, len(selected), len(chosen) - kept_before,
                                 self.unidentified_total - unidentified_before)
            if selected or len(chosen) > kept_before:  # decode now (a deferred group judged again
                # included): a group that is dropped is replaced by the next selection
                read_s += self._settle([read for _, read, _ in chosen])
                begun = self.clock()
                dropped_small += self._drop_incomplete(chosen, decoded, accounted, versions, source_groups,
                                                       source_tokens, position_sources)
                decode_s += self.clock() - begun
            if selected:
                progress = self.clock()  # the samplers are writing: the wait starts again
            if len(chosen) >= groups or selected:
                continue
            if self._deferred:
                # groups wait for their lease (the gate fails closed and warns while it
                # cannot read it): neither the data end nor the no-new-group timeout
                # applies to them; they are judged again after the pause
                self._deferred.sort(key=lambda pending: pending[0])
                self.sleep(self.poll_s)
                continue
            if self.buffer.data_ended:
                rejected, again_deferred = self._recheck(chosen, accounts, lease_reads)
                fenced += rejected
                deferred_now += again_deferred
                if rejected or again_deferred:  # replaced if the buffer holds more; or waits
                    continue
                break
            if self.clock() - progress > self.wait_s:
                raise self._ended(SampleBufferRolloutError(
                    f"rollout {rollout_id}: {len(chosen)}/{groups} groups after {self.wait_s:g} s "
                    f"without a new group (buffer position {self._cursor()})"),
                    stale=stale, stale_incarnation=fenced, quarantined=quarantined, dropped_small=dropped_small)
            self.sleep(self.poll_s)
        self._deferred.sort(key=lambda pending: pending[0])
        end = self._cursor()
        carried_out = padded = 0
        if self.group_size is not None and chosen:
            skipped: dict[str, Any] = {"stale": 0, "stale_positions": [], "fenced": 0, "quarantined": 0,
                                       "nonfinite": 0, "nonfinite_logprob": 0}
            # with a group deferred, nothing behind it is selected: the round is padded
            end, carried_out, padded, more_small = self._fill(
                int(groups) * self.group_size, end, rollout_id, floor, chosen, decoded, accounted, versions,
                source_groups, source_tokens, position_sources, skipped, lease_reads=lease_reads)
            dropped_small += more_small
            stale += skipped["stale"]
            stale_positions += skipped["stale_positions"]
            fenced += skipped["fenced"]
            quarantined += skipped["quarantined"]
            self._nonfinite += skipped["nonfinite"]
            self._nonfinite_lp += skipped["nonfinite_logprob"]
            incomplete = sum(1 for position, _, _ in chosen if len(decoded[position]) < self.group_size)
            deferred_now += skipped.get("deferred", 0)
        if self._deferred:  # the watermark never passes a group the gate could not judge yet
            self._deferred.sort(key=lambda pending: pending[0])
            end = min(end, self._deferred[0][0])
        chosen.sort(key=lambda c: c[0])  # buffer order (a group judged again joins late)
        out = [(position, decoded[position]) for position, _, _ in chosen]
        stale += len(self._stale_payloads)
        stale_positions += [p for p in self._stale_payloads if p is not None]
        nonfinite = self._nonfinite
        nonfinite_lp = self._nonfinite_lp
        damaged = list(self._damaged)
        duplicates = list(self._duplicates)
        if not out:
            raise self._ended(SampleBufferExhausted(
                f"rollout {rollout_id}: the sample buffer ended with no trainable group (position {end}, "
                f"{stale} passed over as too old, {fenced} from fenced incarnations, {dropped_small} too "
                f"small, {nonfinite} without a finite reward, {nonfinite_lp} with a non-finite log-prob, "
                f"{len(damaged)} with a damaged payload, "
                f"{len(duplicates)} of a task already trained)"),
                stale=stale - len(self._stale_payloads), stale_incarnation=fenced, quarantined=quarantined,
                dropped_small=dropped_small)
        if self.tasks is not None:
            self.tasks.commit(end)
        self._ends[rollout_id] = end
        if self.cursor_log is not None:
            positions_of = getattr(self.buffer, "shard_positions", None)
            self.cursor_log.append(rollout_id, [position for position, _ in out], end,
                                   shards=positions_of(end) if callable(positions_of) else None)
        if self.prefetch and not self.buffer.data_ended:
            next_floor = None if current_version is None else floor + 1  # the next take's
            self._start_prefetching(int(groups), next_floor)
        # fenced writes the merge dropped since the last take (shard ownership records)
        take_dropped = getattr(self.buffer, "take_fenced_dropped", None)
        fenced_shards = list(take_dropped()) if callable(take_dropped) else []
        fenced += sum(int(d["groups"]) for d in fenced_shards)
        counters = getattr(self.buffer, "merge_counters", None)
        merge = dict(counters()) if callable(counters) else {}
        return out, TakeReport(stale_incarnation_shards=fenced_shards,
                               duplicate_merged_records=int(merge.get("duplicate_merged_records", 0)),
                               unmerged_shard_gaps=int(merge.get("unmerged_shard_gaps", 0)),
                               bad_shard_records=int(merge.get("bad_shard_records", 0)),
                               rollout_id=rollout_id, groups=len(out), stale_groups=stale,
                               start=start, end=end, behavior_versions=dict(sorted(versions.items())),
                               wait_s=round(self.clock() - started, 3), select_s=round(select_s, 3),
                               read_s=round(read_s, 3), decode_s=round(decode_s, 3),
                               prefetched=sum(1 for _, _, early in chosen if early),
                               stale_positions=stale_positions,
                               gate_version=None if current_version is None else int(current_version),
                               source_groups=dict(sorted(source_groups.items())),
                               source_tokens=dict(sorted(source_tokens.items())),
                               position_sources=position_sources, quarantined_groups=quarantined,
                               stale_incarnation_groups=fenced, lease_deferred_groups=deferred_now,
                               incomplete_groups=incomplete,
                               dropped_small_groups=dropped_small, carried_over_groups=carried_out,
                               padded_samples=padded, nonfinite_reward_groups=nonfinite,
                               nonfinite_logprob_groups=nonfinite_lp,
                               damaged_payload_groups=len(damaged), damaged_payloads=damaged,
                               stale_payload_groups=len(self._stale_payloads),
                               duplicate_task_groups=len(duplicates),
                               duplicate_task_unverified_groups=sum(
                                   1 for d in duplicates if not d["first_trained_here"]),
                               duplicate_tasks=duplicates,
                               unidentified_policy_groups=sum(self._unidentified["versions"].values()),
                               unidentified_policy_versions=dict(sorted(self._unidentified["versions"].items())),
                               unidentified_policy_positions=list(self._unidentified["positions"]))

    @staticmethod
    def _choose(position: int, group: list, version: int, tokens: int, source: str, chosen: list,
                decoded: dict, accounted: dict, versions: dict, source_groups: dict, source_tokens: dict,
                position_sources: dict) -> None:
        """Add an already decoded group to the take and its accounting."""
        done: Future = Future()
        done.set_result(None)
        chosen.append((position, done, False))
        decoded[position] = group
        versions[version] = versions.get(version, 0) + 1
        accounted[position] = (version, tokens)
        source_groups[source] = source_groups.get(source, 0) + 1
        source_tokens[source] = source_tokens.get(source, 0) + tokens
        position_sources[position] = source

    def _fill(self, target: int, end: int, rollout_id: int, floor: int | None, chosen: list, decoded: dict,
              accounted: dict, versions: dict, source_groups: dict, source_tokens: dict,
              position_sources: dict, skipped: dict | None = None,
              lease_reads: dict[str, float] | None = None) -> tuple[int, int, int, int]:
        """Top the chosen groups up to ``target`` samples with whole groups selected
        one by one, without waiting (what is not in the buffer yet is padded). The
        first group that would overflow is carried over (``self._carry``) and the
        round ends at its position, so the watermark never passes it untrained.
        Returns ``(end, carried groups, padded samples, groups dropped as too small)``; the
        groups passed over (too old, a fenced incarnation's, quarantined) are counted in
        ``skipped`` like the main selection counts them; a damaged payload is recorded
        by :meth:`_group_at`. A group joins the round as it is judged, so its shard's
        lease is read again unless the round's recheck read it just now (``lease_reads``)."""
        skipped = skipped if skipped is not None else {"stale": 0, "stale_positions": [], "fenced": 0,
                                                       "quarantined": 0}
        missing = target - sum(len(decoded[position]) for position, _, _ in chosen)
        dropped = 0
        while missing > 0 and not self._deferred:  # nothing joins behind a deferred group
            selected = self._select(1)
            if not selected:
                break
            item = selected[0]
            end = self._cursor()
            if floor is not None and item.entry.behavior_version < floor:
                skipped["stale"] += 1  # too old: consumed unread, as in the main selection
                dispensed = _dispense_position(item.entry)
                if dispensed is not None:
                    skipped["stale_positions"].append(dispensed)
                continue
            if self._is_unidentified(item.entry):
                continue
            verdict = self._gate_verdict(item, lease_reads)
            if verdict is _DEFER:  # strict order: nothing behind it joins this round
                skipped["deferred"] = skipped.get("deferred", 0) + 1
                break
            if verdict is not None:
                skipped["fenced"] += 1
                continue
            source = source_key(item.entry.provenance, self._shard_at(item.position))
            if self._quarantined(source, rollout_id):
                skipped["quarantined"] += 1
                continue
            duplicate = self._duplicate(item.entry, item.position, unknown=_DEFER)
            if duplicate is _DEFER:  # strict order, as for the gate's own verdict
                self._deferred.append((int(item.position), item))
                skipped["deferred"] = skipped.get("deferred", 0) + 1
                break
            if duplicate:
                continue
            self._entry_of[int(item.position)] = item.entry
            group = self._group_at(item.position, self._read(item.entry.payload_ref, inline=True))
            if group is None:
                self._note_source(source, "damaged_payload")
                continue
            if self._stale_payload(item.position, group):
                continue
            if not MIN_GROUP_SAMPLES <= len(group) <= int(self.group_size):
                self._note_source(source, "group_size")
                dropped += 1
                continue
            if not finite_rewards(group):
                self._note_source(source, "nonfinite_reward")
                skipped["nonfinite"] = skipped.get("nonfinite", 0) + 1
                continue
            if nonfinite_logprob(group) is not None:
                self._note_source(source, "nonfinite_logprob")
                skipped["nonfinite_logprob"] = skipped.get("nonfinite_logprob", 0) + 1
                continue
            self._note_source(source, None)
            position, version, tokens = int(item.position), item.entry.behavior_version, int(item.entry.tokens or 0)
            oldest = payload_oldest_version(group)
            if oldest is not None:  # what the next take's age gate compares if this group is carried over
                version = min(int(version), oldest)
            if len(group) > missing:
                # it holds its task when the next take chooses it
                self._carry = (position, group, version, tokens, source, _dispense_position(item.entry),
                               task_key(item.entry))
                return position, 1, missing, dropped
            self._choose(position, group, version, tokens, source, chosen, decoded, accounted, versions,
                         source_groups, source_tokens, position_sources)
            self._accept(item.entry, position)
            missing -= len(group)
        return end, 0, max(0, missing), dropped

    def _drop_incomplete(self, chosen: list, decoded: dict, accounted: dict, versions: dict, source_groups: dict,
                         source_tokens: dict, position_sources: dict) -> int:
        """Decode the chosen groups not yet decoded; drop (in place) those with a
        damaged payload (:meth:`_group_at`, counted apart) and, with a ``group_size``,
        those whose sample count is outside ``MIN_GROUP_SAMPLES..group_size`` or with a
        non-finite reward (:func:`finite_rewards`, counted apart), then those whose
        store-dispenser task an earlier group holds (:meth:`_duplicate`), and undo their
        accounting. Returns the number dropped for their size. Their positions stay
        consumed (the cursor moved past them)."""
        fresh = [(position, read) for position, read, _ in chosen if position not in decoded]
        if not fresh:
            return 0
        self._settle([read for _, read in fresh])
        drop: set[int] = set()
        nonfinite = nonfinite_lp = damaged = duplicate = 0
        for position, read in fresh:
            group = self._group_at(position, read)
            if group is not None and self._stale_payload(position, group):
                group, aged = None, True
            else:
                aged = False
            sized = group is not None and (
                self.group_size is None or MIN_GROUP_SAMPLES <= len(group) <= int(self.group_size))
            bad_lp = sized and nonfinite_logprob(group) is not None
            good = sized and not bad_lp and (self.group_size is None or finite_rewards(group))
            if not aged:  # a group older than the floor says nothing about its source's data
                self._note_source(position_sources.get(int(position)), None if good else (
                    "damaged_payload" if group is None else "nonfinite_logprob" if bad_lp
                    else "nonfinite_reward" if sized else "group_size"))
            if good:
                # in buffer order, after the payload checks: an earlier group of this
                # selection that is trained holds the task; one just dropped does not
                entry = self._entry_of.get(int(position))
                # (judged as no duplicate at its selection: an unreadable lease keeps that)
                if entry is None or not self._duplicate(entry, position, unknown=False):
                    decoded[position] = group
                    if entry is not None:
                        self._accept(entry, position)
                    continue
                duplicate += 1
            elif aged:
                damaged += 1  # not a size drop: counted in the take's stale groups
            elif group is None:
                damaged += 1
            elif bad_lp:
                nonfinite_lp += 1
            elif sized:
                nonfinite += 1
            drop.add(position)
            self._unchoose(position, decoded, accounted, versions, source_groups, source_tokens,
                           position_sources)
        if drop:
            chosen[:] = [c for c in chosen if c[0] not in drop]
        self._nonfinite += nonfinite
        self._nonfinite_lp += nonfinite_lp
        return len(drop) - nonfinite - nonfinite_lp - damaged - duplicate

    def _recheck(self, chosen: list, accounts: tuple, lease_reads: dict[str, float]) -> tuple[int, int]:
        """Judge the ``chosen`` groups again, each shard's lease read again (at most
        once per shard while the take does not wait, :meth:`IncarnationGate.recheck`):
        a group fenced since its selection is dropped (counted stale, its task free:
        a group this take passed over for that task is judged again); from the first
        one whose lease is unreadable on, every group waits in ``self._deferred``
        (strict order), judged again. Returns ``(dropped, deferred)``."""
        recheck = getattr(self.gate, "recheck", None)
        if not chosen or not callable(recheck):
            return 0, 0
        entries = None
        items: list[tuple[int, Any]] = []
        for position, _, _ in chosen:
            entry = self._entry_of.get(int(position))
            if entry is None:  # carried over from the last take
                entries = self.buffer.entries() if entries is None else entries
                entry = entries[int(position)]
            items.append((int(position), entry))
        items.sort(key=lambda pair: pair[0])
        verdicts = recheck([(entry, position) for position, entry in items], lease_reads)
        drop: set[int] = set()
        defer_from = None
        for (position, entry), verdict in zip(items, verdicts):
            if isinstance(verdict, LeaseUnknown):
                defer_from = position
                break
            if verdict is not None:
                drop.add(position)
        waiting = [] if defer_from is None else [(p, e) for p, e in items if p >= defer_from]
        removed = drop | {position for position, _ in waiting}
        # a group passed over by this take because a group now removed held its task is
        # judged again (the task is free, or waits with the removed group)
        freed = [d for d in self._duplicates if d["first_trained_here"] and d["first_position"] in removed]
        if freed:
            self._duplicates = [d for d in self._duplicates if d not in freed]
            entries = self.buffer.entries() if entries is None else entries
            waiting += [(int(d["position"]), entries[int(d["position"])]) for d in freed]
        for position, entry in waiting:
            self._deferred.append((position, Selected(position=position, entry=entry)))
        self._deferred.sort(key=lambda pending: pending[0])
        if removed:
            chosen[:] = [c for c in chosen if int(c[0]) not in removed]
            for position in removed:
                self._unchoose(position, *accounts)
                if self.tasks is not None:
                    self.tasks.release(int(position))
        return len(drop), len(waiting) - len(freed)

    @staticmethod
    def _unchoose(position: int, decoded: dict, accounted: dict, versions: dict, source_groups: dict,
                  source_tokens: dict, position_sources: dict) -> None:
        """Undo the accounting of a chosen group (it is not in the round)."""
        decoded.pop(int(position), None)
        version, tokens = accounted.pop(int(position))
        versions[version] -= 1
        if not versions[version]:
            del versions[version]
        source = position_sources.pop(int(position), None)
        if source is not None:
            source_groups[source] -= 1
            source_tokens[source] -= tokens
            if not source_groups[source]:
                del source_groups[source]
                source_tokens.pop(source, None)

    def _gate_verdict(self, item: Any, lease_reads: dict[str, float] | None = None) -> Any:
        """The incarnation gate's verdict on a selected ``item``: None (train it), the
        stale detail, or :data:`_DEFER` when its lease could not be read (the item is
        kept in ``self._deferred``, neither trained nor counted stale). With
        ``lease_reads`` (a group trained without a later recheck) the lease is read
        again first (:meth:`IncarnationGate.recheck`)."""
        if self.gate is None:
            return None
        recheck = getattr(self.gate, "recheck", None)
        if lease_reads is not None and callable(recheck):
            [verdict] = recheck([(item.entry, int(item.position))], lease_reads)
            if isinstance(verdict, LeaseUnknown):
                self._deferred.append((int(item.position), item))
                return _DEFER
            return verdict
        try:
            return self.gate.verdict(item.entry, int(item.position))
        except LeaseUnknown:
            self._deferred.append((int(item.position), item))
            return _DEFER

    def _shard_at(self, position: int) -> str | None:
        shard_at = getattr(self.buffer, "shard_at", None)
        return shard_at(int(position)) if callable(shard_at) else None

    def _settle(self, reads: list[Future]) -> float:
        """Wait for ``reads`` (their errors surface at ``result()``); the seconds waited."""
        pending = [read for read in reads if not read.done()]
        if not pending:
            return 0.0
        begun = self.clock()
        # no deadline: a read that never ends has no safe outcome here (the caller
        # needs its payload, and a refresh must not overlap it), so warn and keep waiting
        while True:
            _, pending = wait(pending, timeout=SETTLE_WARN_S)
            if not pending:
                return self.clock() - begun
            print(f"[sample-buffer] {len(pending)} payload read(s) still open after "
                  f"{self.clock() - begun:.0f}s; still waiting", flush=True)

    # ------------------------------------------------------------ prefetch
    def _start_prefetching(self, groups: int, floor: int | None) -> None:
        # under the stop lock: a take that ends after ``executor_set_mode(SAMPLER)``
        # stopped the read-ahead must not start it again in the sampler stint
        with self._stop_lock:
            if self._sampler_stint:
                return
            self._stop_prefetch.clear()
            self._prefetcher = threading.Thread(target=self._prefetch_loop, args=(groups, floor),
                                                name="buffer-prefetch", daemon=True)
            self._prefetcher.start()

    def _set_sampler_stint(self, sampler: bool) -> None:
        """Mark the convertible node's stint; entering a sampler stint stops the read-ahead."""
        with self._stop_lock:
            self._sampler_stint = bool(sampler)
        if sampler:  # takes the stop lock itself (not reentrant)
            self._stop_prefetching()

    def _stop_prefetching(self) -> None:
        """Stop the prefetch thread and wait for its reads (so a refresh may reload)."""
        # ``executor_set_mode`` and ``reset``/``take`` may stop from different threads:
        # take the thread out under the lock, and finish the stop before the next caller
        with self._stop_lock:
            prefetcher, self._prefetcher = self._prefetcher, None
            if prefetcher is not None:
                self._stop_prefetch.set()
                prefetcher.join()
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

    def trained_before(self, rollout_id: int) -> int:
        """The buffer position the policy of version ``rollout_id`` is trained up to:
        the end of the last round below ``rollout_id`` this reader took, or the
        position it started from (the last committed watermark) when it took none.
        Unlike ``buffer.read_cursor`` it does not move with a round selected later
        or a group carried over to the next round."""
        done = [rid for rid in self._ends if rid < int(rollout_id)]
        return int(self._ends[max(done)]) if done else int(self.buffer.watermark)

    def reset(self, watermark: int) -> int:
        """Continue at the restored cut's watermark (rounds after it are selected again)."""
        # the read-ahead of the last take: its reads end before the restore's refresh
        # (a Volume reload), and it reads ahead of a position that is abandoned now
        self._stop_prefetching()
        self._reads.clear()
        self.buffer.reset_to(int(watermark))
        self._ends.clear()
        self._carry = None  # its position is at or past the watermark: selected again
        self._retry = []  # so are the groups of a take that raised
        self._deferred = []  # likewise
        if self.tasks is not None:
            self.tasks.reset(int(watermark))
        return int(self.buffer.watermark)


# ---------------------------------------------------------------------- executor link
_ACTIVE: dict[str, BatchReader] = {}
# the mode of a convertible node's rollout function (``ConvertibleRolloutFn``):
# SAMPLER generates into the node's shard, TRAINER reads the buffer
_MODE: dict[str, str] = {CONVERTIBLE: SAMPLER}


def r3_layer_layout(hf_checkpoint: str | None) -> tuple[int, int] | None:
    """``(num_hidden_layers, first_k_dense_replace)`` of the checkpoint's (text) config,
    or None when there is no readable config.json (R3 is then passed to Miles as is)."""
    import json
    from pathlib import Path

    if not hf_checkpoint or not (Path(hf_checkpoint) / "config.json").is_file():
        return None
    config = json.loads((Path(hf_checkpoint) / "config.json").read_text())
    text = config.get("text_config") or config
    layers = text.get("num_hidden_layers")
    if layers is None:
        return None
    return int(layers), int(text.get("first_k_dense_replace") or 0)


def r3_num_experts(hf_checkpoint: str | None) -> int | None:
    """The routed expert count of the checkpoint's (text) config, or None."""
    import json
    from pathlib import Path

    if not hf_checkpoint or not (Path(hf_checkpoint) / "config.json").is_file():
        return None
    config = json.loads((Path(hf_checkpoint) / "config.json").read_text())
    text = config.get("text_config") or config
    value = text.get("n_routed_experts", text.get("num_experts"))
    return None if value is None else int(value)


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
    reader = _ACTIVE.get(TRAINER)
    if reader is not None:
        # the trainer stint is over: its read-ahead would poll the index (a Volume
        # reload every poll) through the rollout and the next stint's start, whose
        # store writes then fail on the remount (EPERM, seen on GPU); a take still
        # ending on another thread does not start it again
        reader._set_sampler_stint(mode == SAMPLER)
    return previous


def reader_basis_version(weight_version: int | None, policy_token: str | None) -> int | None:
    """The version a trainer's ``BatchReader.take`` measures policy age from.

    The driver accepts the round against its ``published_version`` and hands
    the rollout side exactly that version in the policy token
    (``yeto:<published_version>:<hash>``, ``set_policy_token``); the reader
    uses the same basis (:func:`yeto.rl.engine.policy_age.age_floor`). Miles'
    ``RolloutFnInput.weight_version`` is only the fallback without a token: it
    can lag the publication by one (the round's publication is announced to
    Miles after the take), so a group the reader kept as exactly
    ``max_policy_age`` old was one version past the driver's window."""
    if policy_token:
        from yeto.rl.engine.policy_age import token_version

        version = token_version(policy_token)
        if version is None:
            raise SampleBufferRolloutError(f"malformed driver policy token {policy_token!r}")
        return version
    return None if weight_version is None else int(weight_version)


def register_reader(reader: BatchReader) -> None:
    """Called by the reader rollout function in the rollout executor's process."""
    _ACTIVE[TRAINER] = reader
    unidentified = _PENDING_UNIDENTIFIED_GATE.pop(TRAINER, None)
    if unidentified is not None:
        reader.set_unidentified_gate(*unidentified)
    pending = _PENDING_RELAY_GATE.pop(TRAINER, None)
    if pending is not None:  # relay mode only (executor_set_relay_gate_version)
        reader.gate_version = pending


# A relay trainer's gate version set before its reader exists (the first
# publication precedes the first rollout call, which builds the reader).
_PENDING_RELAY_GATE: dict[str, int] = {}
# Relay mode: the round and the gate version of the last take, for the
# trajectory evidence of that round (full_parameter.evidence). Rollout-process
# state under this key (adapters.miles.proc_state): upstream hands rollout
# functions and hooks a frozen view of the args.
EXTERNAL_CUT_ROUND_ATTR = "yeto_rl_external_cut_round"


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


# set before the reader exists (the driver starts before the first rollout call builds it)
_PENDING_UNIDENTIFIED_GATE: dict[str, tuple[int, list[int]]] = {}


def executor_set_unidentified_gate(_executor: Any, start_version: int, known_versions: list[int]) -> bool:
    """Runs INSIDE the rollout executor actor: :meth:`BatchReader.set_unidentified_gate`
    (kept for the reader when it does not exist yet). True: set on the reader now."""
    reader = _ACTIVE.get(TRAINER)
    if reader is None:
        _PENDING_UNIDENTIFIED_GATE[TRAINER] = (int(start_version), [int(v) for v in known_versions])
        return False
    reader.set_unidentified_gate(int(start_version), known_versions)
    return True


def executor_trained_before(_executor: Any, rollout_id: int) -> int | None:
    """Runs INSIDE the rollout executor actor: :meth:`BatchReader.trained_before`
    (None before the reader exists)."""
    reader = _ACTIVE.get(TRAINER)
    return None if reader is None else reader.trained_before(int(rollout_id))


def executor_read_shards(_executor: Any) -> dict[str, int] | None:
    """Runs INSIDE the rollout executor actor: the read cursor per sampler shard
    (None before the reader exists or when the buffer has no merged shards)."""
    reader = _ACTIVE.get(TRAINER)
    if reader is None:
        return None
    return reader.buffer.shard_positions(reader.buffer.read_cursor) or None


def executor_set_relay_gate_version(_executor: Any, version: int) -> int | None:
    """:func:`executor_set_gate_version` of a relay trainer: a gate set
    before the reader exists is kept and applied when the reader registers, so
    the first take already gates on the merged version."""
    if _ACTIVE.get(TRAINER) is None:
        _PENDING_RELAY_GATE[TRAINER] = int(version)
        return None
    return executor_set_gate_version(_executor, version)


# A merged-lane synced trainer (not its own lane, not relay): merged release
# version -> the release's policy hash, pushed by the driver at every publish
# (executor_set_merged_releases). Its groups were generated by merged releases,
# not by this island's publications, so the round's evidence names each
# trajectory's release as an external merged cut (yeto.rl.engine.external_policy;
# digest = the release's policy hash) and counts its age in merged versions.
_MERGED_RELEASES: dict[int, str] = {}
# where the rollout process re-reads the release records on a miss: a release
# can reach a sampler (and its groups the buffer) before the driver's next push
_MERGED_RELEASE_STORE: dict[str, str] = {}
_MERGED_MODE: dict[str, bool] = {}  # set by the first push, also before any release exists


def executor_set_merged_releases(_executor: Any, releases: Mapping[int, str],
                                 gate_version: int, store: str | None = None,
                                 release_dir: str | None = None) -> int:
    """Runs INSIDE the rollout executor actor: record the merged releases the
    driver knows and set the merged gate (kept until the reader exists, like the
    relay gate). Returns how many releases are known."""
    for version, policy_hash in releases.items():
        _MERGED_RELEASES[int(version)] = str(policy_hash)
    if store and release_dir:
        _MERGED_RELEASE_STORE.update(store=str(store), release_dir=str(release_dir))
    _MERGED_MODE[TRAINER] = True
    executor_set_relay_gate_version(_executor, int(gate_version))
    return len(_MERGED_RELEASES)


def _refresh_merged_releases() -> None:
    if not _MERGED_RELEASE_STORE:
        return
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.resume import store_for

    releases = PolicyReleaseStore(store_for(_MERGED_RELEASE_STORE["store"]),
                                  release_dir=_MERGED_RELEASE_STORE["release_dir"])
    for record in releases.records():
        if record.derived_from is None and record.policy_hash:
            _MERGED_RELEASES[int(record.version)] = str(record.policy_hash)


def merged_release_mode() -> bool:
    return bool(_MERGED_MODE.get(TRAINER))


def stamp_merged_release_cuts(group: list[dict[str, Any]]) -> bool:
    """Name every sample's oldest behavior version as the merged release cut that
    generated it (sample metadata :data:`~yeto.rl.engine.external_policy.CUT_METADATA_KEY`).

    Returns False, stamping nothing, for a group of a release the store already
    pruned before this trainer saw it (a relaunched trainer's lane backlog: the
    store keeps the newest few releases, the age gate may be wider): its cut
    cannot be named, so the caller passes it over as too old. A release newer
    than every known one is not visible yet and still raises."""
    from yeto.rl.engine.buffer_payload import behavior_versions
    from yeto.rl.engine.external_policy import CUT_METADATA_KEY, cut_record

    cuts = []
    for sample in group:
        oldest = min(behavior_versions(sample))
        if oldest not in _MERGED_RELEASES:
            _refresh_merged_releases()
        policy_hash = _MERGED_RELEASES.get(oldest)
        if policy_hash is None:
            later = [v for v in _MERGED_RELEASES if v > 0]
            if oldest > 0 and later and oldest < min(later):
                return False
            raise SampleBufferRolloutError(
                f"buffered group of merged release v{oldest}: that release is not known to "
                f"this trainer (known: {sorted(_MERGED_RELEASES)})")
        cuts.append((sample, cut_record(oldest, policy_hash)))
    for sample, cut in cuts:
        meta = sample.setdefault("metadata", {})
        if not isinstance(meta, dict):
            raise SampleBufferRolloutError("buffered sample metadata is not an object")
        meta[CUT_METADATA_KEY] = cut
    return True


def xhw_round(groups: Sequence[tuple[int, list[dict[str, Any]]]], report: TakeReport,
              actions: Mapping[str, Mapping[str, Any]], *, std_normalized: bool,
              enabled: bool = True) -> tuple[list[tuple[int, list[dict[str, Any]]]], dict[str, Any]]:
    """The per-source side of one take (yeto.rl.engine.xhw_guard): the groups with the
    ``downweight`` actions applied, and the round metadata the driver's guard reads
    (``buffer_source_tokens``, ``buffer_source_stats``, ``buffer_xhw_quarantined``).
    Downweighting scales the centered rewards, which is the advantage scale only
    without GRPO std normalization; with it the action is refused. ``enabled`` False
    (``YETO_RL_XHW_GUARD`` unset): the groups unchanged and no metadata."""
    if not enabled:
        return list(groups), {}
    out, by_source = [], {}
    downweighted = 0
    for position, group in groups:
        source = getattr(report, "position_sources", {}).get(int(position))
        action = (actions.get(source) or {}) if source is not None else {}
        if action.get("action") == "downweight":
            if std_normalized:
                raise SampleBufferRolloutError(
                    "xhw_guard downweight scales the centered rewards; GRPO std normalization "
                    "cancels it (use action=quarantine, or turn std normalization off)")
            group = downweight_group(group, float(action.get("weight", 1.0)))
            downweighted += 1
        by_source.setdefault(source or "unknown", []).extend(group)
        out.append((position, group))
    meta = {"buffer_source_tokens": {k: int(v) for k, v in getattr(report, "source_tokens", {}).items()},
            "buffer_source_stats": {k: {**rollout_stats(v), "groups": report.source_groups.get(k, 0)}
                                    for k, v in sorted(by_source.items())},
            "buffer_xhw_quarantined": int(report.quarantined_groups),
            "buffer_xhw_downweighted": downweighted}
    return out, meta


def check_xhw_downweight(*, std_normalized: bool, environ: Mapping[str, str] | None = None) -> None:
    """At the reader's start: ``YETO_RL_XHW_GUARD`` with ``action=downweight`` under GRPO
    std normalization (Miles' default) is refused here, not at the first round the guard
    trips (:func:`xhw_round` cannot apply it then, and the trainer would stop mid-run)."""
    from yeto.rl.engine.xhw_guard import XhwGuard  # noqa: PLC0415

    guard = XhwGuard.from_env(os.environ if environ is None else environ)
    if guard is not None and guard.action == "downweight" and std_normalized:
        raise SampleBufferRolloutError(
            "YETO_RL_XHW_GUARD action=downweight scales the centered rewards; GRPO std "
            "normalization (on) cancels it: use action=quarantine, or turn std normalization off")


def executor_set_xhw_actions(_executor: Any, actions: Mapping[str, Mapping[str, Any]]) -> dict | None:
    """Runs INSIDE the rollout executor actor: the xhw_guard actions the next takes
    follow (None before the reader exists)."""
    reader = _ACTIVE.get(TRAINER)
    return None if reader is None else reader.set_xhw_actions(actions)


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
