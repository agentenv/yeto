"""The trainer's sample-buffer reader, shared by every RL backend.

A trainer island over the durable sample buffer (:mod:`yeto.rl.engine.sample_buffer`)
selects its rounds here, whatever framework trains them: Miles
(``adapters.miles.buffer_rollout``, which re-exports this module), verl
(``adapters.verl.buffer_feed`` uses the judgments and the breaker reports below) and
ms-swift (``yeto.swift_megatron.rl_buffer``).

What lives here, so that every backend judges a group the same way and reports the same
``trainer.*`` breakers (``yeto.breakers.layers.trainer``, owner of those entries):

* opening the trainer's buffer (:func:`open_buffer`, shard merge, lanes);
* :class:`BatchReader`: the round selection with its policy-age gate
  (``trainer.policy_age``), the incarnation gate, one group per dispenser task, the
  damaged-payload bound (``trainer.buffer_damaged``), group integrity
  (``trainer.group_integrity``) and the wait for new groups (``trainer.sample_wait``);
* the round judgments a backend applies after conversion: groups that do not convert
  to the trainer's form (``trainer.sample_alignment``, :func:`report_unaligned`,
  :func:`unconvertible_round`) and rounds of compacted trials it cannot train whole
  (``trainer.compaction_round``, :func:`check_compaction_round`).

The backend only turns a selected group into its own inputs. This module imports no
training framework.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.engine import buffer_payload as _payload_format
from yeto.rl.engine import trainer_breakers
from yeto.rl.engine.buffer_payload import (
    PAYLOAD_SCHEMA,
    DamagedPayload,
    DamagedPayloadLimit,
    SampleBufferExhausted,
    SampleBufferRolloutError,
    compaction_meta,
    decode_group,
    finite_rewards,
    has_siblings,
    nonfinite_logprob,
    passed_over_text,
    payload_is_missing,
    payload_oldest_version,
    trial_count,
    trial_rows,
)
from yeto.rl.engine.incarnation_lease import LeaseUnknown
from yeto.rl.engine.merged_release import (
    LANE_ENV,
    BufferLane,
    LaneCursorLog,
    ObjectLaneCursorLog,
    gate_version,
    lane_cursor_log,
    merges_shards,
)
from yeto.rl.engine.policy_age import age_floor
from yeto.rl.engine.trainer_lifecycle import Selected
from yeto.rl.engine.xhw_guard import source_key

BUFFER_ROOT_ENV = "YETO_SAMPLE_BUFFER_ROOT"  # checkpoint store path holding sample-buffer/
SAMPLER_ID_ENV = "YETO_SAMPLE_BUFFER_SAMPLER_ID"  # prefix of the sample ids a sampler writes
WAIT_ENV = "YETO_SAMPLE_BUFFER_WAIT_S"  # reader: max seconds without a new group while a round fills
POLL_ENV = "YETO_SAMPLE_BUFFER_POLL_S"  # reader: seconds between index reads while waiting
# reader: seconds between rl_sample_wait records while a take waits for new groups (the
# launcher sets it from --rl-stall-timeout, trainer_breakers.wait_heartbeat_env)
HEARTBEAT_ENV = trainer_breakers.WAIT_HEARTBEAT_ENV
DEFAULT_HEARTBEAT_S = trainer_breakers.MIN_WAIT_HEARTBEAT_S
WAIT_EVENT = "rl_sample_wait"
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
# trainer: its learner id, the island_id of the records its rollout process echoes
# (rl_sample_wait, the reader's breaker trips): a fleet bills them to their island
ISLAND_ID_ENV = "YETO_SAMPLE_BUFFER_ISLAND_ID"

DEFAULT_WAIT_S = 3600.0
DEFAULT_POLL_S = 2.0
TRAINER_CONSUMER = "trainer"
# reader: while a take waits for groups, the trainer's read heartbeat (where it reads, the
# groups it passed over included) is written again at least this often (seconds)
DEFAULT_READ_HEARTBEAT_S = 30.0
READ_HEARTBEAT_FAILED_EVENT = "rl_buffer_read_position_failed"

_DEFER = object()  # BatchReader._gate_verdict: the lease could not be read


# ---------------------------------------------------------------------- the trainer's buffer
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

# ---------------------------------------------------------------------- round judgments
def segment_row_width(meta: Any) -> tuple[int, str] | None:
    """``(tokens, what)``: how wide one compaction segment's row can be. A segment can
    grow past ``context_budget`` to the engine context (a large tool output after the last
    step below the trigger, then the summary): the sampler's ``max_model_len`` when the
    segment carries it, else (older samples) ``context_budget``. None: neither is known."""
    for name in ("max_model_len", "context_budget"):
        value = meta.get(name)
        if type(value) is int and value > 0:
            return value, ("the sampler's max_model_len" if name == "max_model_len" else "context budget")
    return None


def check_compaction_round(groups: Sequence[tuple[int, Sequence[dict[str, Any]]]], *, seq_length: Any,
                           schedule_refusal: Callable[[list[int]], str | None] | None = None,
                           seq_flag: str = "--seq-len", emit: Callable[..., Any] | None = None) -> None:
    """Refuse a round (its converted ``groups``) with multi-sample trials (compaction
    segments) that this trainer cannot train whole (``trainer.compaction_round``):
    ``schedule_refusal(positions)`` names why the backend would not train them one rollout
    per trial (None: it would); segments that can be wider than the trainer's
    ``seq_length`` (:func:`segment_row_width`: the sampler's ``max_model_len``, else
    ``context_budget``; its summary, at the end, would be cut); then the sibling check of
    the loss side (``yeto.rl.algos.segment_weights.sibling_problems``) over all the
    round's samples. A round without such trials: nothing is checked."""
    every = groups
    groups = [(position, group) for position, group in every if has_siblings(group)]
    if not groups:
        return
    positions = [position for position, _ in groups]
    refusal = None if schedule_refusal is None else schedule_refusal(positions)
    if refusal:
        raise refuse_compaction_round(refusal, emit=emit)
    widths = [segment_row_width(meta) for _, group in groups for d in group
              if (meta := compaction_meta(d)) is not None]
    widths = [w for w in widths if w is not None]
    if widths and (not isinstance(seq_length, int) or seq_length < max(w for w, _ in widths)):
        width, what = max(widths)
        raise refuse_compaction_round(
            f"groups at buffer positions {positions[:8]} hold compaction segments that can reach {width} "
            f"tokens ({what}); this trainer's --seq-length {seq_length} does not hold one (a segment "
            f"ends with its summary, which would be cut): raise {seq_flag} to {width}", emit=emit)
    from yeto.rl.algos.segment_weights import sibling_problems  # noqa: PLC0415

    problems = sibling_problems([d for _, group in every for d in group])
    if problems:
        raise refuse_compaction_round("compaction segments are not rollout siblings: " + "; ".join(problems),
                                      emit=emit)


def refuse_compaction_round(message: str, *, emit: Callable[..., Any] | None = None) -> SampleBufferRolloutError:
    """A round of compacted trials this trainer cannot train whole (``trainer.compaction_round``);
    the caller raises the error returned."""
    error = SampleBufferRolloutError(message)
    trainer_breakers.trip(trainer_breakers.COMPACTION_ROUND, _echo_event if emit is None else emit, detail=message,
                          legacy_event="SampleBufferRolloutError")
    return error


def stale_segment_buckets(report: Any, pruned: Sequence[Sequence[dict[str, Any]]] = (), *,
                          compacted: bool = False) -> dict[str, int]:
    """The round's stale trials by segment count (``rl_buffer_stale_skipped.stale_by_segments``):
    the take's (``TakeReport.stale_by_segments``) plus the groups of pruned merged
    releases (counted in the round's stale groups too). {} unless compaction shows in the
    round (a trained or a stale group held segments): the default record is unchanged."""
    from yeto.rl.engine.timeline import segment_bucket  # noqa: PLC0415

    counts = dict(getattr(report, "stale_by_segments", None) or {})
    for group in pruned:
        compacted = compacted or has_siblings(group)
        for trial in trial_rows(group):
            bucket = segment_bucket(len(trial))
            counts[bucket] = counts.get(bucket, 0) + 1
    if not (compacted or getattr(report, "stale_compacted", False)):
        return {}
    return {bucket: int(n) for bucket, n in sorted(counts.items()) if n}


def compaction_batch_line(rollout_id: int, groups: Sequence[Sequence[dict[str, Any]]]) -> str:
    """The trainer log line of a round with compacted trials: its trials (rollouts), its
    samples, and how many trials have 1, 2, ... samples (every sample of a trial is in
    this round, so Miles trains them in one step)."""
    sizes = [len(trial) for group in groups for trial in trial_rows(group)]
    per_trial: dict[int, int] = {}
    for size in sorted(sizes):
        per_trial[size] = per_trial.get(size, 0) + 1
    histogram = ",".join(f"{k}:{v}" for k, v in per_trial.items())
    return (f"[sample-buffer] compaction_batch: rollout={int(rollout_id)} trials={len(sizes)} "
            f"segments={sum(sizes)} per_trial={{{histogram}}}")


def report_unaligned(problems: Sequence[str], *, emit: Callable[..., Any] | None = None) -> None:
    """Groups of a take left out of the round because their samples do not convert to
    the trainer's form (``trainer.sample_alignment``, action drop): one warning line per
    group and one breaker record for the take. Nothing when ``problems`` is empty."""
    for problem in problems:
        print(f"[sample-buffer] WARNING: group left out of the round: {problem}", flush=True)
    if problems:
        trainer_breakers.trip(trainer_breakers.SAMPLE_ALIGNMENT, _echo_event if emit is None else emit,
                              count=len(problems), detail=problems[-1])


def unconvertible_round(rollout_id: int, takes: int, problem: str, *, form: str = "the trainer's samples",
                        emit: Callable[..., Any] | None = None) -> SampleBufferRolloutError:
    """The error that ends a round after ``takes`` takes in a row without a group that
    converts to ``form``, reported as a ``trainer.sample_alignment`` stop."""
    error = SampleBufferRolloutError(f"rollout {rollout_id}: {takes} takes in a row with no group that "
                                     f"converts to {form}: {problem}")
    trainer_breakers.trip(trainer_breakers.SAMPLE_ALIGNMENT, _echo_event if emit is None else emit, action="stop",
                          value=takes, limit=takes, detail=str(error), legacy_event="SampleBufferRolloutError")
    return error


def trip_sample_wait(emit: Callable[..., Any] | None, *, waited_s: float, limit_s: float,
                     error: BaseException) -> None:
    """``trainer.sample_wait``: no new group reached the buffer for ``limit_s`` seconds while
    a round filled; the caller raises ``error``."""
    trainer_breakers.trip(trainer_breakers.SAMPLE_WAIT, emit, value=round(float(waited_s), 3), limit=limit_s,
                          detail=str(error), legacy_event="SampleBufferRolloutError")


def check_damaged_run(*, run: int, damaged: int, decoded: int, limit: int, position: int, sample_id: Any,
                      reason: str, emit: Callable[..., Any] | None) -> None:
    """``trainer.buffer_damaged``: raise :class:`DamagedPayloadLimit` once ``run`` damaged
    payloads came in a row, or ``damaged`` (at least ``limit``) outnumber the ``decoded``
    ones of the take; ``limit`` 0: no bound."""
    if limit > 0 and (run >= limit or (damaged >= limit and damaged > decoded)):
        error = DamagedPayloadLimit(
            f"{damaged} damaged payloads in this take ({run} in a row, "
            f"{decoded} decoded), the last at buffer position {int(position)} "
            f"({sample_id}: {reason}); this is not one damaged object: check the "
            f"samplers' payload schema against this reader's ({PAYLOAD_SCHEMA}). Nothing was consumed; "
            f"{MAX_DAMAGED_RUN_ENV} (now {limit}, 0: no limit) sets the bound")
        trainer_breakers.trip(trainer_breakers.BUFFER_DAMAGED, emit, value=run,
                              limit=limit, detail=str(error), legacy_event="DamagedPayloadLimit")
        raise error


def max_damaged_run_limit(environ: Mapping[str, str] | None = None) -> int:
    """The ``trainer.buffer_damaged`` bound a reader applies (0: no bound), resolved from
    the environment as the Miles reader resolves it (``trainer_breakers.reader_limits``)."""
    rb = trainer_breakers.resolved(environ, names=(trainer_breakers.BUFFER_DAMAGED,))
    if not rb.enabled(trainer_breakers.BUFFER_DAMAGED):
        return 0
    return int(rb.get(f"{trainer_breakers.BUFFER_DAMAGED}.max_run_n"))


def report_round_guards(emit: Callable[..., Any] | None, *, rollout_id: int, max_policy_age: int,
                        stale_groups: int, dropped_small: int = 0, nonfinite_reward: int = 0,
                        nonfinite_logprob: int = 0) -> None:
    """One ``rl_breaker`` record per round for each guard that passed groups over:
    ``trainer.policy_age`` (too old) and ``trainer.group_integrity`` (too few or too many
    trials, a non-finite reward or log-prob). Nothing when no group was passed over."""
    rollout = f"rollout {rollout_id}"
    if stale_groups:
        trainer_breakers.trip(trainer_breakers.POLICY_AGE, emit, limit=max_policy_age,
                              count=stale_groups, legacy_event="rl_buffer_stale_skipped",
                              detail=f"{rollout}: {stale_groups} groups older than "
                              f"{max_policy_age} versions")
    bad = {"dropped_small": dropped_small, "nonfinite_reward": nonfinite_reward,
           "nonfinite_logprob": nonfinite_logprob}
    if sum(bad.values()):
        trainer_breakers.trip(trainer_breakers.GROUP_INTEGRITY, emit, count=sum(bad.values()),
                              detail=f"{rollout}: " + ", ".join(f"{n} {why}" for why, n in bad.items() if n))


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
    # the trials of ``stale_groups`` by segment count (``timeline.segment_bucket``:
    # "1", "2", "3", "4+"), and whether any of those groups held compaction segments.
    # A group the age gate passed over unread (its index record was too old) is
    # counted as ``group_size`` trials under "unknown" (its payload is never read)
    stale_by_segments: dict[str, int] = field(default_factory=dict)
    stale_compacted: bool = False
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


def _echo_island_id(environ: Mapping[str, str] | None = None) -> int | None:
    """The trainer's learner id (:data:`ISLAND_ID_ENV`); None when unset or not a number."""
    raw = ((os.environ if environ is None else environ).get(ISLAND_ID_ENV) or "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _echo_event(event: str, **fields: Any) -> None:
    """One ``YETO_RL_EVENT`` stdout record from the rollout executor's process (it has
    no event tape of its own; the island's log stream carries the record). Its
    ``island_id`` is the trainer's learner id (:data:`ISLAND_ID_ENV`) unless given."""
    from yeto.rl.event_echo import format_record

    island = fields.pop("island_id", None)
    print(format_record({"island_id": _echo_island_id() if island is None else island, "time_unix": time.time(),
                         "event": str(event), **fields}), flush=True)


DEFAULT_READ_WORKERS = 8  # payload reads in flight (one Modal Volume file each)

# Raw payload bytes the reader holds at once: reads in flight, read and not decoded
# yet, read ahead (prefetch). A payload's bytes are dropped once it is decoded.
# Unset or 0: no limit, the reader of before (every payload of a round read at once,
# one GET each, no HEAD). RECOMMENDED_262K_INFLIGHT_BYTES is for a K3.1 262k run
# (2400 GiB trainer head): the optimizer offload takes ~1.9 TB (KIMI_K31_ASYNC_RL_GUIDE
# 10.4), which leaves ~550 GiB for the decoded round (262k without compaction: ~270
# GiB of arrays per 32 groups, twice that once Miles widens R3 to int32) and one
# group's decode (a v1 payload peaks at ~4.6x its bytes, ~50 GB at 11 GB). 64 GiB
# still reads five full 262k groups (11 GB each) at once.
MAX_INFLIGHT_BYTES_ENV = "YETO_SAMPLE_BUFFER_MAX_INFLIGHT_BYTES"
DEFAULT_MAX_INFLIGHT_BYTES = 0
RECOMMENDED_262K_INFLIGHT_BYTES = 64 << 30
_BYTE_UNITS = {"": 1, "b": 1, "k": 1 << 10, "kib": 1 << 10, "m": 1 << 20, "mib": 1 << 20,
               "g": 1 << 30, "gib": 1 << 30, "t": 1 << 40, "tib": 1 << 40}


def inflight_bytes_limit(environ: Mapping[str, str] | None = None) -> int | None:
    """:data:`MAX_INFLIGHT_BYTES_ENV`: bytes, or with a unit (``64GiB``, ``512MiB``;
    K / M / G / T are binary too); unset or 0: no limit (None)."""
    raw = (os.environ if environ is None else environ).get(MAX_INFLIGHT_BYTES_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_INFLIGHT_BYTES or None
    number = raw.rstrip("bBiIkKmMgGtT ")
    unit = raw[len(number):].strip().lower()
    try:
        value = int(float(number) * _BYTE_UNITS[unit])
    except (KeyError, ValueError):
        raise ValueError(f"{MAX_INFLIGHT_BYTES_ENV}={raw!r}: expected bytes, e.g. 68719476736 or 64GiB") from None
    if value < 0:
        raise ValueError(f"{MAX_INFLIGHT_BYTES_ENV}={raw!r}: expected >= 0")
    return value or None


class ReadTurn:
    """What a take's read waits on besides room: ``urgent`` (the take waits for this
    read now) and ``abandoned`` (its bytes are not wanted any more)."""

    __slots__ = ("ticket", "urgent", "abandoned")

    def __init__(self, ticket: int) -> None:
        self.ticket = ticket
        self.urgent = False
        self.abandoned = False


class ByteBudget:
    """The raw payload bytes a :class:`BatchReader` holds (``limit`` None: no limit).

    A take's read waits in a worker thread for its turn (tickets, handed out in the
    order the take asks for its reads) and for room: ``used + size <= limit``, or
    nothing held at all (a payload larger than the limit is read alone, never
    refused). A read ahead never waits: it is made only when it fits now and no take
    read is waiting (:meth:`try_acquire`); otherwise the prefetch stops and the take
    reads the rest. A read on the caller's thread (no read pool, or one group topping
    up a round) is counted without waiting (:meth:`force`).

    No deadlock: the take decodes its round in buffer order, and the reads ahead it
    uses hold their room until it decodes them; a take read that comes before them
    in that order (a read ahead that failed, a group the read ahead passed over, a
    deferred group judged again) would wait for room only the take's later decodes
    free. So the read the take waits for now is :meth:`urge` d: it goes at once,
    whatever its turn and the room (at most one payload over the limit, and only
    while the take waits for it). An :meth:`abandon` ed read that has not started
    takes no room and its worker returns at once."""

    def __init__(self, limit: int | None) -> None:
        self.limit = None if not limit else int(limit)
        self.used = 0
        self.peak = 0  # the most bytes held at once
        self.waits = 0  # reads that waited for room
        self._cond = threading.Condition()
        self._issued = 0
        self._turn = 0
        self._granted: set[int] = set()  # tickets let through before their turn
        self._warned = False
        self.urged = 0  # reads let through by urge() without room or turn

    def ticket(self) -> ReadTurn:
        with self._cond:
            self._issued += 1
            return ReadTurn(self._issued - 1)

    def urge(self, turn: "ReadTurn | None") -> None:
        if turn is None:
            return
        with self._cond:
            turn.urgent = True
            self._cond.notify_all()

    def abandon(self, turn: "ReadTurn | None") -> None:
        if turn is None:
            return
        with self._cond:
            turn.abandoned = True
            self._cond.notify_all()

    def _grant(self, ticket: int) -> None:
        self._granted.add(ticket)
        while self._turn in self._granted:
            self._granted.discard(self._turn)
            self._turn += 1

    def _fits(self, size: int) -> bool:
        return self.limit is None or self.used == 0 or self.used + size <= self.limit

    def _take(self, size: int) -> None:
        self.used += int(size)
        self.peak = max(self.peak, self.used)
        if self.limit is not None and size > self.limit and not self._warned:
            self._warned = True
            print(f"[sample-buffer] a payload of {size / (1 << 30):.1f} GiB is above the read budget "
                  f"({MAX_INFLIGHT_BYTES_ENV}={self.limit / (1 << 30):.1f} GiB): it is read alone", flush=True)

    def acquire(self, turn: ReadTurn, size: int) -> bool:
        """Wait for ``turn``'s turn and room for ``size`` bytes, then count them; False
        (nothing counted) when the read was abandoned meanwhile."""
        with self._cond:
            waited = False
            while not (turn.urgent or turn.abandoned) and (turn.ticket != self._turn or not self._fits(size)):
                waited = waited or turn.ticket == self._turn
                self._cond.wait()
            self.waits += int(waited)
            in_turn = turn.ticket == self._turn and self._fits(size)
            self._grant(turn.ticket)
            self._cond.notify_all()
            if turn.abandoned:
                return False
            self.urged += int(not in_turn)
            self._take(size)
            return True

    def try_acquire(self, size: int) -> bool:
        with self._cond:
            if self._turn != self._issued or not self._fits(size):
                return False
            self._take(size)
            return True

    def force(self, size: int) -> None:
        with self._cond:
            self._take(size)
            self._cond.notify_all()

    def release(self, size: int) -> None:
        if not size:
            return
        with self._cond:
            self.used -= int(size)
            self._cond.notify_all()


class _Decoded:
    """A payload a read thread decoded already (:func:`decodes_in_worker`)."""

    __slots__ = ("group",)

    def __init__(self, group: list[dict[str, Any]]) -> None:
        self.group = group


def decodes_in_worker(data: Any) -> bool:
    """Whether a budgeted read decodes this payload in its read thread: format v2
    (``buffer_payload``'s array blobs). Its zstd and numpy copies run outside the GIL,
    so the read threads decode a round's v2 groups in parallel, and their peak is
    ~1.4-2x the payload. A v1 payload (JSON + base64, GIL-bound, ~4.6x peak) is
    decoded on the caller's thread, one at a time, as before."""
    magic = _payload_format._V2_MAGIC
    return bytes(data[:len(magic)]) == magic


def group_nbytes(group: Sequence[Mapping[str, Any]]) -> int:
    """About the memory of a decoded group: its arrays' bytes, 32 B per list item
    (what a decoded group counts in the read budget)."""
    total = 0
    for sample in group:
        for value in sample.values():
            nbytes = getattr(value, "nbytes", None)
            if nbytes is not None:
                total += int(nbytes)
            elif isinstance(value, (list, tuple)):
                total += 32 * len(value)
    return total


class _Abandoned(Exception):
    """A take's read abandoned before it read (``BatchReader._abandon``)."""


def _takes_size(read_payload: Any) -> bool:
    """Whether a buffer's ``read_payload`` takes ``size=`` (``FileSampleBuffer``)."""
    import inspect

    try:
        return "size" in inspect.signature(read_payload).parameters
    except (TypeError, ValueError):
        return False


def _placeholder() -> Future:
    """A done read without bytes: what ``chosen`` keeps once a payload is decoded."""
    done: Future = Future()
    done.set_result(None)
    return done


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
                 max_damaged_run: int = DEFAULT_MAX_DAMAGED_RUN, one_group_per_task: bool = True,
                 max_inflight_bytes: Any = "auto", heartbeat_s: float = DEFAULT_HEARTBEAT_S,
                 read_heartbeat: Callable[[int, dict[str, int] | None], Any] | None = None,
                 read_heartbeat_s: float = DEFAULT_READ_HEARTBEAT_S) -> None:
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
        ``group_size``: the round must hold exactly ``groups * group_size`` trials
        (Miles' global batch counts rollouts; a trial is one sample, or the samples of its
        compaction segments, ``buffer_payload.trial_count``). A short group (a sampler
        lost trials) is trained with its own trials (``TakeReport.incomplete_groups``);
        one with fewer than ``MIN_GROUP_SAMPLES`` or more than ``group_size`` trials is
        consumed, not trained.
        A round that falls short selects more whole groups while they fit; the first
        that does not is carried over to the next take (the round's end, hence the
        watermark, stays before it); what is still missing is reported as
        ``padded_samples`` (:func:`padding_samples` builds them). None: no check.
        ``max_damaged_run``: the damaged payloads at which a take raises (the class
        doc); 0: no limit. ``one_group_per_task``: a second group of a store-dispenser
        task is passed over (:class:`TaskSlots`, ``TakeReport.duplicate_task_groups``).
        ``max_inflight_bytes``: the raw payload bytes held at once (:class:`ByteBudget`;
        ``"auto"``: :func:`inflight_bytes_limit` from the environment; None or 0: no limit).
        ``heartbeat_s``: while a take waits for new groups (the ``wait_s`` bound), one
        :data:`WAIT_EVENT` record per ``heartbeat_s`` seconds, so the launcher's stall clock
        (``launch.stall``, any island event) does not take the wait for a hang; a wait for
        unreadable leases (no ``wait_s`` bound) gets none. 0: no heartbeat.
        ``read_heartbeat``: ``callable(read cursor, {shard: position} or None)``, called
        while a take waits for new groups whenever the read cursor moved since its last
        call, and at least every ``read_heartbeat_s`` seconds: it publishes where this
        trainer reads to the samplers (:func:`trainer_read_heartbeat`). Without it a
        sampler counts its backlog from the read position of the newest policy release,
        which only a trained round moves: groups the take passed over (too old, fenced,
        a duplicate task) then hold the sampler at its backlog cap while the take waits
        for groups the sampler no longer generates. None: nothing is published."""
        if int(max_policy_age) < 1:
            raise ValueError("max_policy_age must be >= 1")
        self.buffer = buffer
        self.max_policy_age = int(max_policy_age)
        self.wait_s = float(wait_s)
        self.heartbeat_s = float(heartbeat_s)
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
        self.budget = ByteBudget(inflight_bytes_limit() if max_inflight_bytes == "auto" else max_inflight_bytes)
        self._open: set[Future] = set()  # budgeted reads whose bytes are still counted
        self._open_lock = threading.Lock()
        self._read_wait_s = 0.0  # seconds a take waited for single reads while decoding
        self._sized_reads = _takes_size(getattr(buffer, "read_payload", None))
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
        self.read_heartbeat = read_heartbeat
        self.read_heartbeat_s = float(read_heartbeat_s)
        self._beat_cursor: int | None = None  # the read cursor read_heartbeat got last
        self._beat_at = -float("inf")  # ... and when (on ``clock``)
        self._beat_failed = False  # the last call raised (reported once per outage)

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
            result = read.result()
            group = result.group if isinstance(result, _Decoded) else decode_group(result)
            del result
            self._damaged_run = 0
            self._decoded_payloads += 1
            return group
        except DamagedPayload as error:
            reason = str(error)
        except Exception as error:  # noqa: BLE001 - only a payload that is gone is passed over
            if ref is None or not payload_is_missing(self.buffer, ref, error):
                raise
            reason = "payload is missing"
        finally:
            self._release(read)  # decoded (or not a group): its bytes no longer count
        dispensed = _dispense_position(entry)
        self._damaged.append({"position": int(position), "sample_id": getattr(entry, "sample_id", None),
                              "payload_ref": ref, "reason": reason,
                              **({} if dispensed is None else {"dispense_position": dispensed})})
        self._damaged_run += 1
        check_damaged_run(run=self._damaged_run, damaged=len(self._damaged), decoded=self._decoded_payloads,
                          limit=self.max_damaged_run, position=int(position),
                          sample_id=getattr(entry, "sample_id", None), reason=reason, emit=self.emit)
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
        self._note_stale(group)
        return True

    def _note_stale(self, group: Sequence[Mapping[str, Any]] | None) -> None:
        """Count a stale group's trials by segment count (``TakeReport.stale_by_segments``);
        None: a group passed over unread (``group_size`` trials, "unknown")."""
        from yeto.rl.engine.timeline import segment_bucket  # noqa: PLC0415

        counts = getattr(self, "_stale_segments", None)
        if counts is None:
            counts = self._stale_segments = {}
        if group is None:
            counts["unknown"] = counts.get("unknown", 0) + int(self.group_size or 1)
            return
        if has_siblings(group):
            self._stale_compacted = True
        for trial in trial_rows(group):
            bucket = segment_bucket(len(trial))
            counts[bucket] = counts.get(bucket, 0) + 1

    def _read(self, ref: str, *, inline: bool = False) -> Future:
        if self.budget.limit is None:  # no budget: as before it
            if self._pool is None or inline:
                done: Future = Future()
                try:
                    done.set_result(self.buffer.read_payload(ref))
                except BaseException as error:  # noqa: BLE001 - raised by result()
                    done.set_exception(error)
                return done
            return self._pool.submit(self.buffer.read_payload, ref)
        held = [0]  # the bytes this read counts in the budget
        if self._pool is None or inline:  # the caller's thread: counted, never waits (ByteBudget)
            done = self._opened(Future(), held)
            try:
                data = self.buffer.read_payload(ref)
                self.budget.force(len(data))
                held[0] = len(data)
                done.set_result(data)
            except BaseException as error:  # noqa: BLE001 - raised by result()
                done.set_exception(error)
            return done
        turn = self.budget.ticket()
        read = self._pool.submit(self._read_in_turn, ref, turn, held)
        read.yeto_turn = turn  # type: ignore[attr-defined]
        return self._opened(read, held)

    def _opened(self, read: Future, held: list[int]) -> Future:
        read.yeto_held = held  # type: ignore[attr-defined]
        with self._open_lock:
            self._open.add(read)
        # a failed read holds no bytes: out of the open set at once
        read.add_done_callback(lambda done: None if done.cancelled() or done.exception() is None
                               else self._release(done))
        return read

    def _payload_size(self, ref: str) -> int | None:
        """The payload's size without reading it (one HEAD on an object store), None
        when the buffer cannot tell: the read is then counted once it arrived."""
        size_of = getattr(self.buffer, "payload_size", None)
        if not callable(size_of):
            return None
        size = size_of(ref)  # a store that does not answer: raised, like a read it does not answer
        return None if size is None else int(size)

    def _read_in_turn(self, ref: str, turn: ReadTurn, held: list[int]) -> Any:
        """A take's read in a worker: its size, its turn and room in the budget, then the
        bytes. Every ticket takes its turn (or the later ones never do), also one that
        fails or is abandoned before it reads."""
        if turn.abandoned:
            self.budget.acquire(turn, 0)
            raise _Abandoned(ref)
        try:
            size = self._payload_size(ref)
        except BaseException:
            self.budget.abandon(turn)
            self.budget.acquire(turn, 0)
            raise
        if not self.budget.acquire(turn, size or 0):
            raise _Abandoned(ref)
        held[0] = size or 0
        return self._read_counted(ref, held, size)

    def _read_counted(self, ref: str, held: list[int], size: int | None = None) -> Any:
        """Read ``ref`` with ``held[0]`` bytes counted for it; the count is set to the
        bytes read (a size that was unknown, or changed), and to 0 when the read fails.
        A known ``size`` goes to a buffer that takes it (a large payload is then read
        by parallel ranges, ``FileSampleBuffer.read_payload``)."""
        try:
            data = self.buffer.read_payload(ref, size=size) if size is not None and self._sized_reads \
                else self.buffer.read_payload(ref)
        except BaseException:
            self.budget.release(held[0])
            held[0] = 0
            raise
        self._recount(held, len(data))
        if not decodes_in_worker(data):
            return data
        try:  # a v2 payload: decoded here; its bytes go, its arrays count instead
            group = decode_group(data)
        except BaseException:  # DamagedPayload included: _group_at records it like a caller decode
            self._recount(held, 0)
            raise
        del data
        self._recount(held, group_nbytes(group))
        return _Decoded(group)

    def _recount(self, held: list[int], size: int) -> None:
        """Count ``size`` bytes for a read that counted ``held[0]``."""
        if size > held[0]:
            self.budget.force(size - held[0])
        else:
            self.budget.release(held[0] - size)
        held[0] = int(size)

    def _read_ahead(self, ref: str) -> Future | None:
        """A read ahead of the next take; None when it does not fit the budget now
        (the prefetch stops there, the take reads the rest)."""
        if self.budget.limit is None:
            return self._read(ref)
        size = self._payload_size(ref)
        if not self.budget.try_acquire(size or 0):
            return None
        held = [size or 0]
        if self._pool is None:
            done = self._opened(Future(), held)
            try:
                done.set_result(self._read_counted(ref, held, size))
            except BaseException as error:  # noqa: BLE001 - raised by result()
                done.set_exception(error)
            return done
        return self._opened(self._pool.submit(self._read_counted, ref, held, size), held)

    def _release(self, read: Future) -> None:
        """The read's bytes no longer count (decoded, passed over or abandoned)."""
        held = getattr(read, "yeto_held", None)
        with self._open_lock:
            self._open.discard(read)
            size = held[0] if held else 0
            if held:
                held[0] = 0
        self.budget.release(size)

    def _abandon(self, reads: Any) -> None:
        """Reads whose bytes are not wanted: one not started yet takes no room (its
        worker returns at once), the others are released once they end."""
        for read in list(reads):
            self.budget.abandon(getattr(read, "yeto_turn", None))
            read.add_done_callback(self._release)

    def _abandon_open(self) -> None:
        with self._open_lock:
            reads = list(self._open)
        self._abandon(reads)

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
        self._stale_segments, self._stale_compacted = {}, False
        self._damaged_run = self._decoded_payloads = 0
        self._duplicates = []
        try:
            out, report = self._take(int(rollout_id), int(groups), current_version)
        except BaseException:
            self._abandon_open()  # its reads: selected again by the next take, read again then
            self._retry = self._taken + self._retry
            self._carry = carried_in
            # the groups deferred by an earlier take wait again (those this take
            # selected and deferred are selected again from ``_retry``)
            self._deferred = deferred_in
            raise
        finally:
            self._taken = []
        self._report_round(report)  # after the take: a report never undoes a round
        return out, report

    def _beat_read_position(self) -> None:
        """While a take waits for new groups: hand the read cursor (every group below it
        is selected: chosen for this round or passed over) to ``read_heartbeat`` when it
        moved since the last call or ``read_heartbeat_s`` passed (the class doc). A
        failure is reported once per outage and never stops the take."""
        publish = self.read_heartbeat
        if publish is None:
            return
        cursor = int(self._cursor())
        now = self.clock()
        due = now - self._beat_at >= self.read_heartbeat_s
        if not due and (cursor == self._beat_cursor or self._beat_failed):
            return  # unchanged, or the last try failed: tried again after read_heartbeat_s
        positions = getattr(self.buffer, "shard_positions", None)
        try:
            shards = positions(cursor) if callable(positions) else None
            publish(cursor, {str(k): int(v) for k, v in shards.items()} if shards else None)
        except Exception as error:  # noqa: BLE001 - the heartbeat never stops a take
            if not self._beat_failed:
                self.emit(READ_HEARTBEAT_FAILED_EVENT, rollout_id=self._rollout_id, read_cursor=cursor,
                          error=f"{type(error).__name__}: {error}"[:300])
            self._beat_failed, self._beat_at = True, now
            return
        self._beat_cursor, self._beat_at, self._beat_failed = cursor, now, False

    def _report_round(self, report: TakeReport) -> None:
        """One ``rl_breaker`` record per round for each guard that passed groups over:
        ``trainer.policy_age`` (too old) and ``trainer.group_integrity`` (too few or too many
        trials, a non-finite reward or log-prob). Nothing when no group was passed over."""
        report_round_guards(self.emit, rollout_id=report.rollout_id, max_policy_age=self.max_policy_age,
                            stale_groups=report.stale_groups, dropped_small=report.dropped_small_groups,
                            nonfinite_reward=report.nonfinite_reward_groups,
                            nonfinite_logprob=report.nonfinite_logprob_groups)

    def _take(self, rollout_id: int, groups: int, current_version: int | None
              ) -> tuple[list[tuple[int, list[dict[str, Any]]]], TakeReport]:
        started = progress = beat = self.clock()
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
                self._note_stale(group)
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
                    self._note_stale(None)
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
            if ahead and self.budget.limit is not None:
                # read ahead for the groups this pass did not take (aged out, fenced, the
                # carried group's place): their bytes would hold the budget the take's own
                # reads wait for; a later pass that selects one reads it again
                self._abandon(ahead.values())
                ahead.clear()
            self._note_selection(rollout_id, len(selected), len(chosen) - kept_before,
                                 self.unidentified_total - unidentified_before)
            if selected or len(chosen) > kept_before:  # decode now (a deferred group judged again
                # included): a group that is dropped is replaced by the next selection
                if self.budget.limit is None:  # with a budget each read is waited for at its decode
                    read_s += self._settle([read for _, read, _ in chosen])
                begun, waited = self.clock(), self._read_wait_s
                dropped_small += self._drop_incomplete(chosen, decoded, accounted, versions, source_groups,
                                                       source_tokens, position_sources)
                waited = self._read_wait_s - waited
                read_s += waited
                decode_s += self.clock() - begun - waited
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
            waited = self.clock() - progress
            if waited > self.wait_s:
                error = SampleBufferRolloutError(
                    f"rollout {rollout_id}: {len(chosen)}/{groups} groups after {self.wait_s:g} s "
                    f"without a new group (buffer position {self._cursor()})")
                trip_sample_wait(self.emit, waited_s=self.clock() - progress, limit_s=self.wait_s, error=error)
                raise self._ended(error, stale=stale, stale_incarnation=fenced, quarantined=quarantined,
                                  dropped_small=dropped_small)
            if self.heartbeat_s > 0 and self.clock() - max(progress, beat) >= self.heartbeat_s:
                # a legitimate wait is not a hang: the launcher's stall clock sees this event
                beat = self.clock()
                self.emit(WAIT_EVENT, rollout_id=rollout_id, waited_s=round(waited, 3), limit_s=self.wait_s,
                          groups=len(chosen), needed=groups, heartbeat_s=self.heartbeat_s)
            self._beat_read_position()
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
            incomplete = sum(1 for position, _, _ in chosen if trial_count(decoded[position]) < self.group_size)
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
                               stale_by_segments=dict(sorted(getattr(self, "_stale_segments", {}).items())),
                               stale_compacted=bool(getattr(self, "_stale_compacted", False)),
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
        """Top the chosen groups up to ``target`` trials with whole groups selected
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
        missing = target - sum(trial_count(decoded[position]) for position, _, _ in chosen)
        dropped = 0
        while missing > 0 and not self._deferred:  # nothing joins behind a deferred group
            selected = self._select(1)
            if not selected:
                break
            item = selected[0]
            end = self._cursor()
            if floor is not None and item.entry.behavior_version < floor:
                skipped["stale"] += 1  # too old: consumed unread, as in the main selection
                self._note_stale(None)
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
            trials = trial_count(group)
            if not MIN_GROUP_SAMPLES <= trials <= int(self.group_size):
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
            if trials > missing:
                # it holds its task when the next take chooses it
                self._carry = (position, group, version, tokens, source, _dispense_position(item.entry),
                               task_key(item.entry))
                return position, 1, missing, dropped
            self._choose(position, group, version, tokens, source, chosen, decoded, accounted, versions,
                         source_groups, source_tokens, position_sources)
            self._accept(item.entry, position)
            missing -= trials
        return end, 0, max(0, missing), dropped

    def _drop_incomplete(self, chosen: list, decoded: dict, accounted: dict, versions: dict, source_groups: dict,
                         source_tokens: dict, position_sources: dict) -> int:
        """Decode the chosen groups not yet decoded; drop (in place) those with a
        damaged payload (:meth:`_group_at`, counted apart) and, with a ``group_size``,
        those whose trial count is outside ``MIN_GROUP_SAMPLES..group_size`` or with a
        non-finite reward (:func:`finite_rewards`, counted apart), then those whose
        store-dispenser task an earlier group holds (:meth:`_duplicate`), and undo their
        accounting. Returns the number dropped for their size. Their positions stay
        consumed (the cursor moved past them)."""
        fresh = [(position, read) for position, read, _ in chosen if position not in decoded]
        if not fresh:
            return 0
        if self.budget.limit is None:
            self._settle([read for _, read in fresh])
        else:  # ``chosen`` keeps a placeholder: a payload's bytes go once it is decoded
            ready = {position for position, _ in fresh}
            chosen[:] = [(p, _placeholder() if p in ready else r, early) for p, r, early in chosen]
        drop: set[int] = set()
        nonfinite = nonfinite_lp = damaged = duplicate = 0
        for position, group in self._decoded_in_order(fresh):
            if group is not None and self._stale_payload(position, group):
                group, aged = None, True
            else:
                aged = False
            sized = group is not None and (
                self.group_size is None or MIN_GROUP_SAMPLES <= trial_count(group) <= int(self.group_size))
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

    def _decoded_in_order(self, fresh: list[tuple[int, Future]]) -> Any:
        """``(position, decoded group or None)`` of ``fresh`` reads, in their order,
        each read waited for at its turn (with a budget the reads wait for room in
        this order, ByteBudget) and dropped from ``fresh`` before its group is
        yielded, so its bytes go with its decode. Reads left when the caller stops
        (a decode that raised) are abandoned: their bytes are released."""
        fresh.reverse()
        try:
            while fresh:
                position, read = fresh.pop()
                if not read.done():
                    # the take waits for this read: it goes now, even past the reads ahead
                    # holding the room until their later decodes (ByteBudget)
                    self.budget.urge(getattr(read, "yeto_turn", None))
                    self._read_wait_s += self._settle([read])
                group = self._group_at(position, read)
                del read
                yield position, group
        finally:
            self._abandon(read for _, read in fresh)

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
                        read = self._read_ahead(ref)
                        if read is None:  # the budget is full: the next take reads the rest
                            return
                        self._reads[ref] = read
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
        self._abandon(self._reads.values())
        self._reads.clear()
        self.buffer.reset_to(int(watermark))
        self._ends.clear()
        self._carry = None  # its position is at or past the watermark: selected again
        self._retry = []  # so are the groups of a take that raised
        self._deferred = []  # likewise
        if self.tasks is not None:
            self.tasks.reset(int(watermark))
        return int(self.buffer.watermark)
