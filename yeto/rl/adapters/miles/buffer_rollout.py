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
  are passed over (consumed, not trained) and counted. The reader is backend-neutral
  and lives in :mod:`yeto.rl.engine.batch_reader` (re-exported here); this module keeps
  the Miles conversion of a selected group.

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
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# The payload format and the group writer are engine-neutral (the
# engine sampler island writes the same payloads); re-exported here.
from yeto.rl.engine.buffer_payload import (  # noqa: F401
    finite_rewards,
    COMPACTION_KEY,
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
    check_trials,
    compaction_meta,
    has_siblings,
    trial_count,
    trial_rows,
)
# The trainer's buffer reader is backend-neutral (verl and ms-swift read through it
# too); re-exported here under its old names.
from yeto.rl.engine.batch_reader import (  # noqa: F401
    BUFFER_ROOT_ENV,
    SAMPLER_ID_ENV,
    WAIT_ENV,
    POLL_ENV,
    HEARTBEAT_ENV,
    DEFAULT_HEARTBEAT_S,
    WAIT_EVENT,
    MAX_POLICY_AGE_ENV,
    ONE_GROUP_PER_TASK_ENV,
    MAX_DAMAGED_RUN_ENV,
    DEFAULT_MAX_DAMAGED_RUN,
    SHARD_ENV,
    SHARDS_ENV,
    ISLAND_ID_ENV,
    DEFAULT_WAIT_S,
    DEFAULT_POLL_S,
    TRAINER_CONSUMER,
    DEFAULT_READ_HEARTBEAT_S,
    READ_HEARTBEAT_FAILED_EVENT,
    _DEFER,
    buffer_shards,
    open_buffer,
    check_unmerged_shards,
    open_lane_log,
    segment_row_width,
    check_compaction_round,
    refuse_compaction_round,
    stale_segment_buckets,
    compaction_batch_line,
    report_unaligned,
    trip_sample_wait,
    check_damaged_run,
    report_round_guards,
    TakeReport,
    _dispense_position,
    task_key,
    TaskSlots,
    unidentified_metadata,
    MIN_GROUP_SAMPLES,
    UNIDENTIFIED_STARVED_EVENT,
    UNIDENTIFIED_STARVED_SELECTS_ENV,
    DEFAULT_UNIDENTIFIED_STARVED_SELECTS,
    SETTLE_WARN_S,
    _starved_selects_limit,
    SOURCE_BAD_STREAK_EVENT,
    SOURCE_BAD_STREAK_ENV,
    DEFAULT_SOURCE_BAD_STREAK,
    _source_bad_streak_limit,
    _echo_island_id,
    _echo_event,
    DEFAULT_READ_WORKERS,
    MAX_INFLIGHT_BYTES_ENV,
    DEFAULT_MAX_INFLIGHT_BYTES,
    RECOMMENDED_262K_INFLIGHT_BYTES,
    _BYTE_UNITS,
    inflight_bytes_limit,
    ReadTurn,
    ByteBudget,
    _Decoded,
    decodes_in_worker,
    group_nbytes,
    _Abandoned,
    _takes_size,
    _placeholder,
    BatchReader,
)
from yeto.rl.engine.batch_reader import unconvertible_round as _unconvertible_round
from yeto.rl.engine.merged_release import LANE_ENV, BufferLane  # noqa: F401
from yeto.rl.engine.data_split import DATA_ROW_KEY, DataSlot  # noqa: F401 - DATA_ROW_KEY re-exported
from yeto.rl.engine import trainer_breakers  # noqa: F401
from yeto.rl.engine.xhw_guard import downweight_group, rollout_stats, source_key  # noqa: F401

SAMPLER, TRAINER, CONVERTIBLE = "sampler", "trainer", "convertible"
ROLES = (SAMPLER, TRAINER, CONVERTIBLE)

DATA_SLOT_ENV = "YETO_SAMPLE_BUFFER_DATA_SLOT"  # sampler: "index/count" of the prompt data
INCARNATION_ENV = "YETO_SAMPLER_INCARNATION"  # set by the sampler island before Ray starts
# the rollout functions run in Ray workers, which inherit the raylet's environment:
# connect_island_ray forwards these to every worker
ISLAND_ENVS = (BUFFER_ROOT_ENV, SAMPLER_ID_ENV, WAIT_ENV, POLL_ENV, MAX_POLICY_AGE_ENV, HEARTBEAT_ENV, SHARD_ENV,
               SHARDS_ENV, INCARNATION_ENV, LANE_ENV, DATA_SLOT_ENV, ISLAND_ID_ENV,
               # the trainer's relay pump (yeto.rl.spot_rollout.relay_buffer.RELAY_ENVS)
               "YETO_SAMPLE_RELAY_ADDR", "YETO_SAMPLE_RELAY_CHANNEL", "YETO_SAMPLE_RELAY_PREFETCH",
               # the pump's boat orphan watchdog (yeto.rl.harness.harbor.boat_labels)
               "YETO_BOAT_RUN_ID", "YETO_BOAT_ORPHAN_LEASE_S", "BOAT_API_KEY")


_MODULE = "yeto.rl.adapters.miles.buffer_rollout_miles"
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
               lane: BufferLane | None = None, data_slot: str | None = None,
               island_id: int | None = None) -> dict[str, str]:
    """The environment the launch layer exports on the island (read in the rollout
    process). A convertible node writes like a sampler, into its own ``shard`` of
    the index, and reads like a trainer. ``shards``: the sampler shards of a
    launch with several sampler islands (a sampler writes the one named by its
    sampler id, the trainer merges them all). ``lane``: a synced trainer's
    buffer lane. ``data_slot``: a sampler's or a convertible node's slot of the
    prompt data (``sample_roles.DataSlot``; several sampler islands, or the nodes of
    a convertible pool, draw disjoint prompts). ``island_id``: a trainer's learner id
    (:data:`ISLAND_ID_ENV`)."""
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
    if island_id is not None and role == TRAINER:
        env[ISLAND_ID_ENV] = str(int(island_id))
    return env


def renumber_group(samples: Sequence[dict[str, Any]], *, position: int,
                   samples_per_group: int) -> list[dict[str, Any]]:
    """Give a group read from the buffer an identity unique in the buffer.

    Miles numbers groups and samples per sampler process from 0, so groups from two
    sampler rounds (or a restarted sampler) repeat them; the trainer's batch checks
    and trajectory ids need them unique. The buffer position is unique for the
    buffer's life and the same after a restore re-reads it, so ``group_index`` is
    the position and ``index`` is ``position * samples_per_group + j``. The
    sampler's numbers stay in ``metadata["yeto_sampler_identity"]``.

    A group with compaction segments (``buffer_payload.has_siblings``: a sample with
    ``metadata.compaction``) is numbered by trial instead: ``j`` is the
    trial's number in the group (:func:`~yeto.rl.engine.buffer_payload.check_trials`),
    and every sample of the trial gets that ``index`` and the same ``rollout_id``. Miles
    then keys one rollout per trial (``rollout_segments``: one advantage per trial; the
    rollout-side DP schedule keeps a trial's samples in one step, and with a
    ``rollout_id`` on the round its samples are not trimmed by count; the trajectory
    evidence binds the segments of one ``index`` as one trial). Any other group (a Miles
    sampler's ``rollout_id`` siblings included) is numbered exactly as before."""
    if not has_siblings(samples):
        trials = None
        if len(samples) > int(samples_per_group):
            raise SampleBufferRolloutError(
                f"group at buffer position {position} has {len(samples)} samples > "
                f"samples per group {samples_per_group}")
    else:
        rows = check_trials(samples)
        if len(rows) > int(samples_per_group):
            raise SampleBufferRolloutError(
                f"group at buffer position {position} has {len(rows)} trials > "
                f"samples per group {samples_per_group}")
        trials = {row: t for t, trial in enumerate(rows) for row in trial}
    out = []
    for j, sample in enumerate(samples):
        sample = dict(sample)
        metadata = dict(sample.get("metadata") or {})
        identity = {"group_index": sample.get("group_index"), "index": sample.get("index")}
        index = int(position) * int(samples_per_group) + (j if trials is None else trials[j])
        if trials is not None:
            identity["rollout_id"] = sample.get("rollout_id")
            sample["rollout_id"] = index
        metadata["yeto_sampler_identity"] = identity
        sample.update(metadata=metadata, group_index=int(position), index=index)
        out.append(sample)
    return out


SCORE_CENTERING_LOSS = "score_centering"  # Miles --loss-type


def trainer_sampling(args: Any) -> dict[str, Any] | None:
    """The sampling a score_centering trainer's behavior distribution holds at (Miles
    ``--rollout-temperature`` / ``--rollout-top-p`` / ``--rollout-top-k``); None when
    the trainer's ``--loss-type`` is not score_centering (no check)."""
    if getattr(args, "loss_type", None) != SCORE_CENTERING_LOSS:
        return None
    return {"temperature": float(getattr(args, "rollout_temperature", 1.0)),
            "top_p": float(getattr(args, "rollout_top_p", 1.0)),
            "top_k": int(getattr(args, "rollout_top_k", -1))}


def check_version_fields(group: Sequence[Mapping[str, Any]], sampling: Mapping[str, Any]) -> bool:
    """A score_centering trainer reads the samples' log-probs and top-K log-probs as the
    behavior distribution of its own ``sampling`` (:func:`trainer_sampling`). Raises
    :class:`SampleBufferRolloutError` when a sample's ``metadata.engine.version_fields``
    (stamped by the engine sampler) name another temperature / top_p / top_k: its
    log-probs are of another distribution than the trainer's. False when no sample of
    the group carries them (a sampler before the stamp, a Miles sampler): let through."""
    from yeto.rl.engine.buffer_payload import ENGINE_META  # noqa: PLC0415
    from yeto.rl.engine.sampler_engine import version_fields_mismatch  # noqa: PLC0415

    stamped = False
    for sample in group:
        fields = ((sample.get("metadata") or {}).get(ENGINE_META) or {}).get("version_fields")
        if not isinstance(fields, Mapping):
            continue
        stamped = True
        differ = version_fields_mismatch(fields, sampling)
        if differ:
            raise SampleBufferRolloutError(
                f"sample {sample.get('index')}: engine.version_fields {', '.join(differ)}: score_centering "
                "trains on the sampler's log-probs as the behavior distribution of the trainer's sampling; "
                "fix the sampler spec's sampling")
    return stamped


def miles_groups(groups: Sequence[tuple[int, Sequence[dict[str, Any]]]], *, samples_per_group: int,
                 r3_layers: tuple[int, int] | None = None, r3_num_experts: int | None = None,
                 sampling: Mapping[str, Any] | None = None, stats: dict[str, int] | None = None,
                 require_known_versions: bool = False,
                 published_versions: Collection[int] | None = None,
                 ) -> tuple[list[tuple[int, list[dict[str, Any]]]], list[str]]:
    """The taken groups renumbered (:func:`renumber_group`) and in Miles' sample form
    (``buffer_payload.miles_sample_dict``), and why each group that cannot be converted
    was left out. One malformed sample (an R3 routing row that mixes pads and ids, an
    expert id past the model's, a layer count of another model) leaves its group out
    of the round, which is padded instead, rather than stopping the trainer (the caller
    takes again when no group of a take converts, and fails after a few such takes).
    ``sampling``: a score_centering trainer's (:func:`trainer_sampling`); a group whose
    ``engine.version_fields`` differ is left out the same way (:func:`check_version_fields`),
    and a group without them is kept and counted in ``stats["no_version_fields_groups"]``.
    ``require_known_versions`` (a full-parameter island, whose trajectory evidence
    names the behavior policy of every trained token): a group with a response token
    of unknown policy version is left out the same way, before it is trained, and
    counted in ``stats["unknown_version_groups"]``; the evidence would otherwise stop
    the island after the take. Samples served by two versions (named by the oldest,
    ``buffer_payload.miles_sample_dict``) are counted in ``stats["cross_version_samples"]``.
    ``published_versions`` (the same island, its trajectories named by the driver's
    published versions, not by external merged cuts): a group with a trajectory whose
    oldest version (the one the evidence names it by) the driver never published -- a
    release an earlier process of the island made and the store no longer names -- is
    left out the same way and counted in ``stats["unpublished_version_groups"]``; the
    evidence would otherwise stop the island ("policy vN ... was not published")."""
    from yeto.rl.engine.buffer_payload import (  # noqa: PLC0415
        cross_version,
        miles_sample_dict,
        oldest_response_version,
        unknown_version_tokens,
    )

    published = None if published_versions is None else frozenset(int(v) for v in published_versions)

    kept, problems = [], []
    unstamped = 0
    for position, group in groups:
        try:
            stamped = sampling is None or check_version_fields(group, sampling)
            converted = [miles_sample_dict(d, r3_layers=r3_layers, r3_num_experts=r3_num_experts)
                         for d in renumber_group(group, position=position, samples_per_group=samples_per_group)]
            crossed = sum(cross_version(d) for d in converted)
            if crossed and stats is not None:
                stats["cross_version_samples"] = stats.get("cross_version_samples", 0) + crossed
            if require_known_versions:
                unknown = sum(unknown_version_tokens(d) for d in converted)
                if unknown:
                    if stats is not None:
                        stats["unknown_version_groups"] = stats.get("unknown_version_groups", 0) + 1
                    raise SampleBufferRolloutError(
                        f"{unknown} response token(s) of unknown policy version (no weight_versions span "
                        "names them): the full-parameter evidence cannot name their behavior policy")
            if published is not None:
                unpublished = sorted({v for v in map(oldest_response_version, converted)
                                      if v is not None and v not in published})
                if unpublished:
                    if stats is not None:
                        stats["unpublished_version_groups"] = stats.get("unpublished_version_groups", 0) + 1
                    raise SampleBufferRolloutError(
                        f"trajectories of policy version(s) {unpublished} this island has not published "
                        f"(published: {sorted(published)}): the full-parameter evidence cannot name "
                        "their behavior policy")
            kept.append((position, converted))
            unstamped += not stamped
        except SampleBufferRolloutError as error:
            problems.append(f"buffer position {position}: {error}")
    if unstamped:
        print(f"[sample-buffer] WARNING: {unstamped} group(s) carry no engine.version_fields: their sampling "
              "is not checked against the score_centering trainer's", flush=True)
        if stats is not None:
            stats["no_version_fields_groups"] = stats.get("no_version_fields_groups", 0) + unstamped
    report_unaligned(problems)
    return kept, problems


def unconvertible_round(rollout_id: int, takes: int, problem: str) -> SampleBufferRolloutError:
    """The error that ends a round after ``takes`` takes in a row without a group that
    converts (:func:`miles_groups`), reported as a ``trainer.sample_alignment`` stop."""
    return _unconvertible_round(rollout_id, takes, problem, form="Miles samples")


def with_reader_edits(converted: Sequence[tuple[int, list[dict[str, Any]]]],
                      groups: Sequence[tuple[int, Sequence[dict[str, Any]]]]
                      ) -> list[tuple[int, list[dict[str, Any]]]]:
    """The converted groups (:func:`miles_groups`) of the positions still in ``groups``,
    with the reader's later edits of those decoded groups carried over: the ``reward``
    (the xhw guard's downweight) and the ``metadata`` (the downweight record, the merged
    release cut stamp). The conversion keeps a sample's other fields and adds only
    ``metadata["yeto_sampler_identity"]`` (kept), so the result is what converting the
    edited groups would give, without copying their rollout arrays (R3) again."""
    edited = {int(position): group for position, group in groups}
    out = []
    for position, group in converted:
        source = edited.get(int(position))
        if source is None:
            continue  # pruned after the conversion: not trained
        if len(source) != len(group):
            raise SampleBufferRolloutError(f"buffer position {position}: {len(source)} decoded samples, "
                                           f"{len(group)} converted")
        samples = []
        for d, e in zip(group, source):
            d = dict(d)
            if "reward" in e:
                d["reward"] = e["reward"]
            identity = (d.get("metadata") or {}).get("yeto_sampler_identity")
            d["metadata"] = {**dict(e.get("metadata") or {}), "yeto_sampler_identity": identity}
            samples.append(d)
        out.append((position, samples))
    return out


PADDING_GROUP_BASE = 1 << 40  # group_index of padding samples: past any buffer position
PADDING_KEY = "yeto_padding"


def padding_samples(groups: Sequence[Sequence[dict[str, Any]]], count: int, *,
                    samples_per_group: int) -> list[dict[str, Any]]:
    """``count`` samples that fill a round to Miles' global batch without training
    (one per missing trial: each is a rollout of its own).

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
        metadata = {**dict(pad.get("metadata") or {}), PADDING_KEY: True}
        metadata.pop(COMPACTION_KEY, None)  # a padding sample is no segment of a trial
        pad["metadata"] = metadata
        pad["group_index"] = PADDING_GROUP_BASE + k
        pad["index"] = (PADDING_GROUP_BASE + k) * int(samples_per_group)
        out.append(pad)
    return out


def segmented_round_problems(args: Any) -> list[str]:
    """Why Miles would not train a round that holds multi-sample trials (compaction
    segments) one rollout per trial; [] when it does.

    With a ``rollout_id`` on the round (:func:`renumber_group`) Miles keeps every sample
    (no trim by count, ``rollout_data_conversion.postprocess_rollout_data``) and its
    rollout-side DP schedule (``train_data_conversion.can_precompute_dp_schedule``,
    ``dp_schedule.build_dp_schedule``) counts ``--global-batch-size`` in rollouts and puts
    all samples of a rollout in one step. That schedule needs a Megatron trainer without
    independent DP, multi-LoRA or multimodal inputs; otherwise the training side splits by
    sample count and a trial's segments can land in different steps or be dropped.
    ``--use-dynamic-global-batch-size`` is only computed for a round without rollout ids
    (Miles asserts on one with them). Without ``--use-dynamic-batch-size`` the schedule
    cuts a step into fixed micro batches and asserts unless the step's sample count (data
    dependent here) is a multiple of DP x micro batch size."""
    problems = []
    if not getattr(args, "use_dynamic_batch_size", False):
        problems.append("no --use-dynamic-batch-size (--rl-max-tokens-per-gpu)")
    if getattr(args, "use_dynamic_global_batch_size", False):
        problems.append("--use-dynamic-global-batch-size")
    if getattr(args, "indep_dp", False):
        problems.append("--indep-dp")
    if getattr(args, "multi_lora", False):
        problems.append("multi-LoRA")
    if getattr(args, "multimodal_keys", None):
        problems.append("--multimodal-keys")
    backend = getattr(args, "train_backend", None)
    if backend not in (None, "megatron"):
        problems.append(f"--train-backend {backend}")
    return problems


def check_segmented_round(args: Any, groups: Sequence[tuple[int, Sequence[dict[str, Any]]]]) -> None:
    """Refuse a round (its converted ``groups``) with multi-sample trials that Miles would
    not train one rollout per trial (:func:`segmented_round_problems`), or whose
    segments can be wider than this trainer's ``--seq-length`` (:func:`segment_row_width`:
    the sampler's ``max_model_len``, else ``context_budget``; its summary, at the end,
    would be cut). Then the sibling check of the
    loss side (``yeto.rl.algos.segment_weights.sibling_problems``) over all
    the round's samples. A round without such trials: nothing is checked."""
    def refusal(positions: list[int]) -> str | None:
        problems = segmented_round_problems(args)
        if not problems:
            return None
        return (f"groups at buffer positions {positions[:8]} hold compacted trials (several samples "
                f"per trial); Miles trains them one rollout per trial only on its rollout-side DP schedule, "
                f"which this trainer does not use ({', '.join(problems)})")

    check_compaction_round(groups, seq_length=getattr(args, "seq_length", None), schedule_refusal=refusal)


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
                    positions: Sequence[int | None] | None = None,
                    extra: Sequence[Mapping[str, Any] | None] | None = None) -> int:
        """Write ``groups`` (``(group key, samples)``, aborted groups already left
        out) of round ``rollout_id``, with one store commit at the end; returns the
        groups written (the first ones, in order). ``positions``: each
        group's task-dispenser position, kept in its round record; ``extra``: each
        group's further round record fields (a store ledger slot's
        ``dispense_attempt`` and ``dispense_plan``)."""
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
                fields = dict((extra[i] if extra is not None else None) or {})
                record = SamplerRound(self.sampler_id, self.incarnation, int(rollout_id),
                                      data_cursor=data_cursor, last=i == len(groups) - 1,
                                      dispense_position=None if positions is None else positions[i],
                                      dispense_attempt=fields.get("dispense_attempt"),
                                      dispense_plan=fields.get("dispense_plan"))
                self._writer.write(samples, group_key=key, sampler_round=record.record())
                written += 1
        if written and self.guard is not None:
            self.guard.written(rollout_id)
        return written


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


def newest_release_identity(buffer: Any) -> dict[str, Any] | None:
    """``{"seq", "policy_hash"}`` of the newest policy release in ``buffer``'s store
    (``policy-releases/LATEST.json``, where a trainer island releases), None when there
    is none yet. A read the store does not answer raises."""
    from yeto.rl.engine.policy_release import LATEST
    from yeto.rl.engine.sample_buffer import buffer_objects
    from yeto.rl.quant.schemes import RELEASE_DIR

    objects = buffer_objects(buffer)
    if objects is not None:
        raw = objects.read(f"{RELEASE_DIR}/{LATEST}")
    else:
        root = getattr(getattr(buffer, "store", None), "root", None)
        path = None if root is None else Path(root) / RELEASE_DIR / LATEST
        raw = path.read_bytes() if path is not None and path.is_file() else None
    if raw is None:
        return None
    record = json.loads(raw)
    return {"seq": int(record["seq"]), "policy_hash": str(record["policy_hash"])}


def trainer_read_heartbeat(buffer: Any, *, clock: Callable[[], float] = time.time
                           ) -> Callable[[int, dict[str, int] | None], str]:
    """The ``read_heartbeat`` of a trainer island's :class:`BatchReader`: writes the
    read heartbeat (``sample_buffer.write_buffer_read_position``: the object
    ``sample-buffer/read-position.json`` on an object store, else the file) naming the
    newest release of the store, so a sampler that follows this trainer counts its
    backlog from where the trainer reads (``sampler_loop.heartbeat_read``) instead of
    from the read position the newest release recorded."""
    from yeto.rl.engine.sample_buffer import write_buffer_read_position

    def beat(cursor: int, shards: dict[str, int] | None) -> str:
        return write_buffer_read_position(buffer, buffer_selected=int(cursor), buffer_shards=shards,
                                          train_step=None, release=newest_release_identity(buffer),
                                          clock=clock)

    return beat


def reader_positions(reader: BatchReader) -> tuple[int, dict[str, int] | None]:
    """``(read cursor, read cursor per sampler shard)`` of ``reader``, read together
    (the shards None when the buffer has no merged shards)."""
    cursor = int(reader.buffer.read_cursor)
    shards = reader.buffer.shard_positions(cursor) or None
    return cursor, None if shards is None else {str(k): int(v) for k, v in shards.items()}


def executor_read_positions(_executor: Any) -> tuple[int, dict[str, int] | None] | None:
    """Runs INSIDE the rollout executor actor: :func:`reader_positions` of the
    trainer's reader (None before the reader exists)."""
    reader = _ACTIVE.get(TRAINER)
    return None if reader is None else reader_positions(reader)


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
            if any(isinstance(d.get("reward"), bool) or not isinstance(d.get("reward"), (int, float))
                   for d in group):
                raise SampleBufferRolloutError(
                    f"xhw_guard downweight of buffer position {position}: a reward is not a scalar "
                    "(a reward dict cannot be scaled; use action=quarantine)")
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


def check_xhw_downweight(*, std_normalized: bool, environ: Mapping[str, str] | None = None,
                         args: Any = None) -> None:
    """At the reader's start: ``YETO_RL_XHW_GUARD`` with ``action=downweight`` under GRPO
    std normalization (Miles' default) is refused here, not at the first round the guard
    trips (:func:`xhw_round` cannot apply it then, and the trainer would stop mid-run).

    The downweight rewrites a group's rewards as ``c + w (r - c)``; Miles' per-rollout
    centering cancels ``c``, so every advantage of the group is exactly ``w`` times its
    own. That holds only for scalar rewards centered as they are: with the Miles ``args``
    also refused are a length penalty (``--length-penalty-coeff``: added after the
    scaling, and its correctness test ``reward == 1`` no longer holds), a reward dict
    (``--reward-key``), and a yeto reward shaper or an advantage transform other than
    ``grpo_default`` (``advantage.reward_shapers`` / ``advantage.transform``)."""
    guard = trainer_breakers.xhw_guard(trainer_breakers.resolved(environ, names=(trainer_breakers.XHW_GUARD,)))
    if guard is None or guard.action != "downweight":
        return
    if std_normalized:
        raise SampleBufferRolloutError(
            "YETO_RL_XHW_GUARD action=downweight scales the centered rewards; GRPO std "
            "normalization (on) cancels it: use action=quarantine, or turn std normalization off")
    if args is None:
        return
    from yeto.rl.algos.reward_pipeline import DEFAULT_TRANSFORM, read_plugins  # noqa: PLC0415

    problems = []
    if float(getattr(args, "length_penalty_coeff", 0.0) or 0.0) > 0:
        problems.append("--length-penalty-coeff > 0")
    if getattr(args, "reward_key", None):
        problems.append("--reward-key (a reward dict)")
    pipeline = dict((read_plugins(args, required=False) or {}).get("reward_pipeline") or {})
    if pipeline.get("reward_shapers"):
        problems.append(f"reward shapers {[dict(s).get('name') for s in pipeline['reward_shapers']]}")
    if pipeline.get("advantage_transform") not in (None, DEFAULT_TRANSFORM):
        problems.append(f"advantage transform {pipeline['advantage_transform']!r}")
    if problems:
        raise SampleBufferRolloutError(
            "YETO_RL_XHW_GUARD action=downweight scales a group's advantages exactly only for "
            f"scalar GRPO-centered rewards; this trainer has {', '.join(problems)}: use action=quarantine")


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

    def __init__(self, in_executor: Callable[..., Any],
                 on_positions: Callable[[Any], None] | None = None) -> None:
        """``on_positions(positions)``: told the reader's positions
        (:func:`executor_read_positions`) after every restore, which moves the read
        cursor (``MilesRolloutPool.note_buffer_read_positions``: the release records
        read that cache instead of the executor)."""
        self.in_executor = in_executor
        self.on_positions = on_positions

    def watermark(self, rollout_id: int) -> int:
        return int(self.in_executor(executor_watermark_before, int(rollout_id)))

    def resume(self, watermark: int) -> int:
        landed = int(self.in_executor(executor_reset, int(watermark)))
        if self.on_positions is not None:
            self.on_positions(self.in_executor(executor_read_positions))
        return landed


def wire_sample_buffer(checkpoint: Any, rollout: Any) -> BufferCutLink:
    """Give a ``RoundCutCheckpoint`` the trainer's buffer watermark and restore hook."""
    in_executor = getattr(rollout, "_in_executor", None)
    if not callable(in_executor):
        raise SampleBufferRolloutError(f"rollout pool {type(rollout).__name__} cannot run "
                                       "code in the rollout executor")
    link = BufferCutLink(in_executor, getattr(rollout, "note_buffer_read_positions", None))
    checkpoint.buffer_watermark = link.watermark
    checkpoint.buffer_resume = link.resume
    return link
