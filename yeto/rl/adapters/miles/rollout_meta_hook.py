"""Metadata extraction that runs *inside* the Miles rollout process (design D3).

Upstream rollout functions call, at the end of every training rollout::

    rollout_sample_filter_path(args, data)                  # the kept groups
    rollout_all_samples_process_path(args, all_samples, data_source)

``record_trained_groups`` remembers which groups were actually kept (the
dynamic-sampling filter plus the batch-size cap decide this, and only ``data``
reflects the cap), without modifying them. ``extract_rollout_metadata`` then
reduces every group to a small JSON-able record (ids, policy token, reward
summary, token counts) and ships it to a sink; tokens, logprobs and tensors
never leave the rollout process.

Sinks (``YETO_ROLLOUT_META_SINK``):

* ``ray:<actor_name>`` (default ``ray:yeto_rollout_meta``) -- a named Ray actor
  created by :class:`~.rollout.RayMetadataSink` in the driver;
* ``dir:<path>`` -- one atomic JSON file per rollout (single-node / tests).

The same sink carries the other direction for one value: the driver-side
``MilesRolloutPool`` stores the published policy token
(``yeto:<rollout_id>:<policy_tensor_hash>``) before each rollout, and
``policy_buffer_filter`` (``--buffer-filter-path``) reads it inside the rollout
process to keep only complete groups of exactly that policy -- the ports
equivalent of legacy ``_complete_group_for_policy`` group-reuse filtering.

Import-light: nothing here imports miles or torch.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

META_SINK_ENV = "YETO_ROLLOUT_META_SINK"
DEFAULT_SINK_ACTOR = "yeto_rollout_meta"
DEFAULT_SINK = f"ray:{DEFAULT_SINK_ACTOR}"
METADATA_SCHEMA = "yeto-rollout-meta-v1"
_TRAINED_ATTR = "_yeto_trained_group_keys"
# S19 #8: kept-sample stop tokens masked for a 0.0 placeholder logprob this rollout.
_PLACEHOLDER_ATTR = "_yeto_placeholder_logprob_tokens"
_BOUNDED_FILTER_STATE_ATTR = "_yeto_bounded_filter_state"
# Per-round algorithm counters reported by rollout-side algorithm code AFTER
# the all-samples hook ran (e.g. the rl-algo-seq-and-adv reward dispatcher's
# non-zero advantage count, computed in reward post-processing). They travel
# as a separate sink record keyed by rollout_id and are merged into that
# rollout's metadata by the driver-side reader, so they are never attributed
# to the next round.
ROUND_META_SCHEMA = "yeto-rl-round-metadata-v1"
ROUND_METADATA_KEYS = frozenset({"nonzero_advantages"})


def current_round_id(samples: Sequence[Any] = (), sink: str | None = None) -> int | None:
    """The training round of the rollout being processed.

    Authoritative source: the policy token the driver published for this
    rollout (``yeto:<rollout_id>:<hash>``). ``Sample.rollout_id`` is NOT used
    when a token exists: in multi-segment agentic rollouts it is a trajectory
    key (Miles ``agentic_tool_call.py``), not the round. Without a token (unit
    fixtures, legacy) the first sample's ``rollout_id`` is the fallback.
    """
    try:
        token = current_policy_token(sink)
    except ImportError:  # no Ray in this process (unit fixtures): no published token
        token = None
    if token:
        from yeto.rl.core import parse_policy_snapshot_token

        return parse_policy_snapshot_token(token)[0]
    return next(
        (s.rollout_id for s in samples if getattr(s, "rollout_id", None) is not None), None
    )


def record_round_metadata(
    args: Any, samples_or_rollout_id: Any, *, sink: str | None = None, **counts: int
) -> None:
    """Rollout side: report this rollout's per-round counters (call once per rollout)."""

    del args
    unknown = sorted(set(counts) - ROUND_METADATA_KEYS)
    if unknown:
        raise RuntimeError(f"unknown per-round metadata keys {unknown}")
    for key, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"per-round metadata {key} must be a non-negative int")
    if isinstance(samples_or_rollout_id, int) and not isinstance(samples_or_rollout_id, bool):
        rollout_id = samples_or_rollout_id
    else:
        rollout_id = current_round_id(_flat(samples_or_rollout_id), sink)
    if rollout_id is None:
        raise RuntimeError("per-round metadata needs the rollout id")
    put_to_sink({"schema": ROUND_META_SCHEMA, "rollout_id": int(rollout_id), **counts}, sink)
POLICY_TOKEN_FILE = "policy-token"
_REUSABLE_STATUSES = frozenset({"completed", "truncated"})


def _flat(group: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in group:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def _group_key(group: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(getattr(s, "index", None) for s in _flat(group))


def _load_miles_tokenizer(args: Any) -> Any:
    from miles.utils.processing_utils import load_tokenizer  # cached by Miles

    return load_tokenizer(args.hf_checkpoint, trust_remote_code=True)


def record_trained_groups(args: Any, data: Sequence[Sequence[Any]]) -> None:
    """``--rollout-sample-filter-path`` hook: remember the kept groups.

    Shared by rl-algo-grpo-knobs (sample filters, D7) and the rl-infra-spec
    3.6 ledger (alignment A2/F5). Spec-selected sample filters run first
    (``yeto.rl.algos.sample_filters``: overlong filter sets
    ``remove_sample=True`` on truncated samples; default config: untouched).
    Filtered samples are the ledger's terminal ``filtered``. Over-sampling
    leaves no reusable remainder: Miles does not return surplus kept groups
    to its buffer (``sglang_rollout.py:505-510``); only samples aborted under
    ``--partial-rollout`` go back.
    """

    from yeto.rl.algos.sample_filters import apply_sample_filters
    from yeto.rl.harness.reasoning_loss_mask import apply_from_args as apply_reasoning_loss_policy

    apply_sample_filters(args, data)
    # Side effect beyond filtering (S17 WP6): with --codex-exclude-reasoning-from-loss
    # this sets loss_mask 1 -> 0 on generated reasoning tokens of the kept samples
    # (before Miles converts them to train data). Default: returns at once, no sample touched.
    apply_reasoning_loss_policy(args, data, tokenizer_loader=_load_miles_tokenizer)
    # S19 #8: a stop token forced by the engine's grammar mask (SGLang returns
    # logprob exactly 0.0 under the masked distribution) is not a model choice:
    # loss_mask 1 -> 0 before Miles converts the samples to train data.
    from yeto.rl.harness.placeholder_logprob_mask import apply_to_groups as mask_placeholder_logprobs

    setattr(args, _PLACEHOLDER_ATTR,
            mask_placeholder_logprobs(data, tokenizer_loader=_load_miles_tokenizer, args=args))
    # CompactionRL: the masks above may drop optimised tokens; the per-segment
    # counts (tokens_after, gae_length) must follow the final masks.
    from yeto.rl.harness.codex.codex_openenv_generate import refresh_segment_counts

    refresh_segment_counts([s for group in data for s in _flat(group)])
    setattr(args, _TRAINED_ATTR, {_group_key(group) for group in data})


def _reward(args: Any, sample: Any) -> float:
    getter = getattr(sample, "get_reward_value", None)
    value = getter(args) if callable(getter) else getattr(sample, "reward", None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def _versions(sample: Any) -> set[str]:
    versions: set[str] = set()
    for call in getattr(sample, "weight_versions", None) or ():
        for span in getattr(call, "spans", None) or ():
            versions.add(str(span.version))
    if not versions:
        meta = getattr(sample, "metadata", None) or {}
        legacy = meta.get("weight_version")
        if legacy is not None:
            versions.add(str(legacy))
    return versions


def _status(sample: Any) -> str:
    status = getattr(sample, "status", None)
    return str(getattr(status, "value", status) or "")


from yeto.rl.adapters.miles import carry_over as _carry  # noqa: E402
from yeto.rl.engine.policy_age import token_version  # noqa: E402


def group_record(args: Any, group: Sequence[Any]) -> dict[str, Any]:
    samples = _flat(group)
    indices = [getattr(s, "index", None) for s in samples]
    group_index = getattr(samples[0], "group_index", None) if samples else None
    if group_index is None:
        # the 3.6 ledger keys groups on Miles' monotonic sample_group_index;
        # a sample index is not a group identity (fail closed)
        raise RuntimeError(
            f"rollout group without group_index (sample indices {indices}); "
            "the yeto ledger needs Miles' data-source group_index"
        )
    versions: set[str] = set()
    for s in samples:
        versions |= _versions(s)
    if len(versions) == 1:
        token = next(iter(versions))
    elif not versions:
        token = ""
    else:  # mixed policies inside one group can never match one snapshot
        token = "mixed:" + "|".join(sorted(versions))
    carry = {}
    if versions and _carry.max_policy_age(args) > 0:
        # agentic-rollout-utilization 4.1: under a limit > 0 a carried-over group
        # spans versions; its token names the OLDEST one (the driver checks it
        # is a published version within the window) and its version segments
        # are listed. Unparsable tokens keep "mixed:" (refused, fail closed).
        numbered = {v: token_version(v) for v in versions}
        if None not in numbered.values():
            token = min(versions, key=lambda v: numbered[v])
            carry = _carry.group_versions_record(group)
    rewards = [_reward(args, s) for s in samples]
    finite = [r for r in rewards if math.isfinite(r)]
    return {
        "group_id": f"g{group_index}",
        "sample_ids": [f"s{i}" for i in indices],
        "policy_token": token,
        "reward_mean": statistics.fmean(finite) if finite else math.nan,
        "reward_std": statistics.pstdev(finite) if finite else math.nan,
        "token_count": int(
            sum(int(getattr(s, "effective_response_length", None) or getattr(s, "response_length", 0) or 0) for s in samples)
        ),
        "aborted": any(_status(s) == "aborted" for s in samples),
        "_key": list(_group_key(group)),
        **carry,
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile (numpy default); None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _advantage(sample: Any) -> float | None:
    for name in ("advantage", "advantages"):
        value = getattr(sample, name, None)
        if value is None:
            value = (getattr(sample, "metadata", None) or {}).get(name)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = statistics.fmean(value) if value else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None
    return None


def batch_summary(args: Any, samples: Sequence[Any]) -> dict[str, float | None]:
    """fleet-dashboard 1.3: summary of the trained samples of one rollout.

    Advantages are computed train-side by Miles; ``adv_*`` is filled only when
    the samples carry an ``advantage`` value (else None, never inferred).
    """
    lengths = [
        float(getattr(s, "effective_response_length", None) or getattr(s, "response_length", 0) or 0)
        for s in samples
    ]
    rewards = [r for r in (_reward(args, s) for s in samples) if math.isfinite(r)]
    advs = [a for a in (_advantage(s) for s in samples) if a is not None]
    n = len(samples)
    return {
        "adv_mean": statistics.fmean(advs) if advs else None,
        "adv_std": statistics.pstdev(advs) if advs else None,
        "resp_len_mean": statistics.fmean(lengths) if lengths else None,
        "resp_len_p95": _quantile(lengths, 0.95),
        "truncated_frac": sum(_status(s) == "truncated" for s in samples) / n if n else None,
        "reward_p10": _quantile(rewards, 0.10),
        "reward_p50": _quantile(rewards, 0.50),
        "reward_p90": _quantile(rewards, 0.90),
    }


def _bucket_of(sample: Any) -> str:
    meta = getattr(sample, "metadata", None) or {}
    value = meta.get("difficulty") if isinstance(meta, dict) else None
    return str(value) if value not in (None, "") else "unknown"


def batch_summary_by_bucket(args: Any, samples: Sequence[Any]) -> dict[str, dict[str, float | int | None]]:
    """rl-eval-difficulty-buckets 4.1 (D5): trained samples grouped by their row's
    ``metadata.difficulty`` (missing = ``unknown``): count, reward mean, success
    rate (reward > 0), truncated fraction, mean response length. Scalars only."""
    groups: dict[str, list[Any]] = {}
    for s in samples:
        groups.setdefault(_bucket_of(s), []).append(s)
    out: dict[str, dict[str, float | int | None]] = {}
    for bucket in sorted(groups):
        rows = groups[bucket]
        rewards = [r for r in (_reward(args, s) for s in rows) if math.isfinite(r)]
        lengths = [float(getattr(s, "effective_response_length", None) or getattr(s, "response_length", 0) or 0)
                   for s in rows]
        n = len(rows)
        out[bucket] = {
            "n": n,
            "reward_mean": statistics.fmean(rewards) if rewards else None,
            "success_rate": sum(r > 0 for r in rewards) / len(rewards) if rewards else None,
            "truncated_frac": sum(_status(s) == "truncated" for s in rows) / n,
            "resp_len_mean": statistics.fmean(lengths) if lengths else None,
        }
    return out


BATCH_SUMMARY_ENV = "YETO_RL_BATCH_SUMMARY"


def batch_summary_enabled(args: Any) -> bool:
    """Opt-in (``--rl-observe-timeline`` sets ``args.yeto_rl_observe_timeline``,
    or ``YETO_RL_BATCH_SUMMARY=1``): the default metadata key set is unchanged."""
    return bool(getattr(args, "yeto_rl_observe_timeline", False)) or os.environ.get(
        BATCH_SUMMARY_ENV) == "1"


def build_metadata(
    args: Any, all_samples: Iterable[Sequence[Any]], sink: str | None = None
) -> dict[str, Any]:
    trained = getattr(args, _TRAINED_ATTR, None)
    if trained is None:
        raise RuntimeError(
            "rollout metadata hook ran without record_trained_groups; "
            "--rollout-sample-filter-path must be the yeto recorder"
        )
    from yeto.rl.algos.sample_filters import group_filtered_samples, metadata_fields

    extra = metadata_fields(args)
    sample_filter_counts = extra.get("filtered_samples")
    all_samples = list(all_samples)  # iterated twice (groups, then harness counters)
    tool_wait = 0.0
    rollout_id = None
    groups, filtered, aborted = [], 0, 0
    trained_samples: list[Any] = []
    for group in all_samples:
        samples = _flat(group)
        if not samples:
            continue
        if rollout_id is None:
            rollout_id = current_round_id(samples, sink)
        record = group_record(args, group)
        tool_wait += sum(float(getattr(x, "non_generation_time", 0.0) or 0.0) for x in samples)
        aborted += int(record.pop("aborted"))
        key = tuple(record.pop("_key"))
        if key in trained:
            if sample_filter_counts:
                # terminal ledger state ``filtered`` (alignment A2/F5)
                record["filtered_samples"] = group_filtered_samples(group)
            groups.append(record)
            trained_samples.extend(samples)
        else:
            filtered += 1
    if len(groups) != len(trained):
        raise RuntimeError(
            f"trained groups ({len(trained)}) not all present in all_samples ({len(groups)} matched)"
        )
    groups.sort(key=lambda g: g["group_id"])
    payload = {
        "schema": METADATA_SCHEMA,
        "rollout_id": rollout_id,
        "groups": groups,
        "completed": len(groups),
        "filtered": filtered,
        "aborted": aborted,
        "trained_sample_indices": sorted(
            int(i[1:]) for g in groups for i in g["sample_ids"] if i[1:].lstrip("-").isdigit()
        ),
        **extra,
    }
    placeholder = int(getattr(args, _PLACEHOLDER_ATTR, 0) or 0)
    if placeholder:  # S19 #8: absent when 0 (old key set kept)
        from yeto.rl.harness.placeholder_logprob_mask import COUNT_KEY as _PLACEHOLDER_KEY

        payload[_PLACEHOLDER_KEY] = placeholder
    limit = _carry.max_policy_age(args)
    if limit > 0:
        # agentic-rollout-utilization 4.1: carry-over accounting (limit > 0 only;
        # the default key set is unchanged).
        payload.update(carry_over_fields(args, trained_samples, sink))
        # 5.1/5.3: agentic suspension between turns (absent when not in use)
        payload.update(_carry.suspend_fields(args, all_samples))
    if tool_wait > 0:
        # 1.7: time trajectories spent outside generation (tool calls), summed
        # over every generated sample (Miles Sample.non_generation_time).
        # Absent when no sample reported any: the default key set is unchanged.
        payload["tool_wait_seconds"] = tool_wait
    if trained_samples and batch_summary_enabled(args):
        payload["batch_summary"] = batch_summary(args, trained_samples)
        if any(_bucket_of(s) != "unknown" for s in trained_samples):
            # 4.1: only when rows carry a difficulty (old key set kept otherwise)
            payload["batch_summary_by_bucket"] = batch_summary_by_bucket(args, trained_samples)
    if trained_samples and batch_summary_enabled(args):
        # rl-fn-codex-rollout 1.0 (observe only): one record per trained sample
        # (task_id / trajectory_id / reward / signed success) so the tape can
        # tell *which* task scored (``rl_trajectory_reward``). Absent on the
        # default path (old key set kept).
        payload[TRAJECTORY_REWARDS_KEY] = trajectory_reward_records(
            args, all_samples, trained, limit=trajectory_records_limit(args))
    harness = harness_counters(all_samples)
    if harness:  # IR-3/IR-4: absent when no sample reported any (old key set kept)
        payload.update(harness)
    if harness.get(TITO_SESSION_MISMATCH_KEY) and batch_summary_enabled(args):
        # S14-M1 (observe only): the per-record mismatches, truncated and capped,
        # so the driver can tape them as ``rl_harness_mismatch``. Absent on the
        # default path and when there is nothing to report (old key set kept).
        records = harness_mismatch_records(
            all_samples, limit=mismatch_records_limit(args), text_limit=mismatch_text_limit(args))
        if records:
            payload[TITO_SESSION_MISMATCH_RECORDS_KEY] = records
    return payload


def carry_over_fields(args: Any, trained_samples: Sequence[Any],
                      sink: str | None = None, scorer: Any = None) -> dict[str, Any]:
    """4.1 rollout-metadata fields under a limit > 0: what the buffer filter carried
    in / discarded this round, the trained cross-version tokens and the estimated
    cross-version IS truncated fraction (:mod:`.carry_over`)."""
    token = current_policy_token(sink)
    version = token_version(token) if token else None
    fields: dict[str, Any] = {"max_policy_age": _carry.max_policy_age(args)}
    fields.update(_carry.take_carry_stats(args, version))
    if version is None:
        return fields
    crossed, total = _carry.cross_version_tokens(trained_samples, version)
    fields["cross_version_tokens"] = crossed
    fields["trained_response_tokens"] = total
    if crossed:
        fields.update(_carry.estimate_cross_version_truncation(
            args, trained_samples, version,
            scorer if scorer is not None else _carry.router_prefill_scorer(args)))
    return fields


# IR-3/IR-4 sample-metadata keys written by agentic generate code (codex-harness
# codex_openenv_generate): summed per rollout into the metadata payload.
# Neutral names and values live in the core port (decoupling 4.11); re-exported.
from yeto.rl.engine.rollout_meta import (  # noqa: E402
    EXPECTED_POLICY_VERSION_KEY,
    POLICY_AGE_VIOLATION_KEY,
    TITO_CHAIN_BREAKS_KEY,
    TITO_SESSION_MISMATCH_KEY,
    counter_value,
)


def expected_policy_version(sample: Any = None, sink: str | None = None) -> str | None:
    """Rollout side (IR-3): the driver's target policy token for this rollout.

    The prompt sample's metadata ``expected_policy_version`` wins when the
    data source carries it; otherwise the token the driver published through
    the metadata sink (``MilesRolloutPool.generate`` -> ``set_policy_token``).
    None = the driver did not publish one (agentic generate must refuse).
    """
    meta = getattr(sample, "metadata", None) if sample is not None else None
    if isinstance(meta, dict) and meta.get(EXPECTED_POLICY_VERSION_KEY):
        return str(meta[EXPECTED_POLICY_VERSION_KEY])
    return current_policy_token(sink)


TITO_SESSION_MISMATCH_RECORDS_KEY = "tito_session_mismatch_records"
MISMATCH_RECORDS_MAX_ENV = "YETO_RL_MISMATCH_TAPE_MAX"
MISMATCH_TEXT_MAX_ENV = "YETO_RL_MISMATCH_TEXT_MAX"


def _limit_from(args: Any, attr: str, env: str, default: int) -> int:
    value = getattr(args, attr, None)
    if value is None:
        value = os.environ.get(env)
    try:
        return max(0, int(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def mismatch_records_limit(args: Any = None) -> int:
    """Per-round cap on taped mismatch records (``args.yeto_rl_mismatch_tape_max``
    or ``YETO_RL_MISMATCH_TAPE_MAX``; default HARNESS_MISMATCH_MAX_PER_ROUND)."""
    from yeto.rl.engine.timeline import HARNESS_MISMATCH_MAX_PER_ROUND

    return _limit_from(args, "yeto_rl_mismatch_tape_max", MISMATCH_RECORDS_MAX_ENV,
                       HARNESS_MISMATCH_MAX_PER_ROUND)


def mismatch_text_limit(args: Any = None) -> int:
    """Per-field text cap (``args.yeto_rl_mismatch_text_max`` or
    ``YETO_RL_MISMATCH_TEXT_MAX``; default HARNESS_MISMATCH_TEXT_MAX)."""
    from yeto.rl.engine.timeline import HARNESS_MISMATCH_TEXT_MAX

    return _limit_from(args, "yeto_rl_mismatch_text_max", MISMATCH_TEXT_MAX_ENV,
                       HARNESS_MISMATCH_TEXT_MAX)


def _clip(value: Any, limit: int) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    return (text[:limit], True) if len(text) > limit else (text, False)


def _int_or(value: Any, default: int) -> int:
    try:
        return default if value is None or isinstance(value, bool) else int(value)
    except (TypeError, ValueError):
        return default


def harness_mismatch_records(
    all_samples: Iterable[Sequence[Any]], *, limit: int, text_limit: int,
) -> list[dict[str, Any]]:
    """S14-M1: the per-record ``tito_session_mismatch`` dicts of every sample
    (upstream ``Mismatch.to_dict``: type/segment_index/expected_text/actual_text/
    detail), flattened in sample order, texts cut to ``text_limit`` chars and at
    most ``limit`` records. Int / bool / dict values of the key (harness bridge
    counters) carry no records and are skipped."""
    out: list[dict[str, Any]] = []
    if limit <= 0:
        return out
    for group in all_samples:
        for s in _flat(group):
            meta = getattr(s, "metadata", None)
            if not isinstance(meta, dict):
                continue
            value = meta.get(TITO_SESSION_MISMATCH_KEY)
            if not isinstance(value, (list, tuple)):
                continue
            for i, rec in enumerate(value):
                if len(out) >= limit:
                    return out
                if not isinstance(rec, dict):
                    rec = {"detail": repr(rec)}
                expected, cut_e = _clip(rec.get("expected_text"), text_limit)
                actual, cut_a = _clip(rec.get("actual_text"), text_limit)
                detail, cut_d = _clip(rec.get("detail"), text_limit)
                kind = rec.get("type", rec.get("kind"))
                kind = getattr(kind, "value", kind)
                out.append({
                    "sample_index": _int_or(getattr(s, "index", None), -1),
                    "group_index": _int_or(getattr(s, "group_index", None), -1),
                    "record_index": i,
                    "kind": "" if kind is None else str(kind),
                    "segment_index": _int_or(rec.get("segment_index"), -1),
                    "expected_text": expected,
                    "actual_text": actual,
                    "detail": detail,
                    "truncated": bool(cut_e or cut_a or cut_d),
                })
    return out


TRAJECTORY_REWARDS_KEY = "trajectory_rewards"
TRAJECTORY_RECORDS_ENV = "YETO_RL_TRAJECTORY_TAPE_MAX"
TRAJECTORY_RECORDS_DEFAULT = 256


def trajectory_records_limit(args: Any = None) -> int:
    return _limit_from(args, "yeto_rl_trajectory_tape_max", TRAJECTORY_RECORDS_ENV, TRAJECTORY_RECORDS_DEFAULT)


def trajectory_reward_records(
    args: Any, all_samples: Iterable[Sequence[Any]], trained: Any = None, *, limit: int,
) -> list[dict[str, Any]]:
    """rl-fn-codex-rollout 1.0 (observe only): per-sample reward records of the
    trained groups (every group when ``trained`` is None): Miles sample/group
    index, ``metadata["task_id"]`` (Terminal-Bench task), ``metadata["trajectory_id"]``
    (or ``sample.rollout_id``: siblings of one trajectory share a reward, R-D5a),
    the reward the trainer sees, the signed ``metadata["success"]`` bit (None =
    no verdict) and the aborted flag; at most ``limit`` records in sample order."""
    out: list[dict[str, Any]] = []
    if limit <= 0:
        return out
    carry_round = current_round_id((), None) if _carry.max_policy_age(args) > 0 else None
    for group in all_samples:
        samples = _flat(group)
        if not samples or (trained is not None and _group_key(group) not in trained):
            continue
        for s in samples:
            if len(out) >= limit:
                return out
            meta = getattr(s, "metadata", None)
            meta = meta if isinstance(meta, dict) else {}
            reward = _reward(args, s)
            success = meta.get("success")
            trajectory = meta.get("trajectory_id")
            if trajectory is None:
                trajectory = getattr(s, "rollout_id", None)
            out.append({
                "sample_index": _int_or(getattr(s, "index", None), -1),
                "group_index": _int_or(getattr(s, "group_index", None), -1),
                "task_id": "" if meta.get("task_id") is None else str(meta.get("task_id")),
                "trajectory_id": "" if trajectory is None else str(trajectory),
                "reward": float(reward) if math.isfinite(reward) else None,
                "success": success if isinstance(success, bool) else None,
                "aborted": _status(s) == "aborted",
                **trajectory_diagnostics(meta),
                **(_carry.trajectory_fields(s, carry_round)
                   if _carry.max_policy_age(args) > 0 else {}),
            })
    return out


_DIAGNOSTIC_COUNTERS = ("turns", "terminal_calls", "submit_calls", "parse_failures",
                        "max_seq_len_hit", "timed_out")


def trajectory_diagnostics(meta: dict[str, Any]) -> dict[str, Any]:
    """S15 stage-2 follow-up: Codex exit status, agent counters and the signed
    ``testsh_rc`` from the harness metadata; only the keys that are present."""
    out: dict[str, Any] = {}
    status = meta.get("exit_status")
    if isinstance(status, str) and status:
        out["exit_status"] = status[:64]
    metrics = meta.get("agent_metrics")
    if isinstance(metrics, dict):
        for key in _DIAGNOSTIC_COUNTERS:
            value = metrics.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                out[key] = value
        reason = metrics.get("end_reason")
        if isinstance(reason, str) and reason:
            out["end_reason"] = reason[:160]
        etime = metrics.get("evaluate_time")  # S17 N13: seconds spent grading
        if isinstance(etime, (int, float)) and not isinstance(etime, bool) and etime >= 0:
            out["evaluate_time"] = round(float(etime), 3)
        kind = metrics.get("end_kind")  # S17 C9
        if isinstance(kind, str) and kind:
            out["end_kind"] = kind[:32]
        for key in ("turn_completion_tokens", "turn_context_tokens", "turn_tool_output_bytes"):
            values = metrics.get(key)
            if isinstance(values, list):
                out[key] = [v if isinstance(v, int) and not isinstance(v, bool) else None
                            for v in values[:256]]
        last = metrics.get("last_completion")
        if isinstance(last, dict):
            for key, limit in (("finish_reason", 16), ("tool_names", 96), ("content_head", 200), ("reasoning_tail", 200)):
                if isinstance(last.get(key), str):
                    out[f"last_{key}"] = last[key][:limit]
            for key in ("content_chars", "reasoning_chars", "tool_calls"):
                value = last.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    out[f"last_{key}"] = value
            tokens = last.get("completion_tokens")
            if tokens is None or (isinstance(tokens, int) and not isinstance(tokens, bool)):
                out["last_completion_tokens"] = tokens
    from yeto.rl.engine.trajectory_timing import timing_fields

    out.update(timing_fields(meta))  # agentic-rollout-utilization 1.3
    log = meta.get("verifier_log")
    if isinstance(log, str):
        out["verifier_log"] = log[-2000:]
    outcome = meta.get("tbench_trusted_outcome")
    if isinstance(outcome, dict) and "testsh_rc" in outcome:
        rc = outcome.get("testsh_rc")
        out["testsh_rc"] = rc if isinstance(rc, int) and not isinstance(rc, bool) else None
    return out


def harness_counters(all_samples: Iterable[Sequence[Any]]) -> dict[str, Any]:
    """Sum the IR-3/IR-4 per-sample counters; only keys with a non-zero total."""
    age = mismatch = 0
    breaks: dict[str, int] = {}
    for group in all_samples:
        for s in _flat(group):
            meta = getattr(s, "metadata", None)
            if not isinstance(meta, dict):
                continue
            age += counter_value(meta.get(POLICY_AGE_VIOLATION_KEY))
            mismatch += counter_value(meta.get(TITO_SESSION_MISMATCH_KEY))
            for reason, n in (meta.get(TITO_CHAIN_BREAKS_KEY) or {}).items():
                breaks[str(reason)] = breaks.get(str(reason), 0) + int(n or 0)
    out: dict[str, Any] = {}
    if age:
        out[POLICY_AGE_VIOLATION_KEY] = age
    if mismatch:
        out[TITO_SESSION_MISMATCH_KEY] = mismatch
    if breaks:
        out[TITO_CHAIN_BREAKS_KEY] = breaks
    return out


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


def _jsonable(payload: dict[str, Any]) -> str:
    def fix(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: fix(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fix(v) for v in value]
        return value

    return json.dumps(fix(payload), sort_keys=True, separators=(",", ":"))


def put_to_sink(payload: dict[str, Any], sink: str | None = None) -> None:
    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    encoded = _jsonable(payload)
    if kind == "dir":
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        rid = payload.get("rollout_id")
        prefix = "round" if payload.get("schema") == ROUND_META_SCHEMA else "rollout"
        name = f"{prefix}-{rid if rid is not None else 'latest'}.json"
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        os.replace(tmp, directory / name)
        return
    if kind == "ray":
        import ray

        ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).put.remote(encoded))
        return
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def put_policy_token(token: str, sink: str | None = None) -> None:
    """Driver side: publish the token the next rollout must be sampled from."""

    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    if kind == "dir":
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        os.replace(tmp, directory / POLICY_TOKEN_FILE)
        return
    if kind == "ray":
        import ray

        ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).set_token.remote(token))
        return
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def sink_available() -> bool:
    """Whether a metadata sink is reachable here (neutral port, decoupling 4.11).

    An explicit ``YETO_ROLLOUT_META_SINK`` counts; otherwise the default Ray
    actor sink needs an initialised Ray in this process.
    """

    if os.environ.get(META_SINK_ENV):
        return True
    try:
        import ray
    except ImportError:
        return False
    return bool(ray.is_initialized())


MAX_POLICY_AGE_FILE = "max-policy-age"


def put_max_policy_age(limit: int, sink: str | None = None) -> None:
    """Driver side (3.5 governor fallback): lower the rollout's policy-age limit."""
    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    if kind == "dir":
        directory = Path(target)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / MAX_POLICY_AGE_FILE).write_text(str(int(limit)), encoding="utf-8")
        return
    if kind == "ray":
        import ray

        ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).set_limit.remote(int(limit)))
        return
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def max_policy_age_override(sink: str | None = None) -> int | None:
    """Rollout side: the limit set by :func:`put_max_policy_age` (None = not lowered)."""
    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    if kind == "dir":
        path = Path(target) / MAX_POLICY_AGE_FILE
        return int(path.read_text(encoding="utf-8")) if path.exists() else None
    if kind == "ray":
        import ray

        return ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).get_limit.remote())
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def current_policy_token(sink: str | None = None) -> str | None:
    """Rollout side: the token stored by :func:`put_policy_token`, if any."""

    sink = sink or os.environ.get(META_SINK_ENV) or DEFAULT_SINK
    kind, _, target = sink.partition(":")
    if kind == "dir":
        path = Path(target) / POLICY_TOKEN_FILE
        return path.read_text(encoding="utf-8") if path.exists() else None
    if kind == "ray":
        import ray

        return ray.get(ray.get_actor(target or DEFAULT_SINK_ACTOR).get_token.remote())
    raise ValueError(f"unknown rollout metadata sink {sink!r}")


def group_matches_policy(group: Sequence[Any], token: str | None, size: int | None) -> bool:
    """Legacy ``_complete_group_for_policy`` over upstream sample versions."""

    samples = _flat(group)
    if not token or not samples or (size is not None and len(group) != size):
        return False
    for sample in samples:
        if _status(sample) not in _REUSABLE_STATUSES or _versions(sample) != {token}:
            return False
    return True


def policy_buffer_filter(args: Any, _rollout_id: Any, buffer: list, num_samples: int) -> list:
    """``--buffer-filter-path``: reuse only complete groups of the published policy.

    Groups from any other policy (or incomplete ones) are dropped from the
    buffer, exactly as legacy filtered its completed-group queue.
    """

    token = current_policy_token()
    limit = _carry.max_policy_age(args)
    if limit > 0:
        # agentic-rollout-utilization 4.1: unfinished groups put back by Miles'
        # partial-rollout abort continue while within the policy-age limit.
        version = token_version(token) if token else None
        if version is None:
            raise RuntimeError(f"carry-over buffer filter without a published policy token ({token!r})")
        return _carry.carry_over_buffer_filter(args, buffer, num_samples,
                                               current_version=version, limit=limit)
    size = getattr(args, "n_samples_per_prompt", None)
    kept = [g for g in buffer if group_matches_policy(g, token, size)]
    selected, buffer[:] = kept[:num_samples], kept[num_samples:]
    return selected


_OFFSET_ATTR = "_yeto_data_source_offset"
_GROUP_INDEX_ATTR = "_yeto_data_source_group_index"


def submitted_groups(args: Any, data_source: Any) -> int | None:
    """Prompt groups drawn from the data source this rollout (over-sampling included).

    Miles ``generate_rollout`` submits ``over_sampling_batch_size`` groups at a
    time and aborts the ones still in flight once enough are accepted; those
    never reach ``all_samples``. The drawn count is the advance of the data
    source's monotonic ``sample_group_index`` since the previous rollout
    (agentic-rollout-utilization 1.1: survives an epoch wrap-around, which a
    small task set hits every round under over-sampling). Sources without it
    fall back to the ``sample_offset`` advance (a wrap-around: unknown). First
    rollout or a buffer source: unknown -> None, never guessed.
    """
    source = getattr(data_source, "__self__", data_source)
    offset = getattr(source, "sample_offset", None)
    index = getattr(source, "sample_group_index", None)
    index = index if isinstance(index, int) and not isinstance(index, bool) else None
    previous_index = getattr(args, _GROUP_INDEX_ATTR, None)
    previous = getattr(args, _OFFSET_ATTR, None)
    setattr(args, _GROUP_INDEX_ATTR, index)
    setattr(args, _OFFSET_ATTR, offset if isinstance(offset, int) else None)
    if getattr(source, "buffer", None):
        return None
    if index is not None:
        if not isinstance(previous_index, int) or index < previous_index:
            return None
        return index - previous_index
    if not isinstance(offset, int) or not isinstance(previous, int) or offset < previous:
        return None
    return offset - previous


_CURSOR_FIELDS = ("sample_offset", "epoch_id", "sample_group_index", "sample_index")


def data_cursor(data_source: Any) -> tuple[dict[str, int] | None, int | None]:
    """rl-infra-spec 4.2: data source position and reuse-buffer length (cut-audit §3).

    Read in the rollout process after the rollout drew its prompts (Miles
    ``RolloutDataSource`` attributes; ``get_buffer_length`` on the buffered
    source). Missing or non-integer fields: unknown (None), never guessed.
    """
    source = getattr(data_source, "__self__", data_source)
    if source is None:
        return None, None
    cursor = {f: getattr(source, f, None) for f in _CURSOR_FIELDS}
    known = {k: int(v) for k, v in cursor.items() if isinstance(v, int) and not isinstance(v, bool)}
    length = None
    getter = getattr(source, "get_buffer_length", None)
    try:
        if callable(getter):
            length = int(getter())
        elif isinstance(getattr(source, "buffer", None), list):
            length = len(source.buffer)
    except Exception:  # noqa: BLE001 - unknown, reported as None
        length = None
    return (known if len(known) == len(_CURSOR_FIELDS) else None), length


ELASTIC_METADATA_ENV = "YETO_RL_ELASTIC_METADATA"


def elastic_metadata_enabled(args: Any) -> bool:
    """Report data cursor/buffer length only when E1/E2 (ledger, cut) is on.

    Off by default so the default metadata (and ``carried_over=None``) is
    unchanged. Set ``args.yeto_rl_elastic_metadata`` or the environment
    variable in the rollout process before it starts.
    """
    import os

    return bool(getattr(args, "yeto_rl_elastic_metadata", False)) or os.environ.get(
        ELASTIC_METADATA_ENV) == "1"


def extract_rollout_metadata(args: Any, all_samples: Any, data_source: Any = None) -> None:
    """``--rollout-all-samples-process-path`` hook."""

    try:
        payload = build_metadata(args, all_samples)
        if elastic_metadata_enabled(args):
            cursor, buffer_length = data_cursor(data_source)
            if cursor is not None:
                payload["data_cursor"] = cursor
            if buffer_length is not None:
                payload["buffer_length"] = buffer_length
        submitted = submitted_groups(args, data_source)
        if submitted is not None and payload.get("max_policy_age"):
            # 4.1: under partial rollout the abort does not discard; the groups
            # submitted (new + resubmitted from the buffer) and not completed
            # went back to the buffer to continue next round.
            generated = payload["completed"] + payload["filtered"]
            payload["submitted_groups"] = submitted
            payload["carried_out_groups"] = max(
                0, submitted + int(payload.get("resubmitted_groups", 0)) - generated)
            if "suspended_groups" in payload:
                # 5.1 agentic: the fork counted the groups it kept at this cut-off
                payload["carried_out_groups"] = int(payload["suspended_groups"])
        elif submitted is not None:
            generated = payload["completed"] + payload["filtered"]
            payload["submitted_groups"] = submitted
            # submitted but not completed when the batch filled: aborted in
            # flight (partial_rollout off -> their prompts are consumed, never
            # trained: terminal, A2/F5 'filtered' with reason aborted_in_flight)
            payload["aborted_in_flight_groups"] = max(0, submitted - generated)
        # agentic-rollout-utilization 1.2: the fork's abort() tally (agentenv/miles
        # s18-abort-discard-stats) is exact, first rollout included; absent on
        # older images -> the offset-derived count above (tokens unknown).
        from yeto.rl.engine.rollout_cutoff import discard_stats_fields

        payload.update(discard_stats_fields(getattr(args, "rollout_abort_discard_stats", None)))
        if hasattr(args, "rollout_abort_discard_stats"):
            args.rollout_abort_discard_stats = None
        # rl-algo-supplement 2.6: the fork's over-sampling tally (absent on
        # older images -> no field, unknown)
        from yeto.rl.engine.ports import over_sampling_fields

        over_sampling = over_sampling_fields(getattr(args, "rollout_over_sampling_stats", None))
        if over_sampling is not None:
            payload["over_sampling"] = over_sampling
        if hasattr(args, "rollout_over_sampling_stats"):
            args.rollout_over_sampling_stats = None
        put_to_sink(payload)
    finally:
        # Reset per-rollout state: the bounded filter keys its memo on
        # ``yeto_rl_policy_version`` which legacy advanced per round; here the
        # rollout boundary is the reset point.
        from yeto.rl.algos.sample_filters import reset as _reset_sample_filters

        _reset_sample_filters(args)
        for attr in (_TRAINED_ATTR, _BOUNDED_FILTER_STATE_ATTR, _PLACEHOLDER_ATTR):
            if hasattr(args, attr):
                setattr(args, attr, None)
