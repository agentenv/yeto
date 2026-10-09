"""Miles single-turn carry-over (agentic-rollout-utilization 4.1, stage 2).

With ``--rl-max-policy-age N > 0`` the island runs Miles with
``--partial-rollout``: at the cut-off (over-sampling target reached) Miles'
``abort`` puts the unfinished groups back into its data buffer, and the next
rollout continues them (stock single-turn generate resumes from the tokens
already generated). This module is the yeto side of that, running inside the
Miles rollout process:

* :func:`response_token_versions` -- the version segments of one sample: the
  yeto policy version of every response token, read from the per-call
  ``weight_versions`` spans SGLang reports (the span version is the yeto policy
  token ``yeto:<version>:<hash>`` the publisher stamps on the engines);
* :func:`carry_over_buffer_filter` -- Miles' ``--buffer-filter-path`` under a
  limit > 0: groups whose oldest token is at most N versions behind the round
  being generated continue; older ones (and groups whose token versions are
  unknown) are discarded and counted (spec "超过上限": discarded, reported);
* :func:`take_carry_stats` -- what the buffer filter carried in / discarded
  this round, for the rollout metadata (driver event ``rl_rollout_carry_over``);
* :func:`estimate_cross_version_truncation` -- the truncated fraction of the
  cross-version importance-sampling correction, estimated on the inference
  side: the engines now serve the current version, so a prefill of a carried
  sample gives the current policy's log-probability of each older token; the
  ratio against its generation log-probability is truncated with Miles' TIS
  bounds (:func:`yeto.rl.engine.version_segments.cross_version_is`). The Miles
  trainer applies the same TIS to these tokens but reports only an all-token
  clip fraction, so this is the per-cross-version-token figure the 3.5
  governor reads (an estimate: inference instead of training log-probs).

Limit 0 never reaches this module (``policy_buffer_filter`` and the metadata
hook keep their byte-identical default behaviour).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from typing import Any

from yeto.rl.engine.policy_age import token_version
from yeto.rl.engine.version_segments import (
    ProvenanceError,
    TokenProvenance,
    batch_truncated_fraction,
    cross_version_is,
)

CARRY_STATS_ATTR = "_yeto_carry_over_stats"
CARRY_FIELDS = (
    "carried_in_groups",
    "carried_in_trajectories",
    "carried_in_tokens",
    "over_age_discarded_groups",
    "over_age_discarded_trajectories",
    "over_age_discarded_tokens",
    "unknown_version_discarded_groups",
    # every buffered group handed back to generation this round (with or
    # without tokens): with the new groups drawn from the data source, what
    # was submitted; the groups not completed went back to the buffer.
    "resubmitted_groups",
)


def max_policy_age(args: Any) -> int:
    """The rollout's effective limit: the configured one, lowered to 0 once the
    driver's 3.5 governor fell back (``MilesRolloutPool.set_max_policy_age``)."""
    limit = int(getattr(args, "yeto_rl_max_policy_age", 0) or 0)
    if limit == 0:
        return 0
    from .rollout_meta_hook import max_policy_age_override

    try:
        override = max_policy_age_override()
    except Exception:  # noqa: BLE001 - no sink here: the configured limit
        override = None
    return limit if override is None else min(limit, int(override))


def _flat(group: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for item in group:
        out.extend(_flat(item) if isinstance(item, list) else [item])
    return out


def _spans(sample: Any) -> list[Any]:
    return [span for call in (getattr(sample, "weight_versions", None) or ())
            for span in (getattr(call, "spans", None) or ())]


def response_token_versions(sample: Any) -> list[int | None]:
    """Per response token: the yeto policy version that generated it (None = unknown)."""
    n = int(getattr(sample, "response_length", 0) or 0)
    start = len(getattr(sample, "tokens", None) or ()) - n
    out: list[int | None] = [None] * n
    for span in _spans(sample):
        version = token_version(str(getattr(span, "version", "")))
        lo, hi = max(int(span.abs_start), start), min(int(span.abs_end), start + n)
        for pos in range(lo, hi):
            out[pos - start] = version
    return out


def sample_versions(sample: Any) -> list[int | None]:
    """Distinct versions of a sample's response tokens (None present = some unknown)."""
    return sorted(set(response_token_versions(sample)), key=lambda v: -1 if v is None else v)


def group_versions(group: Sequence[Any]) -> tuple[set[int], bool]:
    """(known versions of the group's response tokens, whether some token is unknown)."""
    known: set[int] = set()
    unknown = False
    for sample in _flat(group):
        for v in response_token_versions(sample):
            if v is None:
                unknown = True
            else:
                known.add(v)
    return known, unknown


def _response_tokens(group: Sequence[Any]) -> int:
    return sum(int(getattr(s, "response_length", 0) or 0) for s in _flat(group))


def _trajectories(group: Sequence[Any]) -> int:
    return sum(1 for s in _flat(group) if int(getattr(s, "response_length", 0) or 0) > 0)


def _stats_for(args: Any, current_version: int) -> dict[str, int]:
    stats = getattr(args, CARRY_STATS_ATTR, None)
    if not isinstance(stats, dict) or stats.get("version") != current_version:
        stats = {"version": current_version, **{k: 0 for k in CARRY_FIELDS}}
        setattr(args, CARRY_STATS_ATTR, stats)
    return stats


def carry_over_buffer_filter(args: Any, buffer: list, num_samples: int, *,
                             current_version: int, limit: int) -> list:
    """Select up to ``num_samples`` buffered groups to continue at ``current_version``.

    Every group older than ``limit`` (oldest token version < current - limit), or
    with response tokens of unknown version, is removed from the buffer and
    counted; the rest keep their order. Counts accumulate per round on ``args``."""
    stats = _stats_for(args, current_version)
    kept: list = []
    for group in buffer:
        known, unknown = group_versions(group)
        if unknown:
            stats["unknown_version_discarded_groups"] += 1
            continue
        if known and current_version - min(known) > limit:
            stats["over_age_discarded_groups"] += 1
            stats["over_age_discarded_trajectories"] += _trajectories(group)
            stats["over_age_discarded_tokens"] += _response_tokens(group)
            continue
        kept.append(group)
    selected, buffer[:] = kept[:num_samples], kept[num_samples:]
    stats["resubmitted_groups"] += len(selected)
    for group in selected:
        if _response_tokens(group):
            stats["carried_in_groups"] += 1
            stats["carried_in_trajectories"] += _trajectories(group)
            stats["carried_in_tokens"] += _response_tokens(group)
    return selected


def take_carry_stats(args: Any, current_version: int | None) -> dict[str, int]:
    """This round's carry-over counts (zeros when nothing was buffered); consumed."""
    stats = getattr(args, CARRY_STATS_ATTR, None)
    if hasattr(args, CARRY_STATS_ATTR):
        setattr(args, CARRY_STATS_ATTR, None)
    if not isinstance(stats, dict) or stats.get("version") != current_version:
        return {k: 0 for k in CARRY_FIELDS}
    return {k: int(stats[k]) for k in CARRY_FIELDS}


def group_versions_record(group: Sequence[Any]) -> dict[str, Any]:
    """Metadata fields of a group under a limit > 0: its version segments."""
    known, _ = group_versions(group)
    return {"policy_versions": sorted(known)} if known else {}


def version_segments(sample: Any) -> list[list[int]]:
    """``[[version, start, end), ...]`` runs over the response (unknown tokens skipped)."""
    out: list[list[int]] = []
    for i, v in enumerate(response_token_versions(sample)):
        if v is None:
            continue
        if out and out[-1][0] == v and out[-1][2] == i:
            out[-1][2] = i + 1
        else:
            out.append([v, i, i + 1])
    return out


def trajectory_fields(sample: Any, current_round: int | None) -> dict[str, Any]:
    """``rl_trajectory_reward`` fields under a limit > 0 (names read by the dashboard):
    ``started_rollout_id`` (Miles' ``start_rollout_id`` of a carried-over sample,
    else this round) and ``policy_versions`` (version segments)."""
    meta = getattr(sample, "metadata", None)
    started = meta.get("start_rollout_id") if isinstance(meta, dict) else None
    if started is None:
        started = current_round
    out: dict[str, Any] = {"policy_versions": version_segments(sample)}
    if isinstance(started, int) and not isinstance(started, bool):
        out["started_rollout_id"] = started
    return out


def cross_version_tokens(samples: Sequence[Any], current_version: int) -> tuple[int, int]:
    """(response tokens generated by an older version, all response tokens)."""
    crossed = total = 0
    for sample in samples:
        versions = response_token_versions(sample)
        total += len(versions)
        crossed += sum(1 for v in versions if v is not None and v != current_version)
    return crossed, total


# ------------------------------------------------------- truncation estimate


Scorer = Callable[[Any], "list[float] | None"]


def router_prefill_scorer(args: Any, timeout_s: float = 120.0) -> Scorer:
    """Score a sample's response tokens under the engines' current weights
    (SGLang prefill through the Miles router); None when scoring fails."""
    import urllib.request

    def score(sample: Any) -> list[float] | None:
        try:
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
            from miles.rollout.generate_utils.prefill_logprobs import (
                _build_prefill_scoring_payload,
                _extract_response_logprobs,
            )

            payload = _build_prefill_scoring_payload(args, sample, {})
            request = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                output = json.loads(response.read().decode("utf-8"))
            return [float(p) for p in _extract_response_logprobs(sample, output["meta_info"])]
        except Exception:  # noqa: BLE001 - an estimate: unknown, never guessed
            return None

    return score


def estimate_cross_version_truncation(args: Any, samples: Sequence[Any], current_version: int,
                                      scorer: Scorer) -> dict[str, Any]:
    """Cross-version IS truncated fraction over the trained samples' older tokens.

    Only samples with an older-version token are scored. Fields: the fraction
    (None when no token crossed a version or no sample could be scored),
    the cross-version tokens it covers, and the samples whose scoring failed."""
    clip_low = float(getattr(args, "tis_clip_low", 0.0) or 0.0)
    clip_high = float(getattr(args, "tis_clip", 2.0) or 2.0)
    corrections = []
    failed = 0
    for sample in samples:
        versions = response_token_versions(sample)
        if not any(v is not None and v != current_version for v in versions):
            continue
        generated = list(getattr(sample, "rollout_log_probs", None) or ())
        current = scorer(sample) if None not in versions and len(generated) == len(versions) else None
        if current is None or len(current) != len(versions):
            failed += 1
            continue
        try:
            provenance = TokenProvenance(tuple(int(v) for v in versions),
                                         tuple(min(float(p), 0.0) for p in generated))
            corrections.append(cross_version_is(provenance, current, current_version,
                                                clip_low=clip_low, clip_high=clip_high))
        except (ProvenanceError, ValueError, OverflowError):
            failed += 1
    fraction = batch_truncated_fraction(corrections)
    return {
        "cross_version_truncated_fraction": None if fraction is None or not math.isfinite(fraction)
        else float(fraction),
        "cross_version_scored_tokens": sum(c.cross_version_tokens for c in corrections),
        "cross_version_unscored_samples": failed,
    }
