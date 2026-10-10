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


# Router policies that route by the X-SMG-Routing-Key header (Miles
# ``policy_uses_routing_key``). Under ``manual`` -- the default once Miles' session
# server is on (agentic runs) -- the router rejects a request without the key
# with a 4xx before any engine sees it (S18 ARU-3 M1: one router "client error"
# per unscored sample, rounds 1-5: 16/8/8/8/8).
ROUTING_KEY_POLICIES = ("consistent_hashing", "manual")
ROUTING_KEY_HEADER = "X-SMG-Routing-Key"
SCORING_ROUTING_KEY = "yeto-cross-version-score"


def scoring_routing_headers(args: Any, sample: Any) -> dict[str, str]:
    """Routing header for a scoring request: the sample's own key (the engine
    that holds its prefix), else a fixed key when the router policy needs one;
    {} when the policy routes without a key."""
    key = getattr(sample, "routing_key", None)
    if key:
        return {ROUTING_KEY_HEADER: str(key)}
    if getattr(args, "sglang_router_policy", None) in ROUTING_KEY_POLICIES:
        return {ROUTING_KEY_HEADER: SCORING_ROUTING_KEY}
    return {}


def router_prefill_scorer(args: Any, timeout_s: float = 120.0) -> Scorer:
    """Score a sample's response tokens under the engines' current weights
    (SGLang prefill through the Miles router); None when scoring fails."""
    import urllib.request

    def score(sample: Any) -> list[float] | None:
        score.last_error = None
        try:
            url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
            from miles.rollout.generate_utils.prefill_logprobs import (
                _build_prefill_scoring_payload,
                _extract_response_logprobs,
            )

            payload = _build_prefill_scoring_payload(args, sample, {})
            request = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **scoring_routing_headers(args, sample)})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                output = json.loads(response.read().decode("utf-8"))
            return [float(p) for p in _extract_response_logprobs(sample, output["meta_info"])]
        except Exception as exc:  # noqa: BLE001 - an estimate: unknown, never guessed
            score.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            return None

    score.last_error = None
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
    ratios: list[float] = []
    failed = 0
    reasons: dict[str, int] = {}
    unknown_tokens = 0
    near_zero = 0
    near_zero_examples: list[dict[str, Any]] = []

    def fail(why: str) -> None:
        nonlocal failed
        failed += 1
        reasons[why] = reasons.get(why, 0) + 1

    for sample in samples:
        versions = trained_token_versions(sample, current_version)
        if not any(v is not None and v != current_version for v in versions):
            continue
        # a trained token whose version no span covers is left out of the
        # estimate (weight 1, counted), instead of dropping the whole sample
        unknown = sum(1 for v in versions if v is None)
        if unknown:
            unknown_tokens += unknown
            versions = [current_version if v is None else v for v in versions]
        generated = list(getattr(sample, "rollout_log_probs", None) or ())
        if len(generated) != len(versions):
            fail(f"rollout_log_probs {len(generated)} != response {len(versions)}")
            continue
        current = scorer(sample)
        if current is None:
            fail(str(getattr(scorer, "last_error", None) or "scorer returned nothing"))
            continue
        if len(current) != len(versions):
            fail(f"scored {len(current)} != response {len(versions)}")
            continue
        try:
            provenance = TokenProvenance(tuple(int(v) for v in versions),
                                         tuple(min(float(p), 0.0) for p in generated))
            corrections.append(cross_version_is(provenance, current, current_version,
                                                clip_low=clip_low, clip_high=clip_high))
            for i, (v, g, c) in enumerate(zip(provenance.versions, provenance.logprobs, current, strict=True)):
                if v == current_version:
                    continue
                r = math.exp(float(c) - g)
                ratios.append(r)
                if r < NEAR_ZERO_RATIO:  # 5.5 follow-up: where do zero ratios come from
                    near_zero += 1
                    if len(near_zero_examples) < NEAR_ZERO_EXAMPLES:
                        near_zero_examples.append(_ratio_example(sample, provenance.versions, i, g, c))
        except (ProvenanceError, ValueError, OverflowError) as exc:
            fail(f"{type(exc).__name__}: {str(exc)[:120]}")
    fraction = batch_truncated_fraction(corrections)
    extra: dict[str, Any] = {}
    if reasons:  # 5.5: why samples could not be scored (absent when all were)
        extra["cross_version_unscored_reasons"] = dict(sorted(reasons.items())[:8])
    if unknown_tokens:
        extra["cross_version_unknown_version_tokens"] = unknown_tokens
    if near_zero:  # absent when none (old key set kept)
        extra["cross_version_ratio_near_zero"] = near_zero
        extra["cross_version_ratio_near_zero_examples"] = near_zero_examples
    return {
        **extra,
        "cross_version_truncated_fraction": None if fraction is None or not math.isfinite(fraction)
        else float(fraction),
        "cross_version_scored_tokens": sum(c.cross_version_tokens for c in corrections),
        "cross_version_unscored_samples": failed,
        **ratio_quantiles(ratios),
    }


NEAR_ZERO_RATIO = 1e-6
NEAR_ZERO_EXAMPLES = 8


def _ratio_example(sample: Any, versions: Sequence[int], i: int, generated: float, current: Any) -> dict[str, Any]:
    """One near-zero cross-version ratio: where in the response it sits and the two
    log-probabilities (non-finite values as strings, JSON-safe)."""
    def num(x: Any) -> Any:
        x = float(x)
        return round(x, 4) if math.isfinite(x) else str(x)

    tokens = list(getattr(sample, "tokens", None) or ())
    n = len(versions)
    token = tokens[len(tokens) - n + i] if len(tokens) >= n else None
    mask = list(getattr(sample, "loss_mask", None) or ())
    return {"index": i, "response_len": n, "version": int(versions[i]),
            "segment_start": i == 0 or versions[i - 1] != versions[i],
            "token": token, "loss_mask": mask[i] if len(mask) == n else None,
            "generation_logprob": num(generated), "current_logprob": num(current)}


def ratio_quantiles(ratios: Sequence[float]) -> dict[str, float]:
    """p50/p90/p99/min/max of the cross-version ratios exp(current - generation)
    (5.5: data to calibrate the warn/fallback thresholds); {} when none."""
    values = sorted(r for r in ratios if math.isfinite(r))
    if not values:
        return {}

    def q(p: float) -> float:
        return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]

    return {"cross_version_ratio_p50": q(0.50), "cross_version_ratio_p90": q(0.90),
            "cross_version_ratio_p99": q(0.99), "cross_version_ratio_min": values[0],
            "cross_version_ratio_max": values[-1]}


def trained_token_versions(sample: Any, current_version: int) -> list[int | None]:
    """Per response token, the version to correct for: tokens outside the loss
    (``loss_mask`` 0: tool/environment output of an agentic sample, never
    generated by the policy) count as current (weight 1, not crossed)."""
    versions = response_token_versions(sample)
    mask = getattr(sample, "loss_mask", None)
    if mask is None or len(mask) != len(versions):
        return versions
    return [v if m else current_version for v, m in zip(versions, mask, strict=True)]


# --------------------------------------------- agentic suspension (5.1/5.3)

SUSPEND_STATS_ATTR = "rollout_suspend_stats"  # Miles fork, set at the cut-off
RESUME_STATS_ATTR = "rollout_resume_stats"  # Miles fork, set at rollout start
SUSPEND_EXPIRED_PREFIX = "CodexSuspendExpired"
POLICY_AGE_EXCEEDED_PREFIX = "policy_age_exceeded"


def _infrastructure_reason(sample: Any) -> str:
    from yeto.rl.harness.codex.tbench_reward import INFRASTRUCTURE_KEY

    meta = getattr(sample, "metadata", None)
    reason = meta.get(INFRASTRUCTURE_KEY) if isinstance(meta, dict) else None
    return str(reason) if reason else ""


def _agent_metric(sample: Any, name: str) -> float:
    meta = getattr(sample, "metadata", None)
    metrics = meta.get("agent_metrics") if isinstance(meta, dict) else None
    value = metrics.get(name) if isinstance(metrics, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.0
    return float(value)


def suspend_fields(args: Any, all_samples: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Rollout-metadata fields of an agentic run under a limit > 0 (stage 3).

    From the Miles fork (consumed): groups suspended at this rollout's cut-off,
    groups resumed at its start, groups cancelled because they would exceed the
    limit. From the samples of this rollout: trajectories discarded because
    their suspension outlived the survival limit, trajectories outside the
    version window, trajectories that were suspended at least once, and the
    seconds their environments stayed alive while parked (survival cost)."""
    suspend = getattr(args, SUSPEND_STATS_ATTR, None)
    resume = getattr(args, RESUME_STATS_ATTR, None)
    if suspend is None and resume is None:
        return {}
    for attr in (SUSPEND_STATS_ATTR, RESUME_STATS_ATTR):
        if hasattr(args, attr):
            setattr(args, attr, None)
    out: dict[str, Any] = {}
    for prefix, stats in (("", suspend), ("", resume)):
        if isinstance(stats, dict):
            out.update({f"{prefix}{k}": int(v) for k, v in stats.items()
                        if isinstance(v, int) and not isinstance(v, bool)})
    expired = over_age = parked = 0
    parked_seconds = 0.0
    retried = 0
    for sample in _flat(list(all_samples)):
        reason = _infrastructure_reason(sample)
        expired += reason.startswith(SUSPEND_EXPIRED_PREFIX)
        over_age += reason.startswith(POLICY_AGE_EXCEEDED_PREFIX)
        if _agent_metric(sample, "suspensions") > 0:
            parked += 1
            parked_seconds += _agent_metric(sample, "suspended_seconds")
            retried += int(_agent_metric(sample, "suspend_retried_turns"))
    out.update({
        "suspend_expired_trajectories": expired,
        "policy_age_exceeded_trajectories": over_age,
        "suspended_trajectories": parked,
        "suspended_env_seconds": round(parked_seconds, 3),
        "suspend_retried_turns": retried,
    })
    return out
