"""Authenticated reward extraction for secrlenv Miles samples."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
import statistics
from pathlib import Path
from typing import Any


OUTCOME_KEY = "secrlenv_trusted_outcome"
MAC_KEY = "secrlenv_trusted_outcome_hmac"
INFRASTRUCTURE_STATUS = "infrastructure_error"
_SIGNED_STATUSES = {
    "completed",
    "timeout",
    "max_turns",
    "max_seq_len",
    INFRASTRUCTURE_STATUS,
}


class UntrustedOutcome(RuntimeError):
    pass


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hmac_key() -> bytes:
    value = os.getenv("SECRLENV_REWARD_HMAC_KEY") or os.getenv("SECRLENV_BEARER_TOKEN", "")
    if not value and os.getenv("SECRLENV_BEARER_TOKEN_FILE"):
        path = Path(os.environ["SECRLENV_BEARER_TOKEN_FILE"])
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise UntrustedOutcome("secrlenv token file is not private")
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise UntrustedOutcome(f"cannot read secrlenv token file: {exc}") from exc
    if len(value) < 32:
        raise UntrustedOutcome("secrlenv reward HMAC key is missing or too short")
    return value.encode("utf-8")


def sign_outcome(outcome: dict[str, Any], *, key: str | bytes | None = None) -> str:
    secret = key if key is not None else _hmac_key()
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    return hmac.new(secret, _canonical(outcome), hashlib.sha256).hexdigest()


def _verified_outcome(metadata: Any) -> tuple[dict[str, Any], float]:
    if not isinstance(metadata, dict):
        raise UntrustedOutcome("sample metadata is not an object")
    outcome = metadata.get(OUTCOME_KEY)
    supplied_mac = metadata.get(MAC_KEY)
    if not isinstance(outcome, dict) or not isinstance(supplied_mac, str):
        raise UntrustedOutcome("sample has no signed secrlenv outcome")
    expected_mac = sign_outcome(outcome)
    if not hmac.compare_digest(expected_mac, supplied_mac):
        raise UntrustedOutcome("secrlenv outcome signature mismatch")
    if outcome.get("schema") != 1 or outcome.get("status") not in _SIGNED_STATUSES:
        raise UntrustedOutcome("secrlenv outcome has an invalid schema or status")
    reward = outcome.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise UntrustedOutcome("secrlenv outcome reward is not numeric")
    value = float(reward)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise UntrustedOutcome("secrlenv outcome reward is outside [0, 1]")
    if not isinstance(outcome.get("episode_id"), str) or not isinstance(
        outcome.get("task_id"), str
    ):
        raise UntrustedOutcome("secrlenv outcome has no episode/task identity")
    return outcome, value


def verify_outcome(metadata: Any) -> float:
    outcome, value = _verified_outcome(metadata)
    if outcome.get("status") == INFRASTRUCTURE_STATUS:
        raise UntrustedOutcome(
            "secrlenv infrastructure outcome is not a grader verdict"
        )
    return value


def _sample_reward(sample: Any) -> float:
    outcome, value = _verified_outcome(getattr(sample, "metadata", None))
    if outcome.get("status") != INFRASTRUCTURE_STATUS:
        return value

    # Miles calls the reward function before the dynamic group filter. Mark a
    # signed infrastructure failure ABORTED here so it is replaced, while still
    # returning the numeric value required by the reward-function API. The
    # group filter never permits this transient zero into training data.
    from miles.utils.types import Sample

    sample.status = Sample.Status.ABORTED
    return 0.0


async def reward_func(args: Any, samples: Any, **_kwargs: Any) -> float | list[float]:
    """Return only HMAC-authenticated host-side rewards.

    Missing or invalid metadata remains fatal. A signed infrastructure marker
    is returned as the numeric zero required by Miles only after marking its
    sample ABORTED, so the group filter replaces it and it never enters training.
    """

    if isinstance(samples, list):
        return [_sample_reward(sample) for sample in samples]
    return _sample_reward(samples)


def _flatten_samples(samples: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for item in samples:
        flattened.extend(item if isinstance(item, list) else [item])
    return flattened


def _group_key(samples: list[Any]) -> tuple[Any, ...]:
    """Identify a generated group across Miles' repeated callbacks.

    Sample indexes are stable across the generation-time and all-samples
    callbacks.  The signed episode identity and MAC prevent an index collision
    from reusing a decision for different external-environment evidence.
    """

    values = []
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        outcome = metadata.get(OUTCOME_KEY) if isinstance(metadata, dict) else None
        supplied_mac = metadata.get(MAC_KEY) if isinstance(metadata, dict) else None
        index = getattr(sample, "index", None)
        values.append(
            (
                index if index is not None else id(sample),
                outcome.get("episode_id") if isinstance(outcome, dict) else None,
                outcome.get("task_id") if isinstance(outcome, dict) else None,
                supplied_mac,
            )
        )
    return tuple(values)


def _replacement_limit(args: Any) -> int | None:
    value = getattr(args, "yeto_rl_dynamic_sampling_max_replacements", None)
    try:
        limit = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return limit if limit is not None and limit >= 0 else None


def check_group(args: Any, samples: list[Any], **_kwargs: Any):
    """Authenticate every group and bound valid zero-variance replacements.

    Aborted samples and unsigned/invalid outcomes are always rejected.  Only a
    fully authenticated zero-variance group consumes the replacement budget;
    after the configured bound, one such group is accepted so a difficult
    flaky task cannot deadlock rollout generation.  Decisions are memoized
    because Miles calls the filter again from Yeto's all-samples hook.
    """

    from miles.rollout.filter_hub.base_types import DynamicFilterOutput
    from miles.utils.types import Sample

    flattened = _flatten_samples(samples)
    if any(sample.status == Sample.Status.ABORTED for sample in flattened):
        return DynamicFilterOutput(keep=False, reason="secrlenv_aborted")
    try:
        verified = [_verified_outcome(sample.metadata) for sample in flattened]
    except UntrustedOutcome:
        return DynamicFilterOutput(keep=False, reason="secrlenv_untrusted_outcome")
    if any(
        outcome.get("status") == INFRASTRUCTURE_STATUS
        for outcome, _ in verified
    ):
        return DynamicFilterOutput(
            keep=False, reason="secrlenv_infrastructure_failure"
        )
    rewards = [value for _, value in verified]

    rollout_id = getattr(args, "yeto_rl_policy_version", None)
    state = getattr(args, "_yeto_secrlenv_filter_state", None)
    if state is None or state.get("rollout_id") != rollout_id:
        state = {
            "rollout_id": rollout_id,
            "rejections": 0,
            "forced": 0,
            "decisions": {},
        }
        args._yeto_secrlenv_filter_state = state

    key = _group_key(flattened)
    previous = state["decisions"].get(key)
    if previous is not None:
        return DynamicFilterOutput(keep=previous[0], reason=previous[1])

    std = statistics.pstdev(rewards) if rewards else 0.0
    if math.isfinite(std) and std > 1e-8:
        decision = (True, None)
    else:
        limit = _replacement_limit(args)
        if limit is None or state["rejections"] < limit:
            state["rejections"] += 1
            value = round(rewards[0], 3) if rewards else 0.0
            decision = (False, f"secrlenv_zero_std_{value}")
        else:
            state["forced"] += 1
            decision = (
                True,
                f"secrlenv_bounded_fallback_after_{limit}_replacements",
            )
    state["decisions"][key] = decision
    return DynamicFilterOutput(keep=decision[0], reason=decision[1])
