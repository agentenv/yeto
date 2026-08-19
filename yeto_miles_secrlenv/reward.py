"""Authenticated reward extraction for secrlenv Miles samples."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import statistics
from pathlib import Path
from typing import Any

OUTCOME_KEY = "secrlenv_trusted_outcome"
MAC_KEY = "secrlenv_trusted_outcome_hmac"
INFRASTRUCTURE_STATUS = "infrastructure_error"
CLEANUP_ERROR_STATUS = "cleanup_error"
INFRASTRUCTURE_503_RETRIES_KEY = "secrlenv_infrastructure_503_retries"
ADMISSION_SCHEMA = 2
ADMISSION_PHASE_KEY = "admission_phase"
ADMISSION_ERROR_CODE_KEY = "admission_error_code"
ADMISSION_NONCE_KEY = "admission_nonce"
ADMISSION_PHASE = "pre_create"
ADMISSION_ERROR_CODE = "infrastructure_error"
_ADMISSION_NONCE = re.compile(r"[0-9a-f]{32}")
_SIGNED_STATUSES = {
    "completed",
    "timeout",
    "max_turns",
    "max_seq_len",
    INFRASTRUCTURE_STATUS,
    CLEANUP_ERROR_STATUS,
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
    schema = outcome.get("schema")
    status = outcome.get("status")
    if schema not in {1, ADMISSION_SCHEMA} or status not in _SIGNED_STATUSES:
        raise UntrustedOutcome("secrlenv outcome has an invalid schema or status")
    infrastructure_503_retries = outcome.get(INFRASTRUCTURE_503_RETRIES_KEY)
    if (
        type(infrastructure_503_retries) is not int
        or infrastructure_503_retries not in {0, 1}
    ):
        raise UntrustedOutcome("secrlenv outcome has an invalid retry ledger")
    reward = outcome.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise UntrustedOutcome("secrlenv outcome reward is not numeric")
    value = float(reward)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise UntrustedOutcome("secrlenv outcome reward is outside [0, 1]")
    if not isinstance(outcome.get("task_id"), str):
        raise UntrustedOutcome("secrlenv outcome has no episode/task identity")
    if schema == ADMISSION_SCHEMA:
        expected_keys = {
            "schema",
            "status",
            "episode_id",
            "task_id",
            "reward",
            "passed",
            "class",
            INFRASTRUCTURE_503_RETRIES_KEY,
            ADMISSION_PHASE_KEY,
            ADMISSION_ERROR_CODE_KEY,
            ADMISSION_NONCE_KEY,
        }
        if (
            set(outcome) != expected_keys
            or status != INFRASTRUCTURE_STATUS
            or outcome.get("episode_id") is not None
            or not outcome["task_id"]
            or outcome.get(ADMISSION_PHASE_KEY) != ADMISSION_PHASE
            or outcome.get(ADMISSION_ERROR_CODE_KEY) != ADMISSION_ERROR_CODE
            or not isinstance(outcome.get(ADMISSION_NONCE_KEY), str)
            or _ADMISSION_NONCE.fullmatch(outcome[ADMISSION_NONCE_KEY]) is None
            or infrastructure_503_retries != 1
            or value != 0.0
            or outcome.get("passed") is not False
            or outcome.get("class") is not None
        ):
            raise UntrustedOutcome("secrlenv admission outcome is invalid")
    elif (
        not isinstance(outcome.get("episode_id"), str)
        or any(
            key in outcome
            for key in (
                ADMISSION_PHASE_KEY,
                ADMISSION_ERROR_CODE_KEY,
                ADMISSION_NONCE_KEY,
            )
        )
    ):
        raise UntrustedOutcome("secrlenv outcome has no episode/task identity")
    return outcome, value


def verify_outcome(metadata: Any) -> float:
    outcome, value = _verified_outcome(metadata)
    if outcome.get("status") in {INFRASTRUCTURE_STATUS, CLEANUP_ERROR_STATUS}:
        raise UntrustedOutcome(
            "secrlenv non-verdict outcome is not a grader verdict"
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
    """Authenticate groups and coordinate bounded same-task replacements."""

    from miles.rollout.filter_hub.base_types import DynamicFilterOutput

    from yeto.rl.miles import (
        SecRLEnvReplacementExhausted,
        SecRLEnvUntrustedEvidence,
    )

    flattened = _flatten_samples(samples)
    rollout_id = getattr(args, "yeto_rl_policy_version", None)
    state = getattr(args, "_yeto_secrlenv_filter_state", None)
    if state is None or state.get("rollout_id") != rollout_id:
        state = {
            "rollout_id": rollout_id,
            "rejections": 0,
            "nontrainable_rejections": 0,
            "forced": 0,
            "decisions": {},
        }
        args._yeto_secrlenv_filter_state = state

    key = _group_key(flattened)
    previous = state["decisions"].get(key)
    if previous is not None:
        return DynamicFilterOutput(keep=previous[0], reason=previous[1])

    try:
        verified = [_verified_outcome(sample.metadata) for sample in flattened]
    except UntrustedOutcome:
        raise SecRLEnvUntrustedEvidence(
            "SecRLEnv rollout evidence authentication failed"
        ) from None

    statuses = []
    for sample in flattened:
        status = getattr(sample, "status", None)
        value = getattr(status, "value", status)
        statuses.append(value if isinstance(value, str) else None)
    infrastructure = any(
        outcome.get("status") == INFRASTRUCTURE_STATUS
        for outcome, _ in verified
    )
    cleanup_failure = any(
        outcome.get("status") == CLEANUP_ERROR_STATUS
        for outcome, _ in verified
    )
    if cleanup_failure:
        from yeto.rl.miles import SecRLEnvRolloutCleanupError

        raise SecRLEnvRolloutCleanupError(
            "SecRLEnv episode cleanup could not be verified"
        )
    incomplete = any(
        status not in {"completed", "truncated"} for status in statuses
    )
    if infrastructure or incomplete:
        if infrastructure:
            reason = "secrlenv_infrastructure_failure"
        elif "aborted" in statuses:
            reason = "secrlenv_aborted"
        elif "pending" in statuses:
            reason = "secrlenv_pending"
        elif "failed" in statuses:
            reason = "secrlenv_failed"
        else:
            reason = "secrlenv_incomplete_status"
        decision = (False, reason)
        state["decisions"][key] = decision
        state["nontrainable_rejections"] += 1
        retry = getattr(args, "_yeto_secrlenv_retry_callback", None)
        if not callable(retry):
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout replacement scheduler is unavailable"
            )
        retry_ledger: dict[int, int] = {}
        for sample, (outcome, _) in zip(flattened, verified, strict=True):
            try:
                index = int(sample.index)
            except (AttributeError, TypeError, ValueError):
                raise SecRLEnvReplacementExhausted(
                    "SecRLEnv rollout replacement task identity is unavailable"
                ) from None
            value = outcome[INFRASTRUCTURE_503_RETRIES_KEY]
            previous_retry_count = retry_ledger.setdefault(index, value)
            if previous_retry_count != value:
                raise SecRLEnvUntrustedEvidence(
                    "SecRLEnv rollout retry ledger is inconsistent"
                )
        retry(
            flattened,
            evidence_key=key,
            reason=reason,
            infrastructure_503_retries=retry_ledger,
        )
        return DynamicFilterOutput(keep=False, reason=reason)

    rewards = [value for _, value in verified]

    std = statistics.pstdev(rewards) if rewards else 0.0
    if math.isfinite(std) and std > 1e-8:
        decision = (True, None)
    else:
        limit = _replacement_limit(args)
        if limit is None:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv zero-variance replacement contract is missing"
            )
        if state["rejections"] < limit:
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
