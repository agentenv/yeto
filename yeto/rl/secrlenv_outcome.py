"""Structural check of a SecRLEnv grader verdict in trajectory metadata.

Kept from the SecRLEnv reward of the removed self-written Codex harness (S26
E0.5, tag s26-pre-harness-removal) so trajectory evidence of the ``secrlenv``
kind still verifies; the same role ``tbench_outcome`` has for Terminal-Bench.
Rewards are not signed (openspec ``rl-drop-reward-tamper-hash``): this checks
shape, status and range only.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

OUTCOME_KEY = "secrlenv_trusted_outcome"
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
_OUTCOME_STATUSES = {
    "completed",
    "timeout",
    "max_turns",
    "max_seq_len",
    INFRASTRUCTURE_STATUS,
    CLEANUP_ERROR_STATUS,
}


class UntrustedOutcome(RuntimeError):
    pass


def canonical_outcome(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verified_outcome(metadata: Any) -> tuple[dict[str, Any], float]:
    if not isinstance(metadata, dict):
        raise UntrustedOutcome("sample metadata is not an object")
    outcome = metadata.get(OUTCOME_KEY)
    if not isinstance(outcome, dict):
        raise UntrustedOutcome("sample has no secrlenv outcome")
    schema = outcome.get("schema")
    status = outcome.get("status")
    if schema not in {1, ADMISSION_SCHEMA} or status not in _OUTCOME_STATUSES:
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
