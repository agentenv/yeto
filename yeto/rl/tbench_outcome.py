"""Validated Terminal-Bench 2.1 verifier outcomes.

This module intentionally uses only the Python standard library so the same
closed contract can run in the isolated OpenEnv agent process, Miles' reward
worker, and Yeto's trajectory-evidence writer.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

OUTCOME_KEY = "tbench_trusted_outcome"
SCHEMA = 1
BENCHMARK = "terminal-bench-2.1"
TIMEOUT_VERIFIER = "not_run_timeout"
NATIVE_VERIFIER = "openenv_native_evaluate"
TEST_SH_VERIFIER = "terminal_bench_test_sh"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+@-]{0,511}\Z")
_STATUSES = frozenset({"completed", "timeout", "max_turns", "max_seq_len"})
_VERIFIERS = frozenset({TIMEOUT_VERIFIER, NATIVE_VERIFIER, TEST_SH_VERIFIER})
_FIELDS = frozenset(
    {
        "schema",
        "benchmark",
        "task_id",
        "sample_id",
        "episode_id",
        "status",
        "reward",
        "passed",
        "verifier",
        "testsh_rc",
    }
)


class UntrustedTBenchOutcome(RuntimeError):
    """Raised when Terminal-Bench metadata is absent or malformed."""


def canonical_outcome(outcome: dict[str, Any]) -> bytes:
    """Return the canonical byte representation used by trajectory evidence."""

    return json.dumps(
        outcome,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise UntrustedTBenchOutcome(
            f"Terminal-Bench outcome has an invalid {name}"
        )
    return value


def _validated(outcome: Any) -> tuple[dict[str, Any], float]:
    if not isinstance(outcome, dict) or set(outcome) != _FIELDS:
        raise UntrustedTBenchOutcome("Terminal-Bench outcome fields are not closed")
    if outcome.get("schema") != SCHEMA or outcome.get("benchmark") != BENCHMARK:
        raise UntrustedTBenchOutcome(
            "Terminal-Bench outcome has an invalid schema or benchmark"
        )
    for name in ("task_id", "sample_id", "episode_id"):
        _identifier(name, outcome.get(name))

    status = outcome.get("status")
    verifier = outcome.get("verifier")
    if status not in _STATUSES or verifier not in _VERIFIERS:
        raise UntrustedTBenchOutcome(
            "Terminal-Bench outcome has an invalid status or verifier"
        )
    reward = outcome.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise UntrustedTBenchOutcome("Terminal-Bench outcome reward is not numeric")
    value = float(reward)
    if not math.isfinite(value) or value not in {0.0, 1.0}:
        raise UntrustedTBenchOutcome(
            "Terminal-Bench outcome reward is not finite and binary"
        )
    if type(outcome.get("passed")) is not bool or outcome["passed"] != (value == 1.0):
        raise UntrustedTBenchOutcome(
            "Terminal-Bench outcome pass bit differs from its reward"
        )

    testsh_rc = outcome.get("testsh_rc")
    if status == "timeout":
        if (
            value != 0.0
            or verifier != TIMEOUT_VERIFIER
            or testsh_rc is not None
        ):
            raise UntrustedTBenchOutcome(
                "Terminal-Bench timeout outcome claims a verifier verdict"
            )
    elif verifier == TIMEOUT_VERIFIER:
        raise UntrustedTBenchOutcome(
            "Terminal-Bench non-timeout outcome has no verifier verdict"
        )
    elif verifier == NATIVE_VERIFIER:
        if testsh_rc is not None:
            raise UntrustedTBenchOutcome(
                "Terminal-Bench native verifier outcome has a test.sh status"
            )
    elif (
        isinstance(testsh_rc, bool)
        or not isinstance(testsh_rc, int)
        or not 0 <= testsh_rc <= 255
    ):
        raise UntrustedTBenchOutcome(
            "Terminal-Bench test.sh verifier outcome has no valid exit status"
        )
    return outcome, value


def build_outcome(
    *,
    task_id: str,
    sample_id: str,
    episode_id: str,
    status: str,
    reward: float,
    verifier: str,
    testsh_rc: int | None,
) -> dict[str, Any]:
    """Build one canonical, closed Terminal-Bench outcome."""

    outcome = {
        "schema": SCHEMA,
        "benchmark": BENCHMARK,
        "task_id": task_id,
        "sample_id": sample_id,
        "episode_id": episode_id,
        "status": status,
        "reward": float(reward),
        "passed": float(reward) == 1.0,
        "verifier": verifier,
        "testsh_rc": testsh_rc,
    }
    _validated(outcome)
    return outcome


def build_outcome_metadata(
    *,
    task_id: str,
    sample_id: str,
    episode_id: str,
    status: str,
    reward: float,
    verifier: str,
    testsh_rc: int | None,
) -> dict[str, Any]:
    """Return the metadata field consumed by reward and evidence validation."""

    return {
        OUTCOME_KEY: build_outcome(
            task_id=task_id,
            sample_id=sample_id,
            episode_id=episode_id,
            status=status,
            reward=reward,
            verifier=verifier,
            testsh_rc=testsh_rc,
        )
    }


def verified_outcome(metadata: Any) -> tuple[dict[str, Any], float]:
    """Validate the outcome in arbitrary sample metadata.

    Any legacy HMAC field written by older producers is ignored.
    """

    if not isinstance(metadata, dict):
        raise UntrustedTBenchOutcome("sample metadata is not an object")
    outcome = metadata.get(OUTCOME_KEY)
    if not isinstance(outcome, dict):
        raise UntrustedTBenchOutcome("sample has no Terminal-Bench outcome")
    _validated(outcome)
    return outcome, float(outcome["reward"])


def verify_outcome(metadata: Any) -> float:
    """Return the validated binary benchmark reward."""

    return verified_outcome(metadata)[1]
