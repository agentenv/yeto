"""Authenticated Terminal-Bench 2.1 verifier outcomes.

This module intentionally uses only the Python standard library so the same
closed contract can run in the isolated OpenEnv agent process, Miles' reward
worker, and Yeto's trajectory-evidence writer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any

OUTCOME_KEY = "tbench_trusted_outcome"
MAC_KEY = "tbench_trusted_outcome_hmac"
HMAC_ENV = "TBENCH_REWARD_HMAC_KEY"
HMAC_FILE_ENV = "TBENCH_REWARD_HMAC_KEY_FILE"
SCHEMA = 1
BENCHMARK = "terminal-bench-2.1"
TIMEOUT_VERIFIER = "not_run_timeout"
NATIVE_VERIFIER = "openenv_native_evaluate"
TEST_SH_VERIFIER = "terminal_bench_test_sh"

_DOMAIN = b"yeto-tbench-outcome-v1\0"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+@-]{0,511}\Z")
_MAC = re.compile(r"[0-9a-f]{64}\Z")
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
_MAX_KEY_BYTES = 4096


class UntrustedTBenchOutcome(RuntimeError):
    """Raised when Terminal-Bench metadata is absent, forged, or malformed."""


def canonical_outcome(outcome: dict[str, Any]) -> bytes:
    """Return the exact byte representation covered by the outcome MAC."""

    return json.dumps(
        outcome,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _secret(key: str | bytes | None = None) -> bytes:
    if key is None:
        direct = os.getenv(HMAC_ENV)
        file_name = os.getenv(HMAC_FILE_ENV)
        if direct and file_name:
            raise UntrustedTBenchOutcome(
                "Terminal-Bench direct and file HMAC keys are mutually exclusive"
            )
        if file_name:
            path = Path(file_name)
            if not path.is_absolute() or path.is_symlink():
                raise UntrustedTBenchOutcome(
                    "Terminal-Bench HMAC key file must be absolute and non-symlink"
                )
            try:
                info = path.stat()
                if (
                    not stat.S_ISREG(info.st_mode)
                    or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
                ):
                    raise UntrustedTBenchOutcome(
                        "Terminal-Bench HMAC key file must be private and regular"
                    )
                with path.open("rb") as handle:
                    value = handle.read(_MAX_KEY_BYTES + 1)
            except OSError as error:
                raise UntrustedTBenchOutcome(
                    f"cannot read Terminal-Bench HMAC key file: {error}"
                ) from error
            value = value.rstrip(b"\r\n")
        else:
            value = direct or ""
    else:
        value = key
    if isinstance(value, str):
        value = value.encode("utf-8")
    if (
        not isinstance(value, bytes)
        or not 32 <= len(value) <= _MAX_KEY_BYTES
    ):
        raise UntrustedTBenchOutcome(
            "Terminal-Bench reward HMAC key is missing or outside its size bound"
        )
    return value


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


def sign_outcome(
    outcome: dict[str, Any], *, key: str | bytes | None = None
) -> str:
    """Authenticate one already-valid outcome with the dedicated TB key."""

    _validated(outcome)
    return hmac.new(
        _secret(key), _DOMAIN + canonical_outcome(outcome), hashlib.sha256
    ).hexdigest()


def build_signed_metadata(
    *,
    task_id: str,
    sample_id: str,
    episode_id: str,
    status: str,
    reward: float,
    verifier: str,
    testsh_rc: int | None,
    key: str | bytes | None = None,
) -> dict[str, Any]:
    """Return the two metadata fields consumed by reward/evidence verification."""

    outcome = build_outcome(
        task_id=task_id,
        sample_id=sample_id,
        episode_id=episode_id,
        status=status,
        reward=reward,
        verifier=verifier,
        testsh_rc=testsh_rc,
    )
    return {OUTCOME_KEY: outcome, MAC_KEY: sign_outcome(outcome, key=key)}


def verified_outcome(metadata: Any) -> tuple[dict[str, Any], float]:
    """Authenticate and validate the signed outcome in arbitrary sample metadata."""

    if not isinstance(metadata, dict):
        raise UntrustedTBenchOutcome("sample metadata is not an object")
    outcome = metadata.get(OUTCOME_KEY)
    supplied_mac = metadata.get(MAC_KEY)
    if not isinstance(outcome, dict) or not isinstance(supplied_mac, str):
        raise UntrustedTBenchOutcome(
            "sample has no signed Terminal-Bench outcome"
        )
    if _MAC.fullmatch(supplied_mac) is None:
        raise UntrustedTBenchOutcome("Terminal-Bench outcome MAC is malformed")
    _validated(outcome)
    expected = sign_outcome(outcome)
    if not hmac.compare_digest(expected, supplied_mac):
        raise UntrustedTBenchOutcome("Terminal-Bench outcome signature mismatch")
    return outcome, float(outcome["reward"])


def verify_outcome(metadata: Any) -> float:
    """Return the authenticated binary benchmark reward."""

    return verified_outcome(metadata)[1]


def validate_hmac_key_source() -> None:
    """Fail closed unless exactly one bounded, private TB key source is ready."""

    _secret()
