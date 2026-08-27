"""SecRLEnv wrapper for pinned Miles' agentic generate function."""

from __future__ import annotations

import copy
import hashlib
import hmac
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from yeto.rl import SECRLENV_GENERATE_SHA256

from .reward import MAC_KEY, OUTCOME_KEY, UntrustedOutcome, _verified_outcome

_CAPTURED_AGENT_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "secrlenv_agent_metadata", default=None
)


def _verify_pinned_source() -> None:
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not hmac.compare_digest(actual, SECRLENV_GENERATE_SHA256):
        raise RuntimeError("secrlenv generate wrapper does not match its Yeto pin")


_verify_pinned_source()


def capture_agent_metadata(metadata: dict[str, Any]) -> None:
    """Retain one agent result inside its generation task only."""

    _CAPTURED_AGENT_METADATA.set(copy.deepcopy(metadata))


def _generated_samples(output: Any) -> list[Any]:
    samples = getattr(output, "samples", None)
    if isinstance(samples, list):
        flattened: list[Any] = []
        pending = list(samples)
        while pending:
            sample = pending.pop(0)
            if isinstance(sample, list):
                pending[0:0] = sample
            else:
                flattened.append(sample)
        return flattened
    return [] if samples is None else [samples]


def _status_value(sample: Any) -> str | None:
    status = getattr(sample, "status", None)
    value = getattr(status, "value", status)
    return value if isinstance(value, str) else None


async def generate(input: Any) -> Any:
    """Preserve verified evidence across Miles' no-record aborted fallback.

    Pinned Miles intentionally returns an aborted copy when a session emitted no
    model records or all records were truncated, but it does so before merging
    the custom agent's metadata.  SecRLEnv may restore that metadata only after
    authenticating the complete signed outcome; every other missing/invalid
    evidence path remains fatal.
    """

    from miles.rollout.generate_hub.agentic_tool_call import generate as upstream

    token = _CAPTURED_AGENT_METADATA.set(None)
    try:
        output = await upstream(input)
        samples = _generated_samples(output)
        if not samples:
            raise UntrustedOutcome("SecRLEnv generation returned no samples")

        evidence_fields = []
        for sample in samples:
            metadata = getattr(sample, "metadata", None)
            evidence_fields.append(
                (
                    isinstance(metadata, dict) and OUTCOME_KEY in metadata,
                    isinstance(metadata, dict) and MAC_KEY in metadata,
                )
            )
        if all(outcome and mac for outcome, mac in evidence_fields):
            return output
        if (
            any(outcome or mac for outcome, mac in evidence_fields)
            or len(samples) != 1
            or _status_value(samples[0]) != "aborted"
            or getattr(samples[0], "response", None) not in {None, ""}
        ):
            raise UntrustedOutcome("SecRLEnv generation lost signed outcome metadata")

        captured = _CAPTURED_AGENT_METADATA.get()
        _verified_outcome(captured)
        if not isinstance(captured, dict):  # Kept explicit for type narrowing.
            raise UntrustedOutcome("SecRLEnv generation has no signed outcome")
        signed_evidence = {
            OUTCOME_KEY: copy.deepcopy(captured[OUTCOME_KEY]),
            MAC_KEY: captured[MAC_KEY],
        }
        metadata = getattr(samples[0], "metadata", None)
        if not isinstance(metadata, dict):
            raise UntrustedOutcome("SecRLEnv sample metadata is not an object")
        metadata.update(signed_evidence)
        return output
    finally:
        _CAPTURED_AGENT_METADATA.reset(token)


def _add_arguments(parser: Any) -> None:
    from miles.rollout.generate_hub.agentic_tool_call import generate as upstream

    upstream.add_arguments(parser)


generate.add_arguments = _add_arguments
