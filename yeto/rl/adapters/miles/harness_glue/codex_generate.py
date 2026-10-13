"""SecRLEnv wrapper for pinned Miles' agentic generate function.

Moved from ``yeto/rl/harness/codex/generate.py`` (yeto-framework-decoupling 5.7):
Miles loads ``generate`` by path (``--custom-generate-function-path``).  The
agent-metadata capture it reads stays neutral in ``yeto.rl.harness.codex.generate``.
This file verifies itself against ``yeto.rl.harness.codex.pins.GENERATE_SHA256``.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
from pathlib import Path
from typing import Any

from yeto.rl.harness.codex.generate import _CAPTURED_AGENT_METADATA
from yeto.rl.harness.codex.pins import GENERATE_SHA256 as SECRLENV_GENERATE_SHA256
from yeto.rl.harness.codex.reward import OUTCOME_KEY, UntrustedOutcome, _verified_outcome


def _verify_pinned_source() -> None:
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not hmac.compare_digest(actual, SECRLENV_GENERATE_SHA256):
        raise RuntimeError("secrlenv generate wrapper does not match its Yeto pin")


_verify_pinned_source()


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
    validating the complete outcome; every other missing/invalid
    evidence path remains fatal.

    The restored sample is marked ``failed``, not ``aborted``: Miles drops a
    group with an aborted sample before the dynamic filter runs, so the
    SecRLEnv group check would never see it and could not schedule the
    same-task retry.  A ``failed`` sample still never trains (the group check
    rejects any status other than completed or truncated).
    """

    from miles.rollout.generate_hub.agentic_tool_call import generate as upstream

    token = _CAPTURED_AGENT_METADATA.set(None)
    try:
        output = await upstream(input)
        samples = _generated_samples(output)
        if not samples:
            raise UntrustedOutcome("SecRLEnv generation returned no samples")

        has_outcome = [
            isinstance(getattr(sample, "metadata", None), dict)
            and OUTCOME_KEY in sample.metadata
            for sample in samples
        ]
        if all(has_outcome):
            return output
        if (
            any(has_outcome)
            or len(samples) != 1
            or _status_value(samples[0]) != "aborted"
            or getattr(samples[0], "response", None) not in {None, ""}
        ):
            raise UntrustedOutcome("SecRLEnv generation lost outcome metadata")

        captured = _CAPTURED_AGENT_METADATA.get()
        _verified_outcome(captured)
        if not isinstance(captured, dict):  # Kept explicit for type narrowing.
            raise UntrustedOutcome("SecRLEnv generation has no outcome")
        evidence = {OUTCOME_KEY: copy.deepcopy(captured[OUTCOME_KEY])}
        metadata = getattr(samples[0], "metadata", None)
        if not isinstance(metadata, dict):
            raise UntrustedOutcome("SecRLEnv sample metadata is not an object")
        metadata.update(evidence)
        from miles.utils.types import Sample

        samples[0].status = Sample.Status.FAILED
        return output
    finally:
        _CAPTURED_AGENT_METADATA.reset(token)


def _add_arguments(parser: Any) -> None:
    from miles.rollout.generate_hub.agentic_tool_call import generate as upstream

    upstream.add_arguments(parser)


generate.add_arguments = _add_arguments
