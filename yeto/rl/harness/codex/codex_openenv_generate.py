"""``--custom-generate-function-path`` wrapper around upstream ``agentic_tool_call.generate``.

Adds the R-D5a / R-IR-3 bookkeeping that the upstream session path does not
know about:

- sibling samples of one trajectory get the same ``group_index`` and
  ``rollout_id`` so ``train_data_conversion`` shares one reward and counts one
  baseline entry per trajectory (never per chain);
- ``expected_policy_version`` (IR-3: ``rollout_meta_hook.expected_policy_version``,
  i.e. prompt metadata first, else the driver token published through the
  metadata sink by ``MilesRolloutPool.generate``) is compared against the
  weight versions SGLang reported for every generation; any drift aborts the
  trajectory (metadata ``policy_age_violation=1`` -> ``harness_counters`` ->
  driver ``PolicyIdentityError``) instead of giving it a 0 reward; a missing
  token refuses generation before upstream is called;
- mask/token/logprob alignment is asserted on every sample (``alignment``).

The upstream function is injected (``upstream``) so this wrapper is testable
without the Miles package.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from yeto.rl.engine.miles_adapter import rollout_meta_hook

from .alignment import AlignmentError, assert_sample_alignment
from .tbench_reward import INFRASTRUCTURE_KEY

UPSTREAM_PATH = "miles.rollout.generate_hub.agentic_tool_call.generate"
POLICY_AGE_KEY = rollout_meta_hook.POLICY_AGE_VIOLATION_KEY  # "policy_age_violation"
EXPECTED_VERSION_KEY = rollout_meta_hook.EXPECTED_POLICY_VERSION_KEY
ACTUAL_VERSIONS_KEY = "policy_versions_actual"


class PolicyVersionMissing(RuntimeError):
    """IR-3: no target policy token for this rollout (driver did not publish one)."""


def expected_policy_version(input_sample: Any) -> str | None:
    """IR-3 target token for ``input_sample`` (metadata first, else the driver sink)."""
    try:
        return rollout_meta_hook.expected_policy_version(input_sample)
    except Exception:  # noqa: BLE001 - unreachable sink == nothing published
        meta = getattr(input_sample, "metadata", None)
        return str(meta[EXPECTED_VERSION_KEY]) if isinstance(meta, dict) and meta.get(EXPECTED_VERSION_KEY) else None


def _load_upstream() -> Callable[[Any], Awaitable[Any]]:
    from miles.rollout.generate_hub.agentic_tool_call import generate as upstream

    return upstream


def _samples_of(output: Any) -> list[Any]:
    samples = getattr(output, "samples", output)
    return list(samples) if isinstance(samples, list) else [samples]


def _mark_aborted(sample: Any, reason: str) -> None:
    try:
        from miles.utils.types import Sample

        sample.status = Sample.Status.ABORTED
    except ImportError:
        sample.status = "ABORTED"
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    metadata.pop("tbench_trusted_outcome", None)
    metadata.pop("tbench_trusted_outcome_hmac", None)
    metadata[INFRASTRUCTURE_KEY] = reason
    sample.metadata = metadata


def actual_versions(sample: Any) -> list[str]:
    versions: list[str] = []
    for call in getattr(sample, "weight_versions", None) or []:
        spans = getattr(call, "spans", None)
        if spans is None and isinstance(call, dict):
            spans = call.get("spans", [])
        for span in spans or []:
            version = getattr(span, "version", None)
            if version is None and isinstance(span, dict):
                version = span.get("version")
            if version is not None:
                versions.append(str(version))
    return versions


def apply_trajectory_bookkeeping(input_sample: Any, samples: list[Any], *, expected_version: str | None = None) -> None:
    """Siblings share ``group_index``/``rollout_id``; policy version and alignment checks."""
    group_index = getattr(input_sample, "group_index", None)
    rollout_key = getattr(input_sample, "index", None)
    if expected_version is None:
        expected_version = expected_policy_version(input_sample)
    for position, sample in enumerate(samples):
        if group_index is not None:
            sample.group_index = group_index
        if rollout_key is not None:
            sample.rollout_id = rollout_key
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        metadata.setdefault("chain_index", position)
        metadata["chains_total"] = len(samples)
        sample.metadata = metadata
        if INFRASTRUCTURE_KEY in metadata:
            continue
        versions = actual_versions(sample)
        metadata[ACTUAL_VERSIONS_KEY] = versions
        if expected_version is not None:
            metadata[EXPECTED_VERSION_KEY] = expected_version
            if not versions or any(v != str(expected_version) for v in versions):
                metadata[POLICY_AGE_KEY] = 1
                _mark_aborted(sample, f"policy_age_violation: expected {expected_version}, saw {versions}")
                continue
        try:
            assert_sample_alignment(sample)
        except AlignmentError as exc:
            _mark_aborted(sample, f"alignment: {exc}")


async def generate(input: Any, *, upstream: Callable[[Any], Awaitable[Any]] | None = None) -> Any:
    expected_version = expected_policy_version(input.sample)
    if expected_version is None:  # IR-3: refuse before any generation happens
        raise PolicyVersionMissing("expected_policy_version missing: the driver published no policy token")
    output = await (upstream or _load_upstream())(input)
    apply_trajectory_bookkeeping(input.sample, _samples_of(output), expected_version=expected_version)
    return output
