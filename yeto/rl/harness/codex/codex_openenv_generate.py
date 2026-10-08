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

import dataclasses
from typing import Any, Awaitable, Callable

from yeto.rl.adapters.miles import rollout_meta_hook

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


# --- CompactionRL segments (design D8; progress.md "S13 Codex 桥压缩拦截") ----
SESSIONS_KEY = "codex_compaction_sessions"  # trusted: pre-created by the agent function
COMPACTION_METRICS_KEY = "codex_compaction"  # untrusted bridge record in agent_metrics
SegmentCollector = Callable[[Any, str, str], Awaitable[tuple[list[Any], dict[str, Any]]]]


async def collect_segment_session(input: Any, router: str, session_id: str) -> tuple[list[Any], dict[str, Any]]:
    """Collect (and delete) one extra session exactly like upstream's tracer does."""
    from miles.rollout.generate_utils.openai_endpoint_utils import (
        COMPUTED_FIELDS,
        ROLLOUT_SAMPLING_MASK_FIELDS,
        OpenAIEndpointTracer,
        should_return_sampling_mask,
    )

    fields = COMPUTED_FIELDS
    if should_return_sampling_mask(input.args, input.sampling_params, evaluation=input.evaluation):
        fields += ROLLOUT_SAMPLING_MASK_FIELDS
    tracer = OpenAIEndpointTracer(router_url=router, session_id=session_id, samples_wire_fields=fields)
    reply = await tracer.collect_samples(input.sample, max_seq_len=getattr(input.args, "max_seq_len", None))
    return list(reply.samples), dict(reply.session_metadata or {})


def _optimized_tokens(sample: Any) -> int:
    mask = getattr(sample, "loss_mask", None) or []
    return int(sum(1 for value in mask if value))


async def assemble_compaction_segments(input: Any, output: Any, *, collect: SegmentCollector | None = None) -> Any:
    """Collect the pre-created segment sessions; one sample per segment.

    Every listed session is collected (which deletes it) even when the
    trajectory failed.  Segment ``s`` carries the rollout's metadata (signed
    outcome -> shared reward) plus ``segment_index``/``num_segments``/
    ``segment_tokens``/``tokens_after`` (N_{>s}, optimised tokens of later
    segments)/``gae_length`` (optimised tokens of the whole rollout).
    """
    samples = _samples_of(output)
    if len(samples) != 1:
        raise ValueError("CompactionRL expects one upstream sample for segment 0")
    first = samples[0]
    meta0 = first.metadata if isinstance(getattr(first, "metadata", None), dict) else {}
    session_ids = meta0.get(SESSIONS_KEY)
    if session_ids is None:
        return output
    collect = collect or collect_segment_session
    router = f"http://{meta0.get('session_server_id', '')}"
    collected: list[tuple[list[Any], dict[str, Any]]] = []
    failure: str | None = None
    for session_id in session_ids if isinstance(session_ids, list) else []:
        try:
            collected.append(await collect(input, router, str(session_id)))
        except Exception as exc:  # noqa: BLE001 - keep deleting the remaining sessions
            failure = failure or f"segment collect: {type(exc).__name__}: {exc}"
            collected.append(([], {}))
    if INFRASTRUCTURE_KEY in meta0:
        return output  # already aborted; extra sessions were only drained
    if not isinstance(session_ids, list):
        failure = "segment sessions metadata is not a list"
    record = (meta0.get("agent_metrics") or {}).get(COMPACTION_METRICS_KEY)
    compactions = record.get("compactions") if isinstance(record, dict) else None
    if failure is None and (type(compactions) is not int or not 0 <= compactions <= len(collected)):
        failure = "segment record missing or out of range"
    if failure is None:
        used, unused = collected[:compactions], collected[compactions:]
        if any(len(got) != 1 for got, _ in used) or any(got for got, _ in unused):
            failure = f"segment sessions do not match {compactions} compactions"
    if failure is not None:
        _mark_aborted(first, failure)
        return output
    segments = [first]
    for got, session_metadata in collected[:compactions]:
        sample = got[0]
        sample.metadata = {**meta0, **session_metadata}
        segments.append(sample)
    tokens = [_optimized_tokens(sample) for sample in segments]
    truncated = meta0.get("exit_status") == "max_seq_len"
    for index, sample in enumerate(segments):
        sample.metadata.update(
            {
                "segment_index": index,
                "num_segments": len(segments),
                "segment_tokens": tokens[index],
                "tokens_after": sum(tokens[index + 1 :]),
                "gae_length": sum(tokens),
                "compactions": compactions,
                "truncated": truncated,
                "chain_index": index,
                "chains_total": len(segments),
                "segment_id": index,
                "chain_break_reason": "compaction_window" if index else None,
                "segment_boundary_reason": "compaction" if index < len(segments) - 1 else None,
            }
        )
    if len(segments) > 1:
        # GenerateFnOutput is a frozen dataclass ("multiple samples" is its documented use).
        output = dataclasses.replace(output, samples=segments) if dataclasses.is_dataclass(output) else segments
    return output


async def generate(
    input: Any,
    *,
    upstream: Callable[[Any], Awaitable[Any]] | None = None,
    collect: SegmentCollector | None = None,
) -> Any:
    expected_version = expected_policy_version(input.sample)
    if expected_version is None:  # IR-3: refuse before any generation happens
        raise PolicyVersionMissing("expected_policy_version missing: the driver published no policy token")
    output = await (upstream or _load_upstream())(input)
    samples = _samples_of(output)
    if len(samples) == 1 and isinstance(getattr(samples[0], "metadata", None), dict) and SESSIONS_KEY in samples[0].metadata:
        output = await assemble_compaction_segments(input, output, collect=collect)
    apply_trajectory_bookkeeping(input.sample, _samples_of(output), expected_version=expected_version)
    return output
