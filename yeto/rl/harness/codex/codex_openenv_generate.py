"""``--custom-generate-function-path`` wrapper around upstream ``agentic_tool_call.generate``.

Adds the R-D5a / R-IR-3 bookkeeping that the upstream session path does not
know about:

- sibling samples of one trajectory get the same ``group_index`` and
  ``rollout_id`` so ``train_data_conversion`` shares one reward and counts one
  baseline entry per trajectory (never per chain);
- ``expected_policy_version`` (IR-3: ``rollout_meta_hook.expected_policy_version``,
  i.e. prompt metadata first, else the driver token published through the
  metadata sink by ``MilesRolloutPool.generate``) is compared against the
  weight versions SGLang reported for every generation; under the default
  policy-age limit 0 any drift aborts the trajectory (metadata ``policy_age_violation=1`` -> ``harness_counters`` ->
  driver ``PolicyIdentityError``) instead of giving it a 0 reward; a missing
  token refuses generation before upstream is called; under a limit > 0
  (agentic-rollout-utilization 5.2: a trajectory suspended between model turns
  continues under the next version) the versions are recorded as segments and
  checked against the version current when the trajectory finishes: oldest at
  most ``limit`` behind, none newer; outside the window the trajectory is
  discarded (infrastructure ``policy_age_exceeded``, not a policy-identity
  error);
- mask/token/logprob alignment is asserted on every sample (``alignment``).

The upstream function is injected (``upstream``) so this wrapper is testable
without the Miles package.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
from typing import Any, Awaitable, Callable

from yeto.rl.engine import rollout_meta

from .alignment import AlignmentError, assert_sample_alignment
from .tbench_reward import INFRASTRUCTURE_KEY

UPSTREAM_PATH = "miles.rollout.generate_hub.agentic_tool_call.generate"
POLICY_AGE_KEY = rollout_meta.POLICY_AGE_VIOLATION_KEY  # "policy_age_violation"
EXPECTED_VERSION_KEY = rollout_meta.EXPECTED_POLICY_VERSION_KEY
ACTUAL_VERSIONS_KEY = "policy_versions_actual"


class PolicyVersionMissing(RuntimeError):
    """IR-3: no target policy token for this rollout (driver did not publish one)."""


def expected_policy_version(input_sample: Any) -> str | None:
    """IR-3 target token for ``input_sample`` (metadata first, else the driver sink)."""
    try:
        return rollout_meta.expected_policy_version(input_sample)
    except Exception:  # noqa: BLE001 - unreachable sink == nothing published
        meta = getattr(input_sample, "metadata", None)
        return str(meta[EXPECTED_VERSION_KEY]) if isinstance(meta, dict) and meta.get(EXPECTED_VERSION_KEY) else None


def _glue():
    """Backend glue: upstream generate, aborted status, session collection (decoupling 5.7)."""
    from yeto.rl.engine import backends

    return backends.module("harness_glue")


def _load_upstream() -> Callable[[Any], Awaitable[Any]]:
    return _glue().load_agentic_upstream()


def _samples_of(output: Any) -> list[Any]:
    samples = getattr(output, "samples", output)
    return list(samples) if isinstance(samples, list) else [samples]


def _mark_aborted(sample: Any, reason: str) -> None:
    try:
        sample.status = _glue().aborted_status()
    except ImportError:  # backend not installed (CPU tests)
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


POLICY_AGE_EXCEEDED = "policy_age_exceeded"
VERSION_SEGMENTS_KEY = "policy_version_segments"


def window_problem(versions: list[str], current_token: str | None, limit: int) -> str | None:
    """Limit > 0: why ``versions`` (one per generation span) cannot be trained at
    ``current_token`` (None = within the window)."""
    from yeto.rl.engine.policy_age import token_version

    current = token_version(str(current_token or ""))
    if current is None:
        return f"current policy token {current_token!r} has no version"
    if not versions:
        return "no generation version recorded"
    numbers = [token_version(v) for v in versions]
    if None in numbers:
        return f"unparsable generation versions {sorted(set(versions))}"
    if max(numbers) > current:
        return f"generation version {max(numbers)} is newer than the current {current}"
    if current - min(numbers) > limit:
        return f"oldest generation version {min(numbers)} is {current - min(numbers)} behind {current} (limit {limit})"
    return None


def apply_trajectory_bookkeeping(input_sample: Any, samples: list[Any], *, expected_version: str | None = None,
                                 max_policy_age: int = 0, current_version: str | None = None) -> None:
    """Siblings share ``group_index``/``rollout_id``; policy version and alignment checks.

    ``max_policy_age`` > 0 (5.2): versions are recorded as segments and checked
    against ``current_version`` (the policy token current when the trajectory
    finished; default ``expected_version``) instead of requiring one version."""
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
        if max_policy_age > 0:
            current = current_version if current_version is not None else expected_version
            metadata[EXPECTED_VERSION_KEY] = current
            metadata[VERSION_SEGMENTS_KEY] = sorted(set(versions))
            problem = window_problem(versions, current, max_policy_age)
            if problem is not None:
                _mark_aborted(sample, f"{POLICY_AGE_EXCEEDED}: {problem}")
                continue
        elif expected_version is not None:
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
ROUTER_KEY = "codex_compaction_router"  # trusted: router those sessions live on
SEGMENT_COLLECT_TIMEOUT_ENV = "YETO_CODEX_SEGMENT_COLLECT_TIMEOUT_S"
DEFAULT_SEGMENT_COLLECT_TIMEOUT_S = 120.0


def segment_collect_timeout_s() -> float:
    """Per-session collect budget (env override, positive seconds)."""
    raw = os.environ.get(SEGMENT_COLLECT_TIMEOUT_ENV)
    try:
        value = float(raw) if raw else DEFAULT_SEGMENT_COLLECT_TIMEOUT_S
    except ValueError:
        value = DEFAULT_SEGMENT_COLLECT_TIMEOUT_S
    return value if value > 0 else DEFAULT_SEGMENT_COLLECT_TIMEOUT_S
COMPACTION_METRICS_KEY = "codex_compaction"  # untrusted bridge record in agent_metrics
SegmentCollector = Callable[[Any, str, str], Awaitable[tuple[list[Any], dict[str, Any]]]]


async def collect_segment_session(input: Any, router: str, session_id: str) -> tuple[list[Any], dict[str, Any]]:
    """Collect (and delete) one extra session exactly like upstream's tracer does (backend glue)."""
    return await _glue().collect_segment_session(input, router, session_id)


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
    # The agent function records the router it created the sessions on
    # (s19-compaction-g1-20261010d: session_server_id is not in the returned
    # metadata, the collect URL became "http:///sessions/..." and retried forever).
    router = meta0.get(ROUTER_KEY)
    if not router and meta0.get("session_server_id"):
        router = f"http://{meta0['session_server_id']}"
    if not isinstance(router, str) or not router.startswith(("http://", "https://")) or router.endswith("//"):
        _mark_aborted(first, "segment collect: no session-server router for the segment sessions")
        return output
    collected: list[tuple[list[Any], dict[str, Any]]] = []
    failure: str | None = None
    for session_id in session_ids if isinstance(session_ids, list) else []:
        try:
            collected.append(await asyncio.wait_for(collect(input, router, str(session_id)),
                                                    timeout=segment_collect_timeout_s()))
        except asyncio.TimeoutError:
            # s19-compaction-g1-20261010d: Miles' http_utils retries a failing
            # collect for a long time; bound it so one rollout cannot stall the round.
            failure = failure or f"segment collect: timed out after {segment_collect_timeout_s():g} s"
            collected.append(([], {}))
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
    limit = int(getattr(getattr(input, "args", None), "yeto_rl_max_policy_age", 0) or 0)
    current = None
    if limit > 0:
        # 5.2: the trajectory may have been suspended across a publish; it is
        # trained at the version current NOW (driver token in the metadata sink).
        try:
            current = rollout_meta.current_policy_token() or expected_version
        except Exception:  # noqa: BLE001 - unreachable sink: the start token
            current = expected_version
    apply_trajectory_bookkeeping(input.sample, _samples_of(output), expected_version=expected_version,
                                 max_policy_age=limit, current_version=current)
    return output


def _add_arguments(parser: Any) -> None:
    """Miles parse_args registers a generate function's own flags through its
    ``add_arguments`` attribute; this wrapper hands the agent flags
    (--custom-agent-function-path, --max-seq-len) to upstream
    agentic_tool_call.generate, so it must register upstream's flags too
    (s19-compaction-g1-20261010c: "unrecognized arguments")."""
    upstream = _load_upstream()
    add = getattr(upstream, "add_arguments", None)
    if callable(add):
        add(parser)


generate.add_arguments = _add_arguments
