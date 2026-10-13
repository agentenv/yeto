"""Miles rollout with yeto's group/version queue contracts (moved from the legacy
engine, rl-code-quality-cleanup 1.3; moved first, split later).

``generate_rollout`` is Miles' ``--rollout-function-path`` /
``--eval-function-path`` for yeto policy-sync runs and the SAO streaming
runtime.  It wraps upstream ``miles.rollout.sglang_rollout.generate_rollout``
and adds: the completed-group queue (restore/save across spot restarts),
policy-version checks per group, the SecRLEnv replacement/cleanup data
source, evaluation scalar metrics and trajectory evidence.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from yeto.rl import (
    SECRLENV_AGENTS,
    SECRLENV_GROUP_FILTER,
)
from yeto.rl.core import (
    PolicySnapshot,
    StrictRlInvariantError,
    parse_policy_snapshot_token,
)
from yeto.rl.engine.progress import (
    _ISLAND_CHECKPOINT_SCHEMA,
    _atomic_save_island_checkpoint,
    _island_checkpoint_config,
    _load_decoupled_checkpoint,
    _save_decoupled_checkpoint,
)
from yeto.rl.compaction import validate_compaction_segments
from yeto.rl.engine.events import append_tape_event as _append_rl_event
from yeto.rl.adapters.miles.publish import (
    _policy_token,
    get_current_published_policy_identity,
)


def _validate_rollout_groups(data: object, groups: int, samples: int) -> None:
    if not isinstance(data, list) or len(data) != groups:
        raise RuntimeError(
            f"Miles produced {len(data) if isinstance(data, list) else 0} "
            f"groups, expected {groups}"
        )
    for index, group in enumerate(data):
        if not isinstance(group, list) or len(group) != samples:
            raise RuntimeError(
                f"Miles group {index} contains "
                f"{len(group) if isinstance(group, list) else 0} trajectories, "
                f"expected {samples}"
            )
        for sample in group:
            segments = sample if isinstance(sample, list) else [sample]
            if not segments:
                raise RuntimeError("Miles returned an empty segmented trajectory")
            for segment in segments:
                status = getattr(getattr(segment, "status", None), "value", None)
                if isinstance(segment, list) or status not in {"completed", "truncated"}:
                    raise RuntimeError("Miles returned an incomplete trajectory")
            if isinstance(sample, list):
                validate_compaction_segments(segments)


def _policy_token_for_rollout(args, rollout_id: int) -> str:
    if getattr(args, "yeto_rl_sync_preset", "strict-avg") not in {
        "decoupled",
        "sao-streaming-full",
    }:
        return _policy_token(rollout_id)
    token = getattr(args, "yeto_rl_policy_token", None)
    if token is None:
        checkpoint = _load_decoupled_checkpoint(args)
        token = None if checkpoint is None else checkpoint["policy_token"]
    try:
        token_rollout_id, _ = parse_policy_snapshot_token(token)
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    if token_rollout_id != rollout_id:
        raise RuntimeError("decoupled rollout ID differs from its policy snapshot")
    return str(token)


def _validate_rollout_policy_versions(data: list[list[Any]], expected: str) -> None:
    for group in data:
        for sample in group:
            for segment in sample if isinstance(sample, list) else [sample]:
                versions = getattr(segment, "weight_versions", None)
                if (
                    not isinstance(versions, list)
                    or not versions
                    or any(
                        not isinstance(version, str) or version != expected
                        for version in versions
                    )
                ):
                    raise StrictRlInvariantError(
                        "mixed_version_group_count",
                        f"Miles rollout did not use only policy {expected}",
                    )


def _record_strict_failure(args, error: StrictRlInvariantError, bridge=None) -> None:
    _append_rl_event(
        args,
        {
            "event": "rl_strict_failure",
            "metric": error.metric,
            "value": 1,
            "error": f"{type(error).__name__}: {error}",
        },
    )
    print(
        f"[yeto-rl-strict-failure] {error.metric}: {error}",
        file=sys.stderr,
        flush=True,
    )
    if bridge is not None:
        bridge.client.close()


def _complete_group_for_policy(
    group: object,
    policy: int | str,
    size: int,
) -> bool:
    expected = _policy_token(policy) if isinstance(policy, int) else policy
    if not isinstance(group, list) or len(group) != size:
        return False
    for sample in group:
        segments = sample if isinstance(sample, list) else [sample]
        if not segments:
            return False
        for segment in segments:
            status = getattr(getattr(segment, "status", None), "value", None)
            versions = getattr(segment, "weight_versions", None)
            if status not in {"completed", "truncated"} or not versions:
                return False
            if any(not isinstance(token, str) or token != expected for token in versions):
                return False
    return True


def _group_indices(group: object) -> tuple[int, ...] | None:
    if not isinstance(group, list):
        return None
    try:
        return tuple(
            int((sample[0] if isinstance(sample, list) and sample else sample).index)
            for sample in group
        )
    except (AttributeError, TypeError, ValueError):
        return None


class SecRLEnvRolloutTerminalError(RuntimeError):
    """Sanitized base class for non-recoverable SecRLEnv rollout failures."""


class SecRLEnvReplacementExhausted(SecRLEnvRolloutTerminalError):
    """The one authenticated same-task replacement did not restore progress."""


class SecRLEnvUntrustedEvidence(SecRLEnvRolloutTerminalError):
    """A rollout group did not carry authentic SecRLEnv evidence."""


class SecRLEnvRolloutCleanupError(SecRLEnvRolloutTerminalError):
    """Pinned Miles could not drain a terminal SecRLEnv rollout."""


_SECRLENV_TERMINAL_CLEANUP_TIMEOUT_SECONDS = 240.0


def _consume_background_task_result(task: asyncio.Task[Any]) -> None:
    """Prevent a timed-out cleanup task from surfacing payload-bearing errors."""

    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        pass


def _retry_group_indices(group: object) -> tuple[int, ...] | None:
    if not isinstance(group, list) or not group:
        return None
    values = []
    for item in group:
        while isinstance(item, list):
            if not item:
                return None
            item = item[0]
        try:
            values.append(int(item.index))
        except (AttributeError, TypeError, ValueError):
            return None
    return tuple(values)


def _sample_statuses(group: object) -> list[str | None]:
    values: list[str | None] = []
    if not isinstance(group, list):
        return values
    pending = list(group)
    while pending:
        item = pending.pop(0)
        if isinstance(item, list):
            pending[0:0] = item
            continue
        status = getattr(item, "status", None)
        value = getattr(status, "value", status)
        values.append(value if isinstance(value, str) else None)
    return values


class _SecRLEnvRetryDataSource:
    """Reserve Miles' one spare slot for an exact same-task retry."""

    def __init__(self, args: Any, source: Any) -> None:
        target = getattr(args, "rollout_batch_size", None)
        capacity = getattr(args, "over_sampling_batch_size", None)
        limit = getattr(
            args,
            "yeto_rl_secrlenv_max_infrastructure_replacements",
            None,
        )
        if (
            type(target) is not int
            or target <= 0
            or type(capacity) is not int
            or capacity != target + 1
            or type(limit) is not int
            or limit != 1
        ):
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout replacement contract is invalid"
            )
        self.args = args
        self.source = source
        self.target = target
        self.capacity = capacity
        self.limit = limit
        self.initialized = False
        self.templates: dict[tuple[int, ...], list[Any]] = {}
        self.queued: list[list[Any]] = []
        self.seen_evidence: set[object] = set()
        self.used = 0
        self.stats = {
            "replacement_attempts": 0,
            "nontrainable_groups": 0,
            "infrastructure_groups": 0,
            "aborted_groups": 0,
            "pending_groups": 0,
            "failed_groups": 0,
        }
        args._yeto_secrlenv_retry_stats = self.stats

    def _remember(self, groups: list[list[Any]]) -> None:
        for group in groups:
            key = _retry_group_indices(group)
            if key is None:
                raise SecRLEnvReplacementExhausted(
                    "SecRLEnv rollout task identity is unavailable"
                )
            self.templates[key] = copy.deepcopy(group)

    def get_samples(self, requested: int) -> list[list[Any]]:
        if type(requested) is not int or requested != self.capacity:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout refill contract is invalid"
            )
        if not self.initialized:
            groups = self.source.get_samples(self.target)
            if not isinstance(groups, list) or len(groups) != self.target:
                raise SecRLEnvReplacementExhausted(
                    "SecRLEnv rollout did not receive its exact task batch"
                )
            self._remember(groups)
            self.initialized = True
            return groups
        if not self.queued:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout refill had no authenticated task retry"
            )
        groups, self.queued = self.queued, []
        return groups

    def add_samples(self, groups: list[list[Any]]) -> None:
        self.source.add_samples(groups)

    def request_retry(
        self,
        group: list[Any],
        *,
        evidence_key: object,
        reason: str,
        infrastructure_503_retries: Mapping[int, int],
    ) -> None:
        if evidence_key in self.seen_evidence:
            return
        self.seen_evidence.add(evidence_key)
        if self.used >= self.limit or self.queued:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout replacement budget was exhausted"
            )
        key = _retry_group_indices(group)
        template = self.templates.get(key) if key is not None else None
        if template is None:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout replacement task identity did not match"
            )
        expected_indices = set(key)
        if (
            not isinstance(infrastructure_503_retries, Mapping)
            or set(infrastructure_503_retries) != expected_indices
            or any(
                type(index) is not int or type(value) is not int or value not in {0, 1}
                for index, value in infrastructure_503_retries.items()
            )
        ):
            raise SecRLEnvUntrustedEvidence(
                "SecRLEnv rollout retry ledger authentication failed"
            )
        retry = copy.deepcopy(template)
        for sample in retry:
            reset = getattr(sample, "reset_for_retry", None)
            if not callable(reset):
                raise SecRLEnvReplacementExhausted(
                    "SecRLEnv rollout sample cannot be reset safely"
                )
            reset()
            metadata = getattr(sample, "metadata", None)
            if isinstance(metadata, dict):
                metadata.pop("secrlenv_trusted_outcome", None)
                metadata.pop("secrlenv_trusted_outcome_hmac", None)
                metadata["secrlenv_infrastructure_503_retries"] = (
                    infrastructure_503_retries[int(sample.index)]
                )
        statuses = _sample_statuses(group)
        self.stats["nontrainable_groups"] += 1
        self.stats["aborted_groups"] += int("aborted" in statuses)
        self.stats["pending_groups"] += int("pending" in statuses)
        self.stats["failed_groups"] += int("failed" in statuses)
        self.stats["infrastructure_groups"] += int(
            reason == "secrlenv_infrastructure_failure"
        )
        self.used += 1
        self.stats["replacement_attempts"] = self.used
        self.queued.append(retry)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)


def _secrlenv_generate_with_cleanup(
    args: Any,
    rollout_id: int,
    data_source: Any,
    *,
    evaluation: bool = False,
):
    """Run pinned Miles and drain its in-flight work on a terminal filter error."""

    from miles.rollout import sglang_rollout
    from miles.utils.async_utils import run

    from yeto.rl.harness.codex.reward import UntrustedOutcome

    if evaluation:
        return sglang_rollout.generate_rollout(
            args, rollout_id, data_source, evaluation=True
        )

    async def generate():
        try:
            output, aborted = await sglang_rollout.generate_rollout_async(
                args, rollout_id, data_source.get_samples
            )
        except (SecRLEnvRolloutTerminalError, UntrustedOutcome) as error:
            if isinstance(error, UntrustedOutcome):
                terminal_error = SecRLEnvUntrustedEvidence(
                    "SecRLEnv rollout evidence authentication failed"
                )
            else:
                terminal_error = error
            try:
                abort_task = asyncio.create_task(sglang_rollout.abort(args, rollout_id))
                done, _pending = await asyncio.wait(
                    {abort_task},
                    timeout=_SECRLENV_TERMINAL_CLEANUP_TIMEOUT_SECONDS,
                )
                if not done:
                    abort_task.add_done_callback(_consume_background_task_result)
                    abort_task.cancel()
                    raise TimeoutError
                await abort_task
                from yeto.rl.harness.codex.agent import require_no_episode_residue

                require_no_episode_residue()
            except Exception:
                raise SecRLEnvRolloutCleanupError(
                    "SecRLEnv terminal rollout cleanup failed"
                ) from None
            raise terminal_error from None
        data_source.add_samples(aborted)
        return output

    return run(generate())


def _serialize_completed_groups(groups: list[list[Any]]) -> list[list[dict[str, Any]]]:
    serialized = []
    for group in groups:
        serialized_group = []
        for sample in group:
            segments = sample if isinstance(sample, list) else [sample]
            values = []
            for segment in segments:
                to_dict = getattr(segment, "to_dict", None)
                if not callable(to_dict):
                    raise TypeError("pinned Miles Sample lacks checkpoint serialization")
                values.append(to_dict())
            serialized_group.append(values if isinstance(sample, list) else values[0])
        serialized.append(serialized_group)
    return serialized


def _deserialize_completed_groups(groups: object) -> list[list[Any]]:
    from miles.utils.types import Sample

    if not isinstance(groups, list):
        return []
    try:
        return [
            [
                [Sample.from_dict(segment) for segment in sample]
                if isinstance(sample, list)
                else Sample.from_dict(sample)
                for sample in group
            ]
            for group in groups
            if isinstance(group, list)
        ]
    except (TypeError, ValueError) as error:
        raise RuntimeError("invalid Miles samples in island checkpoint") from error


def _restore_completed_groups(args, policy_version: int, data_source) -> None:
    if getattr(data_source, "_yeto_rl_checkpoint_loaded", False):
        return
    data_source._yeto_rl_checkpoint_loaded = True
    if getattr(args, "yeto_rl_sync_preset", "strict-avg") == "decoupled":
        try:
            payload = _load_decoupled_checkpoint(args)
        except RuntimeError as error:
            print(f"[rl] discarded unreadable decoupled island checkpoint: {error}")
            return
        expected_token = _policy_token_for_rollout(args, policy_version)
        if payload is None or payload.get("policy_token") != expected_token:
            return
        groups = [
            group
            for group in _deserialize_completed_groups(payload.get("completed_groups"))
            if _complete_group_for_policy(
                group,
                expected_token,
                args.n_samples_per_prompt,
            )
        ]
        if groups:
            data_source.add_samples(groups)
        return
    path = Path(args.yeto_rl_completed_groups_path).expanduser()
    if not path.is_file():
        return
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # local queue is disposable, global state is not
        print(f"[rl] discarded unreadable island checkpoint {path}: {error}")
        return
    expected_config = _island_checkpoint_config(args)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _ISLAND_CHECKPOINT_SCHEMA
        or payload.get("policy_version") != policy_version
        or payload.get("config") != expected_config
    ):
        print(
            f"[rl] discarded island checkpoint outside policy/config contract: {path}"
        )
        return
    groups = [
        group
        for group in _deserialize_completed_groups(payload.get("completed_groups"))
        if _complete_group_for_policy(group, policy_version, args.n_samples_per_prompt)
    ]
    if groups:
        data_source.add_samples(groups)


def _save_completed_groups(
    args,
    policy_version: int,
    local_round_id: int,
    data_source,
    metrics: Mapping[str, Any] | None,
) -> None:
    buffer = getattr(data_source, "buffer", None)
    if not isinstance(buffer, list):
        raise RuntimeError("Miles data source lacks a completed-group queue")
    policy_token = _policy_token_for_rollout(args, policy_version)
    completed = [
        group
        for group in buffer
        if _complete_group_for_policy(group, policy_token, args.n_samples_per_prompt)
    ]
    buffer[:] = completed
    numeric_metrics = {
        str(name): float(value)
        for name, value in (metrics or {}).items()
        if isinstance(value, (int, float))
    }
    if getattr(args, "yeto_rl_sync_preset", "strict-avg") == "decoupled":
        payload = _load_decoupled_checkpoint(args)
        if (
            payload is None
            or payload.get("next_rollout_id") != policy_version
            or payload.get("policy_token") != policy_token
        ):
            raise RuntimeError("decoupled RL progress changed during rollout")
        _save_decoupled_checkpoint(
            args,
            snapshot=PolicySnapshot(
                policy_version,
                tuple(payload["fragment_versions"]),
                payload["policy_hash"],
            ),
            optimizer_steps=payload["optimizer_steps"],
            action_tokens=payload["action_tokens"],
            rollout_metrics=numeric_metrics,
            local_round_stats=payload.get("local_round_stats"),
            completed_groups=_serialize_completed_groups(completed),
        )
        return
    _atomic_save_island_checkpoint(
        Path(args.yeto_rl_completed_groups_path).expanduser(),
        {
            "schema_version": _ISLAND_CHECKPOINT_SCHEMA,
            "config": _island_checkpoint_config(args),
            "local_round_id": local_round_id,
            "policy_version": policy_version,
            "rollout_metrics": numeric_metrics,
            "local_round_stats": None,
            "completed_groups": _serialize_completed_groups(completed),
        },
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _run_rollout_with_metrics(generate, args, rollout_id, data_source, evaluation):
    from miles.rollout import sglang_rollout

    state = {
        "active": 0,
        "peak_active": 0,
        "cancelled": 0,
        "durations": [],
    }
    generate_state = getattr(sglang_rollout, "GenerateState", None)
    original = getattr(generate_state, "submit_generate_tasks", None)
    if original is None:
        raise RuntimeError("pinned Miles lacks GenerateState.submit_generate_tasks")

    def submit_generate_tasks(self, samples):
        before = set(self.pendings)
        original(self, samples)
        started = time.monotonic()
        tasks = self.pendings - before
        state["active"] += len(tasks)
        state["peak_active"] = max(state["peak_active"], state["active"])

        def finished(task):
            state["active"] -= 1
            state["durations"].append(time.monotonic() - started)
            if task.cancelled():
                state["cancelled"] += 1
                return
            try:
                group = task.result()
            except Exception:
                state["cancelled"] += 1
                return
            statuses = _sample_statuses(group)
            if any(status not in {"completed", "truncated"} for status in statuses):
                state["cancelled"] += 1

        for task in tasks:
            task.add_done_callback(finished)

    generate_state.submit_generate_tasks = submit_generate_tasks
    try:
        return generate(args, rollout_id, data_source, evaluation=evaluation), state
    finally:
        generate_state.submit_generate_tasks = original


class _SuppressMilesEvalPayload(logging.Filter):
    """Drop Miles' one prompt/response example while retaining scalar logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(
            "eval_rollout_single_dataset example data:"
        )


def _attach_eval_scalar_metrics(output: Any) -> list[dict[str, Any]]:
    """Add payload-free SecRLEnv result/pass@1 metrics to Miles' eval output."""

    data = getattr(output, "data", None)
    if not isinstance(data, Mapping) or not data:
        raise RuntimeError("Miles evaluation returned no named datasets")
    metrics = dict(getattr(output, "metrics", None) or {})
    results = []
    for dataset_name, dataset in data.items():
        rewards = dataset.get("rewards") if isinstance(dataset, Mapping) else None
        if not isinstance(rewards, list) or not rewards:
            raise RuntimeError(
                f"Miles evaluation dataset {dataset_name!r} returned no rewards"
            )
        samples = dataset.get("samples")
        if not isinstance(samples, list) or len(samples) != len(rewards):
            raise RuntimeError("Miles evaluation returned incomplete sample evidence")
        statuses = [
            getattr(getattr(sample, "status", None), "value", None)
            for sample in samples
        ]
        if any(status not in {"completed", "truncated"} for status in statuses):
            raise RuntimeError("Miles evaluation returned an aborted sample")
        values = []
        for reward in rewards:
            if isinstance(reward, bool):
                raise RuntimeError("Miles evaluation returned a non-numeric reward")
            try:
                value = float(reward)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "Miles evaluation returned a non-numeric reward"
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError("Miles evaluation reward is outside [0, 1]")
            values.append(value)
        prefix = f"eval/{dataset_name}"
        result = statistics.fmean(values)
        pass_at_1 = statistics.fmean(1.0 if value == 1.0 else 0.0 for value in values)
        metrics[prefix] = result
        metrics[f"{prefix}-pass@1"] = pass_at_1
        results.append(
            {
                "dataset_name": str(dataset_name),
                "sample_count": len(values),
                "rl/eval/result": result,
                "rl/eval/pass_at_1": pass_at_1,
            }
        )
    output.metrics = metrics
    return results


def _persist_trajectory_evidence(args: Any, rollout_id: int, groups: Any) -> None:
    """Persist the closed accepted batch before Miles discards Sample metadata."""

    directory = getattr(args, "yeto_rl_trajectory_evidence_dir", None)
    if directory is None:
        return
    _, policy_hash = get_current_published_policy_identity(
        args,
        expected_policy_version=rollout_id,
    )
    reward_hash = getattr(args, "yeto_rl_reward_sha256", None)
    evidence_kind = getattr(args, "yeto_rl_trajectory_evidence_kind", None)
    if evidence_kind not in {"secrlenv", "terminal-bench-2.1"}:
        raise RuntimeError(
            "trajectory evidence kind is not bound by the launch context"
        )
    evidence_schema_version = getattr(
        args, "yeto_rl_trajectory_evidence_schema_version", None
    )
    if evidence_schema_version not in {1, 2}:
        raise RuntimeError(
            "trajectory evidence schema is not bound by the launch context"
        )
    from yeto.rl.trajectory_evidence import (
        build_trajectory_batch_evidence,
        write_trajectory_batch_evidence,
    )

    evidence = build_trajectory_batch_evidence(
        groups,
        rollout_id=rollout_id,
        behavior_policy_hash=policy_hash,
        reward_contract_hash=reward_hash,
        evidence_kind=evidence_kind,
        schema_version=evidence_schema_version,
    )
    write_trajectory_batch_evidence(directory, evidence)


def generate_rollout(args, rollout_id: int, data_source, evaluation: bool = False):
    """Keep Miles rollout, adding group/version queue contracts."""

    from miles.rollout.sglang_rollout import generate_rollout as miles_generate

    if not evaluation:
        args.yeto_rl_policy_version = rollout_id
        args._yeto_dynamic_sampling_stats = None
        expected_policy_token = _policy_token_for_rollout(args, rollout_id)
        _restore_completed_groups(args, rollout_id, data_source)
        buffer = getattr(data_source, "buffer", None)
        if not isinstance(buffer, list):
            raise RuntimeError("Miles data source lacks a completed-group queue")
        buffer[:] = [
            group
            for group in buffer
            if _complete_group_for_policy(
                group, expected_policy_token, args.n_samples_per_prompt
            )
        ]
    rollout_started = time.monotonic()
    rollout_data_source = data_source
    rollout_generate = miles_generate
    retry_callback = None
    if (
        not evaluation
        and getattr(args, "custom_agent_function_path", None) in SECRLENV_AGENTS
    ):
        if getattr(args, "dynamic_sampling_filter_path", None) != SECRLENV_GROUP_FILTER:
            raise SecRLEnvReplacementExhausted(
                "SecRLEnv rollout group filter contract is invalid"
            )
        rollout_data_source = _SecRLEnvRetryDataSource(args, data_source)
        retry_callback = rollout_data_source.request_retry
        args._yeto_secrlenv_retry_callback = retry_callback
        rollout_generate = _secrlenv_generate_with_cleanup
    payload_filter = None
    upstream_logger = None
    if evaluation:
        payload_filter = _SuppressMilesEvalPayload()
        upstream_logger = logging.getLogger("miles.rollout.sglang_rollout")
        upstream_logger.addFilter(payload_filter)
    try:
        output, lifecycle = _run_rollout_with_metrics(
            rollout_generate,
            args,
            rollout_id,
            rollout_data_source,
            evaluation,
        )
    finally:
        if (
            retry_callback is not None
            and getattr(args, "_yeto_secrlenv_retry_callback", None) == retry_callback
        ):
            del args._yeto_secrlenv_retry_callback
        if payload_filter is not None and upstream_logger is not None:
            upstream_logger.removeFilter(payload_filter)
    if evaluation:
        policy_version = int(getattr(args, "yeto_rl_eval_policy_version", -1))
        if policy_version < 0:
            raise RuntimeError("Miles evaluation has no applied policy version")
        policy_hash = getattr(args, "yeto_rl_eval_policy_hash", None)
        for result in _attach_eval_scalar_metrics(output):
            _append_rl_event(
                args,
                {
                    "event": "rl_eval_result",
                    "rollout_id": int(rollout_id),
                    "policy_version": policy_version,
                    **(
                        {"sync/global_policy_hash": policy_hash}
                        if isinstance(policy_hash, str)
                        else {}
                    ),
                    **result,
                },
            )
    if not evaluation:
        if lifecycle["active"] != 0:
            raise RuntimeError("Miles returned with rollout groups still active")
        _validate_rollout_groups(
            output.samples,
            args.rollout_batch_size,
            args.n_samples_per_prompt,
        )
        try:
            _validate_rollout_policy_versions(output.samples, expected_policy_token)
        except StrictRlInvariantError as error:
            _record_strict_failure(args, error)
            raise
        _persist_trajectory_evidence(args, rollout_id, output.samples)
        consumed = {_group_indices(group) for group in output.samples}
        data_source.buffer[:] = [
            group
            for group in data_source.buffer
            if _group_indices(group) not in consumed
        ]
        samples = [
            segment
            for group in output.samples
            for sample in group
            for segment in (sample if isinstance(sample, list) else [sample])
        ]
        tool_wait_seconds = sum(
            float(getattr(sample, "non_generation_time", 0.0)) for sample in samples
        )
        round_metrics = {
            "rollout_seconds": time.monotonic() - rollout_started,
            "active_groups": lifecycle["peak_active"],
            "cancelled_groups": lifecycle["cancelled"],
            "tool_wait_seconds": tool_wait_seconds,
            "group_p50_seconds": _percentile(lifecycle["durations"], 0.50),
            "group_p95_seconds": _percentile(lifecycle["durations"], 0.95),
            "group_p99_seconds": _percentile(lifecycle["durations"], 0.99),
        }
        dynamic_stats = getattr(args, "_yeto_dynamic_sampling_stats", None) or {}
        retry_stats_value = getattr(args, "_yeto_secrlenv_retry_stats", None)
        retry_stats = (
            retry_stats_value if isinstance(retry_stats_value, Mapping) else {}
        )
        retry_attempts = int(retry_stats.get("replacement_attempts", 0))
        nontrainable_groups = int(retry_stats.get("nontrainable_groups", 0))
        generated_groups = int(
            dynamic_stats.get("generated_groups", args.rollout_batch_size)
        )
        generated_groups = max(
            generated_groups, args.rollout_batch_size + retry_attempts
        )
        dropped_groups = (
            int(dynamic_stats.get("dropped_groups", 0)) + nontrainable_groups
        )
        round_metrics.update(
            {
                "rl/dynamic_filter/enabled": float(
                    getattr(args, "dynamic_sampling_filter_path", None) is not None
                ),
                "rl/dynamic_filter/generated_groups": float(generated_groups),
                "rl/dynamic_filter/accepted_groups": float(
                    dynamic_stats.get("accepted_groups", args.rollout_batch_size)
                ),
                "rl/dynamic_filter/dropped_groups": float(dropped_groups),
                "rl/dynamic_filter/replacement_attempts": float(
                    max(
                        retry_attempts,
                        generated_groups - args.rollout_batch_size,
                    )
                ),
            }
        )
        if isinstance(retry_stats_value, Mapping):
            for name in (
                "infrastructure_groups",
                "aborted_groups",
                "pending_groups",
                "failed_groups",
            ):
                round_metrics[f"rl/secrlenv/{name}"] = float(retry_stats.get(name, 0))
        bounded_state = getattr(args, "_yeto_bounded_filter_state", None)
        if isinstance(bounded_state, Mapping):
            round_metrics["rl/dynamic_filter/forced_groups"] = float(
                bounded_state.get("forced", 0)
            )
        for reason, count in dynamic_stats.get("drop_reasons", {}).items():
            round_metrics[f"rl/dynamic_filter/drop_reason/{reason}"] = float(count)
        _save_completed_groups(
            args,
            rollout_id,
            rollout_id + 1,
            data_source,
            {**(output.metrics or {}), **round_metrics},
        )
    return output


# ---------------------------------------------------------------------------
# Miles --rollout-all-samples-process-path: keep complete oversampled groups
# (moved from the legacy engine with the dynamic-sampling filter helpers,
# rl-code-quality-cleanup 1.7; used by the strict/decoupled policy-sync runs).
# ---------------------------------------------------------------------------
def _dynamic_sampling_filter(args):
    """Load the optional Miles group filter once per learner process."""

    path = getattr(args, "dynamic_sampling_filter_path", None)
    if not path:
        return None
    cached = getattr(args, "_yeto_dynamic_sampling_filter", None)
    if cached is None:
        from miles.utils.misc import load_function

        cached = load_function(path)
        args._yeto_dynamic_sampling_filter = cached
    return cached


def _dynamic_sampling_keep(args, group: object) -> tuple[bool, str | None]:
    filter_fn = _dynamic_sampling_filter(args)
    if filter_fn is None:
        return True, None
    result = filter_fn(args, group)
    keep = bool(getattr(result, "keep", result))
    reason = getattr(result, "reason", None)
    return keep, None if reason is None else str(reason)


def queue_completed_groups(args, all_samples, data_source) -> None:
    """Retain complete oversampling results until this round selects its batch."""

    source = getattr(data_source, "__self__", None)
    if source is None or not callable(getattr(source, "add_samples", None)):
        raise RuntimeError("pinned Miles did not provide a bound data source method")
    complete = [
        group
        for group in all_samples
        if _complete_group_for_policy(
            group,
            _policy_token_for_rollout(
                args,
                args.yeto_rl_policy_version,
            ),
            args.n_samples_per_prompt,
        )
    ]
    kept = []
    dropped = 0
    drop_reasons: dict[str, int] = {}
    for group in complete:
        keep, reason = _dynamic_sampling_keep(args, group)
        if keep:
            kept.append(group)
            continue
        dropped += 1
        if reason:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
    args._yeto_dynamic_sampling_stats = {
        "generated_groups": len(complete),
        "accepted_groups": len(kept),
        "dropped_groups": dropped,
        "drop_reasons": drop_reasons,
    }
    _emit_sample_outcomes(args, complete)
    source.add_samples(kept)


def _emit_sample_outcomes(args, complete: list[list[Any]]) -> None:
    """rlf-codex-rl 5.8: per-sample reward/success onto the event tape.

    The ports bridge can only persist an empty completed-group queue, so the
    RL gate's sample-level evidence must ride the tape (which is mirrored to
    the Modal volume and survives the island). Fail-closed: a sample without
    a numeric reward is skipped, and the gate rejects a tape with gaps.
    """
    if not getattr(args, "yeto_rl_event_tape", None):
        return
    from yeto.rl.engine.events import write_event

    policy_version = int(getattr(args, "yeto_rl_policy_version", -1))
    for group_index, group in enumerate(complete):
        for index, sample in enumerate(group):
            first = sample[0] if isinstance(sample, list) else sample
            to_dict = getattr(first, "to_dict", None)
            if not callable(to_dict):
                continue
            data = to_dict()
            reward = data.get("reward")
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                continue
            metadata = data.get("metadata") or {}
            write_event(
                args,
                {
                    "event": "rl_sample_outcome",
                    "policy_version": policy_version,
                    "group_index": group_index,
                    "index": index,
                    "task_id": metadata.get("rlf_task_id") or metadata.get("task_id"),
                    "reward": float(reward),
                    "success": bool(metadata["success"]) if "success" in metadata else None,
                },
            )
