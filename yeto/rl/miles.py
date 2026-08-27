"""Adapter for the pinned Miles commit's Megatron/SGLang runtime."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib
import json
import logging
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from . import (
    MILES_COMMIT,
    MILES_REPOSITORY,
    SECRLENV_AGENTS,
    SECRLENV_GROUP_FILTER,
)
from .core import (
    STRICT_STREAM_CHUNK_BYTES,
    CanonicalLoraState,
    LocalRoundStats,
    PolicySnapshot,
    StrictRlInvariantError,
    canonical_state,
    canonical_state_from_validated_owned_tensors,
    parse_policy_snapshot_token,
    policy_hash,
    policy_tensor_hash,
)
from .decoupled import BroadcastBatch, BudgetConsolidation, FragmentSubmission


def miles_execution_source_sha256(root: str | Path) -> str:
    """Hash the complete Miles Python/runtime-data source used by an RL job."""

    unresolved = Path(root).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise RuntimeError("Miles source root is not a real directory")
    root = unresolved.resolve()
    sources = [root / "train.py"]
    for package_name in ("miles", "miles_plugins"):
        package = root / package_name
        if package.is_symlink() or not package.is_dir():
            raise RuntimeError(f"Miles source package is not a real directory: {package}")
        for path in package.rglob("*"):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise RuntimeError(f"Miles source contains a symlink: {relative}")
            if path.is_file():
                sources.append(path)
            elif not path.is_dir():
                raise RuntimeError(f"Miles source contains a special file: {relative}")
    if not sources[0].is_file() or sources[0].is_symlink():
        raise RuntimeError("Miles source has no regular train.py")

    digest = hashlib.sha256(b"yeto-miles-execution-source-v1\0")
    for path in sorted(sources, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode()
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after or len(payload) != after.st_size:
            raise RuntimeError(f"Miles source changed while hashing: {relative!r}")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def verify_miles_revision(
    root: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> Path:
    """Verify either the pinned clean commit or an exact staged source tree."""

    root = Path(root).expanduser().resolve()

    if expected_source_sha256 is not None:
        expected = expected_source_sha256.lower()
        if len(expected) != 64 or any(value not in _LOWER_HEX for value in expected):
            raise ValueError("Miles source SHA256 is malformed")
        actual = miles_execution_source_sha256(root)
        if actual != expected:
            raise RuntimeError(
                f"Miles source mismatch: expected {expected}, got {actual}"
            )
        miles = importlib.import_module("miles")
        package_path = Path(miles.__file__).resolve()
        if not package_path.is_relative_to(root):
            raise RuntimeError(
                f"imported Miles package {package_path} is outside {root}"
            )
        return root

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot verify Miles checkout at {root}") from exc

    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    origin = git("config", "--get", "remote.origin.url")
    if commit != MILES_COMMIT:
        raise RuntimeError(
            f"Miles revision mismatch: expected {MILES_COMMIT}, got {commit}"
        )
    if origin.removesuffix(".git").rstrip("/") != MILES_REPOSITORY:
        raise RuntimeError(
            f"Miles origin mismatch: expected {MILES_REPOSITORY}, got {origin}"
        )
    if branch != "HEAD":
        raise RuntimeError("Miles checkout is not detached")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Miles checkout is not clean")

    miles = importlib.import_module("miles")
    package_path = Path(miles.__file__).resolve()
    if not package_path.is_relative_to(root):
        raise RuntimeError(f"imported Miles package {package_path} is outside {root}")
    return root


def _policy_token(version: int) -> str:
    return f"yeto:{version}"


_PUBLISHED_POLICY_IDENTITY_ATTRIBUTE = "_yeto_rl_current_published_policy_identity"
_LOWER_HEX = frozenset("0123456789abcdef")


def _validate_published_policy_identity(
    policy_version: Any,
    policy_hash: Any,
) -> tuple[int, str]:
    if type(policy_version) is not int or policy_version < 0:
        raise ValueError("published policy version must be a non-negative integer")
    if (
        type(policy_hash) is not str
        or len(policy_hash) != 64
        or any(character not in _LOWER_HEX for character in policy_hash)
    ):
        raise ValueError("published policy hash must be a lowercase SHA256")
    return policy_version, policy_hash


def set_current_published_policy_identity(
    args: Any,
    *,
    policy_version: int,
    policy_hash: str,
) -> tuple[int, str]:
    """Expose an inference-ready policy identity to trajectory admission.

    The full-parameter policy hook calls this only after the exact policy has
    been applied atomically to Miles and published successfully to SGLang.
    """

    identity = _validate_published_policy_identity(policy_version, policy_hash)
    previous = getattr(args, _PUBLISHED_POLICY_IDENTITY_ATTRIBUTE, None)
    if previous is not None:
        if type(previous) is not tuple or len(previous) != 2:
            raise RuntimeError("current published policy identity is malformed")
        try:
            previous_version, previous_hash = _validate_published_policy_identity(
                *previous
            )
        except ValueError as error:
            raise RuntimeError(
                "current published policy identity is malformed"
            ) from error
        if policy_version < previous_version:
            raise RuntimeError("published policy identity cannot move backwards")
        if policy_version == previous_version and policy_hash != previous_hash:
            raise RuntimeError("published policy hash changed at the same version")
    setattr(args, _PUBLISHED_POLICY_IDENTITY_ATTRIBUTE, identity)
    # The RolloutManager owns a serialized args namespace.  Installing the
    # identity through its remote setter is therefore also the only reliable
    # way to tell rollout admission and evaluation which exact published
    # policy they are sampling.
    args.yeto_rl_policy_token = f"yeto:{policy_version}:{policy_hash}"
    args.yeto_rl_eval_policy_version = policy_version
    args.yeto_rl_eval_policy_hash = policy_hash
    return identity


def get_current_published_policy_identity(
    args: Any,
    *,
    expected_policy_version: int,
) -> tuple[int, str]:
    """Return the exact inference-ready identity or fail closed."""

    _validate_published_policy_identity(expected_policy_version, "0" * 64)
    identity = getattr(args, _PUBLISHED_POLICY_IDENTITY_ATTRIBUTE, None)
    if type(identity) is not tuple or len(identity) != 2:
        raise RuntimeError("Miles has no current published policy identity")
    try:
        policy_version, policy_hash = _validate_published_policy_identity(*identity)
    except ValueError as error:
        raise RuntimeError("current published policy identity is malformed") from error
    if policy_version != expected_policy_version:
        raise RuntimeError(
            "current published policy version does not match the rollout"
        )
    return policy_version, policy_hash


def _version_from_token(token: object) -> int:
    prefix, separator, value = str(token).partition(":")
    if prefix != "yeto" or not separator:
        raise RuntimeError(f"invalid rollout policy token {token!r}")
    try:
        version = int(value)
    except ValueError as exc:
        raise RuntimeError(f"invalid rollout policy token {token!r}") from exc
    if version < 0:
        raise RuntimeError(f"invalid rollout policy token {token!r}")
    return version


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
                metadata = [getattr(segment, "metadata", None) for segment in segments]
                trajectory_ids = {
                    value.get("compaction_trajectory_id")
                    for value in metadata
                    if isinstance(value, Mapping)
                }
                indices = [
                    value.get("compaction_segment_index")
                    if isinstance(value, Mapping)
                    else None
                    for value in metadata
                ]
                types = [
                    value.get("compaction_segment_type")
                    if isinstance(value, Mapping)
                    else None
                    for value in metadata
                ]
                context_budgets = {
                    value.get("compaction_context_budget")
                    for value in metadata
                    if isinstance(value, Mapping)
                }
                if (
                    len(trajectory_ids) != 1
                    or not all(
                        isinstance(value, str) and value for value in trajectory_ids
                    )
                    or indices != list(range(len(segments)))
                    or types != [
                        "execution" if value % 2 == 0 else "summary"
                        for value in range(len(segments))
                    ]
                    or types[-1] != "execution"
                    or any(
                        not isinstance(value, Mapping)
                        or value.get("compaction_schema_version") != 1
                        for value in metadata
                    )
                    or len(context_budgets) != 1
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in context_budgets
                    )
                ):
                    raise RuntimeError("Miles returned malformed compaction segments")
                identities = {
                    (getattr(segment, "group_index", None), getattr(segment, "index", None))
                    for segment in segments
                }
                if len(identities) != 1 or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for identity in identities
                    for value in identity
                ):
                    raise RuntimeError("Miles compaction segments changed trajectory identity")


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


def _append_rl_event(args, event: dict[str, Any]) -> None:
    path = Path(args.yeto_rl_event_tape).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "island_id": int(args.yeto_rl_learner_id),
        "time_unix": time.time(),
        **event,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    # The tape is the island's telemetry; W&B is a second reader of it, not
    # a second instrumentation pass. Writing the file first keeps the tape
    # authoritative when the network is not.
    from .wandb_rl import tee as _wandb_tee

    _wandb_tee(args, event)


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


_ISLAND_CHECKPOINT_SCHEMA = 3
_DECOUPLED_CHECKPOINT_SCHEMA = 4


def _island_checkpoint_config(args) -> dict[str, Any]:
    config = {
        "actor_num_gpus_per_node": args.actor_num_gpus_per_node,
        "actor_num_nodes": args.actor_num_nodes,
        "pipeline_model_parallel_size": args.pipeline_model_parallel_size,
        "advantage_estimator": args.advantage_estimator,
        "model": args.yeto_rl_model,
        "dataset": args.yeto_rl_data,
        "base_model_revision": args.yeto_rl_base_model_revision,
        "rollout_model_revision": getattr(
            args, "yeto_rl_rollout_model_revision", args.yeto_rl_base_model_revision
        ),
        "data_revision": args.yeto_rl_data_revision,
        "seq_length": args.seq_length,
        "seed": args.seed,
        "expert_model_parallel_size": args.expert_model_parallel_size,
        "layout_hash": args.yeto_rl_layout_hash,
        "lr": args.lr,
        "lora_config_hash": args.yeto_rl_lora_config_hash,
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "num_steps_per_rollout": args.num_steps_per_rollout,
        "over_sampling_batch_size": args.over_sampling_batch_size,
        "dynamic_sampling_filter_path": getattr(
            args, "dynamic_sampling_filter_path", None
        ),
        "dynamic_sampling_max_replacements": getattr(
            args, "yeto_rl_dynamic_sampling_max_replacements", None
        ),
        "secrlenv_max_infrastructure_replacements": getattr(
            args,
            "yeto_rl_secrlenv_max_infrastructure_replacements",
            None,
        ),
        "rl_offload_train": bool(getattr(args, "offload_train", False)),
        "rl_distributed_timeout_minutes": getattr(
            args, "distributed_timeout_minutes", 10
        ),
        "reward_sha256": args.yeto_rl_reward_sha256,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_max_response_len": args.rollout_max_response_len,
        "custom_generate_function_path": args.custom_generate_function_path,
        "use_session_server": args.use_session_server,
        "tito_model": args.tito_model,
        "codex_backend_profile": getattr(
            args, "yeto_rl_codex_backend_profile", None
        ),
    }
    codex_harness_contract = getattr(
        args,
        "yeto_rl_codex_harness_contract",
        None,
    )
    if codex_harness_contract is not None:
        config["codex_harness_contract"] = codex_harness_contract
    initial_adapter_sha256 = getattr(
        args,
        "yeto_rl_initial_adapter_sha256",
        None,
    )
    if initial_adapter_sha256 is not None:
        config["initial_adapter_sha256"] = initial_adapter_sha256
    return config


def _decoupled_checkpoint_config(args) -> dict[str, Any]:
    return {
        **_island_checkpoint_config(args),
        "sync_preset": "decoupled",
        "learner_id": args.yeto_rl_learner_id,
        "source_sha256": args.yeto_rl_source_sha256,
        "num_fragments": args.yeto_rl_num_fragments,
        "pipeline": args.yeto_rl_pipeline,
        "local_horizon": args.yeto_rl_local_horizon,
        "total_sweeps": args.yeto_rl_total_sweeps,
        "total_fragment_steps": args.yeto_rl_total_fragment_steps,
        "sync_layout_fingerprint": args.yeto_rl_sync_layout_fingerprint,
        "learner_budget_steps": getattr(
            args,
            "yeto_rl_learner_budget_steps",
            None,
        ),
    }


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

    from yeto_miles_secrlenv.reward import UntrustedOutcome

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
                from yeto_miles_secrlenv.agent import require_no_episode_residue

                require_no_episode_residue()
            except Exception:
                raise SecRLEnvRolloutCleanupError(
                    "SecRLEnv terminal rollout cleanup failed"
                ) from None
            raise terminal_error from None
        data_source.add_samples(aborted)
        return output

    return run(generate())


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
    source.add_samples(kept)


def _atomic_save_island_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_decoupled_checkpoint(args) -> dict[str, Any] | None:
    path = Path(args.yeto_rl_completed_groups_path).expanduser()
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError("cannot read decoupled RL island checkpoint") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _DECOUPLED_CHECKPOINT_SCHEMA
        or payload.get("config") != _decoupled_checkpoint_config(args)
    ):
        raise RuntimeError("decoupled RL island checkpoint configuration changed")
    next_rollout_id = payload.get("next_rollout_id")
    optimizer_steps = payload.get("optimizer_steps")
    action_tokens = payload.get("action_tokens")
    versions = payload.get("fragment_versions")
    token = payload.get("policy_token")
    policy_digest = payload.get("policy_hash")
    try:
        token_rollout_id, token_digest = parse_policy_snapshot_token(token)
    except ValueError as error:
        raise RuntimeError("invalid decoupled RL island checkpoint token") from error
    if (
        not isinstance(next_rollout_id, int)
        or next_rollout_id < 0
        or optimizer_steps != next_rollout_id
        or not isinstance(action_tokens, int)
        or action_tokens < 0
        or token_rollout_id != next_rollout_id
        or token_digest != policy_digest
        or not isinstance(versions, list)
        or len(versions) != args.yeto_rl_num_fragments
        or any(
            not isinstance(version, int)
            or version < 0
            or version > args.yeto_rl_total_fragment_steps
            or (
                version > 0
                and (version - 1) % args.yeto_rl_num_fragments != fragment_id
            )
            for fragment_id, version in enumerate(versions)
        )
        or not isinstance(payload.get("completed_groups"), list)
        or not isinstance(payload.get("rollout_metrics"), Mapping)
        or not isinstance(payload.get("local_round_stats"), Mapping)
    ):
        raise RuntimeError("invalid decoupled RL island checkpoint progress")
    return payload


def _save_decoupled_checkpoint(
    args,
    *,
    snapshot: PolicySnapshot,
    optimizer_steps: int,
    action_tokens: int,
    rollout_metrics: Mapping[str, Any],
    local_round_stats: Mapping[str, Any] | None,
    completed_groups: list[list[dict[str, Any]]],
) -> None:
    if (
        optimizer_steps != snapshot.rollout_id
        or action_tokens < 0
        or len(snapshot.fragment_versions) != args.yeto_rl_num_fragments
    ):
        raise ValueError("decoupled RL checkpoint progress is inconsistent")
    numeric_metrics = {
        str(name): float(value)
        for name, value in rollout_metrics.items()
        if isinstance(value, (int, float))
    }
    _atomic_save_island_checkpoint(
        Path(args.yeto_rl_completed_groups_path).expanduser(),
        {
            "schema_version": _DECOUPLED_CHECKPOINT_SCHEMA,
            "config": _decoupled_checkpoint_config(args),
            "next_rollout_id": snapshot.rollout_id,
            "optimizer_steps": optimizer_steps,
            "action_tokens": action_tokens,
            "policy_token": snapshot.token,
            "policy_hash": snapshot.policy_hash,
            "fragment_versions": list(snapshot.fragment_versions),
            "rollout_metrics": numeric_metrics,
            "local_round_stats": dict(local_round_stats or {}),
            "completed_groups": completed_groups,
        },
    )


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
    from .trajectory_evidence import (
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
        if getattr(args, "yeto_rl_sync_preset", None) == "dense-full":
            identity = get_current_published_policy_identity(
                args,
                expected_policy_version=policy_version,
            )
            if identity is None or identity[1] != policy_hash:
                raise RuntimeError(
                    "dense Miles evaluation has no exact published policy identity"
                )
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


class _BridgeRuntime:
    def __init__(self, initial: CanonicalLoraState, args) -> None:
        self.initial = initial
        self.args = args

    def initialize(self) -> CanonicalLoraState:
        if self.initial is None:
            raise RuntimeError("Miles initial policy was already consumed")
        initial = self.initial
        self.initial = None
        return initial

    def apply_global_policy(self, _state: CanonicalLoraState) -> None:
        pass

    def record_local_round(self, stats: LocalRoundStats) -> None:
        path = Path(self.args.yeto_rl_completed_groups_path).expanduser()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise RuntimeError("cannot update Miles island checkpoint") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _ISLAND_CHECKPOINT_SCHEMA
            or payload.get("policy_version") != stats.base_policy_version
            or payload.get("config") != _island_checkpoint_config(self.args)
        ):
            raise RuntimeError("Miles island checkpoint changed before round commit")
        payload["local_round_id"] = stats.local_round_id
        payload["local_round_stats"] = asdict(stats)
        _atomic_save_island_checkpoint(path, payload)

    def shutdown(self) -> None:
        pass


class MilesPolicySync:
    """Yeto synchronization hook called by the Miles training loop."""

    def __init__(self, args) -> None:
        self.args = args
        self.actor_model = None
        self.rollout_manager = None
        self.bridge = None
        self.current = None
        self.permit = None
        self.optimizer_reset_count = 0

    def _canonical_state(self, state) -> CanonicalLoraState:
        normalize = (
            canonical_state_from_validated_owned_tensors
            if getattr(state, "_yeto_owned_tensors", None) == "canonical-v1"
            else canonical_state
        )
        canonical = normalize(
            state.policy_version,
            state.tensors,
            base_model_revision=self.args.yeto_rl_base_model_revision,
            lora_config_hash=self.args.yeto_rl_lora_config_hash,
            layout_hash=self.args.yeto_rl_layout_hash,
        )
        if state.layout_hash != canonical.layout_hash:
            raise StrictRlInvariantError(
                "layout_hash_mismatch",
                "Miles and Yeto computed different LoRA layouts",
            )
        return canonical

    def _strict_export_groups(self):
        if self.bridge is None:
            raise RuntimeError("Miles strict bridge is not initialized")
        return self.bridge.export_tensor_groups()

    @staticmethod
    def _group_names(groups) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(spec.name for spec in group) for group in groups)

    def _requires_chunked_export(self) -> bool:
        if self.bridge is None:
            return False
        return (
            sum(spec.numel for spec in self.bridge.specs) * 4
            > STRICT_STREAM_CHUNK_BYTES
        )

    def _streamed_policy_hash(
        self,
        exported,
        expected: CanonicalLoraState,
    ) -> str:
        groups = self._strict_export_groups()
        expected_names = tuple(spec.name for group in groups for spec in group)
        if (
            getattr(exported, "_yeto_chunked_export", None) != "owner-sharded-v1"
            or exported.policy_version != expected.policy_version
            or tuple(exported.expected_names) != expected_names
        ):
            discard = getattr(exported, "discard", None)
            if callable(discard):
                discard()
            raise StrictRlInvariantError(
                "layout_hash_mismatch",
                "chunked applied policy identity or canonical order changed",
            )
        digest = hashlib.sha256()
        digest.update(b"yeto-rl-policy-v1\0")
        digest.update(expected.base_model_revision.encode("ascii"))
        digest.update(expected.lora_config_hash.encode("ascii"))
        digest.update(expected.layout_hash.encode("ascii"))
        digest.update(expected.policy_version.to_bytes(8, "little"))
        try:
            total_groups = len(groups)
            for group_index, group in enumerate(groups):
                tensors = exported.take_tensors(group)
                for spec in group:
                    value = tensors[spec.name]
                    digest.update(spec.name.encode("utf-8"))
                    digest.update(memoryview(value.numpy()).cast("B"))
                del tensors
                completed_groups = group_index + 1
                if self._policy_progress_due(completed_groups, total_groups):
                    self._record_policy_apply_progress(
                        expected.policy_version,
                        "hash_progress",
                        completed_groups=completed_groups,
                        total_groups=total_groups,
                    )
            exported.finish()
        except BaseException:
            exported.discard()
            raise
        return digest.hexdigest()

    async def _engines(self) -> list[Any]:
        info = await self.rollout_manager.get_updatable_engines_and_lock.remote()
        engines = list(info.rollout_engines)
        if not engines:
            raise RuntimeError("Miles created no updatable SGLang engine")
        return engines

    async def _set_rollout_token(self, token: str) -> None:
        await asyncio.gather(
            *(
                engine.update_weight_version.remote(token)
                for engine in await self._engines()
            )
        )

    async def _set_rollout_version(self, version: int) -> None:
        await self._set_rollout_token(_policy_token(version))

    async def _apply_global_policy(self, state: CanonicalLoraState) -> None:
        from miles.backends.megatron_utils.trainable_state import make_trainable_state

        self._record_policy_apply_progress(state.policy_version, "begin")
        reset_count = await self.actor_model.apply_trainable_state(
            make_trainable_state(state.policy_version, state.tensors),
            reset_optimizer=True,
        )
        self._record_policy_apply_progress(state.policy_version, "apply_finished")
        export_chunks = getattr(
            self.actor_model,
            "export_trainable_state_chunks",
            None,
        )
        self._record_policy_apply_progress(state.policy_version, "hash_begin")
        if callable(export_chunks):
            groups = self._strict_export_groups()
            exported = await export_chunks(self._group_names(groups))
            applied_hash = self._streamed_policy_hash(exported, state)
        else:
            if self._requires_chunked_export():
                raise RuntimeError(
                    "large strict policy runtime lacks bounded chunk export"
                )
            applied = self._canonical_state(
                await self.actor_model.export_trainable_state()
            )
            applied_hash = policy_hash(applied)
        if applied_hash != policy_hash(state):
            raise StrictRlInvariantError(
                "policy_hash_mismatch_after_apply",
                "policy hash mismatch after trainer apply",
            )
        self._record_policy_apply_progress(state.policy_version, "hash_finished")
        await self._set_rollout_version(state.policy_version)
        self.optimizer_reset_count += 1
        self._append_event(
            {
                "event": "rl_policy_apply",
                "policy_version": state.policy_version,
                "optimizer_reset_count": self.optimizer_reset_count,
                "reset_parameter_count": reset_count,
                "rl/global_policy_version": state.policy_version,
                "rl/optimizer_reset_count": self.optimizer_reset_count,
                "sync/global_policy_hash": applied_hash,
            }
        )

    def _rollout_batches(self, data_pack) -> list[Mapping[str, Any]]:
        import ray

        references = data_pack.get("data_ref")
        if not isinstance(references, list):
            raise RuntimeError("Miles returned an invalid rollout shard list")
        world_size = int(self.args.actor_num_nodes) * int(
            self.args.actor_num_gpus_per_node
        )
        model_parallel_size = (
            int(getattr(self.args, "tensor_model_parallel_size", 1))
            * int(getattr(self.args, "pipeline_model_parallel_size", 1))
            * int(getattr(self.args, "context_parallel_size", 1))
        )
        if world_size <= 0 or model_parallel_size <= 0:
            raise RuntimeError("Miles returned an invalid actor topology")
        if world_size % model_parallel_size:
            raise RuntimeError(
                "Miles actor world size is not divisible by its "
                "TP/PP/CP model-parallel size"
            )
        expected = world_size // model_parallel_size
        if len(references) != expected:
            raise RuntimeError(
                f"Miles returned {len(references)} DP shards, expected {expected}"
            )
        return [ray.get(reference.inner) for reference in references]

    def _rollout_metrics(self, policy_version: int) -> dict[str, float]:
        if getattr(self.args, "yeto_rl_sync_preset", "strict-avg") == "decoupled":
            payload = _load_decoupled_checkpoint(self.args)
            if (
                payload is None
                or payload.get("next_rollout_id") != policy_version
                or payload.get("policy_token")
                != _policy_token_for_rollout(self.args, policy_version)
                or not isinstance(payload.get("rollout_metrics"), Mapping)
            ):
                raise RuntimeError("Miles island checkpoint lacks rollout metrics")
            return {
                name: float(value) for name, value in payload["rollout_metrics"].items()
            }
        path = Path(self.args.yeto_rl_completed_groups_path).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != _ISLAND_CHECKPOINT_SCHEMA
            or payload.get("policy_version") != policy_version
            or payload.get("config") != _island_checkpoint_config(self.args)
            or not isinstance(payload.get("rollout_metrics"), Mapping)
        ):
            raise RuntimeError("Miles island checkpoint lacks rollout metrics")
        try:
            return {
                name: float(value) for name, value in payload["rollout_metrics"].items()
            }
        except (TypeError, ValueError) as error:
            raise RuntimeError("Miles returned invalid Yeto group metrics") from error

    def _round_stats(self, rollout_id: int, data_pack, train_state) -> LocalRoundStats:
        batches = self._rollout_batches(data_pack)
        expected_samples = self.args.rollout_batch_size * self.args.n_samples_per_prompt
        response_lengths = [
            int(value)
            for batch in batches
            for value in batch.get("response_lengths", [])
        ]
        sample_indices = [
            int(value) for batch in batches for value in batch.get("sample_indices", [])
        ]
        if (
            len(response_lengths) != expected_samples
            or len(sample_indices) != expected_samples
            or len(set(sample_indices)) != expected_samples
        ):
            raise RuntimeError("Miles DP rollout shards do not form one complete batch")
        raw_rewards = batches[0].get("raw_reward")
        if not isinstance(raw_rewards, list) or len(raw_rewards) != expected_samples:
            raise RuntimeError("Miles rollout lacks scalar raw rewards")
        if any(batch.get("raw_reward") != raw_rewards for batch in batches[1:]):
            raise RuntimeError("Miles DP rollout shards disagree on raw rewards")
        try:
            rewards = [float(value) for value in raw_rewards]
        except (TypeError, ValueError) as error:
            raise RuntimeError("Miles RL v0 requires scalar rewards") from error
        try:
            mean_kl = (
                None
                if train_state.train_rollout_kl is None
                else float(train_state.train_rollout_kl)
            )
            ess_ratio = (
                None if train_state.ess_ratio is None else float(train_state.ess_ratio)
            )
            clip_fraction = (
                None
                if train_state.pg_clipfrac is None
                else float(train_state.pg_clipfrac)
            )
            train_seconds = float(train_state.train_seconds)
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Miles did not export valid round train statistics"
            ) from error
        telemetry = getattr(train_state, "telemetry", None)
        if telemetry is not None:
            from .deepseek_v4_expert_full_runtime import _TrainTelemetry

            if not isinstance(telemetry, _TrainTelemetry):
                raise RuntimeError("Miles exported invalid typed training telemetry")
            # ``deepseek_v4_expert_full_runtime`` has already reduced the
            # metrics-owner dictionary to a closed scalar-only capsule.  Never
            # forward the original Miles record or inspect arbitrary keys here.
            mean_kl = telemetry.kl if telemetry.kl is not None else mean_kl
            ess_ratio = telemetry.ess if telemetry.ess is not None else ess_ratio
            clip_fraction = (
                telemetry.clipfrac if telemetry.clipfrac is not None else clip_fraction
            )
        if train_seconds < 0:
            raise RuntimeError("Miles exported a negative train duration")
        metrics = self._rollout_metrics(rollout_id)
        group_size = self.args.n_samples_per_prompt
        return LocalRoundStats(
            island_id=int(self.args.yeto_rl_learner_id),
            local_round_id=rollout_id + 1,
            base_policy_version=rollout_id,
            active_groups=int(metrics["active_groups"]),
            completed_groups=self.args.rollout_batch_size,
            cancelled_groups=int(metrics["cancelled_groups"]),
            completed_trajectories=expected_samples,
            action_tokens=sum(response_lengths),
            tool_wait_seconds=metrics["tool_wait_seconds"],
            group_p50_seconds=metrics["group_p50_seconds"],
            group_p95_seconds=metrics["group_p95_seconds"],
            group_p99_seconds=metrics["group_p99_seconds"],
            reward_mean=statistics.fmean(rewards),
            reward_std=statistics.pstdev(rewards),
            zero_variance_group_ratio=sum(
                len(set(rewards[index : index + group_size])) == 1
                for index in range(0, expected_samples, group_size)
            )
            / self.args.rollout_batch_size,
            mean_kl=mean_kl,
            ess_ratio=ess_ratio,
            clip_fraction=clip_fraction,
            delta_l2_norm=0.0,
            rollout_seconds=metrics["rollout_seconds"],
            train_seconds=train_seconds,
            train_step=None if telemetry is None else telemetry.step,
            loss=None if telemetry is None else telemetry.loss,
            pg_loss=None if telemetry is None else telemetry.pg_loss,
            grad_norm=None if telemetry is None else telemetry.grad_norm,
            lr=None if telemetry is None else telemetry.lr,
            pass_rate=None if telemetry is None else telemetry.pass_rate,
            dynamic_filter_generated_groups=int(
                metrics.get("rl/dynamic_filter/generated_groups", 0)
            ),
            dynamic_filter_dropped_groups=int(
                metrics.get("rl/dynamic_filter/dropped_groups", 0)
            ),
            dynamic_filter_replacement_attempts=int(
                metrics.get("rl/dynamic_filter/replacement_attempts", 0)
            ),
        )

    async def _initialize(self, *, actor_model, rollout_manager) -> None:
        from .bridge import StrictRlBridge

        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        initial = self._canonical_state(await actor_model.export_trainable_state())
        runtime = _BridgeRuntime(initial, self.args)
        self.bridge = StrictRlBridge(runtime, self.args.yeto_rl_bridge_config)
        del initial
        self.bridge.start()
        self.current = self.bridge.wait_for_initial_policy()
        await self._apply_global_policy(self.current)
        self.args.start_rollout_id = self.current.policy_version
        if self.current.policy_version < self.args.num_rollout:
            self.permit = self.bridge.wait_for_round()

    async def initialize(self, *, actor_model, rollout_manager) -> None:
        try:
            await self._initialize(
                actor_model=actor_model,
                rollout_manager=rollout_manager,
            )
        except StrictRlInvariantError as error:
            self._record_strict_failure(error)
            raise

    async def _after_local_train(
        self, *, rollout_id, actor_model, rollout_data
    ) -> None:
        if (
            actor_model is not self.actor_model
            or self.current is None
            or self.permit is None
        ):
            raise RuntimeError(
                "Miles called policy synchronization outside an active round"
            )
        if rollout_id != self.current.policy_version:
            raise RuntimeError(
                "Miles rollout ID differs from the global policy version"
            )
        export_chunks = getattr(actor_model, "export_trainable_state_chunks", None)
        base_released = False

        def release_base_before_commit() -> None:
            nonlocal base_released
            if base_released:
                raise RuntimeError("strict base release barrier ran more than once")
            self.bridge.release_current(rollout_id)
            self.current = None
            base_released = True

        if callable(export_chunks):
            groups = self._strict_export_groups()
            train_state = await export_chunks(self._group_names(groups))
            try:
                stats = self._round_stats(rollout_id, rollout_data, train_state)
                self.bridge.submit_chunked_local_state(
                    self.permit,
                    self.current,
                    train_state,
                    stats,
                    before_last_enqueue=release_base_before_commit,
                )
            except BaseException:
                train_state.discard()
                raise
        else:
            if self._requires_chunked_export():
                raise RuntimeError(
                    "large strict policy runtime lacks bounded chunk export"
                )
            train_state = await actor_model.export_trainable_state()
            local = self._canonical_state(train_state)
            stats = self._round_stats(rollout_id, rollout_data, train_state)
            self.bridge.submit_local_state(self.permit, self.current, local, stats)
            release_base_before_commit()
        if not base_released:
            raise RuntimeError("strict base was not released before PUSH commit")
        self.current = self.bridge.wait_for_global_policy(rollout_id + 1)
        await self._apply_global_policy(self.current)
        if self.current.policy_version < self.args.num_rollout:
            self.permit = self.bridge.wait_for_round()
        else:
            self.permit = None

    async def after_local_train(self, *, rollout_id, actor_model, rollout_data) -> None:
        try:
            await self._after_local_train(
                rollout_id=rollout_id,
                actor_model=actor_model,
                rollout_data=rollout_data,
            )
        except StrictRlInvariantError as error:
            self._record_strict_failure(error)
            raise

    async def finalize(self) -> None:
        if self.bridge is None or self.current is None:
            raise RuntimeError("Miles finalized an uninitialized policy synchronizer")
        final = self.bridge.finalize()
        if policy_hash(final) != policy_hash(self.current):
            raise RuntimeError("final policy differs from the committed global policy")
        self.bridge.client.close()

    def _append_event(self, event: dict[str, Any]) -> None:
        _append_rl_event(self.args, event)

    @staticmethod
    def _policy_progress_due(completed: int, total: int) -> bool:
        if completed <= 0 or completed > total or total <= 0:
            return False
        stride = max(1, (total + 15) // 16)
        return completed == total or completed % stride == 0

    def _record_policy_apply_progress(
        self,
        policy_version: int,
        phase: str,
        **progress: int,
    ) -> None:
        marker = {
            "event": "rl_policy_apply_progress",
            "phase": phase,
            "policy_version": int(policy_version),
            **{name: int(value) for name, value in progress.items()},
        }
        self._append_event(marker)
        print(
            "[yeto-rl-policy-apply-progress] "
            + json.dumps(marker, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _record_strict_failure(self, error: StrictRlInvariantError) -> None:
        _record_strict_failure(self.args, error, self.bridge)


class DecoupledMilesPolicySync(MilesPolicySync):
    """Miles hook for full-snapshot RL with pipelined fragment rounds."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self.snapshot: PolicySnapshot | None = None
        self.optimizer_steps = 0
        self.action_tokens = 0
        self.finished = False

    def _canonical_at_progress(self, state, policy_version: int) -> CanonicalLoraState:
        canonical = self._canonical_state(state)
        return canonical_state(
            policy_version,
            canonical.tensors,
            base_model_revision=canonical.base_model_revision,
            lora_config_hash=canonical.lora_config_hash,
            layout_hash=canonical.layout_hash,
            expected_specs=canonical.specs,
        )

    async def _apply_decoupled_policy(
        self,
        state: CanonicalLoraState,
        *,
        reset_optimizer: bool,
    ) -> CanonicalLoraState:
        from miles.backends.megatron_utils.trainable_state import make_trainable_state

        apply_started = time.monotonic()
        reset_count = await self.actor_model.apply_trainable_state(
            make_trainable_state(state.policy_version, state.tensors),
            reset_optimizer=reset_optimizer,
        )
        if not reset_optimizer and reset_count != 0:
            raise RuntimeError("Miles reset optimizer state during fragment apply")
        applied = self._canonical_at_progress(
            await self.actor_model.export_trainable_state(),
            state.policy_version,
        )
        applied_hash = policy_tensor_hash(applied)
        if applied_hash != policy_tensor_hash(state):
            raise StrictRlInvariantError(
                "policy_hash_mismatch_after_apply",
                "policy hash mismatch after trainer apply",
            )
        if reset_optimizer:
            self.optimizer_reset_count += 1
        self._append_event(
            {
                "event": "rl_policy_apply",
                "policy_version": state.policy_version,
                "partial_fragment_apply": not reset_optimizer,
                "optimizer_reset_count": self.optimizer_reset_count,
                "reset_parameter_count": reset_count,
                "rl/optimizer_reset_count": self.optimizer_reset_count,
                "sync/global_policy_hash": applied_hash,
                "sync/apply_seconds": time.monotonic() - apply_started,
            }
        )
        return applied

    def _save_progress(
        self,
        snapshot: PolicySnapshot,
        *,
        stats: LocalRoundStats | None,
    ) -> None:
        previous = _load_decoupled_checkpoint(self.args)
        metrics = {} if previous is None else previous.get("rollout_metrics", {})
        completed_groups = []
        if (
            previous is not None
            and previous.get("policy_token") == snapshot.token
            and previous.get("policy_hash") == snapshot.policy_hash
            and previous.get("fragment_versions") == list(snapshot.fragment_versions)
        ):
            completed_groups = previous["completed_groups"]
        _save_decoupled_checkpoint(
            self.args,
            snapshot=snapshot,
            optimizer_steps=self.optimizer_steps,
            action_tokens=self.action_tokens,
            rollout_metrics=metrics,
            local_round_stats=None if stats is None else asdict(stats),
            completed_groups=completed_groups,
        )

    async def _publish_snapshot(self, snapshot: PolicySnapshot) -> None:
        self.args.yeto_rl_policy_token = snapshot.token
        await self._set_rollout_token(snapshot.token)
        self._append_event(
            {
                "event": "rl_policy_snapshot",
                "rl/rollout_id": snapshot.rollout_id,
                "rl/policy_token": snapshot.token,
                "rl/policy_hash": snapshot.policy_hash,
                "rl/fragment_versions": list(snapshot.fragment_versions),
                "rl/canonical_layout_hash": self.current.layout_hash,
                "rl/sync_layout_fingerprint": (
                    self.args.yeto_rl_sync_layout_fingerprint
                ),
                "rl/mixed_version_group_count": 0,
            }
        )

    def _record_local_round(
        self,
        stats: LocalRoundStats,
        *,
        rollout_id: int,
        batch: BroadcastBatch | None = None,
        submissions: tuple[FragmentSubmission, ...] = (),
        additional_payload_bytes_received: int = 0,
    ) -> None:
        train_metrics = {}
        if stats.train_step is not None:
            train_metrics = {
                "train/step": stats.train_step,
                **{
                    name: value
                    for name, value in (
                        ("train/loss", stats.loss),
                        ("train/pg_loss", stats.pg_loss),
                        ("train/grad_norm", stats.grad_norm),
                        ("train/train_rollout_kl", stats.mean_kl),
                        ("train/ess_ratio", stats.ess_ratio),
                        ("train/pg_clipfrac", stats.clip_fraction),
                        ("train/lr", stats.lr),
                        ("train/pass_rate", stats.pass_rate),
                    )
                    if value is not None
                },
            }
        self._append_event(
            {
                "event": "rl_local_round",
                **asdict(stats),
                **train_metrics,
                "rl/rollout_id": rollout_id,
                "rl/policy_hash": self.snapshot.policy_hash,
                "rl/fragment_versions": list(self.snapshot.fragment_versions),
                "rl/active_groups": stats.active_groups,
                "rl/completed_groups": stats.completed_groups,
                "rl/cancelled_groups": stats.cancelled_groups,
                "rl/completed_trajectories": stats.completed_trajectories,
                "rl/action_tokens": stats.action_tokens,
                "rl/tool_wait_seconds": stats.tool_wait_seconds,
                "rl/reward_mean": stats.reward_mean,
                "rl/reward_std": stats.reward_std,
                "rl/rollout_seconds": stats.rollout_seconds,
                "rl/group_p50_seconds": stats.group_p50_seconds,
                "rl/group_p95_seconds": stats.group_p95_seconds,
                "rl/group_p99_seconds": stats.group_p99_seconds,
                "rl/zero_variance_group_ratio": stats.zero_variance_group_ratio,
                "rl/mixed_version_group_count": 0,
                "rl/local_delta_norm": stats.delta_l2_norm,
                "rl/current_vs_rollout_kl": stats.mean_kl,
                "rl/ess_ratio": stats.ess_ratio,
                "rl/clip_fraction": stats.clip_fraction,
                "sync/applied_fragments": list(
                    () if batch is None else batch.fragment_ids
                ),
                "sync/fragment_payload_bytes_received": additional_payload_bytes_received
                + (0 if batch is None else batch.bytes_received),
                "sync/submitted_fragments": [
                    submission.fragment_id for submission in submissions
                ],
                "sync/fragment_payload_bytes_sent": sum(
                    submission.payload_bytes for submission in submissions
                ),
            }
        )
        for broadcast in () if batch is None else batch.broadcasts:
            self._append_event(
                {
                    "event": "rl_fragment_bcast",
                    "fragment_id": broadcast.fragment_id,
                    "version": broadcast.version,
                    "payload_bytes": broadcast.payload_bytes,
                    "queue_seconds": broadcast.queue_seconds,
                }
            )
        for submission in submissions:
            self._append_event(
                {
                    "event": "rl_fragment_push",
                    **asdict(submission),
                    "realized_h": submission.c_steps,
                }
            )

    async def _initialize(self, *, actor_model, rollout_manager) -> None:
        from .decoupled import DecoupledRlBridge

        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        initial = self._canonical_state(await actor_model.export_trainable_state())
        initial_adapter = getattr(self.args, "yeto_rl_initial_adapter", None)
        initial_adapter_sha256 = getattr(
            self.args,
            "yeto_rl_initial_adapter_sha256",
            None,
        )
        if (initial_adapter is None) != (initial_adapter_sha256 is None):
            raise ValueError(
                "decoupled RL initial adapter path and SHA256 must be set together"
            )
        parent_policy_hash = None
        if initial_adapter is not None:
            from .initial_adapter import load_initial_adapter

            initial = load_initial_adapter(
                initial_adapter,
                initial_adapter_sha256,
                model=self.args.yeto_rl_model,
                expected=initial,
            )
            parent_policy_hash = policy_tensor_hash(initial)
        checkpoint_error = None
        try:
            checkpoint = _load_decoupled_checkpoint(self.args)
        except RuntimeError as error:
            checkpoint = None
            checkpoint_error = error
        if checkpoint is not None:
            self.optimizer_steps = checkpoint["optimizer_steps"]
            self.action_tokens = checkpoint["action_tokens"]

        self.bridge = DecoupledRlBridge(
            initial,
            self.args.yeto_rl_bridge_config,
        )
        self.bridge.start()
        cut = self.bridge.wait_for_initial_cut(
            optimizer_steps=self.optimizer_steps,
            action_tokens=self.action_tokens,
        )
        versions = cut.fragment_versions
        if any(versions) and checkpoint is None:
            detail = "missing" if checkpoint_error is None else str(checkpoint_error)
            raise RuntimeError(
                f"nonzero decoupled RL cut has no valid island checkpoint: {detail}"
            )
        if (
            parent_policy_hash is not None
            and not any(versions)
            and policy_tensor_hash(cut.state) != parent_policy_hash
        ):
            raise RuntimeError(
                "decoupled RL initial adapter policy differs from version-zero cut"
            )
        if parent_policy_hash is not None and checkpoint is None:
            self._append_event(
                {
                    "event": "rl_initial_adapter",
                    "parent_adapter_sha256": initial_adapter_sha256,
                    "parent_policy_hash": parent_policy_hash,
                }
            )
        if checkpoint is not None and any(
            current < saved
            for current, saved in zip(
                versions,
                checkpoint["fragment_versions"],
            )
        ):
            raise RuntimeError("decoupled RL syncer cut predates island checkpoint")

        self.current = canonical_state(
            self.optimizer_steps,
            cut.state.tensors,
            base_model_revision=cut.state.base_model_revision,
            lora_config_hash=cut.state.lora_config_hash,
            layout_hash=cut.state.layout_hash,
            expected_specs=cut.state.specs,
        )
        self.current = await self._apply_decoupled_policy(
            self.current,
            reset_optimizer=True,
        )
        self.bridge.commit_initial_cut(
            cut,
            optimizer_steps=self.optimizer_steps,
            action_tokens=self.action_tokens,
        )
        self.snapshot = PolicySnapshot.create(
            self.optimizer_steps,
            self.current,
            versions,
        )
        self._save_progress(self.snapshot, stats=None)
        startup_manifest = getattr(self.bridge, "startup_final_manifest", None)
        if startup_manifest is not None:
            self.bridge.acknowledge_finalization(startup_manifest)
        await self._publish_snapshot(self.snapshot)
        self.args.start_rollout_id = self.snapshot.rollout_id
        if startup_manifest is not None:
            self._record_final_payload(self.bridge.final_payload_bytes_received)
            self.finished = True
            self.args.external_policy_sync_run_until_stop = False
            self.args.num_rollout = self.args.start_rollout_id

    async def _finish(
        self,
        *,
        policy_version: int,
        stats: LocalRoundStats,
    ) -> bool:
        manifest, final = self.bridge.wait_for_final_cut(
            policy_version=policy_version,
        )
        return await self._commit_final_cut(
            manifest,
            final,
            stats=stats,
            final_payload_bytes_received=self.bridge.final_payload_bytes_received,
        )

    async def _commit_final_cut(
        self,
        manifest,
        final: CanonicalLoraState,
        *,
        stats: LocalRoundStats,
        final_payload_bytes_received: int = 0,
    ) -> bool:
        self.current = await self._apply_decoupled_policy(
            final,
            reset_optimizer=False,
        )
        self.snapshot = PolicySnapshot.create(
            final.policy_version,
            self.current,
            manifest.versions,
        )
        self._save_progress(self.snapshot, stats=stats)
        self.bridge.acknowledge_finalization(manifest)
        await self._publish_snapshot(self.snapshot)
        if final_payload_bytes_received:
            self._record_final_payload(final_payload_bytes_received)
        self.finished = True
        return True

    def _record_final_payload(self, payload_bytes_received: int) -> None:
        self._append_event(
            {
                "event": "rl_final_cut",
                "sync/fragment_payload_bytes_received": payload_bytes_received,
            }
        )

    async def _after_local_train(
        self,
        *,
        rollout_id,
        actor_model,
        rollout_data,
    ) -> bool:
        if (
            actor_model is not self.actor_model
            or self.current is None
            or self.snapshot is None
            or rollout_id != self.snapshot.rollout_id
            or self.args.yeto_rl_policy_token != self.snapshot.token
        ):
            raise RuntimeError(
                "Miles called decoupled synchronization outside its policy snapshot"
            )
        next_rollout_id = rollout_id + 1
        exported = await actor_model.export_trainable_state()
        local = self._canonical_at_progress(exported, next_rollout_id)
        stats = self._round_stats(rollout_id, rollout_data, exported)
        self.optimizer_steps += 1
        self.action_tokens += stats.action_tokens
        if self.optimizer_steps != next_rollout_id:
            raise RuntimeError(
                "decoupled RL optimizer progress diverged from rollout ID"
            )
        if self.bridge.finalizing:
            self._record_local_round(stats, rollout_id=rollout_id)
            return await self._finish(
                policy_version=next_rollout_id,
                stats=stats,
            )
        if (
            getattr(self.args, "yeto_rl_learner_budget_steps", None)
            == self.optimizer_steps
        ):
            consolidation: BudgetConsolidation = self.bridge.consolidate_budget(
                local,
                optimizer_steps=self.optimizer_steps,
                action_tokens=self.action_tokens,
            )
            stats = replace(
                stats,
                delta_l2_norm=math.sqrt(
                    sum(
                        submission.delta_l2_norm**2
                        for submission in consolidation.submissions
                    )
                ),
            )
            self._record_local_round(
                stats,
                rollout_id=rollout_id,
                submissions=consolidation.submissions,
                additional_payload_bytes_received=consolidation.bytes_received,
            )
            return await self._commit_final_cut(
                consolidation.manifest,
                consolidation.state,
                stats=stats,
            )

        batch = self.bridge.drain_broadcasts(
            local,
            optimizer_steps=self.optimizer_steps,
            action_tokens=self.action_tokens,
        )
        current = batch.state
        if batch.fragment_ids:
            current = await self._apply_decoupled_policy(
                current,
                reset_optimizer=False,
            )
            self.bridge.commit_broadcasts(
                batch,
                optimizer_steps=self.optimizer_steps,
                action_tokens=self.action_tokens,
            )
        submissions = self.bridge.submit_ready(
            current,
            optimizer_steps=self.optimizer_steps,
            action_tokens=self.action_tokens,
        )
        stats = replace(
            stats,
            delta_l2_norm=math.sqrt(
                sum(submission.delta_l2_norm**2 for submission in submissions)
            ),
        )
        self._record_local_round(
            stats,
            rollout_id=rollout_id,
            batch=batch,
            submissions=submissions,
        )
        if self.bridge.finalizing or any(
            submission.global_step == self.args.yeto_rl_total_fragment_steps
            for submission in submissions
        ):
            return await self._finish(
                policy_version=next_rollout_id,
                stats=stats,
            )

        self.current = current
        self.snapshot = PolicySnapshot.create(
            next_rollout_id,
            current,
            self.bridge.fragment_versions,
        )
        self._save_progress(self.snapshot, stats=stats)
        await self._publish_snapshot(self.snapshot)
        return False

    async def after_local_train(
        self,
        *,
        rollout_id,
        actor_model,
        rollout_data,
    ) -> bool:
        hook_started = time.monotonic()
        try:
            should_stop = await self._after_local_train(
                rollout_id=rollout_id,
                actor_model=actor_model,
                rollout_data=rollout_data,
            )
        except StrictRlInvariantError as error:
            self._record_strict_failure(error)
            raise
        self._append_event(
            {
                "event": "rl_sync_hook",
                "rl/rollout_id": rollout_id,
                "sync/hook_seconds": time.monotonic() - hook_started,
                "sync/remote_quorum_wait_seconds": 0.0,
                "sync/finalization": should_stop,
            }
        )
        return should_stop

    async def finalize(self) -> None:
        if not self.finished:
            raise RuntimeError("Miles stopped before decoupled RL finalization")
        self.bridge.close()


def create_policy_sync(args) -> MilesPolicySync:
    if getattr(args, "yeto_rl_sync_preset", "strict-avg") == "decoupled":
        return DecoupledMilesPolicySync(args)
    return MilesPolicySync(args)
