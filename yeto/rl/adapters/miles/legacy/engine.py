"""Adapter for the pinned Miles commit's Megatron/SGLang runtime."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import math
import statistics
import subprocess
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from yeto.rl import (
    MILES_LEGACY_PINS,
    MilesRevisionPins,
)
from yeto.rl.core import (
    STRICT_STREAM_CHUNK_BYTES,
    CanonicalLoraState,
    LocalRoundStats,
    PolicySnapshot,
    StrictRlInvariantError,
    canonical_state,
    canonical_state_from_validated_owned_tensors,
    policy_hash,
    policy_tensor_hash,
    require_nonzero_learning_rate,
)
from yeto.rl.decoupled import BudgetConsolidation
# Island progress formats moved to the engine core (decoupling 2.4); same objects re-exported.
from yeto.rl.engine.progress import (  # noqa: F401
    _DECOUPLED_CHECKPOINT_SCHEMA,
    _ISLAND_CHECKPOINT_SCHEMA,
    DecoupledProgress,
    _atomic_save_island_checkpoint,
    _decoupled_checkpoint_config,
    _island_checkpoint_config,
    _load_decoupled_checkpoint,
    _save_decoupled_checkpoint,
    record_strict_local_round,
)


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


def _overlay_tree_matches(root: Path, base_commit: str) -> bool:
    """A dirty checkout is accepted only when the island setup applied the critic
    overlay (record present, same patch sha256, same base) AND the whole worktree,
    untracked files included, hashes to the fork result tree (S13 forkg1 fix)."""

    import os
    import tempfile

    from yeto.rl.adapters.miles.overlay import (CRITIC_C357_BASE_COMMIT, CRITIC_C357_PATCH_SHA256,
                                CRITIC_C357_RESULT_TREE, read_applied_record)

    record = read_applied_record()
    if not record or record.get("patch_sha256") != CRITIC_C357_PATCH_SHA256 \
            or base_commit != CRITIC_C357_BASE_COMMIT:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        try:
            subprocess.run(["git", "-C", str(root), "read-tree", "HEAD"], env=env, check=True, capture_output=True)
            subprocess.run(["git", "-C", str(root), "add", "-A"], env=env, check=True, capture_output=True)
            tree = subprocess.run(["git", "-C", str(root), "write-tree"], env=env, check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return False
    return tree == CRITIC_C357_RESULT_TREE


def verify_miles_revision(
    root: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected: MilesRevisionPins = MILES_LEGACY_PINS,
) -> Path:
    """Verify either the pinned clean commit or an exact staged source tree.

    ``expected`` selects the constant group (legacy ``MILES_*`` or ports
    ``MILES_NEXT_*``); both engines share the same checks.
    """

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
    if commit != expected.commit:
        raise RuntimeError(
            f"Miles revision mismatch: expected {expected.commit}, got {commit}"
        )
    if origin.removesuffix(".git").rstrip("/") != expected.repository:
        raise RuntimeError(
            f"Miles origin mismatch: expected {expected.repository}, got {origin}"
        )
    if branch != "HEAD":
        raise RuntimeError("Miles checkout is not detached")
    if git("status", "--porcelain", "--untracked-files=all"):
        if not _overlay_tree_matches(root, expected.commit):
            raise RuntimeError("Miles checkout is not clean")

    miles = importlib.import_module("miles")
    package_path = Path(miles.__file__).resolve()
    if not package_path.is_relative_to(root):
        raise RuntimeError(f"imported Miles package {package_path} is outside {root}")
    return root


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


# Tape writer moved to the engine core (decoupling 2.3); same object re-exported.
from yeto.rl.engine.events import append_tape_event as _append_rl_event  # noqa: E402
# Published policy identity lives in the ports publisher (rl-code-quality-cleanup 1.4).
from yeto.rl.adapters.miles.publish import (  # noqa: E402
    _LOWER_HEX,
    _policy_token,
)
# generate_rollout and its queue/SecRLEnv helpers (rl-code-quality-cleanup 1.3).
from yeto.rl.adapters.miles.models.rollout_legacy_port import (  # noqa: E402
    _complete_group_for_policy,
    _policy_token_for_rollout,
    _record_strict_failure,
)


def _require_training_progress(stats: LocalRoundStats) -> None:
    """Fail a round that trained on real advantages yet produced no gradient.

    A round whose advantages are all zero (every GRPO group scored the same)
    legitimately has nothing to learn from, so a zero gradient norm is fine
    there.  A round with nonzero advantages and a gradient norm of exactly
    ``0.0`` means the gradients never reached the optimizer; committing its
    state would hand the syncer fp32 round-trip noise as if it were learning.
    """

    if stats.grad_norm is None or stats.grad_norm != 0.0:
        return
    if not stats.nonzero_advantage_count:
        return
    raise StrictRlInvariantError(
        "zero_grad_norm_with_nonzero_advantages",
        f"local round {stats.local_round_id} (global policy "
        f"{stats.base_policy_version}) reported grad_norm={stats.grad_norm!r} "
        f"although {stats.nonzero_advantage_count} of "
        f"{stats.completed_trajectories} trained samples had nonzero "
        "advantages; the round's local state was not submitted",
    )


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

    record_local_round = record_strict_local_round

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
        nonzero_advantage_count = self._nonzero_advantage_count(
            [value for batch in batches for value in batch.get("rewards", [])],
            expected_samples,
        )
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
        from yeto.rl.adapters.miles.models.deepseek_v4.expert_full_runtime import (
            round_stats_with_telemetry,
        )

        telemetry = getattr(train_state, "telemetry", None)
        mean_kl, ess_ratio, clip_fraction = round_stats_with_telemetry(
            telemetry, mean_kl, ess_ratio, clip_fraction
        )
        if train_seconds < 0:
            raise RuntimeError("Miles exported a negative train duration")
        metrics = self._rollout_metrics(rollout_id)
        group_size = self.args.n_samples_per_prompt
        applied_lrs = self._applied_lrs(rollout_id)
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
            nonzero_advantage_count=nonzero_advantage_count,
            applied_lr=None if applied_lrs is None else min(applied_lrs),
            applied_lrs=applied_lrs,
        )

    def _applied_lrs(self, rollout_id: int) -> tuple[float, ...] | None:
        """LRs the round's optimizer steps applied, from the before-train-step hook.

        ``yeto.rl.applied_lr.before_train_step`` records them in the actor
        (fork untouched); ``None`` only when the learner did not arm the hook.
        """

        from yeto.rl.applied_lr import APPLIED_LR_DIR_ATTR, collect

        directory = getattr(self.args, APPLIED_LR_DIR_ATTR, None)
        if not directory:
            return None
        return collect(directory, rollout_id, int(self.args.num_steps_per_rollout))

    def _nonzero_advantage_count(
        self, advantages: list[Any], expected_samples: int
    ) -> int | None:
        # Miles' GRPO/GSPO path trains on the group-normalized ``rewards`` list
        # verbatim as each sample's advantage (``get_grpo_returns``), and that
        # list is split across the DP shards like ``response_lengths``.  Other
        # estimators derive advantages later inside the actor, where the hook
        # cannot see them; report "unknown" rather than guess.
        if getattr(self.args, "advantage_estimator", None) not in ("grpo", "gspo"):
            return None
        if len(advantages) != expected_samples:
            raise RuntimeError("Miles DP rollout shards lack per-sample advantages")
        try:
            values = [float(value) for value in advantages]
        except (TypeError, ValueError) as error:
            raise RuntimeError("Miles rollout advantages are not scalars") from error
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError("Miles rollout advantages are not finite")
        return sum(1 for value in values if value != 0.0)

    async def _initialize(self, *, actor_model, rollout_manager) -> None:
        from yeto.rl.bridge import StrictRlBridge

        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        initial = self._canonical_state(await actor_model.export_trainable_state())
        runtime = _BridgeRuntime(initial, self.args)
        if getattr(self.args, "yeto_rl_island_scheduling", "legacy") == "elastic":
            raise RuntimeError("--rl-island-scheduling elastic runs on --rl-engine ports "
                               "(ElasticAvgSync); the legacy Miles bridge path is strict only")
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
                _require_training_progress(stats)
                require_nonzero_learning_rate(
                    stats, final_round=stats.local_round_id >= self.args.num_rollout
                )
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
            _require_training_progress(stats)
            require_nonzero_learning_rate(
                stats, final_round=stats.local_round_id >= self.args.num_rollout
            )
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

    _save_progress = DecoupledProgress._save_progress

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

    _record_local_round = DecoupledProgress._record_local_round

    async def _initialize(self, *, actor_model, rollout_manager) -> None:
        from yeto.rl.decoupled import DecoupledRlBridge

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
            from yeto.rl.initial_adapter import load_initial_adapter

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

    _record_final_payload = DecoupledProgress._record_final_payload

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
        _require_training_progress(stats)
        # Decoupled: the round is final once the syncer's final cut is known
        # (finalizing) or it exhausts the optional step budget (design D4).
        require_nonzero_learning_rate(
            stats,
            final_round=bool(self.bridge.finalizing)
            or getattr(self.args, "yeto_rl_learner_budget_steps", None)
            == self.optimizer_steps + 1,
        )
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
