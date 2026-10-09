"""Island progress files in the legacy formats (yeto-framework-decoupling 2.4). Import-light.

Strict progress (schema 3) and decoupled progress (schema 4), moved unchanged
from ``yeto.rl.adapters.miles.legacy.engine`` (which re-exports the same objects) so the engine
core's bridges no longer import the legacy Miles engine. Neither file ever
contains LoRA tensors or optimizer state. Serialization is byte-for-byte the
old code (``torch.save`` of the same dicts in the same key order).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from yeto.rl.core import LocalRoundStats, PolicySnapshot, parse_policy_snapshot_token

if TYPE_CHECKING:
    from yeto.rl.decoupled import BroadcastBatch, FragmentSubmission

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



def record_strict_local_round(self, stats: LocalRoundStats) -> None:
    """Strict schema-3 round commit (legacy ``_BridgeRuntime.record_local_round``); needs only ``self.args``."""
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


class DecoupledProgress:
    """Decoupled progress/event helpers shared by the legacy Miles sync and the ports bridge.

    Needs ``self.args``, ``self.snapshot``, ``self.current``, ``self.optimizer_steps``,
    ``self.action_tokens`` and ``self._append_event``.
    """

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

    def _record_final_payload(self, payload_bytes_received: int) -> None:
        self._append_event(
            {
                "event": "rl_final_cut",
                "sync/fragment_payload_bytes_received": payload_bytes_received,
            }
        )
