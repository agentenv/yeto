"""Bind a checked SAO streaming runtime contract into a ports Miles island.

``yeto.rl.sao_streaming_runtime`` loads and checks the engine-neutral
contract.  This module turns its two role contracts into the one two-role
:class:`FullParameterSyncConfig` (actor and critic, P0-D8), checks the parsed
Miles namespace against the contract and creates the private evidence
directory.  The island's ``IslandDriver`` owns the loop, rollout and
publication, so nothing here installs a Miles training-loop callback.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

from yeto.rl.sao_streaming_runtime import SaoRoleContract, SaoStreamingRuntime

from .role_stream import RoleStreamConfig
from .session import CONFIG_ATTRIBUTE, FullParameterSyncConfig

SAO_POLICY_OBJECTIVE = "sao_dis"
_UNSAFE_DEBUG_FLAGS = (
    "debug_skip_weight_update",
    "debug_disable_optimizer",
    "debug_train_only",
    "debug_rollout_only",
)
_UNSAFE_DEBUG_VALUES = (
    "load_debug_rollout_data",
    "ci_inject_rollout_data_path",
    "ci_ft_test_actions",
    "debug_exit_after_rollout",
)


def _role_stream(
    contract: SaoRoleContract, runtime: SaoStreamingRuntime, algorithm_spec_sha256: str
) -> RoleStreamConfig:
    return RoleStreamConfig(
        algorithm="sao",
        role=contract.role,
        component=contract.component,
        syncer_addr=contract.syncer_addr,
        learner_id=contract.learner_id,
        learner_generation=contract.learner_generation,
        learner_generations=contract.learner_generations,
        total_fragment_steps=contract.total_fragment_steps,
        expected_fragments=contract.expected_fragments,
        expected_layout_hash=contract.expected_layout_hash,
        local_horizon=contract.local_horizon,
        optimizer_steps_per_round=contract.optimizer_steps_per_round,
        training_contract_hash=contract.training_contract_hash,
        syncer_profile_hash=runtime.syncer_profile.sha256,
        pipeline_depth=runtime.syncer_profile.pipeline,
        algorithm_spec_sha256=algorithm_spec_sha256,
        wan_streams=contract.wan_streams,
        wait_timeout=contract.wait_timeout,
        poll_seconds=contract.poll_seconds,
        max_fragment_bytes=contract.max_fragment_bytes,
        max_chunk_bytes=contract.max_chunk_bytes,
    )


def sao_sync_config(
    runtime: SaoStreamingRuntime, *, algorithm_spec_sha256: str
) -> FullParameterSyncConfig:
    """The two-role (actor, critic) streaming sync the contract describes.

    Both role sessions bind the island's algorithm spec hash (P0-4.8).
    """

    return FullParameterSyncConfig(
        (
            _role_stream(runtime.actor, runtime, algorithm_spec_sha256),
            _role_stream(runtime.critic, runtime, algorithm_spec_sha256),
        )
    )


def validate_miles_args(args: Any, runtime: SaoStreamingRuntime) -> None:
    """Check the parsed Miles namespace against the SAO runtime contract."""

    actor = runtime.actor
    critic = runtime.critic
    enabled = [name for name in _UNSAFE_DEBUG_FLAGS if bool(getattr(args, name, False))]
    enabled += [
        name for name in _UNSAFE_DEBUG_VALUES if getattr(args, name, None) is not None
    ]
    if enabled:
        raise ValueError(
            "SAO streaming rejects state-bypassing debug options: " + ", ".join(enabled)
        )
    if getattr(args, "policy_objective", None) != SAO_POLICY_OBJECTIVE:
        raise ValueError(
            "SAO streaming requires the SAO algorithm spec "
            "(loss.policy_objective='sao_dis')"
        )
    if not getattr(args, "use_critic", False):
        raise ValueError("SAO streaming requires the Miles critic")
    if int(getattr(args, "num_critic_only_steps", 0)) != 0:
        raise ValueError(
            "SAO streaming requires actor and critic lockstep from rollout zero"
        )
    if int(getattr(args, "start_rollout_id", 0)) != 0:
        raise ValueError("SAO streaming currently requires a version-zero start")
    if int(getattr(args, "lora_rank", 0)) > 0:
        raise ValueError("SAO streaming requires full-parameter training")
    if getattr(args, "yeto_rl_learner_id", None) != actor.learner_id:
        raise ValueError("SAO streaming learner ID differs from the SecRLEnv context")
    if (
        getattr(args, "yeto_rl_base_model_revision", None)
        != actor.component.model_revision
    ):
        raise ValueError("SAO actor revision differs from the SecRLEnv base model")

    actor_steps = getattr(args, "num_steps_per_rollout", None)
    critic_epochs = getattr(args, "num_critic_epochs", None)
    if (
        type(actor_steps) is not int
        or actor_steps < 1
        or type(critic_epochs) is not int
        or critic_epochs < 1
        or actor.optimizer_steps_per_round != actor_steps
        or critic.optimizer_steps_per_round != actor_steps * critic_epochs
    ):
        raise ValueError("SAO role optimizer accounting differs from the Miles recipe")

    evidence = runtime.trajectory_evidence_dir
    parent = evidence.parent
    if (
        evidence.exists()
        or evidence.is_symlink()
        or not parent.is_dir()
        or parent.is_symlink()
    ):
        raise ValueError(
            "SAO trajectory evidence path must be fresh with a real parent"
        )


def bind_sao_streaming_runtime(
    args: Any, runtime: SaoStreamingRuntime, *, algorithm_spec_sha256: str
) -> Path:
    """Install the two-role sync config and create its private evidence dir.

    ``args`` is the ports island's parsed Miles namespace; the island sets
    ``yeto_rl_sync_preset`` itself.  The ports rollout hook writes each
    round's trajectory evidence into the directory; the full-parameter sync
    reads it back at the round boundary.
    """

    validate_miles_args(args, runtime)
    sync = sao_sync_config(runtime, algorithm_spec_sha256=algorithm_spec_sha256)
    evidence = runtime.trajectory_evidence_dir
    try:
        evidence.mkdir(mode=0o700)
        evidence.chmod(0o700)
    except OSError as error:
        raise ValueError(
            "failed to create private SAO trajectory evidence directory"
        ) from error
    if stat.S_IMODE(evidence.stat().st_mode) != 0o700:
        raise RuntimeError("SAO trajectory evidence directory is not private")

    actor = runtime.actor
    setattr(args, CONFIG_ATTRIBUTE, sync)
    args.yeto_rl_trajectory_evidence_dir = str(evidence)
    args.yeto_rl_trajectory_evidence_kind = runtime.trajectory_evidence_kind
    args.yeto_rl_trajectory_evidence_schema_version = (
        runtime.trajectory_evidence_schema_version
    )
    args.yeto_rl_learner_generation = actor.learner_generation
    args.yeto_rl_learner_generations = actor.learner_generations
    args.yeto_rl_num_fragments = actor.expected_fragments
    args.yeto_rl_total_fragment_steps = actor.total_fragment_steps
    return evidence
