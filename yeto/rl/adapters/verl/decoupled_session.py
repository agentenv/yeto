"""Decoupled DiLoCo (``--rl-sync-preset decoupled``, LoRA fragments) on a verl island.

The island runs the engine-neutral :class:`yeto.rl.engine.bridges.DecoupledSync`: the
same safe-boundary state machine, fragment bridge (:mod:`yeto.rl.decoupled`), schema-4
island progress file and policy snapshot token (``yeto:<rollout_id>:<policy_tensor_hash>``)
as a Miles island.  What verl already does is used as is:

* the trainable LoRA is verl FSDP2's PEFT model (``actor_rollout_ref.model.lora_rank``;
  fork ``verl/workers/engine/fsdp/transformer_impl.py:FSDPEngine._build_lora_module``,
  ``peft.get_peft_model``);
* the canonical tensors are read and written by :class:`.ports_impl.VerlPolicyState`
  (``full_tensor`` / ``distribute_tensor`` on the PEFT DTensors, canonical names from
  :mod:`.param_names`), the same port the strict-avg island uses;
* the publication is verl's colocated ``checkpoint_manager.update_weights`` (a LoRA
  update goes to vLLM as a ``TensorLoRARequest``) plus the yeto read-back
  (:class:`.ports_impl.VerlPublisher`).

This module adds only what verl has no notion of: the island plan section (refusing
the shapes the session does not cover yet) and the configuration ``DecoupledSync``
reads, built from the verl plan instead of Miles' parsed arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

PRESET = "decoupled"
PLAN_KEY = "decoupled"


def requested(args: Any) -> bool:
    """The island was launched with ``--sync-preset decoupled``."""
    return getattr(args, "sync_preset", None) == PRESET


def island_problems(args: Any, *, megatron: bool, buffer_trainer: bool, max_policy_age: int,
                    rollout_ahead: int, separate: bool, resume: Mapping[str, Any] | None) -> list[str]:
    """Why this verl island cannot run the decoupled session ([] when it can).

    Covered: the synchronous colocated FSDP2 LoRA island that generates and trains.
    The launcher refuses the same shapes first (``launch_flags.decoupled_problems``)."""
    problems = []
    if getattr(args, "rl_single_island_no_sync", False):
        problems.append("a syncer (decoupled is an outer sync; drop --rl-single-island-no-sync)")
    if getattr(args, "parameter_mode", "lora") != "lora":
        problems.append("--parameter-mode lora (full parameters: --rl-sync-preset full-parameter "
                        "--rl-sync-full-parameter-rounds async, the multi-fragment counterpart)")
    if megatron:
        problems.append("the FSDP2 engine (TP=PP=EP=CP=1; the Megatron island trains full parameters)")
    if buffer_trainer:
        problems.append("no sampler / buffer-trainer split")
    if max_policy_age or rollout_ahead:
        problems.append("--rl-max-policy-age 0 without --rl-rollout-ahead (the synchronous trainer)")
    if separate:
        problems.append("--rl-placement colocated")
    if resume is not None:
        problems.append("no round or island cuts (the schema-4 progress file is the restart record)")
    if getattr(args, "bcast_delta", False):
        problems.append("no --bcast-delta (decoupled broadcasts f32 fragments)")
    fragments, pipeline = int(args.fragments), int(args.pipeline)
    if fragments < 2:
        problems.append("--fragments >= 2")
    if not 1 <= pipeline <= max(fragments, 1):
        problems.append("--pipeline between 1 and --fragments")
    if int(args.local_horizon) < 2:
        problems.append("--local-horizon >= 2")
    if args.total_fragment_steps != args.global_rounds * fragments:
        problems.append("--total-fragment-steps = --global-rounds x --fragments")
    for flag, value in (("--completed-groups-path", args.completed_groups_path),
                        ("--source-sha256", args.source_sha256),
                        ("--reward-sha256", args.reward_sha256)):
        if not value:
            problems.append(flag)
    if (args.initial_adapter is None) != (args.initial_adapter_sha256 is None):
        problems.append("--initial-adapter and --initial-adapter-sha256 together")
    return problems


def plan_section(args: Any, *, trainer_gpus: int) -> dict[str, Any]:
    """``plan["decoupled"]``: what :func:`build_sync` needs on the trainer side."""
    pool = None
    if getattr(args, "rl_island_pool", False):
        from yeto.rl.pool_client import DEFAULT_ISLAND_LEASE_S

        lease = getattr(args, "rl_sync_island_lease_s", None)
        pool = {"lease_s": DEFAULT_ISLAND_LEASE_S if lease is None else float(lease),
                "syncer_epoch": int(getattr(args, "rl_syncer_epoch", 0) or 0)}
    return {
        "fragments": int(args.fragments), "pipeline": int(args.pipeline),
        "local_horizon": int(args.local_horizon), "total_fragment_steps": int(args.total_fragment_steps),
        "matrix_merge": getattr(args, "rl_sync_matrix_merge", "avg"),
        "completed_groups_path": str(args.completed_groups_path),
        "source_sha256": args.source_sha256, "reward_sha256": args.reward_sha256,
        "model": args.model, "data": args.data, "data_revision": args.data_revision,
        "initial_adapter": args.initial_adapter, "initial_adapter_sha256": args.initial_adapter_sha256,
        "inner_lr": float(args.inner_lr), "seq_len": int(args.seq_len), "seed": int(args.seed),
        "rollout_max_response_len": int(args.rollout_max_response_len),
        "actor_num_nodes": int(getattr(args, "actor_num_nodes", 1) or 1),
        "actor_num_gpus_per_node": int(trainer_gpus), "island_pool": pool,
    }


def sync_layout_fingerprint(specs: Sequence[Any], fragments: int, matrix_merge: str) -> str:
    """The fragment layout fingerprint the syncer's session contract is built from
    (the Miles island's ``yeto_rl_sync_layout_fingerprint``)."""
    from yeto.protocol import layout_fingerprint
    from yeto.rl.core import build_rl_fragment_layout

    return layout_fingerprint(build_rl_fragment_layout(specs, fragments, matrix_merge=matrix_merge)).hex()


def session_args(plan: Mapping[str, Any], specs: Sequence[Any], *, lora_hash: str, layout_hash: str,
                 identity_sha256: str | None, compat_group: str | None) -> SimpleNamespace:
    """The configuration :class:`~yeto.rl.engine.bridges.DecoupledSync` and its schema-4
    progress file (:mod:`yeto.rl.engine.progress`) read, from the verl plan.

    The progress file's ``config`` block is compared on restart, so every value is a
    deterministic function of the launch.  Miles-only knobs have their verl constants:
    one optimizer step per round, no pipeline or expert parallelism, GRPO, no
    oversampling, no custom generation."""
    from yeto.rl.decoupled import DecoupledBridgeConfig

    section = plan[PLAN_KEY]
    fragments = int(section["fragments"])
    host, _, port = str(plan["syncer"]).rpartition(":")
    pool = section.get("island_pool") or {}
    bridge = DecoupledBridgeConfig(
        syncer_addr=(host, int(port)), learner_id=int(plan["learner_id"]),
        total_fragment_steps=int(section["total_fragment_steps"]), num_fragments=fragments,
        pipeline=int(section["pipeline"]), local_horizon=int(section["local_horizon"]),
        expected_specs=tuple(specs), base_model_revision=plan["model_revision"], lora_config_hash=lora_hash,
        canonical_layout_hash=layout_hash, wan_streams=int(plan.get("wan_streams", 4)),
        backend_identity_sha256=identity_sha256, compat_group=compat_group,
        matrix_merge=section["matrix_merge"],
        island_lease_s=pool.get("lease_s"), syncer_epoch=pool.get("syncer_epoch"))
    groups = int(plan["groups_per_round"])
    return SimpleNamespace(
        # DecoupledSync
        yeto_rl_bridge_config=bridge,
        yeto_rl_completed_groups_path=section["completed_groups_path"],
        yeto_rl_policy_token=None,
        yeto_rl_total_fragment_steps=int(section["total_fragment_steps"]),
        yeto_rl_learner_budget_steps=None,
        yeto_rl_sync_layout_fingerprint=sync_layout_fingerprint(specs, fragments, section["matrix_merge"]),
        yeto_rl_initial_adapter=section.get("initial_adapter"),
        yeto_rl_initial_adapter_sha256=section.get("initial_adapter_sha256"),
        yeto_rl_model=section["model"],
        # schema-4 progress config (progress._decoupled_checkpoint_config)
        actor_num_gpus_per_node=int(section["actor_num_gpus_per_node"]),
        actor_num_nodes=int(section["actor_num_nodes"]),
        pipeline_model_parallel_size=1, expert_model_parallel_size=1,
        advantage_estimator="grpo",
        yeto_rl_data=section["data"], yeto_rl_data_revision=section.get("data_revision"),
        yeto_rl_base_model_revision=plan["model_revision"],
        seq_length=int(section["seq_len"]), seed=int(section["seed"]),
        yeto_rl_layout_hash=layout_hash, yeto_rl_lora_config_hash=lora_hash,
        lr=float(section["inner_lr"]),
        n_samples_per_prompt=int(plan["samples_per_group"]), num_steps_per_rollout=1,
        rollout_batch_size=groups, over_sampling_batch_size=groups,
        rollout_max_response_len=int(section["rollout_max_response_len"]),
        yeto_rl_reward_sha256=section["reward_sha256"],
        custom_generate_function_path=None, use_session_server=False, tito_model=None,
        yeto_rl_learner_id=int(plan["learner_id"]), yeto_rl_source_sha256=section["source_sha256"],
        yeto_rl_num_fragments=fragments, yeto_rl_pipeline=int(section["pipeline"]),
        yeto_rl_local_horizon=int(section["local_horizon"]),
        yeto_rl_total_sweeps=int(plan["global_rounds"]),
    )


def build_sync(plan: Mapping[str, Any], specs: Sequence[Any], *, lora_hash: str, layout_hash: str,
               identity_sha256: str | None, compat_group: str | None):
    """The ``DecoupledSync`` of a decoupled verl island."""
    from yeto.rl.engine.bridges import DecoupledSync

    return DecoupledSync(session_args(plan, specs, lora_hash=lora_hash, layout_hash=layout_hash,
                                      identity_sha256=identity_sha256, compat_group=compat_group))


def progress_for(sync: Any):
    """The driver's progress store of a decoupled session (the Miles island's
    ``DecoupledIslandProgress``), None for any other sync."""
    from yeto.rl.engine.bridges import DecoupledIslandProgress, DecoupledSync

    return DecoupledIslandProgress(sync.args) if isinstance(sync, DecoupledSync) else None
