"""RL launch-argument preparation (``_prepare_rl_args``) and the SAO
streaming / dense full-sweep contract adoption.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from ..gpu_spec import parse_gpu_spec
from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESET, SAO_STREAMING_SYNC_PRESET
from . import REPO_ROOT
from .rl_backend import (
    _rl_backend_image_ok, _rl_backend_module, _rl_callable, _rl_miles_function,
    resolve_default_rl_image,
)
from .topology import (
    rl_actor_gpus_per_node, rl_colocated_engine_check, rl_cross_node_switches, rl_island_args,
    rl_island_count, rl_island_placement_args, rl_island_spec, rl_min_nodes, rl_trainer_shape,
)
from .syncer import (
    ROLE_SYNCERS, island_scheduling_mode, rl_needs_critic, rl_syncer_profile, rl_trainable_roles,
)
from .storage import _rl_checkpoint_mount
from .rl_ports import _prepare_ports_algorithm


# Base port for Miles session servers when --session-server-port is unset
# (instance i listens on base + i); applied in _prepare_rl_args.
DEFAULT_SESSION_SERVER_PORT = 9301


def _default_session_server_port(args) -> None:
    """Give Miles session servers a fixed base port when none was set.

    Unset, Miles picks each session server port dynamically in the 20000+
    range, where the engine's other ports (NCCL, dist-init, ...) are also
    picked; S18 ARU-3 lost 1 of 4 launches to 20012 "address already in use"
    (which process took it is not known). A fixed base outside that range and
    below the Linux ephemeral range (32768+) avoids the shared range; Miles
    adds the instance index and starts each server on that static port.
    Miles checks each static port is free when it allocates it, but the server
    binds later; s19-compaction-g1-20261010f lost 31809 in that window. Under
    gVisor (Modal) the ephemeral range is 16000-65535 (gvisor
    pkg/tcpip/ports: firstEphemeral = 16000), so 31801+ was inside it. The base
    now sits below every shared range: gVisor/Linux ephemeral ports, Ray worker
    ports (10002-19999), Miles dynamic ports (20000+), and off the fixed
    6379/8000/8265/9000.
    An explicit --session-server-port is kept unchanged.
    """
    if args.use_session_server and args.session_server_port is None:
        args.session_server_port = [DEFAULT_SESSION_SERVER_PORT]


def _f32_shortest(value: float) -> float:
    """The shortest decimal of an f32 value, as the Rust syncer parses it back."""
    return float(f"{value:.7g}")


def _adopt_contract_settings(args, adopted: dict, *, source: str) -> None:
    """Set launch settings a frozen contract owns.

    A setting at its CLI default takes the contract value; any other value
    that differs from the contract is a conflict, not silently overridden.
    (A value typed equal to the CLI default cannot be told apart from the
    default and is treated as not given.)"""
    import argparse

    from ..cli import _add_launch_args

    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    conflicts = []
    for dest, value in adopted.items():
        given = getattr(args, dest)
        if given != parser.get_default(dest) and given != value:
            conflicts.append(f"--{dest.replace('_', '-')} {given} (contract: {value})")
    if conflicts:
        raise ValueError(f"settings conflict with the {source}: " + "; ".join(conflicts))
    for dest, value in adopted.items():
        setattr(args, dest, value)


def _prepare_sao_streaming(args, *, num_islands: int) -> None:
    """sao-streaming-full (P0-1.7b.3): actor and critic full-parameter streams.

    The contract bundle freezes the syncer semantic profile; the launcher
    adopts it, starts one syncer per role with exactly that profile and checks
    that its own rendering hashes to the frozen value."""
    from ..rl.sao_streaming_manifest import load_sao_streaming_manifest

    problems = []
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        problems.append("--rl-placement fixed-partition (dedicated rollout GPUs)")
    if not getattr(args, "rl_sao_contracts", None):
        problems.append("--rl-sao-contracts <final manifest>")
    if args.experimental_rl_sync:
        problems.append("no --experimental-rl-sync")
    if island_scheduling_mode(args) != "legacy":
        problems.append("--rl-island-scheduling legacy (no fragment frames in elastic mode)")
    if rl_trainable_roles(args) != ("actor", "critic"):
        problems.append("an algorithm with a critic (the SAO spec trains actor and critic)")
    if getattr(args, "rl_single_island_no_sync", False):
        problems.append("the role syncers (no --rl-single-island-no-sync)")
    if problems:
        raise ValueError("--rl-sync-preset sao-streaming-full requires " + "; ".join(problems))
    manifest = load_sao_streaming_manifest(args.rl_sao_contracts)
    profile = manifest.profile
    if profile.learners != num_islands:
        raise ValueError(f"the SAO contracts name {profile.learners} islands, the launch "
                         f"has {num_islands}")
    served = tuple((role, ROLE_SYNCERS[role].port) for role in rl_trainable_roles(args))
    if any(island.role_ports != served for island in manifest.islands):
        raise ValueError("the SAO contracts name role syncer ports the launch does not "
                         f"serve ({served})")
    horizons = {island.local_horizon for island in manifest.islands}
    if len(horizons) != 1:
        raise ValueError("the SAO contracts name different local horizons per island")
    adopted = {
        "quorum": profile.quorum,
        "grace_ms": profile.grace_ms,
        "grace_gamma": profile.grace_gamma,
        "grace_tau": profile.grace_tau,
        "pipeline": profile.pipeline,
        "sync_interval_steps": profile.sync_interval_steps,
        "delta_correction": profile.delta_correction,
        # The syncer takes the outer optimizer settings as f32.
        "outer_lr": _f32_shortest(profile.outer_lr),
        "outer_momentum": _f32_shortest(profile.outer_momentum),
        "fragments": manifest.expected_fragments,
        "local_rl_rounds_per_sync": horizons.pop(),
    }
    _adopt_contract_settings(args, adopted, source="SAO contracts")
    args.rl_sao_manifest = manifest
    args.rl_total_fragment_steps = profile.total_steps
    rendered = rl_syncer_profile(args, num_islands)
    if rendered.sha256 != profile.sha256:
        raise ValueError("the launcher's role syncers do not reproduce the frozen SAO "
                         "syncer profile")


def sao_contract_mounts(args, learner_id: int) -> dict[str, str]:
    """This island's runtime contract and layout attestation, at the paths the
    contract names."""
    island = args.rl_sao_manifest.island(learner_id)
    return {shipped.container_path: str(shipped.host_path)
            for shipped in (island.runtime, island.layout_attestation)}


def _prepare_dense_full_sweep(args, *, num_islands: int) -> None:
    """full-parameter preset (P0-D8): one actor role swept as whole-model fragments.

    The syncer's strict policy-sweep profile fixes its merge semantics
    (every island per round, no grace, serial rounds, plain averaging);
    the fragment count P comes from the Miles layout probe."""
    problems = []
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        problems.append("--rl-placement fixed-partition (dedicated rollout GPUs)")
    if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, "rl_parameter_layout_sha256", None) or ""):
        problems.append("--rl-parameter-layout-sha256 <64 lowercase hex>")
    if args.experimental_rl_sync:
        problems.append("no --experimental-rl-sync")
    if args.local_rl_rounds_per_sync != 1 or int(getattr(args, "rl_optimizer_steps", 1) or 1) != 1:
        problems.append("--local-rl-rounds-per-sync 1 and one optimizer step (strict sweep H=1)")
    if args.fragments < 1:
        problems.append("--fragments >= 1")
    if island_scheduling_mode(args) != "legacy":
        problems.append("--rl-island-scheduling legacy (no fragment frames in elastic mode)")
    if rl_needs_critic(args):
        problems.append("a critic-free algorithm (the strict sweep trains the actor only)")
    if problems:
        raise ValueError(f"--rl-sync-preset {FULL_PARAMETER_SYNC_PRESET} requires " + "; ".join(problems))
    args.quorum = num_islands
    args.grace_ms = 0
    args.pipeline = 1
    args.sync_interval_steps = 0.0
    args.delta_correction = "none"
    args.outer_lr = 1.0
    args.outer_momentum = 0.0
    args.merge_alpha = 0.0
    args.wire_dtype = "f32"
    args.rl_total_fragment_steps = args.total_steps * args.fragments


def _prepare_rl_args(
    args,
    *,
    allow_local_data: bool = False,
    allow_remote_model: bool = False,
) -> None:
    resolve_default_rl_image(args)
    if getattr(args, "training_mode", "sft") != "rl":
        if getattr(args, "rl_initial_adapter", None) is not None or getattr(
            args, "rl_initial_adapter_sha256", None
        ) is not None:
            raise ValueError("--rl-initial-adapter requires --training-mode rl")
        if getattr(args, "codex_reasoning_effort", None) is not None:
            raise ValueError("--codex-reasoning-effort requires --training-mode rl")
        return

    from ..adapter_lifecycle import directory_sha256, selected_parent

    if selected_parent(args)[1] is not None:
        raise ValueError(
            "--resume-from/--branch-from are not supported with --training-mode rl"
        )
    initial_adapter = getattr(args, "rl_initial_adapter", None)
    expected_initial_sha256 = getattr(args, "rl_initial_adapter_sha256", None)
    if initial_adapter is None and expected_initial_sha256 is not None:
        raise ValueError(
            "--rl-initial-adapter-sha256 requires --rl-initial-adapter"
        )
    if initial_adapter is not None:
        if args.rl_sync_preset != "decoupled":
            raise ValueError(
                "--rl-initial-adapter is only supported with "
                "--rl-sync-preset decoupled"
            )
        root = Path(initial_adapter).expanduser()
        if not root.is_dir():
            raise ValueError("--rl-initial-adapter must be a local directory")
        actual_initial_sha256 = directory_sha256(root)
        if (
            expected_initial_sha256 is not None
            and actual_initial_sha256 != expected_initial_sha256.lower()
        ):
            raise ValueError(
                "RL initial adapter SHA256 mismatch: expected "
                f"{expected_initial_sha256.lower()}, got {actual_initial_sha256}"
            )
        args.rl_initial_adapter = str(root.resolve())
        args.rl_initial_adapter_sha256 = actual_initial_sha256

    from ..models import resolve_model_kind

    from ..rl.engine.selection import require_ports_supported

    # Rejected before any cloud or GPU work (spec rl-engine-selection).
    for iargs in rl_island_placement_args(args):
        require_ports_supported(
            sync_preset=args.rl_sync_preset,
            model_kind=resolve_model_kind(args.model, args.model_kind),
            tuning=args.tuning,
            expert_full_count=getattr(args, "expert_full_count", 0) or 0,
            rollout_num_gpus=getattr(iargs, "rollout_num_gpus", None),
            placement=getattr(iargs, "rl_placement", "colocated"),
            island_scheduling=getattr(args, "rl_island_scheduling", "legacy"),
        )
    if resolve_model_kind(args.model, args.model_kind) != "causal-lm":
        raise ValueError("RL v0 supports only causal language models")
    if args.tuning != "lora":
        raise ValueError("RL v0 requires --tuning lora")
    if args.lora_r <= 0:
        raise ValueError("RL v0 requires a positive LoRA rank")
    if args.total_steps <= 0:
        raise ValueError("RL v0 requires --total-steps > 0")
    rollout_model = getattr(args, "rollout_model", None)
    rollout_revision = getattr(args, "rollout_model_revision", None)
    if rollout_model is None and rollout_revision is not None:
        raise ValueError("--rollout-model-revision requires --rollout-model")
    if rollout_model is not None and rollout_model != args.model:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(rollout_revision or "")):
            raise ValueError(
                "a distinct --rollout-model requires an immutable "
                "--rollout-model-revision"
            )
        args.rollout_model_revision = rollout_revision.lower()
    if args.rollout_batch_size <= 0 or args.n_samples_per_prompt <= 0:
        raise ValueError("RL v0 requires positive rollout batch and sample counts")
    if args.rollout_max_response_len <= 0:
        raise ValueError("RL v0 requires --rollout-max-response-len > 0")
    if args.cybergym_timeout <= 0:
        raise ValueError("RL v0 requires a positive CyberGym timeout")
    if args.rl_distributed_timeout_minutes <= 0:
        raise ValueError(
            "RL v0 requires --rl-distributed-timeout-minutes > 0"
        )
    if args.dynamic_sampling_max_replacements is not None and (
        args.dynamic_sampling_max_replacements < 0
    ):
        raise ValueError(
            "RL --dynamic-sampling-max-replacements must be non-negative"
        )
    if not args.inner_lr > 0:
        raise ValueError(f"RL requires --inner-lr > 0 (got {args.inner_lr})")
    if args.rl_sync_preset in ("strict-avg", FULL_PARAMETER_SYNC_PRESET):
        if args.local_rl_rounds_per_sync != 1:
            raise ValueError(f"--rl-sync-preset {args.rl_sync_preset} requires "
                             "--local-rl-rounds-per-sync 1 (strict sweep H=1)")
    elif args.rl_sync_preset == "decoupled" and args.local_rl_rounds_per_sync < 2:
        raise ValueError("decoupled RL requires at least 2 local RL rounds per sync")
    if args.seq_len < 2:
        raise ValueError("RL v0 requires --seq-len >= 2")
    args.seq_len = max(args.seq_len, args.rollout_max_response_len)
    # agentic-rollout-utilization 2.1: the policy-age limit is checked against
    # what the selected backend supports before any cloud or GPU work.
    from ..rl.engine.policy_age import validate_limit

    max_policy_age = validate_limit(getattr(args, "rl_max_policy_age", 0) or 0)
    policy_age_module = _rl_backend_module(args, "policy_age")
    policy_age_module.SUPPORT.check(max_policy_age)
    task_check = getattr(policy_age_module, "check_task", None)
    if callable(task_check):  # 4.1: stage 2 is single-turn only (agentic = stage 3)
        task_check(max_policy_age,
                   custom_generate=getattr(args, "custom_generate_function_path", None),
                   custom_agent=getattr(args, "custom_agent_function_path", None))
    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size
    elif args.over_sampling_batch_size < args.rollout_batch_size:
        raise ValueError(
            "RL --over-sampling-batch-size must be at least --rollout-batch-size"
        )
    # agentic-rollout-utilization 6.2: a backend without over-sample cut-off
    # refuses it before launch instead of silently ignoring it.
    over_sampling_check = getattr(
        _rl_backend_module(args, "run_config_rules"), "check_over_sampling", None)
    if callable(over_sampling_check):
        over_sampling_check(args.rollout_batch_size, args.over_sampling_batch_size)
    _rl_miles_function(args.custom_generate_function_path)
    custom_agent = getattr(args, "custom_agent_function_path", None)
    _rl_miles_function(custom_agent, "--custom-agent-function-path")
    from ..rl import (
        SECRLENV_AGENTS,
        SECRLENV_GENERATE,
        SECRLENV_GROUP_FILTER,
        SECRLENV_INFRASTRUCTURE_REPLACEMENTS,
        SECRLENV_REWARD,
        SECRLENV_ZERO_VARIANCE_REPLACEMENTS,
        SIGNED_CODEX_AGENTS,
    )

    if custom_agent is not None:
        if args.custom_generate_function_path is None:
            raise ValueError(
                "--custom-agent-function-path requires "
                "--custom-generate-function-path"
            )
        if not args.use_session_server:
            raise ValueError(
                "--custom-agent-function-path requires --use-session-server"
            )
        if args.tito_model is None:
            raise ValueError(
                "--custom-agent-function-path requires --tito-model"
            )
        if (
            args.apply_chat_template_kwargs is not None
            and custom_agent not in SIGNED_CODEX_AGENTS
        ):
            raise ValueError(
                "agentic session rollouts preserve raw messages and do not accept "
                "--apply-chat-template-kwargs"
            )

    codex_reasoning_effort = getattr(args, "codex_reasoning_effort", None)
    codex_backend_profile = getattr(args, "codex_backend_profile", None)
    if custom_agent in SIGNED_CODEX_AGENTS:
        from ..rl.codex_backend import (
            stock_codex_backend_profile,
            validate_stock_codex_fields,
        )

        codex_backend_profile = codex_backend_profile or str(args.tito_model)
        args.codex_backend_profile = codex_backend_profile
        codex_profile = stock_codex_backend_profile(codex_backend_profile)
        codex_chat_template_kwargs = codex_profile["chat_template_kwargs"]
        if args.apply_chat_template_kwargs is None:
            args.apply_chat_template_kwargs = codex_chat_template_kwargs
        validate_stock_codex_fields(
            tito_model=str(args.tito_model),
            codex_backend_profile=codex_backend_profile,
            rl_model_recipe=getattr(args, "rl_model_recipe", "generic"),
            model=str(args.model),
            model_revision=str(args.model_revision),
            rollout_model=getattr(args, "rollout_model", None),
            rollout_model_revision=getattr(args, "rollout_model_revision", None),
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            tito_allowed_append_roles=list(
                getattr(args, "tito_allowed_append_roles", None) or ()
            ),
            codex_reasoning_effort=codex_reasoning_effort,
            lora_targets=str(args.lora_targets),
            expert_full_count=int(getattr(args, "expert_full_count", 0)),
            lora_expert_rank=int(getattr(args, "rl_lora_expert_rank", None) or 0),
        )
    elif codex_reasoning_effort is not None or codex_backend_profile is not None:
        raise ValueError(
            "--codex-reasoning-effort/--codex-backend-profile requires a signed Codex agent"
        )
    if getattr(args, "codex_exclude_reasoning_from_loss", False):
        if custom_agent not in SIGNED_CODEX_AGENTS:
            raise ValueError(
                "--codex-exclude-reasoning-from-loss requires a signed Codex agent"
            )
        from ..rl.codex_backend import stock_codex_reasoning_markers

        if stock_codex_reasoning_markers(codex_backend_profile) is None:
            raise ValueError(
                f"Codex profile {codex_backend_profile!r} declares no reasoning "
                "markers; --codex-exclude-reasoning-from-loss is unavailable"
            )
    secrlenv_infrastructure_replacements = getattr(
        args, "secrlenv_max_infrastructure_replacements", None
    )
    if custom_agent in SECRLENV_AGENTS:
        if args.custom_generate_function_path != SECRLENV_GENERATE:
            raise ValueError(
                "the SecRLEnv agents require the signed SecRLEnv generate wrapper"
            )
        if args.reward_function != SECRLENV_REWARD:
            raise ValueError(
                "the SecRLEnv agents require the signed SecRLEnv reward function"
            )
        configured_filter = getattr(args, "dynamic_sampling_filter_path", None)
        if configured_filter not in {None, SECRLENV_GROUP_FILTER}:
            raise ValueError(
                "the SecRLEnv agents require the exact signed group filter"
            )
        zero_variance_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        if zero_variance_replacements not in {
            None,
            SECRLENV_ZERO_VARIANCE_REPLACEMENTS,
        } or isinstance(zero_variance_replacements, bool):
            raise ValueError(
                "the SecRLEnv agents require zero variance replacements"
            )
        if secrlenv_infrastructure_replacements not in {
            None,
            SECRLENV_INFRASTRUCTURE_REPLACEMENTS,
        } or isinstance(secrlenv_infrastructure_replacements, bool):
            raise ValueError(
                "the SecRLEnv agents require exactly one infrastructure replacement"
            )
        expected_oversampling = args.rollout_batch_size + 1
        if args.over_sampling_batch_size == args.rollout_batch_size:
            args.over_sampling_batch_size = expected_oversampling
        elif args.over_sampling_batch_size != expected_oversampling:
            # agentic-rollout-utilization 1.1 (10-09 ruling): the SecRLEnv +1
            # is a reserved same-task retry slot (_SecRLEnvRetryDataSource), not
            # over-sampling; over-sample/cut-off applies to the other agents.
            raise ValueError(
                "the SecRLEnv agents require oversampling equal to the training "
                "batch plus one (a reserved same-task retry slot; over-sampling "
                "with cut-off is not available to SecRLEnv agents)"
            )
        args.dynamic_sampling_filter_path = SECRLENV_GROUP_FILTER
        args.dynamic_sampling_max_replacements = (
            SECRLENV_ZERO_VARIANCE_REPLACEMENTS
        )
        args.secrlenv_max_infrastructure_replacements = (
            SECRLENV_INFRASTRUCTURE_REPLACEMENTS
        )
    elif secrlenv_infrastructure_replacements is not None:
        raise ValueError(
            "--secrlenv-max-infrastructure-replacements requires a signed "
            "SecRLEnv agent"
        )
    agent_max_seq_len = getattr(args, "agent_max_seq_len", None)
    if agent_max_seq_len is not None:
        if custom_agent is None:
            raise ValueError(
                "--agent-max-seq-len requires --custom-agent-function-path"
            )
        if agent_max_seq_len <= 0 or agent_max_seq_len > args.seq_len:
            raise ValueError(
                "--agent-max-seq-len must be positive and no greater than --seq-len"
            )
    dynamic_filter = getattr(args, "dynamic_sampling_filter_path", None)
    _rl_miles_function(
        dynamic_filter,
        "--dynamic-sampling-filter-path",
    )
    if (
        dynamic_filter is not None
        and args.over_sampling_batch_size <= args.rollout_batch_size
    ):
        raise ValueError(
            "variance-aware RL filtering requires --over-sampling-batch-size "
            "greater than --rollout-batch-size"
        )
    if args.dynamic_sampling_max_replacements is not None:
        if dynamic_filter is None:
            raise ValueError(
                "--dynamic-sampling-max-replacements requires "
                "--dynamic-sampling-filter-path"
            )
        stock_filter = (
            "miles.rollout.filter_hub.dynamic_sampling_filters."
            "check_reward_nonzero_std"
        )
        bounded_filter = "yeto.rl.filters.bounded_nonzero_reward_std"
        if dynamic_filter == stock_filter:
            args.dynamic_sampling_filter_path = bounded_filter
        elif dynamic_filter not in {
            bounded_filter,
            SECRLENV_GROUP_FILTER,
        }:
            raise ValueError(
                "--dynamic-sampling-max-replacements is only supported with "
                "an approved bounded nonzero-variance filter"
            )
    # After the agent contract binds its filter (SecRLEnv) and the over-sampling
    # count: the expected spec hash must be the one the island computes.
    _prepare_ports_algorithm(args)
    if not args.use_session_server and (
        args.session_server_ip is not None
        or args.session_server_port is not None
        or getattr(args, "session_server_workers", None) is not None
        or args.tito_model is not None
        or getattr(args, "codex_backend_profile", None) is not None
        or getattr(args, "tito_allowed_append_roles", None) is not None
    ):
        raise ValueError(
            "--session-server-ip/--session-server-port/--session-server-workers/"
            "--tito-model/--codex-backend-profile/"
            "--tito-allowed-append-roles requires --use-session-server"
        )
    roles = getattr(args, "tito_allowed_append_roles", None)
    if roles is not None:
        args.tito_allowed_append_roles = list(dict.fromkeys(roles))
    if args.session_server_port is not None and (
        len(args.session_server_port) not in {1, 2}
        or any(port <= 0 or port > 65535 for port in args.session_server_port)
        or (
            len(args.session_server_port) == 2
            and args.session_server_port[1] <= args.session_server_port[0]
        )
    ):
        raise ValueError(
            "--session-server-port requires one positive port or an increasing range"
        )
    if getattr(args, "session_server_workers", None) is not None and int(args.session_server_workers) < 1:
        raise ValueError("--session-server-workers must be at least 1")
    _default_session_server_port(args)

    per_island = [rl_island_args(args, m) for m in range(rl_island_count(args))]
    specs = [rl_island_spec(iargs, spec) for iargs, spec in zip(per_island, parse_gpu_spec(args.gpu))]
    if getattr(args, "external_learners", 0):
        raise ValueError("RL v0 does not support external learner slots")
    if args.tensor_parallel <= 0 or args.pipeline_parallel <= 0:
        raise ValueError("RL tensor and pipeline parallelism must be positive")
    model_parallel = args.tensor_parallel * args.pipeline_parallel
    if args.expert_parallel is not None:
        if args.expert_parallel <= 0:
            raise ValueError("RL expert parallelism must be positive")
        if any(spec.total_gpus % args.expert_parallel for spec in specs):
            raise ValueError("RL expert parallelism must divide every island")
    for iargs, spec in zip(per_island, specs):
        if getattr(iargs, "rl_placement", "colocated") == "fixed-partition":
            # rl-infra-spec 2.1: the node's GPUs are trainer + rollout + standby;
            # parallel-size checks below apply to the trainer part.
            if spec.num_nodes == 1 and rl_actor_gpus_per_node(iargs, spec) < 1:
                raise ValueError(
                    "--rl-placement fixed-partition needs one node with "
                    "--rollout-num-gpus + --rl-standby-gpus < GPUs per node"
                )
            if int(iargs.rollout_num_gpus) % args.rollout_num_gpus_per_engine:
                raise ValueError("--rollout-num-gpus must be a multiple of --rollout-num-gpus-per-engine")
            from ..gpu_spec import require_min_nodes

            require_min_nodes(spec, rl_min_nodes(iargs, spec))
            actor_nodes, actor = rl_trainer_shape(iargs, spec)
            node_parallel = args.tensor_parallel * int(getattr(args, "context_parallel", 1) or 1)
            cross_tp, cross_engine = rl_cross_node_switches(args)
            if spec.num_nodes > 1 and not cross_tp and (node_parallel > spec.gpus_per_node
                                                        or spec.gpus_per_node % node_parallel):
                raise ValueError("RL TP*CP must fit and divide one node (rl-multinode-island Q1/Q3: "
                                 "TP stays inside a node by default; EP/PP may span nodes; "
                                 "--rl-allow-cross-node-tp or cfg parallel.allow_cross_node_tp lifts it)")
            if spec.num_nodes > 1 and cross_tp:
                print("[launcher] WARNING: --rl-allow-cross-node-tp: the trainer TP*CP group "
                      f"({node_parallel}) may span nodes (NCCL over the inter-node fabric); "
                      "journaled as topology.layout.cross_node_tp=1")
            if spec.num_nodes > 1 and cross_engine:
                print("[launcher] WARNING: --rl-allow-cross-node-engine-tp: a rollout engine of "
                      f"{args.rollout_num_gpus_per_engine} GPUs may span whole nodes (sglang "
                      "nnodes>1; scales as one replica)")
            spec = dataclasses.replace(spec, num_nodes=actor_nodes, gpus_per_node=actor)
        elif spec.num_nodes > 1:
            # m5 (ruling 2026-10-04 v2): a colocated multi-node island carves its engines out
            # of the whole island; an engine larger than a node (sglang nnodes>1) needs the
            # explicit opt-in and must take whole nodes, before any cloud work.
            rl_colocated_engine_check(iargs, spec)
        if spec.total_gpus % model_parallel:
            raise ValueError(
                "RL TP*PP must divide every island GPU world size"
            )
        dp = spec.total_gpus // model_parallel
        if args.rollout_batch_size * args.n_samples_per_prompt % dp:
            raise ValueError(
                "RL rollout_batch_size*n_samples_per_prompt must be divisible "
                "by every island data-parallel size"
            )
        if (
            args.rollout_num_gpus_per_engine <= 0
            or spec.total_gpus % args.rollout_num_gpus_per_engine
        ):
            raise ValueError(
                "RL --rollout-num-gpus-per-engine must be positive and divide "
                "every island GPU world size"
            )
    for name in ("sglang_tp_size", "sglang_dp_size", "sglang_ep_size"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"RL --{name.replace('_', '-')} must be positive")
    if not 0.0 < args.sglang_mem_fraction_static < 1.0:
        raise ValueError("RL --sglang-mem-fraction-static must be between 0 and 1")
    if getattr(args, "rl_sampler_snapshot", False):
        # L8.2 (design D5): refuse before any GPU is provisioned; the island
        # translation re-checks against the placement it actually builds.
        from yeto.rl.adapters.miles import sampler_snapshot

        sampler_snapshot.check_launch(getattr(args, "rl_placement", "colocated"))
    if (
        getattr(args, "sglang_deterministic_inference", True)
        and getattr(args, "sglang_attention_backend", None) in {"dsv4", "compressed"}
    ):
        raise ValueError(
            "DeepSeek V4 dsv4/compressed attention requires "
            "--no-sglang-deterministic-inference"
        )
    expert_full_count = getattr(args, "expert_full_count", 0)
    selection_sha256 = getattr(args, "expert_selection_sha256", None)
    selection_contract_sha256 = getattr(
        args,
        "expert_selection_contract_sha256",
        None,
    )
    if expert_full_count < 0 or expert_full_count > 32:
        raise ValueError("--expert-full-count must be between 1 and 32 when enabled")
    if expert_full_count == 0:
        if selection_sha256 is not None or selection_contract_sha256 is not None:
            raise ValueError("expert selection hash requires --expert-full-count")
    else:
        if getattr(args, "rl_model_recipe", "generic") != "deepseek-v4-flash":
            raise ValueError(
                "--expert-full-count is only supported by "
                "--rl-model-recipe deepseek-v4-flash"
            )
        if args.lora_targets != "attention":
            raise ValueError("expert-full RL requires --lora-targets attention")
        if not 0 < args.expert_full_lr < float("inf"):
            raise ValueError("--expert-full-lr must be finite and positive")
        for flag, value in (
            ("--expert-selection-sha256", selection_sha256),
            (
                "--expert-selection-contract-sha256",
                selection_contract_sha256,
            ),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value or "")):
                raise ValueError(f"{flag} must be 64 lowercase hex characters")
    if getattr(args, "rl_model_recipe", "generic") == "deepseek-v4-flash":
        expected_lora_targets = (
            "attention" if expert_full_count else "attention-routed-experts"
        )
        if args.lora_targets != expected_lora_targets:
            raise ValueError(
                "expanded DeepSeek V4 Flash recipe requires "
                f"--lora-targets {expected_lora_targets}"
            )
        if (
            args.tensor_parallel != 8
            or args.expert_parallel != 8
            or args.rollout_num_gpus_per_engine != 8
            or args.sglang_tp_size != 8
            or args.sglang_ep_size != 8
            or args.sglang_dp_size not in (None, 1)
        ):
            raise ValueError(
                "expanded DeepSeek V4 Flash requires TP8/EP8 pipeline stages "
                "and per-node eight-GPU rollout replicas"
            )
        if args.sglang_attention_backend not in {"dsv4", "compressed"}:
            raise ValueError(
                "DeepSeek V4 Flash recipe requires --sglang-attention-backend dsv4"
            )
        if args.sglang_page_size != 256:
            raise ValueError(
                "DeepSeek V4 Flash recipe requires --sglang-page-size 256"
            )
    for name in (
        "sglang_page_size",
        "sglang_max_running_requests",
        "sglang_chunked_prefill_size",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"RL --{name.replace('_', '-')} must be positive")

    _rl_callable(args.reward_function, "--reward-function", required=True)
    if not re.fullmatch(
        r"docker:[^\s@]+@sha256:[0-9a-fA-F]{64}", args.rl_image or ""
    ) and not _rl_backend_image_ok(args):
        raise ValueError(
            "--rl-image must be docker:<repository>@sha256:<64 hex digest>"
        )
    if getattr(args, "learner_image", None) is not None:
        raise ValueError("RL uses digest-pinned --rl-image, not --learner-image")
    args.rl_total_fragment_steps = args.total_steps
    if args.rl_sync_preset != FULL_PARAMETER_SYNC_PRESET and getattr(args, "rl_parameter_layout_sha256", None):
        raise ValueError("--rl-parameter-layout-sha256 only applies to "
                         f"--rl-sync-preset {FULL_PARAMETER_SYNC_PRESET}")
    if args.rl_sync_preset != SAO_STREAMING_SYNC_PRESET and getattr(args, "rl_sao_contracts", None):
        raise ValueError("--rl-sao-contracts only applies to --rl-sync-preset sao-streaming-full")
    if args.rl_sync_preset == "decoupled":
        if args.experimental_rl_sync:
            raise ValueError("decoupled RL does not accept --experimental-rl-sync")
        if args.fragments < 2:
            raise ValueError("decoupled RL requires at least 2 fragments")
        if not 1 <= args.pipeline <= args.fragments:
            raise ValueError("decoupled RL pipeline must be between 1 and fragments")
        if args.fragment_pattern != "binpack":
            raise ValueError("decoupled RL requires --fragment-pattern binpack")
        args.quorum = len(specs)
        args.grace_ms = 0
        args.sync_interval_steps = float(args.local_rl_rounds_per_sync)
        args.delta_correction = "none"
        args.outer_lr = 0.7
        args.outer_momentum = 0.9
        args.merge_alpha = 0.0
        args.wire_dtype = "f32"
        args.rl_total_fragment_steps = args.total_steps * args.fragments
    elif args.rl_sync_preset == FULL_PARAMETER_SYNC_PRESET:
        _prepare_dense_full_sweep(args, num_islands=len(specs))
    elif args.rl_sync_preset == SAO_STREAMING_SYNC_PRESET:
        _prepare_sao_streaming(args, num_islands=len(specs))
    elif not args.experimental_rl_sync:
        args.fragments = 1
        args.quorum = len(specs)
        args.grace_ms = 0
        args.pipeline = 1
        args.sync_interval_steps = 0.0
        args.delta_correction = "none"
        args.outer_lr = 1.0
        args.outer_momentum = 0.0
        args.merge_alpha = 0.0
        args.wire_dtype = "f32"
    else:
        for name, expected, flag in (
            ("fragments", 1, "--fragments 1"),
            ("pipeline", 1, "--pipeline 1"),
            ("merge_alpha", 0.0, "--merge-alpha 0"),
            ("wire_dtype", "f32", "--wire-dtype f32"),
        ):
            if getattr(args, name) != expected:
                raise ValueError(
                    f"RL's one-fragment f32 bridge still requires {flag}"
                )
    if args.spot:
        _rl_checkpoint_mount(args.rl_completed_groups_path)

    provenance = getattr(args, "_provenance", None)
    if provenance:
        provenance["rl_engine"] = "ports"
    if provenance:
        for name in ("model", "dataset"):
            source = provenance.get(name) or {}
            if (
                name == "dataset"
                and allow_local_data
                and source.get("source") == "local"
                and source.get("resolved_revision") is None
            ):
                continue
            if (
                name == "model"
                and allow_remote_model
                and source.get("source") == "remote-local"
                and source.get("resolved_revision")
            ):
                continue
            if source.get("source") != "huggingface" or not source.get(
                "resolved_revision"
            ):
                raise ValueError(
                    f"RL v0 requires a revision-pinned Hugging Face {name}"
                )
        from ..provenance import python_spec_path, python_spec_sha256

        reward_path = python_spec_path(args.reward_function, base_dir=REPO_ROOT)
        try:
            reward_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                "RL reward source must be inside the synced Yeto workdir"
            ) from exc
        args.reward_sha256 = python_spec_sha256(
            args.reward_function, base_dir=REPO_ROOT
        )
    # The fixed Miles commit enables remote model code in its internal
    # Megatron and SGLang loaders. Keep that trust decision explicit through
    # Yeto's existing flag.
    if not args.trust_remote_code:
        raise ValueError("pinned Miles requires explicit --trust-remote-code")
