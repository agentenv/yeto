"""The SkyPilot task of one RL (Miles/ports) island."""

from __future__ import annotations

import json
import os
import shlex
import sys
import time

from ..gpu_spec import ClusterSpec
from ..hw.catalog import COMPAT_GROUP_ENV, compat_group
from .code_pack import code_pack_island_flag, code_workdir
from .peer_recovery import island_flags as peer_recovery_island_flags
from .rl_backend import _miles_overlay, _rl_backend_module
from .remote_setup import (
    HF_TOKEN_ENV, HF_TOKEN_PATH, PORTS_MEGATRON_PATH, WAN_TUNING, island_pre_run_block,
    model_prefetch,
)
from .topology import (
    multinode_env_prelude, multinode_network_tier, rl_cross_node_flags, rl_island_bundle_map_flag,
    rl_island_cpus, rl_island_gpu_env, rl_island_ray_gpus, rl_island_ray_spill_dir_prelude,
    rl_island_ray_storage, rl_island_spec, rl_trainer_shape,
)
from .syncer import (
    critic_syncer_address, elastic_debug_delay_env, elastic_debug_pause_env, island_hmac_secret,
    rl_dense_sweep, rl_full_parameter_async, rl_full_parameter_final_cut, rl_needs_critic,
    rl_sao_streaming, rl_syncer_profile,
)
from .credentials import registry_login_for, split_secret_envs
from .storage import (
    ELASTIC_CHECKPOINT_STORE_MOUNT, NODE0_LOCAL_CHECKPOINT_WARNING, RL_INITIAL_ADAPTER_PATH,
    _rl_checkpoint_mount, _rl_checkpoint_storage_name, modal_volume_store_env, model_store_env,
    model_store_filesystem, nebius_baked_image_id, rl_checkpoint_store_plan, store_object_envs,
)
from .island_tasks import island_task_data
from .rl_ports import _ports_algorithm_flags
from .rl_args import sao_contract_mounts
from .images import _miles_source_setup
from ..rl.engine.start_phases import SUBMITTED_ENV, shell_mark, shell_reset
from .in_vm_docker import (
    DOCKER_LOGIN_UNSET, IN_VM_DOCKER_CLOUDS, in_vm_docker_run, in_vm_docker_setup,
)
from .harness_env import check_compaction_launch


def submitted_unix() -> float:
    """Phase ``submitted`` of the island being built (a seam for byte-equal task tests)."""

    return time.time()


def _wan_transport_flags(args) -> str:
    """S26 E3b island transport options; emitted only when not the default."""
    flags = ""
    schedule = getattr(args, "wan_stream_schedule", "rr")
    if schedule != "rr":
        flags += f" --wan-stream-schedule {schedule}"
    stall = float(getattr(args, "wan_stall_timeout", 0.0) or 0.0)
    if stall < 0:
        raise ValueError("--wan-stall-timeout must be >= 0 (0 = off)")
    if stall > 0:
        flags += f" --wan-stall-timeout {stall!r}"
    min_up = float(getattr(args, "wan_probe_min_mbps", 0.0) or 0.0)
    min_down = float(getattr(args, "wan_probe_min_down_mbps", 0.0) or 0.0)
    if min_up < 0 or min_down < 0:
        raise ValueError("--wan-probe-min-mbps / --wan-probe-min-down-mbps must be >= 0 (0 = off)")
    if min_up > 0 or min_down > 0:
        if getattr(args, "rl_elastic", False):
            # Only the legacy syncer serves MSG_BW_PROBE; on the elastic server
            # every probe would fail and every island would be rejected.
            raise ValueError("--wan-probe-min-mbps / --wan-probe-min-down-mbps need the "
                             "legacy syncer; the elastic syncer has no link probe")
        mib = int(getattr(args, "wan_probe_mib", 256))
        if not 1 <= mib <= 1024:
            raise ValueError("--wan-probe-mib must be between 1 and 1024")
        flags += f" --wan-probe-min-mbps {min_up!r} --wan-probe-min-down-mbps {min_down!r} --wan-probe-mib {mib}"
    return flags


def _rl_matrix_merge_flag(args) -> str:
    """S27 P1: decoupled islands build their fragment layout with the syncer's
    merge modes; omitted at the default so the island command is unchanged."""
    merge = getattr(args, "rl_sync_matrix_merge", "avg")
    return "" if merge == "avg" else f" --rl-sync-matrix-merge {merge}"


def _rl_layout_flags(args) -> str:
    """The shape-free layout flags; "" with the default shape-bound layout (argv unchanged)."""

    if getattr(args, "rl_sync_layout", "shape-bound") != "shape-free":
        return ""
    flags = " --rl-sync-layout shape-free"
    for flag, dest, default in (
        ("--rl-sync-shape-free-tp-units", "rl_sync_shape_free_tp_units", 1),
        ("--rl-sync-shape-free-expert-classes", "rl_sync_shape_free_expert_classes", 8),
        ("--rl-sync-shape-free-dense-groups", "rl_sync_shape_free_dense_groups", 8),
    ):
        value = int(getattr(args, dest, default))
        if value != default:
            flags += f" {flag} {value}"
    if getattr(args, "rl_sync_shape_free_split_layers", False):
        flags += " --rl-sync-shape-free-split-layers"
    return flags


def _rl_trainable_island_flags(args, learner_id: int, num_learners: int) -> str:
    """LoRA rank/targets; for the full-parameter preset the sweep identity (the
    probed layout hash, the frozen roster size and the syncer profile hash);
    for sao-streaming-full this island's hash-bound runtime contract.

    the full-parameter preset trains every parameter, unless DeepSeek V4 expert-full selects
    the LoRA hybrid (attention LoRA plus clone experts, P0 1.7b.2)."""
    if rl_sao_streaming(args):
        island = args.rl_sao_manifest.island(learner_id)
        return (" --parameter-mode full"
                f" --sao-streaming-runtime {shlex.quote(island.runtime.container_path)}"
                f" --sao-streaming-runtime-sha256 {island.runtime.sha256}"
                f" --sao-context-sha256 {island.context_sha256}")
    lora = f" --lora-r {args.lora_r} --lora-targets {args.lora_targets}"
    if not rl_dense_sweep(args):
        if getattr(args, "tuning", "lora") == "full":  # L4.6 buffer island (checked at launch)
            return " --parameter-mode full"
        return lora
    sweep = (f" --parameter-layout-sha256 {args.rl_parameter_layout_sha256}"
             f" --num-learners {num_learners}"
             f" --syncer-profile-sha256 {rl_syncer_profile(args, num_learners).sha256}")
    if rl_full_parameter_async(args):
        sweep += " --full-parameter-rounds async"
    if getattr(args, "rl_sync_expert_token_weights", False):
        sweep += " --sweep-expert-token-weights"
    if rl_full_parameter_final_cut(args) == "ack":
        sweep += " --sweep-final-cut ack"
    if getattr(args, "rl_sync_init_mode", "auto") != "auto":
        sweep += f" --init-mode {args.rl_sync_init_mode}"
    if getattr(args, "rl_sync_push_dtype", "f32") != "f32":
        sweep += f" --sweep-push-dtype {args.rl_sync_push_dtype}"
    if getattr(args, "rl_sync_overlap", False):
        sweep += " --sweep-overlap"
    if getattr(args, "rl_sync_push_no_tensor_blocks", False):
        sweep += " --sweep-push-no-tensor-blocks"
    if getattr(args, "rl_sync_max_local_rounds_ahead", None) is not None:
        sweep += f" --sweep-max-local-rounds-ahead {int(args.rl_sync_max_local_rounds_ahead)}"
    if float(getattr(args, "rl_sync_wait_timeout_s", 900.0)) != 900.0:
        sweep += f" --sweep-wait-timeout-s {float(args.rl_sync_wait_timeout_s):g}"
    if getattr(args, "rl_sync_fragment_max_bytes", 2 << 30) != 2 << 30:
        sweep += f" --sweep-fragment-max-bytes {int(args.rl_sync_fragment_max_bytes)}"
    if getattr(args, "rl_sync_push_dtype", "f32") == "int8":
        sweep += f" --sweep-push-int8-block {int(getattr(args, 'rl_sync_push_int8_block', 128))}"
    if getattr(args, "rl_sync_push_dtype", "f32") == "int4":
        sweep += f" --sweep-push-int4-block {int(getattr(args, 'rl_sync_push_int4_block', 128))}"
        if getattr(args, "rl_sync_push_int4_stochastic", False):
            sweep += " --sweep-push-int4-stochastic"
        error_feedback = getattr(args, "rl_sync_push_int4_error_feedback", "off")
        if error_feedback != "off":
            sweep += f" --sweep-push-int4-error-feedback {error_feedback}"
    sweep += peer_recovery_island_flags(args)
    if getattr(args, "expert_full_count", 0):
        return lora + sweep
    return " --parameter-mode full" + sweep


def make_miles_island_task(
    args,
    spec: ClusterSpec,
    learner_id: int,
    num_learners: int,
    syncer_addr: str,
):
    """Create one Ray/Miles island from the pinned Miles checkout."""

    import sky

    from ..models import resolve
    from ..provenance import is_local_reference

    if not getattr(args, "source_sha256", None) or not getattr(
        args, "reward_sha256", None
    ):
        raise ValueError("RL task requires prepared source and reward provenance")
    island_data, island_data_flags, data_mounts = island_task_data(args, learner_id)
    check_compaction_launch(args)
    # --rl-island-use-gpus-per-node: island rules see M GPUs/node; resources stay physical.
    island = rl_island_spec(args, spec)

    flags = (
        f" --model {shlex.quote(args.model)}"
        f" --rl-model-recipe {shlex.quote(args.rl_model_recipe)}"
        + (
            f" --rl-transformer-impl {args.rl_transformer_impl}"
            if getattr(args, "rl_transformer_impl", "transformer_engine") != "transformer_engine"
            else ""
        )
        + (
            f" --rl-training-attention-backend {args.rl_training_attention_backend}"
            if getattr(args, "rl_training_attention_backend", "auto") not in (None, "auto")
            else ""
        )
        + f" --data {shlex.quote(island_data)}{island_data_flags}"
        + (
            " --rl-single-island-no-sync"
            if getattr(args, "rl_single_island_no_sync", False)
            else " --syncer $SYNCER_ADDR"
            + (" --critic-syncer $CRITIC_SYNCER_ADDR" if rl_needs_critic(args) else "")
        )
        + (" --rl-echo-events" if _echoes_events(args, spec) else "")
        + " --learner-id $LEARNER_ID"
        f" --reward-function {shlex.quote(args.reward_function)}"
        f" --reward-sha256 {shlex.quote(args.reward_sha256)}"
        f" --source-sha256 {shlex.quote(args.source_sha256)}"
        + code_pack_island_flag(args)
        + f" --global-rounds {args.total_steps}"
        f" --sync-preset {args.rl_sync_preset}"
        f" --fragments {args.fragments}"
        f" --pipeline {args.pipeline}"
        f" --local-horizon {args.local_rl_rounds_per_sync}"
        f" --total-fragment-steps {args.rl_total_fragment_steps}"
        f" --groups-per-round {args.rollout_batch_size}"
        f" --samples-per-group {args.n_samples_per_prompt}"
        f" --over-sampling-batch-size {args.over_sampling_batch_size}"
        f" --rl-distributed-timeout-minutes {args.rl_distributed_timeout_minutes}"
        f" --optimizer-steps {int(getattr(args, 'rl_optimizer_steps', 1) or 1)}"
        f" --rollout-max-response-len {args.rollout_max_response_len}"
        f" --completed-groups-path {shlex.quote(args.rl_completed_groups_path)}"
        f" --event-tape ~/yeto-output/rl-island-{learner_id}.jsonl"
        f" --actor-num-nodes {rl_trainer_shape(args, island)[0]}"
        f" --actor-num-gpus-per-node {rl_trainer_shape(args, island)[1]}"
        + (f" --rl-island-gpus-per-node {island.gpus_per_node}" if island.num_nodes > 1 else "")
        + rl_island_bundle_map_flag(args, island)
        + (rl_cross_node_flags(args) if spec.num_nodes > 1 else "")
        + f" --tensor-parallel {args.tensor_parallel}"
        f" --pipeline-parallel {args.pipeline_parallel}"
        f" --rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine}"
        f" --sglang-mem-fraction-static {args.sglang_mem_fraction_static}"
        + _rl_trainable_island_flags(args, learner_id, num_learners)
        + (" --bcast-delta" if int(getattr(args, "rl_sync_bcast_delta_full_every", 0) or 0) else "")
        + _rl_matrix_merge_flag(args)
        + _rl_layout_flags(args)
        + f" --inner-lr {args.inner_lr}"
        f" --seq-len {args.seq_len}"
        + (" --rl-activation-recompute" if getattr(args, "rl_activation_recompute", False) else "")
        + (" --rl-gradient-accumulation-fusion"
           if getattr(args, "rl_gradient_accumulation_fusion", False) else "")
        + (f" --rl-max-tokens-per-gpu {int(args.rl_max_tokens_per_gpu)}"
           if getattr(args, "rl_max_tokens_per_gpu", None) is not None else "")
        + (f" --rl-optimizer-offload-fraction {float(args.rl_optimizer_offload_fraction)!r}"
           if float(getattr(args, "rl_optimizer_offload_fraction", 0.0) or 0.0) > 0 else "")
        + (f" --rl-dp-optimizer {args.rl_dp_optimizer}"
           if (getattr(args, "rl_dp_optimizer", None) or "auto") != "auto" else "")
        + f" --seed {args.seed}"
        f" --wan-streams {args.wan_streams}"
        + _wan_transport_flags(args)
        + " --miles-root ~/miles"
    )
    if getattr(args, "wandb", False):
        flags += " --wandb"
        flags += f" --wandb-project {shlex.quote(args.wandb_project)}"
        flags += f" --wandb-mode {args.wandb_mode}"
        if getattr(args, "wandb_entity", None):
            flags += f" --wandb-entity {shlex.quote(args.wandb_entity)}"
    if getattr(args, "rollout_model", None):
        flags += f" --rollout-model {shlex.quote(args.rollout_model)}"
        flags += (
            " --rollout-model-revision "
            f"{shlex.quote(args.rollout_model_revision)}"
        )
    if getattr(args, "expert_full_count", 0):
        flags += f" --expert-full-count {args.expert_full_count}"
        flags += f" --expert-full-lr {args.expert_full_lr}"
        flags += (
            " --expert-selection-sha256 "
            f"{shlex.quote(args.expert_selection_sha256)}"
        )
        flags += (
            " --expert-selection-contract-sha256 "
            f"{shlex.quote(args.expert_selection_contract_sha256)}"
        )
    dynamic_filter = getattr(args, "dynamic_sampling_filter_path", None)
    if dynamic_filter:
        flags += (
            " --dynamic-sampling-filter-path "
            f"{shlex.quote(dynamic_filter)}"
        )
    if args.dynamic_sampling_max_replacements is not None:
        flags += (
            " --dynamic-sampling-max-replacements "
            f"{args.dynamic_sampling_max_replacements}"
        )
    if args.rl_offload_train:
        flags += " --rl-offload-train"
    algorithm_prelude, algorithm_flags = _ports_algorithm_flags(args)
    flags += algorithm_flags
    if args.expert_parallel is not None:
        flags += f" --expert-parallel {args.expert_parallel}"
    if int(getattr(args, "context_parallel", 1) or 1) > 1:
        flags += f" --context-parallel {int(args.context_parallel)}"
    for flag, name in (
        ("--rl-prompt-column", "rl_prompt_column"),
        ("--rl-label-column", "rl_label_column"),
        ("--sglang-tp-size", "sglang_tp_size"),
        ("--sglang-dp-size", "sglang_dp_size"),
        ("--sglang-ep-size", "sglang_ep_size"),
        ("--sglang-attention-backend", "sglang_attention_backend"),
        ("--sglang-page-size", "sglang_page_size"),
        ("--sglang-max-running-requests", "sglang_max_running_requests"),
        ("--sglang-chunked-prefill-size", "sglang_chunked_prefill_size"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            flags += f" {flag} {shlex.quote(str(value))}"
    if getattr(args, "rl_rollout_quant_tier", 0):  # S26 E6 (validated by the launcher)
        flags += f" --rl-rollout-quant-tier {int(args.rl_rollout_quant_tier)}"
        from yeto.rl.quant.tiers import FP8_SOURCE_ENGINE, fp8_source

        source = fp8_source(args)
        if source != FP8_SOURCE_ENGINE:
            flags += f" --rl-rollout-quant-fp8-source {shlex.quote(source)}"
    if args.use_rollout_routing_replay:
        flags += " --use-rollout-routing-replay"
    if getattr(args, "observe_training_entropy", False):
        flags += " --observe-training-entropy"
    if getattr(args, "mamba_training_ssm_states_dtype", None):
        flags += f" --mamba-training-ssm-states-dtype {args.mamba_training_ssm_states_dtype}"
    if getattr(args, "apply_chat_template_kwargs", None):
        chat_kwargs = json.dumps(
            args.apply_chat_template_kwargs, sort_keys=True, separators=(",", ":")
        )
        flags += f" --apply-chat-template-kwargs {shlex.quote(chat_kwargs)}"
    if not getattr(args, "sglang_deterministic_inference", True):
        flags += " --no-sglang-deterministic-inference"
    if getattr(args, "rl_sampler_snapshot", False):
        flags += " --rl-sampler-snapshot"
    if args.custom_generate_function_path:
        flags += (
            " --custom-generate-function-path "
            f"{shlex.quote(args.custom_generate_function_path)}"
        )
    if getattr(args, "custom_agent_function_path", None):
        flags += (
            " --custom-agent-function-path "
            f"{shlex.quote(args.custom_agent_function_path)}"
        )
    if getattr(args, "agent_max_seq_len", None) is not None:
        flags += f" --agent-max-seq-len {args.agent_max_seq_len}"
    if args.use_session_server:
        flags += " --use-session-server"
        if args.session_server_ip:
            flags += f" --session-server-ip {shlex.quote(args.session_server_ip)}"
        if args.session_server_port:
            ports = " ".join(str(port) for port in args.session_server_port)
            flags += f" --session-server-port {ports}"
        if getattr(args, "session_server_workers", None) is not None:
            flags += f" --session-server-workers {int(args.session_server_workers)}"
        if args.tito_model:
            flags += f" --tito-model {shlex.quote(args.tito_model)}"
        if getattr(args, "tito_allowed_append_roles", None):
            roles = " ".join(
                shlex.quote(role) for role in args.tito_allowed_append_roles
            )
            flags += f" --tito-allowed-append-roles {roles}"
    if args.model_revision:
        flags += f" --model-revision {shlex.quote(args.model_revision)}"
    if args.data_revision:
        flags += f" --data-revision {shlex.quote(args.data_revision)}"
    if args.trust_remote_code:
        flags += " --trust-remote-code"
    if getattr(args, "rl_initial_adapter", None) is not None:
        flags += (
            f" --initial-adapter {RL_INITIAL_ADAPTER_PATH}"
            " --initial-adapter-sha256 "
            f"{shlex.quote(args.rl_initial_adapter_sha256)}"
        )
    if getattr(_rl_backend_module(args, "launch_flags"), "NEEDS_MILES_SOURCE", True):
        miles_setup, sglang_setup = _miles_source_setup(_miles_overlay().resolve_overlay(args))
    else:  # rl-verl-backend: the engine image carries its own pinned source
        miles_setup = sglang_setup = ": # no Miles/SGLang checkout for this backend"
    model = resolve(args.model)
    if is_local_reference(model):
        prefetch = ": # local model; no Hub prefetch"
    else:
        prefetch = model_prefetch(model, args.model_revision)
    worker_model_fetch = ""
    if spec.num_nodes > 1 and not is_local_reference(model):
        # Same call and cache as the learner's own snapshot_download on the
        # head (python3 of the image; `huggingface-cli` is not on the run
        # shell's PATH there: s1-mn-20261003f), so the two nodes resolve the
        # same snapshot directory.
        fetch_py = (
            "from huggingface_hub import snapshot_download; "
            f"print(snapshot_download(repo_id={model!r}, revision={args.model_revision!r}))"
        )
        worker_model_fetch = (
            f"  python3 -c {shlex.quote(fetch_py)} >/tmp/hf-prefetch-worker.log 2>&1 "
            "|| { echo '[yeto-island] worker could not fetch the model snapshot:' >&2; "
            "tail -5 /tmp/hf-prefetch-worker.log >&2; exit 1; }\n"
        )
    file_mounts = dict(data_mounts)
    if getattr(args, "rl_initial_adapter", None) is not None:
        file_mounts[RL_INITIAL_ADAPTER_PATH] = os.path.expanduser(
            args.rl_initial_adapter
        )
    local_token = os.path.expanduser(HF_TOKEN_PATH)
    if os.path.isfile(local_token):
        file_mounts[HF_TOKEN_PATH] = local_token
    if rl_sao_streaming(args):
        file_mounts.update(sao_contract_mounts(args, learner_id))
    envs = {
        "SYNCER_ADDR": syncer_addr,
        **({"CRITIC_SYNCER_ADDR": critic_syncer_address(syncer_addr)}
           if rl_needs_critic(args) and syncer_addr != "none" else {}),
        "LEARNER_ID": str(learner_id),
        # rl-trainer-cold-start-cost L0: phase ``submitted`` (launcher clock)
        SUBMITTED_ENV: f"{submitted_unix():.3f}",
        # decoupling 7.7a: "<vendor>-<card>" from the card catalog; an unknown
        # card raises here, before any cloud spend (no default).
        COMPAT_GROUP_ENV: compat_group(spec.gpu),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Megatron refuses TP>1 or CP>1 without this; it is exported before
        # `ray start` so every Ray worker inherits it.  Harmless at TP1.
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # No NVTE_*_ATTN pins here: the learner passes --attention-backend
        # per recipe (unfused for the generic provider, flash for
        # gated-delta-net hybrids and DeepSeek V4) and Megatron sets the
        # three NVTE variables itself from that choice; a pin that
        # disagrees makes it assert at model construction.
        "CYBERGYM_URL": args.cybergym_url,
        "CYBERGYM_AGENT_ID": args.cybergym_agent_id,
        "CYBERGYM_TIMEOUT": str(args.cybergym_timeout),
    }
    from yeto.rl.profiles import qwen3_8_next as _fn
    if str(args.model or "").rstrip("/").rsplit("/", 1)[-1] in _fn.MODEL_NAMES.values():
        # run_qwen3_8_next.py extra_env_vars, exported before `ray start`
        envs.update(_fn.PORTS_RUNTIME_ENV)
    if args.rl_model_recipe == "deepseek-v4-flash":
        envs.update(
            {
                "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
                "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
                "SGLANG_OPT_FP8_WO_A_GEMM": "0",
                "SGLANG_OPT_FUSE_WQA_WKV": "0",
            }
        )
    if getattr(args, "expert_full_count", 0):
        envs.update(
            {
                "YETO_DSV4_EXPERT_CLONE": "1",
                "YETO_DSV4_EXPERT_FULL_PORTS": "1",
                "YETO_DSV4_EXPERT_FULL_COUNT": str(args.expert_full_count),
                "YETO_DSV4_EXPERT_FULL_LR": str(args.expert_full_lr),
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM": "0",
            }
        )
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if os.environ.get("CYBERGYM_API_KEY"):
        envs["CYBERGYM_API_KEY"] = os.environ["CYBERGYM_API_KEY"]
    for name in ("CYBERGYM_REWARD_SCHEME", "CYBERGYM_REWARD_VIEW"):
        if os.environ.get(name):
            envs[name] = os.environ[name]
    envs.update(elastic_debug_delay_env(args))  # 0.25 ({} unless the test switch is given)
    envs.update(elastic_debug_pause_env(args))  # link-pause test switch ({} by default)
    if getattr(args, "wandb", False):
        # RL islands join the same fleet group as the syncer's tape run.
        envs["YETO_RUN_GROUP"] = args.cluster_prefix
        if os.environ.get("WANDB_API_KEY"):
            envs["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    # rl-trainer-cold-start-cost L0: the setup shell records its start/end and
    # the run shell records ray_ready (start_phases.MARK_DIR); the island replays them.
    setup_steps = [shell_reset(), shell_mark("container_ready"),
                   WAN_TUNING, HF_TOKEN_ENV, miles_setup, sglang_setup]
    if getattr(args, "wandb", False):
        setup_steps.append(
            "pip install -q wandb || echo '[yeto-setup] wandb install failed; "
            "the island will train without telemetry' >&2"
        )
    if getattr(args, "rl_initial_adapter", None) is not None:
        setup_steps.append(f"chmod -R a-w {RL_INITIAL_ADAPTER_PATH}")
    store_envs = store_object_envs(args, spec.cloud, spec.num_nodes)
    if store_envs:  # the store's object API (yeto.cloud.s3_store) needs boto3 in the image
        setup_steps.append(STORE_CLIENT_SETUP)
    store_fs = model_store_filesystem(getattr(args, "model_store", None), spec.cloud, spec.region)
    store_env = ""
    if (spec.cloud == "modal" and getattr(args, "modal_model_volume", None)
            and not is_local_reference(model)):
        store_env = modal_volume_store_env(model, args.model_revision) + "\n"
        setup_steps.append(store_env.rstrip("\n"))
        prefetch = f'[ -n "$YETO_MODEL_STORE_HIT" ] || {prefetch}'
    elif store_fs and not is_local_reference(model):
        store_env = model_store_env(model, args.model_revision) + "\n"
        setup_steps.append(store_env.rstrip("\n"))
        prefetch = f'[ -n "$YETO_MODEL_STORE_HIT" ] || {prefetch}'
    setup_steps.append(prefetch)
    # Ports images (radixark/miles) ship their own Miles at /root/miles on
    # PYTHONPATH; the pinned fork checkout must shadow it.
    island_pythonpath = "$HOME/miles:"
    # Ports images install Megatron-LM with `pip install -e .`, whose editable
    # finder maps only megatron.core and megatron.training; megatron.
    # post_training (imported by megatron.training.get_model whenever
    # nvidia-modelopt is importable, i.e. every raw/native-provider run such
    # as Flash-Next) is then unresolvable (S11 try25 fnA).  Miles' own
    # launchers always put the checkout on PYTHONPATH; do the same, after
    # our own sources so it shadows nothing of ours.  The driver's
    # PYTHONPATH reaches every Ray actor through the job runtime_env.
    island_megatron_path = f":{PORTS_MEGATRON_PATH}"
    # Private --rl-image (registry_login_for: SKYPILOT_DOCKER_* in the
    # environment, or --rl-image-private): the login goes into the task SECRETS, SkyPilot's supported
    # form: every Task load re-derives the DockerLoginConfig from them
    # (sky/task.py _with_docker_login_config). A DockerLoginConfig placed in
    # Resources does not survive sky 0.13's YAML round trip (Resources.
    # from_yaml_config keeps a dict, the next to_yaml_config calls
    # dataclasses.asdict on it: "asdict() should be called on dataclass
    # instances"; B1 nsmoke). SkyPilot exports secrets into setup/run, so both
    # scripts unset them first. Default (public image): nothing injected.
    setup_steps.append(shell_mark("setup_done"))
    registry_login = registry_login_for(args)
    login_unset = DOCKER_LOGIN_UNSET if registry_login else ""
    setup_script = login_unset + "\n".join(setup_steps)
    run_script = (
            f"{login_unset}{HF_TOKEN_ENV}\n"
            f"{store_env}"
            "set -e\n"
            "cd ~/sky_workdir\n"
            f"{island_pre_run_block(getattr(args, 'rl_island_pre_run_script', None))}"
            f"{multinode_env_prelude(spec.cloud, spec.num_nodes)}"
            'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
            # The island's Ray lives in its own temp dir so that cleanup can
            # target it by path.  A whole-machine `ray stop` would also kill
            # SkyPilot's runtime Ray (port 6380, /tmp/ray), after which every
            # status refresh marks the cluster INIT and the head relaunches
            # it forever.  Ray processes carry their session dir on the
            # command line, so pkill by that path never touches sky's.
            f"{rl_island_gpu_env(args, spec)}"
            'MILES_RAY_DIR="$HOME/miles-ray"\n'
            f"{rl_island_ray_spill_dir_prelude(args)}"
            # Same escalation as `ray stop --force` (TERM, short grace, KILL)
            # but scoped to this dir: gcs_server, raylet and the autoscaler
            # monitor ignore a bare SIGTERM for 30 s or more.
            "stop_miles_ray() {\n"
            '  pkill -f "$MILES_RAY_DIR/" >/dev/null 2>&1 || true\n'
            "  for _ in 1 2 3 4 5 6 7 8 9 10; do\n"
            '    pgrep -f "$MILES_RAY_DIR/" >/dev/null 2>&1 || return 0\n'
            "    sleep 1\n"
            "  done\n"
            '  pkill -KILL -f "$MILES_RAY_DIR/" >/dev/null 2>&1 || true\n'
            "}\n"
            "stop_miles_ray\n"
            'if [ "$SKYPILOT_NODE_RANK" = "0" ]; then\n'
            "  ray start --head --node-ip-address=\"$MASTER_ADDR\" "
            # Dashboard on: Miles' --pin-rollout-manager-to-head lists
            # nodes through Ray's state API, which the dashboard serves.
            # On a multi-node island the dashboard must listen on the node
            # ip, not 127.0.0.1: Ray registers the dashboard address in GCS
            # as given, and a Miles manager actor scheduled on a worker node
            # then asks 127.0.0.1:8265 and fails (rl-multinode-island G1,
            # 2026-10-03: ServerUnavailable in compute_ray_pin_head_options).
            '--port=6379 --include-dashboard=true --temp-dir="$MILES_RAY_DIR"'
            + rl_island_ray_gpus(args, spec)
            + rl_island_ray_storage(args, head=True)
            + (" --dashboard-host=0.0.0.0" if spec.num_nodes > 1 else "") + "\n"
            f"  {shell_mark('ray_ready')}\n"
            "  trap stop_miles_ray EXIT\n"
            # Miles calls ray.init(address="auto"), which reads RAY_ADDRESS
            # first and otherwise sky's /tmp/ray/ray_current_cluster file.
            f"{algorithm_prelude}"
            '  RAY_ADDRESS="$MASTER_ADDR:6379" '
            f"PYTHONPATH={island_pythonpath}$HOME/sglang/python:$HOME/sky_workdir{island_megatron_path}"
            "${PYTHONPATH:+:$PYTHONPATH} "
            f"{getattr(args, 'rl_learner_launch_prefix', '')}python3 -m {_rl_backend_module(args, 'launch_flags').ISLAND_ENTRY_MODULE}{flags}"
            f"{_island_post_cmd(args)}\n"
            "else\n"
            # rl-multinode-island D1: the trap is armed before the join loop so a
            # worker killed while joining still cleans its Miles Ray; the join is
            # bounded (head never came up -> the node exits non-zero instead of
            # looping forever).
            "  trap stop_miles_ray EXIT\n"
            # rl-multinode-island G1 (2026-10-03): the learner (head) resolves
            # the model to its own HF-cache snapshot path and hands that path
            # to the rollout engines; an engine on a worker node needs the
            # same snapshot in the worker's cache.  The setup prefetch is a
            # background download, so wait for it here (foreground, same
            # revision, idempotent) before joining the Ray; a failed fetch
            # ends this node non-zero instead of a later engine start error.
            f"{worker_model_fetch}"
            "  for _ in $(seq 1 150); do "
            'ray start --address="$MASTER_ADDR:6379" --temp-dir="$MILES_RAY_DIR"'
            f"{_worker_node_ip_flag(spec.cloud)}"
            f"{rl_island_ray_gpus(args, spec)}{rl_island_ray_storage(args, head=False)} && break; "
            "sleep 2; done\n"
            "  ray status --address=\"$MASTER_ADDR:6379\" >/dev/null 2>&1 "
            "|| { echo '[yeto-island] worker could not join the Ray head' >&2; exit 1; }\n"
            "  while ray status --address=\"$MASTER_ADDR:6379\" "
            ">/dev/null 2>&1; do sleep 5; done\n"
            "fi"
    )
    envs.update(store_envs)
    envs, env_secrets = split_secret_envs(envs)
    in_vm_docker = spec.cloud in IN_VM_DOCKER_CLOUDS
    if in_vm_docker and code_workdir(args) is None:
        # the in-VM `docker run` bind-mounts the host ~/sky_workdir over the image's link
        raise ValueError(f"--rl-code-pack release is not supported on {spec.cloud} "
                         "(in-VM docker mounts the host workdir)")
    if in_vm_docker:
        # C block: sky's docker runtime is unavailable here; the same setup
        # and run execute inside `docker run` of the same pinned image.
        setup_script, run_script = (
            in_vm_docker_setup(args.rl_image, login=bool(registry_login)),
            in_vm_docker_run(args.rl_image, setup_script, run_script,
                             [*envs.keys(), *env_secrets.keys(), *island_hmac_secret(args).keys()],
                             learner_id),
        )
    island_secrets = {**dict(registry_login or {}), **env_secrets, **island_hmac_secret(args)}
    task = sky.Task(
        name=f"yeto-rl-island-{learner_id}",
        setup=setup_script,
        **({"secrets": island_secrets} if island_secrets else {}),
        run=run_script,
        envs=envs,
        num_nodes=spec.num_nodes,
        workdir=code_workdir(args),
        file_mounts=file_mounts or None,
    )
    resources = {
        "infra": f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud,
        "accelerators": spec.accelerators,
        "cpus": rl_island_cpus(args, spec.cloud),
        "instance_type": args.learner_instance_type,
        "use_spot": args.spot,
        "disk_size": args.disk_size,
    }
    if not in_vm_docker:  # Verda: the VM image stays the cloud default; the container is ours
        resources["image_id"] = nebius_baked_image_id(args.rl_image, spec.cloud, spec.region)
    if spec.num_nodes > 1 and getattr(args, "rl_island_network_tier", "auto") != "none":
        tier = multinode_network_tier(spec.cloud, spec.gpu, spec.gpus_per_node)
        if tier:
            resources["network_tier"] = tier
    if spec.cloud != "modal":  # Modal islands take only run + envs (see build_modal_island_config)
        task.set_resources(sky.Resources(**resources))
    storage_mounts = {}
    from ..cloud.droppable import island_keeps_spot_volume

    if island_keeps_spot_volume(args):
        checkpoint_mount = _rl_checkpoint_mount(args.rl_completed_groups_path)
        storage_mounts[checkpoint_mount] = sky.Storage(
            name=_rl_checkpoint_storage_name(args.cluster_prefix, learner_id),
            persistent=False,
            mode=sky.StorageMode.MOUNT,
            sync_on_reconstruction=True,
        )
    if getattr(args, "rl_checkpoint_store", None) and not getattr(args, "rl_elastic", False):
        store = rl_checkpoint_store_plan(args)  # resume store: same bucket mount on sky
        if store is not None and store[1] is not None:
            storage_mounts[ELASTIC_CHECKPOINT_STORE_MOUNT] = sky.Storage(
                source=store[1], mode=sky.StorageMode.MOUNT, persistent=True)
    if getattr(args, "rl_elastic", False):
        # rl-multinode-island Q4 (C5): the consistent checkpoint (journal + cuts) must
        # outlive node0 for a rebuild on other machines; a bucket URI is mounted on
        # every island node, a shared path is left to the deployment (fail open with
        # a warning: single-node and node0-local behavior is unchanged).
        store = rl_checkpoint_store_plan(args)
        if store is not None and store[1] is not None:
            storage_mounts[ELASTIC_CHECKPOINT_STORE_MOUNT] = sky.Storage(
                source=store[1], mode=sky.StorageMode.MOUNT, persistent=True)
        elif store is None and spec.num_nodes > 1:
            print(NODE0_LOCAL_CHECKPOINT_WARNING, file=sys.stderr)
    if storage_mounts:
        task.set_storage_mounts(storage_mounts)
    return task


STORE_CLIENT_SETUP = (
    'python -c "import boto3" 2>/dev/null || pip install -q boto3 '
    "|| { echo '[yeto-setup] boto3 install failed: the s3 store object API is unavailable' >&2; exit 1; }"
)


def _worker_node_ip_flag(cloud: str) -> str:
    """Modal only (CLOUD-OPTIONS-S16 A.5 item 4): a Modal container has both an
    i6pn IPv6 and an IPv4 interface, so a worker lets Ray pick and may register an
    unreachable address.  Pin it to this rank's IPv4 from SKYPILOT_NODE_IPS
    (modal_runner maps the IPv4 list, rank order).  Other clouds: unchanged argv."""
    if cloud != "modal":
        return ""
    return (' --node-ip-address="$(echo "$SKYPILOT_NODE_IPS" '
            '| sed -n "$((SKYPILOT_NODE_RANK + 1))p")"')


def _island_post_cmd(args) -> str:
    """``--rl-island-post-cmd``: shell appended to the rank-0 learner line.

    The learner's exit code is captured with ``|| LRC=$?`` (so ``set -e``
    does not end the script first), the command runs with its output in
    ``~/yeto-output/post-cmd.txt`` (the Modal tape mirror picks ``.txt`` up),
    and the script exits with the learner's code, so the island verdict is
    unchanged.  Empty when the flag is unset (every other run unchanged).
    """
    cmd = getattr(args, "rl_island_post_cmd", None)
    if not cmd:
        return ""
    return (
        " || LRC=$?\n"
        "  mkdir -p \"$HOME/yeto-output\"\n"
        f"  ( set +e; {cmd} ) > \"$HOME/yeto-output/post-cmd.txt\" 2>&1; "
        "echo \"[yeto-island] post-cmd rc=$?\"\n"
        "  exit ${LRC:-0}"
    )


def _echoes_events(args, spec) -> bool:
    """Whether this island's tape is echoed to its log and rebuilt locally."""

    # Every RL island (sky too): rl_learner_finalized tells the launcher that a
    # later non-zero exit is a shutdown error, not an island failure.
    return getattr(args, "training_mode", "sft") == "rl"
