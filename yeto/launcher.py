"""SkyPilot orchestration: one syncer VM + one cluster per learner.

Flow:
  1. build the Rust syncer locally (release) — the binary is file-mounted to
     a cheap CPU VM whose TCP port is opened to the learners; a submitter
     that is not x86_64 Linux (e.g. a Mac) instead has the VM build the
     syncer from the synced repo (SYNCER_REMOTE_BUILD);
  2. launch the syncer cluster, read its head public IP;
  3. launch all learner clusters in parallel (each pinned to its cloud/region
     from the --gpu spec), with the repo synced as the workdir and the syncer
     address passed via env;
  4. stream all job logs with per-cluster prefixes while a fleet controller
     polls job/cluster health, re-provisions failed or preempted clusters
     with their original spec, and abandons (tears down) any learner that
     cannot be restored within --recover-timeout — the run continues with
     the shrunken fleet;
  5. tear everything down (unless --keep; abandoned learners are always
     torn down).

Layout: this module keeps the entry points (``run``, ``dry_run_plan``) and the
orchestration; the pieces it composes live in ``yeto.launch``. Names that
callers reach as ``yeto.launcher.<name>`` are re-exported below.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import delivery
from .gpu_spec import ClusterSpec, parse_gpu_spec
from .models import MODEL_WEIGHT_GB
from .rl.event_echo import TapeCollector as EventCollector
from .launch.rl_backend import _rl_backend_module
from .launch.topology import rl_island_args, rl_island_count, rl_island_placement_records, rl_island_spec
from .launch.syncer import (
    SYNCER_PORT, island_scheduling_mode, island_syncer_addrs, make_syncer_task, probe_syncer_ports,
    resolve_island_syncer_epoch, rl_role_syncers, role_syncer_commands, syncer_event_tape,
    syncer_shard_addrs, syncer_status_reader,
)
from .launch.storage import (
    droppable_rejoins_in_new_container, resumes_in_new_container,
)
from .launch.island_tasks import island_task_plan
from .launch.args import prepare_launch_args
from .launch.rl_island_task import _echoes_events, make_miles_island_task
from .launch.learner_task import make_learner_task
from .launch.sky_ops import (
    SkySDKOps, _tail, learner_cluster_names, maybe_install_nebius_reclaim_hook, sky_cluster_name,
    submit_island_launch,
)
from .launch.tapes import (
    excuse_departed_islands, settle_echo_tapes, syncer_pool_departures, wait_for_tapes,
)
from .launch.modal_islands import (
    _modal_tape_run_dir, _tail_modal, _verify_modal_app_stopped, build_modal_island_config,
    container_set_poller_for, effective_recover_timeout, pull_modal_tapes,
    require_modal_for_gpu_exact, wan_probe_relaunches,
)
from .launch.fleet import (
    CONTAINER_CHANGED_EXIT, DEFAULT_RL_STALL_TIMEOUT_S, FAILED_RUN_DRAIN_S, ISLAND_FAILED_EXIT,
    RUN_STALLED_EXIT, RUN_STOPPED_EXIT, TEARDOWN_UNVERIFIED_EXIT, TRAINER_GAVE_UP_EXIT,
    FixedRosterIslandAbandoned, FleetController,
    RunStalled, RunStopRequested, controller_fixed_roster,
)
from .launch.teardown import collect_diagnostics_parallel, teardown_island
from .launch.verda import _rename_hook, launch_verda_island, prepare_verda_islands
from .launch.manifest import (
    _dashboard_fleet_log, _no_sync_events_dir, _record_sky_trainer_start, _record_trainer_start,
    _write_run_manifest,
)

# Re-exported: callers and tests reach these as ``yeto.launcher.<name>``.
from .launch import REPO_ROOT  # noqa: F401
from .launch.code_pack import prepare_run_code_pack
from .launch.rl_backend import _rl_backend_image_ok, resolve_default_rl_image  # noqa: F401
from .launch.remote_setup import (  # noqa: F401
    HF_TOKEN_ENV, HF_TOKEN_PATH, ISLAND_PRE_RUN_DONE, ISLAND_PRE_RUN_FAILED, ISLAND_PRE_RUN_START,
    NVME_ENV, NVME_SETUP, PORTS_MEGATRON_PATH, TORCH_SETUP, WAN_TUNING, island_pre_run_block,
)
from .launch.topology import (  # noqa: F401
    ISLAND_ROLES, island_role_env, multinode_env_prelude, multinode_network_tier,
    rl_actor_gpus_per_node, rl_colocated_engine_check, rl_cross_node_flags, rl_cross_node_switches,
    rl_island_bundle_map_flag, rl_island_layout, rl_min_nodes, rl_trainer_shape,
)
from .launch.syncer import (  # noqa: F401
    CRITIC_SYNCER_PORT, DEFAULT_ISLAND_LEASE_S, PROBE_BANNER, RL_SYNCER_CHECKPOINT,
    RL_SYNCER_EVENT_TAPE, ROLE_SYNCERS, SYNCER_EVENT_TAPE, SYNCER_REMOTE_BUILD,
    _syncer_expected_contract, _syncer_quorum_timeout, build_syncer_binary, critic_syncer_address,
    elastic_debug_delay_env, elastic_debug_pause_env, island_hmac_secret, probe_listener_command,
    rl_syncer_profile, syncer_command, syncer_ports, syncer_tape_sidecar, tcp_probe, ufw_setup,
)
from .launch.loss import PICKLED_LOSS_PREFIX, pickled_loss_path, resolve_loss_function  # noqa: F401
from .launch.credentials import (  # noqa: F401
    CLOUD_CREDENTIAL_ENV_NAMES, CLOUD_CREDENTIAL_PATHS, _secret_value, _sky_docker_login_config,
    docker_config_login, fleet_clouds, head_cloud, head_cloud_credentials, registry_login_for,
    split_secret_envs,
)
from .launch.storage import (  # noqa: F401
    ELASTIC_CHECKPOINT_STORE_MOUNT, ELASTIC_ISLAND_STATE_DIR, MODEL_STORE_MOUNT,
    RL_HEAD_INITIAL_ADAPTER_PATH, _rl_checkpoint_storage_name, modal_checkpoint_store,
    modal_volume_store_env, model_store_env, nebius_baked_image_id, rl_checkpoint_store_plan,
    model_store_filesystem, model_store_sky_config,
)
from .launch.island_tasks import island_task_data  # noqa: F401
from .launch.rl_ports import (  # noqa: F401
    _ELASTIC_PAUSE_FLAGS, _EVAL_OPTIONAL_FLAGS, EVAL_INLINE_MAX_BYTES, RESTART_LOOP_FN,
    _check_ports_infra_switches, _check_resume_launch_flags, _ports_algorithm_flags,
    _ports_infra_flags, _prepare_ports_algorithm,
)
from .launch.rl_args import (  # noqa: F401
    DEFAULT_SESSION_SERVER_PORT, _default_session_server_port, _prepare_rl_args,
    sao_contract_mounts,
)
from .launch.args import check_cloud_prerequisites  # noqa: F401
from .launch.images import (  # noqa: F401
    GPU_IMAGE_OVERRIDES, MEGATRON_IMAGE, _miles_source_setup, learner_image_for, parse_image_spec,
)
from .launch.in_vm_docker import (  # noqa: F401
    DOCKER_LOGIN_UNSET, IN_VM_DOCKER_CLOUDS, in_vm_docker_image, in_vm_docker_run,
)
from .launch.codex import (  # noqa: F401
    CODEX_BUNDLE_DIR_ENV, CODEX_COMPACTION_ENV, CODEX_CONTAINER_DIR,
    HARNESS_ENVIRONMENT_PROVIDER_ENV, HARNESS_PASSTHROUGH_ENV, HARNESS_PREFLIGHT_ENV,
    HEAD_CODEX_BUNDLE_PATH, MODAL_CLIENT_SETUP, MODAL_SANDBOX_PROVIDER, MODAL_SANDBOX_PROVIDERS,
    PREBAKED_MODAL_SANDBOX_PROVIDER, SANDBOX_MODAL_TOKEN_ENVS, SANDBOX_TOKEN_IN_ISLAND_CHECK,
    codex_bundle_contract, codex_harness_launch, codex_head_staging, modal_sandbox_secret_name,
    require_sandbox_modal_token,
)
from .launch.rl_island_task import _worker_node_ip_flag  # noqa: F401
from .launch.diffusion_sample import make_diffusion_sample_task, run_diffusion_sample  # noqa: F401
from .launch.tapes import _recover_echo_tape  # noqa: F401
from .launch.modal_islands import (  # noqa: F401
    MODAL_STOP_VERIFY_ATTEMPTS, MODAL_STOP_VERIFY_DELAY_S, _list_modal_apps, modal_tape_subdir,
)
from .launch.fleet import (  # noqa: F401
    ABANDONED, FIXED_ROSTER_MAX_RELAUNCHES, RECOVERY_STABLE_S, RUNNING, _RlStrictFailure,
)
from .launch.local_syncer import LocalSyncer  # noqa: F401
from .launch.teardown import (  # noqa: F401
    _cloud_live_instances_probe, _verda_teardown_check, collect_teardown_diagnostics,
    terminate_and_verify,
)
from .launch.verda import VERDA_ISLANDS, _verda_copy_override, remote_sky_api_server  # noqa: F401
from .launch.manifest import _island_tasks_manifest, _miles_overlay_manifest  # noqa: F401


# Rough per-GPU training capacity sanity check (bf16 LoRA, GB).
GPU_MEM_GB = {"A100": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 180, "L4": 24, "A10G": 24, "T4": 16, "V100": 16, "L40S": 48, "RTX-PRO-6000": 96, "RTX-6000-Ada": 48}


def _island_overrides(args, specs) -> dict:
    """launch-preflight-guards 3.2: validated per-island overrides ({} normally)."""
    from .island_overrides import overrides_of
    from .launch_preflight import check_policy_age_spec

    check_policy_age_spec(args)  # dry-run reaches here without pre_cloud_checks
    return overrides_of(args, len(specs))


def _island_role(args, specs, m) -> str | None:
    """rl-spot-cost-saving phase 2: island m's role (anchor | droppable), None without
    --rl-island-role. Admission already ran in pre_cloud_checks; it is re-run here
    (pure, cheap) so every caller sees the same answer."""
    from .cloud import droppable

    return droppable.admit(args, [s.cloud for s in specs]).get(m)


def _island_task(task_factory, args, spec, m, num_learners, syncer_addr, overrides, role=None):
    """The island task, built from a checked args copy for an overridden island
    (sky and Modal islands both start from this task). ``role`` (phase 2) sets the
    island's billing and drops the checkpoint store of a droppable island."""
    from . import island_overrides as io
    from .cloud import droppable
    from .rl.engine import sample_roles

    iargs = droppable.role_args(io.island_args(rl_island_args(args, m), m, overrides), role)
    # L5.4 pair: the island's sample role; it runs as learner 0 of 1 (no syncer).
    iargs = sample_roles.island_args(iargs, m, rl_island_count(args))
    learner_id, island_learners = sample_roles.island_learner(args, m, num_learners)
    task = task_factory(iargs, spec, learner_id, island_learners, syncer_addr)
    io.apply_to_task(task, {**io.island_env(args, m, overrides),
                            **(droppable.role_env(role, spec.cloud, spec.region) if role else {}),
                            **sample_roles.island_env(iargs),
                            **_spot_sampler_cloud_env(iargs, getattr(spec, "cloud", None))})
    return task


def _spot_sampler_cloud_env(iargs, cloud: str | None) -> dict[str, str]:
    """A spot sampler island on Nebius: the VM stop reaches the sampler through the
    host unit of ``yeto.cloud.nebius_reclaim`` (Modal signals the container itself)."""
    from .rl.engine import sample_roles

    if cloud != "nebius" or not getattr(iargs, "spot", False) or not sample_roles.island_env(iargs):
        return {}
    from .cloud import nebius_reclaim

    return nebius_reclaim.spot_sampler_env()


def _ssh_trainer_spec(args, spec: ClusterSpec, task):
    """The SSH spec of a held-back ``ssh:`` trainer island (None on other clouds)."""
    if spec.cloud != "ssh":
        return None
    from .launch.ssh_trainer_island import island_spec_from_inventory

    return island_spec_from_inventory(args, spec, task)


def warn_if_model_wont_fit(args, specs: list[ClusterSpec]) -> None:
    weight_gb = MODEL_WEIGHT_GB.get(args.model)
    if weight_gb is None:
        return
    for spec in specs:
        vram = GPU_MEM_GB.get(spec.gpu, 0) * spec.total_gpus
        if vram < weight_gb:
            print(
                f"[launcher] WARNING: {spec} has ~{vram} GB VRAM but {args.model} "
                f"needs ~{weight_gb} GB for frozen bf16 weights alone — expect OOM.",
                file=sys.stderr,
            )


def run(args, on_clusters=None, local_syncer=None, on_instance_ids=None) -> int:
    """Provision and supervise the fleet; returns the run's exit code.

    `on_clusters`, if given, is called once with the full list of cluster
    names for this run (syncer first). Names are deterministic from the
    args, so the callback fires before provisioning starts — callers (the
    CLI's run registry) can record them for status/teardown even if the
    launch dies mid-provision. Optional and best-effort: existing callers
    need not pass it, and a failing hook never aborts the run.

    `local_syncer` switches on head controller mode: this process is
    running ON the head VM, the syncer is the given LocalSyncer subprocess
    (already started by the caller), and no separate syncer cluster is
    launched — learners connect to this host's public IP
    ($SYNCER_PUBLIC_IP, injected by the submitting CLI).

    `on_instance_ids(cluster, ids)`, if given, receives the Verda instance
    ids of each Verda island once it is up (the CLI saves them in the run
    record; recovery and teardown are checked against them).
    """
    import sky

    if getattr(args, "rl_single_island_no_sync", False) and (
        getattr(args, "training_mode", "sft") != "rl"
    ):
        # Checked first: the flag must never turn an SFT launch syncer-less.
        raise ValueError("--rl-single-island-no-sync requires --training-mode rl")
    from . import launch_preflight, sky_patches
    from .launch import trainer_loop

    # launch-preflight-guards 1.3: thread count before anything touches a cloud.
    args._launch_preflight_manifest = {"preflight": {"threads": launch_preflight.thread_preflight(args)}}
    sky_patches.install()
    prepare_launch_args(args)
    # rl-identity-tiering L9.4: the run code pack (first start makes it; later starts reuse it).
    prepare_run_code_pack(args)
    # 2.3 / 3.1 / 3.6: memory estimate, island overrides, negative-test resume guard.
    launch_preflight.pre_cloud_checks(args, threads=False)
    trainer_loop.check(args)  # rl-trainer-start-stop L5.3b: refuse before any launch
    _write_run_manifest(args)
    head_mode = local_syncer is not None
    # --rl-single-island-no-sync: one ports island, no syncer at all
    # (validated in _prepare_ports_algorithm before any cloud work).
    no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
    if no_sync and head_mode:
        raise ValueError("--rl-single-island-no-sync has no syncer; do not run it as a head")
    specs = parse_gpu_spec(args.gpu)
    nodes_by_cluster = dict(zip(learner_cluster_names(args.cluster_prefix, specs),
                                (s.num_nodes for s in specs)))  # rl-multinode-island D10
    no_sync_incomplete: list[str] = []
    # rl-trainer-start-stop L5.3c: the trainer scheduler starts and stops this island
    # (one Modal call per incarnation); the fleet controller never sees it.
    held = trainer_loop.held_island(args)
    held_name = None if held is None else learner_cluster_names(args.cluster_prefix, specs)[held]
    held_cfg = None
    held_task = None
    # Islands whose tape travels over the log stream (no fetchable
    # ~/yeto-output): every Modal RL ports island, and a no-sync island.
    echo_names = {
        name for name, spec in zip(learner_cluster_names(args.cluster_prefix, specs), specs)
        if _echoes_events(args, spec) and name != held_name
    }
    events_dir = _no_sync_events_dir(args) if echo_names else None
    if echo_names:
        # Refuse to mix runs in one tape, before anything is provisioned.
        existing = sorted(str(p) for p in events_dir.glob("*.jsonl*")) if events_dir.exists() else []
        if existing:
            raise ValueError(
                f"event tapes already exist for run {args.cluster_prefix!r}: {existing}; "
                "use another --cluster-prefix or remove them"
            )
    # External learners (machines sky cannot provision — e.g. Macs running
    # yeto.mlx.learner) get the ids AFTER the cloud learners; the syncer
    # counts them in --learners and its port is already public, so they
    # simply dial in with the printed join command.
    external = max(0, getattr(args, "external_learners", 0) or 0)
    num_learners = len(specs) + external
    require_modal_for_gpu_exact(args, specs)
    warn_if_model_wont_fit(args, specs)
    prefix = args.cluster_prefix
    syncer_cluster = None if head_mode or no_sync else sky_cluster_name(f"{prefix}-syncer")
    learner_names = learner_cluster_names(prefix, specs)
    if on_clusters is not None:
        try:
            on_clusters(([] if syncer_cluster is None else [syncer_cluster]) + learner_names)
        except Exception as e:
            print(f"[launcher] on_clusters hook failed: {e}", file=sys.stderr)
    clusters: list[str] = []
    controller = None
    teardown_unverified = False
    run_started_unix = time.time()  # Modal rows created after this are this run's app
    verda_names = [n for n, sp in zip(learner_names, specs) if sp.cloud == "verda"]
    verda = prepare_verda_islands(verda_names, on_instance_ids) if verda_names else None
    # Read by the finally block (Modal stop / tape pull) even on an early failure.
    modal_ops = None
    modal_cfgs: dict[str, object] = {}

    try:
        # 1. Syncer: a subprocess on this host (head mode) or its own VM.
        if no_sync:
            syncer_task = syncer_job = None
            syncer_addr = "none"  # the island command carries no --syncer
            print("[launcher] --rl-single-island-no-sync: no syncer, no outer sync")
        elif head_mode:
            syncer_task = syncer_job = None
            syncer_addr = syncer_shard_addrs(os.environ['SYNCER_PUBLIC_IP'], args)
            print(f"[launcher] syncer runs on this head node at {syncer_addr}")
        else:
            if getattr(args, "wandb", False):
                print(
                    "[launcher] --wandb: the syncer cluster carries its own "
                    "event-tape forwarder (log: /tmp/yeto-tape-wandb.log)"
                )
            print(f"[launcher] launching syncer cluster {syncer_cluster} in {args.syncer_region}")
            syncer_task = make_syncer_task(args, num_learners)
            rid = sky.launch(syncer_task, cluster_name=syncer_cluster)
            syncer_job, syncer_handle = sky.stream_and_get(rid)
            clusters.append(syncer_cluster)
            syncer_addr = syncer_shard_addrs(str(syncer_handle.head_ip), args)
            print(f"[launcher] syncer up at {syncer_addr}")
            probe_syncer_ports(args, str(syncer_handle.head_ip))
        if not no_sync and island_scheduling_mode(args) == "elastic" \
                and getattr(args, "rl_syncer_epoch", None) is None:
            # 0.17: the islands JOIN with the term the syncer actually runs.
            resolve_island_syncer_epoch(args, syncer_status_reader(
                local_tape=syncer_event_tape(args) if head_mode else None,
                ssh_host=None if head_mode else syncer_cluster))

        if external:
            for x in range(external):
                external_provenance_flags = ""
                if getattr(args, "model_revision", None):
                    external_provenance_flags += (
                        f" --model-revision {shlex.quote(args.model_revision)}"
                    )
                if getattr(args, "data_revision", None):
                    external_provenance_flags += (
                        f" --data-revision {shlex.quote(args.data_revision)}"
                    )
                if getattr(args, "trust_remote_code", False):
                    external_provenance_flags += " --trust-remote-code"
                if getattr(args, "source_sha256", None):
                    external_provenance_flags += (
                        f" --source-sha256 {shlex.quote(args.source_sha256)}"
                    )
                for flag_name in (
                    "model_requested_identifier",
                    "model_requested_revision",
                    "data_requested_identifier",
                    "data_requested_revision",
                ):
                    flag_value = getattr(args, flag_name, None)
                    if flag_value is not None:
                        external_provenance_flags += (
                            f" --{flag_name.replace('_', '-')} "
                            f"{shlex.quote(str(flag_value))}"
                        )
                print(
                    f"[launcher] external learner slot {len(specs) + x}: join with\n"
                    f"    python -m yeto.mlx.learner --model {shlex.quote(args.model)} "
                    f"--data {shlex.quote(args.data)} --syncer {shlex.quote(syncer_addr)} "
                    f"--learner-id {len(specs) + x} --num-learners {num_learners} "
                    f"--data-format {getattr(args, 'data_format', 'auto')} "
                    f"--train-on {args.train_on} "
                    f"--assistant-mask-mode "
                    f"{getattr(args, 'assistant_mask_mode', 'native')} "
                    f"--seed {getattr(args, 'seed', 0)} "
                    f"--tuning {args.tuning} --lora-r {args.lora_r} "
                    f"--lora-targets {getattr(args, 'lora_targets', 'auto')} "
                    f"--seq-len {args.seq_len} --fragments {args.fragments} "
                    f"--fragment-pattern {args.fragment_pattern} "
                    f"--merge-alpha {args.merge_alpha} --wire-dtype {args.wire_dtype}"
                    f"{external_provenance_flags}"
                )
            print(
                f"[launcher] the syncer will wait for all {num_learners} learners "
                f"({len(specs)} cloud + {external} external) before training starts"
            )

        # 2. Learners, in parallel. Modal entries are not sky clusters: the
        #    factory still builds their task (run script, envs) and the
        #    Modal runner executes it in a GPU container.
        tasks = {}
        rids = {}
        # Bound before any island is built: the failure path below stops the
        # Modal app when modal_ops is set, and must not NameError otherwise.
        modal_island_ops = None
        modal_ops = None
        task_factory = (
            make_miles_island_task
            if getattr(args, "training_mode", "sft") == "rl"
            else make_learner_task
        )
        modal_addr = None
        if no_sync:
            modal_addr = syncer_addr  # nothing to reach
        elif any(spec.cloud == "modal" for spec in specs):
            from .modal_runner import resolve_syncer_for_modal

            # Fails BEFORE any Modal container starts when the syncer is
            # not reachable from Modal's network.
            modal_addr = resolve_syncer_for_modal(
                syncer_addr, getattr(args, "syncer_public_addr", None)
            )
        island_overrides = _island_overrides(args, specs)
        for m, spec in enumerate(specs):
            name = learner_names[m]
            role = _island_role(args, specs, m)
            task = _island_task(task_factory, args, spec, m, num_learners, syncer_addr,
                                island_overrides, role)
            if spec.cloud == "modal":
                from .cloud.droppable import role_args

                cfg = build_modal_island_config(role_args(args, role), spec, m, task, modal_addr)
                print(f"[launcher] {name} dials {island_syncer_addrs(cfg.envs)}", flush=True)
                if name == held_name:
                    held_cfg = cfg  # defined on the app below; the scheduler spawns it
                    print(f"[launcher] {name} is the trainer island: the trainer scheduler "
                          "starts and stops it", flush=True)
                    continue
                tasks[name] = cfg
                modal_cfgs[name] = cfg
                continue
            if name == held_name:
                held_task = task  # the scheduler launches one cluster per incarnation
                print(f"[launcher] {name} is the trainer island: the trainer scheduler "
                      "starts and stops it", flush=True)
                continue
            tasks[name] = task
            print(f"[launcher] {name} dials {island_syncer_addrs(getattr(task, 'envs', None))}", flush=True)
            alloc = (rl_island_spec(rl_island_args(args, m), spec)
                     if getattr(args, "training_mode", "sft") == "rl" else spec)
            print(f"[launcher] launching learner {m} on {spec} as {name}"
                  + (f" (physical {spec.gpus_per_node} GPUs/node billed, island allocated "
                     f"{alloc.gpus_per_node}/node = GPUs 0..{alloc.gpus_per_node - 1})"
                     if alloc is not spec else ""))
            if spec.cloud == "verda":
                # Launched in resolve(): live stock -> ordered any_of, with
                # refresh + bounded backoff on capacity failures.
                rids[name] = (m, ("verda", spec))
                continue
            rids[name] = (m, submit_island_launch(sky, args, spec, task, name))

        results = {}
        errors = {}

        def resolve(name: str, m: int, rid) -> None:
            try:
                if isinstance(rid, tuple) and rid and rid[0] == "verda":
                    from .cloud.droppable import role_args

                    results[name] = launch_verda_island(
                        sky, tasks[name], name, rid[1],
                        role_args(args, _island_role(args, specs, m)))
                    verda["guard"].record(
                        name, getattr(results[name][1], "cluster_name_on_cloud", None)
                    )
                else:
                    results[name] = sky.stream_and_get(rid)
                    maybe_install_nebius_reclaim_hook(tasks[name], results[name], name)
            except Exception as e:
                errors[name] = e

        threads = [
            threading.Thread(target=resolve, args=(n, m, r), daemon=True)
            for n, (m, r) in rids.items()
        ]
        for t in threads:
            t.start()
        if modal_cfgs or held_cfg is not None:
            from .modal_runner import ModalIslandOps, ModalOps, modal_app_name

            modal_ops = ModalOps(modal_app_name(prefix))
            for cfg in [*modal_cfgs.values(), *([] if held_cfg is None else [held_cfg])]:
                modal_ops.define(cfg)
            print(f"[launcher] deploying Modal app {modal_ops.app_name} "
                  f"({len(modal_cfgs) + (held_cfg is not None)} island(s))")
            modal_ops.deploy()
            modal_island_ops = ModalIslandOps(modal_ops)
            for name, cfg in modal_cfgs.items():
                print(f"[launcher] launching learner {cfg.learner_id} on Modal as {name}")
                call_id = modal_island_ops.relaunch(cfg, name)
                if call_id is None:
                    errors[name] = RuntimeError("Modal refused to start the island")
                else:
                    results[name] = (call_id, None)
        for t in threads:
            t.join()
        for name in list(rids) + list(modal_cfgs):
            if name not in errors:
                clusters.append(name)
        if errors:
            for name, e in errors.items():
                print(f"[launcher] ERROR launching {name}: {e}", file=sys.stderr)
            raise RuntimeError(f"{len(errors)} learner cluster(s) failed to provision")

        # 3. Stream logs while the fleet controller polls health, recovers
        #    failed/preempted clusters, and abandons learners that stay down
        #    past --recover-timeout.
        def spawn_tail(name: str, job_id) -> None:
            label = "syncer" if name == syncer_cluster else name
            collector = None
            if name in echo_names:
                # No syncer tape and (on Modal) no fetchable ~/yeto-output: the
                # island echoes its tape into the log; rebuild it locally.
                collector = event_collectors.get(name)
                if collector is None:  # a relaunch keeps appending to the same tape
                    collector = event_collectors[name] = EventCollector(
                        events_dir / f"{name}.jsonl", fresh=False
                    )
            if modal_ops is not None and name in modal_cfgs:
                from .modal_runner import ContainerIdGuard

                rejoins = droppable_rejoins_in_new_container(args, modal_cfgs[name])
                guard = ContainerIdGuard(
                    on_change=lambda m, _r=rejoins: on_container_change(m, droppable_rejoin=_r))
                guard.expected = rejoins
                container_guards.append(guard)
                thread = threading.Thread(
                    target=_tail_modal, args=(modal_ops, job_id, label, collector), kwargs={"guard": guard},
                    daemon=True
                )
            else:
                thread = threading.Thread(
                    target=_tail, args=(name, job_id, label, collector), daemon=True
                )
            thread.start()
            tail_threads.append(thread)

        tail_threads: list[threading.Thread] = []
        event_collectors: dict[str, EventCollector] = {}
        container_guards: list = []

        resume_on_new_container = resumes_in_new_container(args)

        def on_container_change(message: str, droppable_rejoin: bool = False) -> None:
            if droppable_rejoin:
                # rl-spot-cost-saving 5.2: a droppable island reclaimed on Modal re-runs
                # in a new container and JOINs again (catch_up) from the syncer base.
                print(f"[launcher] WARNING: {message}; droppable island, it rejoins from "
                      "the syncer base (no failure)", file=sys.stderr, flush=True)
                if events_dir is not None:
                    try:
                        with open(Path(events_dir) / "launcher-errors.jsonl", "a") as fh:
                            fh.write(json.dumps({"event": "modal_container_changed", "level": "warning",
                                                 "droppable_rejoin": True, "message": message,
                                                 "time": time.time()}) + "\n")
                    except OSError:
                        pass
                return
            if resume_on_new_container:
                # rl-resume-from-checkpoint: the Volume store carries LATEST; the re-run
                # container resumes from the newest complete cut instead of starting over.
                print(f"[launcher] WARNING: {message}; the island resumes from its "
                      "modal-volume checkpoint store (no failure)", file=sys.stderr, flush=True)
                if events_dir is not None:
                    try:
                        with open(Path(events_dir) / "launcher-errors.jsonl", "a") as fh:
                            fh.write(json.dumps({"event": "modal_container_changed", "level": "warning",
                                                 "resume": True, "message": message,
                                                 "time": time.time()}) + "\n")
                    except OSError:
                        pass
                return
            print(f"[launcher] ERROR: {message}; failing the run (exit {CONTAINER_CHANGED_EXIT}) "
                  "and stopping the Modal app", file=sys.stderr, flush=True)
            if events_dir is not None:
                try:
                    with open(Path(events_dir) / "launcher-errors.jsonl", "a") as fh:
                        fh.write(json.dumps({"event": "modal_container_changed", "level": "error",
                                             "message": message, "time": time.time()}) + "\n")
                except OSError:
                    pass
            try:
                modal_ops.stop_app()
            except Exception as e:  # noqa: BLE001 - teardown verification reports the rest
                print(f"[launcher] modal app stop failed: {e}", file=sys.stderr, flush=True)

        def container_changed() -> bool:
            return not resume_on_new_container and any(
                g.tripped and not getattr(g, "expected", False) for g in container_guards)

        if syncer_cluster is not None:
            spawn_tail(syncer_cluster, syncer_job)
        for name, (job_id, _handle) in results.items():
            spawn_tail(name, job_id)
        container_poller = container_set_poller_for(args, modal_ops, modal_cfgs, on_container_change)
        if container_poller is not None:
            container_guards.append(container_poller)

        from . import runs
        from .modal_runner import RoutingOps

        sky_sdk_ops = SkySDKOps()
        # rl-multinode-island D10: the failure-path teardown verifies every node instance
        sky_sdk_ops.nodes_by_cluster = dict(nodes_by_cluster)
        controller = FleetController(
            learners={name: (tasks[name], job_id) for name, (job_id, _h) in results.items()},
            syncer=None if syncer_cluster is None else (syncer_cluster, syncer_task, syncer_job),
            sky_ops=RoutingOps(sky_sdk_ops, modal_island_ops),
            poll_interval=args.controller_poll,
            recover_timeout=effective_recover_timeout(args),
            on_relaunch=spawn_tail,
            syncer_probe=local_syncer.probe if head_mode else None,
            syncer_restart=local_syncer.restart if head_mode else None,
            # 10-09 ruling of the main agent for the user: elastic is NOT a fixed
            # roster (an elastic island that fails for good is dropped, the rest go on).
            fixed_roster=controller_fixed_roster(args),
            stop_flag=runs.stop_flag_path(args.cluster_prefix),
            finalized_probe=(
                (lambda name: name in event_collectors and event_collectors[name].finalized)
                if echo_names else None
            ),
            progress_probe=(
                (lambda: sum(c.count for c in list(event_collectors.values())))
                if echo_names else None
            ),
            stall_timeout=float(
                getattr(args, "rl_stall_timeout", None) or DEFAULT_RL_STALL_TIMEOUT_S
            ) if getattr(args, "rl_stall_timeout", None) != 0 else 0.0,
            keep_abandoned=bool(getattr(args, "keep_abandoned", False)),
            no_recover=verda["no_recover"] if verda else (),
            instance_guard=verda["guard"] if verda else None,
            on_rename=_rename_hook(clusters, on_clusters, [] if head_mode else [syncer_cluster]),
            fleet_log=_dashboard_fleet_log(args, tasks, results),
            syncer_finished_probe=local_syncer.finished if head_mode else None,
            left_pool_probe=(
                (lambda name: name in event_collectors and event_collectors[name].left_pool)
                if echo_names else None
            ),
            elastic=island_scheduling_mode(args) == "elastic",
            no_island_relaunch=bool(getattr(args, "no_island_relaunch", False)),
            wan_reject_probe=(
                (lambda name: event_collectors[name].wan_rejects if name in event_collectors else 0)
                if echo_names else None
            ),
            wan_probe_relaunches=wan_probe_relaunches(args),
        )
        def drain_tapes(limit: float = NO_SYNC_EVENT_DRAIN_S) -> None:
            if echo_names:
                # The island's last events (finalization) must be on disk before
                # teardown: the log streams end when the island exits; bounded wait.
                wait_for_tapes(event_collectors, echo_names, tail_threads, limit)
                modal_tape_dir = None
                if modal_ops is not None and any(n in modal_cfgs for n in echo_names):
                    # Modal islands: the tape Volume (committed every 30 s and at
                    # exit) is the island-side source the echo mirrors.
                    modal_tape_dir = _modal_tape_run_dir(args) / "modal-tape"
                    pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
                no_sync_incomplete.extend(settle_echo_tapes(
                    event_collectors, echo_names, modal_cfgs, events_dir,
                    modal_tape_dir=modal_tape_dir))
                _record_trainer_start(args, events_dir)

        try:
            exit_codes = trainer_loop.run_with_trainer_loop(
                args, controller.run,
                island=None if held_name is None else trainer_loop.TrainerIsland(
                    held_name, specs[held].cloud, held_cfg, modal_ops, sky_task=held_task,
                    spec=specs[held], events_dir=events_dir,
                    ssh=_ssh_trainer_spec(args, specs[held], held_task)),
                abort=controller.abort)
        except trainer_loop.TrainerLoopGaveUp as error:
            # the trainer scheduler released the trainer and stopped; the samplers
            # cannot finish the run usefully: tear everything down
            print(f"[launcher] ERROR: {error}; stopping the run (exit {TRAINER_GAVE_UP_EXIT})",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT if container_changed() else TRAINER_GAVE_UP_EXIT
        except RunStalled as error:
            print(f"[launcher] ERROR: {error}; stopping the run (exit {RUN_STALLED_EXIT})",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT if container_changed() else RUN_STALLED_EXIT
        except RunStopRequested as error:
            # yeto stop-run: the trainer loop has stopped its trainer (cut) on the way
            # out; secure the tapes (bounded), then the finally block tears down
            print(f"[launcher] {error}: stopping the run (exit {RUN_STOPPED_EXIT})",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return RUN_STOPPED_EXIT
        except FixedRosterIslandAbandoned as error:
            # A fixed-roster island failed for good: the run cannot finish.
            # Secure the tapes (bounded), then the finally block tears every
            # recorded resource down. Other controller errors (strict syncer /
            # strict RL job failure, all learners abandoned, bugs) propagate.
            print(f"[launcher] ERROR: {error}; stopping the run (exit "
                  f"{ISLAND_FAILED_EXIT})", file=sys.stderr)
            # surviving islands still stream (blocked on the syncer): short wait
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            if no_sync_incomplete:
                print(f"[launcher] event tape incomplete for {no_sync_incomplete} "
                      "(expected after an island failure)", file=sys.stderr)
            return CONTAINER_CHANGED_EXIT if container_changed() else ISLAND_FAILED_EXIT
        except Exception as error:
            if not container_changed():
                raise
            # the app stop that follows a container change makes the controller error out
            print(f"[launcher] controller ended after a Modal container change: {error}",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT
        drain_tapes()
        if container_changed():
            return CONTAINER_CHANGED_EXIT
        if (no_sync_incomplete and head_mode and island_scheduling_mode(args) == "elastic"):
            # S16 §6.6: an island the syncer removed from the pool for an excused
            # reason (finished, stopped, preempted, lease expiry) owes no
            # rl_learner_finalized; report it, do not exit 3. Failure codes still fail.
            departures = syncer_pool_departures(
                local_syncer.event_tape, getattr(local_syncer, "_event_offset", 0))
            still, left = excuse_departed_islands(no_sync_incomplete, departures)
            no_sync_incomplete[:] = still
            for item in left:
                print(f"[launcher] run summary: island {item['island']} ({item['name']}) "
                      f"left the pool mid-run (syncer pool_leave reason={item['reason']}); "
                      "no rl_learner_finalized required", file=sys.stderr)
            if left:
                try:
                    from . import runs as _runs
                    _runs.update_run(args.cluster_prefix, islands_left_pool=left)
                except Exception as e:  # noqa: BLE001 - summary is best effort
                    print(f"[launcher] WARN: could not record islands_left_pool ({e})",
                          file=sys.stderr)
        failed = [n for n, s in exit_codes.items() if "SUCCEEDED" not in s]
        if no_sync_incomplete:
            print(
                f"[launcher] event tape incomplete for {no_sync_incomplete}; "
                f"exit {NO_SYNC_INCOMPLETE_EXIT}",
                file=sys.stderr,
            )
            if no_sync:  # nothing else to secure: no syncer checkpoint
                return NO_SYNC_INCOMPLETE_EXIT

        # Secure the artifact BEFORE the finally block tears learners down:
        # fetch ~/yeto-output from the winning learner onto this machine
        # (the head, or the local worker), then deliver to --output.
        done = [n for n, s in exit_codes.items() if "SUCCEEDED" in s]
        if not done:
            print("[launcher] no learner succeeded; recover from the syncer "
                  "checkpoint with yeto-export", file=sys.stderr)
            return 1
        rl_mode = getattr(args, "training_mode", "sft") == "rl"
        # Prefer a sky learner as the artifact source (rsync); a Modal
        # island's ~/yeto-output is not reachable that way.
        sky_done = [n for n in done if n not in modal_cfgs]
        source = (
            syncer_cluster
            if rl_mode and syncer_cluster is not None
            else next((n for n in sky_done if "-l0-" in n), (sky_done or done)[0])
        )
        output = getattr(args, "output", None)
        local_dest = (
            os.path.expanduser(output)
            if output and delivery.kind(output) == "local" and not head_mode
            else os.path.expanduser("~/yeto-output")
        )
        os.makedirs(local_dest, exist_ok=True)
        if rl_mode and head_mode:
            print(f"[launcher] committed RL checkpoint retained at {local_dest}")
        elif source in modal_cfgs and no_sync:
            print(
                f"[launcher] --rl-single-island-no-sync island ran on Modal ({source}); "
                "its ~/yeto-output is not fetchable over ssh and there is no syncer "
                "checkpoint -- use the streamed island log / event tape as the evidence",
                file=sys.stderr,
            )
            # S17 N16: the run itself is judged by the island outcome (its tape
            # was already checked complete above).  Only an explicit --output
            # the launcher cannot honour stays exit 2.
            if output:
                print(f"[launcher] --output {output} cannot be delivered from a "
                      "Modal no-sync island; exit 2", file=sys.stderr)
                return 2
            return 1 if failed else 0
        elif source in modal_cfgs:
            print(
                f"[launcher] every successful learner ran on Modal ({source}); its "
                "~/yeto-output is not fetchable over ssh — recover the model from "
                "the syncer checkpoint with yeto-export",
                file=sys.stderr,
            )
            return 2
        else:
            try:
                subprocess.run(delivery.fetch_cmd(source, local_dest), check=True)
                artifact = "committed RL checkpoint" if rl_mode else "fine-tuned model"
                print(f"[launcher] {artifact} fetched to {local_dest}")
            except subprocess.CalledProcessError as e:
                print(
                    f"[launcher] fetching {source}:~/yeto-output failed ({e}); "
                    "recover from the syncer checkpoint with yeto-export",
                    file=sys.stderr,
                )
                return 2
        if delivery.is_remote(output):
            try:
                delivery.deliver(output, local_dest)
                print(f"[launcher] output uploaded to {output}")
            except Exception as e:
                print(f"[launcher] upload to {output} failed: {e}; the model "
                      f"remains at {local_dest}", file=sys.stderr)
                return 2
        if failed:
            return 1
        # A synced run still fetches/delivers its checkpoint first, then
        # fails closed on an incomplete island tape.
        return NO_SYNC_INCOMPLETE_EXIT if no_sync_incomplete else 0
    finally:
        if verda is not None and verda.get("pth_written"):
            from .sky_patches import remove_local_pth

            remove_local_pth(verda["pth_written"])
        # Clusters the controller already tore down (abandoned learners, or
        # the syncer after a total loss) are skipped — even with --keep.
        downed = controller.downed_clusters if controller is not None else set()
        remaining = [c for c in clusters if c not in downed]
        if modal_ops is not None and modal_cfgs:
            # Containers commit their tape every 30 s and at exit; pulled
            # here (before the app stop) and again after it below.
            pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
        if args.keep:
            print(f"[launcher] keeping clusters: {remaining}")
        else:
            unverified = []
            collect_diagnostics_parallel(
                [n for n in remaining if n not in modal_cfgs], head=head_mode
            )
            for name in remaining:
                print(f"[launcher] tearing down {name}")
                if name in modal_cfgs:
                    try:
                        if modal_island_ops is not None:
                            modal_island_ops.down(name)
                    except Exception as e:  # noqa: BLE001 - app stop below is the backstop
                        print(f"[launcher] cancel of {name} failed: {e}", file=sys.stderr)
                    continue
                if not teardown_island(sky, name, head=head_mode, collect=None,
                                       num_nodes=nodes_by_cluster.get(name, 1)):
                    unverified.append(name)
            # rl-trainer-cold-start-cost L0.2: sky islands without an echoed tape
            # (their tapes are in the pre-teardown diagnostics, also for islands
            # the controller tore down earlier) still give the manifest T_s.
            _record_sky_trainer_start(
                args, [n for n in nodes_by_cluster if n not in modal_cfgs and n not in echo_names])
            if modal_ops is not None:
                # Belt and braces: stop the whole per-run Modal app so no
                # function call of this run outlives the launcher.
                try:
                    modal_ops.stop_app()
                except Exception as e:  # noqa: BLE001
                    print(f"[launcher] Modal app stop failed: {e}", file=sys.stderr)
                if not _verify_modal_app_stopped(modal_ops, args,
                                                 run_started_unix=run_started_unix):
                    teardown_unverified = True
                if modal_cfgs:
                    # The final at-exit commits land once the containers are gone.
                    pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
            if unverified:
                # The head must NOT self-terminate: it is the only thing that
                # can still reach these orphaned learner clusters via sky.
                args._teardown_incomplete = True
                print(
                    f"[launcher] WARNING: could not confirm termination of "
                    f"{unverified}; leaving the head up so they stay reachable. "
                    f"Terminate them from the cloud console, then: yeto down {prefix}",
                    file=sys.stderr,
                )
        if head_mode:
            # The head VM cannot tear itself down here (the teardown would
            # kill this very process mid-flight). With a remote --output the
            # caller (cmd_head) self-terminates via the EC2 API after this
            # returns cleanly; otherwise the head stays up so the fetched
            # model and syncer checkpoint remain reachable.
            head_cluster = sky_cluster_name(f"{prefix}-head")
            if args.keep:
                print(
                    f"[launcher] run finished; clusters left up: "
                    f"{remaining + [head_cluster]}; tear everything down "
                    f"with: yeto down {prefix}",
                    flush=True,
                )
            elif delivery.is_remote(getattr(args, "output", None)):
                print("[launcher] run finished; head will self-terminate "
                      "after delivery", flush=True)
            else:
                print(
                    f"[launcher] run finished; model + checkpoint live on the "
                    f"head — tear it down with: yeto down {prefix}",
                    flush=True,
                )
        if teardown_unverified and sys.exc_info()[0] is None:
            # the run's own outcome is secondary to a possibly still-running app
            return TEARDOWN_UNVERIFIED_EXIT  # noqa: B012


NO_SYNC_EVENT_DRAIN_S = 120.0
# no-sync run whose island tape lacks rl_learner_finalized (differs from 2:
# "artifact not fetchable").
NO_SYNC_INCOMPLETE_EXIT = 3


def _dry_run_role_syncers(args, num_learners: int) -> list[dict]:
    commands = role_syncer_commands(args, num_learners)
    return [
        {
            "role": syncer.role,
            "port": syncer.port,
            "address": syncer.address,
            "layout": f"{syncer.role}_layout_hash",
            "command": commands[syncer.role],
            "checkpoint": syncer.checkpoint,
            "event_tape": syncer.event_tape,
        }
        for syncer in rl_role_syncers(args)
    ]


def dry_run_plan(args) -> dict:
    """``yeto launch --dry-run``: what a launch would request, creating nothing.

    Runs on prepared args (``prepare_launch_args`` already validated the
    request, including the ports algorithm/capability checks) and builds the
    island tasks in memory only to read the learner command; no sky/Modal
    call is made.
    """

    no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
    rl = getattr(args, "training_mode", "sft") == "rl"
    if no_sync and not rl:
        raise ValueError("--rl-single-island-no-sync requires --training-mode rl")
    head = getattr(args, "controller", "local") == "head"
    if no_sync and head:
        raise ValueError("--rl-single-island-no-sync has no syncer; use --controller local")
    from .launch import trainer_loop
    from .rl.engine import sample_roles

    trainer_loop.check(args)  # the launch refuses the same scheduler combinations
    specs = parse_gpu_spec(args.gpu)
    external = max(0, getattr(args, "external_learners", 0) or 0)
    islands = []
    island_overrides = _island_overrides(args, specs) if rl else {}
    from .launch_preflight import dry_run_memory

    memory = dry_run_memory(args, specs)
    task_plan = island_task_plan(args) if rl else None
    placements = rl_island_placement_records(args) if rl else []
    for learner_id, spec in enumerate(specs):
        entry = {
            "learner_id": learner_id,
            "cloud": spec.cloud,
            "region": spec.region,
            "gpu": spec.gpu,
            "num_nodes": spec.num_nodes,
            "gpus_per_node": spec.gpus_per_node,
            "total_gpus": spec.total_gpus,
        }
        if rl:
            entry["placement"] = placements[learner_id]
        if rl and rl_island_spec(rl_island_args(args, learner_id), spec) is not spec:
            entry["allocated_gpus_per_node"] = rl_island_spec(
                rl_island_args(args, learner_id), spec).gpus_per_node
        if memory is not None:
            entry["memory_estimate"] = memory[learner_id]
        if rl:
            role = _island_role(args, specs, learner_id)
            task = _island_task(make_miles_island_task, args, spec, learner_id,
                                len(specs) + external, "$SYNCER_ADDR", island_overrides, role)
            if role is not None:
                from .cloud.droppable import billing, rejoin_source

                entry["island_role"] = {"role": role, "billing": billing(role),
                                        "rejoin": rejoin_source(role)}
            if sample_roles.is_store_coupled(args):  # L5.4/L7.5: per-island role, learner 0 of 1
                entry["sample_role"] = sample_roles.island_args(
                    args, learner_id, len(specs)).rl_sample_role
            if learner_id == trainer_loop.held_island(args):
                entry["held_by_trainer_scheduler"] = True  # the fleet never starts it
            if task_plan is not None:
                entry["island_tasks"] = task_plan[learner_id].record()
            if island_overrides.get(learner_id):
                from .island_overrides import records

                entry["island_overrides"] = [r for r in records(args, island_overrides)
                                             if r["island"] == learner_id]
            entry["learner_command"] = next(
                (line.strip() for line in task.run.splitlines() if _rl_backend_module(args, "launch_flags").ISLAND_ENTRY_MODULE in line),
                None,
            )
        islands.append(entry)
    return {
        "dry_run": True,
        "training_mode": getattr(args, "training_mode", "sft"),
        "rl_engine": "ports" if rl else None,
        "controller": "head" if head else "local",
        "islands": len(specs) + external,
        "external_learners": external,
        "total_gpus": sum(s.total_gpus for s in specs),
        "syncer": None if no_sync else ("head VM" if head else f"{args.cluster_prefix}-syncer"),
        # P0-D8: one syncer per trainable role (actor first); none without outer sync
        "role_syncers": [] if not rl or no_sync else _dry_run_role_syncers(
            args, len(specs) + external),
        "outer_sync": not no_sync,
        # rl-trainer-start-stop L5.3b/L5.4: the head's scheduler and the island it owns
        "trainer_scheduler": (None if not trainer_loop.enabled(args) else {
            "mode": trainer_loop.scheduler_mode(args), "island": trainer_loop.held_island(args),
            "convertible_pool": trainer_loop.convertible_pool(args),
            "interval_s": args.rl_trainer_scheduler_interval}),
        "algorithm_spec_sha256": getattr(args, "rl_expected_algorithm_sha256", None),
        "algorithm_spec": getattr(args, "rl_algorithm_spec_json", None),
        "unverified_mechanisms": list(getattr(args, "rl_allow_unverified_mechanism", None) or ()),
        "clusters": [] if not islands else learner_cluster_names(args.cluster_prefix, specs),
        "island_requests": islands,
    }
