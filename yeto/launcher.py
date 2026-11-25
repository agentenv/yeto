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
    SYNCER_PORT, island_pool, island_syncer_addrs, make_syncer_task, probe_syncer_ports,
    resolve_island_syncer_epoch, rl_role_syncers, role_syncer_commands, role_syncer_shards,
    syncer_event_tape, syncer_shard_addrs, syncer_status_reader,
)
from .launch.storage import (
    droppable_rejoins_in_new_container, resumes_in_new_container,
)
from .launch.island_tasks import data_split_plan, island_task_plan
from .launch.args import prepare_launch_args
from .launch import engine_cluster
from .launch.rl_island_task import _echoes_events, make_miles_island_task
from .launch.learner_task import make_learner_task
from .launch.sky_ops import (
    SkySDKOps, _tail, learner_cluster_names, maybe_install_nebius_reclaim_hook, sky_cluster_name,
    submit_island_launch,
)
from .launch.trainer_provision import resident_ssh_names as _resident_ssh_names
from .launch.trainer_provision import ssh_trainer_spec as _ssh_trainer_spec
from .launch.tapes import (
    excuse_departed_islands, settle_echo_tapes, syncer_pool_departures, wait_for_tapes,
)
from .launch.modal_islands import (
    _modal_tape_run_dir, _tail_modal, _verify_modal_app_stopped, build_modal_island_config,
    container_log_follower_for, container_set_poller_for, effective_recover_timeout,
    finalized_island_containers, pull_modal_tapes, require_modal_for_gpu_exact, wan_probe_relaunches,
)
from .launch.peer_recovery import relaunch_envs as peer_recovery_relaunch_envs
from .launch.fleet import (
    CONFIG_ERROR_EXIT, CONTAINER_CHANGED_EXIT, FAILED_RUN_DRAIN_S,
    ISLAND_FAILED_EXIT, RUN_STALLED_EXIT, RUN_STOPPED_EXIT, TEARDOWN_UNVERIFIED_EXIT,
    TRAINER_GAVE_UP_EXIT, FixedRosterIslandAbandoned, FleetController, IslandConfigError,
    RunStalled, RunStopRequested, controller_fixed_roster, effective_stall_timeout,
)
from .launch import trips
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
    rl_syncer_profile, syncer_command, syncer_disk_resources, syncer_ports, syncer_tape_sidecar, tcp_probe,
    ufw_setup,
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
from .launch.harness_env import (  # noqa: F401
    check_compaction_launch, compactionrl_launch_env, modal_sandbox_secret_name,
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
from .launch.manifest import _island_tasks_manifest  # noqa: F401


# Rough per-GPU training capacity sanity check (bf16 LoRA, GB).
GPU_MEM_GB = {"A100": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 180, "B300": 288, "L4": 24, "A10G": 24, "T4": 16, "V100": 16, "L40S": 48, "RTX-PRO-6000": 96, "RTX-6000-Ada": 48}


def _rollout_controller(args, *, ops=None, **kw):
    """The :class:`~yeto.launch.pool.controller.RolloutController` of ``--rollout-pool``
    (every pool config in one controller, each built like its ``nc pool`` loop), None
    without the flag. The fleet controller starts, checks and drains it."""
    paths = [os.path.expanduser(p) for p in (getattr(args, "rollout_pool", None) or [])]
    if not paths:
        return None
    from .launch.nc import cli as nc_cli
    from .launch.pool.controller import RolloutController, nc_pool

    slots = {}
    try:
        for path in paths:
            with open(path) as fh:
                cfg = json.load(fh)
            name = str(cfg.get("shard_base") or path)
            if name in slots:
                raise SystemExit(f"--rollout-pool: two pools share shard_base {name!r}")
            slots[name] = nc_pool(cfg, path, ops=(ops or {}).get(name))
    except BaseException:
        for slot in slots.values():
            if slot.lock is not None:
                slot.lock.close()
        raise
    interval = float(getattr(args, "controller_poll", 30) or 30)
    nc_cli._log(f"rollout: {len(slots)} pool(s) in the launcher's controller ({', '.join(slots)})")  # noqa: SLF001
    return RolloutController(slots, interval=interval, **kw)


def _follower_islands(args, specs, learner_names) -> list[str]:
    """Cluster names of a synced group's sampler islands that run until their
    trainers end (the fleet controller ends them once the trainers are done)."""
    if getattr(args, "training_mode", "sft") != "rl":
        return []
    from .rl.engine import sample_roles

    if not sample_roles.is_synced_group(args):
        return []
    stops = sample_roles.sampler_stop_trainers(args, len(specs))
    return [learner_names[m] for m, n in sorted(stops.items()) if n and m < len(learner_names)]


def _island_overrides(args, specs) -> dict:
    """launch-preflight-guards 3.2: validated per-island overrides ({} normally)."""
    from .island_overrides import overrides_of
    from .launch_preflight import check_compat_groups, check_policy_age_spec

    check_policy_age_spec(args)  # dry-run reaches here without pre_cloud_checks
    check_compat_groups(args, specs)
    return overrides_of(args, len(specs))


def _island_engine_args(args, m):
    """hetero-1114 A22: island m's own image / backend (``--rl-island-engine``),
    None without the flag (callers then read the run-wide ``args``)."""
    from .rl.engine import island_engine

    if getattr(args, "training_mode", "sft") != "rl" or not island_engine.requested(args):
        return None
    return island_engine.island_args(args, m, rl_island_count(args))


def _island_engine_rows(args, m) -> list[dict]:
    from .rl.engine import island_engine

    if not island_engine.requested(args):
        return []
    return [r for r in island_engine.records(args, rl_island_count(args)) if r["island"] == m]


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
    # the island's sample role; it runs as learner 0 of 1 (no syncer).
    iargs = sample_roles.island_args(iargs, m, rl_island_count(args))
    learner_id, island_learners = sample_roles.island_learner(args, m, num_learners, rl_island_count(args))
    task = task_factory(iargs, spec, learner_id, island_learners, syncer_addr)
    from .launch.modal_islands import parse_island_env

    io.apply_to_task(task, {**parse_island_env(args), **io.island_env(args, m, overrides),
                            **(droppable.role_env(role, spec.cloud, spec.region) if role else {}),
                            **sample_roles.island_env(iargs),
                            **_wait_heartbeat_env(args, iargs),
                            **_spot_sampler_cloud_env(iargs, getattr(spec, "cloud", None)),
                            **_resume_marker_env(iargs, getattr(spec, "cloud", None))})
    return task


def _wait_heartbeat_env(args, iargs) -> dict[str, str]:
    """A buffer-trainer island (it reads the sample buffer): the interval of its wait
    heartbeat, derived from this launch's stall clock (``trainer_breakers.wait_heartbeat_env``)."""
    from .launch.fleet import effective_stall_timeout
    from .rl.engine import sample_roles, trainer_breakers

    if sample_roles.sample_role(iargs) not in (sample_roles.TRAINER, sample_roles.CONVERTIBLE):
        return {}
    return trainer_breakers.wait_heartbeat_env(effective_stall_timeout(args))


def _resume_marker_env(iargs, cloud: str | None) -> dict[str, str]:
    """A resumable island on Modal (``--rl-checkpoint-store`` without
    ``--rl-elastic``: the island's ``--rl-resume-store``): the reclaim marker path.
    Modal signals only the container's main process (the runner), so
    the runner writes the marker and the driver's watcher thread -- in a Ray actor
    on verl -- writes the store's preemption notice."""
    if (cloud != "modal" or not getattr(iargs, "rl_checkpoint_store", None) or getattr(iargs, "rl_elastic", False)
            or getattr(iargs, "rl_checkpoint_store_export_only", False)):  # no resume store
        return {}
    from .rl.engine.sampler_resume import DEFAULT_RECLAIM_MARKER, RECLAIM_MARKER_ENV

    return {RECLAIM_MARKER_ENV: DEFAULT_RECLAIM_MARKER}


def _spot_sampler_cloud_env(iargs, cloud: str | None) -> dict[str, str]:
    """A spot sampler island on Nebius: the VM stop reaches the sampler through the
    host unit of ``yeto.cloud.nebius_reclaim`` (Modal signals the container itself)."""
    from .rl.engine import sample_roles

    if cloud != "nebius" or not getattr(iargs, "spot", False) or not sample_roles.island_env(iargs):
        return {}
    from .cloud import nebius_reclaim

    return nebius_reclaim.spot_sampler_env()


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


def _relaunch_of_same_run(args) -> bool:
    """True when this cluster prefix already has a run manifest (an earlier launch of the
    same run: a resume or a restart), so its old event tapes belong to this run."""
    from .launch.code_pack import _previous_manifest

    return _previous_manifest(args.cluster_prefix) is not None


def island_lifecycle_enabled(args, specs) -> bool:
    """Whether this launch records its islands in the run store and honors each island's
    own STOP (``yeto.launch.island_lifecycle``; ``--island-lifecycle``). ``auto`` (the
    default): an RL launch of two or more trainer islands (an island fleet); a launch of
    one trainer island records nothing, reads nothing and behaves as before. ``on``: any
    launch with a run store; ``off``: none."""
    mode = getattr(args, "island_lifecycle", None) or "auto"
    if mode == "off" or getattr(args, "dry_run", False):
        return False
    if mode == "on":
        return True
    if getattr(args, "training_mode", "sft") != "rl":
        return False
    from .rl.engine import sample_roles

    samplers = set(sample_roles.sampler_islands(args, len(specs)))
    return sum(1 for m in range(len(specs)) if m not in samplers) >= 2


def island_outcome(exit_codes) -> tuple[list[str], list[str], list[str]]:
    """The islands that succeeded, those ended on their own island STOP (a stop by the
    decision table: neither a success nor a failure; the fleet stopped only when every
    island did) and those that failed, by the controller's exit strings."""
    from .launch.fleet import ISLAND_STOPPED_PREFIX

    done = [n for n, s in exit_codes.items() if "SUCCEEDED" in str(s)]
    stopped = [n for n, s in exit_codes.items()
               if n not in done and str(s).startswith(f"{ISLAND_STOPPED_PREFIX} (island STOP")]
    failed = [n for n in exit_codes if n not in done and n not in stopped]
    return done, stopped, failed


def _island_refs(args, specs, names, *, modal_names=(), ssh_names=(), events_dir=None, echo_names=()) -> dict:
    """Launch island name -> :class:`~yeto.launch.island_lifecycle.IslandRef` (learner id =
    the island's position in ``--gpu``; ``tape``: its echoed event tape, when it has one)."""
    from .launch.island_lifecycle import IslandRef
    from .rl.engine import sample_roles

    samplers = set(sample_roles.sampler_islands(args, len(specs)))
    out = {}
    for m, (name, spec) in enumerate(zip(names, specs)):
        source = ("modal" if name in modal_names else "ssh" if name in ssh_names
                  else "verda" if spec.cloud == "verda" else "sky")
        tape = str(Path(events_dir) / f"{name}.jsonl") if events_dir is not None and name in echo_names else None
        out[name] = IslandRef(m, source, "sampler" if m in samplers else "trainer", tape)
    return out


def _adopt_probes(args) -> dict:
    """The source probes of a relaunch's adoption plan (``yeto.launch.island_adopt``)."""
    from .launch.island_adopt import ModalProbe, SkyProbe
    from .modal_runner import ModalOps, modal_app_name

    modal_ops = ModalOps(modal_app_name(args.cluster_prefix))
    return {"sky": SkyProbe(args.cluster_prefix), "verda": SkyProbe(args.cluster_prefix),
            "modal": ModalProbe(modal_ops.status)}


def _adopt_syncer(states, cluster: str, adoption: dict):
    """``(job id, head ip)`` of the syncer cluster an earlier launcher of this run started,
    when an island is adopted and that syncer still runs (its record ``fleet/syncer.json``,
    the cluster UP, its job alive); else None (the syncer is launched as usual)."""
    if states is None or not any(a.adopt for a in adoption.values()):
        return None
    record = states.read_syncer(strict=True) or {}
    if record.get("cluster") != cluster or record.get("job_id") is None or not record.get("head_ip"):
        return None
    from .launch.island_adopt import AdoptionUnknown, SkyProbe

    try:
        running, why = SkyProbe(cluster)({"cluster": cluster, "job_id": record["job_id"]})
    except Exception as e:  # noqa: BLE001 - unknown is not "gone": a second syncer would split the fleet
        raise AdoptionUnknown(f"syncer {cluster}: whether its job {record['job_id']} still runs is "
                              f"unknown ({type(e).__name__}: {e})"[:400]) from e
    if not running:
        print(f"[launcher] syncer {cluster}: not adopted ({why}); launching it", file=sys.stderr)
        return None
    print(f"[launcher] syncer {cluster}: adopted ({why}); no new syncer job", file=sys.stderr)
    return record["job_id"], str(record["head_ip"])


def _record_ordering(recorder, name: str, adoption: dict) -> None:
    """The island's record before its order: ``ordering``, its cluster name and incarnation,
    so a launcher that dies mid-order leaves the machine findable."""
    if recorder is None:
        return
    recorder.store.guard_order(name)  # the lock still ours, checked now (raises: nothing ordered)
    a = adoption.get(name)
    try:
        recorder.record(name, "ordering", handle={"cluster": name},
                        incarnation=0 if a is None else a.incarnation,
                        relaunches=0 if a is None else a.relaunches, reason="ordering")
    except Exception as e:  # noqa: BLE001 - a view only
        print(f"[launcher] island record of {name}: {e}", file=sys.stderr)


def _record_call(recorder, name: str, call_id: str, adoption: dict) -> None:
    """A Modal island's function call id, persisted the moment it is spawned (a relaunch's
    record carries the new call id once the controller sees it running)."""
    try:
        recorder.record(name, None, handle={"cluster": name, "call_id": str(call_id)})
    except Exception as e:  # noqa: BLE001 - a view only
        print(f"[launcher] island record of {name}: {e}", file=sys.stderr)


ISLAND_RECORDS_LOCK = "fleet/trainers/lock"  # the island records' single-writer lock (StoreLock)
ISLAND_RECORDS_LOCK_TTL_S = 120.0  # a dead launcher's lease runs out this long after its last renewal
ISLAND_RECORDS_LOCKED_EXIT = 12  # another live launcher holds the island records of this run
ISLAND_RECORDS_UNKNOWN_EXIT = 13  # an island record, or whether its machine runs, did not read
# exits after which nothing of the run may be torn down: another launcher (may) supervise it
HANDS_OFF_EXITS = frozenset({ISLAND_RECORDS_LOCKED_EXIT, ISLAND_RECORDS_UNKNOWN_EXIT})


def _plan_island_records(args, specs, names, island_states, island_lock, *, relaunch, ssh_names, events_dir,
                         echo_names, held_name, syncer_cluster, two_cluster_names=()):
    """Under the island records' lock: the records (read strictly), the fleet recorder and
    the adoption plan of a restart, and the syncer to keep. Returns ``(recorder, adoption,
    kept_syncer)``. Raises ``IslandRecordsUnreadable`` / ``island_adopt.AdoptionUnknown``
    when a record or a machine's state does not read: nothing may be ordered then."""
    from .launch import island_adopt
    from .launch.island_lifecycle import FleetRecorder

    # Adoption goes by the run store's records (the local run manifest is only a cache: a
    # launcher restarted on another machine finds the islands there too).
    records = island_states.read_all(range(len(specs)), strict=True)  # also what the writes merge into
    restart = relaunch or any(r.get("state") in island_adopt.ADOPTABLE for r in records.values())
    if not restart:
        for m in range(len(specs)):  # a new run: an island STOP an earlier run of this name left is void
            island_states.clear_stop(m)
    refs = _island_refs(args, specs, names,
                        modal_names={n for n, sp in zip(names, specs) if sp.cloud == "modal"},
                        ssh_names=ssh_names, events_dir=events_dir, echo_names=echo_names)
    refs.pop(held_name, None)  # the trainer scheduler's island has its own journal
    recorder = FleetRecorder(island_states, refs)
    adoption: dict = {}
    if restart and island_lock is not None:  # adopt only under the lock
        adoption = island_adopt.plan(records, refs, probes=_adopt_probes(args))
    two_cluster = sorted(n for n, a in adoption.items() if a.adopt and n in two_cluster_names)
    if two_cluster:
        raise island_adopt.AdoptionUnknown(
            f"two-cluster island(s) {two_cluster} still run; a two-cluster island is not adopted (its engine "
            "cluster has no record): take it down (yeto down) or let it end, then launch again")
    kept = None
    if syncer_cluster is not None:
        kept = _adopt_syncer(island_states, syncer_cluster, adoption)
    return recorder, adoption, kept


def _lost_island_records(island_states) -> bool:
    """This launcher no longer holds the island records (checked now): another launcher
    supervises the fleet, so this one tears nothing of it down."""
    from .launch.island_lifecycle import IslandRecordsFenced, IslandRecordsUnconfirmed

    if island_states is None or island_states.fence is None:
        return False
    try:
        island_states.guard_order("teardown")
    except IslandRecordsFenced:
        return True
    except IslandRecordsUnconfirmed as error:
        # not confirmed within this holder's lease: still ours, it tears down; past it: another
        # launcher may hold the records, nothing is torn down
        return error.lease_over
    return False


def _island_records_lock(args) -> str:
    """The single-writer lock of this launch's island records: the run's
    (:data:`ISLAND_RECORDS_LOCK`), or, for an island launch of a fleet launched island by
    island (fleet role ``island``, learner id K: ``args.rl_fleet_role``, ``args.rl_fleet_learner_id``), its own island's
    (``fleet/trainers/<k>/lock``): each island launch writes only its own record, next to
    the other islands' launches, and none waits for or fences another."""
    if getattr(args, "rl_fleet_role", None) == "island" and getattr(args, "rl_fleet_learner_id", None) is not None:
        return f"fleet/trainers/{int(args.rl_fleet_learner_id)}/lock"
    return ISLAND_RECORDS_LOCK


def _claim_island_records(states, *, prefix: str = ISLAND_RECORDS_LOCK, wait_s: float | None = None,
                          sleep=None, clock=None, others=()):
    """Claim the run store's single-writer lock on the island records
    (``yeto.launch.pool.store_lock.StoreLock`` under ``prefix``: :func:`_island_records_lock`)
    and make it the store's fence; a lease that runs (a launcher that just died, or a live
    one) is waited on up to ``wait_s``. ``others``: lock prefixes that exclude this one (a
    whole-fleet launch and the island-by-island launches of one run): one held live refuses
    the claim. Returns ``(lock, None)``; ``(None, why)`` when a live launcher keeps it;
    ``(None, None)`` on a store without an atomic create (records are kept, nothing is
    adopted)."""
    from .launch.pool.store_lock import StoreLock, StoreLockError

    sleep, clock = sleep or time.sleep, clock or time.time
    deadline = clock() + (ISLAND_RECORDS_LOCK_TTL_S + 30.0 if wait_s is None else float(wait_s))
    while True:
        try:
            lock = StoreLock.acquire(states.objects, prefix, ttl_s=ISLAND_RECORDS_LOCK_TTL_S, clock=clock)
        except StoreLockError as error:
            print(f"[launcher] island records: no single-writer lock on this store ({error}); records are "
                  "kept, no island is adopted", file=sys.stderr)
            return None, None
        if lock is not None:
            try:
                busy = [o for o in others if StoreLock.holder_of(states.objects, o, now=clock()) is not None]
            except BaseException:
                lock.release()  # the other kind's locks did not read: this claim is given back
                raise
            if busy:
                lock.release()
                return None, (f"another launcher of this run holds its island records under {busy[0]} (a "
                              "whole-fleet launch and island-by-island launches of one run exclude each other)")
            states.fence = lock
            print(f"[launcher] island records: lock {prefix} token {lock.token} held by this launcher",
                  flush=True)
            return lock, None
        holder = StoreLock.holder_of(states.objects, prefix, now=clock()) or {}
        now = clock()
        if now >= deadline:
            return None, (f"another launcher holds the island records of this run (lock token "
                          f"{holder.get('token')}, {holder.get('holder')}, lease until "
                          f"{time.strftime('%H:%M:%S', time.gmtime(float(holder.get('until') or now)))} UTC)")
        until = float(holder.get("until") or now)
        print(f"[launcher] island records: lock token {holder.get('token')} held by {holder.get('holder')} "
              f"(lease {max(0.0, until - now):.0f} s left); waiting for it", flush=True)
        sleep(max(1.0, min(10.0, until - now + 1.0, deadline - now)))


def _island_state_store(args):
    """The run store's island records (None: no object API for the store, or none)."""
    from .launch.island_lifecycle import open_state_store
    from .launch.storage import store_client_environ

    store = getattr(args, "rl_checkpoint_store", None)
    if not store:
        print("[launcher] --island-lifecycle: no --rl-checkpoint-store; island records are not kept",
              file=sys.stderr)
        return None
    # background: the fleet controller's poll loop never waits on the store
    states = open_state_store(store, store_client_environ(args), background=True)
    if states is None:
        print(f"[launcher] --island-lifecycle: the store {store} has no object API here; island records "
              "are not kept", file=sys.stderr)
    return states


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
    relaunch = _relaunch_of_same_run(args)  # before this launch writes its own manifest
    import sky

    from .launch.island_lifecycle import IslandRecordsFenced, IslandRecordsUnconfirmed

    hands_off_exit = None  # set once this launcher must not tear the run down (exit 12 / 13)

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
    # The run code pack (first start makes it; later starts reuse it).
    prepare_run_code_pack(args)
    # 2.3 / 3.1 / 3.6: memory estimate, island overrides, negative-test resume guard.
    launch_preflight.pre_cloud_checks(args, threads=False)
    trainer_loop.check(args)  # Refuse before any launch
    from .launch import ssh_resident

    ssh_resident.check(args, parse_gpu_spec(args.gpu))  # resident ssh: islands, before any cloud work
    _write_run_manifest(args)
    head_mode = local_syncer is not None
    # --rl-single-island-no-sync: one ports island, no syncer at all
    # (validated in _prepare_ports_algorithm before any cloud work).
    no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
    if no_sync and head_mode:
        raise ValueError("--rl-single-island-no-sync has no syncer; do not run it as a head")
    specs = parse_gpu_spec(args.gpu)
    nodes_by_cluster = dict(zip(learner_cluster_names(args.cluster_prefix, specs),
                                (s.num_nodes for s in specs)))
    # two-cluster islands ({} without --rl-engine-cluster: nothing changes)
    splits = engine_cluster.plan(
        args, specs, learner_cluster_names(args.cluster_prefix, specs),
        role_of=lambda m: _island_role(args, specs, m)) if engine_cluster.requested(args) else {}
    if splits:
        nodes_by_cluster.update(engine_cluster.nodes_by_cluster(splits))
    mixed: dict[str, engine_cluster.MixedIsland] = {}
    no_sync_incomplete: list[str] = []
    # The trainer scheduler starts and stops this island
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
    prefix = args.cluster_prefix
    syncer_cluster = None if head_mode or no_sync else sky_cluster_name(f"{prefix}-syncer")
    learner_names = learner_cluster_names(prefix, specs)
    ssh_names = _resident_ssh_names(args, learner_names, specs)
    # yeto.launch.island_lifecycle: each trainer island's record and STOP in the run store
    # (an island fleet by default; None for a single trainer island: nothing changes). One
    # launcher writes them: the run store's single-writer lock is claimed first (a dead
    # launcher's lease is waited out), before anything of the run is registered, moved or
    # ordered; held by a live launcher, nothing is launched. The lease is renewed from here on.
    island_states = _island_state_store(args) if island_lifecycle_enabled(args, specs) else None
    island_lock = None
    recorder = None
    adoption: dict = {}  # name -> island_adopt.Adoption (a relaunch of the same run)
    kept_syncer = None  # (job id, head ip) of an adopted syncer
    if island_states is not None:
        from .launch.island_adopt import AdoptionUnknown
        from .launch.island_lifecycle import IslandRecordsUnreadable

        # From the claim until this launcher supervises the fleet (the try below), any way out
        # is hands off: another launcher may hold the records, or this one is about to adopt
        # the islands its ledger still lists (a backstop cleanup would tear them down).
        args._hands_off_exit = ISLAND_RECORDS_UNKNOWN_EXIT
        lock_prefix = _island_records_lock(args)
        island_lock, refusal = _claim_island_records(
            island_states, prefix=lock_prefix,
            others=[ISLAND_RECORDS_LOCK] if lock_prefix != ISLAND_RECORDS_LOCK else
            [f"fleet/trainers/{k}/lock" for k in range(len(specs))])
        if refusal is not None:
            args._hands_off_exit = ISLAND_RECORDS_LOCKED_EXIT
            island_states.close()
            print(f"[launcher] ERROR: {refusal}; nothing is launched and nothing of the run is torn down "
                  f"(exit {ISLAND_RECORDS_LOCKED_EXIT})", file=sys.stderr)
            return ISLAND_RECORDS_LOCKED_EXIT
        island_states.keep_lease()
        try:
            recorder, adoption, kept_syncer = _plan_island_records(
                args, specs, learner_names, island_states, island_lock, relaunch=relaunch, ssh_names=ssh_names,
                events_dir=events_dir, echo_names=echo_names, held_name=held_name, syncer_cluster=syncer_cluster,
                two_cluster_names=set(splits))
        except (IslandRecordsUnreadable, AdoptionUnknown) as error:
            island_states.close()
            if island_lock is not None:
                island_lock.release()
            print(f"[launcher] ERROR: {error}; nothing is launched and nothing of the run is torn down "
                  f"(exit {ISLAND_RECORDS_UNKNOWN_EXIT}): launch again once the store and the cloud answer",
                  file=sys.stderr)
            return ISLAND_RECORDS_UNKNOWN_EXIT
    adopted_names = {name for name, a in adoption.items() if a.adopt}
    if echo_names:
        # Refuse to mix runs in one tape, before anything is provisioned. An adopted island's
        # tape is continued (its trainer keeps running): it stays in place.
        existing = sorted(str(p) for p in events_dir.glob("*.jsonl*")
                          if p.name.split(".jsonl")[0] not in adopted_names) if events_dir.exists() else []
        if existing and relaunch:
            # a relaunch of this run (resume, or a restart after a failed start): the
            # earlier attempt's tapes move aside, so the new incarnation's tape is clean
            prev = events_dir / "previous" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            prev.mkdir(parents=True, exist_ok=True)
            for old in existing:
                Path(old).rename(prev / Path(old).name)
            print(f"[launcher] relaunch of run {args.cluster_prefix!r}: {len(existing)} earlier event "
                  f"tape(s) moved to {prev}", flush=True)
            existing = []
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
    from .rl.engine import sample_roles

    # the sampler islands of a synced group are no learners of the syncer
    num_learners = sample_roles.syncer_learner_count(args, len(specs)) + external
    require_modal_for_gpu_exact(args, specs)
    warn_if_model_wont_fit(args, specs)
    if on_clusters is not None:
        try:
            on_clusters(([] if syncer_cluster is None else [syncer_cluster]) + learner_names
                        + [s.engine_name for s in splits.values()])
        except Exception as e:
            print(f"[launcher] on_clusters hook failed: {e}", file=sys.stderr)
    clusters: list[str] = []
    controller = None
    teardown_unverified = False
    run_started_unix = time.time()  # Modal rows created after this are this run's app
    # Every billed resource of this run goes into its crash-safe ledger; the cost guard
    # (budget breaker, idle shutdown, NC lease expiry) runs until the teardown below.
    cost_ledger, cost_guard = _start_cost_guard(args, events_dir)
    verda_names = [n for n, sp in zip(learner_names, specs) if sp.cloud == "verda"]
    verda = prepare_verda_islands(verda_names, on_instance_ids) if verda_names else None
    # Read by the finally block (Modal stop / tape pull) even on an early failure.
    modal_ops = None
    modal_island_ops = None
    rids: dict = {}  # the islands submitted so far (named by an exit 13 at an order)
    modal_cfgs: dict[str, object] = {}
    # Resident ssh: trainer islands (yeto.launch.ssh_resident): the fleet supervises them
    # over SSH; the finally block releases them and verifies every node.
    ssh_islands = None
    # The log streams and their echo-tape collectors (an ssh: island streams from its start).
    tail_threads: list[threading.Thread] = []
    event_collectors: dict[str, EventCollector] = {}

    try:
        if island_states is not None:  # supervising from here on: the finally decides hands off
            args._hands_off_exit = None
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
            syncer_task = make_syncer_task(args, num_learners)
            if kept_syncer is not None:  # the adopted islands keep talking to the syncer that still runs
                syncer_job, head_ip = kept_syncer
            else:
                if island_states is not None:
                    island_states.guard_order(syncer_cluster)  # the lock still ours (raises: nothing ordered)
                print(f"[launcher] launching syncer cluster {syncer_cluster} in {args.syncer_region}")
                cost_ledger.created("sky_cluster", syncer_cluster, syncer_cluster, role="syncer")
                rid = sky.launch(syncer_task, cluster_name=syncer_cluster)
                syncer_job, syncer_handle = sky.stream_and_get(rid)
                head_ip = str(syncer_handle.head_ip)
                if island_states is not None:
                    island_states.write_syncer(cluster=syncer_cluster, job_id=syncer_job, head_ip=head_ip)
            clusters.append(syncer_cluster)
            syncer_addr = syncer_shard_addrs(head_ip, args)
            print(f"[launcher] syncer up at {syncer_addr}")
            probe_syncer_ports(args, head_ip)
        if not no_sync and island_pool(args) \
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
        # Each island goes to the adapter of its source (yeto.launch.trainer_provision):
        # sky / Verda clusters are ordered during the walk and waited on in threads, the
        # Modal islands deploy on one app and the resident ssh: islands start after it.
        from .launch.trainer_provision import TrainerProvision

        provision = TrainerProvision(
            args, sky, specs, learner_names, ssh_names=ssh_names, syncer_addr=syncer_addr, no_sync=no_sync,
            engine_args=lambda m: _island_engine_args(args, m), verda=verda,
            on_order=lambda name: _record_ordering(recorder, name, adoption),
            on_call=None if recorder is None else (
                lambda name, call_id: _record_call(recorder, name, call_id, adoption)),
            adopted=lambda name: (adoption[name].job_id,) if name in adoption and adoption[name].adopt else None,
            guard_order=None if island_states is None else island_states.guard_order,
            ledger=cost_ledger, events_dir=events_dir, collectors=event_collectors)
        island_overrides = _island_overrides(args, specs)
        for m, spec in enumerate(specs):
            name = learner_names[m]
            role = _island_role(args, specs, m)
            task = _island_task(task_factory, args, spec, m, num_learners, syncer_addr,
                                island_overrides, role)
            island = provision.island(name, m, spec, role, task)
            if name == held_name:
                held_cfg, held_task = provision.hold(island)  # the scheduler starts it from these
                print(f"[launcher] {name} is the trainer island: the trainer scheduler "
                      "starts and stops it", flush=True)
                continue
            tasks[name] = island.task
            provision.add(island)
            if not provision.ordered(island):
                continue  # started with the other islands of its source after the walk
            provision.announce(island)
            if name in splits:
                split_tasks = engine_cluster.build_tasks(
                    sky, args, splits[name],
                    lambda m=m, spec=spec, role=role: _island_task(
                        task_factory, args, spec, m, num_learners, syncer_addr, island_overrides, role))
                mixed[name] = engine_cluster.MixedIsland(
                    splits[name], split_tasks,
                    submit=lambda s_, t_, n_: submit_island_launch(sky, args, s_, t_, n_))
                if island_states is not None:  # its engine relaunches are orders too
                    mixed[name].before_order = island_states.guard_order
                tasks[name] = split_tasks.trainer  # the fleet's record: the trainer cluster
                _record_ordering(recorder, name, adoption)
                rids[name] = (m, ("mixed", mixed[name].submit()))
                continue
            if name in adoption and adoption[name].adopt:
                # its record names a cluster and a job that still run: no order, no restart
                rids[name] = (m, ("adopted", adoption[name].job_id))
                continue
            rids[name] = (m, ("order", provision.order(island)))
        modal_cfgs.update({n: i.task for n, i in provision.batch("modal").items()})

        results = provision.results
        errors = provision.errors

        def resolve(name: str, m: int, rid) -> None:
            kind, handle = rid
            try:
                if kind == "adopted":
                    results[name] = (handle, None)
                elif kind == "mixed":
                    results[name] = mixed[name].finish(sky, handle)
                else:
                    results[name] = provision.wait(provision.islands[name], handle)
            except Exception as e:
                errors[name] = e

        threads = [
            threading.Thread(target=resolve, args=(n, m, r), daemon=True)
            for n, (m, r) in rids.items()
        ]
        for t in threads:
            t.start()
        modal_ops = provision.modal_app()  # bound first: a deploy that fails is still stopped
        if modal_ops is not None:
            modal_island_ops = provision.spawn_modal(modal_ops)
        ssh_islands = provision.start_ssh()
        if ssh_islands is not None:
            if recorder is not None:  # the journal decided: reattached (adopted) or a new incarnation
                from .launch.island_adopt import Adoption

                for name in ssh_names:
                    if name in results and name in recorder.islands:
                        known = adoption.get(name)
                        incarnation = int(results[name][0])
                        adopted = name in ssh_islands.reattached
                        relaunches = 0 if known is None else known.relaunches + (0 if adopted else 1)
                        adoption[name] = Adoption(name, recorder.islands[name].island, adopted, incarnation,
                                                  relaunches if known is not None and known.why != "no record"
                                                  else 0, job_id=incarnation, cluster=name, why="ssh journal")
        for t in threads:
            t.join()
        for name in list(rids) + list(modal_cfgs) + sorted(ssh_names):
            if name not in errors:
                clusters.append(name)
        for name, (_m, (kind, _handle)) in rids.items():  # sky clusters (a failed launch may still have left one up)
            # an ordered cluster carries what prices it (the unified table's row, as a sky sampler's)
            cost_ledger.created("sky_cluster", name, name,
                                **(provision.ledger_attrs(provision.islands[name]) if kind == "order" else {}))
        for split in splits.values():  # both clusters may exist even after an error
            for name in (split.name, split.engine_name):
                if name not in clusters:
                    clusters.append(name)
        if errors:
            for name, e in errors.items():
                print(f"[launcher] ERROR launching {name}: {e}", file=sys.stderr)
            raise RuntimeError(f"{len(errors)} learner cluster(s) failed to provision")

        # 3. Stream logs while the fleet controller polls health, recovers
        #    failed/preempted clusters, and abandons learners that stay down
        #    past --recover-timeout.
        def spawn_tail(name: str, job_id) -> None:
            if ssh_islands is not None and ssh_islands.owns(name):
                # node 0's stream already feeds the collector (ssh_resident sink)
                tail_threads.extend(ssh_islands.threads(name))
                return
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
                modal_id_guards[name] = guard
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

        container_guards: list = []
        modal_id_guards: dict = {}  # Modal island name -> its ContainerIdGuard

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
            if name in mixed and mixed[name].engine_job is not None:
                spawn_tail(mixed[name].split.engine_name, mixed[name].engine_job)
        def attributed_containers() -> dict:
            # container id -> (island, rank) from each island's first container line
            return {cid: key for g in list(container_guards)
                    for key, cid in dict(getattr(g, "first", None) or {}).items()}

        container_poller = container_set_poller_for(
            args, modal_ops, modal_cfgs, on_container_change, attributed=attributed_containers,
            # a finalized island's container leaving the app is its normal end
            finished=lambda: finalized_island_containers(modal_id_guards, event_collectors))
        if container_poller is not None:
            container_guards.append(container_poller)

        def route_container(learner_id: int):
            # the island of a rescheduled container's lines (unique learner id only:
            # the islands of a pair all run as learner 0)
            names = [n for n, cfg in modal_cfgs.items() if int(cfg.learner_id) == int(learner_id)]
            if len(names) != 1:
                return None
            return names[0], (event_collectors.get(names[0]) if names[0] in echo_names else None)

        # daemon threads; each container's log ends when the app stops
        container_log_follower_for(args, modal_ops, modal_cfgs, route_container, poller=container_poller)

        from . import runs
        from .modal_runner import RoutingOps

        sky_sdk_ops = SkySDKOps()
        # the failure-path teardown verifies every node instance
        sky_sdk_ops.nodes_by_cluster = dict(nodes_by_cluster)
        controller = FleetController(
            learners={name: (tasks[name], job_id) for name, (job_id, _h) in results.items()},
            syncer=None if syncer_cluster is None else (syncer_cluster, syncer_task, syncer_job),
            sky_ops=RoutingOps(
                engine_cluster.EngineClusterOps(sky_sdk_ops, mixed, sky) if mixed else sky_sdk_ops,
                modal_island_ops, ssh_ops=ssh_islands),
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
            stall_timeout=effective_stall_timeout(args),
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
            awaiting_syncer_probe=(
                (lambda name: name in event_collectors and event_collectors[name].awaiting_syncer)
                if echo_names else None
            ),
            elastic=island_pool(args),
            no_island_relaunch=bool(getattr(args, "no_island_relaunch", False)),
            wan_reject_probe=(
                (lambda name: event_collectors[name].wan_rejects if name in event_collectors else 0)
                if echo_names else None
            ),
            wan_probe_relaunches=wan_probe_relaunches(args),
            island_relaunch_envs=peer_recovery_relaunch_envs(args),
            followers=_follower_islands(args, specs, learner_names),
            rollout=_rollout_controller(args),
            recorder=recorder,
            incarnations={name: {"incarnation": a.incarnation, "relaunches": a.relaunches, "adopted": a.adopt}
                          for name, a in adoption.items()},
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
                    modal_tape_dir=modal_tape_dir,
                    recover_sky=lambda name, c: (ssh_islands.recover_tape(name, c)
                                                 if ssh_islands is not None and ssh_islands.owns(name)
                                                 else _recover_echo_tape(name, c))))
                _record_trainer_start(args, events_dir)

        # a supervisor per two-cluster island relaunches only its
        # engine cluster when the engine run is lost (none without the recover flag)
        engine_watch_stop = threading.Event()
        if engine_cluster.recover_s(args) > 0:
            for name, island in mixed.items():
                job = results[name][0]
                threading.Thread(
                    target=island.watch, args=(sky, engine_watch_stop),
                    kwargs={"trainer_alive": lambda name=name, job=job: sky_sdk_ops.job_alive(name, job)},
                    name=f"yeto-engine-watch-{name}", daemon=True).start()
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
            limit = trainer_loop.max_failures(args)
            trips.trip(args.cluster_prefix, "launch.trainer_failures", limit, limit,
                       legacy_event="TrainerLoopGaveUp", detail=str(error))
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT if container_changed() else TRAINER_GAVE_UP_EXIT
        except RunStalled as error:
            print(f"[launcher] ERROR: {error}; stopping the run (exit {RUN_STALLED_EXIT})",
                  file=sys.stderr)
            trips.trip(args.cluster_prefix, "launch.stall", error.silent_s, error.timeout_s,
                       legacy_event="RunStalled", detail=str(error))
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT if container_changed() else RUN_STALLED_EXIT
        except RunStopRequested as error:
            # yeto stop-run: the trainer loop has stopped its trainer (cut) on the way
            # out; secure the tapes (bounded), then the finally block tears down
            print(f"[launcher] {error}: stopping the run (exit {RUN_STOPPED_EXIT})",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return RUN_STOPPED_EXIT
        except IslandConfigError as error:
            # An island refused its launch options: every relaunch would fail the same
            # way. Secure the tapes (bounded), then the finally block tears down.
            print(f"[launcher] ERROR: {error}; stopping the run (exit {CONFIG_ERROR_EXIT}): fix "
                  "the launch options, then launch again", file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT if container_changed() else CONFIG_ERROR_EXIT
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
        if (no_sync_incomplete and head_mode and island_pool(args)):
            # an island the syncer removed from the pool for an excused
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
        done, stopped, failed = island_outcome(exit_codes)
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
        if not done and stopped and not failed:
            # every island ended on its own STOP (yeto.launch.island_lifecycle): the
            # fleet stopped, as on the run's STOP flag
            print(f"[launcher] every island ended on its island STOP ({', '.join(stopped)}): the fleet "
                  f"stopped (exit {RUN_STOPPED_EXIT}); recover from the syncer checkpoint with "
                  "yeto-export", file=sys.stderr)
            return RUN_STOPPED_EXIT
        if not done:
            print("[launcher] no learner succeeded; recover from the syncer "
                  "checkpoint with yeto-export", file=sys.stderr)
            return 1
        rl_mode = getattr(args, "training_mode", "sft") == "rl"
        # Prefer a sky learner as the artifact source (rsync); a Modal
        # island's ~/yeto-output is not reachable that way.
        sky_done = [n for n in done if n not in modal_cfgs and n not in ssh_names]
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
        elif source in ssh_names:
            # A resident ssh: island (no-sync): its ~/yeto-output stays on the nodes; the
            # run's store holds its releases and the echo tape is the evidence.
            print(f"[launcher] the trainer ran on the resident ssh: island {source}; its releases are in "
                  "--rl-checkpoint-store and its event tape is the evidence", file=sys.stderr)
            if output:
                print(f"[launcher] --output {output} cannot be delivered from an ssh: island; exit 2",
                      file=sys.stderr)
                return 2
            return 1 if failed else 0
        elif source in modal_cfgs and no_sync:
            print(
                f"[launcher] --rl-single-island-no-sync island ran on Modal ({source}); "
                "its ~/yeto-output is not fetchable over ssh and there is no syncer "
                "checkpoint -- use the streamed island log / event tape as the evidence",
                file=sys.stderr,
            )
            # the run itself is judged by the island outcome (its tape
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
    except IslandRecordsFenced as error:
        # another launcher took the island records over: it supervises (and tears down) the fleet
        hands_off_exit = ISLAND_RECORDS_LOCKED_EXIT
        print(f"[launcher] ERROR: {error}; nothing of the run is torn down here "
              f"(exit {ISLAND_RECORDS_LOCKED_EXIT})", file=sys.stderr)
        return ISLAND_RECORDS_LOCKED_EXIT
    except IslandRecordsUnconfirmed as error:
        # the store did not confirm the lock at an order: another launcher may hold it now
        hands_off_exit = ISLAND_RECORDS_UNKNOWN_EXIT
        up = sorted(set(clusters) | set(rids) | set(
            (modal_island_ops.calls if modal_island_ops is not None else None) or {}))
        print(f"[launcher] ERROR: {error}; nothing of the run is torn down here "
              f"(exit {ISLAND_RECORDS_UNKNOWN_EXIT}): launch again once the store answers (the next launcher "
              f"adopts or takes down what is up: {up or 'nothing ordered yet'})", file=sys.stderr)
        return ISLAND_RECORDS_UNKNOWN_EXIT
    finally:
        if cost_guard is not None:
            cost_guard.stop()  # the teardown below releases everything; no forced stop mid-way
        # a launcher that lost the island records' lock tears nothing down: the holder owns the fleet
        if hands_off_exit is None and _lost_island_records(island_states):
            # 12: another launcher holds the records; 13: not confirmed past this holder's lease
            hands_off_exit = (ISLAND_RECORDS_LOCKED_EXIT if island_states.fenced is not None
                              else ISLAND_RECORDS_UNKNOWN_EXIT)
        hands_off = hands_off_exit is not None
        if hands_off:
            args._hands_off_exit = hands_off_exit  # the worker runs no backstop cleanup either
        if verda is not None and verda.get("pth_written"):
            from .sky_patches import remove_local_pth

            remove_local_pth(verda["pth_written"])
        for island in mixed.values():  # no engine relaunch during the teardown
            island.stop_watch = True
        # Clusters the controller already tore down (abandoned learners, or
        # the syncer after a total loss) are skipped — even with --keep.
        downed = controller.downed_clusters if controller is not None else set()
        # an ssh: island the controller released is released again: that call verifies it
        remaining = [c for c in clusters if c not in downed or c in ssh_names]
        # teardowns the controller skipped while its lock was not confirmed (a renamed relaunch
        # target is not always in clusters): torn down here when the end confirms the lock
        pending = controller.pending_downs if controller is not None else {}
        remaining += [c for c in sorted(pending) if c not in remaining]
        if modal_ops is not None and modal_cfgs:
            # Containers commit their tape every 30 s and at exit; pulled
            # here (before the app stop) and again after it below.
            pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
        if hands_off:
            print(f"[launcher] not tearing down {remaining} or the Modal app: another launcher holds this "
                  "run's island records and supervises them", file=sys.stderr)
        elif args.keep:
            print(f"[launcher] keeping clusters: {remaining}")
        else:
            unverified = []
            unverified_modal: list[str] = []
            collect_diagnostics_parallel(
                [n for n in remaining if n not in modal_cfgs and n not in ssh_names], head=head_mode
            )
            for island in mixed.values():  # two-cluster islands: peer rules first (never raises)
                island.unlink()
            for name in remaining:
                print(f"[launcher] tearing down {name}")
                if name in ssh_names:
                    # every node must show neither process nor container (else exit 5)
                    if ssh_islands is None or not ssh_islands.down(name):
                        unverified.append(name)
                        teardown_unverified = True
                    continue
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
                else:
                    cost_ledger.released("sky_cluster", name, name)
            # Sky islands without an echoed tape
            # (their tapes are in the pre-teardown diagnostics, also for islands
            # the controller tore down earlier) still give the manifest T_s.
            _record_sky_trainer_start(
                args, [n for n in nodes_by_cluster if n not in modal_cfgs and n not in echo_names
                       and n not in {s.engine_name for s in splits.values()}])
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
                    unverified_modal = list(modal_cfgs)
                if modal_cfgs:
                    # The final at-exit commits land once the containers are gone.
                    pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
            # The islands this teardown stopped get their island_stop in fleet.jsonl (a normal
            # end wrote it already; a STOP flag, breaker or signal did not), so the page stops
            # billing them; an island whose stop is not confirmed stays open. Never raises.
            _close_fleet_islands(prefix, skip=[*unverified, *unverified_modal])
            if unverified:
                # The head must NOT self-terminate: it is the only thing that
                # can still reach these orphaned learner clusters via sky.
                args._teardown_incomplete = True
                trips.trip(prefix, "launch.teardown_verify", len(unverified), 0, state="warn",
                           detail=f"termination not confirmed: {', '.join(unverified)}",
                           scope={"clusters": ",".join(unverified)})
                print(
                    f"[launcher] WARNING: could not confirm termination of "
                    f"{unverified}; leaving the head up so they stay reachable. "
                    f"Terminate them from the cloud console, then: yeto down {prefix}",
                    file=sys.stderr,
                )
        if ssh_islands is not None:
            ssh_islands.close()  # streams and locks; a kept island is reattached by the next launch
        if island_states is not None:
            island_states.close()  # the queued island records, bounded (never holds the exit)
        if island_lock is not None:
            island_lock.release()  # the next launcher of this run takes the records at once
        if getattr(args, "training_mode", None) == "rl" and not getattr(args, "dry_run", False):
            # the dispenser log of a syncer on this machine, with its check
            from .launch.dispense_evidence import collect_dispense_log

            collect_dispense_log(args, _modal_tape_run_dir(args),
                                 since_unix=None if head_mode else run_started_unix)
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
            trips.trip(prefix, "launch.teardown_verify", None, None,
                       detail=f"teardown not confirmed (exit {TEARDOWN_UNVERIFIED_EXIT})")
            return TEARDOWN_UNVERIFIED_EXIT  # noqa: B012
        if hands_off:
            # fenced on any way out (a signal, an error, a normal end): the exit says hands off
            return hands_off_exit  # noqa: B012


def _close_fleet_islands(run: str, *, skip=()) -> list[str]:
    """``island_stop`` for the run's islands still open in its ``fleet.jsonl`` (best effort)."""
    try:
        from . import runs
        from .dashboard.fleet import FLEET_FILE, close_open_islands, teardown_stop_reason

        return close_open_islands(runs.run_dir(run) / FLEET_FILE, teardown_stop_reason(), skip=skip)
    except Exception as e:  # noqa: BLE001 - telemetry never breaks the teardown
        print(f"[launcher] fleet.jsonl: island_stop not written ({e})", file=sys.stderr)
        return []


def _start_cost_guard(args, events_dir):
    """(ledger, guard or None) of this run. The ledger always exists (cleanup needs
    it); the guard is off with --no-cost-guard or --dry-run, and never fails the launch."""
    from . import runs
    from .cost import ledger as cost_ledger_mod
    from .cost.guard import CostGuard, TapeTail, guard_config_from_args, guard_enabled

    name = args.cluster_prefix
    if cost_ledger_mod.current() is None and not guard_enabled(args) \
            and getattr(args, "no_cost_guard", None) is None:
        # a programmatic launch without the CLI's cost flags (tests): no ledger file
        return cost_ledger_mod.NullLedger(), None
    ledger = cost_ledger_mod.current() or cost_ledger_mod.ResourceLedger.for_run(name)
    if not guard_enabled(args) or getattr(args, "dry_run", False):
        return ledger, None
    try:
        from .cost.billing import ModalBilling, Rates
        from .cost.report import default_tape_globs

        globs = default_tape_globs(name) + list(getattr(args, "cost_tapes", None) or [])
        if events_dir is not None:
            globs.append(str(Path(events_dir) / "*.jsonl"))
        cfg = guard_config_from_args(args)
        guard = CostGuard(name, cfg, ledger=ledger, tapes=TapeTail(globs),
                          rates=Rates.from_env(getattr(args, "cost_prices", None)),
                          billing=ModalBilling(refresh_s=max(cfg.poll_s, 300.0)),
                          log=lambda m: print(m, flush=True))
        print(f"[launcher] cost guard on: budget {cfg.budget_usd} USD (warn {cfg.warn_usd}), idle "
              f"warn {cfg.idle.idle_after_s / 60:.0f} min + stop {cfg.idle.stop_after_s / 60:.0f} min, "
              f"load grace {cfg.idle.load_grace_s / 60:.0f} min; ledger {ledger.path}", flush=True)
        runs.update_run(name, resource_ledger=str(ledger.path))
        return ledger, guard.start()
    except Exception as e:  # noqa: BLE001
        print(f"[launcher] WARNING: cost guard not started: {e}", file=sys.stderr, flush=True)
        return ledger, None


NO_SYNC_EVENT_DRAIN_S = 120.0
# no-sync run whose island tape lacks rl_learner_finalized (differs from 2:
# "artifact not fetchable").
NO_SYNC_INCOMPLETE_EXIT = 3


def _dry_run_role_syncers(args, num_learners: int) -> list[dict]:
    commands = role_syncer_commands(args, num_learners)
    plans = []
    for syncer in rl_role_syncers(args):
        plan = {
            "role": syncer.role,
            "port": syncer.port,
            "address": syncer.address,
            "layout": f"{syncer.role}_layout_hash",
            "command": commands[syncer.role],
            "checkpoint": syncer.checkpoint,
            "event_tape": syncer.event_tape,
        }
        shards = role_syncer_shards(args, syncer)
        if len(shards) > 1:
            plan["shards"] = [{"port": shard.port, "checkpoint": shard.checkpoint,
                               "event_tape": shard.event_tape} for shard in shards]
        plans.append(plan)
    return plans


def _dry_run_ssh_island(args, spec: ClusterSpec, m: int, num_islands: int, task) -> dict:
    """The per-node commands and the rank layout of an ``ssh:`` island (resident, or the
    trainer scheduler's); a spec the launch would refuse is refused here."""
    from .launch import trainer_loop
    from .launch.ssh_resident import dry_run_record, is_resident
    from .launch.ssh_trainer_island import SshIslandError

    if not is_resident(args, spec, m, num_islands) and trainer_loop.held_island(args) != m:
        return {"mode": "sky-ssh-node-pool"}  # a sampler island: SkyPilot's SSH node pool
    try:
        ssh_spec = _ssh_trainer_spec(args, spec, task)
    except SshIslandError as error:
        raise ValueError(f"ssh: island {m}: {error}") from None
    name = learner_cluster_names(args.cluster_prefix, parse_gpu_spec(args.gpu))[m]
    return dry_run_record(args, ssh_spec, name,
                          mode="resident" if is_resident(args, spec, m, num_islands) else "trainer-scheduler")


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
    from .launch import ssh_resident

    try:
        ssh_resident.check(args, parse_gpu_spec(args.gpu))
    except SystemExit as error:
        raise ValueError(str(error)) from None
    if rl:
        from .launch.code_pack import run_code_pack_problem

        problem = run_code_pack_problem(args)  # the launch's pinned-pack refusal, found early
        if problem is not None:
            raise ValueError(problem)
    specs = parse_gpu_spec(args.gpu)
    external = max(0, getattr(args, "external_learners", 0) or 0)
    islands = []
    island_overrides = _island_overrides(args, specs) if rl else {}
    from .launch_preflight import dry_run_memory, print_dry_run_memory

    memory = dry_run_memory(args, specs)
    print_dry_run_memory(memory)
    if rl:
        from .launch.downlink_demand import downlink_preflight

        downlink_preflight(args)  # a WARNING only; not in the plan (byte-locked)
        from .launch.topology import fabric_bandwidth_notes

        for note in fabric_bandwidth_notes(args, specs):  # notes only; not in the plan
            print(note, flush=True)
    from .launch.model_check import preflight_trainer_model

    trainer_model = preflight_trainer_model(args)  # ModelCheckError (a ValueError): rejected
    from .launch.trainer_image_check import check_trainer_images

    check_trainer_images(args, specs)  # the launch preflight's image / card refusal, found early
    from .launch.trainer_provision import check_sky_prices

    check_sky_prices(args, specs)  # the launch preflight's budget / price refusal, found early
    task_plan = island_task_plan(args) if rl else None
    split_plan = data_split_plan(args) if rl else None
    placements = rl_island_placement_records(args) if rl else []
    splits = engine_cluster.plan(
        args, specs, learner_cluster_names(args.cluster_prefix, specs),
        role_of=lambda m: _island_role(args, specs, m)) if engine_cluster.requested(args) else {}
    split_by_id = {split.learner_id: split for split in splits.values()}
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
                                sample_roles.syncer_learner_count(args, len(specs)) + external,
                                "$SYNCER_ADDR", island_overrides, role)
            if role is not None:
                from .cloud.droppable import billing, rejoin_source

                entry["island_role"] = {"role": role, "billing": billing(role),
                                        "rejoin": rejoin_source(role)}
            if (sample_roles.is_store_coupled(args) or sample_roles.is_synced_group(args)
                    or sample_roles.is_external_pool_group(args)):
                # per-island role, learner 0 of 1; synced group; trainers of external pools
                entry["sample_role"] = sample_roles.island_args(
                    args, learner_id, len(specs)).rl_sample_role
            if learner_id in (bindings := sample_roles.sampler_bindings(args, len(specs))):
                entry["feeds_trainer"] = bindings[learner_id]  # --rl-sampler-follow trainer
                if sample_roles.is_mixed_group(args) and learner_id in (
                        ledgers := sample_roles.sampler_dispense_stores(args, len(specs))):
                    entry["dispense_store"] = ledgers[learner_id]  # its rows from the pools' store ledger
            if (substore := sample_roles.island_substore(args, learner_id, len(specs))) is not None:
                entry["sample_substore"] = substore  # trainer<k>, k the trainer's learner id
            if (pool_shards := sample_roles.plan_buffer_shards(args, learner_id, len(specs))) is not None:
                entry["buffer_shards"] = pool_shards  # a trainer's shards with external pools
            if learner_id in (stops := sample_roles.sampler_stop_trainers(args, len(specs))):
                entry["sampler_until_trainers_end"] = stops[learner_id]  # no fixed round count
            if learner_id == trainer_loop.held_island(args):
                entry["held_by_trainer_scheduler"] = True  # the fleet never starts it
            if task_plan is not None:
                entry["island_tasks"] = task_plan[learner_id].record()
            if split_plan.several_islands:
                entry["data_split"] = split_plan.island_record(learner_id)
            if (engine_rows := _island_engine_rows(args, learner_id)):
                entry["island_engine"] = engine_rows[0]  # --rl-island-engine (A22)
            if island_overrides.get(learner_id):
                from .island_overrides import records

                entry["island_overrides"] = [r for r in records(args, island_overrides)
                                             if r["island"] == learner_id]
            entry["learner_command"] = next(
                (line.strip() for line in task.run.splitlines() if _rl_backend_module(args, "launch_flags").ISLAND_ENTRY_MODULE in line),
                None,
            )
            if spec.cloud == "ssh" and getattr(args, "rl_node_inventory_nodes", None):
                entry["ssh_island"] = _dry_run_ssh_island(args, spec, learner_id, len(specs), task)
            if learner_id in split_by_id:
                entry["engine_cluster"] = engine_cluster.dry_run_record(
                    split_by_id[learner_id], lambda learner_id=learner_id, spec=spec, role=role: _island_task(
                        make_miles_island_task, args, spec, learner_id,
                        sample_roles.syncer_learner_count(args, len(specs)) + external,
                        "$SYNCER_ADDR", island_overrides, role), args)
        islands.append(entry)
    return {
        "dry_run": True,
        "training_mode": getattr(args, "training_mode", "sft"),
        "rl_engine": "ports" if rl else None,
        "controller": "head" if head else "local",
        "islands": len(specs) + external,
        # which rows of --data each training island reads
        **({"data_split": {**split_plan.record(), "warning": split_plan.warning()}}
           if split_plan is not None else {}),
        # Kimi-K3.1 full form: the checked selection and a rough per-GPU memory budget
        **({"kimi_k31_full": args._kimi_k31_full}
           if getattr(args, "_kimi_k31_full", None) else {}),
        # the trainer's model: source, problems, warnings (absent: --preflight-model off)
        **({"trainer_model": trainer_model} if trainer_model is not None else {}),
        "external_learners": external,
        "total_gpus": sum(s.total_gpus for s in specs),
        "syncer": None if no_sync else ("head VM" if head else f"{args.cluster_prefix}-syncer"),
        # one syncer per trainable role (actor first); none without outer sync
        "role_syncers": [] if not rl or no_sync else _dry_run_role_syncers(
            args, sample_roles.syncer_learner_count(args, len(specs)) + external),
        "outer_sync": not no_sync,
        # the head's scheduler and the island it owns
        "trainer_scheduler": (None if not trainer_loop.enabled(args) else {
            "mode": trainer_loop.scheduler_mode(args), "island": trainer_loop.held_island(args),
            "convertible_pool": trainer_loop.convertible_pool(args),
            "interval_s": args.rl_trainer_scheduler_interval}),
        "algorithm_spec_sha256": getattr(args, "rl_expected_algorithm_sha256", None),
        "algorithm_spec": getattr(args, "rl_algorithm_spec_json", None),
        "unverified_mechanisms": list(getattr(args, "rl_allow_unverified_mechanism", None) or ()),
        "clusters": [] if not islands else (learner_cluster_names(args.cluster_prefix, specs)
                                            + [split.engine_name for split in splits.values()]),
        "island_requests": islands,
    }
