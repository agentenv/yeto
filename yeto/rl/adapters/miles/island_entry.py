"""Entrypoint for one pinned-Miles RL island."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.bridge import BridgeConfig
from yeto.rl.decoupled import DecoupledBridgeConfig
from yeto.rl.engine.full_parameter_identity import (
    full_parameter_identity,
    identity_sha256,
    is_sha256,
    local_init as local_init_rule,
)
from yeto.rl.engine.run_flags import (
    FULL_PARAMETER_ASYNC_MAX_PIPELINE,
    FULL_PARAMETER_MAX_LOCAL_ROUNDS_AHEAD as MAX_LOCAL_ROUNDS_AHEAD,
    ISLAND,
    add_rl_flags,
    rl_option_pairs,
)
from yeto.rl.adapters.miles.models.deepseek_v4.expert_full import expert_full_specs
from yeto.rl.export import adapter_targets, derive_peft_lora_specs
from yeto.rl.adapters.miles.revision import verify_miles_revision


def build_parser() -> argparse.ArgumentParser:
    """The island argv parser: the schema's island flags plus the hook-owned ones."""

    from yeto.rl.adapters.miles.elastic_hook import add_recommend_arguments
    from yeto.wandb_logger import add_arguments as add_wandb_arguments

    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.adapters.miles.island_entry")
    add_rl_flags(parser, ISLAND)  # yeto/rl/engine/run_flags.py (P0 4.7)
    add_recommend_arguments(parser)  # D2 elastic hook
    add_wandb_arguments(parser)
    return parser


def verify_island_code_pack(digest: str | None, tree: str | Path | None = None, *,
                            mode: str | None = None, release_manifest: str | Path | None = None,
                            log=print) -> dict | None:
    """rl-identity-tiering L9.5: refuse to start when the workdir is not the run pack.

    Release mode (L9.12): the tree must be the image's source root with the source
    hash its build recorded; returns the release manifest. The image digest
    (``digest``) is checked by the registry pull, not here."""

    if not digest:
        return None
    from yeto import codepack

    root = Path(tree) if tree is not None else Path(__file__).resolve().parents[4]
    if mode == codepack.CODE_PACK_MODE_RELEASE:
        manifest = codepack.verify_release_tree(root, release_manifest or codepack.RELEASE_MANIFEST)
        log(f"[yeto-island] code pack release {digest} ok: commit {manifest.get('yeto_commit')}, "
            f"tree {root.resolve()}")
        return manifest
    codepack.verify_tree(root, digest)
    log(f"[yeto-island] code pack {digest} ok: tree {root}")
    return None


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.parameter_mode == "lora" and (
        args.lora_r is None or args.lora_targets is None
    ):
        parser.error("LoRA mode requires --lora-r and --lora-targets")
    from yeto.rl.engine.selection import (
        FULL_PARAMETER_SYNC_PRESET,
        FULL_PARAMETER_SYNC_PRESETS,
        SAO_STREAMING_SYNC_PRESET,
    )

    if (args.parameter_mode == "full" and args.sync_preset not in FULL_PARAMETER_SYNC_PRESETS
            and not _no_sync_island(args)):
        parser.error(
            "--parameter-mode full requires a full-parameter --sync-preset "
            f"({', '.join(sorted(FULL_PARAMETER_SYNC_PRESETS))}) or "
            "--rl-single-island-no-sync"
        )
    if args.sync_preset == FULL_PARAMETER_SYNC_PRESET and not _trains_base_parameters(args):
        parser.error(
            f"--sync-preset {FULL_PARAMETER_SYNC_PRESET} requires --parameter-mode full "
            "or --expert-full-count"
        )
    if args.sync_preset == SAO_STREAMING_SYNC_PRESET and args.parameter_mode != "full":
        parser.error("--sync-preset sao-streaming-full requires --parameter-mode full")
    sao_runtime = (
        args.sao_streaming_runtime,
        args.sao_streaming_runtime_sha256,
        args.sao_context_sha256,
    )
    if (args.sync_preset == SAO_STREAMING_SYNC_PRESET) != all(value is not None for value in sao_runtime):
        parser.error(
            "--sync-preset sao-streaming-full requires --sao-streaming-runtime, "
            "--sao-streaming-runtime-sha256 and --sao-context-sha256, and only it takes them"
        )
    if not args.eval_only and not args.inner_lr > 0:
        # A zero LR would only surface as the zero-LR invariant failing round 1.
        parser.error(f"--inner-lr must be > 0 for a training run (got {args.inner_lr})")
    try:
        _require_ports_supported(args)
    except ValueError as error:
        parser.error(str(error))
    if (args.rl_critic_load is None) != (args.rl_critic_init_sha256 is None):
        parser.error("--rl-critic-load and --rl-critic-init-sha256 go together")
    if args.rl_critic_baseline_rounds < 0:
        parser.error("--rl-critic-baseline-rounds must be >= 0")
    if args.rl_critic_baseline_run and not getattr(args, "rl_single_island_no_sync", False):
        parser.error("--rl-critic-baseline-run needs --rl-single-island-no-sync")
    if args.rl_critic_baseline_rounds and not (
        getattr(args, "rl_single_island_no_sync", False) and args.rl_critic_load is None
    ):
        parser.error("--rl-critic-baseline-rounds needs --rl-single-island-no-sync "
                     "and no --rl-critic-load")
    try:
        _check_ports_infra_switches(args)
        _check_single_island_no_sync(args)
        _check_ports_algorithm_options(
            args, outer_sync=not getattr(args, "rl_single_island_no_sync", False)
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def build_ports_launch(args, run_config, extra_argv=()):
    """The ports learner's spec -> Miles argv step (design D8).

    ``--rl-algorithm-spec`` (else the legacy CLI, as in R0), then the mapped
    extra-argv flags are absorbed by the translation; the run config's
    estimator follows the absorbed spec.
    """

    import dataclasses

    from yeto.rl.engine.algorithm import resolve_ports_algorithm
    from yeto.rl.adapters.miles.algorithm_flags import absorb_extra_argv
    from yeto.rl.adapters.miles.config import translate_run_config

    base_algorithm = resolve_ports_algorithm(args)
    absorbed, _, _ = absorb_extra_argv(base_algorithm, tuple(extra_argv))
    run_config = dataclasses.replace(
        run_config,
        algorithm=dataclasses.replace(
            run_config.algorithm, advantage_estimator=absorbed.advantage_estimator
        ),
    )
    return translate_run_config(run_config, base_algorithm, extra_argv=tuple(extra_argv),
                                max_policy_age=int(getattr(args, "rl_max_policy_age", 0) or 0),
                                async_buffer_rounds=getattr(args, "rl_async_buffer_rounds", None),
                                full_parameter_sync=not _no_sync_island(args),
                                rollout_ahead=int(getattr(args, "rl_rollout_ahead", 0) or 0))


_ELASTIC_COMPANIONS = rl_option_pairs(
    "rl_elastic_resources",
    "rl_elastic_attestation",
    "rl_elastic_state_dir",
    "rl_elastic_initial_config",
    "rl_elastic_cells",
    "rl_elastic_quorum_timeout_s",
    "rl_elastic_idle_flow_timeout_s",
    "rl_elastic_pause_margin",
    "rl_elastic_drain_timeout_s",
    "rl_elastic_recovery_timeout_s",
    "rl_elastic_max_recovery_attempts",
    "rl_elastic_checkpoint_store",
)
_ELASTIC_PAUSE = ("rl_elastic_quorum_timeout_s", "rl_elastic_idle_flow_timeout_s",
                  "rl_elastic_pause_margin", "rl_elastic_drain_timeout_s",
                  "rl_elastic_recovery_timeout_s")
# --rl-elastic-cells is optional: without it the island lists the fork's
# declared cells (InferenceController.describe_cells, fork F-R1) at compose time.
_ELASTIC_REQUIRED = ("rl_elastic_resources", "rl_elastic_state_dir",
                     "rl_elastic_initial_config")


def _check_ports_infra_switches(args) -> None:
    """``--rl-elastic`` and its companions: explicit, never implied."""

    given = [flag for name, flag in _ELASTIC_COMPANIONS if getattr(args, name, None) is not None]
    if getattr(args, "rl_elastic_tool_wait_board", False):
        given.append("--rl-elastic-tool-wait-board")
    if getattr(args, "rl_elastic_trainer_edges", False):
        given.append("--rl-elastic-trainer-edges")
    if getattr(args, "rl_elastic_accept_rebind", False):
        given.append("--rl-elastic-accept-rebind")
    from yeto.rl.adapters.miles.elastic_hook import check_recommend_flags
    check_recommend_flags(args)
    # S27 G-DI design v2: a two-cluster island with an engine outage budget declares its
    # rollout cells without --rl-elastic (the launcher sets the env with the flags)
    from yeto.rl.engine.engine_outage import outage_budget_s

    outage_cells = (bool(getattr(args, "rl_elastic_declare_cells", False))
                    and not getattr(args, "rl_elastic", False) and outage_budget_s(os.environ) is not None)
    if getattr(args, "rl_elastic_declare_cells", False) and not getattr(args, "rl_elastic", False) \
            and not outage_cells:
        raise ValueError("--rl-elastic-declare-cells needs --rl-elastic")
    _check_resume_switches(args)
    _check_hf_exports(args)
    _check_sample_store(args)
    if not getattr(args, "rl_elastic", False):
        if outage_cells:
            if not _elastic_cells(args.rl_elastic_cells):
                raise ValueError("--rl-elastic-declare-cells needs --rl-elastic-cells (the names)")
            given = [flag for flag in given if flag != "--rl-elastic-cells"]
        if given:
            raise ValueError(", ".join(given) + " need --rl-elastic")
        return
    missing = [flag for name, flag in _ELASTIC_COMPANIONS
               if name in _ELASTIC_REQUIRED and not getattr(args, name, None)]
    if missing:
        raise ValueError("--rl-elastic needs " + ", ".join(missing))
    if args.rl_elastic_cells is not None and not _elastic_cells(args.rl_elastic_cells):
        raise ValueError("--rl-elastic-cells names no cell")
    if getattr(args, "rl_elastic_declare_cells", False) and not _elastic_cells(args.rl_elastic_cells):
        raise ValueError("--rl-elastic-declare-cells needs --rl-elastic-cells (the names)")
    for name in _ELASTIC_PAUSE + ("rl_elastic_max_recovery_attempts",):
        value = getattr(args, name, None)
        if value is not None and not value > 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def _check_sample_store(args) -> None:
    """S25 C3: ``--rl-sample-store`` / ``--rl-sample-lanes`` (engine-neutral rule)."""
    from yeto.rl.engine.sample_roles import check_sample_store

    check_sample_store(args)


_CUT_FLAGS = (("rl_cut_every", "--rl-cut-every"), ("rl_cut_keep", "--rl-cut-keep"),
              ("rl_stop_after_rounds", "--rl-stop-after-rounds"),
              ("rl_cut_auto_mtbf_s", "--rl-cut-auto-mtbf-s"),
              ("rl_cut_max_interval_s", "--rl-cut-max-interval-s"))


def _check_resume_switches(args) -> None:
    """rl-resume-from-checkpoint: ``--rl-resume-store`` is single-island/no-sync;
    ``--rl-island-cut-store`` (S27 R-2) is a synced island's; the cut cadence flags
    need a store (elastic, resume or island cuts)."""
    from yeto.rl.engine.cut_interval import cut_interval_for

    store = getattr(args, "rl_resume_store", None)
    island_store = getattr(args, "rl_island_cut_store", None)
    has_store = bool(store) or bool(island_store) or bool(getattr(args, "rl_elastic_checkpoint_store", None))
    for name, flag in _CUT_FLAGS:
        value = getattr(args, name, None)
        if value is not None and not value > 0:
            raise ValueError(f"{flag} must be positive" if isinstance(value, float) else f"{flag} must be >= 1")
    cut_interval_for(every=getattr(args, "rl_cut_every", None),
                     auto_mtbf_s=getattr(args, "rl_cut_auto_mtbf_s", None),
                     max_interval_s=getattr(args, "rl_cut_max_interval_s", None))  # exclusive modes
    given = [f for n, f in _CUT_FLAGS if getattr(args, n, None) is not None]
    if getattr(args, "rl_resume_allow_config_change", False):
        given.append("--rl-resume-allow-config-change")
    if given and not has_store:
        raise ValueError(", ".join(given) + " need --rl-resume-store (or --rl-elastic-checkpoint-store "
                         "or --rl-island-cut-store)")
    if getattr(args, "rl_cut_keep", None) is not None and not (store or island_store):
        raise ValueError("--rl-cut-keep applies to --rl-resume-store or --rl-island-cut-store "
                         "(the elastic store keeps every cut)")
    if island_store:
        _check_island_cut_store(args)
    if getattr(args, "rl_stop_after_rounds", None) is not None and not store:
        raise ValueError("--rl-stop-after-rounds needs --rl-resume-store")
    if not store:
        return
    if getattr(args, "rl_elastic", False):
        raise ValueError("--rl-resume-store is the non-elastic store; with --rl-elastic use "
                         "--rl-elastic-checkpoint-store")
    if not getattr(args, "rl_single_island_no_sync", False):
        raise ValueError("--rl-resume-store needs --rl-single-island-no-sync (multi-island resume "
                         "is not implemented yet: rl-resume-from-checkpoint 3.3)")


def _check_island_cut_store(args) -> None:
    """S27 R-2: island round cuts belong to a synced island (a syncer, no elastic store,
    no resume store); its cuts are written only, so the resume switches do not apply."""
    if getattr(args, "rl_single_island_no_sync", False) or getattr(args, "rl_elastic", False):
        raise ValueError("--rl-island-cut-store is a synced island's (no --rl-single-island-no-sync, "
                         "no --rl-elastic)")
    if getattr(args, "rl_resume_store", None):
        raise ValueError("--rl-island-cut-store and --rl-resume-store exclude each other")
    for name, flag in (("rl_stop_after_rounds", "--rl-stop-after-rounds"),
                       ("rl_resume_allow_config_change", "--rl-resume-allow-config-change")):
        if getattr(args, name, None):
            raise ValueError(f"{flag}: a synced island does not resume from its round cuts "
                             "(the syncer is authoritative)")


def _check_hf_exports(args) -> None:
    """``--rl-export-hf-rounds`` (``yeto.rl.engine.hf_export``): the launcher sends it
    with ``--rl-export-hf-store`` to islands that generate and train."""
    from yeto.rl.engine import sample_roles
    from yeto.rl.engine.hf_export import EXPORT_HF_FLAG, ExportRounds

    rounds = ExportRounds.parse(getattr(args, "rl_export_hf_rounds", None))
    if bool(rounds) != bool(getattr(args, "rl_export_hf_store", None)):
        raise ValueError(f"{EXPORT_HF_FLAG} and --rl-export-hf-store go together")
    if rounds and getattr(args, "rl_elastic", False):
        raise ValueError(f"{EXPORT_HF_FLAG} is not wired for --rl-elastic")
    if rounds and sample_roles.sample_role(args) != sample_roles.BOTH:
        raise ValueError(f"{EXPORT_HF_FLAG} needs an island that generates and trains (sample role both)")


def hf_export_config(args) -> dict[str, Any] | None:
    """The island's ``--rl-export-hf-rounds``: ``{"rounds"`` (the flag's canonical
    text), ``"store"`` (its own export store)``}``, None when off."""
    from yeto.rl.engine.hf_export import ExportRounds, export_store_path

    rounds = ExportRounds.parse(getattr(args, "rl_export_hf_rounds", None))
    if not rounds:
        return None
    return {"rounds": str(rounds),
            "store": export_store_path(str(args.rl_export_hf_store), learner_id=int(args.learner_id))}


def _elastic_cells(value: str | None) -> tuple[str, ...]:
    return tuple(c.strip() for c in (value or "").split(",") if c.strip())


def apply_sample_role(args, miles_args, environ=None) -> None:
    """L4.6 (--rl-sample-role sampler|trainer): the buffer config on ``miles_args``
    and the environment of the buffer rollout functions (forwarded to the Ray
    workers by ``connect_island_ray``). Role ``both`` sets nothing."""
    from yeto.rl.engine import sample_roles

    from .buffer_rollout import island_env

    env = os.environ if environ is None else environ
    if int(getattr(args, "rl_spot_subscribers", 0) or 0) > 0:
        # S27 G-EA: island 0 publishes its fragment layout for the spot subscribers
        from yeto.rl.spot_rollout.layout_manifest import PUBLISH_ENV

        env[PUBLISH_ENV] = "1"
    if sample_roles.sample_role(args) == sample_roles.BOTH:
        return
    config = sample_roles.resolve(args, groups_per_round=args.groups_per_round)
    miles_args.yeto_rl_sample_buffer = config
    node = getattr(args, "rl_convertible_node", None)  # L7.5: a convertible node's own shard
    # a pair island runs as learner 0: its sampler id is its node name (S25 C1)
    sampler = node or getattr(args, "rl_sampler_node", None) or f"island{int(args.learner_id)}"
    env.update(island_env(
        config.role, store_root=config.store, sampler_id=sampler,
        max_policy_age=int(getattr(args, "rl_max_policy_age", 0) or 0), wait_s=config.wait_s,
        shard=node, shards=getattr(args, "rl_buffer_shards", None),
        lane=sample_roles.buffer_lane(args), data_slot=getattr(args, "rl_sampler_data_slot", None)))
    miles_args.yeto_rl_sample_lane = sample_roles.buffer_lane(args)  # S25 C3: None off a synced trainer
    if getattr(args, "rl_sample_relay", False) and config.role == sample_roles.TRAINER:
        env.update(sample_relay_env(args))
    if getattr(args, "rl_spot_syncer", None) and config.role == sample_roles.SAMPLER:
        env.update(spot_sampler_env(args))


def spot_sampler_env(args) -> dict[str, str]:
    """S27 G-EA (--rl-spot-syncer, sampler island): weights from that syncer as a
    receive-only subscriber, groups to its relay channel --rl-spot-channel."""
    from yeto.rl.spot_rollout.relay_buffer import RELAY_ADDR_ENV, RELAY_CHANNEL_ENV, SPOT_SYNCER_ENV

    host, port = _syncer_address(args.rl_spot_syncer)
    return {SPOT_SYNCER_ENV: f"{host}:{port}", RELAY_ADDR_ENV: f"{host}:{port}",
            RELAY_CHANNEL_ENV: str(int(getattr(args, "rl_spot_channel", 0) or 0))}


def sample_relay_env(args, *, tmp_root: str | None = None) -> dict[str, str]:
    """S27 G-EA (--rl-sample-relay, trainer island): the spot samplers' groups come
    through the head's sample relay. The trainer reads them from a buffer on local
    disk (the relay pump fills it: ``yeto.rl.spot_rollout.relay_buffer``) and makes
    no policy releases (the samplers follow the syncer as subscribers)."""
    import tempfile

    from yeto.rl.spot_rollout.relay_buffer import RELAY_ADDR_ENV, RELAY_CHANNEL_ENV, RELAY_PREFETCH_ENV

    from .buffer_rollout import BUFFER_ROOT_ENV

    host, port = _syncer_address(args.syncer)
    root = Path(tmp_root or tempfile.gettempdir()) / "yeto-relay-buffer" / f"island{int(args.learner_id)}"
    root.mkdir(parents=True, exist_ok=True)
    prefetch = getattr(args, "rl_sample_relay_prefetch", None)
    if prefetch is None:
        prefetch = 2 * int(args.groups_per_round)
    return {BUFFER_ROOT_ENV: str(root), RELAY_ADDR_ENV: f"{host}:{port}",
            RELAY_CHANNEL_ENV: str(int(args.learner_id)), RELAY_PREFETCH_ENV: str(int(prefetch))}


def _cut_auto(args) -> dict[str, float]:
    """The timed cut interval flags (empty: every N rounds)."""
    mtbf = getattr(args, "rl_cut_auto_mtbf_s", None)
    if mtbf is None:
        return {}
    cap = getattr(args, "rl_cut_max_interval_s", None)
    return {"auto_mtbf_s": float(mtbf), **({} if cap is None else {"max_interval_s": float(cap)})}


def apply_ports_infra_switches(args, miles_args, environ=None) -> None:
    """Carry the opt-in 2.3/3.x switches onto ``miles_args`` before any rollout
    process starts. Without them nothing is set (default path unchanged)."""

    if getattr(args, "rl_overlap_eval", False):
        miles_args.yeto_rl_overlap_eval = True
    if getattr(args, "rl_verify_every_round", False):  # E1: read by entry (driver.verify_every_round)
        miles_args.yeto_rl_verify_every_round = True
    if getattr(args, "rl_verify_every", None) is not None:  # S27 C5: read by entry (driver.verify_every)
        from yeto.rl.engine.run_flags import check_verify_every

        check_verify_every(args)
        miles_args.yeto_rl_verify_every = int(args.rl_verify_every)
    if getattr(args, "rl_profile_trainer", False):  # E1: read by state_plugin.train_one_step
        miles_args.yeto_rl_profile_trainer = True
        from yeto.rl.engine import phase_profile

        phase_profile.enable()  # samples the thread that drives the phases
    if getattr(args, "rl_debug_dump", False):
        out = os.path.dirname(os.path.abspath(os.path.expanduser(
            getattr(args, "event_tape", None) or "~/yeto-output/rl-island.jsonl")))
        miles_args.save_debug_rollout_data = f"{out}/miles-rollout-{{rollout_id}}.pt"
        miles_args.save_debug_train_data = f"{out}/miles-train-{{rollout_id}}_{{rank}}.pt"
    if getattr(args, "rl_max_policy_age", 0):  # agentic-rollout-utilization 2.1
        from .harbor_agent import AGENT_FUNCTION as HARBOR_AGENT_FUNCTION
        from .policy_age import SUPPORT, check_task

        SUPPORT.check(int(args.rl_max_policy_age))
        check_task(int(args.rl_max_policy_age),
                   custom_generate=getattr(miles_args, "custom_generate_function_path", None),
                   custom_agent=getattr(miles_args, "custom_agent_function_path", None),
                   recompute_prefill=bool(getattr(miles_args, "recompute_logprobs_via_prefill", False)))
        if getattr(miles_args, "custom_agent_function_path", None) == HARBOR_AGENT_FUNCTION:
            from .harbor_agent import check_suspend_config

            check_suspend_config(environ)  # S26 E9.5: stage 3 holds model calls at the gate relay
        miles_args.yeto_rl_max_policy_age = int(args.rl_max_policy_age)
    if getattr(args, "rl_rollout_ahead", 0):  # read by entry (driver) and carry_over
        miles_args.yeto_rl_rollout_ahead = int(args.rl_rollout_ahead)
    apply_sample_role(args, miles_args, environ)
    exports = hf_export_config(args)
    if exports is not None:  # read by entry.hf_exports_for
        miles_args.yeto_rl_hf_exports = exports
    observe = bool(getattr(args, "rl_observe_timeline", False))
    if observe:
        miles_args.yeto_rl_observe_timeline = True
    from yeto.rl.engine.telemetry import DEFAULT_HEARTBEAT_INTERVAL_S, DEFAULT_RESOURCE_INTERVAL_S

    for flag, attr, default in (
        ("rl_heartbeat_interval", "yeto_rl_heartbeat_interval_s", DEFAULT_HEARTBEAT_INTERVAL_S),
        ("rl_resource_sample_interval", "yeto_rl_resource_sample_interval_s",
         DEFAULT_RESOURCE_INTERVAL_S),
    ):
        value = getattr(args, flag, None)
        value = (default if observe else None) if value is None else float(value)
        if value:
            if value < 0:
                raise ValueError(f"--{flag.replace('_', '-')} must be >= 0")
            setattr(miles_args, attr, value)
    from yeto.rl.adapters.miles.elastic_hook import apply_recommend_flags
    apply_recommend_flags(args, miles_args)
    if getattr(args, "rl_deterministic_trainer", False):
        from yeto.rl.adapters.miles.entry import DETERMINISM_ENV

        (os.environ if environ is None else environ).update(DETERMINISM_ENV)
    # ruling 2026-10-04 v2: the rollout bind path (bind_members) reads the engine opt-in
    if getattr(args, "rl_allow_cross_node_engine_tp", False):
        miles_args.yeto_rl_allow_cross_node_engine_tp = True
    if getattr(args, "rl_allow_cross_node_tp", False):
        miles_args.yeto_rl_allow_cross_node_tp = True
    auto = _cut_auto(args)
    if auto:
        miles_args.yeto_rl_cut_auto = auto  # S27 R-2: a timed interval (any cut store)
    island_store = getattr(args, "rl_island_cut_store", None)
    if getattr(args, "rl_resume_store", None) or island_store:
        every = getattr(args, "rl_cut_every", None)
        miles_args.yeto_rl_resume = {
            "store": (str(args.rl_resume_store) if not island_store
                      else f"{str(island_store).rstrip('/')}/island{int(args.learner_id)}"),
            "state_dir": str(getattr(args, "rl_resume_state_dir", None) or "~/yeto-rl/resume-state"),
            # None: the default timed interval (cut_interval module doc)
            "every": None if auto or every is None else int(every),
            **auto,
            "keep": int(getattr(args, "rl_cut_keep", None) or 2),
            "allow_config_change": bool(getattr(args, "rl_resume_allow_config_change", False)),
            "stop_after": (None if getattr(args, "rl_stop_after_rounds", None) is None
                           else int(args.rl_stop_after_rounds)),
            # S27 R-2: a synced island writes its cuts only (no resume from them)
            **({"write_only": True} if island_store else {}),
        }
        from yeto.rl.adapters.miles.rollout_meta_hook import ELASTIC_METADATA_ENV

        # the cut needs the rollout data cursor (rollout-side metadata), as elastic does
        miles_args.yeto_rl_elastic_metadata = True
        (os.environ if environ is None else environ)[ELASTIC_METADATA_ENV] = "1"
    elif getattr(args, "rl_cut_every", None) is not None:
        miles_args.yeto_rl_resume_every = int(args.rl_cut_every)
    if not getattr(args, "rl_elastic", False):
        return
    miles_args.yeto_rl_elastic = {
        "resources": args.rl_elastic_resources,
        "attestation": getattr(args, "rl_elastic_attestation", None),
        "state_dir": args.rl_elastic_state_dir,
        "initial_config": args.rl_elastic_initial_config,
        "declared_cells": _elastic_cells(args.rl_elastic_cells),
    }
    if getattr(args, "rl_elastic_tool_wait_board", False):
        miles_args.yeto_rl_elastic["tool_wait_board"] = True
    if getattr(args, "rl_elastic_trainer_edges", False):
        miles_args.yeto_rl_elastic["trainer_edges"] = True
    if getattr(args, "rl_elastic_accept_rebind", False):
        miles_args.yeto_rl_elastic["accept_rebind"] = True
    if getattr(args, "rl_elastic_checkpoint_store", None):
        miles_args.yeto_rl_elastic["checkpoint_store"] = str(args.rl_elastic_checkpoint_store)
    if getattr(args, "rl_island_pool", False):
        miles_args.yeto_rl_elastic["island_scheduling"] = "elastic"
    for name in _ELASTIC_PAUSE:
        if getattr(args, name, None) is not None:
            miles_args.yeto_rl_elastic[name.removeprefix("rl_elastic_")] = float(getattr(args, name))
    if getattr(args, "rl_elastic_max_recovery_attempts", None) is not None:
        miles_args.yeto_rl_elastic["max_recovery_attempts"] = int(args.rl_elastic_max_recovery_attempts)
    # M1: rollout metadata carries data_cursor/buffer_length only when asked.
    # The attribute covers the driver process; the env var reaches Ray workers
    # (where the metadata hook runs) through connect_island_ray's job-level
    # runtime_env, since workers inherit the raylet's env, not the driver's.
    from yeto.rl.adapters.miles.rollout_meta_hook import ELASTIC_METADATA_ENV

    miles_args.yeto_rl_elastic_metadata = True
    (os.environ if environ is None else environ)[ELASTIC_METADATA_ENV] = "1"


def _check_single_island_no_sync(args) -> None:
    """``--rl-single-island-no-sync``: explicit, ports-only, one island, no syncer."""

    if not getattr(args, "rl_single_island_no_sync", False):
        if getattr(args, "syncer", None) is None:
            raise ValueError("--syncer is required (unless --rl-single-island-no-sync)")
        return
    problems = []
    if getattr(args, "syncer", None) is not None:
        problems.append("--syncer")
    if int(getattr(args, "num_learners", 1) or 1) != 1:
        problems.append(f"--num-learners {args.num_learners}")
    if int(getattr(args, "learner_id", 0)) != 0:
        problems.append(f"--learner-id {args.learner_id}")
    if getattr(args, "sync_preset", "strict-avg") != "strict-avg":
        problems.append(f"--sync-preset {args.sync_preset}")
    if getattr(args, "initial_adapter", None):
        problems.append("--initial-adapter")
    if problems:
        raise ValueError(
            "--rl-single-island-no-sync (one ports island, no syncer, no outer sync) "
            "cannot be combined with: " + ", ".join(problems)
        )


def _check_ports_algorithm_options(args, *, outer_sync: bool = True) -> None:
    """Startup refusal of the ports algorithm options (D8/D11), before any work."""

    from yeto.rl.engine.algorithm import check_unverified_allowance

    # D11 as written: any outer sync refuses the allowance. The learner CLI
    # always joins a syncer (--syncer is required), so the island count it
    # receives (--num-learners is not sent to Miles islands) is not relied on.
    check_unverified_allowance(
        getattr(args, "rl_allow_unverified_mechanism", None) or (),
        islands=int(getattr(args, "num_learners", 1) or 1),
        outer_sync=outer_sync,
        sync_preset=getattr(args, "sync_preset", "strict-avg"),
    )


def install_event_echo() -> bool:
    """Echo every tape record of this island to stdout (``YETO_RL_EVENT``).

    For islands whose ~/yeto-output cannot be fetched (Modal, no-sync): sets
    ``YETO_RL_ECHO_EVENTS=1`` so the single low-level tape writer
    (``yeto.rl.event_echo.append_record``, used by the driver, bridge, learner
    and adapter events) prints each line it writes; Ray workers inherit it
    (``entry.connect_island_ray``). Idempotent; True when enabled now.
    """

    from yeto.rl.engine import driver
    from yeto.rl.event_echo import echo_enabled, enable_echo

    if getattr(driver, "_ECHO_EVENTS", False):
        driver._ECHO_EVENTS = False  # would print driver records a second time
    if echo_enabled():
        return False
    enable_echo()
    # The tape writer (yeto.rl.engine.events.append_tape_event) writes through
    # event_echo.append_record, so enabling the echo is enough; the old
    # code-object probe for a pre-append_record writer is gone
    # (rl-code-quality-cleanup 1.9).
    return True


def _append_ports_event(args, miles_args, event: dict) -> None:
    from yeto.rl.engine.events import append_tape_event as _append_rl_event

    miles_args.yeto_rl_event_tape = args.event_tape
    miles_args.yeto_rl_learner_id = args.learner_id
    _append_rl_event(miles_args, event)


class AlgorithmMismatchError(RuntimeError):
    """This island's algorithm hash differs from the launcher's expectation."""


def verify_ports_algorithm(args, miles_args, launch) -> None:
    """D9/D11: before the island joins outer sync (no bridge exists yet).

    Plugins are re-hashed and imported (a stale source hash: warning event,
    rl-identity-tiering L9.2); the island's algorithm hash is
    compared with ``--rl-expected-algorithm-sha256`` (mismatch: event +
    refusal; missing: warning event, for manually started islands);
    absorbed flags and unverified allowances are recorded.
    """

    algorithm = launch.algorithm
    for warning in algorithm.verify_plugins():
        _append_ports_event(args, miles_args, {
            "event": "rl_plugin_source_mismatch",
            "level": "warning",
            "message": warning,
        })
    from yeto.rl.engine.algorithm import island_problems

    problems = island_problems(algorithm, {
        "base_model_revision": getattr(args, "model_revision", None),
        "base_model": getattr(args, "model", None),
        "ref_load_override": getattr(args, "megatron_ref_load", None),
        # rl-algo-critic-family 2.3 (critic run-level rejections)
        "sync_preset": getattr(args, "sync_preset", "strict-avg"),
        "elastic": bool(getattr(args, "rl_elastic", False)),
    })
    if problems:
        _append_ports_event(args, miles_args, {
            "event": "rl_algorithm_island_rejected",
            "rl/algorithm_spec_sha256": launch.algorithm_sha256,
            "problems": problems,
        })
        raise AlgorithmMismatchError(
            f"island {args.learner_id} algorithm spec rejected before joining outer sync: "
            + "; ".join(problems)
        )
    actual = launch.algorithm_sha256
    expected = getattr(args, "rl_expected_algorithm_sha256", None)
    if expected is None:
        _append_ports_event(args, miles_args, {
            "event": "rl_algorithm_expected_hash_missing",
            "level": "warning",
            "rl/algorithm_spec_sha256": actual,
        })
    elif expected.lower() != actual:
        _append_ports_event(args, miles_args, {
            "event": "rl_algorithm_mismatch",
            "rl/algorithm_spec_sha256": actual,
            "rl/expected_algorithm_spec_sha256": expected.lower(),
            "rl/algorithm_spec": algorithm.canonical_json(),
        })
        raise AlgorithmMismatchError(
            f"island {args.learner_id} algorithm hash {actual} differs from the launcher's "
            f"expected {expected.lower()}; refusing to join outer sync"
        )
    _append_correction_record(args, miles_args, algorithm)
    miles_args.yeto_rl_algorithm_absorbed_flags = dict(launch.absorbed_flags)
    miles_args.yeto_rl_outer_sync = not getattr(args, "rl_single_island_no_sync", False)
    # INFRA compares it with the runtime AlgorithmSpec before connect_island_ray
    # (a partitioned run without it is refused there).
    miles_args.yeto_rl_expected_algorithm_sha256 = (
        expected.lower() if expected is not None else None
    )
    miles_args.yeto_rl_unverified_mechanisms = tuple(
        sorted(set(getattr(args, "rl_allow_unverified_mechanism", None) or ()))
    )


def _append_correction_record(args, miles_args, algorithm) -> None:
    """rl-offpolicy-correction-passthrough D8: the island tape records the full
    correction group, the Miles values and ``prox_anchor``. A field Miles
    cannot map refuses the start (phase 2, L6b.2), after the record and its
    ``correction_config_warning`` events are on the tape."""
    from yeto.rl.engine.offpolicy_correction import (CORRECTION_RECORD_EVENT, correction_record,
                                                     refuse_correction_warnings)

    from .offpolicy_correction import BACKEND, correction_mapping

    if algorithm.correction.is_default():
        return  # default tapes unchanged; the run manifest still records the group
    mapping = correction_mapping(algorithm)
    _append_ports_event(args, miles_args, {
        "event": CORRECTION_RECORD_EVENT,
        **correction_record(algorithm.correction, mapping, backend=BACKEND),
    })
    warnings = [warning.event() for warning in mapping.warnings]
    for warning in warnings:
        _append_ports_event(args, miles_args, warning)
    refuse_correction_warnings(warnings)


def _require_ports_supported(args, extra_argv: Sequence[str] = ()) -> None:
    """Reject combinations outside the R0 ports matrix before any startup."""

    from yeto.rl.engine.selection import require_ports_supported

    require_ports_supported(
        sync_preset=getattr(args, "sync_preset", "strict-avg"),
        parameter_mode=getattr(args, "parameter_mode", "lora"),
        expert_full_count=getattr(args, "expert_full_count", 0) or 0,
        rollout_num_gpus=getattr(args, "rollout_num_gpus", None),
        placement=getattr(args, "rl_placement", "colocated"),
        island_scheduling="elastic" if getattr(args, "rl_island_pool", False) else "legacy",
        extra_argv=tuple(extra_argv),
    )


_canonical_sha256 = identity_sha256  # yeto/rl/engine/full_parameter_identity.py


# Miles flags whose values may differ between islands of one full-parameter run:
# routing ports, the rollout seed, the training data (P3: each island may train
# on its own task list) and evaluation (P3-Q3: eval runs on the eval island, so
# training islands may carry none, or their own). Value counts: one, except
# where listed in the two maps below.
_DENSE_ISLAND_LOCAL_MILES_FLAGS = frozenset(
    {
        "--prompt-data",
        "--rollout-seed",
        "--rollout-engine-base-port",
        "--session-server-port",
        "--sglang-router-port",
        "--sglang-router-prometheus-port",
        "--train-master-base-port",
        "--eval-interval",
        "--n-samples-per-eval-prompt",
        "--eval-temperature",
        "--eval-top-p",
        "--eval-max-prompt-len",
        "--eval-max-response-len",
        "--eval-max-context-len",
    }
)

# One or more values (up to the next flag).
_DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS = frozenset(
    {
        "--session-server-port",
        "--eval-prompt-data",  # name + path
    }
)

# No value.
_DENSE_ISLAND_LOCAL_MILES_SWITCHES = frozenset(
    {
        "--skip-eval-before-train",
        "--log-passrate",
    }
)

# Placement flags left out of the contract only when the launch gives islands
# their own placement (P4 --rl-per-island-placement): the GPU split, trainer
# shape and weight transfer path then differ by design. A uniform launch keeps
# them in, so islands with different GPU splits are still refused. The rollout
# engine size (--rollout-num-gpus-per-engine) always stays in: it sets the
# engine TP, which can change sampling numerics, and has no per-island value.
_DENSE_PLACEMENT_MILES_FLAGS = frozenset(
    {
        "--actor-num-nodes",
        "--actor-num-gpus-per-node",
        "--num-gpus-per-node",
        "--rollout-num-gpus",
        "--update-weight-transfer-mode",
        "--yeto-placement-map",
    }
)
_DENSE_PLACEMENT_MILES_SWITCHES = frozenset(
    {
        "--colocate",
        "--offload-train",
        "--no-offload-train",
    }
)

# Parallel-shape flags left out of the contract only with a shape-free
# full-parameter layout (--rl-sync-layout shape-free): each island then trains
# with its own TP/PP/CP/EP split and micro-batching, which change how the
# global batch is computed but not what it computes. Expert tensor parallelism
# stays in (the shape-free layout needs 1 everywhere).
_SHAPE_FREE_PARALLEL_MILES_FLAGS = frozenset(
    {
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--context-parallel-size",
        "--expert-model-parallel-size",
        "--decoder-first-pipeline-num-layers",
        "--decoder-last-pipeline-num-layers",
        "--num-layers-per-virtual-pipeline-stage",
        "--max-tokens-per-gpu",
        "--micro-batch-size",
    }
)
_SHAPE_FREE_PARALLEL_MILES_SWITCHES = frozenset({"--sequence-parallel"})

_DENSE_FULL_QUANTIZATION_CONFIG_KEYS = frozenset(
    {
        "compression_config",
        "modelopt_quant",
        "quantization",
        "quantization_config",
        "torchao_config",
    }
)


def _require_env(name: str, value: str) -> None:
    current = os.environ.get(name)
    if current not in (None, value):
        raise ValueError(f"{name} must be unset or {value}, got {current!r}")
    os.environ[name] = value


def _configure_deepseek_v4_env(args, *, clone_only_lora: bool, expert_full: bool) -> None:
    """Process-wide switches the DeepSeek V4 runtime patches read (sitecustomize).

    Ray workers inherit them.
    """

    from yeto.rl.adapters.miles.models.deepseek_v4.ports_runtime import (
        CLONE_ONLY_ENV_FLAG,
        ENV_FLAG,
    )

    if clone_only_lora or expert_full:
        _require_env("YETO_DSV4_EXPERT_CLONE", "1")
    if clone_only_lora:
        _require_env(CLONE_ONLY_ENV_FLAG, "1")
    if clone_only_lora or expert_full:
        if os.environ.get("SGLANG_OPT_FUSE_WQA_WKV") not in (None, "0"):
            raise ValueError("SGLANG_OPT_FUSE_WQA_WKV must be unset or 0 for V4 attention LoRA")
        os.environ["SGLANG_OPT_FUSE_WQA_WKV"] = "0"
    if expert_full:
        _require_env(ENV_FLAG, "1")
        _require_env("YETO_DSV4_EXPERT_FULL_COUNT", str(args.expert_full_count))
        _require_env("YETO_DSV4_EXPERT_FULL_LR", str(args.expert_full_lr))
        _require_env("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "0")


def _trains_base_parameters(args) -> bool:
    """Full-parameter or DeepSeek V4 expert-full: the full-parameter sync carries the cut."""

    return (
        getattr(args, "parameter_mode", "lora") == "full"
        or int(getattr(args, "expert_full_count", 0) or 0) > 0
    )


_is_sha256 = is_sha256  # yeto/rl/engine/full_parameter_identity.py
_full_parameter_identity = full_parameter_identity  # the same rule for every engine


def _no_sync_island(args) -> bool:
    """A no-sync island (a sampler / trainer island, L4.6, or a single island
    with both roles, S26 E11): it has no outer sync, so full parameter mode runs
    without the full-parameter sync preset."""
    return bool(getattr(args, "rl_single_island_no_sync", False))


def _full_parameter_mode(args, sync_preset: str) -> bool:
    """True for a full-parameter island.

    Full parameter mode needs a full-parameter preset, and a full-parameter
    preset needs full parameter mode, except that the full-parameter sweep also
    carries the DeepSeek V4 expert-full LoRA hybrid (P0 1.7b.2).
    """

    from yeto.rl.engine.selection import FULL_PARAMETER_SYNC_PRESET, FULL_PARAMETER_SYNC_PRESETS

    full_mode = getattr(args, "parameter_mode", "lora") == "full"
    full_preset = sync_preset in FULL_PARAMETER_SYNC_PRESETS
    if full_mode and not full_preset and not _no_sync_island(args):
        raise ValueError("full parameter mode needs a full-parameter sync preset "
                         "(or --rl-single-island-no-sync)")
    if full_preset and not _trains_base_parameters(args):
        raise ValueError(f"the {sync_preset} sync needs full parameter mode or expert-full tuning")
    if full_preset and not full_mode and sync_preset != FULL_PARAMETER_SYNC_PRESET:
        raise ValueError(f"the {sync_preset} sync needs full parameter mode")
    return full_mode


def _check_full_parameter_sync(args, *, yeto_policy_sync: bool) -> None:
    """Every full-parameter island runs the ports full-parameter sync (P0-D8)."""

    if not yeto_policy_sync:
        raise ValueError("full-parameter training requires Yeto policy sync")
    if getattr(args, "eval_only", False):
        raise ValueError("full-parameter mode does not run eval-only")


def _check_dense_full_sweep(args, *, yeto_policy_sync: bool) -> None:
    """Dense full-parameter GRPO is the strict-sweep profile of the ports
    full-parameter sync (P0-D8): one role, H=1, whole-round fragment sweeps.

    The Milestone-1 limits (version-0 start, generation-0 roster, pinned Miles
    source tree) are gone (P0-D7); the syncer's authoritative fragments decide
    the start version and the launcher owns the roster.
    """

    _check_full_parameter_sync(args, yeto_policy_sync=yeto_policy_sync)
    if _full_parameter_async(args):
        # S27 P2 asynchronous rounds: H local rounds per fragment push, at
        # most FULL_PARAMETER_ASYNC_MAX_PIPELINE fragment rounds in flight;
        # no sweep-only options.
        if args.optimizer_steps != 1 or getattr(args, "local_horizon", 1) < 1:
            raise ValueError("dense full-parameter asynchronous rounds need H >= 1 "
                             "and one optimizer step")
        if not 1 <= getattr(args, "pipeline", 1) <= min(FULL_PARAMETER_ASYNC_MAX_PIPELINE, args.fragments):
            raise ValueError("dense full-parameter asynchronous rounds need --pipeline "
                             f"1..min({FULL_PARAMETER_ASYNC_MAX_PIPELINE}, P)")
        if getattr(args, "sweep_overlap", False):
            raise ValueError("--sweep-overlap needs the strict sweep")
    elif args.optimizer_steps != 1 or getattr(args, "local_horizon", 1) != 1:
        raise ValueError("the dense full-parameter strict sweep requires H=1")
    elif getattr(args, "rl_sync_matrix_merge", "avg") != "avg":
        raise ValueError("the dense full-parameter strict sweep merges by plain averaging")
    elif getattr(args, "sweep_expert_token_weights", False):
        raise ValueError("--sweep-expert-token-weights needs asynchronous rounds "
                         "(the strict sweep merges with equal weights)")
    elif int(getattr(args, "rl_max_policy_age", 0) or 0) != 0:
        # The sweep publishes whole rounds in lockstep with the syncer. The
        # asynchronous rounds take a policy-age limit counted in local
        # publications (yeto.rl.engine.full_parameter_sync module doc).
        raise ValueError("the dense full-parameter strict sweep requires --rl-max-policy-age 0")
    if (
        type(args.global_rounds) is not int
        or args.global_rounds < 1
        or type(args.fragments) is not int
        or args.fragments < 1
        or args.total_fragment_steps != args.global_rounds * args.fragments
        or not _is_sha256(args.parameter_layout_sha256)
    ):
        raise ValueError("dense full-parameter fragment budget must equal rounds*fragments")
    if not _is_sha256(getattr(args, "syncer_profile_sha256", None)):
        raise ValueError("dense full-parameter mode requires --syncer-profile-sha256")


def _private_trajectory_evidence_dir(args) -> Path:
    """A fresh owner-only directory for this island's trajectory evidence."""

    parent = (
        Path(args.audit_dir).expanduser()
        if getattr(args, "audit_dir", None)
        else Path(args.completed_groups_path).expanduser().parent
    )
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix=f"trajectory-evidence-{args.learner_id}-", dir=parent)
    )
    if directory.stat().st_mode & 0o077:
        raise RuntimeError("trajectory evidence directory is not private")
    return directory


def _roster_generations(args) -> tuple[int, ...]:
    """The launcher's island roster: every peer at generation 0, this island at its own."""

    generations = [0] * int(getattr(args, "num_learners", 1))
    generations[args.learner_id] = int(getattr(args, "learner_generation", 0))
    return tuple(generations)


def _full_parameter_async(args) -> bool:
    """S27 P2: ``--full-parameter-rounds async`` (asynchronous sharded rounds)."""
    return getattr(args, "full_parameter_rounds", "sweep") == "async"


def _local_init(args, async_rounds: bool) -> bool:
    """Local init (S27): every island launched by one run loads the same base
    checkpoint, so async rounds keep it as version 0 by default; the syncer
    refuses the run if the islands' fragment digests disagree."""
    return local_init_rule(getattr(args, "init_mode", "auto"), async_rounds)


def _dense_full_sync_config(
    args, *, model_config_sha256: str, training_contract_sha256: str, algorithm_spec_sha256: str
):
    """The dense full-parameter sync as a one-role :class:`FullParameterSyncConfig`:
    the strict sweep, or (S27 P2) the streaming profile with a partial quorum."""

    from yeto.rl.adapters.miles.full_parameter.role_stream import RoleStreamConfig
    from yeto.rl.adapters.miles.full_parameter.session import FullParameterSyncConfig
    from yeto.rl.local_learner import ComponentIdentity

    async_rounds = _full_parameter_async(args)
    syncers = _syncer_addresses(args.syncer) if async_rounds else [_syncer_address(args.syncer)]
    actor = RoleStreamConfig(
        algorithm="grpo",
        role="actor",
        component=ComponentIdentity(
            role="actor",
            model_revision=args.model_revision.lower(),
            config_hash=model_config_sha256,
        ),
        syncer_addr=syncers[0],
        syncer_shards=tuple(syncers) if len(syncers) > 1 else (),
        learner_id=args.learner_id,
        learner_generation=int(getattr(args, "learner_generation", 0)),
        learner_generations=_roster_generations(args),
        total_fragment_steps=args.total_fragment_steps,
        expected_fragments=args.fragments,
        expected_layout_hash=args.parameter_layout_sha256,
        local_horizon=int(args.local_horizon) if async_rounds else 1,
        optimizer_steps_per_round=args.optimizer_steps,
        training_contract_hash=training_contract_sha256,
        syncer_profile_hash=args.syncer_profile_sha256,
        pipeline_depth=int(args.pipeline) if async_rounds else 1,
        algorithm_spec_sha256=algorithm_spec_sha256,
        wait_timeout=_sweep_wait_timeout_s(args),
        wan_streams=args.wan_streams,
        wan_stream_schedule=getattr(args, "wan_stream_schedule", "rr"),
        wan_stall_timeout=float(getattr(args, "wan_stall_timeout", 0.0) or 0.0),
        push_dtype=getattr(args, "sweep_push_dtype", "f32"),
        push_int8_block=int(getattr(args, "sweep_push_int8_block", 128)),
        max_fragment_bytes=int(getattr(args, "sweep_fragment_max_bytes", 2 << 30)),
        push_int4_block=int(getattr(args, "sweep_push_int4_block", 128)),
        push_tensor_blocks=not getattr(args, "sweep_push_no_tensor_blocks", False),
        push_int4_stochastic=bool(getattr(args, "sweep_push_int4_stochastic", False)),
        push_int4_error_feedback=getattr(args, "sweep_push_int4_error_feedback", "off"),
        expert_token_weights=bool(getattr(args, "sweep_expert_token_weights", False)),
        final_cut=getattr(args, "sweep_final_cut", "download"),
        verify_ahead=getattr(args, "sweep_verify_ahead", "auto") != "off",
        bcast_delta=bool(getattr(args, "bcast_delta", False)),
        # The rollout policy age stays 0 (``_check_dense_full_sweep``); the
        # overlap's outer-step lag is bound into the session contract.
        overlap=bool(getattr(args, "sweep_overlap", False)),
        partial_quorum=async_rounds,
        local_init=_local_init(args, async_rounds),
        matrix_merge=getattr(args, "rl_sync_matrix_merge", "avg"),
        shape_free=shape_free_granularity(args),
        peer_recovery=peer_recovery_options(args),
        # an island pool sets the epoch (and lease); else peer recovery may
        **{"syncer_epoch": _island_syncer_epoch(args), **_island_pool_membership(args)},
    )
    return FullParameterSyncConfig((actor,), sweep=not async_rounds,
                                   max_local_rounds_ahead=_max_local_rounds_ahead(args, async_rounds))


def _max_local_rounds_ahead(args, async_rounds: bool) -> int | None:
    """``--sweep-max-local-rounds-ahead`` (launcher ``--rl-sync-max-local-rounds-ahead``):
    the S27 G-AG lead cap of the asynchronous rounds; None (off) by default."""
    value = getattr(args, "sweep_max_local_rounds_ahead", None)
    if value is None:
        return None
    if not async_rounds:
        raise ValueError("--sweep-max-local-rounds-ahead needs --full-parameter-rounds async")
    if type(value) is not int or not 1 <= value <= MAX_LOCAL_ROUNDS_AHEAD:
        raise ValueError(f"--sweep-max-local-rounds-ahead must be an integer within 1..{MAX_LOCAL_ROUNDS_AHEAD}")
    return value


def _sweep_wait_timeout_s(args) -> float:
    """``--sweep-wait-timeout-s`` (launcher ``--rl-sync-wait-timeout-s``); 900 s by default."""
    value = float(getattr(args, "sweep_wait_timeout_s", 900.0) or 900.0)
    if not 0 < value <= 7 * 86400:
        raise ValueError("--sweep-wait-timeout-s must be within (0, 604800] seconds")
    return value


PEER_RECOVER_ENV = "YETO_RL_PEER_RECOVER"  # "1": the launcher relaunched this island (launch.peer_recovery)


def _island_syncer_epoch(args) -> int | None:
    """S27 E-3: the HELLO carries the syncer epoch only when the launcher sent one."""
    value = getattr(args, "rl_syncer_epoch", None)
    return None if value is None or getattr(args, "rl_peer_recovery_store", None) is None else int(value)


def peer_recovery_options(args, environ=None):
    """S27 E-3: peer recovery of a full-parameter island, or None without
    ``--rl-peer-recovery-store``. The island recovers when ``--rl-peer-recover``
    is given or the launcher relaunched it (:data:`PEER_RECOVER_ENV`)."""
    from yeto.rl.adapters.miles.full_parameter.peer_recovery import PeerRecoveryOptions

    store = getattr(args, "rl_peer_recovery_store", None)
    env = os.environ if environ is None else environ
    recover = bool(getattr(args, "rl_peer_recover", False)) or env.get(PEER_RECOVER_ENV) == "1"
    if store is None:
        if recover or getattr(args, "rl_peer_recovery_staging", None):
            raise ValueError("--rl-peer-recover / --rl-peer-recovery-staging need --rl-peer-recovery-store")
        return None
    ttl = getattr(args, "rl_peer_recovery_ttl_s", None)
    return PeerRecoveryOptions(
        store=str(store),
        recover=recover,
        ttl_s=3600.0 if ttl is None else float(ttl),
        staging=getattr(args, "rl_peer_recovery_staging", None),
    )


def _island_pool_membership(args) -> dict:
    """Island pool (``--rl-island-pool``, set by the launcher with any
    --rl-sync-* membership option): the syncer lease the session heartbeats
    for and the syncer epoch of its HELLO. {} for a fixed roster."""
    if not getattr(args, "rl_island_pool", False):
        return {}
    from yeto.rl.pool_client import DEFAULT_ISLAND_LEASE_S

    lease = getattr(args, "rl_sync_island_lease_s", None)
    return {"island_lease_s": DEFAULT_ISLAND_LEASE_S if lease is None else float(lease),
            "syncer_epoch": int(getattr(args, "rl_syncer_epoch", 0) or 0)}


def _load_sao_streaming_runtime(args):
    """Load the hash-bound SAO streaming contract before any GPU work.

    The role pairing is checked when the runtime is bound (still before Ray
    starts), once the island's algorithm spec hash is known.
    """

    from yeto.rl.sao_streaming_runtime import load_sao_streaming_runtime

    runtime = load_sao_streaming_runtime(
        args.sao_streaming_runtime,
        expected_sha256=args.sao_streaming_runtime_sha256,
        expected_sao_context_sha256=args.sao_context_sha256,
    )
    if runtime.actor.learner_id != args.learner_id:
        raise ValueError("SAO streaming runtime names a different learner")
    if not getattr(args, "critic_syncer", None):
        raise ValueError("SAO streaming requires --critic-syncer")
    runtime = runtime.with_syncer_addresses(
        actor=_syncer_address(args.syncer),
        critic=_syncer_address(args.critic_syncer),
    )
    return runtime


def _require_unquantized_full_parameter_rollout_model(model_path: str | Path) -> None:
    """Enable the pinned SGLang BF16 hook fallback only for an exact safe model.

    The public SGLang pin used by the Milestone-1 gate predates the optional
    begin/end weight-update endpoints.  Miles independently rechecks its live
    server arguments before accepting a missing endpoint; this early check
    prevents Yeto from opting into that compatibility path for a quantized
    rollout model in the first place.
    """

    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise ValueError("full-parameter rollout model has no config.json")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate rollout config key: {name!r}")
            result[name] = value
        return result

    try:
        config = json.loads(
            config_path.read_bytes(),
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("full-parameter rollout config is malformed") from exc
    if not isinstance(config, dict):
        raise TypeError("full-parameter rollout config must be an object")
    configured = sorted(
        name
        for name in _DENSE_FULL_QUANTIZATION_CONFIG_KEYS
        if config.get(name) is not None
    )
    if configured:
        raise ValueError(
            "legacy SGLang weight-update hooks require an unquantized rollout "
            f"model; configured fields: {', '.join(configured)}"
        )


def shape_free_granularity(args: Any):
    """The island's shape-free granularity (``--rl-sync-shape-free-*``), or None
    with the shape-bound layout (``--rl-sync-layout``, default shape-bound)."""

    from yeto.rl.engine.shape_free_layout import LAYOUT_SHAPE_FREE, Granularity

    if getattr(args, "rl_sync_layout", "shape-bound") != LAYOUT_SHAPE_FREE:
        return None
    return Granularity(
        tp_units=int(getattr(args, "rl_sync_shape_free_tp_units", 1)),
        expert_classes=int(getattr(args, "rl_sync_shape_free_expert_classes", 8)),
        dense_groups=int(getattr(args, "rl_sync_shape_free_dense_groups", 8)),
        split_layers=bool(getattr(args, "rl_sync_shape_free_split_layers", False)),
    )


def shape_free_layout_contract(args: Any) -> dict[str, Any] | None:
    """The shape-free granularity as a contract entry; None with the shape-bound layout."""

    granularity = shape_free_granularity(args)
    if granularity is None:
        return None
    return {
        "tp_units": granularity.tp_units,
        "expert_classes": granularity.expert_classes,
        "dense_groups": granularity.dense_groups,
        "split_layers": granularity.split_layers,
    }


def _expert_full_contract(args: Any) -> dict[str, Any]:
    """Expert-full selection that reaches Miles through the environment, not argv.

    Empty for dense full-parameter runs, so their contract hash is unchanged.
    """

    count = int(getattr(args, "expert_full_count", 0) or 0)
    if not count:
        return {}
    return {
        "expert_full": {
            "count": count,
            "lr": float(args.expert_full_lr),
            "selection_sha256": args.expert_selection_sha256,
            "selection_contract_sha256": args.expert_selection_contract_sha256,
        }
    }


def _lora_config_hash(args: Any, canonical_targets: Sequence[str]) -> str:
    """The LoRA identity hash; expert-full adds its expert selection.

    Plain LoRA keeps the canonical hash unchanged, so existing identities hold.
    """

    from yeto.rl.core import canonical_lora_config_hash

    base = canonical_lora_config_hash(rank=args.lora_r, target_modules=canonical_targets)
    expert_full = _expert_full_contract(args)
    if not expert_full:
        return base
    return _canonical_sha256(
        {"schema": "yeto-lora-expert-full-config-v1", "lora_config_sha256": base, **expert_full}
    )


def _dense_shared_miles_argv(
    miles_argv: Sequence[str], *, per_island_placement: bool = False, shape_free: bool = False
) -> list[str]:
    """``miles_argv`` without the island-local flags and their values.

    ``per_island_placement`` also drops the placement flags and switches;
    ``shape_free`` (a shape-free full-parameter layout) the parallel-shape ones.
    """

    local_flags = _DENSE_ISLAND_LOCAL_MILES_FLAGS | _DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS
    local_switches = _DENSE_ISLAND_LOCAL_MILES_SWITCHES
    if per_island_placement:
        local_flags |= _DENSE_PLACEMENT_MILES_FLAGS
        local_switches |= _DENSE_PLACEMENT_MILES_SWITCHES
    if shape_free:
        local_flags |= _SHAPE_FREE_PARALLEL_MILES_FLAGS
        local_switches |= _SHAPE_FREE_PARALLEL_MILES_SWITCHES
    shared_argv = []
    index = 0
    while index < len(miles_argv):
        value = miles_argv[index]
        index += 1
        if value in local_switches:
            continue
        if value not in local_flags:
            shared_argv.append(value)
            continue
        value_count = 0
        while index < len(miles_argv) and not miles_argv[index].startswith("--"):
            index += 1
            value_count += 1
            if value not in _DENSE_ISLAND_LOCAL_MILES_NARGS_FLAGS:
                break
        if value_count < 1:
            raise ValueError(f"Miles flag has no value: {value}")
    return shared_argv


def _dense_full_training_contract_sha256(
    args: Any,
    miles_argv: Sequence[str],
    *,
    model_config_sha256: str,
) -> str:
    """Hash shared training semantics while excluding island-local routing,
    data, eval and (with per-island placement) placement.

    Each island may train on its own task list (P3), so the training data
    (``--data``, its revision and the prompt file) stays out of the contract.
    Evaluation does not change the trained weights and runs on the eval island
    (P3-Q3), so its settings stay out too. With per-island placement (P4) the
    placement flags stay out as well; ``per_island_placement`` records that (v6).
    """

    per_island_placement = bool(getattr(args, "rl_per_island_placement", False))
    shape_free = shape_free_layout_contract(args)
    shared_argv = _dense_shared_miles_argv(
        miles_argv, per_island_placement=per_island_placement, shape_free=shape_free is not None
    )
    explicit_rollout_seed = getattr(args, "rollout_seed", None)
    rollout_seed_contract = (
        {"mode": "fixed", "value": explicit_rollout_seed}
        if explicit_rollout_seed is not None
        else {"mode": "base-plus-learner-id", "base_seed": args.seed}
    )
    return _canonical_sha256(
        {
            **_expert_full_contract(args),
            # A shape-bound layout keeps its v6 contract hash unchanged.
            **({} if shape_free is None else {"shape_free_layout": shape_free}),
            "schema": "yeto-dense-full-training-contract-v6",
            "shared_miles_argv": shared_argv,
            "per_island_placement": per_island_placement,
            "rollout_seed_contract": rollout_seed_contract,
            "model": args.model,
            "model_revision": args.model_revision.lower(),
            "model_config_sha256": model_config_sha256,
            "yeto_source_sha256": args.source_sha256,
            "miles_backend_identity_sha256": _backend_identity_sha256(args),
            "reward_sha256": args.reward_sha256,
            # The Codex harness contract left with the harness (S26 E0.5); the
            # key stays so the v6 contract hash of existing runs is unchanged.
            "codex_harness_contract": None,
        }
    )


def miles_revision_pins(args: Any) -> Any:
    """The Miles commit the island checks for ``--miles-commit`` (full SHA); None
    without it (``verify_miles_revision`` then checks the pinned MILES_NEXT_PINS)."""
    from yeto.provenance import is_immutable_commit
    from yeto.rl.adapters.miles.pins import (
        MILES_NEXT_FORMER_REPOSITORIES,
        MILES_NEXT_REPOSITORY,
        MilesRevisionPins,
    )

    commit = getattr(args, "miles_commit", None)
    if not commit:
        return None
    if not is_immutable_commit(commit) or len(str(commit)) != 40:
        raise ValueError(f"--miles-commit must be a full 40-hex commit, got {commit!r}")
    return MilesRevisionPins(
        MILES_NEXT_REPOSITORY, str(commit).lower(), MILES_NEXT_FORMER_REPOSITORIES
    )


def _miles_callable(spec: str) -> str:
    module, separator, function = spec.partition(":")
    if not separator or not module or not function.isidentifier():
        raise ValueError("RL callable must be package.module:function")
    return f"{module}.{function}"


def _column(row: dict[str, Any], column: str, flag: str) -> Any:
    """A user-named column; a row without it is an error naming the flag,
    never a silent fallback to another column."""
    if column not in row:
        raise ValueError(f"RL rows have no column {column!r} ({flag})")
    return row[column]


def _messages(row: dict[str, Any], prompt_column: str | None = None) -> list[dict[str, Any]]:
    if prompt_column is not None:
        value = _column(row, prompt_column, "--rl-prompt-column")
    else:
        value = row.get("messages", row.get("prompt", row.get("input")))
    if isinstance(value, str):
        value = [{"role": "user", "content": value}]
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, dict) for item in value)
    ):
        raise ValueError("RL rows must contain messages or a string prompt/input")
    return value


def _prompt_split(source, revision: str | None) -> str:
    """`train`, or a Hub dataset's only split when it has no `train`
    (``yeto.data.prompt_split``; the launcher's label check reads the same split)."""
    from yeto.data import prompt_split

    return prompt_split(source, revision)


def prepare_prompt_data(
    source: str,
    revision: str | None,
    output_path: str | Path,
    prompt_column: str | None = None,
    label_column: str | None = None,
    slot: Any = None,
    number_rows: bool = False,
) -> Path:
    """Write ``source`` (local file or Hub dataset) as Miles prompt rows.

    ``slot`` (``data_split.DataSlot``, S27 task assignment): keep only the
    slot's rows, in data order; each kept row records its row number in the
    unsplit data as ``metadata["yeto_data_row"]``. ``number_rows`` (the task
    dispenser, S27 P2): keep every row and record every row number."""
    from yeto.data import load_rows
    from yeto.rl.engine.data_split import DATA_ROW_KEY

    split = _prompt_split(source, revision)
    if split == "train":
        rows = load_rows(source, revision=revision)
    else:
        rows = load_rows(source, split=split, revision=revision)
    split_rows = slot is not None and not slot.whole
    if split_rows and number_rows:
        raise ValueError("a data slot and the task dispenser exclude each other")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for number, raw in enumerate(rows):
            if split_rows and not slot.owns(number):
                continue
            row = dict(raw)
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"messages", "metadata"}
                }
            )
            if split_rows or number_rows:
                metadata[DATA_ROW_KEY] = number
            normalized = {
                "messages": _messages(row, prompt_column),
                "label": (
                    _column(row, label_column, "--rl-label-column")
                    if label_column is not None
                    else row.get("label")
                ),
                "metadata": metadata,
            }
            if "tools" in row:
                normalized["tools"] = row["tools"]
            handle.write(
                json.dumps(
                    normalized,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("RL prompt dataset is empty" if not split_rows
                         else f"data slot {slot.env()} of the RL prompt dataset is empty")
    os.replace(temporary, output)
    return output


def _verify_island_task_data(args) -> dict[str, Any] | None:
    """rl-per-island-tasks: check this island's own prompt file against the
    launcher's SHA256 before any training; None when the island shares ``--data``."""

    expected = getattr(args, "data_sha256", None)
    if not expected:
        return None
    from yeto.rl.island_tasks import verify_island_data

    return verify_island_data(args.data, expected)


def _island_dispense(args) -> bool:
    """S27 P2: the island claims its rows from the actor syncer's dispenser
    (``--rl-data-dispense``). Refused where it does not apply (as the slot)."""

    if not getattr(args, "rl_data_dispense", False):
        return False
    from yeto.rl.engine import sample_roles

    if getattr(args, "rl_data_slot", None):
        raise ValueError("--rl-data-dispense and --rl-data-slot exclude each other")
    if getattr(args, "data_sha256", None):
        raise ValueError("--rl-data-dispense and per-island task data (--data-sha256) exclude each other")
    role = sample_roles.sample_role(args)
    if getattr(args, "rl_sampler_data_slot", None):
        raise ValueError("--rl-data-dispense and --rl-sampler-data-slot exclude each other")
    if role == sample_roles.SAMPLER:  # S27 P4: a sampler island claims its rows too
        if not getattr(args, "rl_dispense_addr", None):
            raise ValueError("--rl-data-dispense on a sampler island needs --rl-dispense-addr")
    elif role != sample_roles.BOTH:
        raise ValueError("--rl-data-dispense belongs to an island that generates (a training "
                         "island that generates and trains, or a sampler island)")
    if getattr(args, "eval_only", False):
        raise ValueError("--rl-data-dispense: an eval-only island evaluates every row")
    return True


def _dispense_config(args, miles_args, prompt_path) -> dict[str, Any] | None:
    """``miles_args.yeto_rl_data_dispense``: what the rollout executor needs to
    open its dispenser connection (``adapters.miles.task_dispense.install``)."""

    if not _island_dispense(args):
        return None
    if getattr(miles_args, "fully_async", False):
        raise ValueError("--rl-data-dispense does not cover Miles fully_async rollouts yet")
    from yeto.rl.engine.task_dispense import file_fingerprint

    from yeto.rl.engine import sample_roles
    from yeto.rl.engine.task_dispense import sampler_holder_id

    rows, digest = file_fingerprint(prompt_path)
    if sample_roles.sample_role(args) == sample_roles.SAMPLER:
        # S27 P4: the sampler's dispenser (a synced group's actor syncer); its island
        # id comes from its sampler id, apart from the training islands' learner ids
        host, port = _syncer_addresses(args.rl_dispense_addr)[0]
        sampler = getattr(args, "rl_sampler_node", None) or f"island{int(args.learner_id)}"
        learner_id = sampler_holder_id(sampler)
    else:
        host, port = _syncer_addresses(args.syncer)[0]  # the actor syncer (shard 0) dispenses
        learner_id = int(args.learner_id)
    block = int(getattr(miles_args, "over_sampling_batch_size", None)
                or getattr(miles_args, "rollout_batch_size"))
    config = {"address": f"{host}:{port}", "learner_id": learner_id, "rows": rows,
              "fingerprint": digest.hex(), "seed": int(getattr(miles_args, "seed", 0) or 0),
              "block": block,
              "prefetch_blocks": int(getattr(args, "rl_dispense_prefetch_blocks", 2) or 2)}
    config.update(_dispense_streams(args, prompt_path, rows, digest))
    return config


def _dispense_streams(args, prompt_path, rows: int, digest: bytes) -> dict[str, Any]:
    """S27 P3: the streams of ``--rl-dispense-mix`` / ``--rl-dispense-by-label``
    and this island's ``--rl-dispense-labels``; {} for the plain P2 data. An
    island whose labels are on no row fails here, before any training."""

    mix = getattr(args, "rl_dispense_mix", None)
    by_label = bool(getattr(args, "rl_dispense_by_label", False))
    labels = getattr(args, "rl_dispense_labels", None)
    if not (mix or by_label or labels):
        return {}
    from yeto.rl.engine.task_dispense import StreamTable

    table = StreamTable.from_prompts(prompt_path, mix=mix.split(",") if mix else None,
                                     by_label=by_label or bool(labels))
    if table.total_rows != rows or table.fingerprint != digest:
        raise ValueError("the prompt file changed while its streams were read")
    accept = table.accept(labels.split(",")) if labels else None
    if table.single and accept is None:
        return {}
    return {"streams": table.to_config(), "accept": accept, "stream_record": table.record()}


def _island_data_slot(args) -> Any:
    """S27 task assignment: the island's ``--rl-data-slot`` (``data_split.DataSlot``),
    None when the island reads every row. A slot belongs to a training island
    that reads ``--data`` itself: refused with per-island task data, on a
    sampler (``--rl-sampler-data-slot``) and on an eval-only island."""

    text = getattr(args, "rl_data_slot", None)
    if not text:
        return None
    from yeto.rl.engine import sample_roles
    from yeto.rl.engine.data_split import DataSlot

    if getattr(args, "data_sha256", None):
        raise ValueError("--rl-data-slot and per-island task data (--data-sha256) exclude each other")
    if sample_roles.sample_role(args) != sample_roles.BOTH or getattr(args, "rl_sampler_data_slot", None):
        raise ValueError("--rl-data-slot belongs to a training island that generates and trains "
                         "(a sampler reads --rl-sampler-data-slot)")
    if getattr(args, "eval_only", False):
        raise ValueError("--rl-data-slot: an eval-only island evaluates every row")
    return DataSlot.parse(text)


def _verify_eval_dataset_identity(args) -> Path | None:
    """Verify the immutable source bytes for an eval-only or training heldout run."""

    if getattr(args, "eval_interval", None) is None:
        if getattr(args, "eval_only", False) or getattr(args, "eval_data", None):
            raise ValueError("--eval-only requires evaluation configuration")
        return None
    eval_only = getattr(args, "eval_only", False)
    from yeto.rl.engine.run_config import ports_training_eval

    train_eval = ports_training_eval(args)
    if not eval_only and not train_eval:
        raise ValueError("evaluation configuration requires --eval-only or the ports engine")
    expected = str(getattr(args, "eval_data_sha256", "") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("evaluation dataset requires an immutable SHA256")
    source_value = args.data if eval_only else getattr(args, "eval_data", None)
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("training-time evaluation requires --eval-data")
    source = Path(source_value).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("evaluation requires one regular local dataset file")
    if train_eval and (
        source.resolve() == Path(args.data).expanduser().resolve()
    ):
        raise ValueError("training-time evaluation must use a distinct heldout dataset")
    from yeto.provenance import file_sha256

    actual = file_sha256(source)
    if actual != expected:
        raise ValueError(
            f"evaluation dataset SHA256 mismatch: expected {expected}, got {actual}"
        )
    return source


def miles_lr_schedule_sha256(miles_args):  # noqa: D103 - see lr_schedule.miles_lr_schedule_sha256
    from yeto.rl.adapters.miles.lr_schedule import miles_lr_schedule_sha256 as _h

    return _h(miles_args)


def _backend_identity_sha256(args) -> str:
    """Decoupling 6.2: this island's backend identity hash (bound into the syncer session)."""
    from yeto.rl.adapters.miles.identity import backend_identity
    from yeto.rl.engine.policy_age import bind_policy_age

    # agentic-rollout-utilization 2.1: a non-zero policy-age limit is bound into
    # the identity the syncer compares (limit 0: unchanged).
    identity = backend_identity(compat_group=getattr(args, "rl_compat_group", None))
    return bind_policy_age(identity.sha256(),
                           int(getattr(args, "rl_max_policy_age", 0) or 0))


from yeto.hw.catalog import HARDWARE_EVENT  # noqa: E402


def _island_code_pack_digest(args) -> str | None:
    """rl-identity-tiering: the run code pack digest (L9.4, checked at startup),
    else (L9.1 transition) the attested source tree hash. Both are lowercase
    sha256 hex; the value is the fourth slot of the island contract (L9.7)."""
    return getattr(args, "code_pack_digest", None) or getattr(args, "source_sha256", None)


def _island_compat_group(args) -> str:
    """Decoupling 7.7a: this island's compat_group, logged once with the driver
    and CUDA versions (recorded only; they never cause a refusal)."""
    from yeto.hw.catalog import COMPAT_GROUP_ENV, island_compat_group, runtime_versions

    group = island_compat_group(getattr(args, "rl_compat_group", None))
    os.environ[COMPAT_GROUP_ENV] = group  # in-process legacy model paths read it
    if not getattr(args, "_yeto_compat_logged", False):
        versions = runtime_versions()
        print(f"[yeto-island] compat_group={group} driver_version={versions['driver_version']} "
              f"cuda_version={versions['cuda_version']}", flush=True)
        if getattr(args, "event_tape", None):
            from yeto.rl.engine.driver import EventTape

            EventTape(args.event_tape, int(getattr(args, "learner_id", 0) or 0)).append(
                {"event": HARDWARE_EVENT, "compat_group": group, **versions})
        args._yeto_compat_logged = True
    return group


def _syncer_address(value: str) -> tuple[str, int]:
    """One ``HOST:PORT``; a sharded head's list is only for
    :func:`_syncer_addresses` (asynchronous full-parameter rounds)."""
    if "," in value:
        raise ValueError("a sharded syncer head (comma-separated --syncer) needs "
                         "--full-parameter-rounds async")
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError("--syncer must be HOST:PORT")
    return host, int(port)


def _syncer_addresses(value: str) -> list[tuple[str, int]]:
    """``HOST:PORT[,HOST:PORT...]``: one syncer, or every shard in shard order."""
    return [_syncer_address(item.strip()) for item in value.split(",")]


def probe_syncers(args, *, connect=None, attempts: int = 10, delay: float = 3.0,
                  timeout: float = 5.0, sleep=time.sleep) -> list[tuple[str, int]]:
    """Island-side TCP probe of every syncer this learner will dial (actor
    --syncer, critic --critic-syncer), printed with the addresses, before any
    model/Miles work. s13-g3-modal-20261007b: both Modal islands sat 15 min in
    the client's silent redial; this fails in ~30 s and names the address."""
    import socket

    connect = connect or socket.create_connection
    addrs = [addr for v in (getattr(args, "syncer", None), getattr(args, "critic_syncer", None))
             if v for addr in _syncer_addresses(v)]
    for host, port in addrs:
        last = "not tried"
        for i in range(max(1, attempts)):
            try:
                connect((host, port), timeout=timeout).close()
                print(f"[rl] syncer probe: {host}:{port} reachable (attempt {i + 1})", flush=True)
                break
            except OSError as error:
                last = f"{type(error).__name__}: {error}"
            if i + 1 < attempts:
                sleep(delay)
        else:
            raise ConnectionError(
                f"syncer {host}:{port} unreachable from this island after {attempts} attempt(s): {last}")
    return addrs


def probe_wan_link(args, *, probe=None) -> None:
    """S26 E3b: measure this container's link to the actor syncer before any
    model work; exit EXIT_SLOW_LINK below the configured rates so the launcher
    can request another container. Off unless a minimum rate is set."""
    min_up = float(getattr(args, "wan_probe_min_mbps", 0.0) or 0.0)
    min_down = float(getattr(args, "wan_probe_min_down_mbps", 0.0) or 0.0)
    if min_up <= 0 and min_down <= 0:
        return
    from yeto.protocol import island_hmac_key_from_env
    from yeto.wan_probe import EXIT_SLOW_LINK, PROBE_EVENT, PROBE_MARKER, probe_link

    result = (probe or probe_link)(
        _syncer_addresses(args.syncer)[0],  # shard 0 of a sharded head
        streams=max(1, int(args.wan_streams)),
        bytes_per_direction=int(getattr(args, "wan_probe_mib", 256)) << 20,
        key=island_hmac_key_from_env(),
    )
    accepted = result.accepted(min_up, min_down)
    record = {"event": PROBE_EVENT, "accepted": accepted,
              "wan_probe/min_upload_mbps": min_up, "wan_probe/min_download_mbps": min_down,
              **result.fields()}
    print(f"{PROBE_MARKER} {'accepted' if accepted else 'rejected'}: up {result.upload_mbps:.1f} MB/s "
          f"(min {min_up}), down {result.download_mbps:.1f} MB/s (min {min_down}), "
          f"{result.streams} stream(s){'' if result.error is None else '; ' + result.error}", flush=True)
    if getattr(args, "event_tape", None):
        from yeto.rl.engine.driver import EventTape

        if getattr(args, "rl_echo_events", False):
            install_event_echo()  # the launcher reads the verdict from the log stream
        EventTape(args.event_tape, int(getattr(args, "learner_id", 0) or 0)).append(record)
    if not accepted:
        raise SystemExit(EXIT_SLOW_LINK)


def require_ports_router_mode(miles_args) -> None:
    """Upstream Miles owns the SGLang router.

    Upstream launches the router as an ``inference-router`` Ray worker and
    removed external router mode: a pre-set ``sglang_router_ip`` without its
    per-model router map is an assertion failure, so it is refused before launch.
    """

    if getattr(miles_args, "sglang_router_ip", None) is not None:
        raise ValueError(
            "an external SGLang router (--sglang-router-ip) is not supported; "
            "upstream Miles launches its own router"
        )


def print_attestation_fingerprint(args, ports_launch, out=None) -> bool:
    """``--rl-print-attestation-fingerprint`` (A5/A9 attestation on CPU): print
    the runtime fingerprint of THIS launch -- computed by the same function
    ``run_ports_island`` uses -- as one JSON line; True when printed."""
    if not getattr(args, "rl_print_attestation_fingerprint", False):
        return False
    from yeto.rl.adapters.miles.entry import ports_runtime_fingerprint

    print(json.dumps({
        "event": "rl_attestation_fingerprint",
        "runtime_fingerprint": ports_runtime_fingerprint(ports_launch),
        "learner_id": getattr(args, "learner_id", None),
        "miles_argv": list(ports_launch.argv),
    }, sort_keys=True), file=out or sys.stdout, flush=True)
    return True


BOOT_ONLY_MARKER = "FN_BOOT_ONLY_OK"
BOOT_ONLY_PATH = "~/yeto-rl/boot_only.json"


def probe_ref_load(args, model_path) -> dict:
    """``--rl-boot-only``: the --megatron-ref-load check result, recorded, never raised."""
    from yeto.rl.engine.run_config import _resolve_ref_load

    configured = getattr(args, "megatron_ref_load", None)
    if configured is None:
        return {"configured": None, "present": False, "reason": "not configured"}
    try:
        return {"configured": configured, "present": True,
                "resolved": _resolve_ref_load(args, model_path), "reason": None}
    except ValueError as exc:
        return {"configured": configured, "present": False, "reason": str(exc)}


def write_boot_only_marker(args, ports_launch, run_config, ref_probe, *,
                           path=None, out=None) -> dict:
    """``--rl-boot-only`` (S11 fnboot): every torch_dist-free check passed (provider
    view, Miles argv built + parsed, algorithm verified); record it and return so the
    learner exits 0 without training."""
    from yeto.rl.adapters.miles.entry import ports_runtime_fingerprint

    present = bool(ref_probe and ref_probe["present"])
    record = {
        "event": "rl_boot_only",
        "marker": BOOT_ONLY_MARKER,
        "learner_id": getattr(args, "learner_id", None),
        "model_recipe": getattr(getattr(run_config, "model_recipe", None), "name", None),
        "ref_load": ref_probe,
        "ref_load_present": present,
        "miles_argc": len(ports_launch.argv),
        "runtime_fingerprint": ports_runtime_fingerprint(ports_launch),
    }
    target = Path(os.path.expanduser(path or BOOT_ONLY_PATH))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    stream = out or sys.stdout
    print(json.dumps(record, sort_keys=True), file=stream, flush=True)
    print(f"{BOOT_ONLY_MARKER} ref_load_present={str(present).lower()} "
          f"recipe={record['model_recipe']}", file=stream, flush=True)
    return record


def run_miles(
    args,
    *,
    model_path: str | Path,
    rollout_model_path: str | Path | None = None,
    prompt_path: str | Path,
    eval_prompt_path: str | Path | None = None,
    island_tasks: Mapping[str, Any] | None = None,
    yeto_policy_sync: bool = True,
    extra_argv: Sequence[str] = (),
) -> None:
    """Run one Miles job, optionally with Yeto's external policy boundary.

    ``island_tasks`` (rl-per-island-tasks) is this island's checked task list,
    recorded in ``rl_driver_start``.
    """

    _check_ports_algorithm_options(args, outer_sync=yeto_policy_sync)
    if (
        getattr(args, "rl_single_island_no_sync", False) or getattr(args, "rl_echo_events", False)
    ) and not getattr(args, "rl_critic_baseline_run", False):
        # The critic baseline run (run_critic_baseline) keeps its records in its
        # own tape file only: its rl_learner_finalized must not reach the
        # launcher's log-stream tape while this island's main run is still ahead.
        install_event_echo()
    if not getattr(args, "rl_critic_baseline_run", False):
        # launch-preflight-guards 3.4/3.6: negative-test island -> override event at
        # the head of the tape; checkpoint store negative-test marker check.
        from yeto.island_overrides import island_startup

        island_startup(args, int(getattr(args, "learner_id", 0) or 0))
    _require_ports_supported(args, extra_argv)
    from yeto.rl.adapters.miles.state import require_run_plugin

    require_run_plugin()
    from yeto.rl.engine.selection import (
        FULL_PARAMETER_SYNC_PRESET,
        FULL_PARAMETER_SYNC_PRESETS,
        SAO_STREAMING_SYNC_PRESET,
    )

    parameter_mode = getattr(args, "parameter_mode", "lora")
    sync_preset = getattr(args, "sync_preset", "strict-avg")
    full_parameter = _full_parameter_mode(args, sync_preset)
    # P0-D8: every full-parameter preset runs the one full-parameter sync; the
    # full-parameter strict sweep also carries DeepSeek V4 expert-full (P0 1.7b.2).
    full_parameter_sync = sync_preset in FULL_PARAMETER_SYNC_PRESETS
    dense_full = sync_preset == FULL_PARAMETER_SYNC_PRESET
    if _trains_base_parameters(args) and not full_parameter_sync and not _no_sync_island(args):
        raise ValueError(f"expert-full tuning needs the {FULL_PARAMETER_SYNC_PRESET} sync")
    sao_runtime = None
    if dense_full:
        _check_dense_full_sweep(args, yeto_policy_sync=yeto_policy_sync)
    elif sync_preset == SAO_STREAMING_SYNC_PRESET:
        _check_full_parameter_sync(args, yeto_policy_sync=yeto_policy_sync)
        sao_runtime = _load_sao_streaming_runtime(args)

    if (
        yeto_policy_sync
        and sync_preset == "decoupled"
        and args.optimizer_steps != 1
    ):
        raise ValueError("decoupled RL requires one optimizer step per rollout")

    clone_only_lora = (
        parameter_mode == "lora"
        and args.lora_targets == "attention-routed-experts"
    )
    expert_full = int(getattr(args, "expert_full_count", 0) or 0) > 0
    if full_parameter and expert_full:
        raise ValueError("full-parameter mode cannot also select expert-full tuning")

    if full_parameter:
        _require_unquantized_full_parameter_rollout_model(
            rollout_model_path or model_path
        )
    if full_parameter_sync:
        _require_env("MILES_EXPERIMENTAL_FT_TRAINER", "0")
    _configure_deepseek_v4_env(args, clone_only_lora=clone_only_lora, expert_full=expert_full)

    if getattr(args, "rl_model_recipe", "generic") == "deepseek-v4-flash":
        from yeto.rl.adapters.miles.models.deepseek_v4.bridge import ensure_deepseek_v4_bridge

        ensure_deepseek_v4_bridge()

    from yeto.rl.adapters.miles.models import flash_next_provider as flash_next

    native_lora_export = None
    if flash_next.is_flash_next(args, model_path):
        # try24: no predicted LoRA layout (specs == ()); the ports state plugin
        # exports through Miles' native qwen3_8_next exporter and the layout
        # hash is learned from the first export (_run_ports).
        native_lora_export = "qwen3_8_next"
        # try22: megatron.bridge AutoBridge cannot parse qwen4_exp (transformers
        # 5.12.1 has no such model type).  Miles registers the HF alias and
        # trains through its own plugin, so build a read-only provider view
        # from the aliased AutoConfig instead (FN-IMAGE-PLAN option (c)).
        if full_parameter or expert_full or clone_only_lora:
            raise ValueError("the Qwen3.8-Flash-Next recipe is native-LoRA only")
        if yeto_policy_sync:
            raise ValueError(
                "Qwen3.8-Flash-Next needs --rl-single-island-no-sync: the Yeto "
                "policy-sync LoRA layout contract is not derivable for qwen4_exp"
            )
        model_bridge = None
        provider = flash_next.flash_next_provider(
            args, model_path, trust_remote_code=args.trust_remote_code
        )
        attention_specs = specs = ()
        canonical_targets = flash_next.target_modules()
    elif getattr(args, "rl_model_recipe", "generic") == "glm5_next":
        # AutoBridge cannot read glm5_next (transformers has no such model type);
        # Miles trains it through its plugin spec, so the island reads the geometry
        # from config.json (models/glm5_next.provider_view). Full-parameter only.
        from yeto.rl.adapters.miles.models import glm5_next as glm5_next_recipe

        if not full_parameter:
            raise ValueError("the glm5_next recipe is full-parameter only")
        model_bridge = None
        provider = glm5_next_recipe.glm5_next_provider(model_path)
        attention_specs = specs = ()
        canonical_targets = []
    else:
        from megatron.bridge import AutoBridge

        from yeto.rl.adapters.miles.models.nemotron_h import register_miles_bridges

        # Nemotron-H: read the checkpoint with the Miles trainer's bridge (MTP
        # dropped; a VL checkpoint as its language model), so the recipe checks
        # the model Miles builds.
        register_miles_bridges(model_path)
        model_bridge = AutoBridge.from_hf_pretrained(
            model_path,
            trust_remote_code=args.trust_remote_code,
        )
        provider = model_bridge.to_megatron_provider(load_weights=False)
        provider.finalize()
        attention_specs = () if full_parameter else derive_peft_lora_specs(
            model_path,
            None,
            rank=args.lora_r,
            targets=args.lora_targets,
            trust_remote_code=args.trust_remote_code,
        )
        specs = tuple(attention_specs)
        if expert_full:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=args.trust_remote_code,
            )
            specs = tuple(
                sorted(
                    specs
                    + expert_full_specs(
                        config,
                        expert_count=args.expert_full_count,
                        expected_selection_sha256=args.expert_selection_sha256,
                        expected_selection_contract_sha256=(
                            args.expert_selection_contract_sha256
                        ),
                    )
                )
            )
        canonical_targets = [] if full_parameter else adapter_targets(attention_specs)

    from yeto.rl.adapters.miles.config import parse_miles_args
    from yeto.rl.engine.run_config import resolve_rl_run_config

    boot_only = bool(getattr(args, "rl_boot_only", False))
    ref_probe = probe_ref_load(args, model_path) if boot_only else None
    run_config = resolve_rl_run_config(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
        prompt_path=prompt_path,
        eval_prompt_path=eval_prompt_path,
        provider=provider,
        # Upstream Miles resolves HF module names itself (canonical_lora
        # via Bridge); the fork's per-layer Megatron names have no Bridge
        # mapping upstream.
        target_modules=canonical_targets,
        yeto_policy_sync=yeto_policy_sync,
        # boot-only: the ref-load was probed (and recorded) above; a
        # missing torch_dist must not stop the argv build.
        verify_ref_load=not boot_only,
    )
    if not boot_only and not getattr(args, "rl_print_attestation_fingerprint", False):
        run_config = _run_critic_warmup_stage(args, run_config, extra_argv)
    ports_launch = build_ports_launch(args, run_config, extra_argv)
    ports_algorithm = ports_launch.algorithm
    miles_argv = list(ports_launch.argv)
    miles_args = parse_miles_args(ports_launch)
    verify_ports_algorithm(args, miles_args, ports_launch)
    if print_attestation_fingerprint(args, ports_launch):
        return  # CPU entry: nothing below (Ray, GPU, sync) runs
    if boot_only:
        write_boot_only_marker(args, ports_launch, run_config, ref_probe)
        return  # boot-only: exit 0 before Ray/GPU work; no training

    # The full-parameter sync contracts bind the spec hash (4.8). The island
    # contract v3 (strict, decoupled, elastic) binds the semantic hash and the
    # run code pack instead (rl-identity-tiering L9.7); placement capabilities
    # stay out of both.
    algorithm_spec_sha256 = ports_algorithm.sha256()
    algorithm_semantic_sha256 = ports_algorithm.semantic_sha256()
    code_pack_digest = _island_code_pack_digest(args)
    # every island (also --rl-single-island-no-sync): the remote-code prewarm reads it
    miles_args.yeto_rl_trust_remote_code = args.trust_remote_code
    if yeto_policy_sync:
        if full_parameter_sync:
            from yeto.provenance import file_sha256

            model_config_path = Path(model_path) / "config.json"
            if not model_config_path.is_file():
                raise ValueError("full-parameter model has no config.json")
            model_config_hash = file_sha256(model_config_path)
            if dense_full:
                training_contract_hash = _dense_full_training_contract_sha256(
                    args, miles_argv, model_config_sha256=model_config_hash
                )
        if full_parameter:
            layout_hash, lora_config_hash = _full_parameter_identity(model_config_hash)
        else:
            from yeto.rl.core import canonical_layout_hash

            layout_hash = canonical_layout_hash(specs)
            lora_config_hash = _lora_config_hash(args, canonical_targets)
        miles_args.yeto_rl_model = args.model
        miles_args.yeto_rl_data = args.data
        miles_args.yeto_rl_base_model_revision = args.model_revision
        miles_args.yeto_rl_rollout_model_revision = (
            getattr(args, "rollout_model_revision", None) or args.model_revision
        )
        miles_args.yeto_rl_data_revision = args.data_revision
        miles_args.yeto_rl_lora_config_hash = lora_config_hash
        miles_args.yeto_rl_layout_hash = layout_hash
        miles_args.yeto_rl_parameter_mode = parameter_mode
        miles_args.yeto_rl_clone_only_lora = clone_only_lora
        if expert_full:
            miles_args.yeto_rl_expected_specs = specs
        if clone_only_lora:
            # The Bridge exposes one representative mapping per packed expert
            # parameter.  Miles needs the sparse canonical clone set at the
            # external policy boundary and reconstructs frozen originals as zeros.
            miles_args.yeto_rl_canonical_lora_names = tuple(
                spec.name for spec in specs
            )
        miles_args.yeto_rl_reward_sha256 = args.reward_sha256
        miles_args.yeto_rl_dynamic_sampling_max_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        miles_args.yeto_rl_completed_groups_path = args.completed_groups_path
        miles_args.yeto_rl_event_tape = args.event_tape
        miles_args.yeto_rl_learner_id = args.learner_id
        # W&B reads the same namespace the rest of the yeto context rides in;
        # yeto.rl.wandb_rl starts the island's run off these.
        miles_args.wandb = getattr(args, "wandb", False)
        miles_args.wandb_project = getattr(args, "wandb_project", "yeto")
        miles_args.wandb_entity = getattr(args, "wandb_entity", None)
        miles_args.wandb_mode = getattr(args, "wandb_mode", "online")
        if getattr(args, "eval_only", False):
            miles_args.yeto_rl_eval_policy_version = args.global_rounds
        miles_args.yeto_rl_sync_preset = sync_preset
        miles_args.external_policy_sync_run_until_stop = sync_preset == "decoupled"
        if dense_full:
            from yeto.rl.adapters.miles.full_parameter.session import CONFIG_ATTRIBUTE
            from yeto.rl.trajectory_evidence import REWARD_FUNCTION_EVIDENCE_KIND

            miles_args.external_policy_identity_setter_path = (
                "yeto.rl.adapters.miles.publish.set_current_published_policy_identity"
            )
            miles_args.yeto_rl_trajectory_evidence_dir = str(
                _private_trajectory_evidence_dir(args)
            )
            miles_args.yeto_rl_trajectory_evidence_kind = REWARD_FUNCTION_EVIDENCE_KIND
            miles_args.yeto_rl_trajectory_evidence_schema_version = 1
            setattr(
                miles_args,
                CONFIG_ATTRIBUTE,
                _dense_full_sync_config(
                    args,
                    model_config_sha256=model_config_hash,
                    training_contract_sha256=training_contract_hash,
                    algorithm_spec_sha256=algorithm_spec_sha256,
                ),
            )
        elif sao_runtime is not None:
            from yeto.rl.adapters.miles.full_parameter.sao_binding import (
                bind_sao_streaming_runtime,
            )

            # Two-role (actor + critic) full-parameter sync; the ports rollout
            # hook writes the round evidence the runtime directory names.
            # The contract names a fresh evidence directory under the island's
            # run root; the island creates that root, the bind the directory
            # (0o700, checked there).  A root that already exists may be
            # readable by others (the evidence directory is private), but no
            # one else may write to it: they could replace the directory.
            evidence_root = sao_runtime.trajectory_evidence_dir.parent
            evidence_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if evidence_root.stat().st_mode & 0o022:
                raise ValueError(
                    f"SAO trajectory evidence root {evidence_root} is writable by others"
                )
            bind_sao_streaming_runtime(
                miles_args, sao_runtime, algorithm_spec_sha256=algorithm_spec_sha256
            )
        elif sync_preset == "decoupled":
            from yeto.protocol import layout_fingerprint
            from yeto.rl.core import build_rl_fragment_layout

            miles_args.yeto_rl_source_sha256 = args.source_sha256
            miles_args.yeto_rl_initial_adapter = getattr(
                args, "initial_adapter", None
            )
            miles_args.yeto_rl_initial_adapter_sha256 = getattr(
                args, "initial_adapter_sha256", None
            )
            total_fragment_steps = args.total_fragment_steps
            if total_fragment_steps != args.global_rounds * args.fragments:
                raise ValueError(
                    "decoupled RL total fragment steps must equal sweeps*fragments"
                )
            matrix_merge = getattr(args, "rl_sync_matrix_merge", "avg")
            layout = build_rl_fragment_layout(
                specs, args.fragments, matrix_merge=matrix_merge
            )
            sync_fingerprint = layout_fingerprint(layout).hex()
            miles_args.yeto_rl_num_fragments = args.fragments
            miles_args.yeto_rl_pipeline = args.pipeline
            miles_args.yeto_rl_local_horizon = args.local_horizon
            miles_args.yeto_rl_total_sweeps = args.global_rounds
            miles_args.yeto_rl_total_fragment_steps = total_fragment_steps
            miles_args.yeto_rl_sync_layout_fingerprint = sync_fingerprint
            miles_args.yeto_rl_learner_budget_steps = getattr(
                args,
                "learner_budget_steps",
                None,
            )
            miles_args.yeto_rl_bridge_config = DecoupledBridgeConfig(
                syncer_addr=_syncer_address(args.syncer),
                learner_id=args.learner_id,
                total_fragment_steps=total_fragment_steps,
                num_fragments=args.fragments,
                pipeline=args.pipeline,
                local_horizon=args.local_horizon,
                expected_specs=specs,
                base_model_revision=args.model_revision,
                lora_config_hash=lora_config_hash,
                canonical_layout_hash=layout_hash,
                wan_streams=args.wan_streams,
                learner_budget_steps=miles_args.yeto_rl_learner_budget_steps,
                backend_identity_sha256=_backend_identity_sha256(args),
                lr_schedule_sha256=miles_lr_schedule_sha256(miles_args),
                algorithm_semantic_sha256=algorithm_semantic_sha256,
                code_pack_digest=code_pack_digest,
                compat_group=_island_compat_group(args),
                matrix_merge=matrix_merge,
                **_island_pool_membership(args),
            )
        else:
            miles_args.yeto_rl_bridge_config = BridgeConfig(
                syncer_addr=_syncer_address(args.syncer),
                learner_id=args.learner_id,
                global_rounds=args.global_rounds,
                groups_per_round=args.groups_per_round,
                samples_per_group=args.samples_per_group,
                local_optimizer_steps=args.optimizer_steps,
                wan_streams=args.wan_streams,
                expected_specs=specs,
                base_model_revision=args.model_revision,
                lora_config_hash=lora_config_hash,
                layout_hash=layout_hash,
                event_tape=args.event_tape,
                audit_dir=args.audit_dir,
                send_initial_params=not getattr(args, "eval_only", False),
                backend_identity_sha256=_backend_identity_sha256(args),
                lr_schedule_sha256=miles_lr_schedule_sha256(miles_args),
                algorithm_semantic_sha256=algorithm_semantic_sha256,
                code_pack_digest=code_pack_digest,
                compat_group=_island_compat_group(args),
                bcast_delta=bool(getattr(args, "bcast_delta", False)),
            )
            if getattr(args, "rl_island_pool", False):
                # a fixed roster sets nothing
                miles_args.yeto_rl_island_pool = True
                miles_args.yeto_rl_syncer_epoch = int(getattr(args, "rl_syncer_epoch", 0))
                miles_args.yeto_rl_island_lease_s = getattr(args, "rl_sync_island_lease_s", None)
            # 4.2.1: the critic syncer (second channel); its BridgeConfig needs
            # the critic specs/layout, built by build_sync once Miles created it.
            miles_args.yeto_rl_critic_syncer_addr = (
                _syncer_address(args.critic_syncer)
                if getattr(args, "critic_syncer", None) else None
            )

    _configure_grad_audit(args, miles_args)
    require_ports_router_mode(miles_args)
    miles_args.yeto_rl_island_tasks = dict(island_tasks) if island_tasks else None
    # S27 task assignment: the island's row slot (rl_driver_start, data passes) and
    # where a full-parameter island keeps its data position across processes
    slot = _island_data_slot(args)
    miles_args.yeto_rl_data_slot = None if slot is None or slot.whole else slot.env()
    miles_args.yeto_rl_data_position_store = getattr(args, "rl_data_position_store", None)
    miles_args.yeto_rl_data_dispense = _dispense_config(args, miles_args, prompt_path)
    if miles_args.yeto_rl_data_dispense is not None:
        miles_args.yeto_rl_data_position_store = None  # the dispenser keeps the position
    if full_parameter and not yeto_policy_sync:
        # A no-sync buffer island (L4.6) trains every parameter without the sync
        # block above: bind the same full-parameter identity for _run_ports.
        from yeto.provenance import file_sha256

        model_config_path = Path(model_path) / "config.json"
        if not model_config_path.is_file():
            raise ValueError("full-parameter model has no config.json")
        miles_args.yeto_rl_parameter_mode = parameter_mode
        miles_args.yeto_rl_layout_hash, miles_args.yeto_rl_lora_config_hash = (
            _full_parameter_identity(file_sha256(model_config_path)))
    miles_args.yeto_rl_code_pack_digest = code_pack_digest
    if getattr(args, "code_pack_digest", None):
        from yeto.codepack import CODE_PACK_MODE_GIT_PATCH

        miles_args.yeto_rl_code_pack_mode = (getattr(args, "code_pack_mode", None)
                                             or CODE_PACK_MODE_GIT_PATCH)
    _run_ports(
        args,
        miles_args,
        ports_launch,
        ports_algorithm,
        specs=specs,
        canonical_targets=canonical_targets,
        yeto_policy_sync=yeto_policy_sync,
        native_lora_export=native_lora_export,
    )
    # Last tape record: a rebuilt (no-sync) tape without it is incomplete.
    _append_ports_event(args, miles_args, {"event": "rl_learner_finalized"})
    print(f"[rl] learner {args.learner_id} finalized")


def _configure_grad_audit(args, miles_args) -> bool:
    """``YETO_RL_AUDIT_GRADS=1`` (teacher forcing, design D12): export pre-clip LoRA grads.

    The grads are written next to the round audit (``args.audit_dir``); the
    ports state plugin arms the capture from its ``train_one_step`` recorder.
    """

    from yeto.rl import grad_audit

    if not grad_audit.enabled():
        return False
    directory = getattr(args, "audit_dir", None)
    if not directory:
        raise ValueError(f"{grad_audit.GRAD_AUDIT_ENV}=1 requires an audit_dir")
    setattr(miles_args, grad_audit.GRAD_AUDIT_DIR_ATTR, str(Path(directory).expanduser()))
    return True


def _ports_spec(args, extra_argv: Sequence[str] = ()):
    from yeto.rl.engine.algorithm import resolve_ports_algorithm
    from yeto.rl.adapters.miles.algorithm_flags import absorb_extra_argv

    spec, _, _ = absorb_extra_argv(resolve_ports_algorithm(args),
                                   tuple(extra_argv))
    return spec


def _run_critic_warmup_stage(args, run_config, extra_argv: Sequence[str] = ()):
    """rl-algo-critic-family 5.3 (design D5): stage W before the ports main stage.

    Only for a critic copied from the actor with ``warmup_steps > 0`` and no
    ``--rl-critic-load``; every other run config is returned unchanged. The
    stage-W argv is the main-stage argv (built with a placeholder product)
    minus the driver hooks; the initial actor is the main stage's ``--ref-load``
    (Miles' ``critic_load`` default in bridge mode, arguments.py:3485-3487/3607).
    """

    import dataclasses

    from yeto.rl.critic_warmup import needs_warmup_stage, run_ports_warmup
    from yeto.rl.engine.run_config import CriticRunConfig

    spec = _ports_spec(args, extra_argv)
    if not needs_warmup_stage(spec, getattr(args, "rl_critic_load", None)):
        return run_config
    def with_critic(critic):
        return dataclasses.replace(
            run_config, algorithm=dataclasses.replace(run_config.algorithm, critic=critic)
        )

    placeholder = with_critic(
        CriticRunConfig(critic_load="<stage-W product>", init_sha256="0" * 64)
    )
    main_argv = build_ports_launch(args, placeholder, extra_argv).argv
    product = run_ports_warmup(
        spec,
        main_argv=main_argv,
        actor_checkpoint=str(run_config.ref_load),
        cache_root=args.rl_critic_warmup_dir,
        miles_root=str(Path(args.miles_root).expanduser().resolve()),
    )
    args.rl_critic_load = product.critic_checkpoint
    args.rl_critic_init_sha256 = product.critic_sha256
    return with_critic(product.critic_run_config())


def run_critic_baseline(args, learner_argv: Sequence[str],
                        run=subprocess.run) -> None:
    """``--rl-critic-baseline-rounds N``: the no-warm-up baseline learner run
    (critic_warmup.baseline_learner_argv) before this run's stage W."""

    from yeto.rl.critic_warmup import baseline_learner_argv

    rounds = int(getattr(args, "rl_critic_baseline_rounds", 0) or 0)
    if not rounds:
        return
    spec = _ports_spec(args)
    spec_path = Path("~/yeto-rl/critic-baseline/algorithm_spec.json").expanduser()
    argv, base = baseline_learner_argv(learner_argv, spec, rounds=rounds,
                                       spec_path=str(spec_path))
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(base.canonical_json())
    command = [sys.executable, "-m", "yeto.rl.adapters.miles.island_entry", *argv, "--rl-critic-baseline-run"]
    print("[rl] critic baseline run (no warm-up, %d rounds, spec %s): %s"
          % (rounds, base.sha256(), " ".join(command)), flush=True)
    from yeto.rl.event_echo import ECHO_ENV

    env = {k: v for k, v in os.environ.items() if k != ECHO_ENV}
    run(command, check=True, env=env)


def _run_test_debug_rollout_replay(args, miles_args) -> bool:
    """TEST ONLY (YETO_TEST_MILES_LOAD_DEBUG_ROLLOUT_DATA): train on a recorded
    rollout dump with upstream Miles' own train loop (train.py). The yeto driver
    publishes to rollout engines, which a dump replay does not start; this checks
    the Miles training half (packing, critic, advantages, train dump) only."""

    from yeto.rl.adapters.miles.config import TEST_DEBUG_ROLLOUT_ENV

    if not (os.environ.get(TEST_DEBUG_ROLLOUT_ENV) and getattr(miles_args, "debug_train_only", False)):
        return False
    import asyncio
    import importlib.util

    path = os.path.join(os.path.expanduser(args.miles_root), "train.py")
    spec = importlib.util.spec_from_file_location("miles_upstream_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from miles.utils.async_utils import with_disposer
    from yeto.rl.adapters.miles.entry import connect_island_ray

    # same job runtime_env as the ports island (PYTHONPATH -> pinned Miles/Megatron in the actors)
    connect_island_ray(miles_args=miles_args)
    print(f"[rl] TEST ONLY: upstream Miles train loop on {miles_args.load_debug_rollout_data}", flush=True)
    asyncio.run(with_disposer(module.train, miles_args))
    print("[rl] TEST ONLY: upstream Miles train loop finished", flush=True)
    return True


def _run_ports(
    args,
    miles_args,
    launch,
    algorithm,
    *,
    specs,
    canonical_targets,
    yeto_policy_sync: bool,
    native_lora_export: str | None = None,
) -> None:
    """Ports path: yeto's IslandDriver over upstream Miles (design D2).

    ``native_lora_export`` (Flash-Next, S11 try24): the LoRA layout is not
    predictable on the CPU side, so no hash is pinned here; the policy state
    learns it from the first export (before any receipt) and pins it after.
    """

    from yeto.rl.core import canonical_layout_hash
    from yeto.rl.adapters.miles.entry import run_ports_island

    if getattr(miles_args, "yeto_rl_parameter_mode", None) == "full":
        # run_miles already bound the full-parameter identity (model config
        # hash); there is no LoRA rank or adapter layout to derive here.
        lora_config_hash = miles_args.yeto_rl_lora_config_hash
        layout_hash = miles_args.yeto_rl_layout_hash
        print(f"[rl] full-parameter identity: layout={layout_hash}", flush=True)
    elif native_lora_export is not None:
        if specs:
            raise ValueError("native LoRA export must not carry predicted specs")
        if yeto_policy_sync:
            raise ValueError("native LoRA export runs only without Yeto policy sync")
        lora_config_hash = _lora_config_hash(args, canonical_targets)
        layout_hash = None
        miles_args.yeto_rl_native_lora_export = native_lora_export
        print(
            f"[rl] LoRA layout: learned from the first {native_lora_export} "
            f"native export (no predicted specs); targets={list(canonical_targets)}",
            flush=True,
        )
    else:
        lora_config_hash = _lora_config_hash(args, canonical_targets)
        layout_hash = canonical_layout_hash(specs)
        print(
            f"[rl] expected LoRA layout: {len(specs)} tensors, hash={layout_hash}: "
            + ", ".join(f"{s.name}{list(s.shape)}" for s in specs[:400]),
            flush=True,
        )
    # The event tape and island identity are needed even without outer sync.
    miles_args.yeto_rl_event_tape = args.event_tape
    miles_args.yeto_rl_learner_id = args.learner_id
    # fleet-dashboard 1.4: the ports EventTape tees into W&B (wandb_rl.tee)
    # off the same namespace fields as run_miles.
    miles_args.wandb = getattr(args, "wandb", False)
    miles_args.wandb_project = getattr(args, "wandb_project", "yeto")
    miles_args.wandb_entity = getattr(args, "wandb_entity", None)
    miles_args.wandb_mode = getattr(args, "wandb_mode", "online")
    apply_ports_infra_switches(args, miles_args)
    if _run_test_debug_rollout_replay(args, miles_args):
        return
    from yeto.rl.engine.sample_roles import SAMPLER, sample_role

    if sample_role(args) == SAMPLER:  # L4.6: generate into the sample buffer, never train
        from yeto.rl.adapters.miles.sampler_island import run_sampler_island

        run_sampler_island(miles_args)
        return
    run_ports_island(
        miles_args,
        launch,
        algorithm,
        learner_id=args.learner_id,
        base_model_revision=args.model_revision,
        lora_config_hash=lora_config_hash,
        layout_hash=layout_hash,
        yeto_policy_sync=yeto_policy_sync,
    )


def main(argv=None) -> None:
    from yeto.modal_runner import allow_any_ptracer

    allow_any_ptracer()  # S27 RAY-SEGV R5: only with YETO_STACK_SAMPLE_S (py-spy sampling on gVisor)
    learner_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    from yeto.provenance import (
        is_immutable_commit,
        is_local_reference,
        python_spec_sha256,
        verify_source_tree_sha256,
    )

    # rl-identity-tiering D6: with a run code pack the pack digest check replaces
    # the source tree check (the launcher attests --source-sha256 on the pack tree).
    if getattr(args, "code_pack_digest", None):
        release = verify_island_code_pack(args.code_pack_digest,
                                          mode=getattr(args, "code_pack_mode", None))
        if release is not None:  # the launcher's --source-sha256 is its local tree, not the image's
            args.source_sha256 = release["source_tree_sha256"]
    else:
        verify_source_tree_sha256(args.source_sha256)
    if not is_immutable_commit(args.model_revision):
        raise ValueError("RL model revision must be an immutable commit")
    if not is_local_reference(args.data) and not is_immutable_commit(
        args.data_revision
    ):
        raise ValueError("RL remote dataset revision must be an immutable commit")
    reward_sha256 = python_spec_sha256(args.reward_function)
    if reward_sha256 != args.reward_sha256.lower():
        raise ValueError(
            f"reward source SHA256 mismatch: expected {args.reward_sha256.lower()}, "
            f"got {reward_sha256}"
        )
    miles_root = str(Path(args.miles_root).expanduser().resolve())
    if miles_root not in sys.path:
        sys.path.insert(0, miles_root)
    pins = miles_revision_pins(args)
    if pins is None:
        verify_miles_revision(miles_root)  # the pinned MILES_NEXT_PINS
    else:
        verify_miles_revision(miles_root, expected=pins)
    probe_syncers(args)
    probe_wan_link(args)

    from miles.utils.misc import load_function

    load_function(_miles_callable(args.reward_function))
    if args.custom_generate_function_path:
        load_function(args.custom_generate_function_path)
    if args.custom_agent_function_path:
        load_function(args.custom_agent_function_path)
    from huggingface_hub import snapshot_download

    from yeto.models import resolve
    model = resolve(args.model)
    if is_local_reference(model):
        model_path = str(Path(model).expanduser().resolve())
    else:
        model_path = snapshot_download(repo_id=model, revision=args.model_revision)
    rollout_model = resolve(args.rollout_model) if args.rollout_model else model
    if rollout_model == model:
        rollout_model_path = model_path
    elif is_local_reference(rollout_model):
        rollout_model_path = str(Path(rollout_model).expanduser().resolve())
        if not Path(rollout_model_path).is_dir():
            raise ValueError("--rollout-model local directory does not exist")
    else:
        rollout_model_path = snapshot_download(
            repo_id=rollout_model,
            revision=args.rollout_model_revision,
        )
    eval_source = _verify_eval_dataset_identity(args)
    island_tasks = _verify_island_task_data(args)
    slot = _island_data_slot(args)
    columns = dict(
        prompt_column=getattr(args, "rl_prompt_column", None),
        label_column=getattr(args, "rl_label_column", None),
    )
    prompt_path = prepare_prompt_data(
        args.data,
        args.data_revision,
        "~/yeto-rl/prompts.jsonl",
        **columns,
        slot=slot,
        number_rows=_island_dispense(args),
    )
    eval_prompt_path = None
    if eval_source is not None:
        eval_prompt_path = (
            prompt_path
            if getattr(args, "eval_only", False)
            else prepare_prompt_data(
                str(eval_source),
                None,
                "~/yeto-rl/eval-prompts.jsonl",
                **columns,
            )
        )
    if getattr(args, "rl_critic_baseline_rounds", 0):
        run_critic_baseline(args, learner_argv)
    run_miles(
        args,
        model_path=model_path,
        rollout_model_path=rollout_model_path,
        prompt_path=prompt_path,
        eval_prompt_path=eval_prompt_path,
        island_tasks=island_tasks,
        # --rl-single-island-no-sync: LocalOnlySync, no syncer connection.
        yeto_policy_sync=not getattr(args, "rl_single_island_no_sync", False),
    )


def _rejoin_exhausted(error: BaseException | None) -> bool:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if type(error).__name__ == "ElasticRejoinError":
            return True
        error = error.__cause__ or error.__context__
    return False


if __name__ == "__main__":
    try:
        from yeto.island_credential_guard import check_island_credentials

        check_island_credentials()  # secret-handling-hardening D4
        main()
    except BaseException as error:  # noqa: BLE001
        if _rejoin_exhausted(error):
            # 0.22: the island gave up re-joining the elastic pool (not 4/6).
            from yeto.rl.pool_client import ELASTIC_REJOIN_FAILED_EXIT

            print(f"[yeto-island] elastic re-JOIN exhausted: {error}", file=sys.stderr)
            raise SystemExit(ELASTIC_REJOIN_FAILED_EXIT) from error
        raise
