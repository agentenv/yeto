"""RL launch-argument preparation (``_prepare_rl_args``) and the SAO
streaming / dense full-sweep contract adoption.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from ..gpu_spec import parse_gpu_spec
from ..rl.engine.run_flags import (
    FULL_PARAMETER_ASYNC_DEFAULT_PIPELINE,
    FULL_PARAMETER_ASYNC_MAX_PIPELINE,
    FULL_PARAMETER_MAX_LOCAL_ROUNDS_AHEAD,
)
from ..rl.engine.selection import (
    ASYNC_GRACE_CAP_MS,
    ASYNC_NESTEROV_MIN_H,
    ASYNC_TERMINAL_GRACE_MS,
    FULL_PARAMETER_SYNC_PRESET,
    SAO_STREAMING_SYNC_PRESET,
    is_sync_mode_default,
    plain,
)
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
    ROLE_SYNCERS, island_pool, rl_dense_sweep, rl_full_parameter_async, rl_full_parameter_final_cut,
    rl_needs_critic, rl_syncer_profile, rl_trainable_roles, syncer_shard_count,
)
from .storage import _rl_checkpoint_mount
from .peer_recovery import check_peer_recovery
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
    if island_pool(args):
        problems.append("a fixed island roster (no island pool options)")
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


def _synced_learners(args, specs) -> int:
    """Islands that join the syncer, the roster every quorum and membership
    check counts: a synced group's sampler islands do not (S26 e9b-sync-b:
    quorum 3 over --learners 2 made the syncer exit)."""
    from ..rl.engine.sample_roles import syncer_learner_count

    return syncer_learner_count(args, len(specs))


_FULL_PARAMETER_SYNC_DEFAULTS = (
    ("rl_sync_fragment_pipeline", 1, "--rl-sync-fragment-pipeline"),
    ("rl_sync_async_checkpoint", False, "--rl-sync-async-checkpoint"),
    ("rl_sync_push_dtype", "f32", "--rl-sync-push-dtype"),
    ("rl_sync_checkpoint_every", 1, "--rl-sync-checkpoint-every"),
    ("rl_sync_push_int8_block", 128, "--rl-sync-push-int8-block"),
    ("rl_sync_fragment_max_bytes", 2 << 30, "--rl-sync-fragment-max-bytes"),
    ("rl_sync_push_int4_block", 128, "--rl-sync-push-int4-block"),
    ("rl_sync_push_int4_stochastic", False, "--rl-sync-push-int4-stochastic"),
    ("rl_sync_push_int4_error_feedback", "off", "--rl-sync-push-int4-error-feedback"),
    ("rl_sync_push_no_tensor_blocks", False, "--rl-sync-push-no-tensor-blocks"),
    ("rl_sync_overlap", False, "--rl-sync-overlap"),
    ("rl_sync_full_parameter_rounds", "sweep", "--rl-sync-full-parameter-rounds"),
    ("rl_sync_expert_token_weights", False, "--rl-sync-expert-token-weights"),
    ("rl_sync_final_cut", "auto", "--rl-sync-final-cut"),
    ("rl_sync_init_mode", "auto", "--rl-sync-init-mode"),
    ("rl_sync_wait_timeout_s", 900.0, "--rl-sync-wait-timeout-s"),
    ("rl_sync_max_local_rounds_ahead", None, "--rl-sync-max-local-rounds-ahead"),
)


def _refuse_full_parameter_sync_options(args) -> None:
    """The S26 E3 sync options exist only for the full-parameter preset
    (the bf16 downlink also for strict-avg, see ``_check_bcast_delta``)."""
    for dest, default, option in _FULL_PARAMETER_SYNC_DEFAULTS:
        if getattr(args, dest, default) != default:
            raise ValueError(f"{option} only applies to --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET}")


def _check_bcast_delta(args) -> None:
    """S26 E3 bf16 downlink: a legacy-scheduled strict-avg or full-parameter
    syncer only (the elastic server and SAO have no downlink anchors)."""
    full_every = int(getattr(args, "rl_sync_bcast_delta_full_every", 0) or 0)
    if full_every == 0:
        return
    if full_every < 2:
        raise ValueError("--rl-sync-bcast-delta-full-every 0 (off) or >= 2")
    if args.rl_sync_preset not in ("strict-avg", FULL_PARAMETER_SYNC_PRESET):
        raise ValueError("--rl-sync-bcast-delta-full-every only applies to --rl-sync-preset "
                         f"strict-avg or {FULL_PARAMETER_SYNC_PRESET}")
    if island_pool(args) or args.experimental_rl_sync:
        raise ValueError("--rl-sync-bcast-delta-full-every needs a fixed island roster (no island "
                         "pool options) and no --experimental-rl-sync")


# S27 P2: asynchronous rounds keep FULL_PARAMETER_ASYNC_DEFAULT_PIPELINE (2)
# fragment rounds in flight by default and at most
# FULL_PARAMETER_ASYNC_MAX_PIPELINE (60); both live in rl.engine.run_flags.


def _prepare_dense_full_sweep(args, *, num_islands: int) -> None:
    """full-parameter preset (P0-D8): one actor role synced as whole-model fragments.

    ``--rl-sync-full-parameter-rounds sweep`` (default, the control) is the
    syncer's strict policy-sweep profile: every island per round, no grace,
    serial rounds, plain averaging. ``async`` (S27 P2) is asynchronous:
    the syncer runs ordinary fragment rounds (H local RL rounds per fragment,
    at most two rounds in flight, the decoupled preset's outer Nesterov), and
    the outer-merge options (quorum K, grace, weights, base lag, RDA) apply.
    The fragment count P comes from the Miles layout probe.

    Placement: islands with engines on the island need dedicated engine GPUs
    (``--rl-placement fixed-partition``). Synced trainers that read their samples
    from the store (``--rl-sample-role trainer``) start no engine, so they take
    no placement (their sampler islands are their own engine partitions)."""
    from ..rl.engine.sample_roles import is_synced_trainer

    async_rounds = rl_full_parameter_async(args)
    problems = []
    fixed_partition = getattr(args, "rl_placement", "colocated") == "fixed-partition"
    if is_synced_trainer(args):
        if fixed_partition:
            problems.append("no --rl-placement fixed-partition with --rl-sample-role trainer (the "
                            "trainers start no engine; the sampler islands generate)")
    elif not fixed_partition:
        problems.append("--rl-placement fixed-partition (dedicated rollout GPUs)")
    if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, "rl_parameter_layout_sha256", None) or ""):
        problems.append("--rl-parameter-layout-sha256 <64 lowercase hex>")
    if args.experimental_rl_sync:
        problems.append("no --experimental-rl-sync")
    if int(getattr(args, "rl_optimizer_steps", 1) or 1) != 1:
        problems.append("one optimizer step per local RL round")
    if args.local_rl_rounds_per_sync < 1 or (not async_rounds and args.local_rl_rounds_per_sync != 1):
        problems.append("--local-rl-rounds-per-sync 1 (strict sweep H=1)" if not async_rounds
                        else "--local-rl-rounds-per-sync >= 1")
    if args.fragments < 1:
        problems.append("--fragments >= 1")
    if island_pool(args) and not async_rounds:
        problems.append("a fixed island roster (membership options need "
                        "--rl-sync-full-parameter-rounds async)")
    if rl_needs_critic(args):
        problems.append("a critic-free algorithm (the full-parameter preset trains the actor only)")
    fragment_pipeline = int(getattr(args, "rl_sync_fragment_pipeline", 1))
    max_pipeline = max(args.fragments, 1)
    if async_rounds:
        max_pipeline = min(max_pipeline, FULL_PARAMETER_ASYNC_MAX_PIPELINE)
    if not 1 <= fragment_pipeline <= max_pipeline:
        problems.append(f"--rl-sync-fragment-pipeline between 1 and min({FULL_PARAMETER_ASYNC_MAX_PIPELINE}, "
                        "--fragments)" if async_rounds
                        else "--rl-sync-fragment-pipeline between 1 and --fragments")
    wait_timeout = float(getattr(args, "rl_sync_wait_timeout_s", 900.0))
    if not 0 < wait_timeout <= 7 * 86400 or wait_timeout != int(wait_timeout):
        problems.append("--rl-sync-wait-timeout-s in whole seconds within 1..604800")
    lead = getattr(args, "rl_sync_max_local_rounds_ahead", None)
    if lead is not None:
        if not async_rounds:
            problems.append("--rl-sync-max-local-rounds-ahead only with --rl-sync-full-parameter-rounds async")
        elif not 1 <= lead <= FULL_PARAMETER_MAX_LOCAL_ROUNDS_AHEAD:
            problems.append(f"--rl-sync-max-local-rounds-ahead within 1..{FULL_PARAMETER_MAX_LOCAL_ROUNDS_AHEAD}")
    checkpoint_every = int(getattr(args, "rl_sync_checkpoint_every", 1))
    if async_rounds:
        # The async checkpoint writer and the sync/train overlap are
        # strict-sweep profiles of the syncer. The checkpoint interval counts
        # fragment rounds: 1 persists every round before its broadcast; N > 1
        # writes after every N-th round and at the end (a crash resumes at the
        # newest checkpoint; a model-sized f32 state per round is too slow).
        if checkpoint_every < 1:
            problems.append("--rl-sync-checkpoint-every >= 1")
        if getattr(args, "rl_sync_async_checkpoint", False):
            problems.append("no --rl-sync-async-checkpoint (sweep only)")
        if getattr(args, "rl_sync_overlap", False):
            problems.append("no --rl-sync-overlap (sweep only)")
    elif checkpoint_every != 1 and (checkpoint_every < 1 or checkpoint_every % max(args.fragments, 1)):
        problems.append("--rl-sync-checkpoint-every 1 or a multiple of --fragments")
    fragment_max_bytes = getattr(args, "rl_sync_fragment_max_bytes", 2 << 30)
    if type(fragment_max_bytes) is not int or not 4 <= fragment_max_bytes <= 2 << 30 or fragment_max_bytes % 4:
        problems.append("--rl-sync-fragment-max-bytes a multiple of 4 between 4 and 2147483648")
    int8_block = int(getattr(args, "rl_sync_push_int8_block", 128))
    if not 32 <= int8_block <= 1 << 20:
        problems.append("--rl-sync-push-int8-block between 32 and 1048576")
    elif int8_block != 128 and getattr(args, "rl_sync_push_dtype", "f32") != "int8":
        problems.append("--rl-sync-push-int8-block only with --rl-sync-push-dtype int8")
    int4_block = int(getattr(args, "rl_sync_push_int4_block", 128))
    int4 = getattr(args, "rl_sync_push_dtype", "f32") == "int4"
    if not 64 <= int4_block <= 256 or int4_block % 2:
        problems.append("--rl-sync-push-int4-block even, between 64 and 256")
    elif int4_block != 128 and not int4:
        problems.append("--rl-sync-push-int4-block only with --rl-sync-push-dtype int4")
    if getattr(args, "rl_sync_push_no_tensor_blocks", False) and getattr(
        args, "rl_sync_push_dtype", "f32"
    ) not in ("int8", "int4", "int4-log"):
        problems.append("--rl-sync-push-no-tensor-blocks only with --rl-sync-push-dtype int8, int4 or int4-log")
    if getattr(args, "rl_sync_push_int4_stochastic", False) and not int4:
        problems.append("--rl-sync-push-int4-stochastic only with --rl-sync-push-dtype int4")
    if getattr(args, "rl_sync_push_int4_error_feedback", "off") != "off":
        if not int4:
            problems.append("--rl-sync-push-int4-error-feedback only with --rl-sync-push-dtype int4")
        elif getattr(args, "rl_sync_push_int4_stochastic", False):
            problems.append("--rl-sync-push-int4-error-feedback without --rl-sync-push-int4-stochastic")
    if getattr(args, "rl_sync_expert_token_weights", False) and not async_rounds:
        # The sweep is the strict equal-weight control.
        problems.append("--rl-sync-expert-token-weights only with --rl-sync-full-parameter-rounds async")
    if getattr(args, "rl_sync_overlap", False) and not async_rounds:
        from ..rl.engine.sync_overlap import overlap_option_problems

        problems += overlap_option_problems(
            max_policy_age=getattr(args, "rl_max_policy_age", 0) or 0,
            fragment_pipeline=fragment_pipeline, fragments=args.fragments)
    if problems:
        hint = (" (a multi-island run defaults to asynchronous rounds; --rl-sync-mode strict "
                "or --rl-sync-full-parameter-rounds sweep selects the strict sweep)"
                if async_rounds and getattr(args, "rl_sync_mode_effective", "strict") == "async"
                else "")
        raise ValueError(f"--rl-sync-preset {FULL_PARAMETER_SYNC_PRESET} requires "
                         + "; ".join(problems) + hint)
    args.quorum = num_islands
    args.grace_ms = 0
    args.pipeline = fragment_pipeline
    args.delta_correction = "none"
    args.merge_alpha = 0.0
    args.wire_dtype = "f32"
    args.rl_total_fragment_steps = args.total_steps * args.fragments
    if async_rounds:
        # The decoupled preset's outer step; the outer-merge options override.
        args.sync_interval_steps = float(args.local_rl_rounds_per_sync)
        args.outer_lr = 0.7
        args.outer_momentum = 0.9
        return
    # Fragment rounds of one sweep are independent (H=1, plain average), so the
    # syncer may gather several at once; it still commits them in step order.
    args.sync_interval_steps = 0.0
    args.outer_lr = 1.0
    args.outer_momentum = 0.0


# The options --rl-sync-mode fills when the command leaves them unset: dest
# -> the async-mode value. None-default options (quorum, grace, outer step)
# are unset when None; the others carry a marked default (selection.py).
_ASYNC_MODE_VALUES = {
    "rl_sync_quorum": 1,
    "rl_sync_grace_ms": ASYNC_GRACE_CAP_MS,
    "rl_sync_max_base_lag": 1,
    "rl_sync_learner_weight": "tokens2-over-steps",
}
_ASYNC_MODE_FULL_PARAMETER_VALUES = {
    "rl_sync_full_parameter_rounds": "async",
    "rl_sync_push_dtype": "int4",
    "rl_sync_terminal_grace_ms": ASYNC_TERMINAL_GRACE_MS,
}
# Marked defaults; the strict mode keeps their (strict) value.
_SYNC_MODE_MARKED = (
    "rl_sync_full_parameter_rounds", "rl_sync_fragment_pipeline", "rl_sync_push_dtype",
    "rl_sync_learner_weight", "rl_sync_max_base_lag", "rl_sync_matrix_merge",
)


def rl_sync_mode_applies(args) -> bool:
    """The async mode fills unset options only for a run with two or more
    islands on strict-avg, decoupled or full-parameter, without a critic or
    --experimental-rl-sync; every other run keeps the strict values."""
    return not rl_sync_mode_off_reasons(args)


def rl_sync_mode_off_reasons(args) -> list[str]:
    """Why the async mode does not apply to this run (empty: it applies)."""
    reasons = []
    preset = getattr(args, "rl_sync_preset", "strict-avg")
    if preset not in ("strict-avg", "decoupled", FULL_PARAMETER_SYNC_PRESET):
        reasons.append(f"preset {preset}")
    if (islands := rl_island_count(args)) < 2:
        reasons.append(f"{islands} island")
    if getattr(args, "experimental_rl_sync", False):
        reasons.append("--experimental-rl-sync")
    if rl_needs_critic(args):
        reasons.append("a critic")
    return reasons


def _apply_rl_sync_mode(args) -> None:
    """Resolve ``--rl-sync-mode`` into concrete option values.

    Only options the command left unset change; an explicit option always
    wins. Afterwards ``args.rl_sync_mode_effective`` is the mode that took
    effect ("strict" when the async mode does not apply) and no option keeps
    its mark. Args never resolved (built in code, or parsed without the RL
    checks) read as strict: every marked default equals its strict value."""
    mode = getattr(args, "rl_sync_mode", "strict")
    explicit_sweep = (getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET
                      and getattr(args, "rl_sync_full_parameter_rounds", "sweep") == "sweep"
                      and not is_sync_mode_default(args.rl_sync_full_parameter_rounds))
    # An explicit full-parameter sweep is the strict control profile.
    if mode == "async" and rl_sync_mode_applies(args) and not explicit_sweep:
        values = dict(_ASYNC_MODE_VALUES)
        nesterov = int(getattr(args, "local_rl_rounds_per_sync", 1)) >= ASYNC_NESTEROV_MIN_H
        values["rl_sync_outer_lr"] = 0.7 if nesterov else 1.0
        values["rl_sync_outer_momentum"] = 0.9 if nesterov else 0.0
        if args.rl_sync_preset == FULL_PARAMETER_SYNC_PRESET:
            values.update(_ASYNC_MODE_FULL_PARAMETER_VALUES)
            # Quantization blocks restart at tensor boundaries by default
            # (--rl-sync-push-no-tensor-blocks turns that off).
        if args.rl_sync_preset in ("decoupled", FULL_PARAMETER_SYNC_PRESET):
            values["rl_sync_matrix_merge"] = "rda"
        # Peer recovery: _default_peer_recovery, right after this function.
        if getattr(args, "rl_sync_quorum_theta", None) is not None:
            values.pop("rl_sync_quorum")  # the capacity quorum replaces the count
        for dest, value in values.items():
            current = getattr(args, dest, None)
            if current is None or is_sync_mode_default(current):
                setattr(args, dest, value)
        if (args.rl_sync_preset == FULL_PARAMETER_SYNC_PRESET
                and args.rl_sync_full_parameter_rounds == "async"
                and is_sync_mode_default(getattr(args, "rl_sync_fragment_pipeline", None))):
            # Two fragment rounds in flight by default; the fragment count is
            # checked (and may be 1) in _prepare_dense_full_sweep.
            args.rl_sync_fragment_pipeline = min(FULL_PARAMETER_ASYNC_DEFAULT_PIPELINE,
                                                 max(int(getattr(args, "fragments", 1) or 1), 1))
    else:
        mode = "strict"
    for dest in _SYNC_MODE_MARKED:
        if hasattr(args, dest):
            setattr(args, dest, plain(getattr(args, dest)))
    args.rl_sync_mode_effective = mode
    if getattr(args, "rl_sync_mode", "strict") == "async" and mode == "strict" and not explicit_sweep:
        _warn_async_mode_off(args)


def _warn_async_mode_off(args) -> None:
    """S27 G-CD: ``--rl-sync-mode async`` (the default) that does not apply to a
    synced run: name the mode in effect and the final cut, which a one-island
    full-parameter run otherwise only sees as an f32 download of the whole model
    (120B: about 494 GB per island, GBC-120B-STAGEB §3)."""
    import sys

    if getattr(args, "rl_single_island_no_sync", False):
        return  # no syncer: neither mode has anything to fill
    final = ""
    if rl_dense_sweep(args):
        cut = rl_full_parameter_final_cut(args)
        final = (f"; final cut: {cut}" + (
            " (every island downloads the f32 model, about 4 bytes per parameter; "
            "--rl-sync-final-cut ack keeps it in the syncer)" if cut == "download" else ""))
    print(f"[launcher] WARNING: --rl-sync-mode async does not apply "
          f"({', '.join(rl_sync_mode_off_reasons(args))}): the run uses the strict sync values "
          f"(rl_sync_mode_effective strict){final}", file=sys.stderr)


# S27 P1: (dest, option, preset-keeping default) of the outer-merge options.
_RL_OUTER_MERGE_OPTIONS = (
    ("rl_sync_quorum", "--rl-sync-quorum", None),
    ("rl_sync_grace_ms", "--rl-sync-grace-ms", None),
    ("rl_sync_grace_min_ms", "--rl-sync-grace-min-ms", None),
    ("rl_sync_grace_arrival_quantile", "--rl-sync-grace-arrival-quantile", None),
    ("rl_sync_terminal_grace_ms", "--rl-sync-terminal-grace-ms", None),
    ("rl_sync_learner_weight", "--rl-sync-learner-weight", "equal"),
    ("rl_sync_max_base_lag", "--rl-sync-max-base-lag", 0),
    ("rl_sync_outer_lr", "--rl-sync-outer-lr", None),
    ("rl_sync_outer_momentum", "--rl-sync-outer-momentum", None),
    ("rl_sync_matrix_merge", "--rl-sync-matrix-merge", "avg"),
)


ISLAND_STATE_SOURCES = ("peer", "catch-up", "cut")


def _default_peer_recovery(args) -> None:
    """Resolve the state source of a replacement island (``--rl-island-state-source``)
    and ``--rl-peer-recovery`` together; the two always agree afterwards.

    Who fills a slot (a relaunch, an island pool member) is not decided here; this
    only picks where that island takes its trainer state. ``--rl-peer-recovery``
    left unset: on in the async mode where it applies (full-parameter async rounds,
    one syncer process, no broadcast deltas, a snapshot store; a fixed roster or an
    island pool), off otherwise; the strict mode and ``--no-rl-peer-recovery`` keep
    it off. Peer recovery on is source ``peer``, off is ``catch-up``."""
    source = getattr(args, "rl_island_state_source", None)
    explicit = getattr(args, "rl_peer_recovery", None)
    if source == "cut":
        raise ValueError("--rl-island-state-source cut: a synced island does not resume from its "
                         "own round cuts yet (the syncer is authoritative); use peer or catch-up")
    if source is not None and explicit is not None and explicit != (source == "peer"):
        flag = "--rl-peer-recovery" if explicit else "--no-rl-peer-recovery"
        raise ValueError(f"--rl-island-state-source {source} contradicts {flag}")
    if source is not None:
        args.rl_peer_recovery = source == "peer"
    elif explicit is None:
        args.rl_peer_recovery = _peer_recovery_applies(args)
    args.rl_island_state_source = "peer" if args.rl_peer_recovery else "catch-up"


def _peer_recovery_applies(args) -> bool:
    if (getattr(args, "rl_sync_mode_effective", "strict") != "async"
            or not rl_full_parameter_async(args)
            or syncer_shard_count(args) > 1
            or getattr(args, "rl_sync_bcast_delta_full_every", 0)):
        return False
    from .peer_recovery import snapshot_store_path

    return snapshot_store_path(args) is not None


def _apply_rl_outer_merge_options(args, *, num_islands: int) -> None:
    """S27 P1: override the outer-merge settings the preset just fixed.

    Every option left at its default keeps the preset's value, so the syncer
    command and semantic profile are unchanged. The full-parameter preset takes
    them only with asynchronous rounds (S27 P2; the sweep keeps its strict
    roster); SAO and --experimental-rl-sync take no overrides."""
    changed = [option for dest, option, default in _RL_OUTER_MERGE_OPTIONS
               if getattr(args, dest, default) != default]
    if not changed:
        return
    where = ", ".join(changed)
    full_parameter = args.rl_sync_preset == FULL_PARAMETER_SYNC_PRESET
    if full_parameter and not rl_full_parameter_async(args):
        raise ValueError(f"{where} need --rl-sync-full-parameter-rounds async "
                         "(the full-parameter sweep keeps its strict roster)")
    if args.rl_sync_preset not in ("strict-avg", "decoupled") and not full_parameter:
        raise ValueError(f"{where} only apply to --rl-sync-preset strict-avg, decoupled "
                         f"or {FULL_PARAMETER_SYNC_PRESET}")
    if args.experimental_rl_sync:
        raise ValueError(f"{where} do not apply with --experimental-rl-sync "
                         "(it takes the generic syncer flags)")
    quorum = args.rl_sync_quorum
    if quorum is not None:
        if not 1 <= quorum <= num_islands:
            raise ValueError(f"--rl-sync-quorum must be between 1 and the {num_islands} islands")
        args.quorum = quorum
    grace_ms = args.rl_sync_grace_ms
    if grace_ms is not None:
        if grace_ms < 0:
            raise ValueError("--rl-sync-grace-ms must be >= 0")
        args.grace_ms = grace_ms
    grace_min_ms = getattr(args, "rl_sync_grace_min_ms", None)
    if grace_min_ms is not None:
        if grace_min_ms < 0:
            raise ValueError("--rl-sync-grace-min-ms must be >= 0")
        args.grace_min_ms = grace_min_ms
    quantile = getattr(args, "rl_sync_grace_arrival_quantile", None)
    if quantile is not None:
        if not 0.0 < quantile <= 1.0:
            raise ValueError("--rl-sync-grace-arrival-quantile must be in (0, 1]")
        args.grace_arrival_quantile = quantile
    terminal_grace_ms = getattr(args, "rl_sync_terminal_grace_ms", None)
    if terminal_grace_ms is not None:
        if terminal_grace_ms < 0:
            raise ValueError("--rl-sync-terminal-grace-ms must be >= 0")
        args.terminal_grace_ms = terminal_grace_ms
    if args.rl_sync_max_base_lag < 0:
        raise ValueError("--rl-sync-max-base-lag must be >= 0")
    outer_lr = args.rl_sync_outer_lr
    if outer_lr is not None:
        if not outer_lr > 0.0:
            raise ValueError("--rl-sync-outer-lr must be positive")
        args.outer_lr = outer_lr
    momentum = args.rl_sync_outer_momentum
    if momentum is not None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("--rl-sync-outer-momentum must be in [0, 1)")
        args.outer_momentum = momentum
    if args.rl_sync_matrix_merge != "avg" and args.rl_sync_preset not in (
            "decoupled", FULL_PARAMETER_SYNC_PRESET):
        raise ValueError("--rl-sync-matrix-merge rda only applies to --rl-sync-preset decoupled "
                         f"or {FULL_PARAMETER_SYNC_PRESET} with asynchronous rounds")


def _check_island_pool(args, *, num_islands: int) -> None:
    """Membership options (--rl-sync-quorum-theta, ...; dynamic membership)
    apply to the strict-avg, decoupled and asynchronous full-parameter
    presets without a critic. The rounds must be able to merge without some
    island: a capacity quorum (--rl-sync-quorum-theta), a count quorum below
    the islands, a grace window or a carried base lag (the syncer refuses
    membership options with a strict fixed roster).
    --rl-sync-quorum-theta replaces the count quorum, so the syncer's
    --quorum is 1."""
    if not island_pool(args):
        if getattr(args, "rl_syncer_epoch", None) is not None and not getattr(args, "rl_peer_recovery", False):
            raise ValueError("--rl-syncer-epoch needs island pool options (e.g. --rl-sync-island-lease-s) "
                             "or --rl-peer-recovery")
        return
    preset = args.rl_sync_preset
    if args.experimental_rl_sync or not (
            preset in ("strict-avg", "decoupled") or rl_full_parameter_async(args)):
        raise ValueError("membership options (--rl-sync-quorum-theta, --rl-sync-island-lease-s, "
                         "...) apply to --rl-sync-preset strict-avg, decoupled, or "
                         f"{FULL_PARAMETER_SYNC_PRESET} with --rl-sync-full-parameter-rounds async")
    if rl_needs_critic(args):
        raise ValueError("membership options do not support a critic yet")
    if int(getattr(args, "syncer_shards", None) or 1) > 1:
        raise ValueError("membership options need one syncer (no --syncer-shards > 1)")
    max_learners = getattr(args, "rl_sync_max_learners", None)
    if max_learners is not None:
        if preset == FULL_PARAMETER_SYNC_PRESET:
            raise ValueError(f"--rl-sync-max-learners does not apply to {FULL_PARAMETER_SYNC_PRESET}: "
                             "its island session binds the launched roster")
        if int(max_learners) < num_islands:
            raise ValueError(f"--rl-sync-max-learners must be at least the {num_islands} islands")
    if getattr(args, "rl_sync_quorum_theta", None) is not None:
        if getattr(args, "rl_sync_quorum", None) is not None:
            raise ValueError("--rl-sync-quorum-theta replaces --rl-sync-quorum; give one of them")
        args.quorum = 1
    fixed = (args.rl_sync_max_base_lag == 0 and args.quorum == num_islands
             and int(args.grace_ms) == 0)
    if fixed:
        raise ValueError(
            "membership options need rounds that may merge without some island: set "
            "--rl-sync-quorum-theta, --rl-sync-quorum below the island count, --rl-sync-grace-ms "
            "or --rl-sync-max-base-lag >= 1")


def _prepare_node_inventory(args) -> None:
    """Validate ``--rl-node-inventory`` before any cloud or SSH work and keep the
    parsed nodes on ``args`` (rl-ssh-node-pool: the launcher reads the nodes from
    an inventory file; a missing device family or an unknown card is refused)."""
    path = getattr(args, "rl_node_inventory", None)
    if path is None:
        args.rl_node_inventory_nodes = None
        return
    from .node_inventory import load_inventory

    args.rl_node_inventory_nodes = load_inventory(path)


def _check_rollout_ahead(args, *, max_policy_age: int, backend_max: int) -> None:
    """--rl-rollout-ahead (``yeto.rl.engine.rollout_ahead``), refused before any cloud work."""
    from ..rl.engine.rollout_ahead import FLAG, launch_problems

    ahead = getattr(args, "rl_rollout_ahead", 0) or 0
    problems = launch_problems(
        ahead,
        backend_max=int(backend_max),
        max_policy_age=max_policy_age,
        placements=[getattr(i, "rl_placement", "colocated") for i in rl_island_placement_args(args)],
        sync_supported=(rl_full_parameter_async(args)
                        or bool(getattr(args, "rl_single_island_no_sync", False))),
        sample_roles=(getattr(args, "rl_sample_role", None) or "both") != "both"
        or bool(getattr(args, "rl_island_sample_role", None)),
        elastic=bool(getattr(args, "rl_elastic", False)),
        island_cuts=(any(getattr(args, name, None) is not None
                         for name in ("rl_cut_every", "rl_cut_auto_mtbf_s"))
                     or (bool(getattr(args, "rl_single_island_no_sync", False))
                         and bool(getattr(args, "rl_checkpoint_store", None)))),
        overlap_eval=bool(getattr(args, "rl_overlap_eval", False)),
    )
    if problems:
        raise ValueError(f"{FLAG} {ahead} requires " + "; ".join(problems))


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

    from yeto.rl.quant.tiers import validate_launch as _validate_quant_tier

    _validate_quant_tier(args)  # S26 E6: refused before any cloud spend
    _prepare_node_inventory(args)
    _apply_rl_sync_mode(args)
    _default_peer_recovery(args)

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
            island_scheduling="elastic" if island_pool(args) else "legacy",
        )
    if resolve_model_kind(args.model, args.model_kind) != "causal-lm":
        raise ValueError("RL v0 supports only causal language models")
    from ..rl.engine import sample_roles

    if args.tuning != "lora" and not sample_roles.buffer_full_parameter(args):
        raise ValueError("RL v0 requires --tuning lora (--tuning full: only with "
                         "--rl-single-island-no-sync)")
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
    if args.rl_sync_preset == "strict-avg" or (
        args.rl_sync_preset == FULL_PARAMETER_SYNC_PRESET and not rl_full_parameter_async(args)
    ):
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
    _check_rollout_ahead(args, max_policy_age=max_policy_age,
                         backend_max=getattr(policy_age_module, "MAX_ROLLOUT_AHEAD", 0))
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
    # S25 N10.4: a backend without the Miles agentic wiring refuses its flags.
    agentic_check = getattr(_rl_backend_module(args, "run_config_rules"), "check_agentic", None)
    if callable(agentic_check):
        agentic_check(args)
    _rl_miles_function(args.custom_generate_function_path)
    custom_agent = getattr(args, "custom_agent_function_path", None)
    _rl_miles_function(custom_agent, "--custom-agent-function-path")
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
        if args.apply_chat_template_kwargs is not None:
            raise ValueError(
                "agentic session rollouts preserve raw messages and do not accept "
                "--apply-chat-template-kwargs"
            )

    # The signed stock-Codex and SecRLEnv agents these flags configured were
    # removed in S26 E0.5 (tag s26-pre-harness-removal); refuse them.
    for flag, attr in (
        ("--codex-reasoning-effort", "codex_reasoning_effort"),
        ("--codex-backend-profile", "codex_backend_profile"),
        ("--codex-exclude-reasoning-from-loss", "codex_exclude_reasoning_from_loss"),
        ("--secrlenv-max-infrastructure-replacements", "secrlenv_max_infrastructure_replacements"),
    ):
        if getattr(args, attr, None) not in (None, False):
            raise ValueError(f"{flag} requires a signed Codex agent; the signed Codex and "
                             "SecRLEnv agents were removed (S26 E0.5)")
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
        elif dynamic_filter != bounded_filter:
            raise ValueError(
                "--dynamic-sampling-max-replacements is only supported with "
                "an approved bounded nonzero-variance filter"
            )
    # After the over-sampling count is final: the expected spec hash must be
    # the one the island computes.
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
    if int(getattr(args, "context_parallel", 1) or 1) <= 0:
        raise ValueError("RL context parallelism must be positive")
    model_parallel = args.tensor_parallel * args.pipeline_parallel * int(getattr(args, "context_parallel", 1) or 1)
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
    if args.rl_sync_preset != FULL_PARAMETER_SYNC_PRESET:
        _refuse_full_parameter_sync_options(args)
    _check_bcast_delta(args)
    learners = _synced_learners(args, specs)
    if args.rl_sync_preset == "decoupled":
        if args.experimental_rl_sync:
            raise ValueError("decoupled RL does not accept --experimental-rl-sync")
        if args.fragments < 2:
            raise ValueError("decoupled RL requires at least 2 fragments")
        if not 1 <= args.pipeline <= args.fragments:
            raise ValueError("decoupled RL pipeline must be between 1 and fragments")
        if args.fragment_pattern != "binpack":
            raise ValueError("decoupled RL requires --fragment-pattern binpack")
        args.quorum = learners
        args.grace_ms = 0
        args.sync_interval_steps = float(args.local_rl_rounds_per_sync)
        args.delta_correction = "none"
        args.outer_lr = 0.7
        args.outer_momentum = 0.9
        args.merge_alpha = 0.0
        args.wire_dtype = "f32"
        args.rl_total_fragment_steps = args.total_steps * args.fragments
    elif args.rl_sync_preset == FULL_PARAMETER_SYNC_PRESET:
        _prepare_dense_full_sweep(args, num_islands=learners)
    elif args.rl_sync_preset == SAO_STREAMING_SYNC_PRESET:
        _prepare_sao_streaming(args, num_islands=len(specs))
    elif not args.experimental_rl_sync:
        args.fragments = 1
        args.quorum = learners
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
    _apply_rl_outer_merge_options(args, num_islands=learners)
    syncer_shard_count(args)  # refuse --syncer-shards outside asynchronous rounds now
    _check_island_pool(args, num_islands=learners)
    check_peer_recovery(args)
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
