"""The syncer (outer optimizer): its command line, role syncers, island
scheduling settings, port probing and the syncer cluster task.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..rl.engine.run_flags import CLI, rl_option_pairs
from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESET, SAO_STREAMING_SYNC_PRESET
from . import REPO_ROOT
from .remote_setup import WAN_TUNING


SYNCER_PORT = 29400
# rl-algo-critic-family 4.2.1 (design D4 plan a): a critic algorithm runs a
# second syncer beside the actor's, on its own port, checkpoint and tape; its
# layout is the critic layout (critic_layout_hash), never the actor LoRA one.
CRITIC_SYNCER_PORT = 29401


def build_syncer_binary() -> Path:
    binary = REPO_ROOT / "syncer/target/release/yeto-syncer"
    print("[launcher] building syncer (cargo build --release)...")
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT / "syncer", check=True)
    return binary


# The syncer's JSONL merge log. Head-controller mode tails it into W&B
# (yeto.wandb_tape), so the path has to be the same on both sides — and RL
# runs put it under the output dir they collect, so it varies by mode.
SYNCER_EVENT_TAPE = "~/yeto-tape.jsonl"
RL_SYNCER_EVENT_TAPE = "~/yeto-output/yeto-tape.jsonl"
RL_CRITIC_SYNCER_EVENT_TAPE = "~/yeto-output/yeto-critic-tape.jsonl"
RL_CRITIC_SYNCER_CHECKPOINT = "~/yeto-output/yeto-critic-state.ckpt"
RL_SYNCER_CHECKPOINT = "~/yeto-output/yeto-state.ckpt"


@dataclass(frozen=True)
class RoleSyncer:
    """Where one trainable role's RL syncer listens and keeps its state (P0-D8)."""

    role: str
    port: int
    address: str  # the island-side env reference
    checkpoint: str
    event_tape: str


ROLE_SYNCERS = {
    "actor": RoleSyncer("actor", SYNCER_PORT, "$SYNCER_ADDR", RL_SYNCER_CHECKPOINT,
                        RL_SYNCER_EVENT_TAPE),
    "critic": RoleSyncer("critic", CRITIC_SYNCER_PORT, "$CRITIC_SYNCER_ADDR",
                         RL_CRITIC_SYNCER_CHECKPOINT, RL_CRITIC_SYNCER_EVENT_TAPE),
}


def rl_needs_critic(args) -> bool:
    """True iff the prepared RL algorithm spec trains a critic (execution.needs_critic)."""
    if getattr(args, "training_mode", "sft") != "rl":
        return False
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if not raw:
        return False
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return False
    return bool(((payload or {}).get("execution") or {}).get("needs_critic"))


def critic_syncer_address(syncer_addr: str) -> str:
    """The critic syncer runs on the actor syncer's host, on CRITIC_SYNCER_PORT."""
    if syncer_addr == "$SYNCER_ADDR":
        return "$CRITIC_SYNCER_ADDR"
    host, _, _port = syncer_addr.rpartition(":")
    if not host:
        raise ValueError(f"syncer address {syncer_addr!r} has no host:port")
    return f"{host}:{CRITIC_SYNCER_PORT}"


def rl_trainable_roles(args) -> tuple[str, ...]:
    """The roles the prepared algorithm trains, actor first; one syncer each."""
    return ("actor", "critic") if rl_needs_critic(args) else ("actor",)


def rl_role_syncers(args) -> tuple[RoleSyncer, ...]:
    return tuple(ROLE_SYNCERS[role] for role in rl_trainable_roles(args))


def role_syncer_commands(args, num_learners: int,
                         binary: str = "~/yeto-syncer") -> dict[str, str]:
    """One strict RL syncer invocation per trainable role (rounds, quorum and
    outer optimizer shared), each on its own port, checkpoint and tape."""
    return {syncer.role: _rl_role_syncer_command(args, num_learners, binary, syncer)
            for syncer in rl_role_syncers(args)}


def syncer_event_tape(args) -> str:
    """Where the syncer writes its merge record for this run's mode.

    syncer_command, LocalSyncer's strict-failure reader, and the W&B tape
    forwarder all resolve the path through here, so none of them can drift
    onto a file the syncer is not writing.
    """
    if getattr(args, "training_mode", "sft") == "rl":
        return RL_SYNCER_EVENT_TAPE
    return SYNCER_EVENT_TAPE


def _resume_if_exists(checkpoint: str) -> str:
    """Shell fragment adding --resume only when the checkpoint file exists.

    The syncer refuses --resume without a checkpoint, so an unconditional
    flag kills every fresh run; evaluated by the shell at each (re)start,
    a restart after the first checkpoint still resumes from it."""
    return f"$(test -f {checkpoint} && echo --resume)"


# syncer/src/main.rs default of --quorum-timeout-s (also the final-ACK timeout
# when --final-ack-timeout-s is omitted, as the launcher does).
SYNCER_DEFAULT_QUORUM_TIMEOUT_S = 900


def _rl_quorum_timeout_s(args) -> int | None:
    """``--rl-elastic-quorum-timeout-s`` (3.8) as whole seconds; None = syncer default."""
    value = getattr(args, "rl_elastic_quorum_timeout_s", None)
    if value is None or not getattr(args, "rl_elastic", False):
        return None
    if float(value) != int(value):
        raise ValueError("--rl-elastic-quorum-timeout-s must be whole seconds (syncer u64)")
    return int(value)


def _syncer_quorum_timeout(args) -> str:
    """The syncer's --quorum-timeout-s, the same value the island's pause
    budget uses; "" by default (syncer 900 s)."""
    value = _rl_quorum_timeout_s(args)
    return "" if value is None else f" --quorum-timeout-s {value}"


def _syncer_checkpoint_every(args) -> int:
    """Per-round checkpoints (1) unless the full-parameter preset sets an interval."""
    return int(getattr(args, "rl_sync_checkpoint_every", 1)) if rl_dense_sweep(args) else 1


def outer_delayed_nesterov_period(args) -> int | None:
    """The syncer's Delayed Nesterov period; None (unset or 1) is plain Nesterov."""
    period = getattr(args, "outer_delayed_nesterov_period", None)
    if period is None:
        return None
    if period < 1:
        raise ValueError("--outer-delayed-nesterov-period must be >= 1")
    return period if period > 1 else None


def _syncer_outer_optimizer(args) -> str:
    period = outer_delayed_nesterov_period(args)
    return (f" --outer-lr {args.outer_lr}"
            f" --outer-momentum {args.outer_momentum}"
            + (f" --outer-delayed-nesterov-period {period}" if period else ""))


def _syncer_bcast_delta(args) -> str:
    """S26 E3 bf16 downlink deltas (checked by rl_args._check_bcast_delta)."""
    full_every = int(getattr(args, "rl_sync_bcast_delta_full_every", 0) or 0)
    return f" --bcast-delta-full-every {full_every}" if full_every else ""


def rl_dense_sweep(args) -> bool:
    return (getattr(args, "training_mode", "sft") == "rl"
            and getattr(args, "rl_sync_preset", None) == FULL_PARAMETER_SYNC_PRESET)


def rl_sao_streaming(args) -> bool:
    return (getattr(args, "training_mode", "sft") == "rl"
            and getattr(args, "rl_sync_preset", None) == SAO_STREAMING_SYNC_PRESET)


def _syncer_profile_binding(args) -> str:
    """Full-parameter presets bind every HELLO to the syncer's semantic profile.

    full-parameter adds strict P-fragment policy sweeps; sao-streaming-full adds the
    frozen contract's timeouts and the publishable final checkpoint."""
    if rl_dense_sweep(args):
        async_checkpoint = " --async-checkpoint" if getattr(args, "rl_sync_async_checkpoint", False) else ""
        return f" --policy-sweep-fragments {args.fragments} --require-profile-binding{async_checkpoint}"
    if rl_sao_streaming(args):
        profile = args.rl_sao_manifest.profile
        return (f" --quorum-timeout-s {profile.quorum_timeout_s}"
                f" --final-ack-timeout-s {profile.final_ack_timeout_s}"
                " --mark-final-checkpoint --require-profile-binding")
    return ""


def rl_syncer_profile(args, num_learners: int):
    """The semantic profile of :func:`_rl_role_syncer_command` for a
    full-parameter preset, as the Rust syncer hashes it (yeto.syncer_profile).
    Islands send its sha256 in HELLO (``--syncer-profile-sha256``,
    P0-1.7b.1-Q6; the SAO runtime contracts carry the same hash)."""
    from ..syncer_profile import SYNCER_SEMANTIC_PROFILE_SCHEMA, SyncerSemanticProfile

    if rl_dense_sweep(args):
        quorum_timeout = _rl_quorum_timeout_s(args) or SYNCER_DEFAULT_QUORUM_TIMEOUT_S
        final_ack_timeout = quorum_timeout
        sweep_fragments, mark_final = args.fragments, False
    elif rl_sao_streaming(args):
        frozen = args.rl_sao_manifest.profile
        quorum_timeout, final_ack_timeout = frozen.quorum_timeout_s, frozen.final_ack_timeout_s
        sweep_fragments, mark_final = None, True
    else:
        raise ValueError("only full-parameter syncers are bound to a semantic profile")
    period = outer_delayed_nesterov_period(args)
    return SyncerSemanticProfile.from_mapping({
        **({"outer_delayed_nesterov_period": period} if period else {}),
        "schema": SYNCER_SEMANTIC_PROFILE_SCHEMA,
        "learners": num_learners,
        "quorum": args.quorum,
        "grace_ms": args.grace_ms,
        "grace_gamma": args.grace_gamma,
        "grace_tau": args.grace_tau,
        "pipeline": args.pipeline,
        "min_round_interval_ms": 0,
        "sync_interval_steps": args.sync_interval_steps,
        "delta_correction": args.delta_correction,
        "quorum_timeout_s": quorum_timeout,
        "final_ack_timeout_s": final_ack_timeout,
        "total_steps": args.rl_total_fragment_steps,
        "policy_sweep_fragments": sweep_fragments,
        "outer_lr": args.outer_lr,
        "outer_momentum": args.outer_momentum,
        "checkpoint_enabled": True,
        "checkpoint_every": _syncer_checkpoint_every(args),
        "resume": False,  # not hashed (v2): fresh start and restart agree
        "mark_final_checkpoint": mark_final,
        "learner_budget_steps": None,
        "max_base_lag": _rl_max_base_lag(args),
        "learner_weight": _rl_learner_weight(args),
        "require_profile_binding": True,
    })


_ISLAND_SCHEDULING_PARAMS = rl_option_pairs(
    "rl_quorum_theta", "rl_carry_gamma", "rl_soft_deadline_s", "rl_q_min", "rl_max_carry_lag",
    "rl_syncer_epoch", "rl_island_lease_s", "rl_final_grace_s",
    layer=CLI,
)

ELASTIC_DEBUG_DELAY_ENV = "YETO_RL_ELASTIC_DEBUG_DELAY"


def elastic_debug_delay_env(args) -> dict[str, str]:
    """0.25 test switch: islands delay each DELTA_TENSOR (soft deadline / carried_over
    on real hardware). Elastic only; {} by default."""
    spec = getattr(args, "rl_elastic_debug_delay_s", None)
    if not spec:
        return {}
    if island_scheduling_mode(args) != "elastic":
        raise ValueError("--rl-elastic-debug-delay-s needs --rl-island-scheduling elastic")
    from ..rl.elastic_client import parse_elastic_debug_delay

    parse_elastic_debug_delay(spec)
    return {ELASTIC_DEBUG_DELAY_ENV: str(spec)}


ELASTIC_DEBUG_PAUSE_ENV = "YETO_RL_ELASTIC_DEBUG_PAUSE"


def elastic_debug_pause_env(args) -> dict[str, str]:
    """Test switch: the given island's link to the syncer goes silent (no heartbeat,
    sends/receives held) for S seconds after base AFTER_V; the process keeps running.
    Used to exercise lease expiry -> "rejoin required" -> automatic re-JOIN on real
    hardware. Elastic only; {} by default."""
    spec = getattr(args, "rl_elastic_debug_pause", None)
    if not spec:
        return {}
    if island_scheduling_mode(args) != "elastic":
        raise ValueError("--rl-elastic-debug-pause needs --rl-island-scheduling elastic")
    from ..rl.elastic_client import parse_elastic_debug_pause

    parse_elastic_debug_pause(spec)
    return {ELASTIC_DEBUG_PAUSE_ENV: str(spec)}


DEFAULT_ISLAND_LEASE_S = 150.0  # passed to the syncer and to every island (heartbeat every lease/3)


def island_lease_s(args) -> float:
    value = getattr(args, "rl_island_lease_s", None)
    value = DEFAULT_ISLAND_LEASE_S if value is None else float(value)
    if not value > 0:
        raise ValueError("--rl-island-lease-s must be > 0")
    return value


def island_syncer_epoch(args) -> int:
    value = getattr(args, "rl_syncer_epoch", None)
    value = 0 if value is None else int(value)
    if value < 0:
        raise ValueError("--rl-syncer-epoch must be >= 0")
    return value


def syncer_status_reader(*, local_tape: str | None = None, ssh_host: str | None = None):
    """0.17: a zero-argument reader of the elastic syncer's status.json (next to its
    --event-tape): a local file on a head node, else ``ssh <syncer cluster> cat``.
    Returns the parsed snapshot, or None when absent / not the elastic schema."""
    from ..rl.engine.island_status import read_syncer_status, STATUS_SCHEMA

    if local_tape is not None:
        return lambda: read_syncer_status(os.path.expanduser(local_tape))

    remote = RL_SYNCER_EVENT_TAPE.rsplit("/", 1)[0] + "/status.json"

    def read() -> dict | None:
        try:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_host, f"cat {remote}"],
                                 capture_output=True, text=True, timeout=30)
            raw = json.loads(out.stdout) if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        return raw if isinstance(raw, dict) and raw.get("schema") == STATUS_SCHEMA else None
    return read


def resolve_island_syncer_epoch(args, read_status, *, timeout_s: float = 120.0,
                                poll_s: float = 2.0, sleep=time.sleep, clock=time.monotonic) -> int:
    """0.17: elastic without an explicit --rl-syncer-epoch takes the term the running
    syncer reports in status.json (and records it on ``args`` so every island
    command carries it). Unreadable within ``timeout_s``: keep 0 with a warning.
    Legacy, or an explicit value: unchanged, nothing is read."""
    if island_scheduling_mode(args) != "elastic" or getattr(args, "rl_syncer_epoch", None) is not None:
        return island_syncer_epoch(args)
    deadline = clock() + timeout_s
    while True:
        status = read_status()
        epoch = status.get("syncer_epoch") if status else None
        if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0:
            args.rl_syncer_epoch = epoch
            print(f"[launcher] syncer_epoch {epoch} read from the syncer's status.json")
            return epoch
        if clock() >= deadline:
            print("[launcher] WARN: could not read syncer_epoch from the syncer's status.json "
                  f"within {timeout_s:.0f}s; islands JOIN with syncer_epoch 0")
            return 0
        sleep(poll_s)


def island_scheduling_mode(args) -> str:
    """``--rl-island-scheduling`` (rl-inter-island-scheduling 0.13); legacy unless
    elastic is given explicitly. Elastic-only parameters with legacy are refused."""
    mode = getattr(args, "rl_island_scheduling", None) or "legacy"
    if mode not in ("legacy", "elastic"):
        raise ValueError(f"--rl-island-scheduling must be legacy|elastic, got {mode!r}")
    if mode == "legacy":
        given = [f for n, f in _ISLAND_SCHEDULING_PARAMS if getattr(args, n, None) is not None]
        if given:
            raise ValueError(", ".join(given) + " need --rl-island-scheduling elastic")
    return mode


ISLAND_HMAC_KEY_ENV = "YETO_ISLAND_HMAC_KEY"


def island_hmac_secret(args) -> dict[str, str]:
    """RL runs: ``{YETO_ISLAND_HMAC_KEY: key}``, to be shipped as a secret (sky
    ``secrets=``, Modal ``cfg.envs`` which becomes a ``modal.Secret``) -- never
    spliced into a command line, log or process list.

    elastic: the key must come from the launching environment (unchanged).
    legacy (secret-handling-hardening): the legacy syncer also requires the key
    (HELLO HMAC). Without one in the environment the launcher generates a
    random per-run key once and keeps it in ``os.environ`` so the syncer task,
    the head job and every island receive the same value. SFT: ``{}`` (its
    syncer runs with --allow-unauthenticated-islands, see design.md)."""
    if getattr(args, "training_mode", "sft") != "rl":
        return {}
    key = os.environ.get(ISLAND_HMAC_KEY_ENV, "")
    if not key:
        if island_scheduling_mode(args) != "legacy":
            raise ValueError(f"--rl-island-scheduling elastic needs {ISLAND_HMAC_KEY_ENV} "
                             "in the launching environment (island message HMAC key)")
        import secrets as _secrets

        key = os.environ[ISLAND_HMAC_KEY_ENV] = _secrets.token_hex(32)
    return {ISLAND_HMAC_KEY_ENV: key}


def _syncer_island_hmac_export(args) -> str:
    """RL: fail fast in the syncer's shell when the secret did not reach it.
    The value itself arrives as a secret env (island_hmac_secret) and the
    syncer reads the env var. "" for SFT."""
    if not island_hmac_secret(args):
        return ""
    return f': "${{{ISLAND_HMAC_KEY_ENV}:?island HMAC key secret missing}}" && '


def _syncer_expected_contract(args) -> str:
    """secret-handling-hardening: the head pins the island contract
    (--rl-island-contract-sha256) instead of letting the first island decide."""
    pinned = getattr(args, "rl_island_contract_sha256", None)
    return f" --expected-island-contract {pinned}" if pinned else ""


def _syncer_island_scheduling(args) -> str:
    """Syncer flags for the island-scheduling mode: "" in legacy, so the legacy
    command line is byte-for-byte the previous one. The syncer folds these into
    its session contract hash (the Python side does not compute that hash)."""
    if island_scheduling_mode(args) == "legacy":
        return ""
    from ..rl.engine.island_ledger import (DEFAULT_GAMMA, DEFAULT_MAX_CARRY_LAG,
                                          DEFAULT_QUORUM_MIN, DEFAULT_THETA)

    theta = getattr(args, "rl_quorum_theta", None)
    gamma = getattr(args, "rl_carry_gamma", None)
    soft = getattr(args, "rl_soft_deadline_s", None)
    if soft is None:
        soft = getattr(args, "rl_elastic_quorum_timeout_s", None) or 900
    q_min = getattr(args, "rl_q_min", None)
    lag = getattr(args, "rl_max_carry_lag", None)
    theta = DEFAULT_THETA if theta is None else float(theta)
    gamma = DEFAULT_GAMMA if gamma is None else float(gamma)
    q_min = DEFAULT_QUORUM_MIN if q_min is None else int(q_min)
    lag = DEFAULT_MAX_CARRY_LAG if lag is None else int(lag)
    if not 0.0 < theta <= 1.0 or not 0.0 <= gamma <= 1.0 or q_min < 1 or lag < 0:
        raise ValueError("--rl-quorum-theta in (0,1], --rl-carry-gamma in [0,1], "
                         "--rl-q-min >= 1, --rl-max-carry-lag >= 0")
    if float(soft) != int(soft) or int(soft) <= 0:
        raise ValueError("--rl-soft-deadline-s must be positive whole seconds")
    return (" --island-scheduling-mode elastic"
            f" --quorum-theta {theta!r}"
            f" --carry-gamma {gamma!r}"
            f" --soft-deadline-s {int(soft)}"
            f" --q-min {q_min}"
            f" --max-carry-lag {lag}"
            f" --syncer-epoch {island_syncer_epoch(args)}"
            f" --island-lease-s {island_lease_s(args)!r}"
            f" --final-grace-s {_final_grace_s(args, int(soft))}")


def _final_grace_s(args, soft_deadline_s: int) -> int:
    """0.26: the syncer's FINISHED window (whole seconds); default the soft deadline."""
    value = getattr(args, "rl_final_grace_s", None)
    value = soft_deadline_s if value is None else int(value)
    if value < 0:
        raise ValueError("--rl-final-grace-s must be >= 0")
    return value


def _rl_max_base_lag(args) -> int:
    """S27 P1 --rl-sync-max-base-lag (checked by rl_args); 0 refuses late pushes."""
    return int(getattr(args, "rl_sync_max_base_lag", 0))


def _rl_learner_weight(args) -> str:
    """S27 P1 --rl-sync-learner-weight (checked by rl_args); RL defaults to equal."""
    return getattr(args, "rl_sync_learner_weight", "equal")


def _rl_role_syncer_command(args, num_learners: int, binary: str,
                            syncer: RoleSyncer) -> str:
    """One role's strict RL syncer; --resume only once its checkpoint exists."""
    total_steps = getattr(args, "rl_total_fragment_steps", args.total_steps)
    return (
        f"{binary}"
        f" --port {syncer.port}"
        f" --learners {num_learners}"
        f" --quorum {args.quorum}"
        f" --grace-ms {args.grace_ms}"
        f" --grace-gamma {args.grace_gamma}"
        f" --grace-tau {args.grace_tau}"
        f" --pipeline {args.pipeline}"
        f" --sync-interval-steps {args.sync_interval_steps}"
        f" --delta-correction {args.delta_correction}"
        f" --total-steps {total_steps}"
        f"{_syncer_quorum_timeout(args)}"
        f"{_syncer_island_scheduling(args)}"
        f"{_syncer_expected_contract(args)}"
        f"{_syncer_profile_binding(args)}"
        f"{_syncer_bcast_delta(args)}"
        f"{_syncer_outer_optimizer(args)}"
        f" --max-base-lag {_rl_max_base_lag(args)}"
        f" --learner-weight {_rl_learner_weight(args)}"
        f" --checkpoint-path {syncer.checkpoint}"
        f" --checkpoint-every {_syncer_checkpoint_every(args)}"
        f" {_resume_if_exists(syncer.checkpoint)}"
        f" --event-tape {syncer.event_tape}"
    )


def syncer_command(args, num_learners: int, binary: str = "~/yeto-syncer") -> str:
    """The syncer invocation shared by the syncer-cluster task (local
    controller mode) and the head-node subprocess (head controller mode).
    --resume makes any restart pick up from the on-disk checkpoint.

    RL runs one syncer per trainable role (P0-D8): every role after the
    actor is backgrounded first; the actor syncer stays the foreground
    process (the job's health)."""
    if getattr(args, "training_mode", "sft") == "rl":
        commands = role_syncer_commands(args, num_learners, binary=binary)
        actor = commands.pop("actor")
        return (
            "mkdir -p ~/yeto-output && "
            + _syncer_island_hmac_export(args)
            + "".join(f"{{ {command} & }} && " for command in commands.values())
            + actor
        )
    if island_scheduling_mode(args) != "legacy":
        raise ValueError("--rl-island-scheduling elastic only applies to --training-mode rl")
    return (
        f"{binary}"
        f" --port {SYNCER_PORT}"
        f" --learners {num_learners}"
        f" --quorum {args.quorum}"
        f" --grace-ms {args.grace_ms}"
        f" --grace-gamma {args.grace_gamma}"
        f" --grace-tau {args.grace_tau}"
        f" --pipeline {getattr(args, 'pipeline', 2)}"
        f" --sync-interval-steps {getattr(args, 'sync_interval_steps', 24.0)}"
        f" --delta-correction {args.delta_correction}"
        f" --total-steps {args.total_steps}"
        f"{_syncer_outer_optimizer(args)}"
        " --checkpoint-path ~/yeto-state.ckpt"
        f" {_resume_if_exists('~/yeto-state.ckpt')}"
        f" --mark-final-checkpoint"
        f" --event-tape {SYNCER_EVENT_TAPE}"
        # secret-handling-hardening: SFT islands do not get the HMAC key yet
        # (known gap in design.md); the opt-out is explicit, never implicit.
        " --allow-unauthenticated-islands"
    )


# The syncer VM is x86_64 Linux; a binary built on any other submitting
# machine (macOS arm64 in particular) is an Exec-format-error away from a
# silent dead fleet, so cross builds happen ON the VM from the synced repo.
# rustup + a release build of the syncer adds ~2-4 min to syncer provision.
SYNCER_REMOTE_BUILD = (
    'if ! command -v cc >/dev/null; then sudo apt-get update -qq && '
    "sudo apt-get install -y -qq build-essential; fi\n"
    "command -v ~/.cargo/bin/cargo >/dev/null || "
    "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q\n"
    "~/.cargo/bin/cargo build --release --quiet "
    "--manifest-path ~/sky_workdir/syncer/Cargo.toml\n"
    "cp ~/sky_workdir/syncer/target/release/yeto-syncer ~/yeto-syncer"
)


def syncer_tape_sidecar(args, num_learners: int, binary: str = "~/yeto-syncer") -> tuple[str, str, dict]:
    """Setup step, run prefix, and envs that put a W&B tape forwarder on the
    syncer VM. Returns empty strings when telemetry is off.

    Local controller mode runs the syncer on its own cluster, so the event
    tape is not reachable from the controller on the submitting machine.
    Rather than shipping the tape back, the reader goes to the tape:
    `yeto.wandb_tape --follow` runs beside the syncer.

    It is backgrounded deliberately. The syncer must stay the job's
    foreground process, because FleetController reads that job's exit code
    as the syncer's health; and as a separate process the forwarder cannot
    take the syncer down with it. The forwarder resumes from its stored
    byte offset, so a restarted job continues the tape instead of logging
    every past merge a second time.

    Only yeto.wandb_tape and yeto.wandb_logger are imported over there, and
    both are stdlib-only — the syncer VM never needs the training stack.
    """
    if not getattr(args, "wandb", False):
        return "", "", {}
    setup = (
        "\npip install -q wandb || echo '[yeto-setup] wandb install failed; "
        "the syncer will run without telemetry' >&2"
    )
    forwarder = (
        f"python3 -m yeto.wandb_tape {syncer_event_tape(args)} --follow"
        f" --wandb-group {shlex.quote(args.cluster_prefix)}"
        f" --wandb-project {shlex.quote(args.wandb_project)}"
        f" --wandb-mode {args.wandb_mode}"
    )
    if getattr(args, "wandb_entity", None):
        forwarder += f" --wandb-entity {shlex.quote(args.wandb_entity)}"
    # Head mode's forwarder is handed the launch namespace directly; the
    # sidecar is a separate process, so the same config rides across as JSON
    # (the head job itself is launched the same way, see cli.cmd_launch_head).
    # build_config drops anything credential-shaped before it is serialized.
    from ..wandb_logger import build_config

    config = build_config(
        args, {"syncer_command": syncer_command(args, num_learners, binary=binary)}
    )
    forwarder += f" --config-json {shlex.quote(json.dumps(config, sort_keys=True))}"
    run_prefix = (
        "(cd ~/sky_workdir && PYTHONPATH=~/sky_workdir nohup "
        f"{forwarder} >/tmp/yeto-tape-wandb.log 2>&1 &) || true\n"
    )
    envs = {"YETO_RUN_GROUP": args.cluster_prefix}
    if os.environ.get("WANDB_API_KEY"):
        envs["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    return setup, run_prefix, envs


# Clouds where sky cannot open ports (sky/clouds/verda.py: no OPEN_PORTS).
# There the syncer's port is reachable because the platform has no
# firewall; yeto closes everything but SSH and the syncer with ufw inside
# the VM and proves reachability with an external TCP probe (design D7).
NO_OPEN_PORTS_CLOUDS = frozenset({"verda"})
PROBE_BANNER = b"yeto-probe"


def syncer_cloud(args) -> str:
    """Cloud of --syncer-region ('region' means AWS, else 'cloud/region')."""
    region = getattr(args, "syncer_region", "") or ""
    return region.split("/", 1)[0] if "/" in region else "aws"


def syncer_ports(args) -> list[int] | None:
    """`ports=` for the syncer/head resources; None where sky can't open them."""
    if syncer_cloud(args) in NO_OPEN_PORTS_CLOUDS:
        return None
    return [syncer.port for syncer in rl_role_syncers(args)]


def ufw_setup(port: int = SYNCER_PORT) -> str:
    """In-VM firewall for a public syncer on a cloud without security
    groups: deny inbound except SSH and the syncer port."""
    return (
        'SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo\n'
        "command -v ufw >/dev/null || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq ufw; }\n"
        "$SUDO ufw default deny incoming\n"
        "$SUDO ufw default allow outgoing\n"
        "$SUDO ufw allow 22/tcp\n"
        f"$SUDO ufw allow {int(port)}/tcp\n"
        "$SUDO ufw --force enable\n"
        "$SUDO ufw status verbose"
    )


def probe_listener_command(port: int = SYNCER_PORT, timeout_s: int = 300) -> str:
    """A one-shot listener on the syncer port that greets the first
    connection with PROBE_BANNER (used before the real syncer starts)."""
    code = (
        "import socket;s=socket.socket();"
        "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
        f"s.bind(('0.0.0.0',{int(port)}));s.listen(1);s.settimeout({int(timeout_s)});"
        f"c,_=s.accept();c.sendall({PROBE_BANNER!r});c.close();s.close()"
    )
    return f'python3 -c "{code}"'


def tcp_probe(host: str, port: int = SYNCER_PORT, *, expect: bytes | None = None, attempts: int = 20,
              delay: float = 3.0, timeout: float = 5.0, connect=None, sleep=time.sleep) -> tuple[bool, str]:
    """Connect from THIS machine (outside the cloud) to host:port.
    With `expect`, the peer must send those bytes first. (ok, detail)."""
    import socket

    connect = connect or socket.create_connection
    last = "not tried"
    for i in range(max(1, attempts)):
        try:
            with connect((host, int(port)), timeout=timeout) as conn:
                if expect is not None:
                    conn.settimeout(timeout)
                    got = conn.recv(len(expect))
                    if got != expect:
                        raise OSError(f"unexpected greeting {got!r}")
            return True, f"{host}:{port} reachable (attempt {i + 1})"
        except OSError as e:
            last = f"{type(e).__name__}: {e}"
        if i + 1 < attempts:
            sleep(delay)
    return False, f"{host}:{port} unreachable after {attempts} attempt(s): {last}"


def island_syncer_addrs(envs) -> str:
    """The syncer address(es) an island dials, for launch.log (no credentials)."""
    envs = envs or {}
    return " ".join(f"{k}={envs[k]}" for k in ("SYNCER_ADDR", "CRITIC_SYNCER_ADDR") if k in envs) or "no syncer"


def probe_syncer_ports(args, host: str, *, probe=None) -> list[str]:
    """Fail fast, before any island is started, unless every syncer port
    (actor 29400 and, for a critic algorithm, critic 29401) accepts a TCP
    connection from this machine. On clouds where sky opens the ports
    (security groups, e.g. Nebius) this is the only check that the rule
    really took effect: s13-g3-modal-20261007b spent the 900 s stall timeout
    with both islands silently redialing a Nebius syncer that never saw a
    connection. A pass here does not prove the islands' egress (Modal), only
    that the head side is open."""
    probe = probe or tcp_probe
    ports = [syncer.port for syncer in rl_role_syncers(args)]
    details = []
    for port in ports:
        ok, detail = probe(host, port)
        if not ok:
            raise RuntimeError(
                f"syncer port {port} not reachable from outside {syncer_cloud(args)} ({detail}); "
                "no island was started"
            )
        print(f"[launcher] syncer probe: {detail}", flush=True)
        details.append(detail)
    return details


def make_syncer_task(args, num_learners: int):
    import platform

    import sky

    cross = (platform.system(), platform.machine()) != ("Linux", "x86_64")
    if cross:
        tape_setup, tape_run, tape_envs = syncer_tape_sidecar(args, num_learners)
        print("[launcher] non-x86-Linux submitter: building the syncer on the syncer VM")
        task = sky.Task(
            name="yeto-syncer",
            setup=WAN_TUNING + "\n" + SYNCER_REMOTE_BUILD + tape_setup,
            run=tape_run + syncer_command(args, num_learners),
            workdir=str(REPO_ROOT),
            envs=tape_envs,
            secrets=island_hmac_secret(args) or None,
        )
        infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
        task.set_resources(
            sky.Resources(
                infra=infra,
                cpus="8+",
                memory=f"{args.syncer_memory}+",
                ports=syncer_ports(args),
                use_spot=False,
            )
        )
        if syncer_ports(args) is None:
            task.setup = ufw_setup() + "\n" + (task.setup or "")
        return task

    binary = build_syncer_binary()
    tape_setup, tape_run, tape_envs = syncer_tape_sidecar(args, num_learners)
    cmd = tape_run + "chmod +x ~/yeto-syncer && " + syncer_command(args, num_learners)
    task = sky.Task(
        name="yeto-syncer",
        setup=WAN_TUNING + tape_setup,
        run=cmd,
        file_mounts={"~/yeto-syncer": str(binary)},
        # The fast path normally ships only the binary; the forwarder needs
        # the yeto package, so telemetry pulls in the same workdir the
        # cross-build path already uses.
        workdir=str(REPO_ROOT) if getattr(args, "wandb", False) else None,
        envs=tape_envs,
        secrets=island_hmac_secret(args) or None,
    )
    # --syncer-region accepts "region" (AWS assumed) or "cloud/region".
    infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
    task.set_resources(
        sky.Resources(
            infra=infra,
            cpus="8+",
            memory=f"{args.syncer_memory}+",
            ports=syncer_ports(args),
            use_spot=False,
        )
    )
    if syncer_ports(args) is None:
        task.setup = ufw_setup() + "\n" + (task.setup or "")
    return task
