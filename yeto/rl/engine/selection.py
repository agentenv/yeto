"""The ports engine's support matrix (task 5.1; rl-code-quality-cleanup 1.8).

Import-light (no torch/ray/miles): the CLI, launcher, SSH harness, learner
and benchmark all call :func:`require_ports_supported` before any remote or
GPU work starts.

The ports engine (``IslandDriver``) is the only Miles engine path.  It runs
causal LMs with LoRA or full parameters, GRPO and PPO critics, colocated or
fixed-partition placement, the strict-avg / decoupled presets and the
full-parameter presets (the strict whole-model sweep, which also carries
DeepSeek V4 expert-full, and SAO streaming).  Every rejection that remains
is a combination no engine runs; each one names its reason below.
"""

from __future__ import annotations

from collections.abc import Sequence

# ``--rl-engine`` keeps accepting its one value so existing commands parse.
RL_ENGINES = ("ports",)
# P0-D8: one full-parameter sync; the preset only picks its profile (the
# strict whole-model sweep, or SAO's actor + critic streaming).
# ``full-parameter`` names the transport (every parameter of the synced state
# is swept), not which parameters train: DeepSeek V4 expert-full rides it too
# (P0-1.7b.2-Q2).  ``dense-full`` is its old name, kept as an alias.
FULL_PARAMETER_SYNC_PRESET = "full-parameter"
SAO_STREAMING_SYNC_PRESET = "sao-streaming-full"
FULL_PARAMETER_SYNC_PRESETS = frozenset({FULL_PARAMETER_SYNC_PRESET, SAO_STREAMING_SYNC_PRESET})
SYNC_PRESETS = ("strict-avg", "decoupled", FULL_PARAMETER_SYNC_PRESET, SAO_STREAMING_SYNC_PRESET)
SYNC_PRESET_ALIASES = {"dense-full": FULL_PARAMETER_SYNC_PRESET}
# S27 P1: the syncer merge of decoupled RL's matrix fragments (yeto.rl.core).
RL_MATRIX_MERGES = ("avg", "rda")
# S27 P2: how the full-parameter preset schedules its fragment rounds. "sweep"
# is the strict policy sweep (every island per round, plain average, the
# control); "async" runs asynchronous sharded fragment rounds.
FULL_PARAMETER_ROUNDS = ("sweep", "async")
# How a multi-island RL run fills the sync options the command leaves unset.
# "async" (default): asynchronous sharded rounds, quorum 1 with the adaptive
# grace window, tokens^2/steps island weights, RDA matrix merge, outer
# Nesterov, int4 push. "strict": the strict-roster control (every island per
# round, plain average, f32), i.e. the values each option had before the
# modes existed. An option given on the command line always wins.
RL_SYNC_MODES = ("async", "strict")
# The adaptive grace window's cap in the async mode. The window itself is
# computed from the islands' measured step time (--grace-gamma, --grace-tau);
# the cap only bounds a slow round, it does not fix the wait.
ASYNC_GRACE_CAP_MS = 300_000
# The async mode's outer step needs enough local progress per merge: with
# fewer than this many local RL rounds per sync (H), the default outer step is
# plain averaging (lr 1.0, momentum 0); from H on it is Nesterov (lr 0.7,
# momentum 0.9). On GPU, 2 full-parameter islands with H = 1 diverged under
# Nesterov 0.7/0.9 and trained normally with plain averaging.
ASYNC_NESTEROV_MIN_H = 4


class SyncModeDefault:
    """Marks an argparse default that ``--rl-sync-mode`` may replace.

    The marked value equals the strict-mode value, so code that reads the
    option before the mode is resolved sees the strict default."""

    __slots__ = ()


class _SyncModeDefaultStr(SyncModeDefault, str):
    __slots__ = ()


class _SyncModeDefaultInt(SyncModeDefault, int):
    __slots__ = ()


def sync_mode_default(value: str | int) -> str | int:
    """``value`` marked as a default the sync mode resolves."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"no sync-mode default for {value!r}")
    return (_SyncModeDefaultStr if isinstance(value, str) else _SyncModeDefaultInt)(value)


def is_sync_mode_default(value: object) -> bool:
    """True when ``value`` is still the unset (marked) default."""

    return isinstance(value, SyncModeDefault)


def plain(value: object) -> object:
    """``value`` without its sync-mode mark."""

    if isinstance(value, str):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return value


def canonical_sync_preset(value: str) -> str:
    """argparse type for the sync preset flags: resolve an alias to its preset."""

    return SYNC_PRESET_ALIASES.get(value, value)


class UnsupportedPortsCombination(ValueError):
    """The requested run is outside what the ports engine runs."""


def _extra_argv_facts(extra_argv: Sequence[str]):
    """Facts read from pass-through backend argv by the Miles adapter (decoupling 4.3, E15)."""

    from . import backends

    return backends.module("selection_argv", backends.DEFAULT_BACKEND).extra_argv_facts(extra_argv)


def ports_rejections(
    *,
    sync_preset: str = "strict-avg",
    parameter_mode: str = "lora",
    model_kind: str = "causal-lm",
    tuning: str = "lora",
    expert_full_count: int = 0,
    rollout_num_gpus: int | None = None,
    use_critic: bool = False,
    advantage_estimator: str = "grpo",
    extra_argv: Sequence[str] = (),
    placement: str = "colocated",
    island_scheduling: str = "legacy",
) -> list[str]:
    """Reasons the combination is outside the ports matrix (empty = OK)."""

    reasons: list[str] = []
    sync_preset = canonical_sync_preset(sync_preset)
    facts = _extra_argv_facts(extra_argv)
    # The island loop, the LoRA layouts and the rollout engine are causal-LM only.
    if model_kind != "causal-lm":
        reasons.append(f"model kind {model_kind!r} (ports supports causal LMs only)")
    if island_scheduling == "elastic" and sync_preset == SAO_STREAMING_SYNC_PRESET:
        # The SAO island session has no pool membership (heartbeat, re-JOIN);
        # the other presets' limits are checked by the launcher (rl_args).
        reasons.append("the sao-streaming-full preset with membership options (island pool)")
    if tuning not in ("lora", "full") or parameter_mode not in ("lora", "full"):
        # Input check: no other parameter mode exists.
        reasons.append(f"tuning {tuning!r} / parameter mode {parameter_mode!r}")
    if sync_preset not in SYNC_PRESETS:
        # Input check: no other sync preset exists.
        reasons.append(f"sync preset {sync_preset!r}")
    # P0 1.7b.2: expert-full is a LoRA hybrid; only the full-parameter sync
    # carries its clone-expert weights.
    if int(expert_full_count or 0) > 0 and sync_preset != FULL_PARAMETER_SYNC_PRESET:
        reasons.append(f"expert-full tuning without --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET}")
    estimator = facts.advantage_estimator or advantage_estimator
    # ``--use-critic`` is not an upstream Miles flag (the critic comes from the
    # algorithm spec), and Miles has no estimator besides GRPO and PPO that the
    # algorithm spec maps; a critic without PPO has no value loss to train.
    if facts.critic_flag or estimator not in ("grpo", "ppo") or (
        use_critic and estimator != "ppo"
    ):
        reasons.append(f"critic / non-GRPO advantage estimator ({estimator})")
    # Input check: no other placement exists.
    if placement not in ("colocated", "fixed-partition"):
        reasons.append(f"placement {placement!r}")
    # The placement translation owns the rollout GPU count; a pass-through
    # flag would bypass it.
    if facts.rollout_gpus_flag:
        reasons.append("fixed partition via --rollout-num-gpus in extra argv (use --rl-placement fixed-partition)")
    if placement == "fixed-partition":
        # rl-infra-spec 2.1: LoRA fixed partition (partitioned-serial driver).
        if rollout_num_gpus is None or int(rollout_num_gpus) < 1:
            reasons.append("fixed partition needs --rollout-num-gpus >= 1")
    elif rollout_num_gpus is not None:
        reasons.append(
            "fixed partition (dedicated rollout GPUs) without --rl-placement fixed-partition"
        )
    return reasons


def require_ports_supported(**kwargs) -> None:
    reasons = ports_rejections(**kwargs)
    if reasons:
        raise UnsupportedPortsCombination("unsupported RL run: " + "; ".join(reasons))

