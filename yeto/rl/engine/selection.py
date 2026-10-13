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
    full_parameter = (
        tuning == "full"
        or parameter_mode == "full"
        or sync_preset in FULL_PARAMETER_SYNC_PRESETS
    )
    if full_parameter and island_scheduling == "elastic":
        # P0-D7 follow-up (tasks 1.7b.6): the elastic syncer has no fragment frames yet.
        reasons.append("full-parameter training with --rl-island-scheduling elastic")
    elif tuning not in ("lora", "full") or parameter_mode not in ("lora", "full"):
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

