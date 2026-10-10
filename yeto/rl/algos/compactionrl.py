"""CompactionRL declaration (change ``rl-algo-critic-family`` task 9.3, design D8).

CompactionRL (arXiv 2607.05378v1, sec. 4-5) = PPO with a critic, trained on
compacted agent rollouts (rollout side: :mod:`yeto.rl.compaction`):
cross-segment GAE (eq. 13-15; ``cross_segment_per_sample``), length-adaptive lambda with alpha 1.5,
token-level loss normalisation (eq. 12), critic lr 3e-6, 2 value updates per
policy update, 50 steps of value pretraining from the policy checkpoint (run
as warm-up stage W, design D5), one rollout per prompt (a run-level setting).

gamma 1.0 and KL 0 come from design D8 / tasks 9.3; the paper does not state
them (see :data:`NOT_IN_PAPER`). Execution is not declared before GPU G1
(task 9.4): ``features:gae_cross_segment`` and ``features:critic_multi_update``
are undeclared, and ``--critic-updates-per-step`` is a requested fork flag.
"""

from __future__ import annotations

from typing import Any

from yeto.rl.engine.algorithm import AlgorithmSpec, load_extensions

# (spec path, value, source).
PAPER_PARAMETERS: tuple[tuple[str, Any, str], ...] = (
    ("advantage.gae_variant", "cross_segment_per_sample",
     "sec. 4.2 eq. 13-15 (cross-trajectory GAE, one sample per segment)"),
    ("advantage.lambd_mode", "length_adaptive", "sec. 5.1: lambda = 1 - 1/(alpha*l)"),
    ("advantage.alpha", 1.5, "sec. 5.1: alpha 1.5"),
    ("loss.aggregation", "token", "sec. 4.2 eq. 12: token-level normalisation"),
    ("critic.critic_lr", 3e-6, "sec. 5.1: critic lr 3e-6"),
    ("critic.critic_updates_per_step", 2, "sec. 5.1: 2 value updates per policy update"),
    ("critic.warmup_steps", 50, "sec. 5.1: critic from policy ckpt, 50 value-pretraining steps"),
)
# Values from design D8 / tasks 9.3 that the paper does not state.
DESIGN_PARAMETERS: tuple[tuple[str, Any, str], ...] = (
    ("advantage.gamma", 1.0, "design D8 / task 9.3 (paper: not given)"),
)

# Run-level settings (not part of the algorithm hash).
PAPER_RUN_SETTINGS: dict[str, Any] = {
    "n_samples_per_prompt": 1,  # sec. 5.1 group size 1
    "actor_lr": 2e-6,  # sec. 5.1 (Adam)
    "global_batch_size": 128,  # sec. 5.1 (unit: segments or rollouts not stated)
    "max_response_tokens": 10240,  # sec. 5.1 per assistant response
    "context_budget": {"GLM-4.7-Flash": 65536, "GLM-4.5-Air-SFT": 81920},  # sec. 5.1 "64k"/"80k"
    "t_comp": 10240,  # sec. 4.1 eq. 7
    "keep_recent_steps": 2,  # eq. 9, k=2, reduced when necessary
    "max_compactions": 3,  # sec. 5.1
}

NOT_IN_PAPER: dict[str, str] = {
    "advantage.gamma": "not given; 1.0 from design D8",
    "kl": "not given; design D8 kl=0 (no KL placement; Miles shared PPO requires kl_coef 0)",
    "loss.eps_clip": "not given numerically; Miles default",
    "critic.value_clip": "not given; Miles default 0.2",
    "advantage.lambd": "unused under length_adaptive; filled with 1.0",
    "length_adaptive l": "paper: 'response length'; per-segment vs whole rollout not stated; "
                         "confirmed (design D8): whole rollout's optimized-token count",
    "segment-end bootstrap": "not stated; confirmed (design D8): V=0 at every segment end",
    "critic target": "not stated; confirmed (design D8): local (uncorrected) return = local "
                     "advantage + V",
    "summary template": "q_sum section names / <analysis> text / u_resume text not given",
}


def compactionrl_whole_rollout_control_spec(**overrides: Any) -> AlgorithmSpec:
    """Ablation arm (task 9.5): the explicit control mode ``cross_segment_whole_rollout``
    (one sample per rollout + per-token segment ids; earlier segments get no terminal
    reward, unlike eq. 15). Everything else equals :func:`compactionrl_spec`."""

    advantage = {"gae_variant": "cross_segment_whole_rollout", **overrides.pop("advantage", {})}
    return compactionrl_spec(advantage=advantage, **overrides)


def compactionrl_spec(**overrides: Any) -> AlgorithmSpec:
    """The CompactionRL spec; ``overrides`` replace group dicts' keys."""

    load_extensions()
    groups: dict[str, dict[str, Any]] = {
        "advantage": {"estimator": "ppo"},
        "execution": {"needs_critic": True},
        "loss": {},
        "critic": {},
    }
    for path, value, _ in PAPER_PARAMETERS + DESIGN_PARAMETERS:
        group, name = path.split(".")
        groups[group][name] = value
    for group, values in overrides.items():
        groups.setdefault(group, {}).update(values)
    return AlgorithmSpec(**groups)


# --- Rollout-side switch (progress.md "S13 Codex 压缩接线") ------------------------
# The trainer-side gae variant decides whether rollouts must be compacted:
# ``cross_segment_per_sample`` (this spec) needs one sample per compaction
# segment, which only the Codex bridge (``harness.codex.compaction_bridge``,
# switched on by ``YETO_CODEX_COMPACTIONRL``) produces.  The control arm
# ``cross_segment_whole_rollout`` needs the whole rollout as ONE sample with
# per-token segment ids; the bridge runs every segment in its own session-server
# session with a rebuilt context, so its segments cannot be concatenated into
# one forward pass -> that arm is rejected on the Codex harness (with or
# without the switch: without compaction there are no segments to ablate).
COMPACTION_SWITCH_ENV = "YETO_CODEX_COMPACTIONRL"
COMPACTION_T_COMP_ENV = "YETO_CODEX_COMPACTIONRL_T_COMP"
COMPACTED_GAE_VARIANT = "cross_segment_per_sample"
# yeto's Codex generate wrapper (segment collection); the launcher checks it
# without importing the adapter layer (import boundary, yeto/rl/engine/README.md).
CODEX_OPENENV_GENERATE = "yeto.rl.harness.codex.codex_openenv_generate.generate"
WHOLE_ROLLOUT_CONTROL_GAE_VARIANT = "cross_segment_whole_rollout"
_SWITCH_FALSE = frozenset({"", "0", "false", "no", "off"})
_SWITCH_TRUE = frozenset({"1", "true", "yes", "on"})


def compaction_switch(raw: Any) -> bool:
    """Parse ``YETO_CODEX_COMPACTIONRL`` (same vocabulary as the bridge)."""
    value = "" if raw is None else str(raw).strip().lower()
    if value in _SWITCH_FALSE:
        return False
    if value in _SWITCH_TRUE:
        return True
    raise ValueError(f"{COMPACTION_SWITCH_ENV} must be a boolean flag, got {raw!r}")


def check_t_comp(raw: Any) -> None:
    if raw is None or str(raw).strip() == "":
        return
    text = str(raw).strip()
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"{COMPACTION_T_COMP_ENV} must be a positive integer, got {raw!r}")


def check_rollout_compaction(
    gae_variant: Any,
    switch_on: bool,
    *,
    harness_compacts: bool,
    t_comp: Any = None,
) -> None:
    """Fail closed unless spec, harness and switch agree.

    ``gae_variant``: the spec's ``advantage.gae_variant`` (= Miles ``--gae-variant``);
    ``harness_compacts``: the run's agent can compact (the Codex OpenEnv agent).
    """
    if gae_variant == WHOLE_ROLLOUT_CONTROL_GAE_VARIANT and harness_compacts:
        raise ValueError(
            f"gae_variant {WHOLE_ROLLOUT_CONTROL_GAE_VARIANT} (CompactionRL control arm) is not "
            "supported on the Codex harness: its compaction runs one session (one sample) per "
            "segment on a rebuilt context, which cannot be concatenated into the single sample "
            "this mode needs"
        )
    if gae_variant == COMPACTED_GAE_VARIANT:
        if not harness_compacts:
            raise ValueError(
                f"gae_variant {COMPACTED_GAE_VARIANT} (CompactionRL) needs compacted rollouts; "
                "only the Codex OpenEnv harness can compact"
            )
        if not switch_on:
            raise ValueError(
                f"gae_variant {COMPACTED_GAE_VARIANT} (CompactionRL) requires "
                f"{COMPACTION_SWITCH_ENV}=1 on the rollout workers"
            )
    elif switch_on:
        raise ValueError(
            f"{COMPACTION_SWITCH_ENV} is on but gae_variant is {gae_variant!r}; rollout "
            f"compaction is only valid with {COMPACTED_GAE_VARIANT} (CompactionRL)"
        )
    if t_comp is not None and str(t_comp).strip() != "":
        if not switch_on:
            raise ValueError(f"{COMPACTION_T_COMP_ENV} is set without {COMPACTION_SWITCH_ENV}")
        check_t_comp(t_comp)
