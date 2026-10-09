"""Miles flag rows and argv translation of the algorithm extensions
(change ``yeto-framework-decoupling`` task 4.3, audit A1).

The extension modules under :mod:`yeto.rl.algos` register only neutral spec
fields, validation and defaults. The "spec field <-> Miles flag" rows and the
spec -> Miles argv functions that they used to register themselves live here,
moved verbatim, so the algorithm layer no longer imports the Miles adapter.

Registration order equals the old import order of
:data:`yeto.rl.algos.EXTENSION_MODULES` (seq_adv, loss_variants, critic, sao),
so :data:`algorithm_flags.MAPPINGS` iterates in the same order and the
generated argv is byte-identical (golden samples, ``test_decoupling_golden``).

Imported once by :mod:`yeto.rl.adapters.miles.algorithm_flags` after
:func:`yeto.rl.engine.algorithm.load_extensions`.
"""

from __future__ import annotations

from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, load_extensions

from . import algorithm_flags as _af
from .algorithm_flags import FlagMapping, _float, _int, _num, _str, _value_row, register_flag

load_extensions()  # neutral fields first, in EXTENSION_MODULES order

from yeto.rl.algos.loss_variants import (  # noqa: E402
    DEFAULT_VARIANT,
    PARAM_FLAGS,
    VARIANT_FLAG,
    VARIANT_PARAMS,
    variant,
)
from yeto.rl.algos.sao import HL_GAUSS_SIGMA_RATIO, SAO_DIS, VALUE_REWARD_RANGE  # noqa: E402

# --------------------------------------------------------------------------
# seq_adv (rl-algo-seq-and-adv)
# --------------------------------------------------------------------------

# --gamma: default 1.0 is not emitted (argv snapshots unchanged); --lambd stays unmapped.
register_flag(_value_row("--gamma", "advantage.gamma", _float, emit=_num, default=1.0))

# --------------------------------------------------------------------------
# loss_variants (rl-algo-loss-variants)
# --------------------------------------------------------------------------


def _translate_variant(spec) -> list[str]:
    name = variant(spec)
    if name == DEFAULT_VARIANT:
        return []  # default GRPO argv unchanged
    argv = [VARIANT_FLAG, name]
    for param, _ in VARIANT_PARAMS[name]:  # explicit, never the fork's default
        argv += [PARAM_FLAGS[param], _num(getattr(spec.loss, param))]
    return argv


register_flag(FlagMapping(
    VARIANT_FLAG, "loss.policy_loss_variant", False, str,
    lambda v: [("loss.policy_loss_variant", v)], _translate_variant,
))
for _param, _flag in PARAM_FLAGS.items():
    register_flag(FlagMapping(
        _flag, f"loss.{_param}", False, _float,
        lambda v, p=_param: [(f"loss.{p}", v)], lambda spec: [],
    ))


# --------------------------------------------------------------------------
# critic (rl-algo-critic-family, design D2)
# --------------------------------------------------------------------------


def critic_argv(spec: AlgorithmSpec) -> list[str]:
    """Shared actor/critic PPO flags of the ports main stage.

    Empty without a critic (default GRPO argv unchanged). gamma / lambd /
    value_clip are always explicit (never Miles' defaults). The main stage runs
    in rebuild mode, which requires ``--num-critic-only-steps 0`` (Miles
    arguments.py:3210-3212); the warm-up is its own stage (design D5) whose
    product the run configuration passes as ``--critic-load``.
    """

    if not spec.execution.needs_critic:
        return []
    a, c = spec.advantage, spec.critic
    argv = ["--gamma", _num(a.gamma), "--lambd", _num(a.lambd),
            "--value-clip", _num(c.value_clip)]
    if c.critic_lr is not None:
        argv += ["--critic-lr", _num(c.critic_lr)]
    if c.critic_lr_warmup is not None:
        argv += ["--critic-lr-warmup-iters", str(c.critic_lr_warmup)]
    if c.critic_updates_per_step != 1:  # fork e07e51c07 (alias of --num-critic-epochs)
        argv += ["--critic-updates-per-step", str(c.critic_updates_per_step)]
    argv += ["--num-critic-only-steps", "0"]
    if c.init == "load":
        argv += ["--critic-load", c.load]
    return argv + gae_variant_argv(spec)


def gae_variant_argv(spec: AlgorithmSpec) -> list[str]:
    """Fork GAE extension flags (yeto-gae-variant ce96fc060, design D6).

    Empty for vanilla GAE with a fixed lambda, so plain PPO argv is unchanged.
    """

    a = spec.advantage
    argv: list[str] = []
    if a.gae_variant == "decoupled":
        argv += ["--gae-variant", "decoupled", "--gae-critic-lambd", _num(a.critic_lambd)]
    elif a.gae_variant in ("cross_segment_per_sample", "cross_segment_whole_rollout"):
        # CompactionRL (design D8): per_sample = the paper's form (one sample per
        # compaction segment, local GAE x (gamma*lambda)^{tokens_after});
        # whole_rollout = explicit control mode for the 9.5 ablation. Same fork names.
        argv += ["--gae-variant", a.gae_variant]
    if a.lambd_mode == "length_adaptive":
        argv += ["--gae-lambd-mode", "length_adaptive", "--gae-length-alpha", _num(a.alpha)]
    return argv


def positive_lm_argv(spec: AlgorithmSpec) -> list[str]:
    """VAPO positive-example LM loss (fork 70e3d7761); empty when unset.

    ``--positive-example-source success`` reads the reward function's boolean
    ``sample.metadata['success'/'is_correct']`` (missing -> fork error); ``reward``
    (reward > threshold) only when the spec declares the reward binary success.
    """

    coef = getattr(spec.loss, "positive_lm_coef", None)
    if coef is None:
        return []
    argv = ["--positive-example-lm-loss-coef", _num(coef),
            "--positive-example-source", spec.loss.positive_lm_source]
    if spec.loss.positive_lm_source == "reward":
        argv += ["--positive-example-reward-threshold",
                 _num(spec.loss.positive_lm_reward_threshold)]
    return argv


def _none(spec: AlgorithmSpec) -> list[str]:
    return []


register_flag(FlagMapping("--lambd", "advantage.lambd", False, _float,
                          lambda v: [("advantage.lambd", v)], critic_argv))
register_flag(FlagMapping("--value-clip", "critic.value_clip", False, _float,
                          lambda v: [("critic.value_clip", v)], _none))
register_flag(FlagMapping("--critic-lr", "critic.critic_lr", False, _float,
                          lambda v: [("critic.critic_lr", v)], _none))
register_flag(FlagMapping("--critic-lr-warmup-iters", "critic.critic_lr_warmup", False, _int,
                          lambda v: [("critic.critic_lr_warmup", v)], _none))
register_flag(FlagMapping("--num-critic-only-steps", "critic.warmup_steps", False, _int,
                          lambda v: [("critic.warmup_steps", v)], _none))
register_flag(FlagMapping("--critic-load", "critic.load", False, _str,
                          lambda v: [("critic.init", "load"), ("critic.load", v)], _none))

def _parse_gae_variant(raw: str) -> str:
    if raw == "cross_segment":
        raise _af.AlgorithmSpecError(
            "--gae-variant cross_segment is ambiguous and refused: use "
            "--gae-variant cross_segment_per_sample (CompactionRL, one sample per segment) or "
            "--gae-variant cross_segment_whole_rollout (explicit control mode)")
    return raw


register_flag(FlagMapping("--gae-variant", "advantage.gae_variant", False, _parse_gae_variant,
                          lambda v: [("advantage.gae_variant", v)], _none))
register_flag(FlagMapping("--gae-lambd-mode", "advantage.lambd_mode", False, _str,
                          lambda v: [("advantage.lambd_mode", v)], _none))
register_flag(FlagMapping("--gae-length-alpha", "advantage.alpha", False, _float,
                          lambda v: [("advantage.alpha", v)], _none))
register_flag(FlagMapping("--gae-critic-lambd", "advantage.critic_lambd", False, _float,
                          lambda v: [("advantage.critic_lambd", v)], _none))


register_flag(FlagMapping("--critic-updates-per-step", "critic.critic_updates_per_step", False,
                          _int, lambda v: [("critic.critic_updates_per_step", v)], _none))
# SAO's spelling of the same fork dest (e07e51c07); critic_argv emits the
# --critic-updates-per-step spelling.
register_flag(FlagMapping("--num-critic-epochs", "critic.critic_updates_per_step", False,
                          _int, lambda v: [("critic.critic_updates_per_step", v)], _none))
register_flag(FlagMapping("--positive-example-lm-loss-coef", "loss.positive_lm_coef", False,
                          _float, lambda v: [("loss.positive_lm_coef", v)], positive_lm_argv))
register_flag(FlagMapping("--positive-example-reward-threshold",
                          "loss.positive_lm_reward_threshold", False, _float,
                          lambda v: [("loss.positive_lm_reward_threshold", v)], _none))
register_flag(FlagMapping("--positive-example-source", "loss.positive_lm_source", False, _str,
                          lambda v: [("loss.positive_lm_source", v)], _none))

# --gamma: seq_adv's row emits it for REINFORCE++; critic_argv owns it under a critic.
_gamma_row = _af.MAPPINGS["--gamma"]
_af.MAPPINGS["--gamma"] = FlagMapping(
    _gamma_row.flag, _gamma_row.field, _gamma_row.switch, _gamma_row.parse, _gamma_row.absorb,
    lambda spec: [] if spec.execution.needs_critic else _gamma_row.translate(spec),
    _gamma_row.emitted_by_config,
)


# --------------------------------------------------------------------------
# sao (rl-algo-critic-family 8.2/8.3)
# --------------------------------------------------------------------------


def sao_fork_argv(spec: AlgorithmSpec) -> list[str]:
    """Fork flags for an SAO spec; empty for every other spec."""

    loss = spec.loss
    if loss.policy_objective != SAO_DIS:
        return []
    a, c = spec.advantage, spec.critic
    argv = ["--policy-objective", SAO_DIS,
            "--sao-dis-eps-low", _num(loss.sao_dis_eps_low),
            "--sao-dis-eps-high", _num(loss.sao_dis_eps_high)]
    if c.value_loss == "hl_gauss":
        argv += ["--value-loss-type", "classification",
                 "--value-num-bins", str(c.hl_gauss_bins),
                 "--value-target-type", "hl_gauss",
                 "--hl-gauss-sigma-ratio", _num(HL_GAUSS_SIGMA_RATIO),
                 "--value-reward-range", _num(VALUE_REWARD_RANGE[0]), _num(VALUE_REWARD_RANGE[1])]
    argv += ["--gae-variant", a.gae_variant, "--gae-lambd-mode", a.lambd_mode]
    if a.lambd_mode == "length_adaptive":
        argv += ["--gae-length-alpha", _num(a.alpha)]
    if a.gae_variant == "decoupled":
        argv += ["--gae-critic-lambd", _num(a.lambd)]
    return argv


register_flag(FlagMapping("--policy-objective", "loss.policy_objective", False, str,
                          lambda v: [("loss.policy_objective", v)]
                          # DIS compares against the rollout policy (fork forces
                          # use_rollout_logprobs), as sao_algorithm_spec declares.
                          + ([("execution.needs_rollout_logprobs", True)] if v == SAO_DIS else []),
                          sao_fork_argv))
register_flag(FlagMapping("--sao-dis-eps-low", "loss.sao_dis_eps_low", False, _float,
                          lambda v: [("loss.sao_dis_eps_low", v)], lambda spec: []))
register_flag(FlagMapping("--sao-dis-eps-high", "loss.sao_dis_eps_high", False, _float,
                          lambda v: [("loss.sao_dis_eps_high", v)], lambda spec: []))


# Value-head flags (fork 6b5bd88c). sao_fork_argv emits them for an hl_gauss
# critic; absorbing them keeps a raw --value-loss-type from bypassing the spec.
# --value-reward-range takes two values and is a translation constant
# (VALUE_REWARD_RANGE): adapter-owned, never absorbed (algorithm_flags._UNMAPPED).
def _value_loss_type(raw: str) -> str:
    if raw not in ("mse", "classification"):
        raise AlgorithmSpecError(f"expected mse or classification, got {raw!r}")
    return raw


def _hl_gauss_target(raw: str) -> str:
    if raw != "hl_gauss":
        raise AlgorithmSpecError(f"only hl_gauss is expressible (critic.value_loss), got {raw!r}")
    return raw


def _sigma_ratio(raw: str) -> float:
    value = _float(raw)
    if value != HL_GAUSS_SIGMA_RATIO:
        raise AlgorithmSpecError(
            f"the HL-Gauss sigma ratio is the translation constant {HL_GAUSS_SIGMA_RATIO}, got {raw!r}")
    return value


register_flag(FlagMapping(
    "--value-loss-type", "critic.value_loss", False, _value_loss_type,
    lambda v: [("critic.value_loss", "hl_gauss" if v == "classification" else "mse")],
    lambda spec: []))
register_flag(FlagMapping("--value-num-bins", "critic.hl_gauss_bins", False, _int,
                          lambda v: [("critic.hl_gauss_bins", v)], lambda spec: []))
register_flag(FlagMapping("--value-target-type", "critic.value_loss", False, _hl_gauss_target,
                          lambda v: [], lambda spec: []))
register_flag(FlagMapping("--hl-gauss-sigma-ratio", "critic.value_loss", False, _sigma_ratio,
                          lambda v: [], lambda spec: []))


# --------------------------------------------------------------------------
# policy-age limit (agentic-rollout-utilization 2.4, design decision 1)
# --------------------------------------------------------------------------
# Miles' boolean switches are derived from the neutral limit --rl-max-policy-age
# (``translate_run_config(max_policy_age=...)`` -> ``policy_age.policy_age_argv``,
# 4.1), never from the algorithm's tolerance (execution.max_policy_staleness,
# which only has to be >= the limit); they are never accepted as pass-through
# argv, which could not express the limit. Limit 0: nothing emitted.


def _refuse_partial_rollout(_value) -> list:
    raise AlgorithmSpecError(
        "Miles partial-rollout switches are derived from --rl-max-policy-age "
        "(agentic-rollout-utilization); do not pass them directly")


register_flag(FlagMapping("--partial-rollout", "execution.max_policy_staleness", True,
                          lambda raw: True, _refuse_partial_rollout, lambda spec: []))
register_flag(FlagMapping("--mask-offpolicy-in-partial-rollout", "execution.max_policy_staleness",
                          True, lambda raw: True, _refuse_partial_rollout, lambda spec: []))
