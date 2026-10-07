"""SAO on the ports engine (change ``rl-algo-critic-family``, tasks 8.2/8.3).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. It
translates the legacy SAO online recipe (``--sao-online-recipe`` applied by
``miles.backends.training_utils.sao.apply_sao_online_recipe`` on
agentenv/miles ``feat/sao-tbench21-e2e-validation`` @ 16a9bea4, the code the
streaming entry ``yeto.rl.sao_streaming_runtime`` runs) into an
:class:`AlgorithmSpec`, so the recipe is part of the algorithm hash instead of
a Miles-side side effect.

What is declared here:

* spec fields in the ``loss`` group (absent from the canonical JSON while
  None, so every pre-existing spec hash is unchanged):
  ``loss.policy_objective`` (``'sao_dis'``), ``loss.sao_dis_eps_low``,
  ``loss.sao_dis_eps_high``;
* :func:`sao_algorithm_spec` -- the recipe as a spec (sao_dis, length-adaptive
  policy lambda alpha=1.5, gamma = critic lambda = 1, decoupled GAE, HL-Gauss
  51-bin value loss, ``critic_updates_per_step = num_critic_epochs``);
* :func:`sao_role_contract` -- the role semantics the streaming runtime keeps
  (two roles with separate layouts, one syncer per role, lockstep paired
  fragments, critic optimizer steps = actor steps x num_critic_epochs);
* :func:`sao_fork_argv` -- the Miles fork flags (``yeto-sao`` branch,
  michaellchung/miles 6b5bd88c + ce96fc06 GAE variants) a spec translates to;
* rejections: SAO DIS fields without ``policy_objective='sao_dis'``, sao_dis
  without explicit bounds, sao_dis combined with another policy-loss variant,
  and sao_dis at the current ports pin (the fork commit carrying it is local
  and not pinned; the old entry stays the way to run SAO).

The second (critic) syncer itself is change task 4.2 and is not implemented
here. Import-light: no torch/miles at import time.
"""

from __future__ import annotations

import math
from typing import Any

from yeto.rl.engine.algorithm import (
    AlgorithmSpec,
    AlgorithmSpecError,
    register_field,
    register_rejection,
)
from yeto.rl.engine.miles_adapter.algorithm_flags import (
    FlagMapping,
    _float,
    _num,
    register_flag,
)

SAO_DIS = "sao_dis"
POLICY_OBJECTIVES = (SAO_DIS,)
# apply_sao_online_recipe (16a9bea4 miles/backends/training_utils/sao.py):
# domain -> (eps_low, eps_high), the two DIS bands of the paper.
SAO_DIS_BANDS: dict[str, tuple[float, float]] = {
    "coding": (0.8, 3.0),
    "reasoning": (0.3, 5.0),
}
SAO_ALPHA = 1.5
SAO_NUM_CRITIC_EPOCHS = 2
SAO_ACTOR_LR = 1e-6
SAO_CRITIC_LR = 5e-6
SAO_CRITIC_LR_WARMUP = 10
SAO_HL_GAUSS_BINS = 51
# Fork value-head constants the spec does not carry (fork defaults, emitted
# explicitly): HL-Gauss sigma / bin width and the categorical support.
HL_GAUSS_SIGMA_RATIO = 0.75
VALUE_REWARD_RANGE = (0.0, 1.0)

# Commits of michaellchung/miles carrying --policy-objective sao_dis and the
# classification value loss. Empty: yeto-sao 6b5bd88c is local (not pushed,
# not in an image), so the ports pin cannot run SAO yet.
FORK_COMMITS: frozenset[str] = frozenset()


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


def _parse_objective(path: str, value: Any) -> str | None:
    if value is None:
        return None
    if value not in POLICY_OBJECTIVES:
        raise AlgorithmSpecError(f"{path} must be one of {list(POLICY_OBJECTIVES)}, got {value!r}")
    return value


def _parse_eps(path: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise AlgorithmSpecError(f"{path} must be a positive finite number, got {value!r}")
    return value


register_field("loss", "policy_objective", default=None, parse=_parse_objective)
register_field("loss", "sao_dis_eps_low", default=None, parse=_parse_eps)
register_field("loss", "sao_dis_eps_high", default=None, parse=_parse_eps)


# --------------------------------------------------------------------------
# recipe -> spec
# --------------------------------------------------------------------------


def sao_algorithm_spec(domain: str, *, num_critic_epochs: int = SAO_NUM_CRITIC_EPOCHS,
                       critic_load: str | None = None) -> AlgorithmSpec:
    """The legacy ``--sao-online-recipe <domain>`` as an AlgorithmSpec."""

    if domain not in SAO_DIS_BANDS:
        raise AlgorithmSpecError(f"unknown SAO online recipe: {domain}")
    eps_low, eps_high = SAO_DIS_BANDS[domain]
    critic: dict[str, Any] = {
        "value_loss": "hl_gauss",
        "hl_gauss_bins": SAO_HL_GAUSS_BINS,
        "critic_lr": SAO_CRITIC_LR,
        "critic_lr_warmup": SAO_CRITIC_LR_WARMUP,
        "critic_updates_per_step": num_critic_epochs,
        "warmup_steps": 0,  # lockstep from rollout zero (no critic-only steps)
        "param_mode": "full",
    }
    if critic_load is not None:
        critic.update(init="load", load=critic_load)
    return AlgorithmSpec(
        advantage={
            "estimator": "ppo",
            "gamma": 1.0,
            # critic targets use the fixed lambda; the actor's lambda is
            # length adaptive (1 - 1/(alpha * l)), i.e. decoupled GAE.
            "lambd": 1.0,
            "lambd_mode": "length_adaptive",
            "alpha": SAO_ALPHA,
            "gae_variant": "decoupled",
        },
        loss={
            "policy_objective": SAO_DIS,
            "sao_dis_eps_low": eps_low,
            "sao_dis_eps_high": eps_high,
        },
        execution={"needs_critic": True, "needs_rollout_logprobs": True},
        entropy_coef=0.0,
        critic=critic,
    )


def recipe_settings(spec: AlgorithmSpec) -> dict[str, Any]:
    """The Miles ``args`` attributes the spec stands for, in the legacy
    recipe's vocabulary (to compare with ``apply_sao_online_recipe``)."""

    a, c, loss = spec.advantage, spec.critic, spec.loss
    if loss.policy_objective != SAO_DIS:
        raise AlgorithmSpecError("not an SAO spec (loss.policy_objective != 'sao_dis')")
    return {
        "advantage_estimator": "gae_adaptive",
        "policy_objective": loss.policy_objective,
        "gae_adaptive_mode": "adaptive" if a.lambd_mode == "length_adaptive" else "fixed",
        "gae_adaptive_alpha": a.alpha,
        "gae_adaptive_min_length": 1,
        "gamma": a.gamma,
        "critic_lambd": a.lambd,
        "num_critic_epochs": c.critic_updates_per_step,
        "critic_lr": c.critic_lr,
        "critic_lr_warmup_iters": c.critic_lr_warmup,
        "kl_coef": spec.kl_coef or 0.0,
        "use_kl_loss": False,
        "entropy_coef": spec.entropy_coef,
        "sao_dis_eps_low": loss.sao_dis_eps_low,
        "sao_dis_eps_high": loss.sao_dis_eps_high,
        "value_loss_type": "classification" if c.value_loss == "hl_gauss" else "mse",
        "value_num_bins": c.hl_gauss_bins,
    }


def sao_role_contract(spec: AlgorithmSpec, actor_steps_per_round: int) -> dict[str, Any]:
    """Role semantics of the SAO streaming runtime the spec keeps.

    Mirrors ``sao_streaming_runtime._validate_miles_runtime`` (optimizer
    accounting) and ``local_learner._ROLES_BY_ALGORITHM['sao']``.
    """

    if isinstance(actor_steps_per_round, bool) or not isinstance(actor_steps_per_round, int) \
            or actor_steps_per_round < 1:
        raise AlgorithmSpecError("actor_steps_per_round must be a positive int")
    epochs = spec.critic.critic_updates_per_step
    return {
        "algorithm": "sao",
        "roles": ("actor", "critic"),
        "separate_layouts": True,   # actor / critic parameter_layout_sha256
        "syncer_per_role": True,    # one syncer stream per role
        "lockstep_paired_fragments": True,
        "num_critic_only_steps": spec.critic.warmup_steps,
        "optimizer_steps_per_round": {
            "actor": actor_steps_per_round,
            "critic": actor_steps_per_round * epochs,
        },
    }


# --------------------------------------------------------------------------
# translation (fork flags)
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
                          lambda v: [("loss.policy_objective", v)], sao_fork_argv))
register_flag(FlagMapping("--sao-dis-eps-low", "loss.sao_dis_eps_low", False, _float,
                          lambda v: [("loss.sao_dis_eps_low", v)], lambda spec: []))
register_flag(FlagMapping("--sao-dis-eps-high", "loss.sao_dis_eps_high", False, _float,
                          lambda v: [("loss.sao_dis_eps_high", v)], lambda spec: []))


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------


def _reject_sao_dis(s: AlgorithmSpec) -> str | None:
    loss = s.loss
    has_eps = loss.sao_dis_eps_low is not None or loss.sao_dis_eps_high is not None
    if loss.policy_objective != SAO_DIS:
        if has_eps:
            return "loss.sao_dis_eps_low/high only apply to loss.policy_objective='sao_dis'"
        return None
    if loss.sao_dis_eps_low is None or loss.sao_dis_eps_high is None:
        return "loss.policy_objective='sao_dis' needs explicit loss.sao_dis_eps_low and _high"
    if not loss.sao_dis_eps_low < 1.0:
        return "loss.sao_dis_eps_low must be in (0, 1)"
    if getattr(loss, "policy_loss_variant", "policy_loss") != "policy_loss":
        return "loss.policy_objective='sao_dis' replaces pg_loss; drop loss.policy_loss_variant"
    if not s.execution.needs_critic:
        return "loss.policy_objective='sao_dis' is the SAO actor objective; it needs the critic"
    return None


def _reject_sao_not_at_pin(s: AlgorithmSpec) -> str | None:
    if s.loss.policy_objective != SAO_DIS:
        return None
    from yeto.rl import MILES_NEXT_COMMIT

    if MILES_NEXT_COMMIT in FORK_COMMITS:
        return None
    return (
        "loss.policy_objective='sao_dis' needs the Miles fork SAO port (yeto-sao, not yet "
        "pinned); run SAO through the streaming entry (yeto.rl.sao_streaming_runtime) "
        "until the pin carries it"
    )


register_rejection("sao_dis_fields", _reject_sao_dis)
register_rejection("sao_not_at_pin", _reject_sao_not_at_pin)
