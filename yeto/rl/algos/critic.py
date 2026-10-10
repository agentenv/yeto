"""Critic family on the ports engine (change ``rl-algo-critic-family``).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. The
identity (``CriticSpec``, the advantage-side ``lambd`` / ``lambd_mode`` /
``alpha`` / ``gae_variant`` fields, the spec-only rejection matrix) lives in
``yeto.rl.engine.algorithm``; this module adds what needs the adapter or the
run configuration:

* flag rows (design D1/D2): ``--lambd``, ``--value-clip``, ``--critic-lr``,
  ``--critic-lr-warmup-iters``, ``--num-critic-only-steps`` (extra argv N =
  the warm-up length, run as its own stage, design D5), ``--critic-load``;
  ``--gamma`` stays the ``seq_adv`` row (``advantage.gamma`` is shared with
  REINFORCE++), and under a critic :func:`critic_argv` emits it instead;
* run-level rejections (design D2), checked before any GPU process by the
  launcher (``launch_problems``) and by the learner before it joins outer
  sync (``island_problems``): elastic reconfiguration / ``--indep-dp``,
  ``--deploy-component trainer``, critic GPU counts different from the
  actor's, decoupled outer sync.

``seq_adv``'s own source is not edited (its PluginRef source hash is part of
the GDPO / MaxRL / MAPO algorithm hashes); its gamma rule and ``--gamma``
translation are wrapped here so a critic algorithm can use ``--gamma``.

Import-light: no torch/miles at import time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import yeto.rl.algos.seq_adv  # noqa: F401  (owns advantage.gamma / --gamma; load first)
from yeto.rl.engine import algorithm as _alg
from yeto.rl.engine.algorithm import (
    AlgorithmSpec,
    register_field,
    register_island_check,
    register_launch_check,
    register_mechanism,
    register_rejection,
)

# Fork-only Miles flags (not in upstream c35702e): yeto-gae-variant ce96fc060
# (--gae-*) and yeto-vapo cbf8c4737 (--positive-example-*).
FORK_FLAGS = frozenset({
    "--gae-variant", "--gae-lambd-mode", "--gae-length-alpha", "--gae-critic-lambd",
    "--positive-example-lm-loss-coef", "--positive-example-reward-threshold",
    "--positive-example-source",
    # CompactionRL 2 / SAO 2 critic updates per policy update: fork e07e51c07
    # (yeto-critic-family), one dest; the mechanism stays undeclared until GPU G1.
    "--critic-updates-per-step", "--num-critic-epochs",
})


# Miles argv translation (critic_argv / gae_variant_argv / positive_lm_argv) and
# flag rows: yeto.rl.adapters.miles.algo_flag_rows (decoupling 4.3).

# --------------------------------------------------------------------------
# VAPO positive-example LM loss (change 7.2; VAPO arXiv 2504.05118 sec. 4.3 eq. 9-10)
# --------------------------------------------------------------------------


def _positive_coef(path: str, value: Any) -> float | None:
    return _alg._number(path, value, low=0.0, low_open=True)


register_field("loss", "positive_lm_coef", default=None, parse=_positive_coef)
register_field("loss", "positive_lm_reward_threshold", default=None,
               parse=lambda path, v: _alg._number(path, v))
# How a sample is judged positive (user decision C, design D7): "success" = the
# reward function's explicit boolean success field; "reward" = reward > threshold,
# allowed only as a declaration that the reward is a binary success signal.
POSITIVE_LM_SOURCES = ("success", "reward")
register_field("loss", "positive_lm_source", default=None,
               parse=lambda path, v: _alg._optional_choice(path, v, POSITIVE_LM_SOURCES))


def _reject_positive_lm(spec: AlgorithmSpec) -> str | None:
    coef = spec.loss.positive_lm_coef
    source = spec.loss.positive_lm_source
    threshold = spec.loss.positive_lm_reward_threshold
    if coef is None:
        if source is not None or threshold is not None:
            return ("loss.positive_lm_source / positive_lm_reward_threshold only apply with "
                    "loss.positive_lm_coef")
        return None
    if source is None:
        return ("loss.positive_lm_coef needs loss.positive_lm_source: 'success' (explicit "
                "success field from the reward function) or 'reward' (only when a positive "
                "reward always means complete success)")
    if source == "reward" and threshold is None:
        return ("loss.positive_lm_source='reward' needs loss.positive_lm_reward_threshold "
                "(positive when reward is strictly greater)")
    if source == "success" and threshold is not None:
        return ("loss.positive_lm_reward_threshold only applies with "
                "loss.positive_lm_source='reward'")
    return None


register_rejection("positive_lm_threshold", _reject_positive_lm)


def positive_lm_success_problems(spec: AlgorithmSpec, values: Mapping[str, Any]) -> list[str]:
    """``loss.positive_lm_source='success'`` needs a reward function declared
    (``yeto.rl.reward_declarations``) to write ``sample.metadata['success']``.

    Without it the Miles fork raises at the first critic step on the island
    (S19 #6 s19-vapo-g3-20261010a: out-of-tree ``gsm8k_reward:score`` wrote no
    flag; S13 G1 passed with ``yeto.rl.gsm8k_reward:score``). Skipped when the
    caller passes no ``reward_function`` or it cannot be imported here.
    """

    if spec.loss.positive_lm_coef is None or spec.loss.positive_lm_source != "success":
        return []
    reward_function = values.get("reward_function")
    if not reward_function:
        return []
    from yeto.rl.reward_declarations import reward_declares_success

    # None (not importable here): no verdict; the Miles fork still refuses on
    # the island. Sub-agent decision: test fixtures use placeholder references.
    if reward_declares_success(reward_function) is not False:
        return []
    return [f"loss.positive_lm_source='success' needs a reward function that writes "
            f"sample.metadata['success']; {reward_function} is not declared with "
            "yeto.rl.reward_declarations.writes_success. Use e.g. "
            "yeto.rl.gsm8k_reward:score, or positive_lm_source='reward' only when a "
            "positive reward always means complete success"]


register_launch_check("positive_lm_success", positive_lm_success_problems)

# --------------------------------------------------------------------------
# Stage-W value-quality gate (user decision B, design D5/D7). Absent (None) =
# record only, never block; absent from the canonical JSON while None, so no
# existing spec hash changes. Checked by critic_warmup.finish_warmup/load_product.
# --------------------------------------------------------------------------
WARMUP_GATE_FIELDS = (
    "warmup_max_value_mse", "warmup_max_value_rel_error",
    "warmup_max_calibration_error", "warmup_min_explained_variance",
)
for _name in WARMUP_GATE_FIELDS[:3]:
    register_field("critic", _name, default=None,
                   parse=lambda path, v: _alg._number(path, v, low=0.0))
register_field("critic", "warmup_min_explained_variance", default=None,
               parse=lambda path, v: _alg._number(path, v))


def _reject_warmup_gate(spec: AlgorithmSpec) -> str | None:
    set_fields = [n for n in WARMUP_GATE_FIELDS if getattr(spec.critic, n, None) is not None]
    if not set_fields:
        return None
    if not spec.execution.needs_critic or not spec.critic.warmup_steps:
        return f"critic.{set_fields} only apply to a critic algorithm with critic.warmup_steps > 0"
    return None


register_rejection("critic_warmup_gate", _reject_warmup_gate)
register_mechanism("features", "positive_example_lm_loss",
                   lambda s: s.loss.positive_lm_coef is not None)
# Fork-only GAE variants: undeclared by the Miles adapter until GPU G1 (task 7.3).
register_mechanism("features", "gae_decoupled",
                   lambda s: s.advantage.gae_variant == "decoupled")
register_mechanism("features", "gae_length_adaptive",
                   lambda s: s.advantage.lambd_mode == "length_adaptive")
# CompactionRL (change 9.3): undeclared until GPU G1 (task 9.4).
register_mechanism("features", "gae_cross_segment",
                   lambda s: s.advantage.gae_variant == "cross_segment_per_sample")
register_mechanism("features", "gae_cross_segment_whole_rollout",
                   lambda s: s.advantage.gae_variant == "cross_segment_whole_rollout")
register_mechanism("features", "critic_multi_update",
                   lambda s: s.execution.needs_critic
                   and s.critic.critic_updates_per_step not in (None, 1))
# seq_adv's gamma rule refuses gamma != 1 outside REINFORCE++; GAE reads it.
_seq_adv_gamma = _alg._REJECTIONS["seq_adv_gamma"]


def _gamma_rule(spec: AlgorithmSpec) -> str | None:
    if spec.execution.needs_critic and spec.advantage.estimator in _alg.CRITIC_ESTIMATORS:
        return None
    return _seq_adv_gamma(spec)


_alg._REJECTIONS["seq_adv_gamma"] = _gamma_rule

# --------------------------------------------------------------------------
# run-level rejections (design D2)
# --------------------------------------------------------------------------

_SHARED = "Miles shared actor/critic PPO"


def _flag_values(argv: Sequence[str], flag: str) -> list[str]:
    tokens = list(argv)
    out = []
    for i, token in enumerate(tokens):
        if token == flag:
            out.append(tokens[i + 1] if i + 1 < len(tokens) else "")
        elif token.startswith(flag + "="):
            out.append(token.split("=", 1)[1])
    return out


def critic_run_problems(spec: AlgorithmSpec, values: Mapping[str, Any]) -> list[str]:
    """Run-configuration combinations a critic cannot run with.

    ``values`` (every key optional; a missing key is not checked): ``elastic``
    (bool), ``sync_preset`` (str), ``extra_argv`` (sequence),
    ``actor_num_nodes`` / ``actor_num_gpus_per_node`` (int).
    """

    if not spec.execution.needs_critic:
        return []
    problems = []
    argv = tuple(values.get("extra_argv") or ())
    if values.get("elastic"):
        problems.append(
            f"elastic reconfiguration with a critic: {_SHARED} does not support --indep-dp "
            "(Miles arguments.py:3593), which elastic reconfiguration and train fault "
            "tolerance need (user decision 4, 2026-10-06)"
        )
    if any(t == "--indep-dp" or t.startswith("--indep-dp=") for t in argv):
        problems.append(
            f"--indep-dp with a critic: {_SHARED} hands the critic outputs to a single trainer "
            "cell; it does not support independent DP (Miles arguments.py:3593)"
        )
    if "trainer" in _flag_values(argv, "--deploy-component"):
        problems.append(
            "--deploy-component trainer with a critic: it deploys exactly one trainer, and a "
            "critic adds a second one (Miles arguments.py:3058)"
        )
    for flag, key in (("--critic-num-nodes", "actor_num_nodes"),
                      ("--critic-num-gpus-per-node", "actor_num_gpus_per_node")):
        actor = values.get(key)
        for raw in _flag_values(argv, flag):
            if actor is None or raw != str(actor):
                problems.append(
                    f"{flag} {raw} differs from the actor's {key} ({actor}): {_SHARED} trains "
                    "the critic on the actor's GPUs (Miles arguments.py:3605-3606 overwrites "
                    "the critic count silently); drop the flag"
                )
    if values.get("sync_preset") == "decoupled":
        problems.append(
            "decoupled outer sync with a critic is not supported (design D4: actor and critic "
            "are averaged together by strict-avg only; decoupled is a later exploration)"
        )
    return problems


register_launch_check("critic_shared_ppo", critic_run_problems)
register_island_check("critic_shared_ppo", critic_run_problems)


def critic_lr_warmup_problems(spec: AlgorithmSpec, values: Mapping[str, Any]) -> list[str]:
    """``critic.critic_lr_warmup`` must stay below the critic's LR decay steps.

    Miles model.py (fork 6e7365b60:88-103) sizes the critic scheduler as
    ``train_iters = num_rollout * rollout_batch_size * n_samples * num_critic_epochs
    // global_batch_size``, ``lr_decay_iters`` defaulting to ``train_iters`` (an
    explicit ``--lr-decay-iters`` is shared with the actor); Megatron then asserts
    ``lr_warmup_steps < lr_decay_steps`` at critic init (S13 G1 SAO: warmup 10,
    decay 3). Checked only when every input is known and not overridden by
    extra argv.
    """

    warmup = spec.critic.critic_lr_warmup if spec.execution.needs_critic else None
    if not warmup:
        return []
    argv = tuple(values.get("extra_argv", ()))
    overridden = ("--lr-decay-iters", "--lr-warmup-fraction", "--num-rollout", "--global-batch-size",
                  "--n-samples-per-prompt", "--rollout-batch-size", "--critic-lr-warmup-iters")
    if any(_flag_values(argv, f) for f in overridden):
        return []
    try:
        rounds, groups = int(values["num_rollout"]), int(values["rollout_batch_size"])
        samples, gbs = int(values["n_samples_per_prompt"]), int(values["global_batch_size"])
    except (KeyError, TypeError, ValueError):
        return []
    if rounds <= 0 or gbs <= 0:
        return []
    decay = values.get("lr_decay_iters")
    if decay is None:
        epochs = spec.critic.critic_updates_per_step or 1
        decay = rounds * groups * samples * epochs // gbs
    if warmup >= decay:
        return [f"critic.critic_lr_warmup={warmup} must be < the critic's LR decay iters ({decay}); "
                "Megatron asserts lr_warmup_steps < lr_decay_steps at critic init. Lower the "
                "warmup or run more rounds"]
    return []


register_launch_check("critic_lr_warmup", critic_lr_warmup_problems)


def critic_lr_horizon_problems(spec: AlgorithmSpec, values: Mapping[str, Any]) -> list[str]:
    """A decaying LR schedule must cover every critic optimizer step.

    yeto's linear/cosine schedule passes an explicit ``--lr-decay-iters`` =
    actor steps, and Miles shares it with the critic (fork model.py only fills
    ``lr_decay_iters`` when it is None).  With ``critic_updates_per_step`` = N > 1
    the critic takes N x actor steps, so it trains at LR 0 for the last
    (N-1)/N of the run (S14 SAO G1 s14-forkg1-sao-20261007a: decay 12, warmup
    10, 24 critic steps -> critic LR 0 from step 12; values stuck near 0.05,
    EV <= 0 every round).  A constant schedule never decays."""

    if not spec.execution.needs_critic:
        return []
    style, decay = values.get("lr_decay_style"), values.get("lr_decay_iters")
    if style in (None, "constant") or not decay:
        return []
    argv = tuple(values.get("extra_argv", ()))
    overridden = ("--lr-decay-iters", "--lr-decay-style", "--num-rollout", "--global-batch-size",
                  "--n-samples-per-prompt", "--rollout-batch-size", "--num-critic-epochs",
                  "--critic-updates-per-step")
    if any(_flag_values(argv, f) for f in overridden):
        return []
    try:
        rounds, groups = int(values["num_rollout"]), int(values["rollout_batch_size"])
        samples, gbs = int(values["n_samples_per_prompt"]), int(values["global_batch_size"])
    except (KeyError, TypeError, ValueError):
        return []
    if rounds <= 0 or gbs <= 0:
        return []
    epochs = spec.critic.critic_updates_per_step or 1
    critic_steps = rounds * groups * samples * epochs // gbs
    if critic_steps > int(decay):
        return [f"lr schedule {style!r} decays over {decay} iters, shared by the critic, but the "
                f"critic takes {critic_steps} optimizer steps (critic_updates_per_step={epochs}); "
                f"it would train at LR 0 for {critic_steps - int(decay)} steps. "
                "Use --rl-lr-schedule constant"]
    return []


register_launch_check("critic_lr_horizon", critic_lr_horizon_problems)
