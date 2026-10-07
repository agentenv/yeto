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
    register_island_check,
    register_launch_check,
)
from yeto.rl.engine.miles_adapter import algorithm_flags as _af
from yeto.rl.engine.miles_adapter.algorithm_flags import (
    FlagMapping,
    _float,
    _int,
    _num,
    _str,
    register_flag,
)

# --------------------------------------------------------------------------
# translation (design D2)
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
    argv += ["--num-critic-only-steps", "0"]
    if c.init == "load":
        argv += ["--critic-load", c.load]
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

# --gamma: seq_adv's row emits it for REINFORCE++; critic_argv owns it under a critic.
_gamma_row = _af.MAPPINGS["--gamma"]
_af.MAPPINGS["--gamma"] = FlagMapping(
    _gamma_row.flag, _gamma_row.field, _gamma_row.switch, _gamma_row.parse, _gamma_row.absorb,
    lambda spec: [] if spec.execution.needs_critic else _gamma_row.translate(spec),
    _gamma_row.emitted_by_config,
)

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
