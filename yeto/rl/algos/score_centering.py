"""Score centering (Miles ``--loss-type score_centering``; Kimi-K3.1 recipe).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. Score
centering (arXiv 2609.20807, upstream Miles ``docs/examples/infra-features/
score-centering.md``) is a REINFORCE-style loss that subtracts the expected
weighted score under the sampler's distribution, approximated from the
sampler's top-``K`` candidates. It replaces the PPO clip objective, so it is
a ``loss.variant`` (``--loss-type``), not a ``loss.policy_loss_variant``.

The module adds, through the registry only:

* the loss variant ``score_centering`` (appended to ``LOSS_VARIANTS``) and its
  mechanism ``losses:score_centering`` -- not declared by the Miles adapter
  until a GPU run shows it takes effect (``--rl-allow-unverified-mechanism``
  admits it on one island);
* spec fields in the ``loss`` group, in the hash exactly when the variant is
  selected: ``score_centering_is`` (``none`` | ``tis`` | ``mis``),
  ``score_centering_tis_clip`` (2.0), ``score_centering_mis_low`` (0.5),
  ``score_centering_mis_high`` (5.0) and ``rollout_top_logprobs_num`` (128);
* rejections: the variant requires GRPO advantages without std normalization,
  per-token aggregation and rollout log-probs as the old policy (the recipe
  of INTERFACES 2.5); it excludes every other off-policy correction
  (``--use-tis`` / OPSM / MIS ratio masks -- the centered weights replace
  them), PPO clip knobs and policy-loss variants; its parameters are refused
  under another variant.

The Miles flag rows live in :mod:`yeto.rl.adapters.miles.algo_flag_rows`.
:func:`spec_from_rl_loss_args` builds the spec from launcher options
(``--rl-loss score_centering --rl-score-centering-is ...``); the default
(``policy_loss``) returns the base spec untouched, so the GRPO argv is
byte-identical.
"""

from __future__ import annotations

import math
from typing import Any

from yeto.rl.engine.algorithm import (
    LOSS_VARIANTS,
    AlgorithmSpec,
    AlgorithmSpecError,
    register_field,
    register_mechanism,
    register_rejection,
)

SCORE_CENTERING = "score_centering"
DEFAULT_LOSS = "policy_loss"
RL_LOSS_CHOICES = (DEFAULT_LOSS, SCORE_CENTERING)
IS_MODES = ("none", "tis", "mis")
DEFAULT_IS = "none"
DEFAULT_TIS_CLIP = 2.0
DEFAULT_MIS_LOW = 0.5
DEFAULT_MIS_HIGH = 5.0
DEFAULT_TOP_LOGPROBS = 128
MECHANISM = ("losses", SCORE_CENTERING)

# Upstream flag names (miles/utils/args/configs/algo.py, rollout.py at origin/main).
FLAGS = {
    "score_centering_is": "--score-centering-is",
    "score_centering_tis_clip": "--score-centering-tis-clip",
    "score_centering_mis_low": "--score-centering-mis-low",
    "score_centering_mis_high": "--score-centering-mis-high",
    "rollout_top_logprobs_num": "--rollout-top-logprobs-num",
}
# Sampling the score-centering recipe pins on the trainer side (INTERFACES 2.5).
SAMPLING_ARGV = ("--rollout-temperature", "1.0", "--rollout-top-p", "1.0", "--rollout-top-k", "-1")

if SCORE_CENTERING not in LOSS_VARIANTS:
    LOSS_VARIANTS.append(SCORE_CENTERING)


def selected(spec: AlgorithmSpec) -> bool:
    return spec.loss.variant == SCORE_CENTERING


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


def _parse_is(path: str, value: Any) -> str:
    if value not in IS_MODES:
        raise AlgorithmSpecError(f"{path} must be one of {list(IS_MODES)}, got {value!r}")
    return value


def _positive_finite(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise AlgorithmSpecError(f"{path} must be a positive finite number, got {value!r}")
    return value


def _non_negative_finite(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise AlgorithmSpecError(f"{path} must be a non-negative finite number, got {value!r}")
    return value


def _positive_int(path: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AlgorithmSpecError(f"{path} must be a positive integer, got {value!r}")
    return value


def _when_selected(group) -> bool:
    return group.variant == SCORE_CENTERING


register_field("loss", "score_centering_is", default=DEFAULT_IS, parse=_parse_is,
               always_emit=_when_selected)
register_field("loss", "score_centering_tis_clip", default=DEFAULT_TIS_CLIP, parse=_positive_finite,
               always_emit=_when_selected)
register_field("loss", "score_centering_mis_low", default=DEFAULT_MIS_LOW, parse=_non_negative_finite,
               always_emit=_when_selected)
register_field("loss", "score_centering_mis_high", default=DEFAULT_MIS_HIGH, parse=_positive_finite,
               always_emit=_when_selected)
register_field("loss", "rollout_top_logprobs_num", default=DEFAULT_TOP_LOGPROBS, parse=_positive_int,
               always_emit=_when_selected)

PARAM_DEFAULTS = {
    "score_centering_is": DEFAULT_IS,
    "score_centering_tis_clip": DEFAULT_TIS_CLIP,
    "score_centering_mis_low": DEFAULT_MIS_LOW,
    "score_centering_mis_high": DEFAULT_MIS_HIGH,
    "rollout_top_logprobs_num": DEFAULT_TOP_LOGPROBS,
}


def params(spec: AlgorithmSpec) -> dict[str, Any]:
    return {name: getattr(spec.loss, name) for name in PARAM_DEFAULTS}


# --------------------------------------------------------------------------
# mechanism
# --------------------------------------------------------------------------

register_mechanism(*MECHANISM, selected)


# --------------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------------


def _reject_requirements(s: AlgorithmSpec) -> str | None:
    if not selected(s):
        return None
    problems = []
    if s.advantage.estimator != "grpo":
        problems.append(f"advantage.estimator={s.advantage.estimator!r} (needs 'grpo')")
    if s.advantage.std_normalization:
        problems.append("advantage.std_normalization=True (needs False: group-centered rewards, "
                        "--disable-grpo-std-normalization)")
    if s.loss.aggregation != "token":
        problems.append(f"loss.aggregation={s.loss.aggregation!r} (needs 'token', --calculate-per-token-loss)")
    if not s.correction.use_rollout_logprobs:
        problems.append("correction.use_rollout_logprobs=False (needs True: the sampler's log-probs are "
                        "the behaviour policy, --use-rollout-logprobs)")
    if problems:
        return ("loss.variant='score_centering' requires the upstream recipe; got "
                + "; ".join(problems))
    return None


def _reject_other_corrections(s: AlgorithmSpec) -> str | None:
    if not selected(s):
        return None
    c = s.correction
    given = []
    if c.method != "none":
        given.append(f"correction.method={c.method!r}")
    if c.opsm_delta is not None:
        given.append("correction.opsm_delta")
    if c.ratio_mask != "none":
        given.append(f"correction.ratio_mask={c.ratio_mask!r}")
    if c.mismatch_metrics:
        given.append("correction.mismatch_metrics")
    if given:
        return (f"{given} with loss.variant='score_centering': the centered importance weights replace "
                "TIS / OPSM / MIS (--use-tis, --use-opsm, --custom-tis-function-path); select them with "
                "loss.score_centering_is='tis'|'mis' instead")
    return None


def _reject_ppo_knobs(s: AlgorithmSpec) -> str | None:
    if not selected(s):
        return None
    stray = [f"loss.{name}" for name in ("eps_clip", "eps_clip_high", "eps_clip_c")
             if getattr(s.loss, name) is not None]
    if s.loss.reducer is not None:
        stray.append("loss.reducer")
    variant = getattr(s.loss, "policy_loss_variant", DEFAULT_LOSS)
    if variant != DEFAULT_LOSS:
        stray.append(f"loss.policy_loss_variant={variant!r}")
    if stray:
        return (f"{stray} have no effect with loss.variant='score_centering' (a REINFORCE loss without "
                "PPO clipping or pg_loss variants); drop them")
    return None


def _reject_params_without_variant(s: AlgorithmSpec) -> str | None:
    if selected(s):
        return None
    stray = sorted(f"loss.{name}" for name, default in PARAM_DEFAULTS.items()
                   if getattr(s.loss, name) != default)
    if stray:
        return f"{stray} only apply to loss.variant='score_centering', got {s.loss.variant!r}"
    return None


def _reject_is_params(s: AlgorithmSpec) -> str | None:
    if not selected(s):
        return None
    mode = s.loss.score_centering_is
    stray = []
    if mode != "tis" and s.loss.score_centering_tis_clip != DEFAULT_TIS_CLIP:
        stray.append("loss.score_centering_tis_clip")
    if mode != "mis" and (s.loss.score_centering_mis_low != DEFAULT_MIS_LOW
                          or s.loss.score_centering_mis_high != DEFAULT_MIS_HIGH):
        stray.append("loss.score_centering_mis_low/high")
    if stray:
        return f"{stray} need loss.score_centering_is={'tis' if 'tis' in stray[0] else 'mis'!r}, got {mode!r}"
    if mode == "mis" and not s.loss.score_centering_mis_low < s.loss.score_centering_mis_high:
        return (f"loss.score_centering_mis_low ({s.loss.score_centering_mis_low}) must be below "
                f"loss.score_centering_mis_high ({s.loss.score_centering_mis_high})")
    return None


register_rejection("score_centering_requirements", _reject_requirements)
register_rejection("score_centering_other_corrections", _reject_other_corrections)
register_rejection("score_centering_ppo_knobs", _reject_ppo_knobs)
register_rejection("score_centering_params_without_variant", _reject_params_without_variant)
register_rejection("score_centering_is_params", _reject_is_params)


# --------------------------------------------------------------------------
# launcher helper (--rl-loss)
# --------------------------------------------------------------------------


def score_centering_spec(
    *,
    base: AlgorithmSpec | None = None,
    importance_sampling: str = DEFAULT_IS,
    tis_clip: float | None = None,
    mis_low: float | None = None,
    mis_high: float | None = None,
    top_logprobs: int | None = None,
) -> AlgorithmSpec:
    """``base`` (default GRPO) turned into the score-centering recipe."""

    payload = (base or AlgorithmSpec()).structured_dict()
    loss = dict(payload.get("loss") or {})
    loss.update({
        "variant": SCORE_CENTERING,
        "aggregation": "token",
        "score_centering_is": importance_sampling,
    })
    if tis_clip is not None:
        loss["score_centering_tis_clip"] = tis_clip
    if mis_low is not None:
        loss["score_centering_mis_low"] = mis_low
    if mis_high is not None:
        loss["score_centering_mis_high"] = mis_high
    if top_logprobs is not None:
        loss["rollout_top_logprobs_num"] = top_logprobs
    payload["loss"] = loss
    advantage = dict(payload.get("advantage") or {})
    advantage["std_normalization"] = False
    payload["advantage"] = advantage
    correction = dict(payload.get("correction") or {})
    correction["use_rollout_logprobs"] = True
    payload["correction"] = correction
    return AlgorithmSpec.from_dict(payload)


def spec_from_rl_loss_args(args: Any, base: AlgorithmSpec | None = None) -> AlgorithmSpec:
    """Apply ``--rl-loss`` and ``--rl-score-centering-*`` launcher options to ``base``.

    ``--rl-loss`` absent or ``policy_loss`` returns ``base`` unchanged.
    """

    loss = getattr(args, "rl_loss", None) or DEFAULT_LOSS
    if loss not in RL_LOSS_CHOICES:
        raise AlgorithmSpecError(f"--rl-loss must be one of {list(RL_LOSS_CHOICES)}, got {loss!r}")
    spec = base if base is not None else AlgorithmSpec()
    if loss == DEFAULT_LOSS:
        for name in ("rl_score_centering_is", "rl_score_centering_tis_clip",
                     "rl_score_centering_mis_low", "rl_score_centering_mis_high"):
            if getattr(args, name, None) is not None:
                raise AlgorithmSpecError(f"--{name.replace('_', '-')} needs --rl-loss score_centering")
        return spec
    return score_centering_spec(
        base=spec,
        importance_sampling=getattr(args, "rl_score_centering_is", None) or DEFAULT_IS,
        tis_clip=getattr(args, "rl_score_centering_tis_clip", None),
        mis_low=getattr(args, "rl_score_centering_mis_low", None),
        mis_high=getattr(args, "rl_score_centering_mis_high", None),
        top_logprobs=getattr(args, "rl_rollout_top_logprobs_num", None),
    )
