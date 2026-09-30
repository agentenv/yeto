"""Token-level policy-loss variants (change ``rl-algo-loss-variants``, route B).

Registration module listed in :data:`yeto.rl.algos.EXTENSION_MODULES`. The
computation lives in the Miles fork (``michaellchung/miles`` ``yeto/ports``,
``--policy-loss-variant``, design D1 route B); yeto only describes, translates
and validates. Through the P0 registry (never by editing the shared files) it
adds:

* spec fields (design D2) in the ``loss`` group
  - ``loss.policy_loss_variant`` in ``policy_loss`` (default, not emitted:
    Miles' PPO clip objective) / ``cispo`` / ``sapo`` / ``gmpo``. P0's
    ``loss.variant`` is already ``--loss-type`` (``policy_loss`` /
    ``custom_loss``); the variants run inside ``--loss-type policy_loss`` and
    only replace its pg_loss, so they are a separate field;
  - ``loss.sapo_tau_pos`` (1.0), ``loss.sapo_tau_neg`` (1.05),
    ``loss.gmpo_log_clip_low`` / ``loss.gmpo_log_clip_high`` (0.4). They enter
    the canonical form (and the hash) exactly when their variant is selected
    -- then always, even at the default value; with another variant a
    non-default value is rejected (one semantics, one hash). CISPO reuses
    ``loss.eps_clip`` / ``loss.eps_clip_high`` as its eps_l / eps_h;
* mechanisms ``losses:cispo`` / ``losses:sapo`` / ``losses:gmpo``: declared
  by the fake engine only. The Miles adapter does not declare them (GPU
  validation paused by the user 2026-09-30, alignment §7b): expressible, not
  opened;
* flag rows (absorption + conflict detection) for the fork flags;
* rejections (design D4), launch checks (fork pin, GMPO context parallel) and
  the GMPO gradient rule (design D5).

Import-light: no torch/miles at import time.
"""

from __future__ import annotations

import math
from typing import Any

from yeto.rl.engine.algorithm import (
    SEQUENCE_RATIO_ESTIMATORS,
    AlgorithmSpecError,
    register_field,
    register_gradient_rule,
    register_launch_check,
    register_mechanism,
    register_rejection,
)
from yeto.rl.engine.miles_adapter.algorithm_flags import (
    FlagMapping,
    _float,
    _num,
    register_flag,
)

DEFAULT_VARIANT = "policy_loss"
VARIANTS = ("cispo", "sapo", "gmpo")
POLICY_LOSS_VARIANTS = (DEFAULT_VARIANT, *VARIANTS)

# variant -> its own parameters (field name, default)
VARIANT_PARAMS: dict[str, tuple[tuple[str, float], ...]] = {
    "cispo": (),  # eps_l / eps_h are loss.eps_clip / loss.eps_clip_high
    "sapo": (("sapo_tau_pos", 1.0), ("sapo_tau_neg", 1.05)),
    "gmpo": (("gmpo_log_clip_low", 0.4), ("gmpo_log_clip_high", 0.4)),
}

# Fork CLI flags (FORK-2b, michaellchung/miles yeto-loss-variants); names
# checked against the fork diff before the pin moves.
VARIANT_FLAG = "--policy-loss-variant"
PARAM_FLAGS = {
    "sapo_tau_pos": "--sapo-tau-pos",
    "sapo_tau_neg": "--sapo-tau-neg",
    "gmpo_log_clip_low": "--gmpo-log-clip-low",
    "gmpo_log_clip_high": "--gmpo-log-clip-high",
}
FORK_FLAGS = frozenset({VARIANT_FLAG, *PARAM_FLAGS.values()})

# Miles commits (full SHA) whose parser/loss implement --policy-loss-variant.
# Empty until the fork branch is reviewed, fast-forwarded into yeto/ports and
# the pin/image are rebuilt by Agent IMG (design Migration Plan, tasks 4.3).
FORK_COMMITS: frozenset[str] = frozenset()

MECHANISMS = {name: ("losses", name) for name in VARIANTS}

# D5: GMPO fully clipped when the clip fraction reaches 1 (float slack).
FULL_CLIP_THRESHOLD = 1.0 - 1e-9


def pinned_miles_commit() -> str:
    from yeto.rl import MILES_NEXT_COMMIT

    return MILES_NEXT_COMMIT


def fork_supports_variants(miles_commit: str | None = None) -> bool:
    return (miles_commit or pinned_miles_commit()) in FORK_COMMITS


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


def _parse_variant(path: str, value: Any) -> str:
    if value not in POLICY_LOSS_VARIANTS:
        raise AlgorithmSpecError(
            f"{path} must be one of {list(POLICY_LOSS_VARIANTS)}, got {value!r}"
        )
    return value


def _positive_finite(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlgorithmSpecError(f"{path} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise AlgorithmSpecError(f"{path} must be a positive finite number, got {value!r}")
    return value


def _selected(group) -> str:
    return group.policy_loss_variant


register_field("loss", "policy_loss_variant", default=DEFAULT_VARIANT, parse=_parse_variant)
for _variant, _params in VARIANT_PARAMS.items():
    for _name, _default in _params:
        register_field(
            "loss", _name, default=_default, parse=_positive_finite,
            always_emit=lambda group, v=_variant: _selected(group) == v,
        )


def variant(spec) -> str:
    return spec.loss.policy_loss_variant


def variant_params(spec) -> dict[str, float]:
    """The selected variant's own parameters (CISPO: its clip bounds)."""

    name = variant(spec)
    if name == "cispo":
        return {"eps_clip": spec.loss.eps_clip, "eps_clip_high": spec.loss.eps_clip_high}
    return {p: getattr(spec.loss, p) for p, _ in VARIANT_PARAMS.get(name, ())}


# --------------------------------------------------------------------------
# mechanisms and flags
# --------------------------------------------------------------------------

for _name in VARIANTS:
    register_mechanism("losses", _name, lambda s, v=_name: variant(s) == v)


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
# rejections (design D4)
# --------------------------------------------------------------------------


def _reject_param_mismatch(s) -> str | None:
    name = variant(s)
    stray = sorted(
        f"loss.{p}" for v, params in VARIANT_PARAMS.items() if v != name
        for p, default in params if getattr(s.loss, p) != default
    )
    if stray:
        return (
            f"{stray} only apply to their own loss.policy_loss_variant, got {name!r}; "
            "drop them or select that variant"
        )
    return None


def _reject_custom_loss(s) -> str | None:
    if variant(s) != DEFAULT_VARIANT and s.loss.variant != "policy_loss":
        return (
            f"loss.policy_loss_variant={variant(s)!r} replaces pg_loss inside Miles' "
            f"policy_loss; loss.variant={s.loss.variant!r} replaces the whole loss function. "
            "Use loss.variant='policy_loss' with the variant, or the custom loss alone"
        )
    return None


def _reject_sequence_ratio(s) -> str | None:
    if variant(s) != DEFAULT_VARIANT and s.advantage.estimator in SEQUENCE_RATIO_ESTIMATORS:
        return (
            f"loss.policy_loss_variant={variant(s)!r} with advantage.estimator="
            f"{s.advantage.estimator!r}: both define the policy ratio (GSPO a sequence-level "
            "ratio). Use advantage.estimator='grpo' with the variant, or GSPO with "
            "loss.policy_loss_variant='policy_loss'"
        )
    return None


def _reject_dual_clip(s) -> str | None:
    if variant(s) != DEFAULT_VARIANT and s.loss.eps_clip_c is not None:
        return (
            f"loss.policy_loss_variant={variant(s)!r} with loss.eps_clip_c (dual-clip): "
            "dual-clip is defined for the PPO clip objective only. Drop loss.eps_clip_c, or "
            "use loss.policy_loss_variant='policy_loss'"
        )
    return None


def _reject_unused_clip(s) -> str | None:
    name = variant(s)
    if name not in ("sapo", "gmpo"):
        return None
    unused = [f for f in ("eps_clip", "eps_clip_high") if getattr(s.loss, f) is not None]
    if unused:
        knobs = ("loss.sapo_tau_pos/loss.sapo_tau_neg" if name == "sapo"
                 else "loss.gmpo_log_clip_low/loss.gmpo_log_clip_high")
        return (
            f"{[f'loss.{f}' for f in unused]} have no effect with "
            f"loss.policy_loss_variant={name!r} (it uses {knobs}); drop them"
        )
    return None


def _reject_cispo_implicit_clip(s) -> str | None:
    # review 2026-09-30 finding 4: CISPO's IS-weight clip range must be in the
    # identity; the engine default (Miles --eps-clip 0.2, --eps-clip-high =
    # --eps-clip when unset) would silently set it outside the hash.
    if variant(s) == "cispo" and (s.loss.eps_clip is None or s.loss.eps_clip_high is None):
        return (
            "loss.policy_loss_variant='cispo' clips the IS weight to [1-eps_l, 1+eps_h]: give "
            "loss.eps_clip (eps_l) and loss.eps_clip_high (eps_h) explicitly so both enter the "
            "algorithm hash and the argv (yeto sets no default; the engine default is "
            "--eps-clip 0.2)"
        )
    return None


register_rejection("loss_variant_params", _reject_param_mismatch)
register_rejection("loss_variant_cispo_clip", _reject_cispo_implicit_clip)
register_rejection("loss_variant_custom_loss", _reject_custom_loss)
register_rejection("loss_variant_sequence_ratio", _reject_sequence_ratio)
register_rejection("loss_variant_dual_clip", _reject_dual_clip)
register_rejection("loss_variant_unused_clip", _reject_unused_clip)


# --------------------------------------------------------------------------
# launch checks (Miles adapter / launcher, before any GPU process)
# --------------------------------------------------------------------------


def launch_problems(spec, values) -> list[str]:
    name = variant(spec)
    if name == DEFAULT_VARIANT:
        return []
    problems = []
    commit = pinned_miles_commit()
    if not fork_supports_variants(commit):
        problems.append(
            f"loss.policy_loss_variant={name!r} needs a Miles fork commit implementing "
            f"{VARIANT_FLAG}; the pinned Miles {commit[:12]} does not (known commits: "
            f"{sorted(FORK_COMMITS) or 'none yet'}). Expressible but not opened"
        )
    # Callers today pass context_parallel_size=1 hard-coded (miles_adapter/config.py
    # and launcher.py do not expose CP yet), so this check is a guard for when
    # CP becomes configurable, not a currently reachable refusal.
    cp = int(values.get("context_parallel_size", 1) or 1)
    if name == "gmpo" and cp != 1:
        problems.append(
            f"GMPO with context parallel size {cp}: the sequence geometric mean gathers "
            "every sequence across CP ranks; only CPU-tested, not GPU-verified. Use "
            "context parallel size 1"
        )
    return problems


register_launch_check("loss_variants", launch_problems)


# --------------------------------------------------------------------------
# gradient rule (design D5)
# --------------------------------------------------------------------------


def gmpo_gradient_rule(spec, batch_summary, step_metrics=None):
    """GMPO: every token with A != 0 clipped in log space -> no gradient.

    Reads ``step_metrics.clip_fraction`` (the round's Miles ``pg_clipfrac``),
    NOT ``masked_fraction``: with corrections on, ``masked_fraction`` carries
    the correction mask (review 2026-09-30, finding 3). Semantics (design D5,
    fork-side the same): among valid tokens with A != 0, the fraction whose
    one-sided log-space clip binds. Clipped tokens carry no gradient and
    A = 0 tokens carry none either, so fraction 1 means a zero gradient is
    legitimate. Unknown / non-finite / < 1 abstains (GRPO rule). The Miles
    trainer collects the fraction for GMPO with
    ``infra-drafts/patches/algo-2b-trainer.patch``; without it the value stays
    None (stricter). CISPO and SAPO register no rule: they never mask a token.
    """

    if variant(spec) != "gmpo":
        return None
    fraction = getattr(step_metrics, "clip_fraction", None)
    if fraction is None or isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        return None
    fraction = float(fraction)
    if math.isfinite(fraction) and fraction >= FULL_CLIP_THRESHOLD:
        return False
    return None


register_gradient_rule("loss_variant_gmpo_full_clip", gmpo_gradient_rule,
                       mechanism="losses:gmpo")
