"""verl mapping of the neutral correction group
(change ``rl-offpolicy-correction-passthrough`` L6.4, L6.6, L6.7, L6.11).

:func:`correction_mapping` turns ``AlgorithmSpec.correction`` into Hydra
overrides under ``algorithm.rollout_correction`` (verl fork ``acad9875``,
``verl/trainer/ppo/rollout_corr_helper.py``). Mapping table (design D3):

==========================  =====================================================
neutral                     verl
==========================  =====================================================
tis, token / sequence       ``rollout_is=token|sequence``, ``rollout_is_threshold=upper``
ratio_mask=icepop           ``rollout_is_threshold='lower_upper'`` (weights outside
                            the range set to 0; ``loss_mask`` unchanged)
ratio_mask=mis              ``rollout_rs=token_k1|seq_mean_k1``,
                            ``rollout_rs_threshold='lower_upper'`` (rejected tokens
                            leave ``response_mask``: the loss denominator shrinks)
batch_normalize             ``rollout_is_batch_normalize=True``
==========================  =====================================================

Rows verl cannot express give a :class:`CorrectionWarning` (phase 1 runs on):
``lower > 0`` on plain TIS (verl clamps the upper side only), level
``sequence`` with the MIS mask (verl has the geometric mean only), ``veto``,
``opsm``, a custom function, and the MIS mask (denominator).

Decoupled PPO (design D2): ``bypass_mode=False`` makes verl recompute
``old_log_probs`` as pi_prox; the fully_async trainer computes it on the
weights of the first local step of the current sync cycle
(``fully_async_trainer.py`` ``_compute_old_log_prob``), hence
``prox_anchor=first_local_step``. ``fully_async_ppo_trainer.yaml`` sets
``bypass_mode: True``, so every mapping writes ``bypass_mode`` explicitly and
:data:`FORBIDDEN` refuses ``bypass_mode=True``.

``actor.use_rollout_log_probs`` (yaml: True) is not forbidden: no Python code
in the fork reads it (``grep`` over ``verl/`` at ``acad9875``: only the config
dataclasses declare it), so it does not change pi_prox (L6.6).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.offpolicy_correction import (
    PROX_FIRST_LOCAL_STEP,
    CorrectionMapping,
    CorrectionWarning,
    ForbiddenSetting,
    refuse_forbidden,
)

BACKEND = "verl"
PROX_ANCHOR = PROX_FIRST_LOCAL_STEP
PREFIX = "algorithm.rollout_correction."
BYPASS_KEY = PREFIX + "bypass_mode"

# verl's own default (rollout_correction.yaml) when the spec gives no upper bound.
DEFAULT_UPPER = 2.0
# verl IcePop needs lower > 0 (_parse_rollout_is_threshold); a missing or zero
# lower runs with this bound, which keeps every positive weight in practice.
ICEPOP_MIN_LOWER = 1e-8

FORBIDDEN = (
    ForbiddenSetting(BACKEND, f"{BYPASS_KEY}=True",
                     "it sets old_log_probs = rollout_log_probs, so pi_prox is lost"),
)


def _num(value: float) -> str:
    return repr(float(value))


def _range(lower: float, upper: float) -> str:
    return f"{_num(lower)}_{_num(upper)}"


def _override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _num(value)
    if isinstance(value, str) and "_" in value:
        return f"'{value}'"  # "lower_upper" stays a string for Hydra
    return str(value)


def _mapped(spec: AlgorithmSpec) -> tuple[dict[str, Any], list[CorrectionWarning]]:
    c = spec.correction
    values: dict[str, Any] = {"bypass_mode": bool(c.use_rollout_logprobs),
                              "rollout_is": None, "rollout_rs": None}
    warnings: list[CorrectionWarning] = []

    def warn(field: str, reason: str, ran_with: Any = None) -> None:
        warnings.append(CorrectionWarning(field, BACKEND, reason, ran_with))

    if c.method == "opsm":
        warn("method", "verl has no OPSM; the run has no correction", "none")
    if c.method == "custom":
        warn("function", "verl takes no custom correction function; the run has no correction",
             "none")
    if c.method != "tis":
        return values, warnings

    upper = DEFAULT_UPPER if c.upper is None else float(c.upper)
    lower = 0.0 if c.lower is None else float(c.lower)
    if c.veto is not None:
        warn("veto", "verl has no per-sequence veto", None)

    if c.ratio_mask == "mis":
        if lower <= 0.0:
            warn("lower", "verl rejection sampling needs lower > 0; the k1 rule uses 1/upper",
                 1.0 / upper)
            lower = 1.0 / upper
        values["rollout_rs"] = "token_k1" if c.level == "token" else "seq_mean_k1"
        values["rollout_rs_threshold"] = _range(lower, upper)
        if c.level == "sequence":
            warn("level", "verl rejection sampling has no product-of-ratios rule; the "
                 "geometric mean (seq_mean_k1) runs", "geometric")
        warn("ratio_mask", "verl removes rejected tokens from response_mask, so the loss "
             "denominator shrinks; the neutral meaning is 'denominator unchanged'",
             "denominator reduced")
        if c.batch_normalize:
            warn("batch_normalize", "verl normalizes IS weights only; the MIS mask maps to "
                 "rejection sampling without weights", False)
        return values, warnings

    values["rollout_is"] = c.level  # token | sequence (geometric needs the MIS mask)
    if c.ratio_mask == "icepop":
        if lower <= 0.0:
            warn("lower", "verl IcePop needs lower > 0", ICEPOP_MIN_LOWER)
            lower = ICEPOP_MIN_LOWER
        values["rollout_is_threshold"] = _range(lower, upper)
    else:
        values["rollout_is_threshold"] = upper
        if lower > 0.0:
            warn("lower", "verl TIS clamps the upper side only (weights below lower pass "
                 "through)", 0.0)
    values["rollout_is_batch_normalize"] = bool(c.batch_normalize)
    return values, warnings


def correction_mapping(spec: AlgorithmSpec) -> CorrectionMapping:
    """The verl settings of ``spec.correction`` with warnings and ``prox_anchor``."""

    values, warnings = _mapped(spec)
    keyed = {PREFIX + k: v for k, v in values.items()}
    overrides = tuple(f"{k}={_override_value(v)}" for k, v in keyed.items())
    return CorrectionMapping(backend=BACKEND, prox_anchor=PROX_ANCHOR, values=keyed,
                             warnings=tuple(warnings), overrides=overrides)


def _normalized(override: str) -> str:
    key, sep, value = override.lstrip("+~").partition("=")
    if sep and value.strip().strip("'\"").lower() in ("true", "false"):
        value = value.strip().strip("'\"").capitalize()
    return f"{key}{sep}{value}"


def refuse_forbidden_overrides(overrides: Iterable[str]) -> None:
    """Refuse Hydra overrides that hold a :data:`FORBIDDEN` assignment (names it)."""

    refuse_forbidden((_normalized(o) for o in overrides), FORBIDDEN)


def apply_to_overrides(overrides: Iterable[str], mapping: CorrectionMapping) -> list[str]:
    """``overrides`` with every ``algorithm.rollout_correction.*`` entry replaced by
    the mapping's (one value per key), then checked against :data:`FORBIDDEN`."""

    out = [o for o in overrides if not o.lstrip("+~").startswith(PREFIX)]
    out += list(mapping.overrides)
    refuse_forbidden_overrides(out)
    return out

