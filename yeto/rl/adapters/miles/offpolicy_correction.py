"""Miles mapping of the neutral correction group
(change ``rl-offpolicy-correction-passthrough`` L6.3, L6.7, L6.9).

:func:`correction_mapping` turns ``AlgorithmSpec.correction`` into Miles
argv (design D2/D3); :func:`algorithm_flags.algorithm_argv` emits the same
argv through its ``--use-tis`` row. Mapping table (design D3):

====================  ==========================================================
neutral               Miles
====================  ==========================================================
tis, token            ``--use-tis --tis-clip upper --tis-clip-low lower``
ratio_mask=icepop     the same plus ``--custom-tis-function-path`` IcePop
ratio_mask=mis        the same with the MIS function; mask mode, ``level``,
                      bounds, ``batch_normalize`` and ``veto`` as namespace
                      attributes (``mismatch_correction.neutral_mis_config``)
opsm                  ``--use-opsm --opsm-delta``
====================  ==========================================================

Rows Miles cannot express give a :class:`CorrectionWarning`; phase 2 (L6b.2)
refuses the start on them: ``level`` other than token without the MIS mask, and ``veto`` /
``batch_normalize`` without the MIS mask.

Decoupled PPO (design D2) needs pi_prox, which Miles recomputes on the
weights before each update (``prox_anchor=per_step``). :data:`FORBIDDEN`
lists the flags that remove it or mask by version; :func:`refuse_forbidden_argv`
refuses them with the flag named.

Admission (design D5, approximate): :func:`admission_argv` maps the
consumption limit ``A_hard`` and an optional admission limit ``k`` to
``--max-weight-staleness`` and the fully_async buffer capacity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from yeto.rl.algos.mismatch_correction import ICEPOP_PATH, MIS_PATH, mis_config
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.offpolicy_correction import (
    PROX_PER_STEP,
    CorrectionMapping,
    CorrectionWarning,
    ForbiddenSetting,
    refuse_forbidden,
)
from yeto.rl.engine.policy_age import validate_limit

BACKEND = "miles"
PROX_ANCHOR = PROX_PER_STEP

FORBIDDEN = (
    ForbiddenSetting(BACKEND, "--use-rollout-logprobs",
                     "it makes the rollout log-probabilities the PPO clip anchor, so pi_prox "
                     "is lost"),
    ForbiddenSetting(BACKEND, "--skip-actor-forward-only",
                     "it skips the actor forward pass that computes pi_prox"),
    ForbiddenSetting(BACKEND, "--mask-offpolicy-in-partial-rollout",
                     "it masks the older-version tokens of a continued sample; every response "
                     "token must enter the loss"),
)
FORBIDDEN_FLAGS = tuple(f.setting for f in FORBIDDEN)

# Miles fully_async data buffer: capacity = factor * rollout_batch_size
# (miles/rollout/fully_async_data_buffer.py:136, Miles fork read 2026-10-14).
BUFFER_CAPACITY_FLAG = "--async-data-buffer-capacity-factor"


def _num(value: float) -> str:
    return repr(float(value))


def _function_path(spec: AlgorithmSpec) -> str | None:
    c = spec.correction
    if c.function is not None:
        return c.function.path
    return {"icepop": ICEPOP_PATH, "mis": MIS_PATH}.get(c.ratio_mask)


def tis_argv(spec: AlgorithmSpec) -> list[str]:
    """``--use-tis`` and its bounds / function for a tis or custom correction."""

    c = spec.correction
    if c.method not in ("tis", "custom"):
        return []
    argv = ["--use-tis"]
    if c.upper is not None:
        argv += ["--tis-clip", _num(c.upper)]
    if c.lower is not None:
        argv += ["--tis-clip-low", _num(c.lower)]
    function = _function_path(spec)
    if function is not None:
        argv += ["--custom-tis-function-path", function]
    return argv


def _warnings(spec: AlgorithmSpec) -> list[CorrectionWarning]:
    c = spec.correction
    if c.method != "tis" or c.ratio_mask == "mis":
        return []
    out = []
    if c.level != "token":
        out.append(CorrectionWarning(
            "level", BACKEND, f"Miles TIS{' / IcePop' if c.ratio_mask == 'icepop' else ''} "
            "weights tokens only; level other than token needs ratio_mask='mis'", "token"))
    if c.veto is not None:
        out.append(CorrectionWarning(
            "veto", BACKEND, "Miles has a veto only in the MIS function (ratio_mask='mis')", None))
    if c.batch_normalize:
        out.append(CorrectionWarning(
            "batch_normalize", BACKEND,
            "Miles normalizes weights only in the MIS function (ratio_mask='mis')", False))
    return out


def _values(argv: Sequence[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            values[flag] = argv[i + 1]
            i += 2
        else:
            values[flag] = True
            i += 1
    return values


def correction_mapping(spec: AlgorithmSpec) -> CorrectionMapping:
    """The Miles settings of ``spec.correction`` with warnings and ``prox_anchor``."""

    c = spec.correction
    argv = tis_argv(spec)
    if c.opsm_delta is not None:
        argv += ["--use-opsm", "--opsm-delta", _num(c.opsm_delta)]
    if c.use_rollout_logprobs:
        argv.append("--use-rollout-logprobs")  # refused by refuse_forbidden_argv
    if c.mismatch_metrics:
        argv.append("--get-mismatch-metrics")
    attrs = mis_config(spec)
    values = {**_values(argv), **{f"namespace.{k}": v for k, v in sorted(attrs.items())}}
    return CorrectionMapping(backend=BACKEND, prox_anchor=PROX_ANCHOR, values=values,
                             warnings=tuple(_warnings(spec)), argv=tuple(argv),
                             runtime_attrs=attrs)


def refuse_forbidden_argv(argv: Sequence[str]) -> None:
    """Refuse a Miles argv that holds a :data:`FORBIDDEN` flag (names the flag)."""

    refuse_forbidden((token.split("=", 1)[0] for token in argv), FORBIDDEN)


def admission_argv(max_policy_age: int, *, admission_limit: int | None = None) -> tuple[str, ...]:
    """Approximate admission and consumption limits on Miles fully_async (D5).

    Consumption: ``--max-weight-staleness A_hard`` drops a group whose oldest
    token is more than ``A_hard`` versions behind (whole group). Admission:
    Miles has no limit by start version; a buffer of ``k`` rounds of groups
    (``--async-data-buffer-capacity-factor k``) bounds how far generation can
    run ahead. ``k`` None keeps Miles' buffer default. ``k <= A_hard``.
    """

    limit = validate_limit(max_policy_age)
    if limit == 0:
        return ()
    argv = ["--max-weight-staleness", str(limit)]
    if admission_limit is not None:
        k = validate_limit(admission_limit)
        if not 1 <= k <= limit:
            raise ValueError(f"admission limit k={k} must be in [1, A_hard={limit}]")
        argv += [BUFFER_CAPACITY_FLAG, str(k)]
    return tuple(argv)
