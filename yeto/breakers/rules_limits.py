"""Dependency rules on limits and shared constants (D17, D18, D21).

The in-run guards and the read-only ``yeto breaker check`` share thresholds
(xhw log-prob limits, NaN skip limits): D17 and D18 report when they drift
apart. D21 extends ``ElasticConfig``'s number checks to every enabled breaker:
its limits must be finite and positive (refused when written in ``breakers:``;
a legacy knob keeps today's meaning and gets a warning).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

from yeto.breakers.rules import Ctx, finding, rule
from yeto.breakers.spec import Finding

XHW_PAIRS = ("abs_diff", "kl", "clip")  # trainer.xhw_guard warn_<x> / act_<x>
# trainer.nan skip thresholds -> the read-only health check's copies (monitor.health_nan_skip)
NAN_SKIP = (("skip_max_consecutive_n", "max_consecutive_n"), ("skip_max_fraction", "max_fraction"),
            ("skip_min_rounds_n", "min_rounds_n"))
LIMIT_UNITS = ("s", "usd", "h", "n")  # params whose 0 means "trip at once" unless declared lo=0


@rule("D17", "xhw_limits")
def xhw_limits(ctx: Ctx) -> Iterator[Finding]:
    """xhw act_* >= warn_*; the health check's amd log-prob limits equal the xhw defaults."""
    if ctx.enabled("trainer.xhw_guard"):
        for pair in XHW_PAIRS:
            warn = ctx.number(f"trainer.xhw_guard.warn_{pair}")
            act = ctx.number(f"trainer.xhw_guard.act_{pair}")
            if warn is not None and act is not None and act < warn:
                yield finding("D17", "warn",
                              f"trainer.xhw_guard: act_{pair} {act:g} < warn_{pair} {warn:g}: the guard acts before "
                              f"it warns; set act_{pair} >= warn_{pair} in YETO_RL_XHW_GUARD",
                              items=("trainer.xhw_guard",), values={"warn": warn, "act": act},
                              suggested={f"trainer.xhw_guard.act_{pair}": warn})
    for kind in ("warn", "act"):
        guard = ctx.number(f"trainer.xhw_guard.{kind}_abs_diff", on=False)
        health = ctx.number(f"monitor.health_logprob.amd_{kind}_abs_diff", on=False)
        if guard is None or health is None or math.isclose(guard, health):
            continue
        yield finding("D17", "info",
                      f"the in-run xhw guard's {kind}_abs_diff {guard:g} != the amd {kind} limit {health:g} of "
                      "yeto breaker check: the two judge one log-prob mismatch by different lines",
                      items=("trainer.xhw_guard", "monitor.health_logprob"), values={"guard": guard, "health": health},
                      suggested={f"monitor.health_logprob.amd_{kind}_abs_diff": guard})


@rule("D18", "nan_skip_limits")
def nan_skip_limits(ctx: Ctx) -> Iterator[Finding]:
    """The trainer's NaN skip thresholds equal the health check's nan_skip limits."""
    if not ctx.get("trainer.nan.skip"):
        return  # strict mode: the skip thresholds mean nothing
    for mine, theirs in NAN_SKIP:
        trainer = ctx.number(f"trainer.nan.{mine}", on=False)
        health = ctx.number(f"monitor.health_nan_skip.{theirs}", on=False)
        if trainer is None or health is None or math.isclose(trainer, health):
            continue
        yield finding("D18", "info",
                      f"trainer.nan {mine} {trainer:g} (YETO_RL_NAN_SKIP) != yeto breaker check's nan_skip "
                      f"{theirs} {health:g}: the check judges the run by other limits than the trainer breaks at",
                      items=("trainer.nan", "monitor.health_nan_skip"), values={"trainer": trainer, "health": health},
                      suggested={f"monitor.health_nan_skip.{theirs}": trainer})


def _limit_problems(entry) -> Iterator[tuple[str, object]]:
    """``(param, value)`` of an enabled breaker's numeric params that are not a
    finite, positive limit (null only where the param is nullable; 0 only where it
    declares ``lo=0``)."""
    for param in entry.spec.params:
        if param.type not in (int, float):
            continue
        value = entry.params.get(param.name)
        if value is None:
            if not param.nullable:
                yield param.name, value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            yield param.name, value
        elif value <= 0 and param.lo is None and param.unit in LIMIT_UNITS:
            yield param.name, value


@rule("D21", "finite_limits")
def finite_limits(ctx: Ctx) -> Iterator[Finding]:
    """Every enabled breaker has a finite, positive limit unless the parameter is nullable."""
    seen = set()
    for resolved in [ctx.resolved, *(p.resolved for p in ctx.facts.pools)]:
        try:
            entries = resolved.entries()
        except (AttributeError, KeyError):
            continue
        for entry in entries:
            if entry.spec.kind != "breaker" or not entry.enabled:
                continue
            for param, value in _limit_problems(entry):
                key = (entry.spec.name, param, repr(value))
                if key in seen:
                    continue
                seen.add(key)
                source = entry.sources.get(param)
                where = f" (from {source.location})" if source is not None else ""
                # a legacy knob keeps today's meaning (budget.usd 0 with nothing priced never trips):
                # a warning; the same value written in the breakers: section is refused
                legacy = source is None or source.kind in ("legacy", "default")
                yield finding("D21", "warn" if legacy else "error",
                              f"{entry.spec.name}.{param} is {value!r}{where}: an enabled breaker needs a finite, "
                              f"positive limit; set a positive number, or turn the entry off with "
                              f"breakers.{entry.spec.name}.enabled: false"
                              + (" (accepted as before for a legacy setting; refused in breakers:)" if legacy else ""),
                              items=(entry.spec.name,), values={param: value})
