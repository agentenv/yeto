"""Dependency rules between breaker thresholds (design section 6, rules D1-D21).

A threshold is often only right relative to another one: the trainer's sample wait
must outlast a trial plus a drain, the dispense lease must outlast a write
deferral, the samplers' backlog caps must hold the groups the trainer reads
between two releases. Each rule here is one such constraint, checked once, for
every entry point (``yeto run up`` preflight, ``yeto launch --dry-run``, the nc
``pool --dry-run`` and ``yeto breaker show``) through :func:`check`.

A rule reads only resolved breaker values (:class:`yeto.breakers.resolve.ResolvedBreakers`)
and the run facts it needs (:mod:`yeto.breakers.rules_facts`: release cadence,
groups per step, sampler counts, dataset rows, ...), and returns
:class:`~yeto.breakers.spec.Finding` s with a level, a message that names the
values, and the value to set (``suggested``). A missing input skips the check
(or leaves a structural hint); a rule never raises on one.

Levels follow decision 6: D2, D3, D6, D8, D15, D16 and D21 are errors and refuse
the start; every other rule warns or informs. D19 keeps the refusal of the check
it replaced (#608: samplers that can never fill the rounds up to the first
release) and warns otherwise. ``breakers.deps.overrides`` changes the level of
one rule (``"D8": "warn"``) or turns it off; the finding then carries
``overridden``.

The rules are grouped by what they protect: :mod:`.rules_timing` (timeouts that
must nest), :mod:`.rules_data` (groups and tasks: coverage, backlog, policy age,
min samples), :mod:`.rules_budget` (one budget, one wall clock) and
:mod:`.rules_limits` (shared constants and finite limits).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from yeto.breakers.spec import Finding

if TYPE_CHECKING:
    from yeto.breakers.resolve import ResolvedBreakers, Sources
    from yeto.breakers.rules_facts import Facts, PoolFacts

Level = Literal["error", "warn", "info"]
LEVELS: tuple[Level, ...] = ("info", "warn", "error")

RULE_IDS: tuple[str, ...] = tuple(f"D{i}" for i in range(1, 22))
# decision 6: the rules whose findings refuse the start (D19: see the module doc)
ERROR_RULES = frozenset({"D2", "D3", "D6", "D8", "D15", "D16", "D21"})


@dataclass(frozen=True)
class Ctx:
    """What one rule sees: the resolved breakers (a pool's own resolve for a
    per-pool rule), the run facts, and the pool being checked (None: run-wide)."""
    resolved: ResolvedBreakers
    facts: Facts
    pool: PoolFacts | None = None

    def get(self, path: str) -> Any:
        """The resolved value of ``<entry>.<param>`` (None: not registered or unset)."""
        try:
            return self.resolved.get(path, None)
        except KeyError:
            return None

    def on(self, path: str) -> Any:
        """:meth:`get` when the entry is enabled, else None (an off entry bounds nothing)."""
        return self.get(path) if self.enabled(path.rsplit(".", 1)[0]) else None

    def number(self, path: str, *, on: bool = True) -> float | None:
        """:meth:`on` (``on=False``: :meth:`get`) as a float; None when it is not a number."""
        return as_number(self.on(path) if on else self.get(path))

    def enabled(self, name: str) -> bool:
        """Whether entry ``name`` is registered and enabled."""
        try:
            return bool(self.resolved.enabled(name))
        except KeyError:
            return False

    def registered(self, name: str) -> bool:
        try:
            self.resolved.entry(name)
        except KeyError:
            return False
        return True

    def source(self, path: str) -> str | None:
        """The source kind of ``<entry>.<param>`` (``breakers`` / ``legacy`` / ``switch`` /
        ``inherit`` / ``default``; None: not registered)."""
        name, param = path.rsplit(".", 1)
        try:
            src = self.resolved.entry(name).sources.get(param)
        except KeyError:
            return None
        return None if src is None else src.kind

    def where(self, message: str) -> str:
        """``message`` prefixed with the pool's name when the run has several pools."""
        if self.pool is None or len(self.facts.pools) < 2:
            return message
        return f"pool {self.pool.name}: {message}"


@dataclass(frozen=True)
class Rule:
    id: str
    slug: str                      # "dispense_coverage": the override key "D8.dispense_coverage"
    scope: Literal["run", "pool"]  # "pool": once per pool, with that pool's resolve
    max_level: Level
    check: Callable[[Ctx], Iterable[Finding]]
    doc: str = ""


_RULES: dict[str, Rule] = {}


def rule(rule_id: str, slug: str, *, scope: Literal["run", "pool"] = "run",
         max_level: Level | None = None) -> Callable[[Callable[[Ctx], Iterable[Finding]]], Callable]:
    """Register a rule function (``max_level``: decision 6 by default)."""
    if rule_id not in RULE_IDS:
        raise ValueError(f"unknown rule id {rule_id!r}")

    def register(fn: Callable[[Ctx], Iterable[Finding]]) -> Callable:
        if rule_id in _RULES:
            raise ValueError(f"rule {rule_id} registered twice")
        level = max_level or ("error" if rule_id in ERROR_RULES else "warn")
        _RULES[rule_id] = Rule(rule_id, slug, scope, level, fn, (fn.__doc__ or "").strip())
        return fn
    return register


def rules() -> list[Rule]:
    """Every rule, in id order (the rule modules register on first use)."""
    from yeto.breakers import rules_budget, rules_data, rules_limits, rules_timing  # noqa: F401, PLC0415

    return sorted(_RULES.values(), key=lambda r: int(r.id[1:]))


def finding(rule_id: str, level: Level, message: str, *, items: Iterable[str] = (),
            values: Mapping[str, Any] | None = None, suggested: Mapping[str, Any] | None = None) -> Finding:
    return Finding(rule=rule_id, level=level, message=message, items=tuple(items),
                   values=dict(values or {}), suggested=dict(suggested or {}))


def as_number(value: Any) -> float | None:
    """``value`` as a finite float (a bool, None, a string, nan or inf: None)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def fmt_s(seconds: float) -> str:
    """``5400 s (90 min)``: seconds with minutes or hours for reading."""
    if seconds >= 7200:
        return f"{seconds:g} s ({seconds / 3600:.3g} h)"
    if seconds >= 120:
        return f"{seconds:g} s ({seconds / 60:.3g} min)"
    return f"{seconds:g} s"


# -------------------------------------------------------------------- entry point

def check(resolved: ResolvedBreakers, *, run_config: Mapping[str, Any] | None = None,
          sources: Sources | None = None,
          pools: Iterable[tuple[str, Mapping[str, Any], str]] | None = None) -> list[Finding]:
    """Every finding of every rule over one run (API section 7).

    ``run_config``: the whole run config (trainer argv, pools, budget); ``sources``:
    the surfaces of a single process (``yeto launch`` argv, one pool's spec and
    elastic config); ``pools``: ``(name, pool config, data path)`` of each pool when
    the caller already loaded them (``yeto run up``), else they are read from
    ``run_config``. Per-pool rules run once per pool on that pool's own resolve.
    ``resolved.deps.overrides`` is applied. Never raises on a missing input."""
    from yeto.breakers import rules_facts  # noqa: PLC0415

    facts = rules_facts.collect(resolved, run_config=run_config, sources=sources, pools=pools)
    return evaluate(resolved, facts)


def evaluate(resolved: ResolvedBreakers, facts: Facts, *, only: Iterable[str] | None = None) -> list[Finding]:
    """The findings of the rules (``only``: these ids) over collected ``facts``, with
    ``resolved.deps.overrides`` applied."""
    wanted = None if only is None else set(only)
    found: list[Finding] = []
    for r in rules():
        if wanted is not None and r.id not in wanted:
            continue
        targets: tuple[PoolFacts | None, ...] = facts.pools if r.scope == "pool" else (None,)
        for pool in targets:
            ctx = Ctx(resolved if pool is None else pool.resolved, facts, pool)
            found += [_capped(r, f) for f in r.check(ctx)]
    overrides = getattr(getattr(resolved, "deps", None), "overrides", None) or {}
    return apply_overrides(found, overrides)


def _capped(r: Rule, f: Finding) -> Finding:
    if LEVELS.index(f.level) <= LEVELS.index(r.max_level):
        return f
    return dataclasses.replace(f, level=r.max_level)


def override_for(rule_id: str, overrides: Mapping[str, str]) -> str | None:
    """The override of ``rule_id``: key ``"D8"`` or ``"D8.<slug>"`` (the slug is
    for reading; any suffix after the id matches)."""
    for key, value in overrides.items():
        if str(key).split(".", 1)[0] == rule_id:
            return str(value)
    return None


def apply_overrides(found: Iterable[Finding], overrides: Mapping[str, str]) -> list[Finding]:
    """``found`` with ``deps.overrides`` applied: a rule set to ``off`` is dropped,
    a level override replaces each finding's level (``overridden`` records it)."""
    out = []
    for f in found:
        value = override_for(f.rule, overrides)
        if value is None:
            out.append(f)
        elif value != "off":
            out.append(dataclasses.replace(f, level=value, overridden=value))
    return out


def errors(found: Iterable[Finding]) -> list[Finding]:
    return [f for f in found if f.level == "error"]


def format_finding(f: Finding) -> str:
    """``D8 error dispense coverage: ... (suggested: sampler... = 6) [override: warn]``."""
    text = f"{f.rule} {f.level} {f.message}"
    if f.suggested:
        text += " (suggested: " + ", ".join(f"{k} = {_show(v)}" for k, v in f.suggested.items()) + ")"
    if f.overridden:
        text += f" [deps.overrides: {f.overridden}]"
    return text


def _show(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def report(found: Iterable[Finding], say: Callable[[str], Any], *, prefix: str = "") -> bool:
    """Print each finding (``prefix`` + ``WARNING:`` / ``ERROR:`` / ``note:``) with
    ``say``; True when none is an error (the start may go on)."""
    ok = True
    for f in found:
        tag = {"error": "ERROR", "warn": "WARNING", "info": "note"}[f.level]
        say(f"{prefix}{tag}: {format_finding(f)}")
        ok &= f.level != "error"
    return ok
