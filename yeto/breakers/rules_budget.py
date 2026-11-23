"""Dependency rules of the one-budget model (D11, D12; decision 3).

``run.budget.usd`` is written once; ``launch.budget`` and ``pool.spend_cap``
inherit it. An explicit override above it (D11) and a launch warn line at another
share of the budget than the run's (D12) are reported.
"""

from __future__ import annotations

from collections.abc import Iterator

from yeto.breakers.rules import Ctx, finding, rule
from yeto.breakers.spec import Finding

EXPLICIT = ("breakers", "legacy")  # an override written by the user, not inherited or a default


@rule("D11", "one_budget")
def one_budget(ctx: Ctx) -> Iterator[Finding]:
    """launch.budget <= run.budget, and every pool's spend_cap <= run.budget (a run
    budget comes from a run config or a ``breakers:`` section, also ``yeto launch --breakers``)."""
    cap = ctx.number("run.budget.usd")
    if cap is None:
        return
    found = []
    launch = ctx.number("launch.budget.usd")
    if launch is not None and launch > cap and ctx.source("launch.budget.usd") in EXPLICIT:
        found.append(("launch.budget", launch, "the trainer's cost guard (--budget-usd)"))
    for pool in ctx.facts.pools:
        sub = Ctx(pool.resolved, ctx.facts, pool)
        spend = sub.number("pool.spend_cap.usd") if sub.source("pool.spend_cap.usd") in EXPLICIT else None
        if spend is not None and spend > cap:
            found.append((f"pool.spend_cap ({pool.name})", spend, f"pool {pool.name}'s max_run_spend_usd"))
    for name, value, what in found:
        yield finding("D11", "warn",
                      f"{name} {value:g} USD > run.budget {cap:g} USD: {what} would let the run spend past its "
                      "budget (the run monitor and the dashboard read run.budget); remove the override to "
                      "inherit run.budget, or lower it",
                      items=(name.split(" ", 1)[0], "run.budget"), values={"usd": value, "run_budget_usd": cap},
                      suggested={f"{name.split(' ', 1)[0]}.usd": cap})


@rule("D12", "one_warn_ratio")
def one_warn_ratio(ctx: Ctx) -> Iterator[Finding]:
    """One warn ratio: run.budget.warn_ratio; the launch cost guard's warn line matches it."""
    if not ctx.enabled("run.budget"):
        return
    ratio = ctx.number("run.budget.warn_ratio", on=False)
    usd, warn = ctx.number("launch.budget.usd"), ctx.number("launch.budget.warn_usd")
    if ratio is None or not usd or warn is None or ctx.source("launch.budget.warn_usd") not in EXPLICIT:
        return
    if abs(warn / usd - ratio) < 1e-6:
        return
    yield finding("D12", "info",
                  f"launch.budget warns at {warn:g} USD ({warn / usd:.0%} of {usd:g}) while run.budget warns at "
                  f"{ratio:.0%}: two warn lines for one budget; drop --budget-warn-usd to warn at "
                  f"run.budget.warn_ratio",
                  items=("launch.budget", "run.budget"), values={"warn_usd": warn, "warn_ratio": ratio},
                  suggested={"launch.budget.warn_usd": round(usd * ratio, 2)})
