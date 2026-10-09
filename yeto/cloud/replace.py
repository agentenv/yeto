"""Replacement candidates and cost scoring after a reclaim (rl-spot-cost-saving 2.1-2.4, D4-D6).

Pure functions plus one small planner object:

* :func:`candidates` -- (cloud, region, gpu) offers -> kept / rejected with a reason
  (same compatibility group only; weights not in the target cloud add copy cost;
  a region probed as sold out is rejected, so the next region is tried).
* :func:`score` -- D5: price / effective compute + prep cost / expected stay +
  reclaim rate x loss per reclaim. Fewer than ``MIN_RECLAIMS`` records in a
  (cloud, region) -> rate "unknown": the term is None, not used for ranking,
  and ``risks`` says so. Every term is returned with its source.
* :class:`ReplacementPlanner` -- D6: auto launch is off by default. Off -> only
  a ``spot_replace_advice`` event, the launch callable is never called. On ->
  check the run budget before every launch; over budget -> ``spot_budget_cap``
  event and no launch.

Phase 2 (one on-demand anchor island + spot islands that may be dropped)
reuses the planner unchanged: the anchor is simply never handed to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

MIN_RECLAIMS = 3  # D5: proposed value, no measured basis; adjust once data exists
ADVICE_EVENT = "spot_replace_advice"
LAUNCH_EVENT = "spot_replace_launch"
BUDGET_EVENT = "spot_budget_cap"


@dataclass(frozen=True)
class Offer:
    cloud: str
    region: str
    gpu: str
    gpus: int
    price_per_hour: float | None
    in_stock: bool | None = None          # capacity probe; None = not probed
    effective_tflops: float | None = None  # per node; None -> price term uses raw price
    has_weights: bool = False
    weight_copy_s: float | None = None     # used when has_weights is False


@dataclass
class Candidate:
    offer: Offer
    score: float | None = None
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)


def candidates(offers: Iterable[Offer], *, compat_gpu: str, origin: tuple[str, str] | None = None
               ) -> tuple[list[Offer], list[tuple[Offer, str]]]:
    kept, rejected = [], []
    for o in offers:
        if o.gpu != compat_gpu:
            rejected.append((o, "compatibility group differs"))
        elif o.in_stock is False:
            why = "origin region out of stock" if origin == (o.cloud, o.region) else "out of stock"
            rejected.append((o, why))
        elif not o.has_weights and o.weight_copy_s is None:
            rejected.append((o, "weights not in target cloud and copy cost unknown"))
        else:
            kept.append(o)
    return kept, rejected


def score(offer: Offer, *, prep_s: float, prep_source: str, expected_stay_s: float,
          reclaim_summary: Mapping[tuple[str, str | None], Mapping[str, Any]] | None = None,
          loss_per_reclaim_usd: float | None = None) -> Candidate:
    """D5 score in USD per hour-equivalent; lower is better."""
    c = Candidate(offer)
    if offer.price_per_hour is None:
        c.risks.append("price unknown")
        return c
    if offer.effective_tflops:
        price_term = offer.price_per_hour / offer.effective_tflops
        c.terms["price"] = {"value": price_term, "source": "price_per_hour / effective_tflops"}
    else:
        price_term = offer.price_per_hour
        c.terms["price"] = {"value": price_term, "source": "price_per_hour (effective compute unknown)"}
        c.risks.append("effective compute unknown")
    total_prep = prep_s + (0.0 if offer.has_weights else (offer.weight_copy_s or 0.0))
    prep_cost = offer.price_per_hour * total_prep / 3600.0
    prep_term = prep_cost / max(expected_stay_s / 3600.0, 1e-9)
    c.terms["prep"] = {"value": prep_term, "prep_s": total_prep,
                       "source": prep_source + ("" if offer.has_weights else " + weight copy")}
    rec = (reclaim_summary or {}).get((offer.cloud, offer.region))
    count = int(rec["count"]) if rec else 0
    if count < MIN_RECLAIMS or loss_per_reclaim_usd is None:
        c.terms["reclaim"] = {"value": None, "records": count,
                              "source": f"fewer than {MIN_RECLAIMS} reclaim records" if count < MIN_RECLAIMS
                              else "loss per reclaim unknown"}
        c.risks.append("reclaim rate unknown")
        reclaim_term = 0.0
    else:
        intervals = rec.get("intervals_s") or []
        mean_gap_h = (sum(intervals) / len(intervals) / 3600.0) if intervals else None
        rate = (1.0 / mean_gap_h) if mean_gap_h else None
        if rate is None:
            c.terms["reclaim"] = {"value": None, "records": count, "source": "no interval"}
            c.risks.append("reclaim rate unknown")
            reclaim_term = 0.0
        else:
            reclaim_term = rate * loss_per_reclaim_usd
            c.terms["reclaim"] = {"value": reclaim_term, "records": count, "rate_per_h": rate,
                                  "source": "spot_reclaim summary"}
    c.score = price_term + prep_term + reclaim_term
    return c


def rank(cands: Iterable[Candidate]) -> list[Candidate]:
    return sorted((c for c in cands if c.score is not None), key=lambda c: c.score)


@dataclass
class ReplacementPlanner:
    """Called by FleetController when an island is recovering (D4/D6)."""

    offers: Callable[[str], Iterable[Offer]]   # island name -> current offers
    compat_gpu: str
    emit: Callable[..., Any]
    prep_s: float
    prep_source: str = "cloud-pool-design §5.3 conservative bound"
    expected_stay_s: float = 3600.0
    reclaim_summary: Callable[[], Mapping] | None = None
    loss_per_reclaim_usd: float | None = None
    auto_launch: bool = False
    budget_usd: float | None = None
    spent_usd: Callable[[], float] = lambda: 0.0
    launch: Callable[[str, Offer], Any] | None = None
    launch_hours: float = 1.0                  # hours charged against the budget per launch

    def __post_init__(self) -> None:
        if self.auto_launch and (self.budget_usd is None or self.launch is None):
            raise ValueError("auto launch needs both a run budget and a launch callable (D6)")

    def plan(self, island: str, origin: tuple[str, str] | None = None) -> dict[str, Any]:
        kept, rejected = candidates(self.offers(island), compat_gpu=self.compat_gpu, origin=origin)
        summary = self.reclaim_summary() if self.reclaim_summary else None
        ranked = rank(score(o, prep_s=self.prep_s, prep_source=self.prep_source,
                            expected_stay_s=self.expected_stay_s, reclaim_summary=summary,
                            loss_per_reclaim_usd=self.loss_per_reclaim_usd) for o in kept)
        advice = {
            "island": island, "auto_launch": self.auto_launch,
            "candidates": [{"cloud": c.offer.cloud, "region": c.offer.region, "gpu": c.offer.gpu,
                            "gpus": c.offer.gpus, "score": c.score, "terms": c.terms, "risks": c.risks}
                           for c in ranked],
            "rejected": [{"cloud": o.cloud, "region": o.region, "gpu": o.gpu, "reason": why}
                         for o, why in rejected],
        }
        self.emit(ADVICE_EVENT, **advice)
        if not self.auto_launch or not ranked:
            return {"action": "advise", **advice}
        best = ranked[0]
        cost = (best.offer.price_per_hour or 0.0) * self.launch_hours
        spent = float(self.spent_usd())
        if spent + cost > float(self.budget_usd):
            self.emit(BUDGET_EVENT, island=island, spent_usd=spent, next_cost_usd=cost,
                      budget_usd=self.budget_usd)
            return {"action": "budget_cap", **advice}
        self.emit(LAUNCH_EVENT, island=island, cloud=best.offer.cloud, region=best.offer.region,
                  gpu=best.offer.gpu, gpus=best.offer.gpus, spent_usd=spent, next_cost_usd=cost,
                  budget_usd=self.budget_usd)
        result = self.launch(island, best.offer)
        return {"action": "launch", "result": result, **advice}
