"""PauseAdvice and its merge with the local pause decision (f-design §1.5, gap G3).

Advice can only tighten: any live veto refuses; the budget is the minimum of
the local budget and all live advice budgets; expired advice does not exist;
a local refusal is never turned into an allowance. ``target_resource_intent``
is carried through untouched and never triggers a cloud action (f-design §1.7).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .pause_audit import PauseDecision


@dataclass(frozen=True)
class PauseAdvice:
    source: str
    issued_at: float
    expires_at: float
    pause_budget_s: float | None = None
    veto: bool = False
    reason: str = ""
    target_resource_intent: Mapping[str, Any] | None = None

    def live(self, now: float) -> bool:
        return self.issued_at <= now < self.expires_at


def merge_pause(local: PauseDecision, advice: Iterable[PauseAdvice], *, now: float,
                expected_pause_s: float | None = None) -> PauseDecision:
    live = [a for a in advice if a.live(now)]
    if not local.allowed:
        return local
    for a in live:
        if a.veto:
            return PauseDecision(False, f"vetoed by {a.source}: {a.reason}", None,
                                 local.stalls_peers, local.profile_hash)
    budgets = [b for b in [local.budget_s, *(a.pause_budget_s for a in live)] if b is not None]
    budget = min(budgets) if budgets else None
    if expected_pause_s is not None and budget is not None and expected_pause_s > budget:
        return PauseDecision(False, f"expected pause {expected_pause_s}s exceeds advised budget "
                             f"{budget}s", budget, local.stalls_peers, local.profile_hash)
    return PauseDecision(True, local.reason, budget, local.stalls_peers, local.profile_hash)


def lease_veto(island_id: str, *, now: float, lease_remaining_s: float,
               margin_s: float = 0.0) -> PauseAdvice:
    """Coordinator advice derived from a heartbeat lease (design Q4).

    The budget is what is left of the lease minus a margin; a non-positive
    remainder is a veto.
    """
    budget = lease_remaining_s - margin_s
    return PauseAdvice(source=f"lease:{island_id}", issued_at=now, expires_at=now + max(budget, 0.0) + 1e-9,
                       pause_budget_s=max(budget, 0.0), veto=budget <= 0,
                       reason="lease_expiring" if budget <= 0 else "lease_budget")


ROLLOUT_ONLY = "rollout_only"


def slow_island_advice(round_wall_s: Mapping[str, float], *, now: float, ratio: float = 2.0,
                       ttl_s: float = 600.0, source: str = "coordinator") -> list[PauseAdvice]:
    """Suggest demoting persistently slow islands to rollout-only islands (tasks 0.11).

    An island whose per-round wall time exceeds ``ratio`` times the median of
    the other islands gets an advice carrying
    ``target_resource_intent={"action": "rollout_only", ...}``. This is a
    suggestion for a human to confirm: it is neither a veto nor a budget, and
    nothing in this module (or reading the advice) executes it.
    """
    out: list[PauseAdvice] = []
    for island, wall in sorted(round_wall_s.items()):
        others = sorted(w for i, w in round_wall_s.items() if i != island)
        if not others or wall is None:
            continue
        mid = len(others) // 2
        median = others[mid] if len(others) % 2 else (others[mid - 1] + others[mid]) / 2
        if median > 0 and wall > ratio * median:
            out.append(PauseAdvice(
                source=source, issued_at=now, expires_at=now + ttl_s,
                reason=f"round wall {wall:.3f}s > {ratio}x median {median:.3f}s of other islands",
                target_resource_intent={"island_id": island, "action": ROLLOUT_ONLY,
                                        "round_wall_s": wall, "median_other_s": median,
                                        "requires_human_confirmation": True}))
    return out
