"""Exact fleet-plan solver: how many learner islands of each shape to run.

Chooses integer counts x_c >= 0 per candidate island shape so as to
lexicographically maximize total effective TFLOPs, then minimize total $/hr
among FLOPs-optimal plans, subject to a $/hr budget (which must also cover a
fixed head-VM cost), per-quota-bucket vCPU caps, and a fleet-wide island cap.

WHY branch-and-bound instead of pulp/CBC: instances are tiny (tens of
candidates, at most ~16 islands), so exact search with two admissible upper
bounds solves them in microseconds and keeps yeto dependency-free.  The
lexicographic objective is also awkward to encode as a single MILP objective
without fragile big-M weights.  Everything here is deterministic: candidate
order, branching order, and tie-breaks are all fixed, so the same inputs
always produce the same plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Float slack for budget/quota comparisons: prices and caps come from provider
# catalogs with a few decimals, so 1e-9 is far below any real distinction.
_EPS = 1e-9


@dataclass(frozen=True)
class Candidate:
    """One launchable island shape in one region.

    An island is an atomic unit: `nodes` machines launched (and lost)
    together, so price, vCPUs, and TFLOPs are all per-island aggregates.
    """

    key: str  # display/launch key, e.g. "aws:3x8xa100@us-east-2"
    region: str
    gpu: str
    instance_type: str
    nodes: int
    gpus_per_node: int
    vcpus_per_island: int  # nodes * vcpus_per_node
    price_per_hour: float  # per island (spot price * nodes)
    eff_tflops: float  # per island
    quota_bucket: tuple[str, str] | None  # (region, quota_code); None = uncapped
    score: int | None  # spot placement score, informational
    # True when the score was unavailable and optimistically assumed instead
    # of measured (default non-strict planning); rendering marks these.
    assumed: bool = False
    # Per-candidate island cap (None = unlimited). Used when the placement
    # score is only verified up to some aggregate capacity (AWS) or stock
    # is limited (RunPod): the shape stays plannable at the safe size
    # instead of being dropped outright.
    max_count: int | None = None
    cloud: str = "aws"  # lowercase sky cloud name


@dataclass
class Plan:
    """A solved fleet plan plus why it stopped growing."""

    counts: dict[str, int]  # candidate.key -> islands (only nonzero entries)
    total_tflops: float
    total_cost: float  # islands + head_cost (0.0 for an empty plan)
    binding: list[str]  # human-readable binding constraints


def _ratio(c: Candidate) -> float:
    """TFLOPs per dollar; free capacity (on-prem) sorts ahead of everything."""
    if c.price_per_hour <= 0:
        return math.inf
    return c.eff_tflops / c.price_per_hour


def _tiebreak(counts: dict[str, int]) -> tuple[list[str], list[tuple[str, int]]]:
    """Deterministic final tie-break so equal-(tflops, cost) plans are stable."""
    return (sorted(counts), sorted(counts.items()))


def solve(
    candidates: list[Candidate],
    budget: float | None,
    quota_limits: dict[tuple[str, str], float],
    max_islands: int = 16,
    head_cost: float = 0.40,
    target_tflops: float | None = None,
) -> Plan:
    """Solve the fleet ILP exactly, in one of two objective modes.

    Budget mode (target_tflops=None): lexicographically maximize TFLOPs,
    then minimize cost, subject to `budget` ($/hr total including the head
    VM, so islands only get `budget - head_cost` to spend).

    Target mode (target_tflops set): minimize cost subject to total TFLOPs
    >= target (then maximize TFLOPs among cost-optimal plans). `budget` may
    be None (uncapped) or double as an additional cap.

    Returns an empty plan (counts={}) with binding explaining why when
    nothing fits — callers render the message.
    """
    room = (budget - head_cost) if budget is not None else math.inf
    fail_binding = ["target-flops"] if target_tflops is not None else ["budget"]
    if room <= 0:
        # The head VM alone exhausts the budget; nothing can launch.
        return Plan(counts={}, total_tflops=0.0, total_cost=0.0, binding=["budget"])

    # Candidates that can never appear (useless or individually unaffordable)
    # are dropped up front so the bounds below stay tight.
    usable = [
        c
        for c in candidates
        if c.eff_tflops > 0 and c.price_per_hour <= room + _EPS
    ]
    if not usable:
        return Plan(counts={}, total_tflops=0.0, total_cost=0.0, binding=list(fail_binding))

    # Best value-per-dollar first so the greedy-ish DFS finds a strong
    # incumbent immediately; key as secondary sort keeps order deterministic.
    order = sorted(usable, key=lambda c: (-_ratio(c), c.key))
    n = len(order)

    # Suffix maxima feeding the two admissible upper bounds:
    #  - budget bound: remaining $ times best remaining TFLOPs/$ (infinite if
    #    any zero-price candidate remains — it adds TFLOPs for free),
    #  - island bound: remaining island slots times best remaining per-island
    #    TFLOPs.  Their min prunes both budget-tight and quota/slot-tight
    #    subtrees.
    suffix_free = [False] * (n + 1)
    suffix_ratio = [0.0] * (n + 1)
    suffix_tflops = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        r = _ratio(order[i])
        suffix_free[i] = suffix_free[i + 1] or math.isinf(r)
        suffix_ratio[i] = max(suffix_ratio[i + 1], 0.0 if math.isinf(r) else r)
        suffix_tflops[i] = max(suffix_tflops[i + 1], order[i].eff_tflops)

    # Budget mode maximizes (tflops, -cost) and the empty plan is a valid
    # incumbent; target mode maximizes (-cost, tflops) over plans meeting
    # the target, so it starts with no incumbent at all.
    best_counts: dict[str, int] = {}
    best_tflops = 0.0
    best_cost = 0.0
    best_found = target_tflops is None
    best_key = (0.0, -0.0)

    cur_counts: dict[str, int] = {}
    quota_used: dict[tuple[str, str], float] = {}

    def consider(tflops: float, cost: float) -> None:
        nonlocal best_counts, best_tflops, best_cost, best_key, best_found
        if target_tflops is not None:
            if tflops + _EPS < target_tflops:
                return
            key = (-round(cost, 9), round(tflops, 9))
        else:
            key = (round(tflops, 9), -round(cost, 9))
        if (
            not best_found
            or key > best_key
            or (key == best_key and _tiebreak(cur_counts) < _tiebreak(best_counts))
        ):
            best_counts = dict(cur_counts)
            best_tflops, best_cost, best_key = tflops, cost, key
            best_found = True

    def dfs(i: int, budget_left: float, islands_left: int, tflops: float, cost: float) -> None:
        if i == n or islands_left == 0:
            consider(tflops, cost)
            return
        max_more = min(
            math.inf if suffix_free[i] else budget_left * suffix_ratio[i],
            islands_left * suffix_tflops[i],
        )
        if target_tflops is not None:
            if round(tflops + max_more, 9) < target_tflops - _EPS:
                return  # the target is unreachable from here
            extra = max(0.0, target_tflops - tflops)
            if extra > 0:
                # Admissible lower bound on the cost still needed to reach
                # the target: buy the deficit at the best remaining rate.
                lb = 0.0 if suffix_free[i] else extra / suffix_ratio[i]
            else:
                lb = 0.0
            if best_found and round(cost + lb, 9) > -best_key[0]:
                return  # cannot get cheaper than the incumbent
        else:
            bound_r = round(tflops + max_more, 9)
            if bound_r < best_key[0]:
                return
            if bound_r == best_key[0] and round(cost, 9) > -best_key[1]:
                # Cannot beat the incumbent's TFLOPs, and cost only grows from
                # here (prices >= 0), so cost cannot beat it either.  Equal cost
                # keeps exploring so the key tie-break stays authoritative.
                return

        c = order[i]
        ub = islands_left
        if c.max_count is not None:
            ub = min(ub, c.max_count)
        if c.price_per_hour > 0 and not math.isinf(budget_left):
            ub = min(ub, int((budget_left + _EPS) // c.price_per_hour))
        cap = quota_limits.get(c.quota_bucket) if c.quota_bucket is not None else None
        if cap is not None and c.vcpus_per_island > 0:
            vcpu_room = cap - quota_used.get(c.quota_bucket, 0.0)
            ub = min(ub, max(0, int((vcpu_room + _EPS) // c.vcpus_per_island)))

        # High counts first: pairs with the ratio ordering to reach a
        # near-greedy incumbent on the first descent, maximizing pruning.
        for x in range(ub, -1, -1):
            if x > 0:
                cur_counts[c.key] = x
                if cap is not None:
                    quota_used[c.quota_bucket] = (
                        quota_used.get(c.quota_bucket, 0.0) + x * c.vcpus_per_island
                    )
            dfs(
                i + 1,
                budget_left - x * c.price_per_hour,
                islands_left - x,
                tflops + x * c.eff_tflops,
                cost + x * c.price_per_hour,
            )
            if x > 0:
                del cur_counts[c.key]
                if cap is not None:
                    quota_used[c.quota_bucket] -= x * c.vcpus_per_island

    dfs(0, room, max_islands, 0.0, 0.0)

    if not best_found:
        # Target mode with an unreachable target: no feasible plan exists.
        return Plan(counts={}, total_tflops=0.0, total_cost=0.0, binding=["target-flops"])

    counts = dict(sorted(best_counts.items()))
    total_cost = best_cost + head_cost if counts else 0.0

    # Report which constraints are tight so callers can explain the plan
    # ("add quota in us-east-2 to grow", "raise the budget", ...).
    binding: list[str] = []
    if budget is not None and min(c.price_per_hour for c in usable) > (budget - total_cost) + _EPS:
        binding.append("budget")
    by_key = {c.key: c for c in usable}
    bucket_used: dict[tuple[str, str], float] = {}
    for key, x in counts.items():
        c = by_key[key]
        if c.quota_bucket is not None and c.quota_bucket in quota_limits:
            bucket_used[c.quota_bucket] = (
                bucket_used.get(c.quota_bucket, 0.0) + x * c.vcpus_per_island
            )
    for bucket, cap in sorted(quota_limits.items()):
        members = [
            c for c in usable if c.quota_bucket == bucket and c.vcpus_per_island > 0
        ]
        if members and all(
            bucket_used.get(bucket, 0.0) + c.vcpus_per_island > cap + _EPS
            for c in members
        ):
            binding.append(f"quota:{bucket[0]}:{bucket[1]}")
    if sum(counts.values()) == max_islands:
        binding.append("max-islands")

    return Plan(
        counts=counts,
        total_tflops=best_tflops,
        total_cost=total_cost,
        binding=binding,
    )
