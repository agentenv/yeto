"""Tests for the fleet-plan ILP solver.

The unit tests pin the individual behaviors (budget, quota, lexicographic
cost, head cost, empty plans); the brute-force cross-check is the real
guarantee — it exhaustively enumerates every feasible integer assignment on
seeded random instances and asserts the branch-and-bound finds the same
lexicographic optimum.
"""

from __future__ import annotations

import itertools
import random

import pytest

from yeto.shape.ilp import Candidate, Plan, solve


def _cand(
    key: str,
    price: float,
    tflops: float,
    *,
    vcpus: int = 32,
    bucket: tuple[str, str] | None = None,
) -> Candidate:
    """Candidate with plausible filler for fields the solver ignores."""
    return Candidate(
        key=key,
        region="us-east-2",
        gpu="a100",
        instance_type="p4d.24xlarge",
        nodes=1,
        gpus_per_node=8,
        vcpus_per_island=vcpus,
        price_per_hour=price,
        eff_tflops=tflops,
        quota_bucket=bucket,
        score=None,
    )


def test_budget_bound_prefers_better_ratio() -> None:
    good = _cand("good", price=2.0, tflops=100.0)
    bad = _cand("bad", price=2.0, tflops=50.0)
    plan = solve([bad, good], budget=10.0, quota_limits={}, head_cost=0.4)
    # 9.6 of island budget fits four $2 islands; the better ratio wins them all.
    assert plan.counts == {"good": 4}
    assert plan.total_tflops == 400.0
    assert plan.total_cost == 8.0 + 0.4
    assert "budget" in plan.binding


def test_quota_bound_caps_count_despite_budget_headroom() -> None:
    bucket = ("us-east-1", "L-1216C47A")
    c = _cand("capped", price=1.0, tflops=100.0, vcpus=32, bucket=bucket)
    plan = solve([c], budget=100.0, quota_limits={bucket: 64.0}, head_cost=0.4)
    # 64 vCPUs of quota holds exactly two 32-vCPU islands.
    assert plan.counts == {"capped": 2}
    assert "quota:us-east-1:L-1216C47A" in plan.binding
    assert "budget" not in plan.binding


def test_lexicographic_equal_tflops_picks_cheaper() -> None:
    pricey = _cand("pricey", price=2.0, tflops=100.0)
    cheap = _cand("cheap", price=1.0, tflops=100.0)
    plan = solve(
        [pricey, cheap], budget=10.4, quota_limits={}, max_islands=5, head_cost=0.4
    )
    # Any 5 islands give 500 TFLOPs; only the all-cheap plan minimizes cost.
    assert plan.counts == {"cheap": 5}
    assert plan.total_tflops == 500.0
    assert plan.total_cost == 5.0 + 0.4


def test_zero_price_onprem_bounded_only_by_max_islands() -> None:
    onprem = _cand("onprem", price=0.0, tflops=50.0, bucket=None)
    plan = solve([onprem], budget=1.0, quota_limits={}, max_islands=3, head_cost=0.4)
    assert plan.counts == {"onprem": 3}
    assert plan.total_cost == 0.4  # head only; the islands are free
    assert "max-islands" in plan.binding
    assert "budget" not in plan.binding


def test_head_cost_shrinks_island_budget_and_is_in_total() -> None:
    too_big = _cand("too-big", price=0.7, tflops=1000.0)
    fits = _cand("fits", price=0.3, tflops=10.0)
    plan = solve([too_big, fits], budget=1.0, quota_limits={}, head_cost=0.4)
    # Only $0.6 remains after the head, so the $0.7 island can never launch.
    assert plan.counts == {"fits": 2}
    assert abs(plan.total_cost - 1.0) < 1e-9  # 2 * 0.3 + 0.4 head
    assert "budget" in plan.binding


def test_nothing_affordable_returns_empty_plan() -> None:
    c = _cand("c", price=5.0, tflops=100.0)
    plan = solve([c], budget=1.0, quota_limits={}, head_cost=0.4)
    assert plan.counts == {}
    assert plan.total_tflops == 0.0
    assert plan.binding == ["budget"]


def test_head_cost_exceeding_budget_returns_empty_plan() -> None:
    c = _cand("c", price=0.1, tflops=100.0)
    plan = solve([c], budget=0.3, quota_limits={}, head_cost=0.4)
    assert plan.counts == {}
    assert plan.binding == ["budget"]


def _brute_force(
    candidates: list[Candidate],
    budget: float,
    quota_limits: dict[tuple[str, str], float],
    max_islands: int,
    head_cost: float,
) -> tuple[float, float]:
    """Exhaustive lexicographic optimum as (tflops, island cost), rounded."""
    room = budget - head_cost
    usable = [
        c for c in candidates if c.eff_tflops > 0 and c.price_per_hour <= room + 1e-9
    ]
    if room <= 0 or not usable:
        return (0.0, 0.0)
    ranges = []
    for c in usable:
        ub = max_islands
        if c.price_per_hour > 0:
            ub = min(ub, int((room + 1e-9) // c.price_per_hour))
        ranges.append(range(ub + 1))
    best = (0.0, -0.0)
    for xs in itertools.product(*ranges):
        if sum(xs) > max_islands:
            continue
        cost = sum(x * c.price_per_hour for x, c in zip(xs, usable))
        if cost > room + 1e-9:
            continue
        used: dict[tuple[str, str], float] = {}
        for x, c in zip(xs, usable):
            if c.quota_bucket is not None and c.quota_bucket in quota_limits:
                used[c.quota_bucket] = used.get(c.quota_bucket, 0.0) + x * c.vcpus_per_island
        if any(v > quota_limits[b] + 1e-9 for b, v in used.items()):
            continue
        tflops = sum(x * c.eff_tflops for x, c in zip(xs, usable))
        key = (round(tflops, 9), -round(cost, 9))
        if key > best:
            best = key
    return (best[0], -best[1])


def test_brute_force_cross_check() -> None:
    rng = random.Random(0)
    buckets = [("us-east-1", "L-AAAA"), ("eu-west-1", "L-BBBB")]
    for instance in range(25):
        n = rng.randint(3, 6)
        n_buckets = rng.randint(0, 2)
        active = buckets[:n_buckets]
        quota_limits = {b: rng.uniform(20.0, 200.0) for b in active}
        cands = [
            _cand(
                f"i{instance}-c{j}",
                price=rng.uniform(0.5, 30.0),
                tflops=rng.uniform(10.0, 500.0),
                vcpus=rng.choice([8, 16, 32, 64, 96]),
                bucket=rng.choice(active + [None]) if active else None,
            )
            for j in range(n)
        ]
        budget = rng.uniform(10.0, 60.0)
        head_cost = 0.4
        max_islands = 5

        plan = solve(
            cands,
            budget=budget,
            quota_limits=quota_limits,
            max_islands=max_islands,
            head_cost=head_cost,
        )
        want_tflops, want_cost = _brute_force(
            cands, budget, quota_limits, max_islands, head_cost
        )
        got_cost = plan.total_cost - head_cost if plan.counts else 0.0
        assert round(plan.total_tflops, 9) == want_tflops, f"instance {instance}"
        assert round(got_cost, 9) == want_cost, f"instance {instance}"


def test_plan_dataclass_shape() -> None:
    plan = solve([], budget=10.0, quota_limits={})
    assert isinstance(plan, Plan)
    assert plan.counts == {}


def test_max_count_caps_a_candidate():
    a = Candidate(
        key="capped", region="r", gpu="G", instance_type="t", nodes=1,
        gpus_per_node=8, vcpus_per_island=96, price_per_hour=1.0,
        eff_tflops=100.0, quota_bucket=None, score=9, max_count=1,
    )
    plan = solve([a], budget=100.0, quota_limits={}, max_islands=16, head_cost=0.0)
    assert plan.counts == {"capped": 1}


def _tcand(key, tflops, price, **kw):
    return Candidate(
        key=key, region="r", gpu="G", instance_type="t-" + key, nodes=1,
        gpus_per_node=8, vcpus_per_island=96, price_per_hour=price,
        eff_tflops=tflops, quota_bucket=None, score=9, **kw,
    )


def test_target_mode_minimizes_cost_then_maximizes_tflops():
    a = _tcand("a", 100.0, 10.0)  # 10 TFLOPs/$
    b = _tcand("b", 60.0, 5.0)  # 12 TFLOPs/$
    plan = solve([a, b], budget=None, quota_limits={}, head_cost=0.0, target_tflops=100.0)
    # 2xB reaches 120 for the same $10 as 1xA's 100: cost ties, TFLOPs wins.
    assert plan.counts == {"b": 2}
    assert plan.total_cost == pytest.approx(10.0)


def test_target_mode_with_budget_cap():
    a = _tcand("a", 100.0, 10.0)
    plan = solve([a], budget=10.5, quota_limits={}, head_cost=0.5, target_tflops=150.0)
    # Reaching 150 needs 2xA ($20) but the cap allows only $10 of islands.
    assert plan.counts == {}
    assert plan.binding == ["target-flops"]


def test_target_mode_unreachable_reports_binding():
    a = _tcand("a", 10.0, 1.0)
    plan = solve([a], budget=None, quota_limits={}, max_islands=3, head_cost=0.0, target_tflops=1000.0)
    assert plan.counts == {} and plan.binding == ["target-flops"]


def test_target_mode_brute_force_cross_check():
    import itertools
    import random

    rng = random.Random(1)
    for _ in range(25):
        cands = [
            _tcand(f"c{i}", rng.uniform(10, 500), rng.uniform(0.5, 30))
            for i in range(rng.randint(3, 5))
        ]
        max_islands = 5
        target = rng.uniform(50, 900)
        best = None  # (cost, -tflops)
        for combo in itertools.product(range(max_islands + 1), repeat=len(cands)):
            if sum(combo) > max_islands:
                continue
            tflops = sum(x * c.eff_tflops for x, c in zip(combo, cands))
            if tflops + 1e-9 < target:
                continue
            cost = sum(x * c.price_per_hour for x, c in zip(combo, cands))
            key = (round(cost, 9), -round(tflops, 9))
            if best is None or key < best:
                best = key
        plan = solve(cands, budget=None, quota_limits={}, max_islands=max_islands, head_cost=0.0, target_tflops=target)
        if best is None:
            assert plan.counts == {}
        else:
            assert (round(plan.total_cost, 9), -round(plan.total_tflops, 9)) == best
