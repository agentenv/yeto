"""End-to-end shape planning against fully faked providers (no network)."""

from __future__ import annotations

import pytest

from yeto.shape import plan as plan_mod
from yeto.shape.catalog import Offering
from yeto.shape.providers import QuotaKey


class FakeAws:
    """Duck-typed AwsProviders: fixed quotas/scores, records asks."""

    def __init__(self, quotas, scores, codes):
        self._quotas = quotas  # {(region, code): limit}
        self._scores = scores  # {(itype, count, region): score}
        self._codes = codes  # {itype: code}

    def quota_code(self, instance_type, use_spot):
        assert use_spot
        return self._codes.get(instance_type)

    def quotas(self, keys):
        return {k: self._quotas.get((k.region, k.code)) for k in keys}

    def placement_scores(self, asks, regions):
        out = {}
        for itype, count in asks:
            for r in regions:
                out[(itype, count, r)] = self._scores.get((itype, count, r))
        return out


OFFERINGS = [
    Offering("A100-80GB", "p4de.24xlarge", 8, 96, "us-west-2", 15.0, 27.4, 80),
    Offering("A100", "p4d.24xlarge", 8, 96, "us-east-2", 8.0, 21.9, 40),
    Offering("H100", "p5.48xlarge", 8, 192, "us-east-2", 34.0, 98.3, 80),
]

QUOTAS = {
    ("us-west-2", "L-7212CCBC"): 128.0,
    ("us-east-2", "L-7212CCBC"): 384.0,
    ("us-east-2", "L-417A185B"): 384.0,
}
SCORES = {
    ("p4de.24xlarge", 1, "us-west-2"): 8,
    ("p4d.24xlarge", 1, "us-east-2"): 1,  # filtered: score too low
    ("p5.48xlarge", 1, "us-east-2"): 9,
    ("p5.48xlarge", 2, "us-east-2"): 8,
}
CODES = {
    "p4de.24xlarge": "L-7212CCBC",
    "p4d.24xlarge": "L-7212CCBC",
    "p5.48xlarge": "L-417A185B",
}


@pytest.fixture()
def fake_env(monkeypatch):
    monkeypatch.setattr(
        plan_mod,
        "list_offerings",
        lambda regions, gpus, cache: [o for o in OFFERINGS if not gpus or o.gpu in gpus],
    )
    monkeypatch.setattr(plan_mod, "model_weights_gb", lambda model, override, cache: override or 66.0)
    return FakeAws(QUOTAS, SCORES, CODES)


def _shape(fake, budget, **kw):
    return plan_mod.build_shape(
        model="gemma4",
        budget=budget,
        regions=["us-west-2", "us-east-2"],
        cache_enabled=False,
        providers=fake,
        **kw,
    )


def test_low_score_candidate_rejected(fake_env):
    result = _shape(fake_env, budget=40.0)
    reasons = {r.key: r.reason for r in result.rejections}
    assert any("score 1" in v for v in reasons.values())
    assert not any(c.gpu == "A100" for c in result.candidates)


def test_budget_picks_best_flops(fake_env):
    # Budget 40 - head 0.4 fits H100 ($34) xor p4de ($15): H100 wins on FLOPs.
    result = _shape(fake_env, budget=40.0)
    assert list(result.plan.counts) == ["aws:8xh100@us-east-2"]
    assert result.plan.total_cost == pytest.approx(34.4)
    # Budget 55 fits both.
    result = _shape(fake_env, budget=55.0)
    assert set(result.plan.counts) == {"aws:8xh100@us-east-2", "aws:8xa100-80gb@us-west-2"}


def test_quota_caps_islands(fake_env):
    # 128 vCPU quota in us-west-2 = one 96-vCPU p4de island even with budget.
    result = _shape(fake_env, budget=500.0, gpus=["A100-80GB"])
    assert result.plan.counts == {"aws:8xa100-80gb@us-west-2": 1}
    assert any(b.startswith("quota:") or "quota" in b for b in result.plan.binding)


def test_multi_node_island_when_model_demands(fake_env):
    # 568 GB lora on 8x80GB: n=1 gives 71+8=79 > 73.6, n=2 gives 43.5 <= 73.6.
    # The 2-node p4de island (192 vCPU) exceeds us-west-2's 128 quota and must
    # be rejected; the H100 pool in us-east-2 (384 vCPU quota) takes it.
    result = _shape(fake_env, budget=80.0, weights_gb_override=568.0, gpus=["A100-80GB", "H100"])
    (key,) = result.plan.counts
    assert key == "aws:2x8xh100@us-east-2"
    c = next(c for c in result.candidates if c.key == key)
    assert c.vcpus_per_island == 384 and c.price_per_hour == pytest.approx(68.0)
    reasons = {r.key: r.reason for r in result.rejections}
    assert "aws:2x8xa100-80gb@us-west-2" in reasons and "vCPUs > quota" in reasons["aws:2x8xa100-80gb@us-west-2"]


def test_launch_line_and_shard(fake_env):
    result = _shape(fake_env, budget=40.0)
    text = plan_mod.render(result, "gemma4", 40.0, "lora", 7)
    assert "yeto launch --gpu aws:8xh100@us-east-2" in text
    assert "--shard fsdp" in text  # 66 GB does not fit a single 80 GB GPU
    assert "rejected:" in text


def test_no_plan_when_budget_below_head(fake_env):
    result = _shape(fake_env, budget=0.3)
    assert result.plan.counts == {}
    assert "budget" in result.plan.binding
