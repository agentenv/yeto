"""End-to-end shape planning against fully faked providers (no network)."""

from __future__ import annotations

import pytest

from yeto.shape import plan as plan_mod
from yeto.shape.catalog import Offering
from yeto.shape.providers import QuotaKey


class FakeAws:
    """Duck-typed AwsProviders: fixed quotas/usage/scores, records asks."""

    def __init__(self, quotas, scores, codes, usage=None, warnings=None):
        self._quotas = quotas  # {(region, code): limit}
        self._scores = scores  # {(itype, count, region): score}
        self._codes = codes  # {itype: code}
        self._usage = usage or {}  # {(region, code): vcpus in use}
        self.warnings = list(warnings or [])
        self.score_asks: list[tuple[str, int]] = []

    def quota_code(self, instance_type, use_spot):
        assert use_spot
        return self._codes.get(instance_type)

    def quotas(self, keys):
        return {k: self._quotas.get((k.region, k.code)) for k in keys}

    def quota_usage(self, keys):
        return {k: self._usage.get((k.region, k.code), 0.0) for k in keys}

    def placement_scores(self, asks, regions):
        self.score_asks.extend(asks)
        out = {}
        for itype, count in asks:
            for r in regions:
                out[(itype, count, r)] = self._scores.get((itype, count, r))
        return out


OFFERINGS = [
    Offering("A100-80GB", "p4de.24xlarge", 8, 96, "us-west-2", 15.0, 27.4, 80),
    Offering("A100", "p4d.24xlarge", 8, 96, "us-east-2", 8.0, 21.9, 40),
    Offering("H100", "p5.48xlarge", 8, 192, "us-east-2", 34.0, 98.3, 80),
    Offering("H100", "p5.48xlarge", 8, 192, "eu-west-1", 40.0, 98.3, 80),
]

QUOTAS = {
    ("us-west-2", "L-7212CCBC"): 128.0,
    ("us-east-2", "L-7212CCBC"): 384.0,
    ("us-east-2", "L-417A185B"): 384.0,
    ("eu-west-1", "L-417A185B"): 192.0,
}
SCORES = {
    ("p4de.24xlarge", 1, "us-west-2"): 8,
    ("p4d.24xlarge", 1, "us-east-2"): 1,  # filtered: score too low
    ("p5.48xlarge", 1, "us-east-2"): 9,
    ("p5.48xlarge", 1, "eu-west-1"): 9,
    ("p5.48xlarge", 2, "us-east-2"): 8,
}
CODES = {
    "p4de.24xlarge": "L-7212CCBC",
    "p4d.24xlarge": "L-7212CCBC",
    "p5.48xlarge": "L-417A185B",
}
US_REGIONS = ["us-west-2", "us-east-2"]


@pytest.fixture()
def fake_env(monkeypatch):
    def offerings(regions, gpus, cache):
        out = [o for o in OFFERINGS if not gpus or o.gpu in gpus]
        if regions is not None:
            out = [o for o in out if o.region in regions]
        return out

    monkeypatch.setattr(plan_mod, "list_offerings", offerings)
    monkeypatch.setattr(plan_mod, "model_weights_gb", lambda model, override, cache: override or 66.0)
    return FakeAws(QUOTAS, SCORES, CODES)


def _shape(fake, budget, regions=US_REGIONS, price_margin=0.0, **kw):
    return plan_mod.build_shape(
        model="gemma4",
        budget=budget,
        regions=regions,
        cache_enabled=False,
        providers=fake,
        price_margin=price_margin,
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
    assert result.est_cost == pytest.approx(34.4)
    # Budget 55 fits both.
    result = _shape(fake_env, budget=55.0)
    assert set(result.plan.counts) == {"aws:8xh100@us-east-2", "aws:8xa100-80gb@us-west-2"}


def test_price_margin_shrinks_usable_budget(fake_env):
    # 34 * 1.20 = 40.8 > 39.6 leftover: H100 no longer affordable, p4de is.
    result = _shape(fake_env, budget=40.0, price_margin=0.20)
    assert list(result.plan.counts) == ["aws:8xa100-80gb@us-west-2"]
    # est cost stays at raw prices; solver cost carries the margin.
    assert result.est_cost == pytest.approx(15.4)
    assert result.plan.total_cost == pytest.approx(15.0 * 1.2 + 0.4)


def test_usage_reduces_quota_room(fake_env):
    # 384 vCPU limit with 150 already in use leaves room 234: one 192-vCPU
    # H100 island fits, a second does not.
    fake = FakeAws(QUOTAS, SCORES, CODES, usage={("us-east-2", "L-417A185B"): 150.0})
    result = _shape(fake, budget=500.0, gpus=["H100"])
    assert result.plan.counts == {"aws:8xh100@us-east-2": 1}
    # And with 384 in use there is no room at all; the rejection says so.
    fake = FakeAws(QUOTAS, SCORES, CODES, usage={("us-east-2", "L-417A185B"): 384.0})
    result = _shape(fake, budget=500.0, gpus=["H100"])
    reasons = {r.key: r.reason for r in result.rejections}
    assert "384/384" in reasons["aws:8xh100@us-east-2"]


def test_aggregate_capacity_score_recheck_caps_not_drops(fake_env):
    # Two 1-node H100 islands plan initially; the score at 2-node aggregate
    # capacity is only 3, so the shape is CAPPED at the verified single
    # island (not dropped) and the freed budget buys the p4de island.
    scores = dict(SCORES)
    scores[("p5.48xlarge", 2, "us-east-2")] = 3
    fake = FakeAws(QUOTAS, scores, CODES)
    result = _shape(fake, budget=80.0)
    assert ("p5.48xlarge", 2) in fake.score_asks  # the re-check happened
    assert result.plan.counts == {
        "aws:8xh100@us-east-2": 1,
        "aws:8xa100-80gb@us-west-2": 1,
    }
    assert any("capped at 1 island" in w for w in result.warnings)


def test_aggregate_recheck_caps_when_score_unavailable(fake_env):
    # A throttled/unknown aggregate score must degrade to the verified
    # capacity, never trust the unknown and never drop the shape.
    scores = {k: v for k, v in SCORES.items() if k != ("p5.48xlarge", 2, "us-east-2")}
    fake = FakeAws(QUOTAS, scores, CODES)
    result = _shape(fake, budget=80.0)
    assert result.plan.counts["aws:8xh100@us-east-2"] == 1
    assert any("score unavailable" in w and "capped" in w for w in result.warnings)


def test_aggregate_recheck_passes_when_score_holds(fake_env):
    # SCORES has (p5, 2) = 8 > 7: two H100 islands survive the re-check.
    result = _shape(fake_env, budget=80.0)
    assert result.plan.counts == {"aws:8xh100@us-east-2": 2}


def test_quota_caps_islands(fake_env):
    # 128 vCPU quota in us-west-2 = one 96-vCPU p4de island even with budget.
    result = _shape(fake_env, budget=500.0, gpus=["A100-80GB"])
    assert result.plan.counts == {"aws:8xa100-80gb@us-west-2": 1}
    assert any("quota" in b for b in result.plan.binding)


def test_multi_node_island_when_model_demands(fake_env):
    # 568 GB lora on 8x80GB: n=1 gives 71+8=79 > 73.6, n=2 gives 43.5 <= 73.6.
    # The 2-node p4de island (192 vCPU) exceeds us-west-2's 128 quota and must
    # be rejected; the H100 pool in us-east-2 (384 vCPU quota) takes it.
    result = _shape(fake_env, budget=80.0, weights_gb_override=568.0, gpus=["A100-80GB", "H100"])
    (key,) = result.plan.counts
    assert key == "aws:2x8xh100@us-east-2"
    reasons = {r.key: r.reason for r in result.rejections}
    assert "vCPUs > remaining quota" in reasons["aws:2x8xa100-80gb@us-west-2"]


def test_regions_all_searches_every_catalog_region(fake_env):
    result = _shape(fake_env, budget=200.0, regions=["all"], gpus=["H100"])
    assert {c.region for c in result.candidates} == {"us-east-2", "eu-west-1"}


def test_warnings_surface_in_result_and_render(fake_env):
    fake = FakeAws(QUOTAS, SCORES, CODES, warnings=["placement scores throttled for p5.48xlarge x1"])
    result = _shape(fake, budget=40.0)
    assert result.warnings and "throttled" in result.warnings[0]
    assert "warning: placement scores throttled" in plan_mod.render(result, "gemma4", 40.0, "lora")


def test_launch_argv_and_render_share_the_command(fake_env):
    result = _shape(fake_env, budget=40.0)
    argv = plan_mod.launch_argv(result, "gemma4", "lora", "org/data")
    assert argv[:3] == ["launch", "--gpu", "aws:8xh100@us-east-2"]
    assert "--shard" in argv and argv[argv.index("--shard") + 1] == "fsdp"
    text = plan_mod.render(result, "gemma4", 40.0, "lora", data="org/data")
    assert "yeto " + " ".join(argv) in text


def test_json_dict_shape(fake_env):
    result = _shape(fake_env, budget=40.0)
    d = plan_mod.to_json_dict(result, "gemma4", 40.0, "lora", "org/data")
    assert d["islands"][0]["instance_type"] == "p5.48xlarge"
    assert d["est_cost_per_hour"] == pytest.approx(34.4)
    assert d["launch_argv"][0] == "yeto"
    import json

    json.dumps(d)  # must be JSON-serializable end to end


def test_missing_credentials_is_a_clear_error(fake_env, monkeypatch):
    monkeypatch.setattr(plan_mod, "credentials_available", lambda: False)
    with pytest.raises(RuntimeError, match="credentials"):
        plan_mod.build_shape(model="gemma4", budget=40.0, providers=None)


def test_no_plan_when_budget_below_head(fake_env):
    result = _shape(fake_env, budget=0.3)
    assert result.plan.counts == {}
    assert "budget" in result.plan.binding


def test_dominated_and_unsupported_shapes_never_ask_for_scores(fake_env, monkeypatch):
    # A 1xH100-per-node offering (needs a 2-node island for 66 GB) is
    # dominated by the 8xH100 single-node island; p3 is unsupported by the
    # scores API; an island pricier than the budget is pointless. None of
    # them may consume a score ask.
    extra = [
        Offering("H100", "p5.4xlarge", 1, 16, "us-east-2", 6.0, 12.0, 80),
        Offering("V100", "p3.16xlarge", 8, 64, "us-east-2", 12.0, 24.5, 16),
        Offering("H200", "p5e.48xlarge", 8, 192, "us-east-2", 90.0, 150.0, 141),
    ]
    monkeypatch.setattr(
        plan_mod, "list_offerings", lambda regions, gpus, cache: OFFERINGS + extra
    )
    fake = FakeAws(QUOTAS, SCORES, {**CODES, "p5.4xlarge": "L-417A185B", "p3.16xlarge": "L-7212CCBC", "p5e.48xlarge": "L-417A185B"})
    result = _shape(fake, budget=40.0)
    asked_types = {t for t, _ in fake.score_asks}
    assert "p5.4xlarge" not in asked_types and "p3.16xlarge" not in asked_types
    assert "p5e.48xlarge" not in asked_types  # $90 island > $39.6 usable budget
    reasons = {r.key: r.reason for r in result.rejections}
    assert "dominated by a fatter-node island" in reasons["aws:2x1xh100@us-east-2"]
    assert "exceeds the budget" in reasons["aws:8xh200@us-east-2"]


def test_score_ask_budget_caps_queries(fake_env, monkeypatch):
    monkeypatch.setattr(plan_mod, "MAX_SCORE_ASKS", 1)
    result = _shape(fake_env, budget=40.0)
    # Only the best-TFLOPs/$ shape (H100) got its ask; the rest are rejected
    # explicitly, not silently.
    assert len(set(fake_env.score_asks)) == 1
    reasons = {r.key: r.reason for r in result.rejections}
    assert any("daily config budget" in v for v in reasons.values())
