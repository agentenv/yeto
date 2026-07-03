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
    def offerings(regions, gpus, cache, clouds=("aws",)):
        out = [o for o in OFFERINGS if not gpus or o.gpu in gpus]
        out = [o for o in out if o.cloud in clouds]
        if regions is not None:
            out = [o for o in out if o.cloud != "aws" or o.region in regions]
        return out

    monkeypatch.setattr(plan_mod, "list_offerings", offerings)
    monkeypatch.setattr(plan_mod, "model_weights_gb", lambda model, override, cache: override or 66.0)
    return FakeAws(QUOTAS, SCORES, CODES)


def _shape(fake, budget, regions=US_REGIONS, price_margin=0.0, clouds=("aws",), **kw):
    return plan_mod.build_shape(
        model="gemma4",
        budget=budget,
        regions=regions,
        cache_enabled=False,
        providers=fake,
        price_margin=price_margin,
        clouds=list(clouds),
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


def test_aggregate_recheck_assumes_when_score_unavailable(fake_env):
    # Default policy: an unfetchable aggregate score is assumed best-case
    # with a warning; the plan keeps both islands.
    scores = {k: v for k, v in SCORES.items() if k != ("p5.48xlarge", 2, "us-east-2")}
    fake = FakeAws(QUOTAS, scores, CODES)
    result = _shape(fake, budget=80.0)
    assert result.plan.counts["aws:8xh100@us-east-2"] == 2
    assert any("score unavailable" in w and "assumed 10" in w for w in result.warnings)


def test_strict_aggregate_recheck_caps_when_score_unavailable(fake_env):
    # --strict-capacity-check restores the conservative behavior: degrade to
    # the verified capacity rather than trust an unknown.
    scores = {k: v for k, v in SCORES.items() if k != ("p5.48xlarge", 2, "us-east-2")}
    fake = FakeAws(QUOTAS, scores, CODES)
    result = _shape(fake, budget=80.0, strict_capacity_check=True)
    assert result.plan.counts["aws:8xh100@us-east-2"] == 1
    assert any("capped at 1 island" in w for w in result.warnings)


def test_unavailable_single_island_score_assumed_by_default(fake_env):
    # p4de's score is missing entirely: default plans it with an assumed
    # score of 10 (goodput 0.98) and a warning; strict rejects it.
    scores = {k: v for k, v in SCORES.items() if k[0] != "p4de.24xlarge"}
    fake = FakeAws(QUOTAS, scores, CODES)
    result = _shape(fake, budget=20.0, gpus=["A100-80GB"])
    (c,) = result.plan.counts
    cand = next(x for x in result.candidates if x.key == c)
    assert cand.assumed and cand.score is None
    assert cand.eff_tflops == pytest.approx(8 * 312 * 0.35 * 0.98)
    assert any("assumed 10" in w and "--strict-capacity-check" in w for w in result.warnings)
    text = plan_mod.render(result, "gemma4", 20.0, "lora")
    assert "~10 (assumed)" in text

    strict = _shape(fake, budget=20.0, gpus=["A100-80GB"], strict_capacity_check=True)
    assert strict.plan.counts == {}
    reasons = {r.key: r.reason for r in strict.rejections}
    assert "unavailable (--strict-capacity-check)" in reasons["aws:8xa100-80gb@us-west-2"]


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


def test_skip_capacity_check_makes_zero_score_calls(fake_env):
    result = _shape(fake_env, budget=80.0, skip_capacity_check=True)
    assert fake_env.score_asks == []  # not one config consumed
    # Quota + budget still apply; even the score-1 p4d shape is plannable now.
    assert result.plan.counts  # a plan exists
    assert all(c.score is None for c in result.candidates)
    assert any("capacity checks skipped" in w for w in result.warnings)


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
        plan_mod, "list_offerings", lambda regions, gpus, cache, clouds=("aws",): OFFERINGS + extra
    )
    fake = FakeAws(QUOTAS, SCORES, {**CODES, "p5.4xlarge": "L-417A185B", "p3.16xlarge": "L-7212CCBC", "p5e.48xlarge": "L-417A185B"})
    result = _shape(fake, budget=40.0)
    asked_types = {t for t, _ in fake.score_asks}
    assert "p5.4xlarge" not in asked_types and "p3.16xlarge" not in asked_types
    assert "p5e.48xlarge" not in asked_types  # $90 island > $39.6 usable budget
    reasons = {r.key: r.reason for r in result.rejections}
    assert "dominated by a fatter-node island" in reasons["aws:2x1xh100@us-east-2"]
    assert "exceeds the budget" in reasons["aws:8xh200@us-east-2"]


def test_quota_dead_shapes_never_consume_score_asks(fake_env):
    # Placement-score configurations are a scarce daily resource: a shape
    # that quota already rules out must be filtered BEFORE the score wave.
    fake = FakeAws(QUOTAS, SCORES, CODES, usage={("us-east-2", "L-417A185B"): 384.0})
    _shape(fake, budget=500.0)
    assert "p5.48xlarge" not in {t for t, _ in fake.score_asks}
    # Sanity: quota-alive shapes still get their asks.
    assert ("p4de.24xlarge", 1) in fake.score_asks


def test_score_ask_budget_caps_queries(fake_env, monkeypatch):
    monkeypatch.setattr(plan_mod, "MAX_SCORE_ASKS", 1)
    result = _shape(fake_env, budget=40.0)
    # Only the best-TFLOPs/$ shape (H100) got its ask; the rest are rejected
    # explicitly, not silently.
    assert len(set(fake_env.score_asks)) == 1
    reasons = {r.key: r.reason for r in result.rejections}
    assert any("daily config budget" in v for v in reasons.values())


def test_unmapped_quota_code_fails_closed(fake_env):
    # An instance type with no quota mapping must be rejected, not treated
    # as unlimited (the bug that once planned 16 P5 islands on 128 vCPUs).
    codes = {k: v for k, v in CODES.items() if k != "p5.48xlarge"}
    fake = FakeAws(QUOTAS, SCORES, codes)
    result = _shape(fake, budget=500.0, gpus=["H100"])
    assert result.plan.counts == {}
    reasons = {r.key: r.reason for r in result.rejections}
    assert "no quota mapping for p5.48xlarge" in reasons["aws:8xh100@us-east-2"]


RUNPOD_OFFERINGS = [
    Offering("H100", "8x_H100_SECURE", 8, 128, "CA", 19.12, 23.12, 80, cloud="runpod"),
    Offering("B200", "8x_B200_SECURE", 8, 224, "CA", 43.92, 47.12, 180, cloud="runpod"),
]


class FakeRunPod:
    def __init__(self, stock):
        self._stock = stock  # {(gpu, gpus_per_node): pseudo-score | None}
        self.warnings = []
        self.asks = []

    def stock_scores(self, asks):
        self.asks.extend(asks)
        return {a: self._stock.get(a) for a in asks}


@pytest.fixture()
def multi_cloud_env(monkeypatch):
    def offerings(regions, gpus, cache, clouds=("aws",)):
        out = [o for o in OFFERINGS + RUNPOD_OFFERINGS if not gpus or o.gpu in gpus]
        out = [o for o in out if o.cloud in clouds]
        if regions is not None:
            out = [o for o in out if o.cloud != "aws" or o.region in regions]
        return out

    monkeypatch.setattr(plan_mod, "list_offerings", offerings)
    monkeypatch.setattr(plan_mod, "model_weights_gb", lambda model, override, cache: override or 66.0)
    return FakeAws(QUOTAS, SCORES, CODES)


def test_runpod_high_stock_competes_without_quota_or_score_asks(multi_cloud_env):
    rp = FakeRunPod({("H100", 8): 9, ("B200", 8): 9})
    result = _shape(
        multi_cloud_env, budget=500.0, clouds=("aws", "runpod"), runpod_providers=rp
    )
    assert ("H100", 8) in rp.asks  # stock was consulted
    assert not any(t.startswith("8x_") for t, _ in multi_cloud_env.score_asks)  # no AWS score asks for pods
    keys = set(result.plan.counts)
    assert any(k.startswith("runpod:") for k in keys)
    # RunPod candidates carry no quota bucket.
    rp_cand = next(c for c in result.candidates if c.cloud == "runpod")
    assert rp_cand.quota_bucket is None


def test_runpod_stock_levels_gate_and_cap(multi_cloud_env):
    # Medium stock (6) fails the default >7 gate...
    rp = FakeRunPod({("H100", 8): 6, ("B200", 8): 0})
    result = _shape(multi_cloud_env, budget=500.0, clouds=("aws", "runpod"), runpod_providers=rp)
    reasons = {r.key: r.reason for r in result.rejections}
    assert "stock score 6 ≤ 7" in reasons["runpod:8xh100@CA"]
    assert "sold out" in reasons["runpod:8xb200@CA"]
    # ...but passes at --min-score 5 with the Medium cap of 4 islands.
    rp = FakeRunPod({("H100", 8): 6, ("B200", 8): 0})
    result = _shape(
        multi_cloud_env, budget=5000.0, clouds=("aws", "runpod"),
        runpod_providers=rp, min_score=5, gpus=["H100"],
    )
    assert result.plan.counts.get("runpod:8xh100@CA") == 4


def test_runpod_unfetchable_stock_follows_score_policy(multi_cloud_env):
    rp = FakeRunPod({})  # every ask -> None
    result = _shape(multi_cloud_env, budget=500.0, clouds=("aws", "runpod"), runpod_providers=rp)
    assumed = [c for c in result.candidates if c.cloud == "runpod" and c.assumed]
    assert assumed  # planned optimistically with a warning
    strict = _shape(
        multi_cloud_env, budget=500.0, clouds=("aws", "runpod"),
        runpod_providers=rp, strict_capacity_check=True,
    )
    reasons = {r.key: r.reason for r in strict.rejections}
    assert "stock score unavailable" in reasons["runpod:8xh100@CA"]


def test_runpod_multi_node_islands_rejected(multi_cloud_env):
    rp = FakeRunPod({("H100", 8): 9})
    result = _shape(
        multi_cloud_env, budget=500.0, clouds=("runpod",),
        runpod_providers=rp, weights_gb_override=568.0, gpus=["H100"],
    )
    reasons = {r.key: r.reason for r in result.rejections}
    assert "multi-node islands unsupported on runpod" in reasons["runpod:2x8xh100@CA"]


def test_runpod_launch_key_parses_in_gpu_grammar(multi_cloud_env):
    from yeto.gpu_spec import parse_gpu_spec

    rp = FakeRunPod({("H100", 8): 9, ("B200", 8): 9})
    result = _shape(multi_cloud_env, budget=500.0, clouds=("aws", "runpod"), runpod_providers=rp)
    rp_keys = [k for k in result.plan.counts if k.startswith("runpod:")]
    assert rp_keys
    (spec,) = parse_gpu_spec(rp_keys[0])
    assert spec.cloud == "runpod" and spec.gpus_per_node == 8


def test_target_flops_mode_minimizes_cost(fake_env):
    # 700 TFLOPs is reachable by one p4de island ($15) — far cheaper than
    # the H100 island ($34) that budget mode would pick.
    result = _shape(fake_env, budget=None, target_tflops=700.0)
    assert result.plan.counts == {"aws:8xa100-80gb@us-west-2": 1}
    text = plan_mod.render(result, "gemma4", None, "lora", target_tflops=700.0)
    assert "target ≥ 700 TFLOPs" in text


def test_objective_required(fake_env):
    with pytest.raises(ValueError, match="--budget and/or --flops"):
        _shape(fake_env, budget=None)
