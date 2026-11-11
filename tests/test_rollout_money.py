"""One budget across providers that price in different currencies
(yeto.launch.pool.money): National Compute in ucr, the clouds in USD, in one pool
(MultiProvider) and in the pool's caps. Fake providers only."""

from __future__ import annotations

import pytest

from tests.test_rollout_island_pool import Clock, FakeProvider, make_pool, names, quiet_elastic, ticks
from yeto.launch.nc import cli
from yeto.launch.pool import money
from yeto.launch.pool.provider import MultiProvider, ProviderCaps


def test_conversions_follow_the_cost_layer_rate(monkeypatch):
    monkeypatch.delenv("YETO_NC_USD_PER_UCR", raising=False)
    assert money.to_usd(24_000_000, "ucr") == pytest.approx(24.0)
    assert money.to_ucr(6.32, "usd") == pytest.approx(6_320_000)
    monkeypatch.setenv("YETO_NC_USD_PER_UCR", "2e-6")
    assert money.to_usd(1_000_000, "ucr") == pytest.approx(2.0) and money.to_ucr(2.0, "usd") == pytest.approx(1e6)
    with pytest.raises(ValueError):
        money.to_ucr(1, "eur")


def _priced(name, clock, *, rate, currency):
    p = FakeProvider(name, clock)
    p.caps = ProviderCaps(currency=currency)
    p.hourly_cost = lambda ticket=None: rate
    return p


def test_a_mixed_currency_pool_adds_up_in_one_unit(monkeypatch):
    monkeypatch.delenv("YETO_NC_USD_PER_UCR", raising=False)
    clock = Clock()
    nc = _priced("nc", clock, rate=24_000_000, currency="ucr")  # 24 USD/h
    sky = _priced("sky", clock, rate=6.0, currency="usd")  # 6 USD/h
    multi = MultiProvider({"nc": nc, "sky": sky}, {"nc": [1], "sky": [1]})
    pool, *_ = make_pool(multi, multi.slots(), target=2, clock=clock)
    ticks(pool, 2, clock, step=1.0)
    by = {m.account: m.hourly_ucr for m in pool.live()}
    assert by == {"nc:1": 24_000_000, "sky:1": 6_000_000}  # the USD rate in ucr-equivalents, not 6 ucr
    start = pool.budget.spent_ucr
    clock.t += 3600
    pool.tick()
    assert pool.budget.spent_ucr - start == pytest.approx(30_000_000, rel=1e-3)
    assert pool.budget.spent_usd == pytest.approx(money.to_usd(pool.budget.spent_ucr, "ucr"))


def test_max_run_spend_usd_caps_orders_across_currencies():
    clock = Clock()
    sky = _priced("sky", clock, rate=10.0, currency="usd")
    pool, *_ = make_pool(sky, [1, 2, 3], target=1, clock=clock, hourly_cost=money.to_ucr(10.0, "usd"),
                         elastic=quiet_elastic(max_run_spend_usd=15.0))
    ticks(pool, 2, clock, step=60)
    assert len(pool.live()) == 1
    clock.t += 3600  # $10 spent; one more node hour ($10) passes the $15 cap
    sky.evict(pool.live()[0].ticket)
    ticks(pool, 3, clock, step=60)
    assert any(e["event"].endswith("order_refused") and "max_run_spend_usd" in e.get("why", "")
               for e in pool.events.events), names(pool)


def test_pool_hourly_of_a_usd_provider_config(monkeypatch):
    monkeypatch.delenv("YETO_NC_USD_PER_UCR", raising=False)
    sky = {"provider": {"kind": "sky", "offers": [{"resources": {"infra": "nebius"}, "price_usd_h": 6.32}]}}
    assert cli.pool_price(sky) == (6.32, "usd") and cli.pool_hourly_ucr(sky) == pytest.approx(6_320_000)
    assert cli.pool_hourly_text(sky, 6_320_000) == "$6.32"
    assert cli.pool_price({"provider": {"kind": "modal", "price_usd_h": 50}}) == (50.0, "usd")
    assert cli.pool_price({}) == (None, "ucr") and cli.pool_hourly_ucr({}) == 0  # the MI355X rate applies
