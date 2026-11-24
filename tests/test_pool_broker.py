"""Capacity broker phase 1 (unified-sampler-pool 6.3): scores, tier order, spread,
hysteresis, the spot gap clock, island floors, the run budget, the slow loop."""

from __future__ import annotations

import math

import pytest

from yeto.cloud import replace as cloud_replace
from yeto.launch.pool import broker as b


class Row:
    """The shape of offer_ledger.OfferStats."""

    def __init__(self, p50=None, p90=None, success=None, hazard=None, tput=None):
        self.start_p50_s, self.start_p90_s, self.start_success_rate, self.hazard_per_h = p50, p90, success, hazard
        self._tput = tput

    def tokens_per_s(self, model):
        return (self._tput or {}).get(model)


class Ledger:
    def __init__(self, rows):
        self.rows = rows

    def get(self, offer):
        return self.rows.get(offer)


class Prices:
    """The shape of yeto.cost.prices.node_usd_h."""

    def __init__(self, table):
        self.table, self.asked = table, []

    def node_usd_h(self, provider, gpu, gpus, *, spot=None, region=None):
        self.asked.append((provider, gpu, gpus, spot, region))
        return self.table.get((provider, gpu, spot))


class Caps:
    def __init__(self, cold_start_s=0.0, lease_s=None, hard_end=None):
        self.cold_start_s, self.lease_s, self.hard_end = cold_start_s, lease_s, hard_end


def offer(pool, name, tier, *, island="0", provider="sky", region="r1", gpu="L4", usd=1.0, cap=None,
          available=True, tput=100.0, cold=600.0, life=3600.0):
    return b.Offer(pool=pool, name=name, island=island, tier=tier, provider=provider, region=region, gpu=gpu,
                   capacity=cap, available=available, usd_h=usd, tokens_per_s=tput, cold_start_s=cold, life_s=life)


# -- the score ------------------------------------------------------------------------


def test_score_is_price_over_throughput_times_availability_via_replace_score():
    o = offer("p", "spot", b.SPOT, usd=6.0, tput=200.0, cold=900.0, life=2700.0)
    s = b.score_offer(o)
    availability = 2700.0 / (2700.0 + 900.0)
    assert s.availability == pytest.approx(availability)
    assert s.score == pytest.approx(6.0 / (200.0 * availability))
    same = cloud_replace.score(cloud_replace.Offer("sky", "r1", "L4", 0, 6.0, has_weights=True),
                               prep_s=900.0, prep_source="x", expected_stay_s=2700.0)
    assert s.score == pytest.approx(same.score / 200.0), "replace.score per useful node hour, over throughput"
    # one offer without a throughput: every offer is ranked per useful node hour
    d = b.decide([o, offer("p", "x", b.SPOT, usd=5.0, tput=None)], [b.Demand("0", want=1)])
    assert d.scores["p/spot"] == pytest.approx(same.score) and d.targets == {"p/spot": 0, "p/x": 1}


def test_score_reads_the_sources_before_the_stated_values():
    o = b.Offer(pool="p", name="nc", island="0", tier=b.PREPAID, provider="nc", gpu="MI355X", gpus=8,
                usd_h=99.0, tokens_per_s=1.0, model="k31")
    prices = Prices({("nc", "MI355X", None): 24.0})
    caps = {"nc": Caps(cold_start_s=3600.0, lease_s=14400.0, hard_end="lease")}
    s = b.score_offer(o, prices=prices, ledger=Ledger({"p/nc": Row(p50=1680.0, tput={"k31": 500.0})}),
                      caps=caps.get)
    avail = (14400.0 - 1680.0) / 14400.0
    assert prices.asked == [("nc", "MI355X", 8, None, None)], "prepaid: spot-ness not asked"
    assert s.usd_h == 24.0 and s.availability == pytest.approx(avail), "NC: 4 h lease minus the measured start"
    assert s.score == pytest.approx(24.0 / (500.0 * avail))
    # no measurement: the capability table's cold start prior
    assert b.score_offer(o, caps=caps.get).availability == pytest.approx((14400.0 - 3600.0) / 14400.0)
    # a hazard cuts the life: (1 - exp(-lam T)) / lam; no hard end: 1 / lam
    lam, span = 1.0 / 3600.0, 7200.0 - 200.0
    modal = b.Offer(pool="m", name="l4", island="0", tier=b.SPOT, provider="modal", gpu="L4", gpus=1,
                    hard_end_s=7200.0, usd_h=0.8)
    row = Row(p50=200.0, hazard=1.0)
    assert b.expected_life_s(modal, row, None) == pytest.approx((1 - math.exp(-lam * span)) / lam)
    assert b.expected_life_s(offer("s", "a", b.SPOT, life=None), Row(hazard=2.0), None) == pytest.approx(1800.0)
    assert b.expected_life_s(offer("s", "a", b.SPOT, life=None), None, None) == b.DEFAULT_LIFE_S
    spot = b.Offer(pool="sky", name="h100", island="0", tier=b.SPOT, provider="sky", cloud="nebius",
                   region="eu-north1", gpu="H100", gpus=8, ledger_key="H100:8@eu-north1/spot")
    prices = Prices({("nebius", "H100", True): 6.32})
    assert b.node_price(spot, prices) == (6.32, "table") and prices.asked == [("nebius", "H100", 8, True, "eu-north1")]
    # no row: the stated price is a second source: used, but flagged in the risks and the decision's notes
    stated = b.Offer(pool="sky", name="h200", island="0", tier=b.SPOT, provider="sky", cloud="nebius", gpu="H200",
                     gpus=8, usd_h=11.0)
    assert b.node_price(stated, prices) == (11.0, "stated")
    assert "not in the price table: stated usd_h 11.0 used" in b.score_offer(stated, prices=prices).risks
    d = b.decide([stated], [b.Demand("0", want=1)], prices=prices)
    assert d.targets == {"sky/h200": 1} and d.notes[0] == "sky/h200: not in the price table: stated usd_h 11.0 used"
    assert all("not in the price" not in r for r in b.score_offer(stated).risks), \
        "no table given (stated prices only): nothing to flag"
    assert b.score_offer(spot, ledger=Ledger({"H100:8@eu-north1/spot": Row(success=0.4)})).risks[-1] == \
        "order success rate 40%", "the ledger is read by the offer's ledger key"
    assert b.score_offer(offer("p", "x", b.SPOT, usd=None)).score is None, "unknown price: not scored"
    own = b.score_offer(offer("p", "own", b.OWN, usd=None))
    assert own.score == 0.0 and own.usd_h == 0.0


def test_the_real_sources_fit_the_protocols():
    from yeto.launch.pool.offer_ledger import OfferLedger

    ledger = OfferLedger.from_doc({"format": 1, "offers": {
        "nc/nc": {"orders": 6, "ready": 5, "start_failed": 1, "ended": 5, "ready_s": 5 * 12000.0,
                  "start_s": [1500.0, 1600.0, 1680.0, 1700.0, 1800.0],
                  "throughput": {"k31": {"tokens": 3.6e6, "s": 3600.0}}}}})
    real = b.real_sources()
    assert real["ledger"] is None
    real["ledger"] = ledger
    nc = b.Offer(pool="nc", name="nc", island="0", tier=b.PREPAID, provider="nc", gpu="MI355X", gpus=8,
                 capacity=2, model="k31")
    spot = b.Offer(pool="sky", name="h100", island="0", tier=b.SPOT, provider="sky", cloud="nebius",
                   region="eu-north1", gpu="H100", gpus=8, tokens_per_s=1000.0)
    s = b.score_offer(nc, **real)
    assert s.usd_h == pytest.approx(24.0) and s.terms["throughput"]["value"] == pytest.approx(1000.0)
    assert s.availability == pytest.approx((14400.0 - 1680.0) / 14400.0)
    assert b.node_price(spot, real["prices"])[0] == pytest.approx(6.32)
    d = b.decide([nc, spot], [b.Demand("0", want=4)], **real)
    assert d.targets == {"nc/nc": 2, "sky/h100": 2}


# -- tier order -----------------------------------------------------------------------


def test_own_machines_first_then_the_rest_by_score():
    offers = [offer("own", "rack", b.OWN, cap=2, usd=None),
              offer("spot", "a", b.SPOT, usd=4.0), offer("spot", "b", b.SPOT, usd=2.0, region="r2"),
              offer("od", "c", b.ON_DEMAND, usd=1.0)]
    d = b.decide(offers, [b.Demand("0", want=6)])
    assert d.targets["own/rack"] == 2, "the 2 idle own machines first"
    assert d.targets["spot/b"] == 4 and d.targets["spot/a"] == 0, "then the best spot score"
    assert d.targets["od/c"] == 0, "on-demand is not used while spot can fill (even when cheaper)"
    assert d.islands == {"0": 6} and d.pools == {"own": 2, "spot": 4, "od": 0}
    assert d.refill == {"own": "own/rack", "spot": "spot/b", "od": "od/c"}


def test_prepaid_before_spot():
    offers = [offer("nc", "nc", b.PREPAID, cap=3, usd=10.0, provider="nc"), offer("sky", "s", b.SPOT, usd=1.0)]
    d = b.decide(offers, [b.Demand("0", want=5)])
    assert d.targets == {"nc/nc": 3, "sky/s": 2}


# -- spread ---------------------------------------------------------------------------


def test_spread_cap_puts_eight_spot_nodes_on_three_places():
    offers = [offer("s", f"o{i}", b.SPOT, region=f"r{i}", usd=1.0 + i / 10) for i in range(4)]
    d = b.decide(offers, [b.Demand("0", want=8)], policy=b.Policy(spread_cap=3))
    placed = {k: n for k, n in d.targets.items() if n}
    assert sum(placed.values()) == 8 and max(placed.values()) <= 3 and len(placed) >= 3
    # two offers of one pool in the same (provider, region, gpu) share the cap
    same = [offer("s", "a", b.SPOT), offer("s", "b", b.SPOT)]
    d = b.decide(same, [b.Demand("0", want=8)], policy=b.Policy(spread_cap=3))
    assert sum(d.targets.values()) == 3 and d.short == {"0": 5}


def test_spot_share_cap():
    offers = [offer("s", "a", b.SPOT, usd=1.0), offer("od", "c", b.ON_DEMAND, usd=5.0)]
    d = b.decide(offers, [b.Demand("0", want=4)], policy=b.Policy(spot_max_frac=0.5))
    assert d.targets == {"s/a": 2, "od/c": 2}, "a shortfall the spot caps leave is no spot gap: on-demand at once"
    d = b.decide([offer("s", "a", b.SPOT), offer("od", "c", b.ON_DEMAND)], [b.Demand("0", want=4)],
                 policy=b.Policy(spread_cap=3))
    assert d.targets == {"s/a": 3, "od/c": 1}
    d = b.decide([offer("s", "a", b.SPOT, cap=2), offer("od", "c", b.ON_DEMAND)], [b.Demand("0", want=4)], now=5.0)
    assert d.targets == {"s/a": 2, "od/c": 0} and d.state.spot_gap_since == {"0": 5.0}, "a full offer is a gap"


# -- hysteresis -----------------------------------------------------------------------


def test_a_slightly_better_score_moves_no_running_node():
    offers = [offer("s", "cur", b.SPOT, usd=1.00), offer("s", "new", b.SPOT, usd=0.95, region="r2")]
    d = b.decide(offers, [b.Demand("0", want=3)], running={"s/cur": 3})
    assert d.targets == {"s/cur": 3, "s/new": 0}, "running nodes keep their place"
    assert d.refill["s"] == "s/cur", "5% better is inside the hysteresis: the refill stays"
    # at a natural boundary (lease end) a 5% better offer still does not win
    d = b.decide(offers, [b.Demand("0", want=3)], running={"s/cur": 3}, at_boundary={"s/cur": 1})
    assert d.targets == {"s/cur": 3, "s/new": 0}
    # 30% better: the node at its boundary goes to the better offer, the others stay
    offers[1] = offer("s", "new", b.SPOT, usd=0.70, region="r2")
    d = b.decide(offers, [b.Demand("0", want=3)], running={"s/cur": 3}, at_boundary={"s/cur": 1})
    assert d.targets == {"s/cur": 2, "s/new": 1}
    assert d.refill["s"] == "s/new"
    d = b.decide(offers, [b.Demand("0", want=3)], running={"s/cur": 3})
    assert d.targets == {"s/cur": 3, "s/new": 0}, "no boundary: even a much better score moves nothing"


def test_a_lower_want_sheds_the_worst_score_first():
    offers = [offer("s", "a", b.SPOT, usd=1.0), offer("od", "c", b.ON_DEMAND, usd=5.0)]
    d = b.decide(offers, [b.Demand("0", want=2)], running={"s/a": 2, "od/c": 2})
    assert d.targets == {"s/a": 2, "od/c": 0}


# -- the spot gap and on-demand ---------------------------------------------------------


def test_on_demand_only_after_the_spot_gap_lasts():
    offers = [offer("s", "a", b.SPOT, available=False), offer("od", "c", b.ON_DEMAND, usd=5.0)]
    policy = b.Policy(on_demand_after_s=1800)
    d = b.decide(offers, [b.Demand("0", want=2)], policy=policy, now=1000.0)
    assert d.targets["od/c"] == 0 and d.short == {"0": 2}
    assert d.state.spot_gap_since == {"0": 1000.0}
    d = b.decide(offers, [b.Demand("0", want=2)], policy=policy, state=d.state, now=2700.0)
    assert d.targets["od/c"] == 0, "1700 s < 1800 s: still no on-demand"
    d = b.decide(offers, [b.Demand("0", want=2)], policy=policy, state=d.state, now=2800.0)
    assert d.targets["od/c"] == 2, "past the wait: on-demand fills the gap"
    # spot is back: the on-demand nodes stay until their boundary, then go to spot
    offers[0] = offer("s", "a", b.SPOT)
    d = b.decide(offers, [b.Demand("0", want=2)], policy=policy, state=d.state, running={"od/c": 2}, now=2900.0)
    assert d.targets == {"s/a": 0, "od/c": 2} and d.state.spot_gap_since == {}
    assert d.refill["od"] == "od/c" and d.refill["s"] == "s/a"
    d = b.decide(offers, [b.Demand("0", want=2)], policy=policy, running={"od/c": 2}, at_boundary={"od/c": 1})
    assert d.targets == {"s/a": 1, "od/c": 1}


def test_a_gap_that_closes_resets_the_clock():
    offers = [offer("s", "a", b.SPOT, available=False), offer("od", "c", b.ON_DEMAND)]
    d = b.decide(offers, [b.Demand("0", want=1)], now=0.0)
    offers[0] = offer("s", "a", b.SPOT)
    d = b.decide(offers, [b.Demand("0", want=1)], state=d.state, now=100.0)
    assert d.state.spot_gap_since == {}
    offers[0] = offer("s", "a", b.SPOT, available=False)
    d = b.decide(offers, [b.Demand("0", want=1)], state=d.state, running={"s/a": 1}, now=5000.0)
    assert d.state.spot_gap_since == {}, "a running node meets the want: no gap"
    d = b.decide(offers, [b.Demand("0", want=1)], state=d.state, now=5000.0)
    assert d.state.spot_gap_since == {"0": 5000.0} and d.targets["od/c"] == 0


def test_no_spot_offer_means_on_demand_at_once():
    d = b.decide([offer("od", "c", b.ON_DEMAND)], [b.Demand("0", want=2)])
    assert d.targets == {"od/c": 2}


# -- islands, floors, budget --------------------------------------------------------------


def test_island_floors_come_before_other_islands_wants():
    offers = [offer("p0", "nc", b.PREPAID, island="0", cap=3, provider="nc", usd=10.0),
              offer("p1", "nc", b.PREPAID, island="1", cap=3, provider="nc", usd=10.0)]
    d = b.decide(offers, [b.Demand("0", want=3), b.Demand("1", want=3, floor=1)], policy=b.Policy(usd_per_h=30.0))
    assert d.islands == {"0": 2, "1": 1}, "island 1 gets its floor even though island 0 comes first"
    assert d.usd_h == pytest.approx(30.0) and d.short == {"0": 1, "1": 2} and d.floor_unmet == {}


def test_the_run_budget_caps_the_hourly_spend():
    offers = [offer("s", "a", b.SPOT, usd=4.0), offer("own", "r", b.OWN, cap=1, usd=None)]
    budget = b.Budget(usd_total=100.0, spent_usd=80.0, hours_left=2.0)  # 10 USD/h left
    d = b.decide(offers, [b.Demand("0", want=5)], budget=budget, policy=b.Policy(usd_per_h=50.0))
    assert d.ceiling_usd_h == pytest.approx(10.0)
    assert d.targets == {"s/a": 2, "own/r": 1}, "own machines cost nothing; 2 spot nodes fit 10 USD/h"
    assert d.usd_h <= 10.0 and any("hourly ceiling" in n for n in d.notes)
    d = b.decide(offers, [b.Demand("0", want=5, floor=4)], budget=budget)
    assert d.floor_unmet == {"0": 1}, "the budget is a hard cap; an unmet floor is reported"
    d = b.decide(offers, [b.Demand("0", want=5)], budget=budget, running={"s/a": 4})
    assert d.targets["s/a"] == 4 and any("above the hourly ceiling" in n for n in d.notes), \
        "running nodes are not stopped by the broker"


def test_an_island_without_demand_keeps_no_new_nodes():
    offers = [offer("p", "a", b.SPOT, island="9")]
    d = b.decide(offers, [b.Demand("0", want=1)], running={"p/a": 2})
    assert d.targets == {"p/a": 0} and any("no demand" in n for n in d.notes)


# -- run config and the loop --------------------------------------------------------------


def test_the_broker_is_off_unless_enabled():
    assert b.config_from_run({}) is None
    assert b.config_from_run({"broker": {"enabled": False, "spread_cap": 3}}) is None
    cfg = b.config_from_run({"broker": {"enabled": True, "spread_cap": 3, "floors": {"0": 1}},
                             "budget": {"usd_per_h": 20}})
    assert cfg.policy.spread_cap == 3 and cfg.policy.usd_per_h == 20.0 and cfg.floors == {"0": 1}
    assert cfg.migrate is False, "phase 2 is off by default"
    for bad, match in (({"broker": {"enabled": True, "spread": 1}}, "unknown broker keys"),
                       ({"broker": {"enabled": True, "hysteresis": 2}}, "hysteresis"),
                       ({"broker": {"enabled": True, "spread_cap": 0}}, "spread_cap"),
                       ({"broker": {"enabled": True, "spread_tiers": ["cheap"]}}, "spread_tiers"),
                       ({"broker": 1}, "must be an object")):
        with pytest.raises(ValueError, match=match):
            b.config_from_run(bad)


class Sink:
    def __init__(self, refuse=()):
        self.calls, self.refuse = [], set(refuse)

    def set_target(self, pool, nodes, *, offers, refill):
        self.calls.append((pool, nodes, dict(offers), refill))
        return pool not in self.refuse


def test_the_slow_loop_sends_changed_counts_only():
    now = [0.0]
    snap = b.Snapshot(offers=[offer("s", "a", b.SPOT), offer("own", "r", b.OWN, cap=1, usd=None)],
                      demands=[b.Demand("0", want=3)])
    sink, events = Sink(refuse={"own"}), []
    loop = b.Broker(b.BrokerConfig(interval_s=300, floors={"0": 1}), lambda: snap, sink,
                    emit=lambda name, **kw: events.append((name, kw)), clock=lambda: now[0])
    dec = loop.tick()
    assert dec.pools == {"s": 2, "own": 1}
    assert sorted(c[:2] for c in sink.calls) == [("own", 1), ("s", 2)]
    assert ("s", 2, {"s/a": 2}, "s/a") in sink.calls
    assert events[0][0] == b.EVENT and events[0][1]["targets"] == {"s/a": 2, "own/r": 1}
    assert loop.tick() is None, "not due before interval_s"
    now[0] = 300.0
    sink.calls.clear()
    loop.tick()
    assert [c[0] for c in sink.calls] == ["own"], "an unchanged count is not sent again; a refused one is"


def test_the_loop_sends_a_new_offer_mix_with_the_same_total():
    offers = [offer("s", "a", b.SPOT, usd=1.0), offer("s", "b", b.SPOT, usd=2.0, region="r2")]
    state = {"running": {}, "at_boundary": {}}
    snap = lambda: b.Snapshot(offers=list(offers), demands=[b.Demand("0", want=2)], **state)  # noqa: E731
    now, sink = [0.0], Sink()
    loop = b.Broker(b.BrokerConfig(interval_s=10), snap, sink, clock=lambda: now[0])
    loop.tick()
    assert sink.calls == [("s", 2, {"s/a": 2, "s/b": 0}, "s/a")]
    offers[0] = offer("s", "a", b.SPOT, usd=1.0, available=False)  # a sold out: same total, other mix
    now[0] = 10.0
    loop.tick()
    assert sink.calls[-1] == ("s", 2, {"s/a": 0, "s/b": 2}, "s/b"), "the mix and the refill changed: sent"
    now[0] = 20.0
    loop.tick()
    assert len(sink.calls) == 2, "nothing changed: nothing sent"
    # only the refill offer changes (a node at its boundary, a much better offer)
    offers[:] = [offer("s", "a", b.SPOT, usd=0.5), offer("s", "b", b.SPOT, usd=2.0, region="r2")]
    state["running"] = {"s/b": 2}
    now[0] = 30.0
    loop.tick()
    assert sink.calls[-1] == ("s", 2, {"s/a": 0, "s/b": 2}, "s/a")


def test_scale_file_sink_writes_an_automatic_scale():
    seen = []

    def scale(cfg, path, target, **kw):
        seen.append((cfg, path, target, kw))
        return 0

    sink = b.ScaleFileSink({"p": ({"run": "r"}, "/x/p.json")}, scale=scale)
    assert sink.set_target("p", 3, offers={"p/a": 3}, refill="p/a")
    assert seen == [({"run": "r"}, "/x/p.json", 3,
                     {"source": "auto", "extra": {"broker": {"offers": {"p/a": 3}, "refill": "p/a"}}})]
    assert not sink.set_target("other", 1, offers={}, refill=None)


def test_through_pool_scale_only_the_total_takes_effect_today(tmp_path, monkeypatch):
    """The pool applies the scale file's target; the offer mix and refill ride along under
    ``broker`` and change nothing yet (TODO with the NodePool split: the pool orders at the
    refill offer). This pins today's behaviour."""
    import json
    import subprocess

    from tests.test_nc_pool_scale import _ready, _scaled_pool
    from yeto.launch.nc import pool_run

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError(a)))
    cfg, path, clock, cloud, pool = _scaled_pool(tmp_path, target=2)
    sink = b.ScaleFileSink({"nc": (cfg, path)})
    orders = len(cloud.kinds("order"))
    assert sink.set_target("nc", 3, offers={"nc/a": 2, "nc/b": 1}, refill="nc/b")
    doc = json.loads(pool_run.scale_path(cfg).read_text())
    assert doc["target"] == 3 and doc["source"] == "auto"
    assert doc["broker"] == {"offers": {"nc/a": 2, "nc/b": 1}, "refill": "nc/b"}
    pool.tick()
    assert pool.config.target == 3 and len(cloud.kinds("order")) == orders + 1
    # same total, another mix: the pool keeps its target and orders nothing
    assert sink.set_target("nc", 3, offers={"nc/a": 0, "nc/b": 3}, refill="nc/b")
    orders = len(cloud.kinds("order"))
    pool.tick()
    assert pool.config.target == 3 and len(cloud.kinds("order")) == orders
    assert len(_ready(pool)) >= 2
