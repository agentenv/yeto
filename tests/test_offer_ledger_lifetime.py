"""The offer ledger and the machine lifetime model (hard_end / preempt_at / lambda)."""

from __future__ import annotations

import json

import pytest

from yeto.launch.pool import offer_ledger as ol
from yeto.launch.pool import status as el
from yeto.launch.pool.status import LeaseStatus

# -- the ledger ------------------------------------------------------------------


def _ev(name, t, ticket=None, prefix="nc_elastic_", **kw):
    return {"event": f"{prefix}{name}", "t": t, **({"ticket": ticket} if ticket else {}), **kw}


def test_start_times_success_and_reclaim_rates_follow_the_event_sequence():
    led = ol.OfferLedger()
    events = [
        _ev("ordered", 0, "a", offer="spot"), _ev("ordered", 0, "b", offer="spot"),
        _ev("ordered", 0, "c", offer="spot"), _ev("ordered", 0, "d", offer="spot"),
        _ev("member_ready", 600, "a", order_to_ready_s=600),
        _ev("member_ready", 1200, "b"),  # no order_to_ready_s: ready - ordered
        _ev("member_ready", 1300, "b"),  # a second join of the same machine counts once
        _ev("start_timeout", 3600, "c"),
        _ev("order_failed", 10, prefix="", offer="spot"), _ev("offer_failed", 10, offer="spot"),  # one failure
        _ev("evicted", 600 + 7200, "a"),  # reclaimed after 2 ready hours
        _ev("released", 1200 + 3600, "b"),  # ended normally after 1 ready hour
        _ev("node_lost", 50, "d"),  # lost before it was ever ready: a start failure
        _ev("sampler_restart", 99, "a"),  # not a ledger event
    ]
    counted = [led.observe(e) for e in events]
    assert counted[-1] is None and counted[6] is None
    row = led.get("spot")
    assert (row.orders, row.ready, row.start_failed, row.reclaimed, row.ended) == (4, 2, 3, 1, 1)
    assert row.start_s == [600.0, 1200.0]
    assert row.start_p50_s == pytest.approx(900.0) and row.start_p90_s == pytest.approx(1140.0)
    assert row.start_success_rate == pytest.approx(2 / 5)
    assert row.reclaim_rate == pytest.approx(0.5)
    assert row.hazard_per_h == pytest.approx(1 / 3)  # one reclaim in 3 ready hours
    assert led.tickets == {}


def test_the_offer_comes_from_the_event_the_ticket_or_the_pool_default():
    led = ol.OfferLedger()
    led.observe(_ev("ordered", 0, "t1"), offer_of=lambda t: "H100:8@nebius/spot")
    led.observe(_ev("member_ready", 300, "t1"))  # remembered from its order
    led.observe(_ev("ordered", 0, "t2"), default_offer="nc:mi355x")
    led.observe(_ev("member_ready", 1680, "t2", order_to_ready_s=1680), default_offer="other")
    assert led.get("H100:8@nebius/spot").start_s == [300.0]
    assert led.get("nc:mi355x").start_s == [1680.0] and led.get("other") is None
    assert led.observe(_ev("ordered", 0, "t3")) is None  # no offer known: not counted


def test_throughput_per_model_and_the_ledger_round_trips_through_the_store():
    led = ol.OfferLedger()
    led.record_throughput("nc:mi355x", "kimi-k31", tokens=6000, seconds=2)
    led.record_throughput("nc:mi355x", "kimi-k31", tokens=3000, seconds=1)
    led.observe(_ev("ordered", 0, "x", offer="nc:mi355x"))
    assert led.get("nc:mi355x").tokens_per_s("kimi-k31") == pytest.approx(3000.0)

    class Store(dict):
        def read(self, key):
            return self.get(key)

        def write(self, key, data):
            self[key] = data

    store = Store()
    assert ol.OfferLedger.load(store).offers == {}
    led.save(store)
    assert set(store) == {ol.DEFAULT_KEY}
    back = ol.OfferLedger.load(store)
    assert back.to_doc() == led.to_doc() and back.tickets == {"x": {"offer": "nc:mi355x", "ordered": 0.0}}
    store[ol.DEFAULT_KEY] = json.dumps({"format": ol.FORMAT + 1}).encode()
    with pytest.raises(ValueError, match="newer"):
        ol.OfferLedger.load(store)


def test_start_samples_keep_the_newest():
    led = ol.OfferLedger()
    for i in range(ol.MAX_SAMPLES + 10):
        led.observe(_ev("ordered", 0, f"t{i}", offer="o"))
        led.observe(_ev("member_ready", i, f"t{i}", order_to_ready_s=i))
    assert len(led.get("o").start_s) == ol.MAX_SAMPLES and led.get("o").start_s[0] == 10.0


def test_the_pool_sink_saves_after_a_machine_came_up_or_ended_and_survives_a_store_error():
    saved, logs = [], []

    class Store:
        fail = False

        def read(self, key):
            return None

        def write(self, key, data):
            if self.fail:
                raise OSError("store down")
            saved.append(key)

    store = Store()
    sink = ol.PoolLedger(ol.OfferLedger(), default_offer="nc:mi355x", store=store, log=logs.append)
    sink(_ev("ordered", 0, "a"))
    assert saved == []
    sink(_ev("member_ready", 1700, "a", order_to_ready_s=1700))
    assert saved == [ol.DEFAULT_KEY]
    store.fail = True
    sink(_ev("expired", 9000, "a"))
    sink(_ev("ordered", 9000, "b"))
    sink(_ev("member_ready", 9100, "b", order_to_ready_s=100))
    assert len(logs) == 1 and "not saved" in logs[0]


# -- the lifetime model --------------------------------------------------------------

CFG = el.ElasticConfig()


def test_nc_lease_end_is_an_expiry_and_the_effective_end():
    s = LeaseStatus("active", lease_until=14400.0)
    assert el.effective_end(s) == 14400.0
    assert el.lifetime(s) == el.Lifetime(hard_end=14400.0)
    ended = LeaseStatus("ended", lease_until=14400.0, end_reason="expired")
    assert el.classify(ended, None, now=14401.0, unreachable_since=None, cfg=CFG, joined=True).kind == el.EXPIRED


def test_a_modal_call_at_its_function_timeout_expires_instead_of_being_evicted():
    active = LeaseStatus("active", hard_end=86400.0)
    assert el.effective_end(active) == 86400.0
    timed_out = LeaseStatus("ended", end_reason="lost", hard_end=86400.0)
    assert el.classify(timed_out, None, now=86400.0 + 5, unreachable_since=None, cfg=CFG,
                       joined=True).kind == el.EXPIRED
    # a failure long before its hard end is still a loss
    assert el.classify(timed_out, None, now=3600.0, unreachable_since=None, cfg=CFG,
                       joined=True).kind == el.EVICTED
    # the call is not an overdue lease: hard_end is no lease_until
    probe = el.Probe(node="ok", sampler="running", joined=True)
    assert el.classify(active, probe, now=86400.0 + 600, unreachable_since=None, cfg=CFG,
                       joined=True).kind == el.HEALTHY


def test_a_spot_reclaim_notice_sets_the_end_and_a_loss_without_notice_is_fenced_after_the_grace():
    notice = LeaseStatus("active", preempt_at=1120.0)
    assert el.effective_end(notice) == 1120.0
    assert el.lifetime(notice, hazard_per_h=0.2) == el.Lifetime(preempt_at=1120.0, hazard_per_h=0.2)
    ok = el.Probe(node="ok", sampler="running", joined=True)
    assert el.classify(notice, ok, now=1000.0, unreachable_since=None, cfg=CFG, joined=True).kind == el.PREEMPT_NOTICE
    silent = LeaseStatus("active")
    assert el.effective_end(silent) is None
    gone = el.Probe(node="unreachable")
    assert el.classify(silent, gone, now=100.0, unreachable_since=0.0, cfg=CFG, joined=True).kind == el.NETWORK_BLIP
    assert el.classify(silent, gone, now=CFG.unreachable_grace_s, unreachable_since=0.0, cfg=CFG,
                       joined=True).kind == el.LOST
    # both a lease and a notice: the earlier one
    assert el.effective_end(LeaseStatus("active", lease_until=5000.0, preempt_at=4000.0)) == 4000.0


def test_the_replacement_lead_is_the_offers_p90_plus_margin_once_measured():
    stats = ol.OfferStats("nc:mi355x", start_s=[1500.0 + 10 * i for i in range(20)])  # NC: ~28 min
    lead, why = ol.replace_lead_s(stats, prior_s=3600)
    assert lead == pytest.approx(stats.start_p90_s * 1.25) and "20 measured" in why
    assert lead < 3600
    few = ol.OfferStats("nc:mi355x", start_s=[1500.0] * 4)
    assert ol.replace_lead_s(few, prior_s=3600)[0] == 3600
    assert ol.replace_lead_s(None, prior_s=900)[0] == 900
    fast = ol.OfferStats("ssh", start_s=[60.0] * 10)
    assert ol.replace_lead_s(fast, prior_s=3600)[0] == pytest.approx(60 + ol.LEAD_MARGIN_MIN_S)


def _pool(auto, ledger=None):
    from yeto.launch.pool.island_pool import IslandPool, Member, PoolConfig

    pool = IslandPool(PoolConfig(target=1, accounts=[1, 2], shard_base="s", replace_lead_auto=auto),
                      ops=object(), log=lambda _m: None)
    pool.offer_ledger = ledger
    return pool, Member(account=1, ticket="t1", shard="s-0")


def test_the_pool_reads_its_replacement_lead_from_the_ledger_only_when_asked():
    led = ol.OfferLedger()
    for i in range(20):
        led.observe(_ev("ordered", 0, f"o{i}", offer="nc:mi355x"))
        led.observe(_ev("member_ready", 1680, f"o{i}", order_to_ready_s=1680))
    sink = ol.PoolLedger(led, default_offer="nc:mi355x")
    pool, m = _pool(False, sink)
    assert pool._replace_lead(m) == 3600  # the default: the configured lead (customer path unchanged)
    pool, m = _pool(True, sink)
    assert pool._replace_lead(m) == pytest.approx(1680 + 420)
    pool, m = _pool(True, ol.PoolLedger(ol.OfferLedger(), default_offer="nc:mi355x"))
    assert pool._replace_lead(m) == 3600  # nothing measured yet: the prior


def test_the_modal_provider_reports_its_calls_hard_end():
    from yeto.launch.nc.modal_provider import ModalProvider

    p = ModalProvider.__new__(ModalProvider)
    p.config = {"timeout_h": 2}
    assert p.timeout_s() == 7200.0
    assert p.hard_end({"spawned_at": 1000.0}) == 8200.0
    assert p.hard_end({}) is None
    p.config = {}
    assert p.timeout_s() == 86400.0
