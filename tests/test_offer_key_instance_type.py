"""The generated offer key names the instance type (two offers of one region and GPU on two
instance types are two offers), and the history counted under the key an older yeto gave
an offer (the key without the instance type) stays readable where it names that offer alone:
the offer ledger's stats, the sky provider's cooling breakers and its cluster records."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_nc_sky_provider import FakeSky
from tests.test_offer_ledger_lifetime import Store
from yeto.launch.nc import pool_run
from yeto.launch.pool.sky_provider import SkyProvider
from yeto.launch.pool import offer_ledger as ol
from yeto.launch.pool import status as el
from yeto.launch.rollout import offers

AWS = {"infra": "aws/us-east-1", "accelerators": "H100:8", "use_spot": True}
TYPE_A = dict(AWS, instance_type="p5.48xlarge")
TYPE_B = dict(AWS, instance_type="p5e.48xlarge")
OLD = "H100:8@aws/us-east-1/spot"  # what an older yeto called both


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("YETO_RESOURCE_LEDGER", raising=False)


def _offers(*resources):
    return offers.offers_from_config({"offers": [{"resources": r, "price_usd_h": 30.0} for r in resources]})


def test_two_instance_types_of_one_region_and_gpu_are_two_offers():
    a, b = _offers(TYPE_A, TYPE_B)
    assert a.key == "H100:8@aws/us-east-1/p5.48xlarge/spot" and b.key == "H100:8@aws/us-east-1/p5e.48xlarge/spot"
    # no instance type: the key is byte for byte the one before
    assert offers._key(AWS) == OLD  # noqa: SLF001
    assert offers._key({"infra": "nebius/eu-north1", "accelerators": "H100:8", "use_spot": True}) \
        == "H100:8@nebius/eu-north1/spot"  # noqa: SLF001
    assert offers._key({"cloud": "aws", "region": "us-west-2", "instance_type": "g6.xlarge"}) \
        == "?@aws/us-west-2/g6.xlarge/od"  # noqa: SLF001
    assert [c.key for c in offers.candidates()] == [o.key for o in offers.offers_from_config(
        {"candidates": {"gpus": ["H100", "H200"], "infra": "nebius"}})]  # the candidate list is unchanged


def test_the_older_key_is_an_alias_only_where_it_named_this_offer_alone():
    [one] = _offers(TYPE_A)
    assert offers.legacy_key(one) == OLD and offers.legacy_aliases([one]) == {one.key: OLD}
    a, b = _offers(TYPE_A, TYPE_B)
    assert offers.legacy_aliases([a, b]) == {}  # both were OLD: that history is mixed, nobody's
    bare, typed = _offers(AWS, TYPE_A)
    assert offers.legacy_aliases([bare, typed]) == {}  # OLD is still the bare offer's own key
    named = offers.offers_from_config({"offers": [{"resources": TYPE_A, "key": "my-p5"}]})
    assert offers.legacy_key(named[0]) is None and offers.legacy_aliases(named) == {}  # a stated key never changed
    [plain] = _offers({"infra": "nebius/eu-north1", "accelerators": "H100:8"})
    assert offers.legacy_key(plain) is None


def _stats(name, ready, start_s):
    row = ol.OfferStats(name, orders=ready, ready=ready, start_s=list(start_s))
    return ol.OfferLedger(offers={name: row})


def test_the_ledger_reads_an_offers_history_under_its_older_key_with_its_own():
    [one] = _offers(TYPE_A)
    led = ol.PoolLedger(_stats(one.key, 1, [600.0]), writer="w", aliases=offers.legacy_aliases([one]))
    led.others = _stats(OLD, 3, [1200.0, 1300.0, 1400.0])  # an older yeto's runs, under OLD
    got = led.stats(one.key)
    assert got.offer == one.key and got.ready == 4 and sorted(got.start_s) == [600.0, 1200.0, 1300.0, 1400.0]
    # two instance types that shared OLD: neither takes its history
    a, b = _offers(TYPE_A, TYPE_B)
    mixed = ol.PoolLedger(ol.OfferLedger(), writer="w", aliases=offers.legacy_aliases([a, b]))
    mixed.others = _stats(OLD, 3, [1200.0])
    assert mixed.stats(a.key) is None and mixed.stats(b.key) is None
    assert mixed.stats(OLD).ready == 3  # still there under its own name


def test_a_sky_pool_attaches_its_ledger_with_the_aliases_and_counts_new_events_under_the_new_key():
    from yeto.launch.pool.island_pool import IslandPool, PoolConfig

    p = SkyProvider("run-k", slots=[1], offers=_offers(TYPE_A), sky=FakeSky())
    store = Store()
    _stats(OLD, 2, [900.0, 1100.0]).save(store, ol.shard_key("older-run-s"))  # an older yeto's shard
    events = el.EventLog(log=lambda _m: None)
    pool = IslandPool(PoolConfig(target=1, accounts=[1], shard_base="s"), ops=SimpleNamespace(provider=p),
                      log=lambda _m: None, events=events)
    cfg = {"shard_base": "s", "run": "r", "run_prefix": "r", "provider": {"kind": "sky"}}
    sink = pool_run.attach_offer_ledger(cfg, pool, store=store, background=False)
    assert sink.aliases == {p.offers[0].key: OLD}
    events("ordered", ticket="t1", offer=p.offers[0].key)
    events("member_ready", ticket="t1", order_to_ready_s=700)
    assert sink.stats(p.offers[0].key).ready == 3
    own = ol.OfferLedger.load(store, ol.shard_key("r-s"))
    assert own.get(p.offers[0].key).ready == 1 and own.get(OLD) is None  # written under the new key only


def test_the_sky_provider_reads_an_older_records_breaker_and_cluster_offer():
    t = [1000.0]
    a, bare = _offers(TYPE_A, dict(AWS, infra="aws/us-west-2"))
    p = SkyProvider("run-k", slots=[1, 2], offers=[a, bare], sky=FakeSky(), clock=lambda: t[0])
    doc = p._doc()  # noqa: SLF001 - the record an older yeto left: its breaker and cluster under OLD
    doc["offers"] = {OLD: {"fails": 1, "open_until": t[0] + 600}}
    doc["clusters"]["old-cluster"] = {"slot": 1, "offer": {"key": OLD, "resources": AWS, "price_usd_h": 30.0}}
    p._save_doc(doc)  # noqa: SLF001
    assert p._cooling(p._doc(), a.key, t[0])  # noqa: SLF001 - still cooling under its older key
    assert p._pick(p._doc()).key == bare.key  # noqa: SLF001 - so the next order takes the other offer
    assert p.offer_of("old-cluster") == a.key  # an older cluster counts as the offer's current key
    p._offer_failed(doc := p._doc(), OLD)  # noqa: SLF001 - its next failure carries the count on
    assert OLD not in doc["offers"] and doc["offers"][a.key]["fails"] == 2
    p._save_doc(doc)  # noqa: SLF001
    assert p.offer_breakers() == {a.key: doc["offers"][a.key]}
    t[0] += 10_000
    assert not p._cooling(p._doc(), a.key, t[0])  # noqa: SLF001


def test_a_launch_of_an_older_cluster_clears_the_older_breaker_too():
    [a] = _offers(TYPE_A)
    p = SkyProvider("run-k", slots=[1], offers=[a], sky=FakeSky())
    doc = p._doc()  # noqa: SLF001
    doc["offers"] = {OLD: {"fails": 1, "open_until": 0}}
    doc["clusters"]["old-cluster"] = {"slot": 1, "offer": {"key": OLD, "resources": AWS, "price_usd_h": 30.0}}
    p._save_doc(doc)  # noqa: SLF001
    p._launch_ok("old-cluster")  # noqa: SLF001
    assert p.offer_breakers() == {}


def test_two_instance_types_keep_their_own_breakers():
    t = [1000.0]
    a, b = _offers(TYPE_A, TYPE_B)
    sky = FakeSky()
    p = SkyProvider("run-k", slots=[1, 2], offers=[a, b], sky=sky, clock=lambda: t[0])
    doc = p._doc()  # noqa: SLF001
    p._offer_failed(doc, a.key)  # noqa: SLF001
    p._save_doc(doc)  # noqa: SLF001
    assert p._cooling(p._doc(), a.key, t[0]) and not p._cooling(p._doc(), b.key, t[0])  # noqa: SLF001
    assert p.has_fallback_offer()


def test_breakers_and_a_tickets_offer_read_by_the_current_key():
    t = [1000.0]
    [a] = _offers(TYPE_A)
    p = SkyProvider("run-k", slots=[1], offers=[a], sky=FakeSky(), clock=lambda: t[0])
    doc = p._doc()  # noqa: SLF001
    doc["offers"] = {OLD: {"fails": 2, "open_until": t[0] + 900}, a.key: {"fails": 1, "open_until": t[0] + 60}}
    p._save_doc(doc)  # noqa: SLF001
    assert p.offer_breakers() == {a.key: {"fails": 2, "open_until": t[0] + 900}}  # the later one, by its key
    # a ticket the ledger knows by the older key (ordered by an older yeto) leads from the merged history
    led = ol.PoolLedger(ol.OfferLedger(), writer="w", aliases=offers.legacy_aliases([a]))
    led.own.tickets["old-t"] = {"offer": OLD}
    assert led.offer("old-t") == a.key
    led.others = _stats(OLD, 3, [1200.0, 1300.0, 1400.0])
    assert led.stats(led.offer("old-t")).ready == 3
