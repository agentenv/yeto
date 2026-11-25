"""The capacity broker drives real pools (6.3 follow-up): the broker writes each pool's
count, offer mix and refill offer through ``pool-scale`` (``ScaleFileSink``), the pools
read the plan (``broker_plan``) and order where the broker placed the node. Over the
fake National Compute of test_nc_elastic, extended with offers (a sky-like provider).
Nothing here orders a real machine, runs a subprocess or opens a socket."""

from __future__ import annotations

import json
import subprocess

import pytest

from tests.test_nc_elastic import Clock, Cloud, _pool
from tests.test_nc_pool_run import _cfg
from yeto.launch.nc import cli, pool_run
from yeto.launch.pool import broker as b
from yeto.launch.pool import broker_plan as plans
from yeto.launch.pool.island_pool import DRAINING, FENCING, GONE

DISPENSE = {"backend": "store", "namespace": "run-a/dispense"}
ACCOUNTS = [1, 4, 5, 6]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", ",".join(map(str, ACCOUNTS)))

    def no_subprocess(*a, **k):
        raise AssertionError(f"no subprocess in these tests: {a}")

    monkeypatch.setattr(subprocess, "run", no_subprocess)


class Key:
    def __init__(self, key):
        self.key = key


class OfferCloud(Cloud):
    """A sky-like provider over the fake cloud: offers in cursor order, ``prefer_offer``,
    ``offer_of``, cooling of an offer that failed to order (sold out)."""

    def __init__(self, clock, offers, **kw):
        super().__init__(clock, **kw)
        self.offers = [Key(k) for k in offers]
        self.provider = self  # ops.provider, as NcSshOps / SkyPoolOps carry it
        self.preferred_offer = None
        self.sold_out: set[str] = set()
        self.cool_until: dict[str, float] = {}
        self.ticket_offer: dict[str, str] = {}
        self.tried: list[str] = []

    def prefer_offer(self, key):
        self.preferred_offer = key

    def cooling_offers(self):
        return [k for k, t in self.cool_until.items() if t > self.clock()]

    def has_fallback_offer(self):
        return any(o.key not in self.cooling_offers() for o in self.offers)

    def offer_of(self, ticket):
        return self.ticket_offer.get(ticket)

    def order(self, account):
        cooling = self.cooling_offers()
        pick = next((o.key for o in self.offers if o.key == self.preferred_offer and o.key not in cooling), None)
        pick = pick or next((o.key for o in self.offers if o.key not in cooling), self.offers[0].key)
        self.tried.append(pick)
        if pick in self.sold_out:
            self.cool_until[pick] = self.clock() + 600.0
            raise RuntimeError(f"{pick}: no capacity")
        ticket = super().order(account)
        self.ticket_offer[ticket] = pick
        return ticket


class Fleet:
    """Pools named by tier, their configs on disk (for pool-scale), one clock."""

    def __init__(self, tmp_path):
        self.tmp, self.clock, self.pools, self.cfgs, self.clouds = tmp_path, Clock(), {}, {}, {}

    def add(self, name, cloud, *, accounts=ACCOUNTS):
        d = self.tmp / name
        d.mkdir()
        cfg, path = _cfg(d, target=1, accounts=list(accounts), shard_base=name, sampler={"dispense": DISPENSE})
        pool = _pool(self.clock, cloud, target=0, accounts=accounts)
        pool.scale_path = pool_run.scale_path(cfg)
        pool.state_path = cli.pool_state_path(cfg)
        pool.state_path.parent.mkdir(parents=True, exist_ok=True)
        pool.save(pool.state_path)
        self.pools[name], self.cfgs[name], self.clouds[name] = pool, (cfg, path), cloud
        return pool

    def counted(self, name):
        return [m for m in self.pools[name].members if m.state not in (GONE, DRAINING, FENCING)
                and m.retiring_since is None]

    def offer_counts(self, name):
        cloud = self.clouds[name]
        out: dict[str, int] = {}
        for m in self.counted(name):
            key = cloud.offer_of(m.ticket) if hasattr(cloud, "ticket_offer") else name
            out[key] = out.get(key, 0) + 1
        return out

    def run(self, seconds, loop=None, step=60.0):
        end = self.clock.t + seconds
        while self.clock.t < end:
            for pool in self.pools.values():
                pool.tick()
                pool.save(pool.state_path)
            if loop is not None:
                loop.tick()
            self.clock.t += step


def own(name="own", cap=2):
    return b.Offer(pool=name, name="rack", island="0", tier=b.OWN, provider="ssh", capacity=cap)


def spot(fleet, key, usd, region):
    cloud = fleet.clouds["spot"]
    return b.Offer(pool="spot", name=key, island="0", tier=b.SPOT, provider="sky", region=region, gpu="L4",
                   usd_h=usd, available=key not in cloud.cooling_offers())


def snapshot(fleet, offers, want):
    def snap():
        running = {}
        for name in fleet.pools:
            for key, n in fleet.offer_counts(name).items():
                full = f"{name}/{key}" if hasattr(fleet.clouds[name], "ticket_offer") else \
                    next(o.key for o in offers() if o.pool == name)
                running[full] = running.get(full, 0) + n
        return b.Snapshot(offers=offers(), demands=[b.Demand("0", want=want)], running=running)
    return snap


def test_own_first_spot_then_on_demand_only_after_the_spot_gap(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.add("own", Cloud(fleet.clock))
    spot_cloud = OfferCloud(fleet.clock, ["a", "b"])
    fleet.add("spot", spot_cloud)
    od_cloud = Cloud(fleet.clock)
    fleet.add("od", od_cloud)
    spot_cloud.sold_out = {"a", "b"}  # the spot market is empty

    def offers():
        return [own(), spot(fleet, "a", 1.0, "r1"), spot(fleet, "b", 1.2, "r2"),
                b.Offer(pool="od", name="x", island="0", tier=b.ON_DEMAND, provider="sky", usd_h=5.0)]

    sink = b.ScaleFileSink(fleet.cfgs)
    loop = b.Broker(b.BrokerConfig(interval_s=300, policy=b.Policy(on_demand_after_s=1800)),
                    snapshot(fleet, offers, want=4), sink, clock=fleet.clock)
    fleet.run(600, loop)
    assert len(fleet.clouds["own"].kinds("order")) == 2, "the two own machines first"
    assert spot_cloud.tried[:2] == ["a", "b"], "spot orders try the broker's offer, then the next"
    assert od_cloud.kinds("order") == [], "no on-demand while the spot gap is young"
    fleet.run(1200, loop)
    assert od_cloud.kinds("order") == [], "still under on_demand_after_s"
    gap_since = loop.state.spot_gap_since["0"]
    fleet.run(1200, loop)
    od_orders = od_cloud.kinds("order")
    assert len(od_orders) == 2 and all(t >= gap_since + 1800 for *_, t in od_orders), \
        "on-demand fills the gap only after it lasted on_demand_after_s"
    assert fleet.pools["own"].config.target == 2 and fleet.pools["od"].config.target == 2
    doc = json.loads(pool_run.scale_path(fleet.cfgs["od"][0]).read_text())
    assert doc["source"] == "auto" and doc["broker"]["offers"] == {"od/x": 2}


def test_the_pool_orders_at_the_broker_offer_and_a_small_score_change_moves_nothing(tmp_path):
    fleet = Fleet(tmp_path)
    cloud = OfferCloud(fleet.clock, ["b", "a"])  # the provider's own cursor would pick b
    fleet.add("spot", cloud)
    price = {"a": 1.0, "b": 1.5}

    def offers():
        return [spot(fleet, "a", price["a"], "r1"), spot(fleet, "b", price["b"], "r2")]

    loop = b.Broker(b.BrokerConfig(interval_s=300), snapshot(fleet, offers, want=3), b.ScaleFileSink(fleet.cfgs),
                    clock=fleet.clock)
    fleet.run(1800, loop)
    assert fleet.offer_counts("spot") == {"a": 3}, "every order went to the broker's offer, not the cursor's"
    assert cloud.tried == ["a", "a", "a"]
    assert any(e["event"].endswith("broker_offer") for e in fleet.pools["spot"].events.events)
    orders = len(cloud.kinds("order"))
    price["b"] = 0.95  # b is now 5% better than a
    fleet.run(1800, loop)
    assert len(cloud.kinds("order")) == orders and cloud.kinds("drain") == [], "no machine is moved"
    assert fleet.offer_counts("spot") == {"a": 3}
    # 30% better and a node lost (a natural boundary): its replacement goes to the better offer
    price["b"] = 0.70
    fleet.run(600, loop)  # the broker sees the new price: the refill offer becomes b, nothing moves yet
    assert fleet.offer_counts("spot") == {"a": 3} and len(cloud.kinds("order")) == orders
    lost = fleet.counted("spot")[0]
    cloud.tickets[lost.ticket].update(state="ended", reason="evicted")
    fleet.run(1800, loop)
    assert fleet.offer_counts("spot") == {"a": 2, "b": 1}
    assert cloud.tried[-1] == "b"


def test_a_pool_without_a_plan_orders_as_before(tmp_path):
    fleet = Fleet(tmp_path)
    cloud = OfferCloud(fleet.clock, ["b", "a"])
    pool = fleet.add("spot", cloud)
    assert pool_run.scale_pool(*fleet.cfgs["spot"], 2) == 0  # a person's scale: no broker key
    fleet.run(1800)
    assert pool.broker_plan is None and cloud.preferred_offer is None
    assert cloud.tried == ["b", "b"], "the provider's own cursor"
    assert not any("broker_offer" in e["event"] for e in pool.events.events)


def test_a_smaller_target_retires_members_on_an_offer_over_its_count(tmp_path):
    fleet = Fleet(tmp_path)
    cloud = OfferCloud(fleet.clock, ["a", "b"])
    pool = fleet.add("spot", cloud)
    sink = b.ScaleFileSink(fleet.cfgs)
    assert sink.set_target("spot", 4, offers={"spot/a": 2, "spot/b": 2}, refill="spot/a")
    fleet.run(2400)
    assert fleet.offer_counts("spot") == {"a": 2, "b": 2}
    assert sink.set_target("spot", 3, offers={"spot/a": 2, "spot/b": 1}, refill="spot/a")
    fleet.run(2400)
    assert fleet.offer_counts("spot") == {"a": 2, "b": 1}, "the member retired was one on b"
    assert pool.config.target == 3


def test_read_plan_and_next_offer():
    assert plans.read_plan({"target": 3}) is None
    plan = plans.read_plan({"broker": {"offers": {"p/H100:8@r1/spot": 2, "p/b": 1, "bad": 1, "p/c": -1},
                                       "refill": "p/b"}})
    assert plan == plans.Plan({"H100:8@r1/spot": 2, "b": 1}, "b"), "an offer key may hold '/'"
    assert plans.next_offer(plan, ["b"]) == "H100:8@r1/spot"
    assert plans.next_offer(plan, ["b"], cooling=["H100:8@r1/spot"]) == "b", "cooling: the refill offer"
    assert plans.next_offer(plan, ["b"], cooling=["H100:8@r1/spot", "b"]) is None, "all cooling: the provider picks"
    # with what ran at the decision: an added node goes to its offer, a lost one to the refill offer
    ran = plans.Plan({"a": 3, "b": 1}, "b", running={"a": 2, "b": 1})
    assert plans.next_offer(ran, ["a", "a", "b"]) == "a", "the broker added one on a"
    assert plans.next_offer(ran, ["a", "a", "b"], ordered={"a": 1}) == "b", "already ordered: a loss refills at b"
    assert plans.next_offer(plan, ["H100:8@r1/spot"] * 2 + ["b"]) == "b", "nothing short: the refill offer"
    assert plans.next_offer(plan, [], known=["b"]) == "b", "an offer the provider does not have is ignored"
    assert plans.next_offer(None, []) is None
    assert plans.victim_rank(plan, "b", ["b", "b"]) == 0 and plans.victim_rank(plan, "b", ["b"]) == 1
    assert plans.victim_rank(None, "b", ["b", "b"]) == 1


def test_a_scale_file_another_writer_replaced_is_written_again(tmp_path):
    fleet = Fleet(tmp_path)
    cloud = OfferCloud(fleet.clock, ["b", "a"])
    pool = fleet.add("spot", cloud)

    def offers():
        return [spot(fleet, "a", 1.0, "r1"), spot(fleet, "b", 1.5, "r2")]

    loop = b.Broker(b.BrokerConfig(interval_s=300), snapshot(fleet, offers, want=2), b.ScaleFileSink(fleet.cfgs),
                    clock=fleet.clock)
    fleet.run(600, loop)
    assert pool.broker_plan is not None and fleet.offer_counts("spot") == {"a": 2}
    seq = json.loads(pool_run.scale_path(fleet.cfgs["spot"][0]).read_text())["seq"]
    fleet.run(600, loop)
    assert json.loads(pool_run.scale_path(fleet.cfgs["spot"][0]).read_text())["seq"] == seq, \
        "nothing changed: the file is not rewritten"
    pool.record_target(2, source="broker-move")  # a writer that keeps no broker plan
    assert "broker" not in json.loads(pool_run.scale_path(fleet.cfgs["spot"][0]).read_text())
    fleet.run(600, loop)
    doc = json.loads(pool_run.scale_path(fleet.cfgs["spot"][0]).read_text())
    assert doc["broker"]["offers"] == {"spot/a": 2, "spot/b": 0} and pool.broker_plan is not None
