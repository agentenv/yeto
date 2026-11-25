"""Capacity broker phase 2 (unified-sampler-pool 6.5): a node an island no longer needs
moves to an island short of nodes (drain the old lane, load the new release, start a new
lane) without a new order or release, and is in the order ledger once. Off by default."""

from __future__ import annotations

from dataclasses import dataclass, field

from yeto.launch.pool import broker as b


def nc(island: str, **kw) -> b.Offer:
    return b.Offer(pool=f"nc-{island}", name="nc", island=island, tier=b.PREPAID, provider="nc", region="us",
                   gpu="MI355X", capacity=3, usd_h=20.0, tokens_per_s=1000.0, cold_start_s=1680.0,
                   life_s=14400.0 - 1680.0, **kw)


@dataclass
class FakeFleet:
    """Two islands' pools on one provider: each pool's members by ticket, a provider whose
    orders and releases are written to one ledger, and lanes per ticket."""

    members: dict[str, list[str]] = field(default_factory=dict)  # pool -> tickets
    lane: dict[str, str] = field(default_factory=dict)  # ticket -> island
    release: dict[str, str] = field(default_factory=dict)  # ticket -> island release loaded
    ledger: list[tuple[str, str]] = field(default_factory=list)  # (op, ticket)
    targets: dict[str, int] = field(default_factory=dict)
    steps: list[tuple[str, str, str]] = field(default_factory=list)
    fail: str | None = None
    seq: int = 0

    def order(self, pool: str) -> str:
        self.seq += 1
        ticket = f"t{self.seq}"
        self.ledger.append(("order", ticket))
        self.members.setdefault(pool, []).append(ticket)
        self.lane[ticket] = pool.split("-", 1)[1]
        return ticket

    # the sink: a pool reaches its target by its own rules (orders or releases)
    def set_target(self, pool, nodes, *, offers, refill, running=None):
        self.targets[pool] = nodes
        have = self.members.setdefault(pool, [])
        while len(have) < nodes:
            self.order(pool)
        while len(have) > nodes:
            self.ledger.append(("release", have.pop()))
        return True

    # the LaneMover
    def drain_lane(self, ticket, island):
        self.steps.append(("drain_lane", ticket, island))
        if self.fail == "drain_lane":
            raise RuntimeError("fence failed")
        assert self.lane.pop(ticket) == island

    def load_release(self, ticket, island):
        self.steps.append(("load_release", ticket, island))
        if self.fail == "load_release":
            raise RuntimeError("release missing")
        self.release[ticket] = island

    def start_lane(self, ticket, island):
        self.steps.append(("start_lane", ticket, island))
        old = next(p for p, ts in self.members.items() if ticket in ts)
        self.members[old].remove(ticket)
        self.members.setdefault(f"nc-{island}", []).append(ticket)
        self.lane[ticket] = island

    def running(self) -> dict[str, int]:
        return {f"{p}/nc": len(ts) for p, ts in self.members.items()}

    def nodes(self, lease_end: float = 20000.0) -> list[b.Node]:
        return [b.Node(t, f"{p}/nc", p.split("-", 1)[1], lease_end=lease_end)
                for p, ts in self.members.items() for t in ts]


def snapshot(fleet: FakeFleet, demands, ended=()):
    return lambda: b.Snapshot(offers=[nc("0"), nc("1")], demands=demands, running=fleet.running(),
                              nodes=fleet.nodes(), ended=tuple(ended))


def test_an_ended_island_hands_its_nodes_to_another_without_a_new_order():
    fleet = FakeFleet()
    for _ in range(3):
        fleet.order("nc-1")
    fleet.order("nc-0")
    orders_before = list(fleet.ledger)
    events = []
    cfg = b.BrokerConfig(migrate=True)
    loop = b.Broker(cfg, snapshot(fleet, [b.Demand("0", want=3)], ended=["1"]), fleet,
                    emit=lambda name, **kw: events.append((name, kw)), mover=fleet, clock=lambda: 0.0)
    dec = loop.tick()
    assert dec.pools == {"nc-0": 3, "nc-1": 0}
    assert fleet.ledger[:4] == orders_before and [op for op, _ in fleet.ledger[4:]] == ["release"], \
        "no machine ordered: two nodes changed lanes; the third has nowhere to go and its pool releases it"
    assert [s[0] for s in fleet.steps] == ["drain_lane", "load_release", "start_lane"] * 2
    moved = [kw["ticket"] for name, kw in events if name == "broker_migrated"]
    assert len(moved) == 2 and all(fleet.release[t] == "0" and fleet.lane[t] == "0" for t in moved)
    assert sorted(fleet.members["nc-0"]) == sorted(["t4", *moved]) and fleet.members["nc-1"] == []
    tickets = [t for op, t in fleet.ledger if op == "order"]
    assert len(tickets) == len(set(tickets)) == 4, "each machine is in the ledger once"
    assert fleet.targets == {"nc-0": 3, "nc-1": 0}


def test_migration_is_off_by_default():
    fleet = FakeFleet()
    for _ in range(2):
        fleet.order("nc-1")
    loop = b.Broker(b.BrokerConfig(), snapshot(fleet, [b.Demand("0", want=2)], ended=["1"]), fleet,
                    mover=fleet, clock=lambda: 0.0)
    loop.tick()
    assert fleet.steps == []
    assert [op for op, _ in fleet.ledger] == ["order", "order", "order", "order", "release", "release"], \
        "without phase 2 the new island orders and the old one releases"


def test_plan_migrations_rules():
    offers = [nc("0"), nc("1"), b.Offer(pool="sky-0", name="l4", island="0", tier=b.SPOT, provider="sky",
                                        gpu="L4", usd_h=1.0)]
    dec = b.decide(offers, [b.Demand("0", want=2), b.Demand("1", want=0)], running={"nc-1/nc": 2})
    assert dec.targets["nc-0/nc"] == 2
    nodes = [b.Node("a", "nc-1/nc", "1", lease_end=10000.0), b.Node("b", "nc-1/nc", "1", lease_end=500.0)]
    running = {"nc-1/nc": 2}
    assert b.plan_migrations(nodes, offers, dec, running=running) == [], "off unless enabled"
    moves = b.plan_migrations(nodes, offers, dec, running=running, enabled=True, now=0.0, min_lease_s=1800.0)
    assert [(m.ticket, m.to_offer, m.reason) for m in moves] == [("a", "nc-0/nc", "island over target")], \
        "a lease ending within min_lease_s is not moved"
    assert b.plan_migrations(nodes, offers, dec, running=running, enabled=True,
                             compatible=lambda src, dst: False) == []
    paused = [b.Node("a", "nc-1/nc", "1", movable=False)]
    assert b.plan_migrations(paused, offers, dec, running=running, enabled=True) == []


def test_a_failed_step_stops_the_move_and_leaves_the_node_to_its_pool():
    fleet = FakeFleet()
    fleet.order("nc-1")
    fleet.fail = "load_release"
    events = []
    move = b.Migration("t1", "1", "0", "nc-1/nc", "nc-0/nc", "island ended")
    assert not b.run_migration(move, fleet, lambda name, **kw: events.append((name, kw)))
    assert [s[0] for s in fleet.steps] == ["drain_lane", "load_release"]
    assert events[-1][0] == "broker_migrate_failed" and events[-1][1]["step"] == "load_release"
    assert fleet.ledger == [("order", "t1")], "no order, no release from the broker"
