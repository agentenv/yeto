"""Capacity broker phase 2 end to end (unified-sampler-pool 6.5): a machine moves from one
island's pool to another's without a new order or a release, every prompt of each
island's dispense ledger is handed out once, the offer ledger counts the machine once,
and its hours are billed once. Two real IslandPools over one fake cloud (tickets by pool
prefix, like sky clusters), the real store dispense ledger on local directories, the real
offer ledger, the real broker loop and :class:`yeto.launch.pool.migrate.PoolMover`."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from tests.test_rollout_island_pool import Clock, quiet_elastic
from tests.test_store_dispense_fleet import NS, ROWS, _prompts, _sampler, _spec
from yeto.launch.pool import broker as b
from yeto.launch.pool import controller as rc
from yeto.launch.pool import migrate
from yeto.launch.pool.island_pool import GONE, READY, IslandPool, PoolConfig
from yeto.launch.pool.offer_ledger import OfferLedger, PoolLedger
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.launch.pool.status import EventLog, LeaseStatus, Probe
from yeto.rl.engine import store_dispense as sd

OFFER = "cloud:H100"
RATE = 3600  # ucr per node hour: one ucr per billed second


class Cloud:
    """One cloud account: machines by ticket; every order and release is recorded."""

    def __init__(self, clock: Clock) -> None:
        self.clock, self.machines, self.orders, self.releases = clock, {}, [], []

    def provider(self, prefix: str) -> PrefixProvider:
        return PrefixProvider(self, prefix)


class PrefixProvider:
    """A pool's view of the cloud: it orders ``<prefix>-t<n>`` and lists its own prefix
    (sky's cluster names); status / release / node work on any ticket of the account, so
    it can take a machine another pool ordered (``take_foreign``: any free slot)."""

    def __init__(self, cloud: Cloud, prefix: str) -> None:
        self.cloud, self.prefix = cloud, prefix

    def order(self, slot):
        ticket = f"{self.prefix}-t{len(self.cloud.orders)}"
        self.cloud.orders.append(ticket)
        self.cloud.machines[ticket] = {"slot": slot, "state": "active", "since": self.cloud.clock()}
        return ticket

    def status(self, slot, ticket):
        return LeaseStatus(self.cloud.machines[ticket]["state"])

    def release(self, slot, ticket):
        self.cloud.machines[ticket]["state"] = "ended"
        self.cloud.releases.append(ticket)
        return {"ended": True}

    def node(self, slot, ticket):
        return {"ticket": ticket}

    def balance(self):
        return None

    def account_free(self, slot):
        return True  # the pool skips slots its live members hold

    def run_tickets(self):
        return [(m["slot"], t) for t, m in self.cloud.machines.items()
                if t.startswith(self.prefix + "-") and m["state"] != "ended"]

    def take_foreign(self, account, ticket, free):
        return free[0] if free else None


class Island:
    """One trainer island's data plane: its own dispense ledger (the real store
    dispenser on a local directory) and the samplers its pool starts (the route)."""

    def __init__(self, tmp_path, k: int) -> None:
        self.k, self.run, self.prompts = k, tmp_path / "run", _prompts(tmp_path, name=f"prompts{k}.jsonl")
        self.objects = sd.DirObjects(self.run / f"trainer{k}")
        self.samplers: dict[str, sd.StoreDispensedPrompts] = {}  # shard -> its live sampler
        self.on_node: dict[str, str] = {}  # ticket -> the shard its sampler serves
        self.held: dict[str, list[int]] = {}  # shard -> slots claimed last step, written next step
        self.claims: list[tuple[int, str]] = []  # (slot, shard) of every claim
        self.written: Counter[int] = Counter()
        self.starts: list[tuple[str, str]] = []
        self.loads: list[str] = []

    # the route (yeto.launch.pool.provider.NodeRoute)
    def start(self, member, node, shards):
        assert member.ticket not in self.on_node, "one sampler per node"
        self.starts.append((member.ticket, member.shard))
        spec = _spec(self.run, self.k, member.shard, self.prompts, shared=False)
        self.samplers[member.shard] = _sampler(spec, inc=f"{member.ticket}-{len(self.starts)}")
        self.on_node[member.ticket] = member.shard

    def restart(self, member, node, shards):
        raise AssertionError("no restart in this test")

    def probe(self, member):
        running = member.ticket in self.on_node
        return Probe(node="ok", sampler="running" if running else "absent", joined=running)

    def stop(self, member):
        shard = self.on_node.pop(member.ticket, None)
        if shard is not None:
            self.samplers.pop(shard).close()
            self.held.pop(shard, None)

    def load_release(self, member, node):
        self.loads.append(member.ticket)

    # the fencer (the pool's side of the work contract)
    def fence_shard(self, shard, by):
        return sd.retire(self.objects, NS, shard, reason="fenced")

    fence_incarnation = fence_shard

    def request_drain(self, shard, seq):
        return sd.request_drain(self.objects, NS, shard, seq=seq)

    def drain_answered(self, shard, seq):
        return sd.drain_answered(self.objects, NS, shard, seq)

    def cancel_drain(self, shard):
        sd.request_drain(self.objects, NS, shard, on=False)

    # one step of every sampler: write what it claimed last step, then claim one more
    def step(self) -> None:
        for shard, s in list(self.samplers.items()):
            slots = self.held.pop(shard, [])
            if slots:
                s.written(0, slots)
                self.written.update(slots)
            state, got = s.poll(1)
            if state == "items":
                self.claims += [(p, shard) for p in got]
                self.held[shard] = got


def _pool(name: str, island: Island, cloud: Cloud, clock: Clock, *, target: int, state_dir) -> IslandPool:
    ops = ProviderPoolOps(cloud.provider(name), island, island)
    config = PoolConfig(target=target, accounts=[0, 1, 2], shard_base=f"{name}-i{island.k}",
                        hourly_cost=RATE, elastic=quiet_elastic(scale_drain_timeout_s=600.0))
    events = EventLog(log=lambda _m: None, clock=clock)
    pool = IslandPool(config, ops, clock=clock, log=lambda _m: None, events=events)
    state_dir.mkdir(parents=True, exist_ok=True)
    pool.state_path = state_dir / f"pool-{name}.json"  # the local state; the hand-over box sits next to it
    pool.offer_ledger = PoolLedger(own=OfferLedger(), writer=name, default_offer=OFFER, background=False)
    events.sinks.append(pool.offer_ledger)
    return pool


def _offer(pool: str, island: str) -> b.Offer:
    return b.Offer(pool=pool, name="h100", island=island, tier=b.SPOT, provider="cloud", region="r1", gpu="H100",
                   capacity=3, usd_h=2.0, tokens_per_s=1000.0, cold_start_s=1680.0, life_s=36000.0)


class Sink:
    """The broker's counts reach each pool's target (what the scale file does in a run)."""

    def __init__(self, pools):
        self.pools, self.sent = pools, []

    def set_target(self, pool, nodes, *, offers, refill, running=None):
        self.sent.append((pool, nodes))
        p = self.pools[pool]
        if nodes < p.config.target:  # as apply_scale does for a lowered target
            p.shrink = max(p.shrink, len(p.counted()) - nodes)
        p.config.target = nodes
        return True


class Fleet:
    def __init__(self, tmp_path, *, migrate_on: bool = True, timeout_s: float = migrate.MOVE_TIMEOUT_S):
        self.clock = Clock()
        self.cloud = Cloud(self.clock)
        self.islands = {"0": Island(tmp_path, 0), "1": Island(tmp_path, 1)}
        self.state_dir = tmp_path / "state"
        self.pools = {"pa": _pool("pa", self.islands["0"], self.cloud, self.clock, target=2, state_dir=self.state_dir),
                      "pb": _pool("pb", self.islands["1"], self.cloud, self.clock, target=1, state_dir=self.state_dir)}
        self.island_of = {"pa": "0", "pb": "1"}
        self.events: list[tuple[str, dict]] = []
        self.slots = {n: rc.PoolSlot(p) for n, p in self.pools.items()}  # the controller's: pool + tick lock
        self.mover = migrate.PoolMover(self.slots, timeout_s=timeout_s, clock=self.clock)
        self.wants = {"0": 2, "1": 1}
        self.broker = b.Broker(b.BrokerConfig(migrate=migrate_on), self.snapshot, Sink(self.pools),
                               emit=lambda name, **kw: self.events.append((name, kw)), mover=self.mover,
                               clock=self.clock)

    def snapshot(self) -> b.Snapshot:
        offers = [_offer("pa", "0"), _offer("pb", "1")]
        running = {f"{n}/h100": len([m for m in p.live() if m.state == READY and not p.moving_out(m)])
                   for n, p in self.pools.items()}
        nodes = migrate.nodes_from_pools(self.pools, island_of=self.island_of, offer_of={"pa": "h100", "pb": "h100"})
        return b.Snapshot(offers=offers, demands=[b.Demand(k, want=w) for k, w in self.wants.items()],
                          running=running, nodes=nodes)

    def run(self, steps: int, *, dt: float = 30.0, skip: tuple[str, ...] = ()) -> None:
        """``skip``: pools whose controller is down meanwhile (they do not tick)."""
        for _ in range(steps):
            for island in self.islands.values():
                island.step()
            for name, pool in self.pools.items():
                if name not in skip:
                    pool.tick()
            self.clock.t += dt

    def replace(self, name: str, pool: IslandPool) -> None:
        self.pools[name] = pool
        self.slots[name] = rc.PoolSlot(pool)

    def moved(self) -> str:
        return next(kw["ticket"] for name, kw in self.events if name == "broker_migrated")

    def ready(self, name: str) -> list:
        return [m for m in self.pools[name].live() if m.state == READY]


def _no_slot_held_twice(island: Island) -> None:
    """At most one live holder per slot at a time: every claim of a slot after the first
    follows a release of it (a fence, a closed sampler), and no slot is written twice."""
    assert all(n == 1 for n in island.written.values()), f"a slot written twice: {island.written}"
    by_slot: dict[int, list[str]] = {}
    for slot, shard in island.claims:
        by_slot.setdefault(slot, []).append(shard)
    for slot, holders in by_slot.items():
        assert len(holders) == 1 or island.written[slot] <= 1, (slot, holders)


def test_a_machine_moves_to_another_island_without_a_new_order_and_is_counted_once(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    assert len(fleet.ready("pa")) == 2 and len(fleet.ready("pb")) == 1
    orders = list(fleet.cloud.orders)
    assert len(orders) == 3

    fleet.wants = {"0": 1, "1": 2}  # island 1 is short, island 0 has one machine to spare
    dec = fleet.broker.tick(force=True)
    assert dec.pools == {"pa": 1, "pb": 2}
    moves = [kw for name, kw in fleet.events if name == "broker_migrated"]
    assert len(moves) == 1 and moves[0]["from_island"] == "0" and moves[0]["to_island"] == "1"
    ticket = moves[0]["ticket"]
    pa, pb = fleet.pools["pa"], fleet.pools["pb"]
    assert pa.config.target == 1 and pb.config.target == 2
    assert pb.moves_expected() == 1, "island 1 holds a place for the machine on its way: it orders nothing"

    fleet.run(20)
    assert fleet.cloud.orders == orders, "no machine was ordered for the move"
    assert fleet.cloud.releases == [], "and none released"
    moved_from = next(m for m in pa.members if m.ticket == ticket)
    assert moved_from.state == GONE and moved_from.end_reason == "moved to pb-i1"
    moved_to = next(m for m in pb.live() if m.ticket == ticket)
    assert moved_to.state == READY and moved_to.shard.startswith("pb-i1-")
    assert len(fleet.ready("pa")) == 1 and len(fleet.ready("pb")) == 2
    assert fleet.islands["1"].loads == [ticket], "the new island's release is loaded on the running machine"
    assert [t for t, _s in fleet.islands["0"].starts].count(ticket) == 1
    assert [t for t, _s in fleet.islands["1"].starts].count(ticket) == 1
    assert pa.moves.get("out", {}).get(ticket, {}).get("state") == "taken" and not pb.moves

    # the work contract: the old lane's sampler is retired, the new one claims from the new island
    old_shard = moved_from.shard
    names = [(e["event"].removeprefix("nc_elastic_"), e.get("shard")) for e in pa.events.events]
    assert names.index(("scale_draining", old_shard)) < names.index(("shard_fenced", old_shard)) \
        < names.index(("member_moved_out", old_shard)), "drained, then fenced, then handed over"
    assert old_shard not in fleet.islands["0"].samplers and ticket not in fleet.islands["0"].on_node
    fleet.run(4 * ROWS)
    for island in fleet.islands.values():
        _no_slot_held_twice(island)
        assert sorted(island.written) == list(range(ROWS)), "every prompt of the island trained once"
    assert any(shard == moved_to.shard for _p, shard in fleet.islands["1"].claims)

    # the offer ledger (both writers' shards merged, as the broker reads it): three machines
    # ordered and come up once each; the moved one is in flight in its new pool's shard only,
    # with its first ready time (its ready hours count from there, once)
    merged = OfferLedger.merged([pa.offer_ledger.own, pb.offer_ledger.own]).get(OFFER)
    assert (merged.orders, merged.ready, merged.start_failed, merged.reclaimed, merged.ended) == (3, 3, 0, 0, 0)
    assert len(merged.start_s) == 3
    assert ticket not in pa.offer_ledger.own.tickets and ticket in pb.offer_ledger.own.tickets
    first_ready = pa.moves["out"][ticket]["ready_at"]
    assert pb.offer_ledger.own.tickets[ticket]["ready"] == first_ready

    # billed once: the moved machine's hours end in its old pool where they start in the new
    # one, and the two pools' spend is every member's billed window once (RATE: 1 ucr / s)
    assert moved_to.granted == pa.moves["out"][ticket]["at"] == moved_from.charged_to
    windows = [m.charged_to - m.granted for p in (pa, pb) for m in p.members if m.granted is not None]
    assert len(windows) == 4  # three machines, the moved one in two pools
    assert pa.budget.spent_ucr + pb.budget.spent_ucr == pytest.approx(sum(windows))
    first_grant = min(m.granted for m in pa.members if m.ticket == ticket)
    assert moved_from.charged_to - first_grant + moved_to.charged_to - moved_to.granted \
        == pytest.approx(moved_to.charged_to - first_grant), "no hour of the moved machine billed twice"


def test_without_migrate_the_short_island_orders_and_the_other_releases(tmp_path):
    fleet = Fleet(tmp_path, migrate_on=False)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    fleet.run(30)
    assert len(fleet.cloud.orders) == 4 and len(fleet.cloud.releases) == 1
    assert not any(name.startswith("broker_migrat") for name, _ in fleet.events)
    assert not fleet.pools["pa"].moves and not fleet.pools["pb"].moves


def test_a_pool_that_never_moved_writes_the_state_it_wrote_before(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    for pool in fleet.pools.values():
        assert "moves" not in pool.snapshot()


def test_a_machine_nobody_takes_is_released_by_its_old_pool_after_the_timeout(tmp_path):
    fleet = Fleet(tmp_path, timeout_s=300.0)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    pb = fleet.pools["pb"]
    fleet.run(30, skip=("pb",))  # the new pool's controller is down meanwhile
    assert ticket in fleet.cloud.releases, "the old pool released the machine nobody took"
    names = [e["event"].removeprefix("nc_elastic_") for e in fleet.pools["pa"].events.events]
    assert names.index("member_moved_out") < names.index("move_out_abandoned") < names.index("released")
    assert not fleet.pools["pa"].moves, "the released machine's move is forgotten once the cloud no longer lists it"
    fleet.run(2)  # its controller is back: the machine is gone, so is the placeholder
    assert not pb.moves, "the placeholder is dropped: its machine went to the old pool's release"
    assert any(e["event"].endswith("move_in_lost") for e in pb.events.events)
    fleet.run(4)
    assert len(fleet.cloud.orders) == 4, "the short island ordered for the place"


def test_the_box_lets_exactly_one_of_take_and_release_win_and_outlives_its_controller(tmp_path):
    objects = sd.DirObjects(tmp_path / "store")
    box = migrate.MoveBox(objects)
    box.offer("t.1", {"to": "pb-i1", "at": 5.0})
    assert box.claim("t.1", "pb-i1") and box.claim("t.1", "pb-i1")
    assert not box.claim("t.1", "pa-i0") and box.claimed_by("t.1") == "pb-i1"
    again = migrate.MoveBox(objects)  # the next controller
    assert again.get("t.1")["to"] == "pb-i1" and again.claimed_by("t.1") == "pb-i1"
    assert not again.claim("t.1", "pa-i0")
    assert again.claim("t.2", "pa-i0"), "a later move of the same machine has its own claim"


def test_a_store_without_an_atomic_create_holds_no_box(tmp_path):
    objects = sd.DirObjects(tmp_path / "store")
    objects.atomic_create = False  # a check-then-write store (the OSS relay)
    with pytest.raises(Exception, match="atomic"):
        migrate.MoveBox(objects)


def test_moves_outlive_a_pool_restart(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    pa, pb = fleet.pools["pa"], fleet.pools["pb"]
    doc_a, doc_b = pa.snapshot(), pb.snapshot()
    assert doc_a["moves"]["out"] and doc_b["moves"]["in"]
    again = IslandPool.from_doc(doc_b, pb.config, pb.ops, clock=fleet.clock, log=lambda _m: None,
                                events=pb.events)
    assert again.moves == pb.moves and again.moves_expected() == 1


def test_a_lost_machine_on_its_way_out_is_retired_and_the_move_cancelled(tmp_path):
    fleet = Fleet(tmp_path, timeout_s=300.0)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = next(kw["ticket"] for name, kw in fleet.events if name == "broker_migrated")
    fleet.cloud.machines[ticket]["state"] = "ended"  # preempted while draining
    fleet.run(30)
    pa = fleet.pools["pa"]
    assert ticket not in pa.moves.get("out", {})
    assert any(e["event"].endswith("move_out_cancelled") for e in pa.events.events)
    assert len(fleet.cloud.orders) == 4, "the short island ordered once its placeholder ran out"


def test_a_pool_that_cannot_stop_a_sampler_alone_never_moves(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    pa = fleet.pools["pa"]
    del pa.ops.stop_sampler
    assert all(not n.movable for n in migrate.nodes_from_pools(fleet.slots, island_of=fleet.island_of)
               if n.offer.startswith("pa/"))
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    assert not any(name == "broker_migrated" for name, _ in fleet.events)
    assert not pa.moves and not fleet.pools["pb"].moves


def test_on_the_store_control_plane_a_restarted_new_pool_takes_the_machine(tmp_path):
    """The store control plane: shard numbers from the store allocator, the box in the run
    store. The new pool's controller dies after the hand-over; the next one rebuilds the
    pool from its store state (the placeholder is in it) and takes the machine."""
    from tests.test_fleet_store_control import _objects
    from yeto.launch.pool import store_state

    fleet = Fleet(tmp_path)
    objects = _objects()
    for name, pool in fleet.pools.items():
        pool.control = store_state.open_pool_control(objects, name, pool.config.shard_base, clock=fleet.clock)
        pool.__post_init__()  # its ops are guarded by the store lock now
        pool.move_box = None  # the run store's box now
    fleet.mover = migrate.PoolMover(fleet.slots, timeout_s=3600.0, clock=fleet.clock)
    fleet.broker.mover = fleet.mover
    fleet.run(6)
    orders = list(fleet.cloud.orders)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    pb = fleet.pools["pb"]
    fleet.run(8, skip=("pb",))  # its controller is gone before the machine arrives
    assert fleet.pools["pa"].moves["out"][ticket]["state"] == "handed_over"
    doc = store_state.read_state(objects, "pb")
    assert ticket in doc["moves"]["in"], "the placeholder is in the store state"
    pb.control.close()
    control = store_state.open_pool_control(objects, "pb", "pb-i1", clock=fleet.clock)
    again = IslandPool.from_doc(doc, pb.config, pb.ops.__dict__["_ops"], control=control,
                                clock=fleet.clock, log=lambda _m: None, events=pb.events)
    again.offer_ledger = pb.offer_ledger
    fleet.replace("pb", again)  # the next controller: the box is the run store's again
    fleet.run(10)
    moved = next(m for m in again.live() if m.ticket == ticket)
    assert moved.state == READY and fleet.cloud.orders == orders and fleet.cloud.releases == []
    assert moved.shard == "pb-i1-1", "its number came from the store allocator"
    assert fleet.pools["pa"].moves["out"][ticket]["state"] == "taken"


def test_a_slot_held_for_a_machine_moving_in_is_not_ordered_on(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    pb = fleet.pools["pb"]
    slot = pb.expect_move_in("pa-t0", move="pa-t0.1", source="pa-i0", account=pb.live()[0].account)
    assert slot != pb.live()[0].account, "the machine keeps no slot the pool's own member holds"
    assert pb._free_account() not in (slot, pb.live()[0].account)
    pb.cancel_move_in("pa-t0", why="test")
    assert pb.config.target == 1 and not pb.moves


def test_an_old_pool_restored_after_the_take_never_releases_the_machine(tmp_path):
    """The new pool took the machine; the old pool's controller died before it saw that.
    Restored from its saved state, it reads the claim in the box (next to the state files)
    and never releases the machine."""
    fleet = Fleet(tmp_path, timeout_s=300.0)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    pa, pb = fleet.pools["pa"], fleet.pools["pb"]
    for _ in range(40):  # until the old pool handed it over
        fleet.run(1)
        if pa.moves["out"][ticket]["state"] == "handed_over":
            break
    doc = pa.snapshot()  # what the old pool saved before its controller died
    fleet.run(3, skip=("pa",))
    assert any(m.ticket == ticket for m in pb.live()), "the new pool took it"
    restored = IslandPool.from_doc(doc, pa.config, pa.ops, clock=fleet.clock, log=lambda _m: None, events=pa.events)
    restored.state_path, restored.offer_ledger = pa.state_path, pa.offer_ledger
    fleet.replace("pa", restored)
    fleet.clock.t += 3600.0
    fleet.run(4)
    assert ticket not in fleet.cloud.releases
    assert restored.moves.get("out", {}).get(ticket, {}).get("state") in ("taken", None)


def test_the_same_machine_moves_there_and_back(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    fleet.run(20)
    assert any(m.ticket == ticket and m.state == READY for m in fleet.pools["pb"].live())
    fleet.events.clear()
    fleet.wants = {"0": 2, "1": 1}
    fleet.clock.t += 600.0
    fleet.broker.tick(force=True)
    back = [kw["ticket"] for name, kw in fleet.events if name == "broker_migrated"]
    assert len(back) == 1
    fleet.run(20)
    assert fleet.cloud.releases == [] and len(fleet.cloud.orders) == 3
    assert len(fleet.ready("pa")) == 2 and len(fleet.ready("pb")) == 1
    assert any(m.ticket == back[0] for m in fleet.ready("pa"))


def test_a_move_waits_while_a_pool_ticks_and_never_runs_without_its_lock(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    lock = fleet.slots["pb"].tick_lock
    lock.acquire()  # the new pool's worker is in a tick
    try:
        fleet.broker.tick(force=True)
    finally:
        lock.release()
    failed = [kw for name, kw in fleet.events if name == "broker_migrate_failed"]
    assert failed and failed[0]["step"] == "begin" and "MoveBusy" in failed[0]["error"]
    assert not fleet.pools["pa"].moves and not fleet.pools["pb"].moves
    bare = migrate.PoolMover(fleet.pools)  # pools without their tick locks: refused
    move = b.Migration(fleet.pools["pa"].live()[0].ticket, "0", "1", "pa/h100", "pb/h100", "test")
    with pytest.raises(migrate.MoveBusy):
        bare.begin(move)


def test_a_sampler_that_does_not_stop_cancels_the_move_and_its_machine_goes(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    island = fleet.islands["0"]

    def stuck(member):
        raise RuntimeError("docker stop timed out")

    island.stop = stuck
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    fleet.run(30)
    pa = fleet.pools["pa"]
    names = [e["event"].removeprefix("nc_elastic_") for e in pa.events.events]
    assert names.count("move_stop_failed") == migrate.MOVE_STOP_TRIES and "member_moved_out" not in names
    assert ticket in fleet.cloud.releases, "the machine whose sampler would not stop is released"
    assert not any(m.ticket == ticket for m in fleet.pools["pb"].members)


def test_a_new_pool_that_dies_right_after_its_claim_completes_the_take(tmp_path):
    """The claim is won only after "taking" is saved: a restart from that state finds its
    own claim in the box and makes the member (no machine left without a pool)."""
    fleet = Fleet(tmp_path)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    pb = fleet.pools["pb"]
    saved = {}
    real = pb._move_in

    def die_after_claim(t, rec, now):
        saved["doc"] = pb.snapshot()
        raise SystemExit("controller died")

    pb._move_in = die_after_claim
    with pytest.raises(SystemExit):
        for _ in range(40):
            fleet.run(1)
    pb._move_in = real
    assert saved["doc"]["moves"]["in"][ticket]["state"] == "taking"
    again = IslandPool.from_doc(saved["doc"], pb.config, pb.ops, clock=fleet.clock, log=lambda _m: None,
                                events=pb.events)
    again.state_path, again.offer_ledger = pb.state_path, pb.offer_ledger
    fleet.replace("pb", again)
    fleet.run(6)
    assert any(m.ticket == ticket and m.state == READY for m in again.live())
    assert fleet.cloud.releases == [] and len(fleet.cloud.orders) == 3


def test_a_destination_whose_provider_cannot_manage_a_foreign_machine_is_refused(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    del fleet.pools["pb"].ops.take_foreign
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    failed = [kw for name, kw in fleet.events if name == "broker_migrate_failed"]
    assert failed and "did not order" in failed[0]["error"]
    assert not fleet.pools["pa"].moves and not fleet.pools["pb"].moves
    assert fleet.pools["pa"].config.target == 2 or fleet.pools["pa"].shrink > 0


def test_a_restarted_worker_keeps_the_newer_moves_it_held(tmp_path):
    """The store control plane: a worker died after its pool took the machine, before the
    store copy said so. The restore keeps the member the memory holds and drops the
    placeholder, so the machine is never taken twice."""
    from tests.test_fleet_store_control import _objects
    from yeto.launch.pool import store_state

    fleet = Fleet(tmp_path)
    objects = _objects()
    for name, pool in fleet.pools.items():
        pool.control = store_state.open_pool_control(objects, name, pool.config.shard_base, clock=fleet.clock)
        pool.__post_init__()
        pool.move_box = None
    fleet.mover = fleet.broker.mover = migrate.PoolMover(fleet.slots, clock=fleet.clock)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    pb = fleet.pools["pb"]
    before = pb.snapshot()  # the store copy with the placeholder still waiting
    for _ in range(40):
        fleet.run(1)
        if any(m.ticket == ticket for m in pb.members):
            break
    pb.control.state.save(before)  # the store copy is older than the memory
    controller = rc.RolloutController({"pb": fleet.slots["pb"]}, interval=0.02, log=lambda _m: None)
    controller._restore_after_death("pb", pb, pb.control)
    assert [m.ticket for m in pb.members].count(ticket) == 1
    assert ticket not in pb.moves.get("in", {})
    fleet.run(6)
    assert [m.ticket for m in pb.members].count(ticket) == 1 and fleet.cloud.orders[:3] == fleet.cloud.orders


def test_a_handed_over_machine_whose_box_is_gone_is_alerted_and_never_released_blind(tmp_path):
    fleet = Fleet(tmp_path, timeout_s=300.0)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = next(kw["ticket"] for name, kw in fleet.events if name == "broker_migrated")
    pa = fleet.pools["pa"]
    for _ in range(40):
        fleet.run(1, skip=("pb",))
        if pa.moves["out"][ticket]["state"] == "handed_over":
            break
    pa.state_path = None  # its box cannot be opened any more (no state kept, no store)
    pa.move_box = None
    fleet.clock.t += 4 * 3600.0
    fleet.run(30, skip=("pb",))
    assert ticket not in fleet.cloud.releases, "its new pool may hold it: never released blind"
    alerts = [e for e in pa.events.events if e["event"].endswith("move_out_unresolved")]
    assert alerts and alerts[0]["level"] == "critical"
    pa.stop("test end")
    assert ticket not in fleet.cloud.releases, "not at the stop either"


def test_a_store_pool_stopped_from_its_local_mirror_never_releases_a_machine_taken_on_the_store(tmp_path):
    """pool-stop / run down after the controller died rebuild the old pool from its local
    mirror without its store control. That pool would open the local box, which holds no
    claim: the move records the box it began on, so it claims and releases nothing there;
    the machine is reported with the members the stop leaves billed."""
    from tests.test_fleet_store_control import _objects
    from yeto.launch.pool import store_state

    fleet = Fleet(tmp_path)
    objects = _objects()
    for name, pool in fleet.pools.items():
        pool.control = store_state.open_pool_control(objects, name, pool.config.shard_base, clock=fleet.clock)
        pool.__post_init__()
        pool.move_box = None
    fleet.mover = fleet.broker.mover = migrate.PoolMover(fleet.slots, clock=fleet.clock)
    fleet.run(6)
    fleet.wants = {"0": 1, "1": 2}
    fleet.broker.tick(force=True)
    ticket = fleet.moved()
    pa, pb = fleet.pools["pa"], fleet.pools["pb"]
    for _ in range(40):
        fleet.run(1, skip=("pa",) if pa.moves.get("out", {}).get(ticket, {}).get("state") == "handed_over" else ())
        if any(m.ticket == ticket for m in pb.live()):
            break
    assert any(m.ticket == ticket for m in pb.live()) and pa.moves["out"][ticket]["state"] == "handed_over"
    mirror = json.loads(pa.state_path.read_text())  # what pool-stop reads with no controller running
    stray = IslandPool.from_doc(mirror, pa.config, pa.ops.__dict__["_ops"], clock=fleet.clock, log=lambda _m: None,
                                events=pa.events)
    stray.state_path = pa.state_path  # no control: box_of opens the local box, not the store's
    left = stray.drain_all("pool-stop", attempts=1, sleep=lambda _s: None)
    assert ticket not in fleet.cloud.releases, "the machine its new pool holds is not released"
    assert any(m.ticket == ticket for m in pb.live()), "its new pool still holds it"
    assert ticket in [m.ticket for m in left], "listed with the members the stop leaves billed"
    assert any(e["event"].endswith("move_out_unresolved") for e in pa.events.events)


def test_a_move_record_that_does_not_read_is_kept_alerted_and_listed_not_fatal(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(3)
    pa = fleet.pools["pa"]
    doc = pa.snapshot()
    doc["moves"] = {"out": {"x": {"state": "handed_over", "account": 2}, "y": "junk"}, "in": "junk"}
    again = IslandPool.from_doc(doc, pa.config, pa.ops, clock=fleet.clock, log=lambda _m: None, events=pa.events)
    assert sorted(again.moves["out"]) == ["x", "y"] and {r["state"] for r in again.moves["out"].values()} == {"unreadable"}
    assert again.moved_away() == {"x", "y"}, "its patrol neither adopts nor releases these tickets"
    assert again.moves_expected() == 0
    again.tick()
    assert any(e["event"].endswith("move_out_unresolved") for e in again.events.events)
    assert sorted(m.ticket for m in again.moves_left()) == ["?", "x", "y"], "the whole unreadable in is listed too"


def test_a_take_whose_box_is_gone_is_alerted_and_listed(tmp_path):
    fleet = Fleet(tmp_path)
    fleet.run(6)
    pb = fleet.pools["pb"]
    pb.moves = {"in": {"pa-t0": {"move": "pa-t0.1", "state": "taking", "source": "pa-i0", "account": 1,
                                 "since": fleet.clock(), "until": fleet.clock(), "box": "elsewhere"}}}
    fleet.run(2)
    assert any(e["event"].endswith("move_in_unresolved") and e["level"] == "critical" for e in pb.events.events)
    assert "pa-t0" in pb.moves["in"], "kept: it may have claimed the machine"
    assert "pa-t0" in [m.ticket for m in pb.moves_left()]


def test_two_spellings_of_one_state_dir_share_one_box(tmp_path):
    fleet = Fleet(tmp_path)
    pa, pb = fleet.pools["pa"], fleet.pools["pb"]
    (tmp_path / "x").mkdir()
    pb.state_path = tmp_path / "x" / ".." / "state" / pb.state_path.name
    pa.move_box = pb.move_box = None
    assert migrate.box_of(pa).where == migrate.box_of(pb).where


@pytest.mark.parametrize("kind", ["out", "in"])
def test_moves_that_do_not_read_as_a_whole_pause_the_patrol(tmp_path, kind):
    fleet = Fleet(tmp_path)
    fleet.run(3)
    pa = fleet.pools["pa"]
    doc = pa.snapshot()
    doc["moves"] = {kind: ["junk"]}
    again = IslandPool.from_doc(doc, pa.config, pa.ops, clock=fleet.clock, log=lambda _m: None, events=pa.events)
    assert again.moves_unknown() == [kind]
    fleet.cloud.machines["pa-t9"] = {"slot": 2, "state": "active", "since": fleet.clock()}  # an unowned ticket
    for _ in range(5):
        again.tick()
        fleet.clock.t += 10
    assert fleet.cloud.machines["pa-t9"]["state"] == "active" and "pa-t9" not in [m.ticket for m in again.members]
    paused = [e for e in again.events.events if e["event"].endswith("patrol_paused")]
    assert len(paused) == 1 and paused[0]["level"] == "critical", "alerted once, then every UNRESOLVED_ALERT_S"
    fleet.clock.t += migrate.UNRESOLVED_ALERT_S
    again.tick()
    assert len([e for e in again.events.events if e["event"].endswith("patrol_paused")]) == 2
    if kind == "in":
        return
    held = {"out": {"t": {"move": "t.1", "state": "unreadable", "account": 1, "since": 1.0}}}
    again.moves = {"out": {"t": {"move": "t.1", "state": "handed_over", "account": 1, "since": 1.0}}}
    again.merge_moves(held)
    assert again.moves["out"]["t"]["state"] == "handed_over", "a copy that reads wins over an unreadable one"
