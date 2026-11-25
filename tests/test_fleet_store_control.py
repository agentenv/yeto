"""The store control plane of a pool (FakeS3, fake providers; no cloud): the pool's
single-writer store lock with its fencing token, the state document in the store
with the one-time import of the local state file, and the store shard allocator."""

from __future__ import annotations

import json
import threading

import pytest

from tests.test_rl_object_store import FakeS3
from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute, quiet_elastic
from yeto.cloud.s3_store import S3ObjectStore
from yeto.launch.pool import store_state
from yeto.launch.pool.island_pool import GONE, IslandPool, PoolConfig
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.launch.pool.shard_alloc import ShardAllocator
from yeto.launch.pool.status import EventLog
from yeto.launch.pool.store_lock import LockLost, StoreLock, StoreLockError


def _objects(client: FakeS3 | None = None) -> S3ObjectStore:
    return S3ObjectStore("s3://bkt/runs/r1", client=client or FakeS3())


# -- the lock -------------------------------------------------------------------------

def test_two_controllers_one_lock_holder():
    objects, clock = _objects(), Clock()
    a = StoreLock.acquire(objects, "fleet/pools/p/lock", holder="host-a", clock=clock)
    b = StoreLock.acquire(objects, "fleet/pools/p/lock", holder="host-b", clock=clock)
    assert a is not None and a.token == 1 and b is None
    assert StoreLock.holder_of(objects, "fleet/pools/p/lock", now=clock())["holder"] == "host-a"
    a.release()
    b = StoreLock.acquire(objects, "fleet/pools/p/lock", holder="host-b", clock=clock)
    assert b is not None and b.token == 2


def test_a_lease_run_out_moves_the_lock_and_fences_the_old_holder():
    objects, clock = _objects(), Clock()
    a = StoreLock.acquire(objects, "L", holder="a", ttl_s=600, clock=clock)
    clock.t += 300
    a.renew()
    clock.t += 599
    assert StoreLock.acquire(objects, "L", holder="b", clock=clock) is None  # renewed: still held
    clock.t += 2
    b = StoreLock.acquire(objects, "L", holder="b", clock=clock)
    assert b is not None and b.token == 2
    with pytest.raises(LockLost):
        a.renew()
    with pytest.raises(LockLost):  # sticky
        a.check()


def test_racing_claimers_get_one_token():
    client = FakeS3()
    clock = Clock()
    won = []

    def claim(i):
        lock = StoreLock.acquire(_objects(client), "L", holder=f"h{i}", clock=clock)
        if lock is not None:
            won.append(lock.token)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert won == [1]


def test_a_store_without_atomic_create_is_refused():
    objects = _objects(FakeS3(conditional=False))
    objects.create("probe", b"{}")  # learns the endpoint has no If-None-Match
    with pytest.raises(StoreLockError):
        StoreLock.acquire(objects, "L")


# -- the shard allocator --------------------------------------------------------------

def test_shard_numbers_are_unique_across_controllers_and_hosts():
    client = FakeS3()
    got: list[int] = []
    lock = threading.Lock()

    def take():
        n = ShardAllocator(_objects(client), "smp").allocate()
        with lock:
            got.append(n)

    threads = [threading.Thread(target=take) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == list(range(10))
    # a new control host (a new client, nothing local) continues past every number issued
    again = ShardAllocator(_objects(client), "smp")
    assert again.allocate() == 10 and again.allocate(floor=3) == 11 and again.allocate(floor=20) == 20
    assert ShardAllocator(_objects(client), "other").allocate() == 0  # one scope per shard_base


def test_an_ended_shard_is_recorded_and_never_issued_again():
    objects = _objects()
    alloc = ShardAllocator(objects, "smp")
    n = alloc.allocate()
    assert alloc.mark_ended(f"smp-{n}", why="retired") and not alloc.mark_ended(f"smp-{n}", why="again")
    assert alloc.ended(f"smp-{n}")["why"] == "retired"
    assert alloc.allocate() == n + 1


# -- the state in the store -----------------------------------------------------------

def _pool(provider, clock, *, control=None, doc=None, target=2, slots=(0, 1, 2), **cfg):
    ops = ProviderPoolOps(provider, FakeRoute(), FakeFencer())
    config = PoolConfig(target=target, accounts=list(slots), shard_base="smp", elastic=quiet_elastic(), **cfg)
    return IslandPool.from_doc(doc, config, ops, control=control, clock=clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=clock))


def test_the_local_state_file_is_imported_once_then_the_store_wins(tmp_path):
    objects, clock = _objects(), Clock()
    legacy = tmp_path / "pool-smp.json"
    legacy.write_text(json.dumps({"format": 2, "next_shard": 7, "members": [], "pending": []}))
    control = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    doc = control.state.load(legacy=legacy)
    assert doc["next_shard"] == 7 and control.state.imported_from == str(legacy)
    assert store_state.read_state(objects, "nc")["imported_from"] == str(legacy)
    legacy.write_text(json.dumps({"format": 2, "next_shard": 99, "members": []}))  # an old loop wrote it later
    control.close()
    control = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    assert control.state.load(legacy=legacy)["next_shard"] == 7 and control.state.imported_from is None
    assert json.loads(legacy.read_text())["next_shard"] == 99  # read only: never rewritten by the import


def test_shards_come_from_the_store_allocator_past_the_imported_next_shard():
    objects, clock = _objects(), Clock()
    control = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    provider = FakeProvider("p", clock)
    pool = _pool(provider, clock, control=control, doc={"format": 2, "next_shard": 4, "members": []})
    pool.tick()
    assert sorted(m.shard for m in pool.live()) == ["smp-4", "smp-5"]
    saved = store_state.read_state(objects, "nc")
    assert saved["control"]["token"] == control.lock.token
    assert sorted(m["shard"] for m in saved["members"]) == ["smp-4", "smp-5"]
    assert [d["shard"] for d in control.allocator.issued()] == ["smp-4", "smp-5"]


def test_the_store_mode_has_no_lifelong_shard_limit_but_keeps_the_index_limit():
    objects, clock = _objects(), Clock()
    control = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    provider = FakeProvider("p", clock)
    pool = _pool(provider, clock, control=control, target=1, shard_limit=1, index_limit=3)
    pool.tick()
    first = pool.live()[0]
    provider.evict(first.ticket)
    for _ in range(4):
        clock.t += 60
        pool.tick()
    assert [m.shard for m in pool.live()] == ["smp-1"]  # past shard_limit 1: the allocator decides
    provider.evict(pool.live()[0].ticket)
    for _ in range(4):
        clock.t += 60
        pool.tick()
    provider.evict(pool.live()[0].ticket)
    for _ in range(4):
        clock.t += 60
        pool.tick()
    assert not pool.live()
    assert any("index limit 3" in str(e.get("why")) for e in pool.events.events if e["event"].endswith("order_refused"))


def test_two_controllers_only_the_lock_holder_writes_and_the_other_takes_over_without_a_drain():
    """Controller A holds pool nc; B is refused. A's host stops renewing (lost); once
    the lease runs out B takes the lock with a higher token and continues A's members
    under their shards. A, back, is fenced: it detaches without draining anything."""
    client, clock = FakeS3(), Clock()
    provider = FakeProvider("p", clock)
    a_ctl = store_state.open_pool_control(_objects(client), "nc", "smp", ttl_s=600, clock=clock)
    with pytest.raises(store_state.PoolLocked):
        store_state.open_pool_control(_objects(client), "nc", "smp", clock=clock)
    a = _pool(provider, clock, control=a_ctl)
    a.tick()
    shards = sorted(m.shard for m in a.live())
    assert shards == ["smp-0", "smp-1"]
    clock.t += 601  # A's host hangs: no renew
    b_ctl = store_state.open_pool_control(_objects(client), "nc", "smp", ttl_s=600, clock=clock)
    assert b_ctl.lock.token == a_ctl.lock.token + 1
    b = _pool(provider, clock, control=b_ctl, doc=b_ctl.state.load())
    assert sorted(m.shard for m in b.live()) == shards
    b.tick()
    b._save_now()  # noqa: SLF001 - the controller saves after every tick
    a.tick()  # A is back: fenced by B's token
    assert a.detached and "lock" in a.detached
    assert provider.released == [] and len(provider.orders) == 2  # nothing drained, nothing ordered twice
    saved = store_state.read_state(_objects(client), "nc")
    assert saved["control"]["token"] == b_ctl.lock.token
    assert sorted(m.shard for m in b.live()) == shards and all(m.state != GONE for m in b.live())


def test_run_up_store_control_plane_is_opt_in(tmp_path, monkeypatch):
    """rollout.control_plane "store" builds every pool on the store lock and state; the
    default leaves the local state files alone (the customer path is unchanged)."""
    from tests.test_rollout_entry import _NoThread, _two_pools  # noqa: PLC0415
    from yeto.launch.pool import controller as ctl  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_cfg, _, _ = _two_pools(tmp_path)
    clock = Clock()
    ops = {name: ProviderPoolOps(FakeProvider(name, clock), FakeRoute(), FakeFencer()) for name in ("nc", "modal")}
    with pytest.raises(ValueError, match="in_process"):
        up.rollout_section(dict(run_cfg, rollout={"control_plane": "store"}))
    objects = _objects()
    cfg = dict(run_cfg, controller="in_process", rollout={"control_plane": "store"})
    c = up.Run(cfg).start_controller(ops=ops, thread_cls=_NoThread, control_objects=objects)
    try:
        assert all(pool.control is not None for pool in c.pools.values())
        c.tick()
        assert store_state.read_state(objects, "nc")["members"] and store_state.read_state(objects, "modal")["members"]
    finally:
        c.stop("test end")
    with ctl._ACTIVE_LOCK:  # noqa: SLF001
        assert c not in ctl._ACTIVE  # noqa: SLF001
    plain = up.Run(dict(run_cfg, controller="in_process")).start_controller(ops=ops, thread_cls=_NoThread)
    try:
        assert all(pool.control is None for pool in plain.pools.values())
    finally:
        plain.stop("test end")


# -- review: every action under a confirmed lock; numbers issued but never ordered -------

class _TimeJump(FakeProvider):
    """A status read during which the clock passes the lock's lease and another controller
    takes the pool (a tick that ran longer than lock_ttl_s)."""

    def __init__(self, *a, takeover=None, **kw):
        super().__init__(*a, **kw)
        self.takeover = takeover

    def status(self, slot, ticket):
        if self.takeover is not None:
            go, self.takeover = self.takeover, None
            go()
        return super().status(slot, ticket)


def test_a_tick_longer_than_the_lock_ttl_acts_no_further_once_the_lock_moved():
    client, clock = FakeS3(), Clock()
    provider = _TimeJump("p", clock)
    a_ctl = store_state.open_pool_control(_objects(client), "nc", "smp", ttl_s=600, clock=clock)
    a = _pool(provider, clock, control=a_ctl)
    a.tick()
    first = sorted(m.ticket for m in a.live())
    taken = {}

    def takeover():
        clock.t += 601  # this tick has run past the lease
        taken["b"] = store_state.open_pool_control(_objects(client), "nc", "smp", ttl_s=600, clock=clock)

    provider.evict(first[0])  # the tick would fence, release and order a replacement
    provider.takeover = takeover
    clock.t += 30
    a.tick()
    assert taken["b"].lock.token == 2
    assert a.detached and "mid-tick" in a.detached  # the guard before the first action stopped it
    assert len(provider.orders) == 2 and provider.released == []  # nothing ordered, released or fenced
    names = [e["event"] for e in a.events.events]
    assert not any(n.endswith(("order_failed", "drain_failed", "fence_failed", "tick_failed")) for n in names)


def test_an_unconfirmed_lock_ends_the_tick_without_an_action():
    client, clock = FakeS3(), Clock()
    provider = FakeProvider("p", clock)
    ctl = store_state.open_pool_control(_objects(client), "nc", "smp", clock=clock)
    pool = _pool(provider, clock, control=ctl)
    real = ctl.lock.objects.list

    def down(prefix, **kw):
        if prefix.startswith("fleet/pools/nc/lock"):
            raise ConnectionError("store down")
        return real(prefix, **kw)

    ctl.lock.renewed_at = clock() - 590  # the lease is nearly over: an unanswered check may not act
    pool._renew_lock = lambda: True  # noqa: SLF001 - the renew at the tick start passed just before the outage
    ctl.lock.objects.list = down
    pool.tick()
    assert provider.orders == [] and not pool.detached
    assert any(e["event"].endswith("lock_unconfirmed") for e in pool.events.events)


def test_numbers_issued_but_never_ordered_are_ended_at_the_takeover(tmp_path):
    """A controller issued shard numbers and died (or detached) before any order took them:
    the next controller ends them (allocator record + a data end in the shard), never issues
    them again, and the trainer's fixed shard list no longer waits for them."""
    from yeto.launch.pool.data_path import R2DataPath  # noqa: PLC0415
    from yeto.launch.nc.elastic import ObjectRecords  # noqa: PLC0415
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index  # noqa: PLC0415

    objects, clock = DirectoryObjectStore(tmp_path), Clock()
    provider = FakeProvider("p", clock)
    data_path = R2DataPath(ObjectRecords(objects))
    ops = ProviderPoolOps(provider, FakeRoute(), FakeFencer())
    ops.data_path = lambda: data_path
    ctl = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    config = PoolConfig(target=1, accounts=[0, 1], shard_base="smp", elastic=quiet_elastic())
    pool = IslandPool.from_doc(None, config, ops, control=ctl, clock=clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=clock))
    pool.tick()
    assert [m.shard for m in pool.live()] == ["smp-0"]
    ctl.allocator.allocate(token=ctl.lock.token, base="smp")  # smp-1: issued, then the controller died
    ctl.allocator.allocate(token=ctl.lock.token, base="smp")  # smp-2
    pool._save_now()  # noqa: SLF001
    ctl.lock.release()  # (a detach; a death lets the lease run out)
    again = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    pool2 = IslandPool.from_doc(again.state.load(), config, ops, control=again, clock=clock,
                                log=lambda _m: None, events=EventLog(log=lambda _m: None, clock=clock))
    assert pool2.end_unused_shards() == ["smp-1", "smp-2"]
    assert pool2.end_unused_shards() == []  # once
    assert again.allocator.ended("smp-1")["why"] == "never ordered"
    assert again.allocator.allocate() == 3  # never issued again
    # the trainer launched with smp-0..smp-2 ends once smp-0 ends: it does not wait for the unused two
    writer = FileSampleBuffer(objects, batch_size=1, index=shard_index("smp-0"))
    writer.mark_data_end()
    trainer = FileSampleBuffer(objects, batch_size=1, merge_shards=True, expected_shards=["smp-0", "smp-1", "smp-2"])
    assert trainer.data_ended


def test_a_reserved_number_of_a_failed_order_is_reused_in_process():
    objects, clock = _objects(), Clock()
    ctl = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    provider = FakeProvider("p", clock)
    provider.fail_orders = 1
    pool = _pool(provider, clock, control=ctl, target=1)
    pool.tick()  # the order fails: its number stays reserved
    clock.t += 60
    pool.tick()
    assert [m.shard for m in pool.live()] == ["smp-0"] and [d["n"] for d in ctl.allocator.issued()] == [0]


def test_an_exhausted_index_budget_says_so_once_with_the_counts():
    objects, clock = _objects(), Clock()
    ctl = store_state.open_pool_control(objects, "nc", "smp", clock=clock)
    provider = FakeProvider("p", clock)
    pool = _pool(provider, clock, control=ctl, target=2, index_limit=1)
    for _ in range(3):
        pool.tick()
        clock.t += 60
    found = [e for e in pool.events.events if e["event"].endswith("pool_shard_budget_exhausted")]
    assert len(found) == 1 and (found[0]["used"], found[0]["limit"]) == (1, 1) and found[0]["ended"] == 0


def test_a_store_blip_during_the_drain_releases_every_member_and_is_no_lost_lock():
    """REV #739: a LIST of the lock that fails while the holder's own lease runs is no reason
    to stop: by the lock's definition nobody else holds it. The drain releases everything."""
    client, clock = FakeS3(), Clock()
    provider = FakeProvider("p", clock)
    ctl = store_state.open_pool_control(_objects(client), "nc", "smp", ttl_s=600, clock=clock)
    pool = _pool(provider, clock, control=ctl)
    pool.tick()
    real = ctl.lock.objects.list

    def flaky(prefix, **kw):
        if prefix.startswith("fleet/pools/nc/lock"):
            raise ConnectionError("store blip")
        return real(prefix, **kw)

    ctl.lock.objects.list = flaky
    left = pool.drain_all("run down", wait_s=0, sleep=lambda _s: None)
    assert left == [] and not pool.detached
    assert sorted(provider.released) == sorted(provider.machines)
    assert not any("lock" in str(e.get("why", "")) for e in pool.events.events if e["event"].endswith("detached"))


def test_past_its_own_lease_an_unconfirmed_lock_is_waited_out_never_a_detach():
    client, clock = FakeS3(), Clock()
    provider = FakeProvider("p", clock)
    ctl = store_state.open_pool_control(_objects(client), "nc", "smp", ttl_s=600, clock=clock)
    pool = _pool(provider, clock, control=ctl)
    pool.tick()
    real = ctl.lock.objects.list
    down = {"n": 0}

    def outage(prefix, **kw):
        if prefix.startswith("fleet/pools/nc/lock") and down["n"] < 2:
            down["n"] += 1
            raise ConnectionError("store down")
        return real(prefix, **kw)

    clock.t += 590  # the lease is nearly over: an outage now must not act
    ctl.lock.objects.list = outage
    waits = []
    left = pool.drain_all("run down", wait_s=0, sleep=waits.append)
    assert waits and not pool.detached and left == []  # waited out, then drained
    assert sorted(provider.released) == sorted(provider.machines)


class _SpotProvider(FakeProvider):
    """Orders are accepted (a ticket) and wait in provisioning; ``lose`` ends one before any grant
    (a spot launch that found no capacity: sky's ResourcesUnavailableError)."""

    def order(self, slot):
        ticket = super().order(slot)
        self.machines[ticket]["state"] = "waiting"
        return ticket

    def lose(self, ticket):
        self.machines[ticket].update(state="ended", end_reason="lost")


def _spot_pool(tmp_path, *, control=True, written=()):
    from yeto.launch.nc.elastic import ObjectRecords  # noqa: PLC0415
    from yeto.launch.pool.data_path import R2DataPath  # noqa: PLC0415
    from yeto.rl.engine.object_store import DirectoryObjectStore  # noqa: PLC0415
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index  # noqa: PLC0415

    objects, clock = DirectoryObjectStore(tmp_path), Clock()
    for shard in written:
        FileSampleBuffer(objects, batch_size=1, index=shard_index(shard)).mark_data_end()
    provider = _SpotProvider("p", clock)
    ops = ProviderPoolOps(provider, FakeRoute(), FakeFencer())
    path = R2DataPath(ObjectRecords(objects))
    ops.data_path = lambda: path
    ctl = store_state.open_pool_control(objects, "nc", "smp", clock=clock) if control else None
    config = PoolConfig(target=1, accounts=[0, 1], shard_base="smp", elastic=quiet_elastic())
    pool = IslandPool.from_doc(None, config, ops, control=ctl, clock=clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=clock))
    return pool, provider, clock


def test_a_number_whose_ticket_is_lost_before_any_grant_goes_to_the_next_order(tmp_path):
    """INT round 0: the order went through (a ticket), the spot launch then found no capacity
    before any grant. No sampler started and its shard holds no record: the next order in this
    process takes the same number (a spot storm never burns the pool's tunnel-port indices)."""
    pool, provider, clock = _spot_pool(tmp_path)
    pool.tick()
    [first] = pool.live()
    assert first.shard == "smp-0"
    provider.lose(first.ticket)
    clock.t += 30
    pool.tick()
    pool.tick()
    live = [m for m in pool.live() if m.ticket != first.ticket]
    assert [m.shard for m in live] == ["smp-0"] and len(provider.orders) == 2
    assert any(e["event"].endswith("shard_given_back") for e in pool.events.events)
    assert pool.control.allocator.high() == 0  # no new number was issued


def test_a_lost_ticket_whose_shard_holds_records_keeps_its_number(tmp_path):
    pool, provider, clock = _spot_pool(tmp_path, written=["smp-0"])
    pool.tick()
    [first] = pool.live()
    provider.lose(first.ticket)
    clock.t += 30
    pool.tick()
    pool.tick()
    assert sorted(m.shard for m in pool.live() if m.ticket != first.ticket) == ["smp-1"]
    assert not any(e["event"].endswith("shard_given_back") for e in pool.events.events)


def test_the_local_control_plane_numbers_as_before_after_a_lost_ticket(tmp_path):
    pool, provider, clock = _spot_pool(tmp_path, control=False)
    pool.tick()
    [first] = pool.live()
    provider.lose(first.ticket)
    clock.t += 30
    pool.tick()
    pool.tick()
    assert sorted(m.shard for m in pool.live() if m.ticket != first.ticket) == ["smp-1"]


def test_a_lost_tickets_number_goes_on_only_once_its_release_is_confirmed(tmp_path):
    """REV #779: a provider whose machine could start a sampler on its own: the number is given
    back only after the lost member's release is confirmed (it is GONE)."""
    pool, provider, clock = _spot_pool(tmp_path)
    real, failures = provider.release, {"n": 1}

    def release(slot, ticket):
        if failures["n"]:
            failures["n"] -= 1
            raise RuntimeError("release not confirmed")
        return real(slot, ticket)

    provider.release = release
    pool.tick()
    [first] = pool.live()
    provider.lose(first.ticket)
    clock.t += 30
    pool.tick()
    assert first.state != GONE and pool._reserved_shard is None  # noqa: SLF001 - not yet
    for _ in range(4):
        clock.t += 120
        pool.tick()
    assert first.state == GONE
    assert sorted(m.shard for m in pool.live()) == ["smp-0"]
