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
