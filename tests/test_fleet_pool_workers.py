"""The rollout controller as a supervisor of one worker per pool (fake providers, real
threads, FakeS3; no cloud): a blocked pool never holds the others, a dead worker is
restarted, releases run in the background on the store control plane, and a detached
controller leaves the samplers running for the next one to adopt under their shards."""

from __future__ import annotations

import threading
import time

import pytest

from tests.test_fleet_store_control import _objects
from tests.test_rl_object_store import FakeS3
from tests.test_rollout_island_pool import Clock, FakeFencer, FakeProvider, FakeRoute, quiet_elastic
from yeto.launch.pool import controller as rc
from yeto.launch.pool import store_state
from yeto.launch.pool.island_pool import DRAINING, GONE, IslandPool, PoolConfig
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.launch.pool.status import EventLog


def _pool(name, provider, clock, *, control=None, target=1, route=None):
    ops = ProviderPoolOps(provider, route or FakeRoute(), FakeFencer())
    config = PoolConfig(target=target, accounts=[0, 1, 2], shard_base=name, elastic=quiet_elastic())
    return IslandPool.from_doc(None, config, ops, control=control, clock=clock, log=lambda _m: None,
                               events=EventLog(log=lambda _m: None, clock=clock))


def _wait_for(check, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return False


def _names(pool):
    return [e["event"].removeprefix("nc_elastic_") for e in pool.events.events]


class HangingProvider(FakeProvider):
    """A provider whose status call hangs while ``gate`` is clear (a cloud API stuck)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = threading.Event()
        self.gate.set()

    def status(self, slot, ticket):
        self.gate.wait(30)
        return super().status(slot, ticket)


def test_a_blocked_pool_never_holds_the_other_pools():
    clock = Clock()
    stuck, fine = HangingProvider("stuck", clock), FakeProvider("fine", clock)
    pools = {"stuck": _pool("stuck", stuck, clock), "fine": _pool("fine", fine, clock)}
    c = rc.RolloutController(pools, interval=0.02, log=lambda _m: None, pool_workers=True, blocked_s=0.2,
                             restart_backoff_s=0)
    c.tick()  # both ordered
    stuck.gate.clear()  # from now on every status read of pool "stuck" hangs
    c.start()
    try:
        assert _wait_for(lambda: c.slots["stuck"].tick_began is not None)
        before = fine.status
        seen = []
        fine.status = lambda slot, ticket: seen.append(ticket) or before(slot, ticket)
        assert _wait_for(lambda: len(seen) >= 5)  # the other pool keeps ticking
        assert _wait_for(lambda: "pool_blocked" in _names(pools["stuck"]))
        assert "blocked_s" in c.status()["pools"]["stuck"] and "blocked_s" not in c.status()["pools"]["fine"]
    finally:
        stuck.gate.set()
        c.stop("test end", join_s=5)
    assert all(not p.live() for p in pools.values())


def test_a_dead_worker_is_restarted():
    clock = Clock()
    provider = FakeProvider("p", clock)
    pool = _pool("p", provider, clock)

    class Boom(BaseException):
        pass

    real, calls = pool.tick, []

    def tick():
        calls.append(1)
        if len(calls) == 1:
            raise Boom("worker bug")
        real()

    pool.tick = tick
    c = rc.RolloutController({"p": pool}, interval=0.02, log=lambda _m: None, pool_workers=True,
                             restart_backoff_s=0)
    c.start()
    try:
        assert _wait_for(lambda: len(calls) >= 3)
        assert c.slots["p"].restarts == 1 and "worker_restarted" in _names(pool)
        assert pool.live()  # the restarted worker ordered
    finally:
        c.stop("test end", join_s=5)


def test_a_restarted_store_worker_reloads_its_state_from_the_store():
    clock = Clock()
    objects = _objects()
    control = store_state.open_pool_control(objects, "p", "p", clock=clock)
    provider = FakeProvider("p", clock)
    pool = _pool("p", provider, clock, control=control)
    c = rc.RolloutController({"p": pool}, interval=0.02, log=lambda _m: None, pool_workers=True,
                             restart_backoff_s=0, thread_cls=_ManualThread)
    c.tick()
    shard = pool.live()[0].shard
    pool.members = []  # what a crash in the middle of a tick may leave in memory
    c.slots["p"].worker = _ManualThread(target=None, dead=True)
    c.slots["p"].worker_error = RuntimeError("crash")
    c.supervise()
    assert [m.shard for m in pool.live()] == [shard] and c.slots["p"].restarts == 1
    c.stop("test end")


class _ManualThread:
    def __init__(self, target=None, args=(), name=None, daemon=None, dead=False):
        self.target, self.args, self.dead = target, args, dead

    def start(self):
        pass

    def is_alive(self):
        return not self.dead

    def join(self, timeout=None):
        pass


class SlowRelease(FakeProvider):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = threading.Event()

    def release(self, slot, ticket):
        self.gate.wait(30)
        return super().release(slot, ticket)


def test_store_pools_release_in_the_background_a_tick_only_decides():
    clock = Clock()
    control = store_state.open_pool_control(_objects(), "p", "p", clock=clock)
    provider = SlowRelease("p", clock)
    pool = _pool("p", provider, clock, control=control, target=2)
    c = rc.RolloutController({"p": pool}, interval=60, log=lambda _m: None, thread_cls=_ManualThread)
    assert pool.async_release
    c.tick()
    victim = pool.live()[0]
    began = time.monotonic()
    pool._drain(victim, "test")  # noqa: SLF001
    c.tick()
    assert time.monotonic() - began < 2 and victim.state == DRAINING  # the release hangs, the ticks do not
    assert "release_started" in _names(pool)
    provider.gate.set()
    assert _wait_for(lambda: pool._releases[victim.ticket].done())  # noqa: SLF001
    c.tick()
    assert victim.state == GONE and provider.released == [victim.ticket]
    c.stop("test end")


def test_detach_then_adopt_keeps_every_sampler_and_its_shard(tmp_path, monkeypatch):
    """rollout.control_plane store: `yeto run detach` stops the controller without a drain;
    the next run up (another controller, new code) adopts every member from the store under
    its shard: no sampler restarted, no machine ordered or released, no new shard."""
    from tests.test_rollout_entry import _NoThread, _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_cfg, _, _ = _two_pools(tmp_path)
    cfg = dict(run_cfg, controller="in_process", rollout={"control_plane": "store", "drain_on_exit": False})
    clock = Clock()
    providers = {name: FakeProvider(name, clock) for name in ("nc", "modal")}
    routes = {name: FakeRoute() for name in providers}
    ops = {name: ProviderPoolOps(providers[name], routes[name], FakeFencer()) for name in providers}
    client = FakeS3()
    run = up.Run(cfg)
    c = run.start_controller(ops=ops, thread_cls=_NoThread, control_objects=_objects(client))
    assert c.pool_workers and c.detach_flag == run.dir / "DETACH"
    for _ in range(3):
        c.tick()
        clock.t += 30
    shards = {name: sorted(m.shard for m in pool.live()) for name, pool in c.pools.items()}
    started = {name: list(r.started) for name, r in routes.items()}
    assert all(shards.values()) and all(started.values())
    assert run.request_detach() == 0
    c.tick()  # the controller sees the DETACH flag
    assert c.detached and all(p.detached for p in c.pools.values())
    assert not any(p.released for p in providers.values()) and not any(r.stopped for r in routes.values())
    assert rc.drain_active("process exit") == {}  # a detached controller is never drained at exit
    again_run = up.Run(cfg)
    (again_run.dir / "DETACH").unlink()  # run up's pool() removes it before the controller starts
    again = again_run.start_controller(ops=ops, thread_cls=_NoThread, control_objects=_objects(client))
    try:
        assert {name: sorted(m.shard for m in pool.live()) for name, pool in again.pools.items()} == shards
        for _ in range(3):
            again.tick()
            clock.t += 30
        assert {name: list(r.started) for name, r in routes.items()} == started  # no sampler restarted
        assert not any(r.restarted for r in routes.values())
        assert sum(len(p.orders) for p in providers.values()) == sum(len(v) for v in shards.values())
        assert {name: sorted(m.shard for m in pool.live()) for name, pool in again.pools.items()} == shards
        tokens = {name: store_state.read_state(_objects(client), name)["control"]["token"] for name in providers}
        assert tokens == {"nc": 2, "modal": 2}  # the lock moved at once (released, not waited out)
    finally:
        again.stop("test end")


def test_run_detach_needs_the_store_control_plane(tmp_path, monkeypatch):
    from tests.test_rollout_entry import _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    run_cfg, _, _ = _two_pools(tmp_path)
    assert up.Run(run_cfg).request_detach() == 2


@pytest.mark.parametrize("workers", [False, True])
def test_the_serial_controller_stays_the_default(workers):
    clock = Clock()
    pool = _pool("p", FakeProvider("p", clock), clock)
    c = rc.RolloutController({"p": pool}, interval=0.02, log=lambda _m: None, pool_workers=workers,
                             restart_backoff_s=0)
    assert c.pool_workers is workers
    c.start()
    try:
        assert _wait_for(lambda: pool.live())
        assert (c.slots["p"].worker is not None) is workers
    finally:
        c.stop("test end", join_s=5)


def test_a_background_release_under_a_lost_lock_is_no_failed_release():
    from tests.test_rl_object_store import FakeS3  # noqa: PLC0415

    client, clock = FakeS3(), Clock()
    control = store_state.open_pool_control(_objects(client), "p", "p", ttl_s=600, clock=clock)
    provider = FakeProvider("p", clock)
    pool = _pool("p", provider, clock, control=control, target=1)
    pool.async_release = True
    pool.tick()
    victim = pool.live()[0]
    clock.t += 601
    store_state.open_pool_control(_objects(client), "p", "p", clock=clock)  # another controller took it
    pool._drain(victim, "test")  # noqa: SLF001 - the release thread's guard finds the higher token
    assert _wait_for(lambda: pool._releases[victim.ticket].done())  # noqa: SLF001
    from yeto.launch.pool.store_lock import LockLost  # noqa: PLC0415

    with pytest.raises(LockLost):  # the release's outcome is the lost lock, not a failed release
        pool._drain(victim, "retry")  # noqa: SLF001
    pool.tick()
    assert pool.detached and provider.released == []
    assert not any(e["event"].endswith("drain_failed") for e in pool.events.events)
