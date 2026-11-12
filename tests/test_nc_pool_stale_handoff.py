"""NC pool: a stop request (handoff record) an earlier attempt of the run left on a shard
name is set aside before a new sampler starts on that name, also when the record names a
shard or a ticket this pool holds again."""

from __future__ import annotations

import json

from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import READY, IslandPool
from yeto.rl.engine.sampler_resume import handoff_requested, request_handoff

from test_nc_elastic import H, Clock, Cloud, CloudWithHandoffs, _names, _pool, _run, _state  # noqa: F401


def _aside(cloud, shard):
    return [json.loads(v) for k, v in cloud.store.objects.data.items()
            if k.startswith(f"sample-buffer/handoffs-aside/{shard}-")]


def test_a_stale_handoff_naming_a_shard_this_pool_orders_again_is_set_aside():
    clock = Clock()
    cloud = CloudWithHandoffs(clock)
    # an earlier attempt of the run: nc-2 continued nc-0 and asked it to stop
    request_handoff(cloud.store, "nc-0", "nc-2", clock=clock)
    # this attempt (the state file was moved away) orders nc-0, nc-1 and nc-2 in one tick
    pool = _pool(clock, cloud, target=3, accounts=(1, 4, 5))
    _run(pool, clock, 1800)
    assert [(m.shard, m.state, m.replaces_shard) for m in pool.members] == [
        ("nc-0", READY, None), ("nc-1", READY, None), ("nc-2", READY, None)]
    assert not handoff_requested(cloud.store, "nc-0")  # it would stop the new nc-0 at once (exit 0)
    assert [r["successor"] for r in _aside(cloud, "nc-0")] == ["nc-2"]
    assert _names(pool).count("stale_handoff_set_aside") == 1


def test_a_stale_fence_of_a_ticket_the_new_pool_adopts_is_set_aside(tmp_path):
    clock = Clock()
    cloud = CloudWithHandoffs(clock)
    first = _pool(clock, cloud, accounts=(1,))
    _run(first, clock, 1800)
    assert first.members[0].state == READY
    cloud.drain_error = 10**6  # the stop fences nc-0 but its ticket cannot be deleted: still open
    first.stop("pool-stop")
    assert handoff_requested(cloud.store, "nc-0") and cloud.tickets["t1"]["state"] == "active"
    cloud.drain_error = 0
    cloud.tickets["t1"]["sampler"] = "absent"
    # the state file is moved away: a fresh pool adopts the open ticket t1 as nc-0
    second = IslandPool.restore(tmp_path / "pool.json", PoolConfig(target=1, accounts=[1], shard_base="nc"), cloud,
                                clock=clock, log=lambda x: None,
                                events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))
    _run(second, clock, clock.t + 1800)
    m = second.members[0]
    assert (m.ticket, m.shard, m.state) == ("t1", "nc-0", READY) and "adopted" in _names(second)
    assert not handoff_requested(cloud.store, "nc-0")
    assert [r["successor"] for r in _aside(cloud, "nc-0")] == ["pool-fence:t1"]


def test_the_handoff_of_this_pools_own_replacement_is_kept_at_a_pool_retry():
    clock = Clock()
    cloud = CloudWithHandoffs(clock)
    pool = _pool(clock, cloud, accounts=(1, 4))
    _run(pool, clock, 1800)
    m = pool.members[0]
    clock.t = cloud.tickets[m.ticket]["until"] - H + 10
    pool.tick()  # its replacement is ordered ...
    clock.t += 120
    pool.tick()  # ... and started: its sampler asks m's to stop
    successor = pool.members[1]
    assert successor.replaces_shard == m.shard and successor.started is not None and m.handed_off
    request_handoff(cloud.store, m.shard, successor.shard, clock=clock)  # what the successor's sampler wrote
    cloud.tickets[m.ticket].update(sampler="exited", exit=2)  # m's sampler refuses its spec
    pool.tick()
    assert m.config_error is not None and pool.halted
    assert pool.retry() == [m.ticket]
    assert handoff_requested(cloud.store, m.shard) and not _aside(cloud, m.shard)
