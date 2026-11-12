"""NC pool: the unreachable grace counts failed node probes in a row. Time in which the
node was not probed (the status call failed, a background start ran, the pool was down)
is no evidence that the node is gone."""

from __future__ import annotations

from yeto.launch.nc import client as ncc
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY, IslandPool

from test_nc_elastic import Clock, Cloud, _names, _pool, _ready_member, _run, _state  # noqa: F401


def test_two_lone_probe_failures_around_a_control_plane_outage_do_not_lose_the_node():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), unreachable_grace_s=300)
    m = _ready_member(pool, clock, cloud)
    orders = len(cloud.kinds("order"))
    cloud.tickets[m.ticket]["reachable"] = False  # one ssh exit 255
    pool.tick()
    clock.t += 60
    cloud.tickets[m.ticket]["reachable"] = True
    good = cloud.poll
    cloud.poll = lambda member: (_ for _ in ()).throw(ncc.NcError("nc-query exit 1: 503"))
    _run(pool, clock, clock.t + 360)  # six ticks without a status: the node is not probed
    cloud.poll = good
    cloud.tickets[m.ticket]["reachable"] = False  # the first probe after it: one more ssh exit 255
    pool.tick()
    assert "node_lost" not in _names(pool) and m.state == READY
    assert not cloud.kinds("fence") and not cloud.kinds("drain")
    clock.t += 60
    cloud.tickets[m.ticket]["reachable"] = True
    pool.tick()
    assert "node_reachable_again" in _names(pool) and m.unreachable_since is None
    assert m.state == READY and len(cloud.kinds("order")) == orders


def test_a_node_unreachable_through_a_control_plane_outage_is_lost_after_the_grace_of_failed_probes():
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), unreachable_grace_s=300)
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket]["reachable"] = False
    _run(pool, clock, clock.t + 180)  # failed probes at 0, 60, 120 s
    good = cloud.poll
    cloud.poll = lambda member: (_ for _ in ()).throw(ncc.NcError("nc-query exit 1: 503"))
    _run(pool, clock, clock.t + 600)
    cloud.poll = good
    assert m.state == READY
    _run(pool, clock, clock.t + 120)  # two more failed probes: 240 s of them, inside the grace
    assert m.state == READY and "node_lost" not in _names(pool)
    _run(pool, clock, clock.t + 120)
    assert m.state == GONE and "node_lost" in _names(pool)


def test_a_probe_failure_before_a_pool_restart_does_not_count_the_time_the_pool_was_down(tmp_path):
    clock = Clock()
    cloud = Cloud(clock)
    pool = _pool(clock, cloud, accounts=(1, 4), unreachable_grace_s=300)
    m = _ready_member(pool, clock, cloud)
    cloud.tickets[m.ticket]["reachable"] = False
    pool.tick()
    path = tmp_path / "pool.json"
    pool.save(path)
    clock.t += 1200  # the pool process is down for 20 minutes
    again = IslandPool.restore(path, PoolConfig(target=1, accounts=[1, 4], shard_base="nc"), cloud, clock=clock,
                               log=lambda x: None, events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))
    again.tick()  # still one failed probe
    restored = again.members[0]
    assert restored.ticket == m.ticket and restored.state == READY and "node_lost" not in _names(again)
    assert not cloud.kinds("fence") and not cloud.kinds("drain")
    _run(again, clock, clock.t + 400)
    assert restored.state == GONE and "node_lost" in _names(again)  # a node really gone is still lost
