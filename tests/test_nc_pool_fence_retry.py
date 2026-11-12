"""NC pool: the fence of a retired shard is a safety step. A fence that fails is tried
again until it succeeds; until then the node is not released, nothing is handed on and no
replacement is ordered. A node that is gone anyway (ticket ended, lease over, pool stop)
is released, but the sampler that continues its shard starts only after the fence."""

from __future__ import annotations

import subprocess

import pytest

from yeto.launch.nc import pool_run
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import FENCING, GONE, READY, IslandPool
from yeto.rl.engine.sampler_resume import handoff_requested

from test_nc_elastic import H, Clock, Cloud, _names, _pool, _ready_member, _run, _state  # noqa: F401


class FlakyFence(Cloud):
    """``fence_errors``: the next N fences fail before they write anything."""

    fence_errors = 0

    def fence(self, m):
        if self.fence_errors:
            self.fence_errors -= 1
            self.log.append(("fence_error", m.ticket, self.clock()))
            raise RuntimeError("503 SlowDown: handoff record not written")
        super().fence(m)


def _setup(**elastic_cfg):
    clock = Clock()
    cloud = FlakyFence(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4), **elastic_cfg)
    return clock, cloud, pool, _ready_member(pool, clock, cloud)


def _lose(pool, clock, cloud, m):
    """The node of ``m`` stops answering; run up to the tick that takes it for lost."""
    cloud.tickets[m.ticket]["reachable"] = False
    while "node_lost" not in _names(pool):
        pool.tick()
        clock.t += 60


def test_a_failed_fence_keeps_the_node_and_orders_nothing_until_it_succeeds():
    clock, cloud, pool, m = _setup()
    cloud.fence_errors = 3
    _lose(pool, clock, cloud, m)
    assert m.state == FENCING and m.fence_pending == "node lost" and m.fence_failures == 1
    assert not cloud.kinds("drain") and not pool.pending and len(cloud.kinds("order")) == 1
    _run(pool, clock, clock.t + 120)  # two more failures
    assert m.state == FENCING and m.fence_failures == 3 and len(cloud.kinds("fence_error")) == 3
    assert not cloud.kinds("drain") and not pool.pending and len(cloud.kinds("order")) == 1
    assert not handoff_requested(cloud.store, m.shard)
    pool.tick()  # the fourth attempt succeeds: the retire goes on
    assert handoff_requested(cloud.store, m.shard) and m.state == GONE and m.fence_pending is None
    assert [e[1] for e in cloud.kinds("drain")] == [m.ticket]
    fenced = [e for e in pool.events.events if e["event"] == "nc_elastic_shard_fenced"]
    assert len(fenced) == 1 and fenced[0]["after_failures"] == 3 and fenced[0]["why"] == "node lost"
    _run(pool, clock, clock.t + 1800)
    successor = pool.members[-1]
    assert successor.state == READY and successor.replaces_shard == m.shard and successor.predecessor_lost
    assert len(cloud.kinds("order")) == 2 and "fence_stuck" not in _names(pool)


def test_a_fence_that_keeps_failing_alerts_once_and_the_member_stays_fencing():
    clock, cloud, pool, m = _setup(fence_alert_failures=4, fence_alert_s=10**9)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    _run(pool, clock, clock.t + 120)
    assert "fence_stuck" not in _names(pool)  # three failures
    _run(pool, clock, clock.t + 1200)
    assert _names(pool).count("fence_stuck") == 1 and m.state == FENCING
    stuck = next(e for e in pool.events.events if e["event"] == "nc_elastic_fence_stuck")
    assert stuck["failures"] == 4 and stuck["shard"] == m.shard and stuck["why"] == "node lost"
    assert not cloud.kinds("drain") and len(cloud.kinds("order")) == 1 and not pool.stopped


def test_the_alert_also_comes_after_fence_alert_s_without_success():
    clock, cloud, pool, m = _setup(fence_alert_failures=10**6, fence_alert_s=300)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    _run(pool, clock, clock.t + 240)
    assert "fence_stuck" not in _names(pool)
    _run(pool, clock, clock.t + 120)
    assert _names(pool).count("fence_stuck") == 1


def test_a_pending_fence_outlives_a_pool_restart(tmp_path):
    clock, cloud, pool, m = _setup()
    cloud.fence_errors = 2
    _lose(pool, clock, cloud, m)
    path = tmp_path / "pool.json"
    pool.save(path)
    again = IslandPool.restore(path, PoolConfig(target=1, accounts=[1, 4], shard_base="nc"), cloud, clock=clock,
                               log=lambda x: None, events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))
    restored = again.members[0]
    assert restored.state == FENCING and restored.fence_pending == "node lost" and restored.fence_failures == 1
    again.tick()  # fails once more
    assert restored.state == FENCING and not cloud.kinds("drain") and len(cloud.kinds("order")) == 1
    clock.t += 60
    again.tick()
    assert restored.state == GONE and handoff_requested(cloud.store, m.shard)
    _run(again, clock, clock.t + 1800)
    assert again.members[-1].state == READY and again.members[-1].replaces_shard == m.shard


def test_an_evicted_node_is_released_at_once_but_its_successor_starts_only_after_the_fence():
    clock, cloud, pool, m = _setup()
    cloud.fence_errors = 5
    cloud.tickets[m.ticket].update(state="ended", reason="preempted")
    pool.tick()
    assert m.state == GONE and [e[1] for e in cloud.kinds("drain")] == [m.ticket]  # nothing to keep billed
    assert m.fence_pending == "evicted" and not handoff_requested(cloud.store, m.shard)
    _run(pool, clock, clock.t + 240)  # the replacement is ordered and granted; four more failures
    successor = pool.members[-1]
    assert successor.replaces_shard == m.shard and successor.continues_from == m.ticket
    assert cloud.tickets[successor.ticket]["state"] == "active" and successor.started is None
    assert [e[1] for e in cloud.kinds("start")] == [m.ticket] and "start_waits_for_fence" in _names(pool)
    assert len(cloud.kinds("fence_error")) == 5
    _run(pool, clock, clock.t + 1800)  # the fence succeeds; then the successor starts
    assert handoff_requested(cloud.store, m.shard) and m.fence_pending is None
    fenced_at = next(e[3] for e in cloud.kinds("fence"))
    started_at = next(e[3] for e in cloud.kinds("start") if e[1] == successor.ticket)
    assert fenced_at <= started_at and successor.state == READY
    assert "start_timeout" not in _names(pool)


def test_a_fencing_node_at_its_lease_end_is_released_and_the_fence_stays_owed():
    clock, cloud, pool, m = _setup(fence_alert_failures=3)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    assert m.state == FENCING
    clock.t = cloud.tickets[m.ticket]["until"] - 300  # inside end_margin_s of its lease
    pool.tick()
    assert m.state == GONE and [e[1] for e in cloud.kinds("drain")] == [m.ticket]
    assert m.fence_pending == "node lost"
    _run(pool, clock, clock.t + 600)
    successor = pool.members[-1]
    assert successor is not m and successor.started is None and len(cloud.kinds("start")) == 1
    cloud.fence_errors = 0
    _run(pool, clock, clock.t + 1800)
    assert successor.state == READY and successor.replaces_shard == m.shard and m.fence_pending is None


def test_a_fencing_node_whose_ticket_ends_is_released_and_the_fence_stays_owed():
    clock, cloud, pool, m = _setup()
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    cloud.tickets[m.ticket].update(state="ended", reason="lost")
    pool.tick()
    assert m.state == GONE and m.fence_pending == "node lost" and m.end_reason == "lost"
    assert len(pool.pending) + len(pool.members) == 2  # its shard is handed on once


def test_stop_releases_every_node_even_when_a_fence_fails_and_says_so():
    clock, cloud, pool, m = _setup()
    cloud.fence_errors = 10**6
    left = pool.drain_all("pool-stop", attempts=3, wait_s=0, sleep=lambda s: None)
    assert not left and m.state == GONE and cloud.tickets[m.ticket]["state"] == "ended"
    assert m.fence_pending == "stop" and len(cloud.kinds("fence_error")) == 3
    assert _names(pool).count("fence_stuck") == 1


def test_stop_retries_a_failed_fence_while_it_drains():
    clock, cloud, pool, m = _setup()
    cloud.fence_errors = 1
    assert not pool.drain_all("pool-stop", attempts=3, wait_s=0, sleep=lambda s: None)
    assert m.fence_pending is None and handoff_requested(cloud.store, m.shard)
    assert "fence_stuck" not in _names(pool)


def test_a_ready_replacement_does_not_release_its_fencing_predecessor():
    clock, cloud, pool, m = _setup()
    clock.t = cloud.tickets[m.ticket]["until"] - H + 10
    pool.tick()  # the replacement is ordered ...
    _run(pool, clock, clock.t + 180)  # ... and started
    successor = pool.members[-1]
    assert successor.replaces == m.ticket and successor.started is not None
    cloud.fence_errors = 2
    cloud.tickets[m.ticket].update(sampler="exited", exit=1)
    pool.config.elastic.max_sampler_restarts = 0  # the crash retires it
    pool.tick()
    assert m.state == FENCING
    cloud.tickets[successor.ticket]["joined_at"] = clock.t  # the replacement is ready meanwhile
    clock.t += 60
    pool.tick()
    assert successor.state == READY and m.state == FENCING and not cloud.kinds("drain")
    clock.t += 60
    pool.tick()
    assert m.state == GONE and [e[1] for e in cloud.kinds("drain")] == [m.ticket]


def test_a_scaled_down_node_is_released_only_after_its_fence(tmp_path, monkeypatch):
    from tests.test_nc_pool_scale_drain import DrainCloud, _elastic, _ready

    monkeypatch.setenv("YETO_NC_ACCOUNTS", "1,4,5")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("no subprocess"))

    class FlakyDrainCloud(FlakyFence, DrainCloud):
        pass

    cfg, path, clock, cloud, pool = _elastic(tmp_path, target=2, cloud_cls=FlakyDrainCloud,
                                             pool_kw={"scale_down": "fence"})
    victim = max(_ready(pool), key=lambda m: m.created)
    cloud.fence_errors = 2
    assert pool_run.scale_pool(cfg, path, 1, retire=[victim.ticket]) == 0
    pool.tick()
    assert victim.state == FENCING and not cloud.kinds("drain") and pool.config.target == 1
    clock.t += 60
    pool.tick()
    assert victim.state == FENCING and not cloud.kinds("drain")
    clock.t += 60
    pool.tick()
    assert victim.state == GONE and [e[1] for e in cloud.kinds("drain")] == [victim.ticket]
    _run(pool, clock, clock.t + 600)
    assert len(_ready(pool)) == 1 and len(cloud.kinds("order")) == 2  # no refill, no second victim


def test_a_node_without_a_lease_end_is_released_after_fence_max_hold_s_and_the_fence_stays_owed():
    clock = Clock()
    cloud = FlakyFence(clock, ready_after=300, lease_s=10**9)  # a provider without leases
    pool = _pool(clock, cloud, accounts=(1, 4), fence_max_hold_s=1800)
    m = _ready_member(pool, clock, cloud)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    _run(pool, clock, clock.t + 1740)
    assert m.state == FENCING and not cloud.kinds("drain") and len(cloud.kinds("order")) == 1
    _run(pool, clock, clock.t + 600)
    assert m.state == GONE and [e[1] for e in cloud.kinds("drain")] == [m.ticket]
    assert _names(pool).count("fence_abandoned_node_released") == 1 and m.fence_pending == "node lost"
    successor = pool.members[-1]
    assert successor is not m and successor.continues_from == m.ticket and successor.started is None
    _run(pool, clock, clock.t + 3600)
    assert successor.started is None and len(cloud.kinds("start")) == 1  # still waiting for the fence
    cloud.fence_errors = 0
    _run(pool, clock, clock.t + 1800)
    assert successor.state == READY and successor.replaces_shard == m.shard and m.fence_pending is None
    assert _names(pool).count("fence_abandoned_node_released") == 1


def test_fence_max_hold_s_null_keeps_the_node_until_the_fence_succeeds():
    clock = Clock()
    cloud = FlakyFence(clock, ready_after=300, lease_s=10**9)
    pool = _pool(clock, cloud, accounts=(1, 4), fence_max_hold_s=None)
    m = _ready_member(pool, clock, cloud)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    _run(pool, clock, clock.t + 6 * H)
    assert m.state == FENCING and not cloud.kinds("drain") and "fence_abandoned_node_released" not in _names(pool)


def test_a_fence_that_keeps_failing_sends_its_first_failure_and_every_tenth():
    clock, cloud, pool, m = _setup(fence_max_hold_s=None)
    cloud.fence_errors = 10**6
    _lose(pool, clock, cloud, m)
    while m.fence_failures < 25:
        pool.tick()
        clock.t += 60
    failed = [e["failures"] for e in pool.events.events if e["event"] == "nc_elastic_fence_failed"]
    assert failed == [1, 10, 20] and len(cloud.kinds("fence_error")) == 25


def test_the_fence_knobs_are_validated():
    assert el.ElasticConfig.from_dict({"fence_alert_failures": 2, "fence_alert_s": 0}).fence_alert_failures == 2
    assert el.ElasticConfig.from_dict({"fence_max_hold_s": None}).fence_max_hold_s is None
    assert el.ElasticConfig().fence_max_hold_s == 7200.0
    with pytest.raises(ValueError, match="fence_alert"):
        el.ElasticConfig.from_dict({"fence_alert_failures": 0})
    with pytest.raises(ValueError, match="fence_max_hold_s"):
        el.ElasticConfig.from_dict({"fence_max_hold_s": -1})
