"""NC pool: a stalled sampler engine (container up, requests in vLLM, no token
progress) is fenced and restarted in place; an engine idle while it waits for the
trainer is left alone; restarts that never bring progress back retire the node."""

from __future__ import annotations

import pytest

from yeto.launch.nc import elastic as el
from yeto.launch.nc.lease_pool import GONE, READY, STARTING
from yeto.rl.engine.incarnation_lease import IncarnationLease, read_lease
from yeto.rl.engine.sampler_resume import handoff_requested

from test_nc_elastic import Clock, Cloud, _names, _pool, _ready_member, _run, _state  # noqa: F401

MIN = 60.0


class MetricsCloud(Cloud):
    """Each node also reports vLLM's /metrics: the token counter and requests."""

    def __init__(self, clock, **kw):
        super().__init__(clock, **kw)
        self.engine = {}  # ticket -> {"tokens", "running", "queued", "rate", "tw"}
        self.fenced_incarnations = []

    def eng(self, t):
        return self.engine.setdefault(t, {"tokens": 0.0, "running": 0, "queued": 0, "rate": 0.0, "tw": False})

    def restart(self, m, shards):
        super().restart(m, shards)
        self.eng(m.ticket)["tokens"] = 0.0  # a new engine process: the counter starts over

    def probe(self, m):
        p = super().probe(m)
        if p.node != "ok" or p.sampler != "running":
            return p
        e = self.eng(m.ticket)
        e["tokens"] += e["rate"]
        return el.Probe(node="ok", sampler="running", joined=p.joined, gen_tokens=e["tokens"],
                        running=e["running"], queued=e["queued"], trainer_wait=e["tw"])

    def fence_incarnation(self, m):
        el.fence_incarnation(self.store, m.shard, by=m.ticket, clock=self.clock)
        self.fenced_incarnations.append(m.ticket)


def _setup(**elastic):
    clock = Clock()
    cloud = MetricsCloud(clock, ready_after=300)
    pool = _pool(clock, cloud, accounts=(1, 4), **elastic)
    m = _ready_member(pool, clock, cloud)
    return clock, cloud, pool, m


def test_probe_line_carries_the_engine_metrics():
    p = el.Probe.parse("YETO_NC_PROBE status=running exit=0 joined=1 age=3 tw=1 tok=12345 run=4 wq=2", 0)
    assert (p.gen_tokens, p.running, p.queued, p.trainer_wait) == (12345.0, 4, 2, True)
    p = el.Probe.parse("YETO_NC_PROBE status=running exit=0 joined=1 age=3 tw=0 tok=- run=- wq=-", 0)
    assert (p.gen_tokens, p.running, p.queued, p.trainer_wait) == (None, None, None, False)
    script = el.probe_script("c", "/x/events-*.jsonl", metrics_port=30000)
    assert "127.0.0.1:30000/metrics" in script and "vllm:generation_tokens_total" in script
    assert "rm " not in script and "stop" not in script
    assert el.ElasticConfig().engine_stall_s == 600 and el.ElasticConfig().max_stall_restart_failures == 3
    with pytest.raises(ValueError):
        el.ElasticConfig.from_dict({"engine_stall_s": 0})
    with pytest.raises(ValueError):
        el.ElasticConfig.from_dict({"max_stall_restart_failures": 0})


def test_sampling_engine_is_healthy():
    clock, cloud, pool, m = _setup()
    cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 60 * MIN)
    assert m.state == READY and "engine_stalled" not in _names(pool) and not cloud.kinds("restart")


def test_idle_engine_waiting_for_the_trainer_is_not_a_stall():
    clock, cloud, pool, m = _setup()
    cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    cloud.eng(m.ticket).update(running=0, queued=0, rate=0.0, tw=True)  # backlog full, no new release
    _run(pool, clock, clock.t + 60 * MIN)
    assert m.state == READY and "engine_stalled" not in _names(pool) and not cloud.kinds("restart")


def test_stalled_engine_is_fenced_and_restarted_in_place():
    clock, cloud, pool, m = _setup()
    cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    old = IncarnationLease(cloud.store, m.shard, "inc-old", clock=clock)
    old.claim()
    cloud.eng(m.ticket).update(running=8, queued=3, rate=0.0)  # RCCL hang: requests in, no tokens
    _run(pool, clock, clock.t + 9 * MIN)
    assert "engine_stalled" not in _names(pool)  # under the 10 min default
    _run(pool, clock, clock.t + 3 * MIN)
    assert "engine_stalled" in _names(pool)
    assert cloud.fenced_incarnations == [m.ticket] and len(cloud.kinds("restart")) == 1
    assert m.state == STARTING and m.stall_failures == 1 and m.ticket in cloud.tickets
    assert len(pool.members) == 1 and not cloud.kinds("drain")  # same node, no new order
    # the stuck incarnation is superseded (its claims free) but the shard is not handed off
    assert old.check_before_write() is True and read_lease(cloud.store, m.shard)["incarnation"].startswith("pool-fence")
    assert not handoff_requested(cloud.store, m.shard)
    cloud.eng(m.ticket).update(rate=5000.0)  # the new engine samples
    _run(pool, clock, clock.t + 15 * MIN)
    assert m.state == READY and m.stall_failures == 0 and "engine_progress_resumed" in _names(pool)


def test_restarts_that_never_progress_retire_the_node():
    clock, cloud, pool, m = _setup(max_stall_restart_failures=2)
    cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    cloud.eng(m.ticket).update(running=8, rate=0.0)  # every incarnation hangs again
    _run(pool, clock, clock.t + 90 * MIN)
    assert len(cloud.kinds("restart")) == 2
    assert "engine_stall_loop" in _names(pool) and m.state == GONE
    assert cloud.kinds("fence") and cloud.tickets[m.ticket]["state"] == "ended"
    assert pool.members[-1].replaces_shard == m.shard  # the slot goes to a new node
