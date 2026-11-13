"""NC pool: a stalled sampler engine (container up, requests in vLLM, no token
progress) is fenced and restarted in place; an engine idle while it waits for the
trainer is left alone; restarts that never bring progress back retire the node."""

from __future__ import annotations

import pytest

from yeto.launch.nc import elastic
from yeto.launch.pool import status as el
from yeto.launch.pool.island_pool import GONE, READY, STARTING
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
        elastic.fence_incarnation(self.store, m.shard, by=m.ticket, clock=self.clock)
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
    script = elastic.probe_script("c", "/x/events-*.jsonl", metrics_port=30000)
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


class _FlakyRestart(MetricsCloud):
    """``fail_restarts``: that many in-place restarts fail on a transient error first."""

    fail_restarts = 0

    def restart(self, m, shards):
        if self.fail_restarts:
            self.fail_restarts -= 1
            raise RuntimeError("ssh exit 255: Connection reset by peer")
        super().restart(m, shards)


def _stall_with_a_failed_restart(**elastic_kw):
    clock = Clock()
    cloud = _StallCloud(clock, ready_after=300)  # a _FlakyRestart (defined below)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4), **elastic_kw)
    _run(pool, clock, clock.t + 1800)
    first, second = [m for m in pool.members if m.state == READY]
    for m in (first, second):
        cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    cloud.eng(first.ticket).update(running=8, queued=3, rate=0.0)  # one engine stalls
    cloud.fail_restarts = 1  # ... and its restart hits one ssh error: the old sampler stays
    _run(pool, clock, clock.t + 12 * MIN)
    assert cloud.fenced_incarnations == [first.ticket] and "start_failed" in _names(pool)
    assert first.fenced_for_restart and not cloud.kinds("restart")
    # the old sampler reads the pool's fence on its lease and leaves FENCED: exit code 0
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    return clock, cloud, pool, first, second


def test_a_sampler_fenced_for_a_restart_that_exits_0_is_restarted_not_the_end_of_the_data():
    """exit 0 of the incarnation the pool itself fenced is that fence, not the data end:
    the shard's sampler is started again and the other samplers keep running."""
    clock, cloud, pool, first, second = _stall_with_a_failed_restart()
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished is None and pool.stopped is None
    assert "sampler_left_fenced" in _names(pool) and "sampler_finished" not in _names(pool)
    assert len(cloud.kinds("restart")) == 1 and not first.fenced_for_restart
    assert second.state == READY and not cloud.kinds("drain")
    cloud.eng(first.ticket).update(rate=5000.0)
    _run(pool, clock, clock.t + 15 * MIN)
    assert first.state == READY and len(pool.members) == 2


def test_a_fenced_sampler_that_cannot_be_restarted_is_replaced():
    clock, cloud, pool, first, second = _stall_with_a_failed_restart(max_sampler_restarts=2)
    cloud.fail_restarts = 99
    _run(pool, clock, clock.t + 10 * MIN)
    assert pool.finished is None and pool.stopped is None and second.state == READY
    assert first.state == GONE and pool.members[-1].replaces_shard == first.shard


def test_exit_0_without_a_fence_still_ends_the_pool():
    clock, cloud, pool, m = _setup()
    cloud.tickets[m.ticket].update(sampler="exited", exit=0)
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished and "sampler_finished" in _names(pool)


def test_a_crash_restart_clears_the_fence_so_the_real_end_of_the_data_ends_the_pool():
    """The fenced sampler died with a non-zero exit and the crash restart replaced it:
    a later exit 0 is the end of the data, with no extra restart and no new order."""
    clock = Clock()
    cloud = _FlakyRestart(clock, ready_after=300)
    pool = _pool(clock, cloud, target=1, accounts=(1, 4), max_sampler_restarts=1)
    _run(pool, clock, clock.t + 1800)
    (first,) = [m for m in pool.members if m.state == READY]
    cloud.eng(first.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    cloud.eng(first.ticket).update(running=8, queued=3, rate=0.0)
    cloud.fail_restarts = 1
    _run(pool, clock, clock.t + 12 * MIN)
    assert first.fenced_for_restart
    cloud.tickets[first.ticket].update(sampler="exited", exit=137)
    _run(pool, clock, clock.t + 2 * MIN)
    assert not first.fenced_for_restart
    cloud.eng(first.ticket).update(running=8, queued=0, rate=5000.0)
    _run(pool, clock, clock.t + 45 * MIN)
    assert first.state == READY and len(pool.members) == 1
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    _run(pool, clock, clock.t + 3 * MIN)
    assert pool.finished and len(pool.members) == 1 and "sampler_left_fenced" not in _names(pool)


def test_a_fenced_sampler_that_left_late_and_cannot_be_restarted_is_still_replaced():
    clock, cloud, pool, first, second = _stall_with_a_failed_restart(max_sampler_restarts=2)
    cloud.tickets[first.ticket].update(sampler="running", exit=None)  # not gone yet
    cloud.eng(first.ticket).update(running=0, queued=0, rate=0.0)  # winding down, no stall
    _run(pool, clock, clock.t + 35 * MIN)
    assert first.fenced_for_restart and first.state == READY
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    cloud.fail_restarts = 10**6
    _run(pool, clock, clock.t + 30 * MIN)
    assert first.state == GONE and pool.members[-1].replaces_shard == first.shard


# -- the pool's note of a stall fence (fenced_for_restart) and the run store's evidence ----

class _Killed(BaseException):
    """The pool process dies (not an ``Exception``: nothing in the tick catches it)."""


class _StallCloud(_FlakyRestart):
    """``fence_answer_lost``: the lease fence is written and the call raises;
    ``restart_answer_lost``: the new sampler is started and the call raises;
    ``restart_error``: raised by a restart instead."""

    fence_answer_lost = False
    restart_answer_lost = 0
    restart_error = None

    def fence_incarnation(self, m):
        super().fence_incarnation(m)
        if self.fence_answer_lost:
            raise RuntimeError("read timeout: no answer to the lease write")

    def restart(self, m, shards):
        if self.restart_error is not None:
            raise self.restart_error
        super().restart(m, shards)
        if self.restart_answer_lost:
            self.restart_answer_lost -= 1
            raise RuntimeError("ssh exit 255 after docker run")


class _StoreStallCloud(_StallCloud):
    """Ops that can read the run store (``exit_evidence``), like the SSH ops on data_path r2."""

    def exit_evidence(self, m):
        from yeto.launch.pool.data_path import R2DataPath

        return R2DataPath(self.store).exit_evidence(m.shard)


def _stalled(cloud_cls=_StallCloud, *, target=2, setup=None, run=True, **elastic_kw):
    """``target`` ready samplers; the first one's engine stalls (``setup(cloud, first)`` first)."""
    clock = Clock()
    cloud = cloud_cls(clock, ready_after=300)
    pool = _pool(clock, cloud, target=target, accounts=(1, 4), **elastic_kw)
    _run(pool, clock, clock.t + 1800)
    ready = [m for m in pool.members if m.state == READY]
    for m in ready:
        cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    first = ready[0]
    if setup is not None:
        setup(cloud, first)
    cloud.eng(first.ticket).update(running=8, queued=3, rate=0.0)
    if run:
        _run(pool, clock, clock.t + 12 * MIN)
    return clock, cloud, pool, first, ready[-1]


def _lost_fence_answer_and_failed_restart(cloud, first):
    cloud.fence_answer_lost, cloud.fail_restarts = True, 1


def test_a_fence_that_took_effect_but_raised_still_marks_the_sampler_fenced():
    clock, cloud, pool, first, second = _stalled(setup=_lost_fence_answer_and_failed_restart)
    assert read_lease(cloud.store, first.shard)["incarnation"] == f"pool-fence:{first.ticket}"
    assert "fence_failed" in _names(pool) and first.fenced_for_restart and not cloud.kinds("restart")
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # the old sampler read the fence
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished is None and pool.stopped is None and "sampler_finished" not in _names(pool)
    assert len(cloud.kinds("restart")) == 1 and second.state == READY


def test_with_the_run_store_a_fence_that_took_effect_but_raised_is_read_from_the_lease():
    def setup(cloud, first):
        IncarnationLease(cloud.store, first.shard, "inc-1", clock=cloud.clock).claim()
        _lost_fence_answer_and_failed_restart(cloud, first)

    clock, cloud, pool, first, second = _stalled(_StoreStallCloud, setup=setup)
    first.fenced_for_restart = False  # even without the pool's own note
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished is None and "sampler_exit_not_data_end" in _names(pool)
    assert len(cloud.kinds("restart")) == 1 and second.state == READY


def _restart_answer_lost(cloud, first):
    cloud.restart_answer_lost = 1


def test_a_restart_that_took_effect_but_raised_does_not_leave_the_fence_note_behind():
    """The new sampler runs although the restart call failed (the ssh dropped after
    ``docker run``): once its engine generates tokens the note is cleared, and its exit 0
    is the end of the data: no extra restart, no node ordered for data that ended."""
    clock, cloud, pool, first, second = _stalled(target=1, setup=_restart_answer_lost, max_sampler_restarts=0)
    assert first.fenced_for_restart and "start_failed" in _names(pool) and len(cloud.kinds("restart")) == 1
    cloud.eng(first.ticket).update(running=8, queued=0, rate=5000.0)  # the new sampler generates
    _run(pool, clock, clock.t + 10 * MIN)
    assert first.state == READY and not first.fenced_for_restart
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # the data ended
    _run(pool, clock, clock.t + 3 * MIN)
    assert pool.finished and "sampler_finished" in _names(pool) and "sampler_left_fenced" not in _names(pool)
    assert len(cloud.kinds("restart")) == 1 and len(cloud.kinds("order")) == 1


def test_with_the_run_store_a_stale_fence_note_does_not_turn_the_end_of_the_data_into_a_restart():
    clock, cloud, pool, first, second = _stalled(_StoreStallCloud, target=1, setup=_restart_answer_lost,
                                                  max_sampler_restarts=0)
    IncarnationLease(cloud.store, first.shard, "inc-2", clock=clock).claim()  # the new sampler joined
    first.fenced_for_restart = True  # the note is stale
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    _run(pool, clock, clock.t + 3 * MIN)
    assert pool.finished and len(cloud.kinds("restart")) == 1 and len(cloud.kinds("order")) == 1


def test_a_pool_retry_after_a_fenced_restart_hit_a_config_error_clears_the_fence_note():
    clock, cloud, pool, first, second = _stall_with_a_failed_restart(stop_on_finish=False)
    error = RuntimeError("image not found")
    error.config_error = True
    cloud.restart_error = error
    _run(pool, clock, clock.t + 2 * MIN)
    assert first.config_error is not None and pool.halted and first.fenced_for_restart
    cloud.restart_error = None
    assert pool.retry() == [first.ticket] and not first.fenced_for_restart
    cloud.eng(first.ticket).update(running=0, queued=0, rate=0.0)
    _run(pool, clock, clock.t + 10 * MIN)
    restarts = len(cloud.kinds("restart"))
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # the data ended
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished and len(cloud.kinds("restart")) == restarts and "sampler_finished" in _names(pool)


def test_the_fence_note_is_saved_before_the_restart_and_outlives_a_killed_pool(tmp_path):
    from yeto.launch.nc.lease_pool import PoolConfig
    from yeto.launch.pool.island_pool import IslandPool

    clock, cloud, pool, first, second = _stalled(run=False)
    path = tmp_path / "pool.json"
    pool.state_path = path
    pool.save(path)
    cloud.restart_error = _Killed()  # the process is killed while the restart's ssh runs
    with pytest.raises(_Killed):
        _run(pool, clock, clock.t + 12 * MIN)
    assert cloud.fenced_incarnations == [first.ticket]
    cloud.restart_error = None
    again = IslandPool.restore(path, PoolConfig(target=2, accounts=[1, 4], shard_base="nc"), cloud, clock=clock,
                               log=lambda x: None, events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))
    restored = next(m for m in again.members if m.ticket == first.ticket)
    assert restored.fenced_for_restart and restored.state == READY
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # the fenced sampler leaves
    _run(again, clock, clock.t + 2 * MIN)
    assert again.finished is None and again.stopped is None and len(cloud.kinds("restart")) == 1


def test_a_paused_fenced_sampler_that_exits_0_is_neither_restarted_nor_the_end_of_the_data():
    clock, cloud, pool, first, second = _stall_with_a_failed_restart()
    first.paused = True
    _run(pool, clock, clock.t + 5 * MIN)
    assert pool.finished is None and pool.stopped is None and not cloud.kinds("restart")
    assert first.fenced_for_restart and first.state == READY


def test_a_fenced_sampler_being_scaled_down_that_exits_0_is_released_not_restarted():
    clock, cloud, pool, first, second = _stall_with_a_failed_restart()
    first.retiring_since = clock.t
    _run(pool, clock, clock.t + 2 * MIN)
    assert first.state == GONE and not cloud.kinds("restart")
    assert pool.finished is None and "sampler_finished" not in _names(pool)


def test_after_the_data_ended_a_fenced_sampler_that_exits_0_is_not_restarted():
    clock, cloud, pool, first, second = _stall_with_a_failed_restart(stop_on_finish=False)
    cloud.tickets[first.ticket].update(sampler="running", exit=None)  # the fenced one has not left yet
    cloud.eng(first.ticket).update(running=0, queued=0, rate=0.0)
    cloud.tickets[second.ticket].update(sampler="exited", exit=0)  # the other sampler: the data ended
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished and first.fenced_for_restart and first.state == READY
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    _run(pool, clock, clock.t + 3 * MIN)
    assert not cloud.kinds("restart") and len(cloud.kinds("order")) == 2
    assert first.state == GONE and pool.stopped


def test_a_stalled_engine_that_recovers_by_itself_keeps_the_fence_note_and_its_exit_0_is_restarted():
    """The restart failed and the old, fenced sampler's engine generates tokens again: its
    counter only grew, which is no proof of a new sampler. It then reads the fence and ends
    with exit 0: restarted, not the end of the data."""
    clock, cloud, pool, first, second = _stall_with_a_failed_restart()
    cloud.tickets[first.ticket].update(sampler="running", exit=None)
    cloud.eng(first.ticket).update(running=8, queued=0, rate=5000.0)  # the old engine recovers
    _run(pool, clock, clock.t + 10 * MIN)
    assert first.fenced_for_restart and first.state == READY and not cloud.kinds("restart")
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # it leaves fenced
    _run(pool, clock, clock.t + 2 * MIN)
    assert pool.finished is None and pool.stopped is None and len(cloud.kinds("restart")) == 1
    assert not first.fenced_for_restart and "sampler_left_fenced" in _names(pool)


def test_a_stalled_engine_is_neither_fenced_nor_restarted_while_its_fence_note_cannot_be_saved(tmp_path):
    """The fence note is saved before the incarnation fence and the restart; a save that
    fails holds both back (a pool killed during the restart would forget the note) until
    a save holds."""
    clock, cloud, pool, m = _setup()
    pool.state_path = tmp_path / "pool.json"
    real_save, broken = pool.save, {"on": True}

    def save(p):
        if broken["on"]:
            raise OSError("disk full")
        real_save(p)

    pool.save = save
    cloud.eng(m.ticket).update(running=8, rate=5000.0)
    _run(pool, clock, clock.t + 5 * MIN)
    cloud.eng(m.ticket).update(running=8, queued=3, rate=0.0)
    _run(pool, clock, clock.t + 15 * MIN)
    assert cloud.fenced_incarnations == [] and not cloud.kinds("restart")
    assert not m.fenced_for_restart and m.state == READY
    broken["on"] = False
    _run(pool, clock, clock.t + 2 * MIN)
    assert cloud.fenced_incarnations == [m.ticket] and len(cloud.kinds("restart")) == 1
