"""NC pool: a sampler that ends with exit 0 did not necessarily finish its data. A fenced
sampler (a stop request on its shard, its incarnation lease taken, its dispenser member
retired) ends cleanly too; before the pool ends on an exit 0 it reads the run's store."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.launch.nc import elastic
from yeto.launch.nc.lease_pool import PoolConfig
from yeto.launch.pool import status as el
from yeto.launch.pool.data_path import R2DataPath
from yeto.launch.pool.island_pool import GONE, READY, IslandPool
from yeto.launch.pool.provider import ProviderPoolOps
from yeto.rl.engine.incarnation_lease import IncarnationLease
from yeto.rl.engine.sampler_resume import Member, handoff_requested, request_handoff, write_member

from test_nc_elastic import Clock, Cloud, Objects, _names, _pool, _run, _ssh_ops, _state  # noqa: F401


class Killed(BaseException):
    """The pool process dies (not an ``Exception``: nothing in the tick catches it)."""


class StoreCloud(Cloud):
    """The pool reads the run's store (``R2DataPath.exit_evidence``) like the SSH ops."""

    evidence_error = None

    def exit_evidence(self, m):
        if self.evidence_error is not None:
            raise self.evidence_error
        return R2DataPath(self.store).exit_evidence(m.shard)


def _sampler_leaves(cloud, m, reason, incarnation="inc-1"):
    """The sampler of ``m`` writes its LEAVE and its container ends with exit 0."""
    now = cloud.clock()
    write_member(cloud.store, Member(sampler=m.shard, incarnation=incarnation, joined_at=now - 1, resume={},
                                     left_at=now, leave_reason=reason))
    cloud.tickets[m.ticket].update(sampler="exited", exit=0)


def _two_ready(**elastic_cfg):
    clock = Clock()
    cloud = StoreCloud(clock)
    pool = _pool(clock, cloud, target=2, accounts=(1, 4, 5), **elastic_cfg)
    _run(pool, clock, 1800)
    first, second = pool.members
    assert first.state == second.state == READY
    return clock, cloud, pool, first, second


def _restored(path, cloud, clock):
    return IslandPool.restore(path, PoolConfig(target=2, accounts=[1, 4, 5], shard_base="nc"), cloud, clock=clock,
                              log=lambda x: None, events=el.EventLog(log=lambda x: None, clock=clock, run="run-a"))


@pytest.mark.parametrize("leave", ["FINISHED", "FENCED"])
def test_a_pool_killed_between_its_fence_and_its_state_save_does_not_end_on_the_fenced_samplers_exit(tmp_path, leave):
    clock = Clock()
    cloud = StoreCloud(clock)
    path = tmp_path / "pool.json"
    pool = _restored(path, cloud, clock)
    while clock.t < 1800:
        pool.tick()
        pool.save(path)  # the loop saves the state at the end of a tick
        clock.t += 60
    victim, other = pool.members
    assert victim.state == other.state == READY
    fence = cloud.fence

    def fence_then_die(m):
        fence(m)  # the handoff record and the lease are in the store ...
        raise Killed  # ... and the process is killed before this tick's save

    cloud.fence = fence_then_die
    cloud.tickets[victim.ticket]["reachable"] = False
    with pytest.raises(Killed):
        while True:
            pool.tick()
            pool.save(path)
            clock.t += 60
    assert handoff_requested(cloud.store, victim.shard)
    cloud.fence = fence
    cloud.tickets[victim.ticket]["reachable"] = True  # the node was only cut off for a while
    again = _restored(path, cloud, clock)
    victim, other = again.members
    assert victim.state == READY  # the state file knows nothing of the fence
    _sampler_leaves(cloud, victim, leave)  # its sampler obeys the stop request: exit 0
    _run(again, clock, clock.t + 3600)
    assert not again.finished and not again.stopped and "sampler_finished" not in _names(again)
    assert victim.state == GONE and other.state == READY and other.restarts == 0
    successor = again.members[-1]
    assert successor.state == READY and successor.replaces_shard == victim.shard
    assert "sampler_exit_not_data_end" in _names(again)


def test_a_sampler_an_operator_retired_leaves_fenced_and_is_restarted_not_taken_for_the_end_of_the_data():
    clock, cloud, pool, first, second = _two_ready()
    IncarnationLease(cloud.store, first.shard, "inc-1", clock=clock).claim()  # its own lease: no pool fence
    _sampler_leaves(cloud, first, "FENCED")  # store_dispense retire --holder: "dispense fenced", exit 0
    pool.tick()
    assert not pool.finished and not pool.stopped and "sampler_finished" not in _names(pool)
    assert [e[1] for e in cloud.kinds("restart")] == [first.ticket] and first.restarts == 1
    _run(pool, clock, clock.t + 1800)
    assert first.state == second.state == READY and len(cloud.kinds("order")) == 2
    assert not cloud.kinds("drain") and not cloud.kinds("fence")


def test_a_sampler_that_keeps_leaving_fenced_is_replaced_after_max_sampler_restarts():
    clock, cloud, pool, first, second = _two_ready(max_sampler_restarts=2)
    for _ in range(3):
        _sampler_leaves(cloud, first, "FENCED")
        pool.tick()
        clock.t += 60
    assert len(cloud.kinds("restart")) == 2 and first.state == GONE and "sampler_crash_loop" in _names(pool)
    assert not pool.finished and not pool.stopped and second.state == READY
    _run(pool, clock, clock.t + 3600)
    assert len(pool.members) == 3 and pool.members[-1].state == READY  # replaced like a crash loop


def test_an_exit_zero_under_a_pool_fence_of_the_incarnation_lease_restarts_the_sampler_in_place():
    clock, cloud, pool, first, second = _two_ready()
    # a stall fence (lease only) whose restart this pool no longer knows of, e.g. after a pool restart
    elastic.fence_incarnation(cloud.store, first.shard, by=first.ticket, clock=clock)
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)  # no LEAVE record read yet
    pool.tick()
    assert not pool.finished and [e[1] for e in cloud.kinds("restart")] == [first.ticket]
    _run(pool, clock, clock.t + 1800)
    assert first.state == second.state == READY and not pool.stopped


def test_a_stop_request_naming_a_live_members_shard_is_a_handoff_not_a_second_successor():
    clock, cloud, pool, first, second = _two_ready()
    request_handoff(cloud.store, first.shard, second.shard, clock=clock)
    _sampler_leaves(cloud, first, "FINISHED")
    pool.tick()
    assert first.handed_off and not pool.finished and not pool.pending
    pool.tick()
    assert first.state == GONE and not pool.stopped and second.state == READY


def test_an_exit_zero_with_a_finished_leave_and_no_fence_still_ends_the_pool():
    clock, cloud, pool, first, second = _two_ready()
    IncarnationLease(cloud.store, first.shard, "inc-1", clock=clock).claim()
    _sampler_leaves(cloud, first, "FINISHED")
    pool.tick()
    assert pool.finished and pool.stopped and not cloud.kinds("restart")
    assert "sampler_finished" in _names(pool) and "sampler_exit_not_data_end" not in _names(pool)


def test_an_exit_zero_with_no_record_in_the_store_still_ends_the_pool():
    clock, cloud, pool, first, second = _two_ready()
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    pool.tick()
    assert pool.finished and pool.stopped


def test_an_unreadable_store_decides_nothing_about_an_exit_zero():
    clock, cloud, pool, first, second = _two_ready()
    _sampler_leaves(cloud, first, "FINISHED")
    cloud.evidence_error = RuntimeError("503 SlowDown")
    _run(pool, clock, clock.t + 300)
    assert not pool.finished and not pool.stopped and first.state == READY
    assert _names(pool).count("exit_evidence_unreadable") == 1 and not cloud.kinds("restart")
    cloud.evidence_error = None
    pool.tick()
    assert pool.finished and pool.stopped


def test_exit_evidence_reads_the_handoff_the_lease_and_the_newest_member_record():
    store = SimpleNamespace(objects=Objects())
    path = R2DataPath(lambda: store)
    assert path.exit_evidence("nc-0") == {"handoff": None, "lease": None, "pool_fenced": False,
                                          "incarnation": None, "leave": None, "superseded": False}
    write_member(store, Member(sampler="nc-0", incarnation="zz-old", joined_at=10.0, resume={}, left_at=20.0,
                               leave_reason="FENCED"))
    write_member(store, Member(sampler="nc-0", incarnation="aa-new", joined_at=30.0, resume={}))
    write_member(store, Member(sampler="nc-1", incarnation="other", joined_at=99.0, resume={}, left_at=100.0,
                               leave_reason="FINISHED"))
    IncarnationLease(store, "nc-0", "aa-new").claim()
    assert path.exit_evidence("nc-0") == {"handoff": None, "lease": "aa-new", "pool_fenced": False,
                                          "incarnation": "aa-new", "leave": None, "superseded": False}
    elastic.fence_shard(store, "nc-0", by="t7")
    before = dict(store.objects.data)
    seen = path.exit_evidence("nc-0")
    assert seen["handoff"] == "pool-fence:t7" and seen["lease"] == "pool-fence:t7" and seen["pool_fenced"]
    assert seen["superseded"]  # the fence replaced the incarnation that held the lease
    assert path.exit_evidence("nc-1")["leave"] == "FINISHED"
    assert store.objects.data == before  # read only


def test_the_ops_pass_the_exit_evidence_of_the_data_path_and_none_without_one(tmp_path):
    ops = _ssh_ops(tmp_path, [])
    m = SimpleNamespace(ticket="abc", shard="nc-x-0")
    ops.data_path = lambda: SimpleNamespace(exit_evidence=lambda shard: {"leave": "FENCED", "shard": shard})
    assert ops.exit_evidence(m) == {"leave": "FENCED", "shard": "nc-x-0"}
    ops.data_path = lambda: SimpleNamespace()  # data_path syncer: no such record
    assert ops.exit_evidence(m) is None
    fencer = SimpleNamespace(fence_shard=None, fence_incarnation=None, exit_evidence=lambda shard: {"shard": shard})
    assert ProviderPoolOps(SimpleNamespace(), fencer=fencer).exit_evidence(m) == {"shard": "nc-x-0"}
    plain = SimpleNamespace(fence_shard=None, fence_incarnation=None)
    assert not hasattr(ProviderPoolOps(SimpleNamespace(), fencer=plain), "exit_evidence")


# -- the real engine sampler: what an operator retire leaves in the store ------------

RETIRE_ENV = "YETO_TEST_RETIRE_STATE"


def retiring_reward(item, completion) -> float:
    """At call ``retire_at`` an operator retires the sampler
    (``python -m yeto.rl.engine.store_dispense retire --holder <shard>``)."""
    import json
    import os
    from pathlib import Path

    from yeto.rl.engine import store_dispense as sd

    path = Path(os.environ[RETIRE_ENV])
    state = json.loads(path.read_text())
    state["n"] += 1
    if state["n"] == state["retire_at"]:
        state["retired"] = sd.retire(sd.DirObjects(state["store"]), "task-dispense", state["holder"],
                                     reason="operator")
    path.write_text(json.dumps(state))
    return 0.5


def test_a_real_sampler_an_operator_retires_exits_zero_fenced_and_the_pool_restarts_it(tmp_path, monkeypatch):
    import json

    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.resume import CheckpointStore

    clock, cloud, pool, first, second = _two_ready()
    store = tmp_path / "store"
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"n": 0, "retire_at": 3, "store": str(store), "holder": first.shard}))
    monkeypatch.setenv(RETIRE_ENV, str(state))
    spec = _sampler_spec(tmp_path, sampler_id=first.shard, num_rollouts=50,
                         reward="tests.test_nc_pool_exit_evidence:retiring_reward")
    (tmp_path / "prompts.jsonl").write_text("".join(
        json.dumps({"q": f"question {i}", "a": str(i), "metadata": {"task_id": f"t{i}"}}) + "\n" for i in range(40)))
    spec["prompts"] = {**spec["prompts"], "dispense": {"backend": "store", "namespace": "task-dispense",
                                                    "scan_s": 0.05, "max_scan_s": 0.05}}
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    # the sampler process: it is retired while it runs, leaves FENCED and ends with exit code 0
    assert entry.main(["--spec", str(path)]) == 0
    assert json.loads(state.read_text())["retired"]["released"] >= 1
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    events = [json.loads(line) for line in tape.read_text().splitlines()]
    (leave,) = [e for e in events if e["event"] == "rl_sampler_leave"]
    assert leave["reason"] == "FENCED" and leave["fenced"] in ("retired", "dispenser")
    assert leave["rounds"] < 50  # the data did not end
    # what the pool reads from the run's store: no handoff, the sampler's own lease, a FENCED leave
    seen = R2DataPath(CheckpointStore(store)).exit_evidence(first.shard)
    assert seen["handoff"] is None and not seen["pool_fenced"] and seen["leave"] == "FENCED"
    assert seen["lease"] == seen["incarnation"] == leave["incarnation"]
    cloud.exit_evidence = lambda m: R2DataPath(CheckpointStore(store)).exit_evidence(m.shard)
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    pool.tick()
    assert not pool.finished and not pool.stopped and [e[1] for e in cloud.kinds("restart")] == [first.ticket]


def test_a_finished_leave_of_an_incarnation_the_pool_fence_superseded_is_restarted_once_then_ends():
    clock, cloud, pool, first, second = _two_ready()
    IncarnationLease(cloud.store, first.shard, "inc-1", clock=clock).claim()
    elastic.fence_incarnation(cloud.store, first.shard, by=first.ticket, clock=clock)  # after its last lease read
    _sampler_leaves(cloud, first, "FINISHED", incarnation="inc-1")
    pool.tick()
    assert not pool.finished and [e[1] for e in cloud.kinds("restart")] == [first.ticket]
    clock.t += 60
    IncarnationLease(cloud.store, first.shard, "inc-2", clock=clock).claim()  # the restarted sampler joins
    _sampler_leaves(cloud, first, "FINISHED", incarnation="inc-2")  # and finds the data ended
    pool.tick()
    assert pool.finished and pool.stopped and len(cloud.kinds("restart")) == 1


def test_a_real_sampler_without_a_lease_that_finishes_under_an_old_pool_fence_ends_the_pool(tmp_path):
    import json

    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.resume import CheckpointStore

    clock, cloud, pool, first, second = _two_ready()
    store = CheckpointStore(tmp_path / "store")
    elastic.fence_incarnation(store, first.shard, by=first.ticket)  # a stall fence; the sampler was restarted
    spec = _sampler_spec(tmp_path, sampler_id=first.shard, num_rollouts=1, incarnation_lease=False)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    assert entry.main(["--spec", str(path)]) == 0  # it never claims the lease and finishes its data
    seen = R2DataPath(store).exit_evidence(first.shard)
    assert seen["pool_fenced"] and seen["leave"] == "FINISHED" and seen["superseded"] is False
    cloud.exit_evidence = lambda m: R2DataPath(store).exit_evidence(m.shard)
    cloud.tickets[first.ticket].update(sampler="exited", exit=0)
    pool.tick()
    assert pool.finished and pool.stopped and not cloud.kinds("restart")
    assert "sampler_exit_not_data_end" not in _names(pool)


def test_hourly_cost_raises_a_type_error_from_inside_the_providers_rate():
    """``hourly_cost(ticket)`` and a no-argument ``hourly_cost()`` are both supported,
    told apart by signature: a TypeError raised inside the rate is a bug and goes up,
    never retried without the ticket."""
    m = SimpleNamespace(account=1, ticket="t-1")
    assert ProviderPoolOps(SimpleNamespace(hourly_cost=lambda: 7.0)).hourly_cost(m) == 7.0
    assert ProviderPoolOps(SimpleNamespace(hourly_cost=lambda ticket=None: ticket)).hourly_cost(m) == "t-1"
    calls = []

    def rate(ticket):
        calls.append(ticket)
        return None + 1  # a real TypeError inside the rate

    with pytest.raises(TypeError, match="unsupported operand"):
        ProviderPoolOps(SimpleNamespace(hourly_cost=rate)).hourly_cost(m)
    assert calls == ["t-1"]
