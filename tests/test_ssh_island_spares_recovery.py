"""Spare nodes of a resident ssh: trainer island, the recovery details
(yeto.launch.island_spares, own-cluster-fleet-stability 4.1-4.3): the usual loss order of a
power cut (rank 0 exits first, the lost node's stream ends minutes later), the longest
wait for a spare, what a lost node owes on the ledger, the per-learner resume point, the
health check a pick and a release need, the lease release, the quarantine shared through
the lease store (and a launcher restarted while it waits), a stale health reading, the
checks outside the ops lock, a store without an atomic create.

Over the fake SSH cluster of test_rl_ssh_resident and the fakes of test_ssh_island_spares."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from test_rl_ssh_resident import Clock, _cards, _crash, b300, make_ops, spec, wait_for  # noqa: F401 - _cards: autouse
from test_ssh_island_spares import (
    NAMES,
    READING,
    Cluster,
    FakeLeases,
    FleetLog,
    _controller,
    _health,  # noqa: F401 - autouse: the fake health checker
    _inventory,
    islands_spec,
    join_stops,
    lost_island,
    make_spares,
    spare,
)
from tests.test_run_up_fleet import _cfg, files  # noqa: F401 - files: fixture
from tests.test_run_up_fleet_runtime import fleet_run  # noqa: F401 - fixture
from yeto.cost.ledger import ResourceLedger
from yeto.launch import island_spares as isp
from yeto.launch.fleet import ABANDONED, RECOVERING, RUNNING, FixedRosterIslandAbandoned, RelaunchBlocked
from yeto.launch.island_spares import IslandSpares, NodeRecords
from yeto.rl.engine.object_store import DirectoryObjectStore


def started(cluster, k):
    return [c[1] for c in cluster.calls if c[0] == "start" and c[2] == k]


def open_rows(tmp_path):
    return sorted(r.id for r in ResourceLedger(tmp_path / "ledger.jsonl").open_resources())


# ------------------------------------------------------------------ H1: rank 0 exits first
def _rank0_exits_while(tmp_path, *, polls=2):
    """An island whose rank 0 exits (its collective failed) while node b300-1 does not answer
    over SSH and its stream still looks open: the release does not confirm b300-1's stop."""
    cluster, clock = Cluster([*NAMES, "b300-3"]), Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=polls)
    s = islands_spec()
    spares, store = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    cluster.unreachable.add("b300-1")
    cluster.exit("b300-0", 0, 1)
    assert wait_for(lambda: rec.job.procs[0].poll() is not None)
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert rec.job.procs[1].poll() is None  # its stop is not confirmed: its stream stays open
    assert rec.job.procs[2].poll() is not None  # a confirmed node's stream is ended
    assert spares.unconfirmed == {1} and spares.lost == {}
    return cluster, ops, spares, store, rec


def test_rank_0_exits_first_and_the_lost_nodes_stream_ends_minutes_later(tmp_path):
    """A power cut: the dead node's stream ends minutes after rank 0 exited; then
    ``unreachable_polls`` checks without an answer make it lost."""
    cluster, ops, spares, store, rec = _rank0_exits_while(tmp_path)
    for _ in range(2):  # its stream still open: two checks are not enough
        assert ops.relaunch(None, "isl") is None
    assert spares.lost == {}
    cluster.drop("b300-1")  # the keepalive gives up: its stream ends
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    assert ops.relaunch(None, "isl") == 1  # the stream ended and no answer: lost, replaced
    assert started(cluster, 1) == ["b300-0", "b300-3", "b300-2"]
    (lost,) = [e for e in ops.island_events("isl") if e["kind"] == isp.NODE_LOST]
    assert lost["node"] == "b300-1" and "its stream ended and it does not answer over SSH (3 checks" in lost["reason"]
    assert str(ops.job_status("isl", 1)) == "RUNNING"
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1", "run1/isl-i1"]
    assert json.loads(store.read("fleet/nodes/b300-1/quarantine.json"))["owed"][0]["incarnation"] == 0


def test_a_node_whose_stream_never_ends_is_lost_after_twice_the_checks(tmp_path):
    cluster, ops, spares, store, rec = _rank0_exits_while(tmp_path)
    for _ in range(3):
        assert ops.relaunch(None, "isl") is None
    assert spares.lost == {} and rec.job.procs[1].poll() is None
    assert ops.relaunch(None, "isl") == 1  # 2 x unreachable_polls checks: lost
    (lost,) = [e for e in ops.island_events("isl") if e["kind"] == isp.NODE_LOST]
    assert "its stop was not confirmed, its stream still open, and it does not answer over SSH (4 checks" \
        in lost["reason"]


def test_sshd_refusing_new_sessions_while_the_old_one_lives_is_no_loss(tmp_path):
    cluster, ops, spares, store, rec = _rank0_exits_while(tmp_path)
    for _ in range(3):  # three refusals in a row, its old connection alive
        assert ops.relaunch(None, "isl") is None
    cluster.unreachable.discard("b300-1")  # sshd takes sessions again
    assert ops.relaunch(None, "isl") == 1
    assert started(cluster, 1) == NAMES and spares.lost == {}  # the same nodes, nothing quarantined
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None


# ------------------------------------------------------------------ M1: the longest wait
def test_a_wait_past_max_wait_gives_the_island_up_with_an_hourly_report(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], max_wait_s=3 * 3600.0)
    spares.records.claim("b300-3", "isl-other")
    wall = [5000.0]
    ops.wall = lambda: wall[0]
    ops.island_events("isl")
    assert ops.relaunch(None, "isl") is None
    (first,) = [e for e in ops.island_events("isl") if e["kind"] == isp.WAITING_SPARE]
    assert first["elapsed_s"] == 0
    journal = json.loads((tmp_path / "runs/run1/ssh-islands/isl-spares.json").read_text())
    assert journal["waiting_since"] == 5000.0  # a restarted launcher keeps the wait's start
    for _ in range(35):  # every 5 minutes, up to 2 h 55 min
        clock.t += 300
        wall[0] += 300
        assert ops.relaunch(None, "isl") is None
    hourly = [e for e in ops.island_events("isl") if e["kind"] == isp.WAITING_SPARE]
    assert [e["elapsed_s"] for e in hourly] == [3600, 7200]
    assert all("b300-3: held by island isl-other" in e["reason"] for e in hourly)
    clock.t += 300
    wall[0] += 300
    with pytest.raises(RelaunchBlocked, match=r"waited 10800 s for a spare node .*--spare-max-wait-s"):
        ops.relaunch(None, "isl")
    (given_up,) = ops.island_events("isl")
    assert given_up["kind"] == isp.WAITING_SPARE and given_up["given_up"] is True


def test_spare_max_wait_s_reaches_the_island(tmp_path):
    s = islands_spec()
    args = SimpleNamespace(rl_node_inventory_nodes=[*s.nodes, spare(3)], cluster_prefix="run1",
                           rl_checkpoint_store=str(tmp_path / "store"), spare_max_wait_s=600.0)
    assert isp.spares_from_args(args, s).max_wait_s == 600.0
    assert isp.spares_from_args(SimpleNamespace(**{**vars(args), "spare_max_wait_s": None}), s).max_wait_s == 86400.0
    assert isp.spare_view(args, s)["max_wait_s"] == 600.0


@pytest.mark.parametrize("fixed_roster", [False, True])
def test_the_fleet_gives_a_waiting_island_up_at_max_wait_also_with_a_fixed_roster(tmp_path, fixed_roster, capsys):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], max_wait_s=1800.0)
    spares.records.claim("b300-3", "isl-other")
    ops.wall = lambda: clock.t
    ctl = _controller(ops, recover_timeout=600.0, fleet_log=FleetLog(), fixed_roster=fixed_roster)
    ctl.ops.sky_ops.t = 0.0
    ctl.ops.now = lambda: clock.t
    rec = ctl.learners["isl"]
    ctl._poll_recorded(rec)
    assert rec["state"] == RECOVERING
    for _ in range(50):  # 25 min: past the recover timeout, short of the longest wait
        clock.t += 30
        ctl._poll_recorded(rec)
    assert rec["state"] == RECOVERING
    err = capsys.readouterr().err
    # while it waits, the controller's retry line comes every 15 min, not every poll
    assert 1 <= err.count("relaunch attempt for isl failed (waiting for a spare node); will retry") <= 2
    assert "relaunch attempt for isl failed; will retry" not in err
    assert err.count("while it waits for a spare node; not given up") == 1
    with pytest.raises(FixedRosterIslandAbandoned) if fixed_roster else _nothing():
        for _ in range(20):
            clock.t += 30
            ctl._poll_recorded(rec)
            if rec["state"] == ABANDONED:
                break
    assert rec["state"] == ABANDONED and "waited" in capsys.readouterr().err


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_each_wait_for_a_spare_is_said_again(tmp_path, capsys):
    extra = [spare(3), spare(4)]
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=extra)
    spares.records.claim("b300-3", "isl-other")
    spares.records.claim("b300-4", "isl-other")
    ctl = _controller(ops, recover_timeout=300.0, fleet_log=FleetLog())
    ctl.ops.sky_ops.t = 0.0
    ctl.ops.now = lambda: clock.t
    rec = ctl.learners["isl"]

    def wait_then_free(name):
        for _ in range(20):  # past the recover timeout while it waits
            clock.t += 30
            ctl._poll_recorded(rec)
        spares.records.unclaim(name, "isl-other")
        for _ in range(40):
            clock.t += 30
            ctl._poll_recorded(rec)
            if rec["state"] == RUNNING:
                return
        raise AssertionError("not replaced")

    ctl._poll_recorded(rec)
    wait_then_free("b300-3")
    cluster.lose("b300-3")  # the spare is lost too: a second wait
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    wait_then_free("b300-4")
    assert capsys.readouterr().err.count("while it waits for a spare node; not given up") == 2


def _runtime(islands, min_islands, records, absent_after_s=3600.0):
    """A fleet runtime of ``islands`` names over island ``records`` (no run behind it)."""
    from yeto.rl import up_fleet_runtime

    rt = up_fleet_runtime.FleetRuntime.__new__(up_fleet_runtime.FleetRuntime)
    rt.plan = SimpleNamespace(min_islands=min_islands, absent_after_s=absent_after_s,
                              islands=[SimpleNamespace(index=k, name=n) for k, n in enumerate(islands)])
    ks = tuple(range(len(islands)))
    stops = SimpleNamespace(running=lambda: ks, fleet_stop_reason=None, stop_reason=lambda k: None)
    rt.stops = lambda: stops
    rt.active = lambda: list(ks)
    rt.trainer_records = lambda now: records
    return rt


def _waiting(since):
    return {"state": "recovering", "spares": {"waiting_spare": True, "wait_reason": "no free spare",
                                              "waiting_since": since}}


def test_a_single_island_fleet_waiting_under_an_hour_goes_on_and_stops_past_it(monkeypatch):
    """A one-island fleet (min_islands 1) is not stopped the moment it waits: its island counts
    for fleet.waiting_absent_after_s (1 h), then as absent; its own give-up stays 24 h."""
    import time

    records = {0: _waiting(10_000.0)}
    rt = _runtime(["east"], 1, records)
    monkeypatch.setattr(time, "time", lambda: 10_000.0 + 3000)
    assert rt.waiting_spare() == {} and rt.all_stopped() is None
    monkeypatch.setattr(time, "time", lambda: 10_000.0 + 3700)
    (why,) = rt.waiting_spare().values()
    assert "waiting 3700 s, past fleet.waiting_absent_after_s 3600 s" in why
    assert rt.all_stopped() == "fleet.min_islands"


def test_in_a_fleet_an_island_waiting_past_the_limit_is_absent(monkeypatch):
    import time

    records = {0: {"state": "running"}, 1: _waiting(10_000.0)}
    monkeypatch.setattr(time, "time", lambda: 10_000.0 + 1800)
    assert _runtime(["east", "west"], 2, records).all_stopped() is None  # still counted
    monkeypatch.setattr(time, "time", lambda: 10_000.0 + 3601)
    assert _runtime(["east", "west"], 2, records).all_stopped() == "fleet.min_islands"
    assert _runtime(["east", "west"], 1, records).all_stopped() is None  # island 0 is enough
    assert _runtime(["east", "west"], 2, records, absent_after_s=7200.0).all_stopped() is None  # configured
    records[1]["spares"]["waiting_spare"] = False
    assert _runtime(["east", "west"], 2, records).all_stopped() is None


def test_an_island_waiting_3599_s_counts_and_3601_s_is_absent(monkeypatch):
    import time

    records = {0: {"state": "running"}, 1: _waiting(10_000.0)}
    rt = _runtime(["east", "west"], 2, records)
    monkeypatch.setattr(time, "time", lambda: 10_000.0 + 3599)
    assert rt.waiting_spare() == {} and rt.all_stopped() is None
    monkeypatch.setattr(time, "time", lambda: 10_000.0 + 3601)
    assert list(rt.waiting_spare()) == [1] and rt.all_stopped() == "fleet.min_islands"


def test_a_wait_without_its_start_counts_from_when_it_is_first_seen():
    rt = _runtime(["east"], 1, {0: _waiting(None)}, absent_after_s=60.0)
    assert rt.waiting_spare(now=5000.0) == {}
    assert rt.waiting_spare(now=5050.0) == {}
    assert list(rt.waiting_spare(now=5061.0)) == [0]


def test_waiting_absent_after_s_is_a_fleet_key_kept_in_the_plan():
    from yeto.rl import fleet

    assert fleet.waiting_absent_after_s({"fleet": {}}) is None
    assert fleet.waiting_absent_after_s({"fleet": {"waiting_absent_after_s": 900}}) == 900.0
    for bad in (0, -1, True, "600"):
        with pytest.raises(ValueError, match="fleet.waiting_absent_after_s must be a number of seconds > 0"):
            fleet.waiting_absent_after_s({"fleet": {"waiting_absent_after_s": bad}})
    plan = fleet.FleetPlan(run="r", store="/s", dispense="fleet", islands=(), waiting_absent_after_s=900.0)
    doc = plan.to_json()
    assert doc["roster"]["waiting_absent_after_s"] == 900.0
    assert fleet.FleetPlan.from_json(doc).absent_after_s == 900.0
    default = fleet.FleetPlan(run="r", store="/s", dispense="fleet", islands=())
    assert "waiting_absent_after_s" not in default.to_json()["roster"]  # unset: the plan is as before
    assert default.absent_after_s == 3600.0


def test_the_compiled_plan_keeps_waiting_absent_after_s(fleet_run, files):  # noqa: F811
    cfg = _cfg(files)
    cfg["fleet"]["waiting_absent_after_s"] = 900
    assert fleet_run(cfg).fleet_plan().absent_after_s == 900.0
    assert fleet_run(_cfg(files)).fleet_plan().absent_after_s == 3600.0


def test_the_island_record_carries_the_waits_start(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    spares.records.claim("b300-3", "isl-other")
    ops.wall = lambda: 7777.0
    assert ops.relaunch(None, "isl") is None
    view = ops.spares_view("isl")
    assert view["waiting_spare"] is True and view["waiting_since"] == 7777.0


# ------------------------------------------------------------------ M2: what a lost node owes
def test_a_lost_node_whose_own_stop_is_confirmed_owes_nothing(tmp_path):
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, store = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], READING)
    join_stops(spares)
    assert spares.records.own_quarantine("b300-2")["owed"] == []  # stopped: owes nothing (a settle marker)
    assert json.loads(store.read("fleet/nodes/b300-2/quarantine.json"))["owed"]  # the record itself is not rewritten
    assert str(ops.job_status("isl", 0)) == "FAILED"
    assert ops.relaunch(None, "isl") == 1
    assert open_rows(tmp_path) == ["run1/isl-i1"]  # no owed row for b300-2


def test_release_node_closes_the_owed_row_and_a_node_picked_again_is_billed_once(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1", "run1/isl-i1"]
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lambda m: None) == 0
    assert open_rows(tmp_path) == ["run1/isl-i1"]  # released: no more billing for its owed stop
    cluster.lose("b300-3")  # the spare is lost; b300-1 (released) is picked again
    rec = ops.islands["isl"]
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 1)
    assert ops.relaunch(None, "isl") == 2
    assert started(cluster, 2) == ["b300-0", "b300-1", "b300-2"]
    assert open_rows(tmp_path) == ["run1/isl-i1-owed-b300-3", "run1/isl-i2"]
    rows = {r.id: r for r in ResourceLedger(tmp_path / "ledger.jsonl").open_resources()}
    assert [n["name"] for n in rows["run1/isl-i2"].attrs["nodes"]].count("b300-1") == 1


def _health_lost_then_picked_again(tmp_path):
    """b300-2 is lost by its health and its stop confirmed (it owes nothing for i0); b300-3
    takes its place (i1); b300-2 is released by hand, then b300-3 loses power and b300-2 is
    picked again (i2)."""
    cluster = Cluster([*NAMES, "b300-3", "b300-4"])
    ops = make_ops(tmp_path, cluster, health_every_s=0, unreachable_polls=2)
    s = islands_spec()
    spares, store = make_spares(tmp_path, s, [spare(3), spare(4)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], READING)
    join_stops(spares)
    assert ("b300-2", 0) in spares.stopped
    assert str(ops.job_status("isl", 0)) == "FAILED" and ops.relaunch(None, "isl") == 1
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3), spare(4)])
    assert isp.release_node_main(inventory=inv, name="b300-2", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lambda m: None) == 0
    cluster.lose("b300-3")
    assert wait_for(lambda: rec.job.procs[2].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 1)
    assert ops.relaunch(None, "isl") == 2 and started(cluster, 2) == ["b300-0", "b300-1", "b300-2"]
    assert not [k for k in spares.stopped if k[0] == "b300-2"]  # picked again: its old confirmation is gone
    return cluster, ops, spares, rec


def test_a_node_picked_again_and_then_power_lost_owes_its_stop_again(tmp_path):
    cluster, ops, spares, rec = _health_lost_then_picked_again(tmp_path)
    cluster.lose("b300-2")
    assert wait_for(lambda: rec.job.procs[2].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 2)
    assert ops.relaunch(None, "isl") == 3
    assert "run1/isl-i2-owed-b300-2" in open_rows(tmp_path)  # cleanup and the reaper still stop it there


def test_a_node_picked_again_and_then_health_lost_with_a_failed_stop_owes_it(tmp_path):
    cluster, ops, spares, rec = _health_lost_then_picked_again(tmp_path)
    cluster.kill_rc["b300-2"] = 1  # its stop does not go through this time
    ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], READING)
    join_stops(spares)
    assert ("b300-2", 2) not in spares.stopped
    assert str(ops.job_status("isl", 0)) == "FAILED" and ops.relaunch(None, "isl") == 3
    assert "run1/isl-i2-owed-b300-2" in open_rows(tmp_path)


# ------------------------------------------------------------------ M3: the per-learner resume point
ARGS = dict(rl_sample_role="trainer", rl_single_island_no_sync=False, rl_island_sample_role=None,
            rl_sampler_follow="trainer", rl_backend="miles", rl_max_policy_age=2, rl_checkpoint_store="/store",
            rl_island_buffer_shards=["0:a", "1:b"], rl_island_release_every=None, rl_island_max_policy_age=None)


def test_the_resume_point_is_the_learners_own(tmp_path):
    args = SimpleNamespace(**ARGS, rl_cut_every=5)
    assert isp.resume_keys(args, 1, 2) == ("island-cuts/island1/LATEST", "trainer1/policy-releases/LATEST.json")
    assert isp.resume_keys(SimpleNamespace(**ARGS), 0, 2) == ("LATEST", "trainer0/policy-releases/LATEST.json")
    assert isp.resume_keys(None, None) == ("LATEST", "policy-releases/LATEST.json")
    store = DirectoryObjectStore(tmp_path / "store")
    store.write("policy-releases/LATEST.json", json.dumps({"version": 99, "path": "x"}).encode())  # not its own
    store.write("trainer1/policy-releases/LATEST.json", json.dumps({"version": 12, "path": "r12"}).encode())
    keys = isp.resume_keys(args, 1, 2)
    assert isp.latest_checkpoint_step(store, keys)["step"] == 12  # no cut yet: its own release
    store.write("LATEST", json.dumps({"cut_id": "r000050-i0"}).encode())  # another island's place
    store.write("island-cuts/island1/LATEST", json.dumps({"cut_id": "r000014-i1"}).encode())
    assert isp.latest_checkpoint_step(store, keys) == {"source": "round_cut", "step": 14, "ref": "r000014-i1",
                                                       "key": "island-cuts/island1/LATEST"}


def test_a_per_island_launch_reads_the_cut_of_its_learner_id_not_its_index():
    """fleet.launch per_island: each launch has one island (index 0) whose learner id is the
    fleet's (``--rl-fleet-learner-id``): learner 2 reads island-cuts/island2."""
    args = SimpleNamespace(rl_fleet_role="island", rl_fleet_learner_id=2, rl_fleet_learners=3,
                           rl_checkpoint_store="/store", rl_cut_every=5)
    assert isp.resume_keys(args, 0, 1)[0] == "island-cuts/island2/LATEST"
    assert isp.spares_from_args(SimpleNamespace(**vars(args), rl_node_inventory_nodes=[*islands_spec().nodes, spare(3)],
                                                cluster_prefix="run1"),
                                islands_spec(), island_index=0, num_islands=1) is not None


def test_start_resident_ssh_gives_each_island_its_index(tmp_path, monkeypatch):
    from yeto.launch import ssh_resident, trainer_provision

    seen = []

    class Ops:
        def __init__(self, **kw):
            pass

        def add(self, name, spec, *, spares=None):
            pass

        def start(self, name):
            return 0

    monkeypatch.setattr(ssh_resident, "SshIslandOps", Ops)
    monkeypatch.setattr(isp, "spares_from_args", lambda args, spec, **kw: seen.append(kw) or None)
    args = SimpleNamespace(cluster_prefix="run1")
    trainer_provision.start_resident_ssh(args, {"a": islands_spec(), "b": islands_spec()}, ["a", "b"], {}, {}, None, {},
                                         island_index={"a": 1, "b": 3}, num_islands=4)
    assert seen == [{"island_index": 1, "num_islands": 4}, {"island_index": 3, "num_islands": 4}]


# ------------------------------------------------------------------ M4: a health check, not an answer
def test_without_a_health_checker_no_spare_is_picked_and_no_node_released(tmp_path, monkeypatch):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    monkeypatch.setattr(isp, "_checker_for", lambda family, **kw: None)
    assert isp.node_check(spare(3), cluster, multi=True) == (
        False, "no health check for device family cuda here (yeto.hw.node_health)")
    assert ops.relaunch(None, "isl") is None
    assert "no health check for device family cuda" in ops.spares_view("isl")["wait_reason"]
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    common = dict(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster, objects=store,
                  kill_grace_s=1, say=lines.append)
    assert isp.release_node_main(**common) == 1
    assert "stays quarantined: no health check" in lines[-1] and "--force" in lines[-1]
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None
    assert isp.release_node_main(**common, force=True) == 0
    assert any(line.startswith("[yeto] WARN: --force: node b300-1 is released without its health check")
               for line in lines)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None


def test_spares_without_a_health_checker_are_named_in_the_dry_run_and_at_the_start(tmp_path, monkeypatch, capsys):
    s = islands_spec()
    args = SimpleNamespace(rl_node_inventory_nodes=[*s.nodes, spare(3)], cluster_prefix="run1",
                           rl_checkpoint_store=str(tmp_path / "store"))
    assert "note" not in isp.spare_view(args, s)  # a checker: nothing said
    monkeypatch.setattr(isp, "_checker_for", lambda family, **kw: None)
    view = isp.spare_view(args, s)
    assert "the spare nodes do not apply; a failed island is relaunched on its own nodes" in view["note"]
    assert "WARN: the inventory declares trainer-spare nodes" in capsys.readouterr().err  # the dry run says it
    lines = []
    assert isp.spares_from_args(args, s, log=lines.append) is None  # as on main
    assert [ln for ln in lines if "WARN" in ln and "device family cuda has no node health check" in ln]
    assert isp.spare_view(SimpleNamespace(rl_node_inventory_nodes=list(s.nodes)), s) is None  # no spares: silent


def test_a_check_that_did_not_run_is_no_pass(monkeypatch):
    class Unchecked:
        def script(self, hcas):
            return "echo yeto-fake-health"

        def parse(self, text, hcas, prev):
            return SimpleNamespace(checked=False, ok_single=None, ok_multi=None, reasons=[])

    cluster = Cluster(["b300-3"])
    monkeypatch.setattr(isp, "_checker_for", lambda family, **kw: Unchecked())
    assert isp.node_check(spare(3), cluster, multi=True) == (False, "health not checked (rc 0)")


def test_a_failed_lease_release_keeps_the_quarantine_and_says_why(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    cluster.unreachable.discard("b300-1")

    class Broken(FakeLeases):
        def force_release(self, node, *, holder=None):
            raise OSError("lease store unreachable")

    lines = []
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    rc = isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                               objects=store, kill_grace_s=1, say=lines.append, leases=Broken())
    assert rc == 1 and "its lease could not be released (OSError: lease store unreachable)" in lines[-1]
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None  # the record stays


# ------------------------------------------------------------------ M5: only this run's lease
def test_release_node_gives_up_only_its_own_lease(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    cluster.unreachable.discard("b300-1")
    leases = FakeLeases(held={"b300-1": "other-run/trainer@b"})
    leases.kept = ["b300-1"]  # kept by that run: only its holder's release may end it
    lines = []
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lines.append, leases=leases) == 0
    assert leases.holders["b300-1"] == "other-run/trainer@b"  # another run's lease is not touched
    assert "its lease is held by other-run/trainer@b, not released" in lines[-1]


def test_release_node_probes_a_live_lease_before_it_forces_it_and_refuses_a_node_still_running(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    cluster.unreachable.discard("b300-1")
    leases = FakeLeases(held={"b300-1": "run1/trainer@launcher-a"})  # live: renewed, not kept
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    common = dict(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster, objects=store,
                  kill_grace_s=1, say=lines.append, leases=leases)
    cluster.still_runs["b300-1"] = ["yeto-run1-isl-i0-n1"]
    assert isp.release_node_main(**common) == 1
    assert "still runs and the node is not confirmed free (node b300-1 still runs run run1" in lines[-1]
    assert leases.holders["b300-1"] == "run1/trainer@launcher-a"
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None
    cluster.still_runs.clear()
    assert isp.release_node_main(**common) == 0
    assert leases.forced == [("b300-1", True)] and "b300-1" not in leases.holders
    probes = [i for i, c in enumerate(cluster.calls) if c == ("lease-probe", "b300-1")]
    assert probes and probes[-1] < len(cluster.calls)  # the probe ran before the forced release


def test_the_quarantine_names_the_lease_holder_and_a_restart_keeps_it(tmp_path):
    leases = FakeLeases(held={n: "run1/trainer@launcher-a" for n in NAMES})
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], leases=leases)
    doc = json.loads(store.read("fleet/nodes/b300-1/quarantine.json"))
    assert doc["lease_holder"] == "run1/trainer@launcher-a"
    spares.records.claim("b300-3", "isl-other")
    assert ops.relaunch(None, "isl") is None  # waiting
    ops.close()
    again_leases = FakeLeases(held={n: "run1/trainer@launcher-a" for n in NAMES})  # the restart took them back
    again = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    spares2, _ = make_spares(tmp_path, islands_spec(), [spare(3)], leases=again_leases)
    again.add("isl", islands_spec(), spares=spares2)
    again.start("isl")
    assert again_leases.kept == ["b300-1"]  # the quarantined node's lease out of the renewals again


# ------------------------------------------------------------------ M6: the shared quarantine
def test_a_quarantine_is_shared_through_the_lease_store_and_other_runs_skip_the_node(tmp_path):
    shared = DirectoryObjectStore(tmp_path / "leases")
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0, unreachable_polls=2)
    s = islands_spec()
    store = DirectoryObjectStore(tmp_path / "store")
    spares = IslandSpares(pool=(*s.nodes, spare(3)), records=NodeRecords(store, run_id="run1", log=lambda m: None,
                                                                          shared=shared))
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    doc = json.loads(shared.read("nodes/b300-1/quarantine.json"))
    assert doc["run"] == "run1" and doc["island"] == "isl"
    shared.write("nodes/b300-2/lease/00000001.json", b"{}")  # node_lease's objects: not quarantines
    shared.write("nodes/b300-2/events/quarantine.json", b"{}")
    assert isp.shared_quarantined_names(shared) == {"b300-1"}  # for another run's pick (excluded)
    # another run on the same inventory: its own run store, the same lease store
    other = IslandSpares(pool=(*spec(2).nodes,), records=NodeRecords(DirectoryObjectStore(tmp_path / "store2"),
                                                                    run_id="run2", log=lambda m: None, shared=shared))
    pick = isp.pick_spare(other, "isl", [b300(0), b300(2)], [1], ops=cluster)
    assert pick.node is None and pick.skipped["b300-1"].startswith(  # who, why and how to release it
        "quarantined: b300-1 by run run1 island isl (rank 1 (b300-1) does not answer")
    assert pick.skipped["b300-1"].endswith("once fixed, run `yeto cluster release-node b300-1` with this inventory "
                                           "and the run store of the run that quarantined it")
    assert "release-node b300-1" in pick.why_none()
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, shared=shared, kill_grace_s=1, say=lambda m: None) == 0
    assert shared.read("nodes/b300-1/quarantine.json") is None and isp.shared_quarantined_names(shared) == set()


def test_a_launcher_restarted_while_waiting_starts_and_waits_on(tmp_path):
    """Formerly a known limit: a launcher restarted while the island waited for a spare
    refused to start (the lost node did not answer). Its quarantined node is lost again at
    the start, the waiting goes on (its start is kept) and a free spare replaces it."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], max_wait_s=7200.0)
    spares.records.claim("b300-3", "isl-other")
    ops.wall = lambda: 1000.0
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    ops.close()
    again = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    again.wall = lambda: 1000.0 + 3600.0
    spares2, _ = make_spares(tmp_path, islands_spec(), [spare(3)], max_wait_s=7200.0)
    again.add("isl", islands_spec(), spares=spares2)
    job = again.start("isl")  # no refusal
    assert job == 0 and str(again.job_status("isl", job)) == "FAILED" and again.waiting_spare("isl")
    assert spares2.wait_since == 1000.0  # the wait's start survived the restart
    assert "b300-1" in str(again.islands["isl"].reason)
    spares2.records.unclaim("b300-3", "isl-other")
    clock.t += 3600
    assert again.relaunch(None, "isl") == 1
    assert started(cluster, 1) == ["b300-0", "b300-3", "b300-2"]
    assert spares2.wait_since is None


def test_a_node_another_run_quarantined_is_replaced_at_the_first_start(tmp_path):
    shared = DirectoryObjectStore(tmp_path / "leases")
    NodeRecords(DirectoryObjectStore(tmp_path / "store2"), run_id="run2", log=lambda m: None,
                shared=shared).quarantine("b300-2", island="x", reason="XID 79 in run2")
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares = IslandSpares(pool=(*s.nodes, spare(3)), records=NodeRecords(DirectoryObjectStore(tmp_path / "store"),
                                                                          run_id="run1", log=lambda m: None,
                                                                          shared=shared))
    ops.add("isl", s, spares=spares)
    assert ops.start("isl") == 0
    assert started(cluster, 0) == ["b300-0", "b300-1", "b300-3"]


def test_a_lost_nodes_lease_is_kept_until_release_node(tmp_path):
    leases = FakeLeases(held={n: "run1/trainer@launcher-a" for n in NAMES})
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], leases=leases)
    assert leases.kept == ["b300-1"]  # held without renewal: no other run gets the broken node
    assert leases.holders["b300-1"] == "run1/trainer@launcher-a"
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, leases=leases, say=lambda m: None) == 0
    assert "b300-1" not in leases.holders


# ------------------------------------------------------------------ L1: a stale health reading
def _real_reading():
    from yeto.hw import node_health

    result = node_health.HealthResult(checked=True, ok_single=False, ok_multi=False, reasons=["XID 79 on GPU 3"],
                                      facts={"gpu_reasons": ["XID 79 on GPU 3"]})
    return node_health.Reading(verdict="lost", result=result, event=node_health.EVENT_LOST)


def test_a_health_reading_taken_before_a_spare_took_the_rank_is_ignored(tmp_path):
    """The real node_health.Reading names no node: the resident tick stamps the reading with
    the rank's node and its replacement count, and the hook drops one taken before."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    rec = ops.islands["isl"]
    old = _real_reading()
    ops._health_reading(rec, 1, old)  # rank 1 (b300-1) judged lost by its health, the hook not run yet
    rec.health_pending = []  # job_status took it (the hook runs after the lock, below)
    assert ops.relaunch(None, "isl") == 1 and rec.spec.nodes[1].name == "b300-3"
    ops._node_health_lost(rec, 1, ["XID 79 on GPU 3"], old)
    assert spares.lost == {} and str(ops.job_status("isl", 1)) == "RUNNING"  # b300-3 is not taken as lost
    fresh = _real_reading()
    ops._health_reading(rec, 1, fresh)  # a reading of b300-3 itself
    rec.health_pending = []
    ops._node_health_lost(rec, 1, ["XID 79 on GPU 3"], fresh)
    assert list(spares.lost) == [1] and str(ops.job_status("isl", 1)) == "FAILED"


def test_a_health_reading_of_the_same_node_back_in_its_rank_after_a_replacement_is_stale(tmp_path):
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    old = _real_reading()
    ops._health_reading(rec, 2, old)
    rec.health_pending = []
    rec.rank_gen[2] = rec.rank_gen.get(2, 0) + 1  # a spare took rank 2 and handed it back to b300-2 since
    ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], old)
    ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], old)  # asked again: still stale
    assert spares.lost == {} and str(ops.job_status("isl", 0)) == "RUNNING"


# ------------------------------------------------------------------ L2 / L3: no SSH under the ops lock
def test_the_lost_nodes_stop_and_the_spares_checks_run_outside_the_ops_lock(tmp_path, monkeypatch):
    cluster = Cluster([*NAMES, "b300-3", "b300-4"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3), spare(4)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    held = []
    real_check, real_run = isp.node_check, cluster.run

    def check(node, o, **kw):
        held.append(("check", node.name, ops._lock._is_owned()))
        return real_check(node, o, **kw)

    def run(node, script, **kw):
        if "kill -TERM" in script and node.name == "b300-2":
            held.append(("stop", node.name, ops._lock._is_owned()))
        return real_run(node, script, **kw)

    monkeypatch.setattr(isp, "node_check", check)
    cluster.run = run
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], READING)
    join_stops(spares)
    cluster.sick.add("b300-3")  # the first candidate fails its check: the next one is checked too
    assert ops.relaunch(None, "isl") == 1
    assert started(cluster, 1) == ["b300-0", "b300-1", "b300-4"]
    assert held == [("stop", "b300-2", False), ("check", "b300-3", False), ("check", "b300-4", False)]


# ------------------------------------------------------------------ L5: an atomic create or nothing
class CheckThenWrite(DirectoryObjectStore):
    atomic_create = False  # e.g. the OSS relay

    def __init__(self, root):
        super().__init__(root)
        self.creates = []

    def create(self, key, data):
        self.creates.append(key)
        return super().create(key, data)


def test_a_store_without_an_atomic_create_is_refused_for_claims(tmp_path):
    store = CheckThenWrite(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    with pytest.raises(isp.SpareError, match="no atomic exclusive create"):
        records.claim("b300-3", "isl")
    assert store.creates == []  # refused before any create (no claim written, even for a moment)
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    ops.add("isl", s, spares=IslandSpares(pool=(*s.nodes, spare(3)), records=records))
    from yeto.launch.ssh_trainer_island import SshIslandError

    with pytest.raises(SshIslandError, match="no atomic exclusive create"):
        ops.start("isl")
    assert cluster.count("start") == 0


def test_a_store_that_finds_out_on_the_create_is_refused_and_leaves_no_claim(tmp_path):
    class Finds(DirectoryObjectStore):
        conditional_writes = True

        def create(self, key, data):
            created = super().create(key, data)
            self.conditional_writes = False  # the endpoint refused If-None-Match: a fallback write
            return created

    store = Finds(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    with pytest.raises(isp.SpareError):
        records.claim("b300-3", "isl")
    assert store.read("fleet/nodes/b300-3/claim.json") is None


# ------------------------------------------------------------------ M7: the release hint
def test_the_lost_node_log_names_release_node(tmp_path):
    lines = []
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0, unreachable_polls=2)
    ops.log = lines.append
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], READING)
    critical = [ln for ln in lines if "CRITICAL" in ln and "is lost" in ln]
    assert critical and "run `yeto cluster release-node b300-2`" in critical[0]
    assert "until released by hand: run `yeto cluster release-node <name>`" in isp.spare_view(
        SimpleNamespace(rl_node_inventory_nodes=[*s.nodes, spare(3)]), s)["quarantine"]


def test_the_trainer_pick_leaves_out_a_node_quarantined_in_the_lease_store(tmp_path, monkeypatch, capsys):
    """A node another run's island lost (quarantined in the lease store, its lease kept or
    released) is no candidate of a new run's trainer pick until release-node."""
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore("s3://leases/x", client=FakeS3()), Clock()
    objects.write("nodes/b300-0/quarantine.json", json.dumps(
        {"node": "b300-0", "run": "run7", "island": "isl0", "reason": "rank 1 (b300-0): no answer"}).encode())
    objects.write("nodes/b300-1/lease/other.json", b"{}")  # node_lease's own keys: not a quarantine
    monkeypatch.setattr(sti, "_lease_record", lambda args: {})
    monkeypatch.setattr(sti, "_record_island_nodes", lambda args, name, nodes: None)
    leases = NodeLeases("s3://leases/x", "run2/trainer", objects=objects, clock=clock, monotonic=clock)
    trainers = [b300(i) for i in range(3)]
    args = SimpleNamespace(cluster_prefix=f"oc-f-{tmp_path.name}")
    picked = sti.lease_trainer_nodes(args, trainers, 2, leases, name="isl", wait_s=0)
    assert [n.name for n in picked] == ["b300-1", "b300-2"] and leases.held() == ["b300-1", "b300-2"]
    assert isp.shared_quarantined_names(objects) == {"b300-0"}
    err = capsys.readouterr().err
    note = "quarantined, left out: b300-0 by run run7 island isl0 (rank 1 (b300-0): no answer)"
    assert f"island isl: {note}" in err and "run `yeto cluster release-node <name>` with this inventory" in err
    with pytest.raises(sti.SshIslandError) as caught:  # only the quarantined node is left: no candidate
        sti.lease_trainer_nodes(args, trainers, 1, leases, name="isl2", wait_s=0)
    assert note in str(caught.value) and "release-node" in str(caught.value)  # not just "no free node"
    assert "b300-0" not in leases.held()


def test_the_lease_wait_line_names_who_quarantined_the_node(tmp_path, monkeypatch, capsys):
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore("s3://leases/x", client=FakeS3()), Clock()
    objects.write("nodes/b300-0/quarantine.json", json.dumps({"run": "run7", "island": "isl0"}).encode())
    NodeLeases("s3://leases/x", "run9/trainer", objects=objects, clock=clock, monotonic=clock).acquire("b300-1")
    monkeypatch.setattr(sti, "_lease_record", lambda args: {})
    monkeypatch.setattr(sti, "_record_island_nodes", lambda args, name, nodes: None)
    from yeto.launch import node_lease

    real = node_lease.pick_nodes

    def pick_nodes(*a, **kw):  # its waiting line (every 5 minutes) goes through the given log
        kw["log"]("[node-lease] 2 node(s) needed, 1 free; held: b300-1 by run run9; waiting 300s so far")
        return real(*a, **kw)

    monkeypatch.setattr(node_lease, "pick_nodes", pick_nodes)
    leases = NodeLeases("s3://leases/x", "run2/trainer", objects=objects, clock=clock, monotonic=clock)
    with pytest.raises(sti.SshIslandError, match="held: b300-1 by run run9.*b300-0 by run run7 island isl0"):
        sti.lease_trainer_nodes(SimpleNamespace(cluster_prefix=f"oc-f-{tmp_path.name}"), [b300(0), b300(1)], 2,
                                leases, name="isl", wait_s=0)
    assert ("waiting 300s so far; quarantined, left out: b300-0 by run run7 island isl0. Once a node is fixed, run "
            "`yeto cluster release-node <name>` with this inventory and the run store of the run that quarantined it") in capsys.readouterr().err


def test_a_start_on_nodes_another_run_holds_is_refused(tmp_path):
    import dataclasses

    from yeto.launch.ssh_trainer_island import SshIslandError

    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=Clock(), health_every_s=0, unreachable_polls=2)
    leases = FakeLeases(held={"b300-2": "run9/trainer@other"})
    s = dataclasses.replace(islands_spec(), leases=leases)
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    with pytest.raises(SshIslandError, match="b300-2 by run run9/trainer@other"):
        ops.start("isl")
    assert not [c for c in cluster.calls if c[0] == "start"]
    assert leases.holders["b300-2"] == "run9/trainer@other"


def test_a_replacement_rebuilds_the_health_watches_on_the_new_nodes(tmp_path, monkeypatch):
    built = []
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    real = ops._health_watches

    def watches(spec_):
        built.append([n.name for n in spec_.nodes])
        return real(spec_)

    monkeypatch.setattr(ops, "_health_watches", watches)
    assert ops.relaunch(None, "isl") == 1
    assert built == [["b300-0", "b300-3", "b300-2"]]


def test_release_node_ends_the_lease_of_the_holder_the_quarantine_names(tmp_path):
    """The command runs under its own lease holder; the lease it ends is the one of the run
    that quarantined the node (the record's ``lease_holder``)."""
    run_leases = FakeLeases(held={n: "run1/trainer@launcher-a" for n in NAMES})
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], leases=run_leases)
    assert run_leases.kept == ["b300-1"]
    cluster.unreachable.discard("b300-1")
    cli = FakeLeases(run_id="yeto-cluster-release-node")
    cli.holders, cli.kept = run_leases.holders, run_leases.kept  # one lease store
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lines.append, leases=cli) == 0
    assert "b300-1" not in run_leases.holders and "not released" not in lines[-1]


def test_the_islands_own_node_leases_hold_the_spares_and_the_shared_quarantine(tmp_path):
    """On the production path the spares get no leases of their own: the island's
    ``spec.leases`` (``trainer_leases``) is used, and its store holds the shared quarantine."""
    import dataclasses

    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=Clock(), health_every_s=0, unreachable_polls=2)
    leases = FakeLeases()
    leases.objects = DirectoryObjectStore(tmp_path / "lease-store")
    s = dataclasses.replace(islands_spec(), leases=leases)
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    assert spares.leases is None and spares.records.shared is None
    ops.add("isl", s, spares=spares)
    assert spares.leases is leases and spares.records.shared is leases.objects
    ops.start("isl")
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    assert leases.objects.read("nodes/b300-1/quarantine.json") is not None
    assert leases.kept == ["b300-1"]


# ------------------------------------------------------------------ REV-885c 2: the leases of a restarted launcher
LEASE_STORE = "s3://leases/site"


def _lease_run(tmp_path, *, extra=(spare(3),)):
    """Island isl of run1 picks b300-0..2 by real node leases (lease record in the run dir,
    real NodeLeases on a fake S3 store); b300-1 loses power and b300-3 takes its place."""
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    args = SimpleNamespace(cluster_prefix="run1", rl_node_inventory_nodes=[*islands_spec().nodes, *extra])

    def leases_of(identity="aaaa"):
        return NodeLeases(LEASE_STORE, "run1/trainer", identity=identity, objects=objects, clock=clock, monotonic=clock)

    def island(leases):
        nodes = sti.lease_trainer_nodes(args, islands_spec().nodes, 3, leases, name="isl", wait_s=0)
        return dataclasses.replace(islands_spec(), nodes=tuple(nodes), leases=leases)

    cluster = Cluster([*NAMES, *[n.name for n in extra]])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    first = leases_of()
    spares, store = make_spares(tmp_path, islands_spec(), list(extra))
    ops.add("isl", island(first), spares=spares)
    ops.start("isl")
    cluster.lose("b300-1")
    rec = ops.islands["isl"]
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    assert ops.relaunch(None, "isl") == 1 and started(cluster, 1) == ["b300-0", "b300-3", "b300-2"]
    assert sti._lease_record(args)["islands"]["isl"] == ["b300-0", "b300-3", "b300-2"]  # (a): the spare recorded
    cluster.unreachable.discard("b300-1")  # repaired, then released by hand
    inv = _inventory(tmp_path, [*islands_spec().nodes, *extra])
    cli = NodeLeases(LEASE_STORE, isp.RELEASE_NODE_RUN, objects=objects, clock=clock, monotonic=clock)
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lambda m: None, leases=cli) == 0
    rec.lock.close()
    for proc in rec.job.procs:  # the launcher goes (no --keep); the island runs on
        proc.terminate()

    def restart(identity="aaaa"):
        leases = leases_of(identity)
        again = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
        fresh, _ = make_spares(tmp_path, islands_spec(), list(extra))
        again.add("isl", island(leases), spares=fresh)
        assert again.start("isl") == 1  # reattached
        return again, leases

    return SimpleNamespace(objects=objects, clock=clock, args=args, leases_of=leases_of, restart=restart,
                           cluster=cluster)


def test_a_restarted_launcher_renews_the_lease_of_its_spare(tmp_path):
    from yeto.launch.node_lease import NodeLeases

    run = _lease_run(tmp_path)
    again, leases = run.restart()
    assert sorted(leases.held()) == ["b300-0", "b300-2", "b300-3"]
    run.clock.t += 3600  # an hour on: the spare's lease is this run's, renewed, not taken over
    leases.renew_due()
    other = NodeLeases(LEASE_STORE, "run9/trainer", objects=run.objects, clock=run.clock, monotonic=run.clock)
    got = other.acquire("b300-3")
    assert not got.ok and got.holder == leases.holder_id


def test_a_restarted_launcher_does_not_hold_a_released_old_node(tmp_path):
    from yeto.launch import ssh_trainer_island as sti

    run = _lease_run(tmp_path)
    # a stale record (the launcher went before it wrote the spare in): the old node is leased again
    sti._record_island_nodes(run.args, "isl", islands_spec().nodes)
    again, leases = run.restart()
    assert "b300-1" not in leases.held() and sorted(leases.held()) == ["b300-0", "b300-2", "b300-3"]
    assert sti._lease_record(run.args)["islands"]["isl"] == ["b300-0", "b300-3", "b300-2"]  # (b): set right again


def test_the_island_reattaches_when_another_run_took_its_old_node(tmp_path):
    from yeto.launch.node_lease import NodeLeases

    run = _lease_run(tmp_path)
    other = NodeLeases(LEASE_STORE, "run9/trainer", objects=run.objects, clock=run.clock, monotonic=run.clock)
    assert other.acquire("b300-1").ok
    again, leases = run.restart()
    assert sorted(leases.held()) == ["b300-0", "b300-2", "b300-3"] and other.held() == ["b300-1"]
    assert run.cluster.count("follow", 1) == 3 and ("follow", "b300-3", 1) in run.cluster.calls  # reattached


# ------------------------------------------------------------------ REV-885c 3: release-node away from the run dir
def test_release_node_away_from_the_runs_ledger_says_where_the_row_is_and_the_launcher_closes_it(
        tmp_path, monkeypatch):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1", "run1/isl-i1"]
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    elsewhere = tmp_path / "elsewhere" / "resources.jsonl"  # another machine: no row of this run
    with monkeypatch.context() as m:
        m.setenv("YETO_RESOURCE_LEDGER", str(elsewhere))
        assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                     objects=store, kill_grace_s=1, say=lines.append) == 0
    assert (f"WARN: its owed ledger row run1/isl-i0-owed-b300-1 is not open in {elsewhere}: the row is in the "
            "run dir of the machine that ran the launcher of run run1 (runs/run1/resources.jsonl)") in lines[-1]
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1", "run1/isl-i1"]  # still open in the run's ledger
    cluster.lose("b300-3")  # the spare is lost; b300-1 (released elsewhere) is picked again
    rec = ops.islands["isl"]
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 1)
    assert ops.relaunch(None, "isl") == 2 and started(cluster, 2) == ["b300-0", "b300-1", "b300-2"]
    assert open_rows(tmp_path) == ["run1/isl-i1-owed-b300-3", "run1/isl-i2"]  # billed once
    rows = [json.loads(ln) for ln in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    closed = [r for r in rows if r.get("id") == "run1/isl-i0-owed-b300-1" and r.get("op") == "release"]
    assert closed and closed[-1]["attrs"]["reason"] == "released elsewhere"



# ------------------------------------------------------------------ REV-885c 4: an unreadable shared quarantine
class _FlakyStore:
    """A lease store whose reads (and, with ``list_fails``, listings) fail."""

    def __init__(self, inner, *, read_fails=(), list_fails=False):
        self.inner, self.read_fails, self.list_fails = inner, set(read_fails), list_fails

    def read(self, key):
        if any(key == f"nodes/{n}/quarantine.json" for n in self.read_fails):
            raise OSError("connection reset")
        return self.inner.read(key)

    def list(self, prefix):
        if self.list_fails and prefix == "nodes/":  # the quarantine listing (node_lease lists deeper)
            raise OSError("list timed out")
        return self.inner.list(prefix)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_an_unreadable_shared_quarantine_keeps_the_spare_out(tmp_path):
    shared = DirectoryObjectStore(tmp_path / "leases")
    shared.write("nodes/b300-3/quarantine.json", b"{not json")  # a broken record
    flaky = _FlakyStore(shared, read_fails={"b300-4"})
    other = IslandSpares(pool=(b300(0), b300(2), spare(3), spare(4)),  # rank 1 is lost (not in the pool)
                         records=NodeRecords(DirectoryObjectStore(tmp_path / "store2"), run_id="run2",
                                             log=lambda m: None, shared=flaky))
    cluster = Cluster([*NAMES, "b300-3", "b300-4"])
    pick = isp.pick_spare(other, "isl", spec(3).nodes, [1], ops=cluster)
    assert pick.node is None
    assert "quarantine record unreadable: not a JSON object" in pick.skipped["b300-3"]
    assert "quarantine record unreadable: OSError: connection reset" in pick.skipped["b300-4"]
    flaky.read_fails.clear()  # the next round reads it again
    assert isp.pick_spare(other, "isl", spec(3).nodes, [1], ops=cluster).node.name == "b300-4"


def test_an_unreadable_shared_quarantine_keeps_the_trainer_node_out(tmp_path, monkeypatch, capsys):
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    inner, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    inner.write("nodes/b300-0/quarantine.json", json.dumps({"run": "run7", "island": "isl0"}).encode())
    flaky = _FlakyStore(inner, read_fails={"b300-1"}, list_fails=True)
    monkeypatch.setattr(sti, "_lease_record", lambda args: {})
    monkeypatch.setattr(sti, "_record_island_nodes", lambda args, name, nodes: None)
    leases = NodeLeases(LEASE_STORE, "run2/trainer", objects=flaky, clock=clock, monotonic=clock)
    picked = sti.lease_trainer_nodes(SimpleNamespace(cluster_prefix="run2"), [b300(i) for i in range(4)], 2,
                                     leases, name="isl", wait_s=0)
    assert [n.name for n in picked] == ["b300-2", "b300-3"]  # b300-0 quarantined, b300-1 unreadable: out
    err = capsys.readouterr().err
    assert "the lease store's quarantine list failed (OSError: list timed out); each node's record is read" in err
    assert "b300-1 (quarantine record unreadable: OSError: connection reset)" in err
    assert "b300-0 by run run7 island isl0" in err


def test_a_store_error_on_the_quarantine_of_a_running_node_is_no_loss(tmp_path):
    shared = DirectoryObjectStore(tmp_path / "leases")
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0, unreachable_polls=2)
    lines = []
    ops.log = lines.append
    s = islands_spec()
    spares = IslandSpares(pool=(*s.nodes, spare(3)), records=NodeRecords(
        DirectoryObjectStore(tmp_path / "store"), run_id="run1", log=lambda m: None,
        shared=_FlakyStore(shared, read_fails={"b300-1"})))
    ops.add("isl", s, spares=spares)
    assert ops.start("isl") == 0 and started(cluster, 0) == NAMES  # not replaced on a store hiccup
    assert any("quarantine record of b300-1 could not be read" in ln for ln in lines)



# ------------------------------------------------------------------ REV-885c 5: down() skips a lost node
def test_down_does_not_wait_for_a_lost_node(tmp_path):
    cluster = Cluster(NAMES)
    ops = make_ops(tmp_path, cluster, clock=Clock(), health_every_s=0, unreachable_polls=2)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    cluster.kill_rc["b300-0"] = 1  # the release at the loss is not confirmed on b300-0
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    assert 1 in spares.lost and 0 not in ops.islands["isl"].released
    cluster.kill_rc.clear()
    before = len(cluster.calls)
    assert ops.down("isl") is True  # confirmed on the nodes that answer; the lost one is not asked
    assert not [c for c in cluster.calls[before:] if c[1] == "b300-1"]
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1"]  # its stop stays owed (cleanup, release-node)


# ------------------------------------------------------------------ REV-885c 6: the spare's ports
def test_a_spares_health_check_reads_the_islands_job_ports(tmp_path, monkeypatch):
    import dataclasses

    from test_ssh_island_spares import FakeChecker
    from yeto.hw import node_health

    seen = []

    class Recording(FakeChecker):
        def script(self, hcas):
            seen.append(list(hcas))
            return super().script(hcas)

    monkeypatch.setattr(isp, "_checker_for", lambda family, **kw: Recording())
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0, unreachable_polls=2)
    s = dataclasses.replace(islands_spec(), env={"NCCL_IB_HCA": "=mlx5_4,mlx5_5"})  # the island's job env wins
    extra = dataclasses.replace(spare(3), env={"NCCL_IB_HCA": "mlx5_0"})
    spares, _ = make_spares(tmp_path, s, [extra])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isl"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    assert ops.relaunch(None, "isl") == 1
    assert seen and all(h == node_health.hcas_of("=mlx5_4,mlx5_5") for h in seen)
    assert isp.hcas_of(extra) == node_health.hcas_of("mlx5_0")  # its own env alone (release-node)


def test_release_node_dry_run_names_the_lease_it_ends_and_the_records_it_removes(tmp_path):
    shared = DirectoryObjectStore(tmp_path / "leases")
    run_leases = FakeLeases(held={n: "run1/trainer@launcher-a" for n in NAMES})
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], leases=run_leases)
    spares.records.shared = shared
    spares.records.quarantine("b300-1", island="isl", reason="lost", lease_holder=run_leases.holder_id)
    cli = FakeLeases(run_id=isp.RELEASE_NODE_RUN)
    cli.holders, cli.kept, cli.store_uri = run_leases.holders, run_leases.kept, "s3://leases/site"
    cli.objects = shared
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lines.append, leases=cli, dry_run=True) == 0
    text = "\n".join(lines)
    assert (f"would remove fleet/nodes/b300-1/quarantine.json in the run store {tmp_path / 'store'} and "
            "nodes/b300-1/quarantine.json in the lease store s3://leases/site") in text
    assert "would end the lease of run1/trainer@launcher-a on b300-1 (a kept one at once" in text
    assert run_leases.holders["b300-1"] == "run1/trainer@launcher-a"  # a dry run ends nothing


def test_release_node_of_a_broken_record_needs_force(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    cluster.unreachable.discard("b300-1")
    store.write("fleet/nodes/b300-1/quarantine.json", b"{truncated")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    lines = []
    common = dict(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster, objects=store,
                  kill_grace_s=1, say=lines.append)
    assert isp.release_node_main(**common) == 1
    assert "its quarantine record is unreadable (quarantine record unreadable: not a JSON object): the stops it " \
           "still owes cannot be known" in "\n".join(lines)
    assert "release it with --force" in lines[-1] and store.read("fleet/nodes/b300-1/quarantine.json") is not None
    assert isp.release_node_main(**common, force=True) == 0
    assert any("released without running any owed stop" in ln for ln in lines)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None


def test_a_lost_node_back_from_a_partition_gets_its_owed_stop_at_the_next_check(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])  # no spare: the island waits
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1"]
    cluster.unreachable.discard("b300-1")  # the partition heals: the old container still runs there
    clock.t += 61
    assert ops.relaunch(None, "isl") is None  # the backoff check: still no spare, the owed stop retried
    join_stops(spares)
    assert ("kill", "b300-1", 0) in cluster.calls
    assert open_rows(tmp_path) == []  # stopped: no more billing
    assert spares.records.own_quarantine("b300-1")["owed"] == []  # it stays quarantined until release-node



def test_a_confirmed_stop_owes_no_row_even_when_its_record_is_not_updated(tmp_path, monkeypatch):
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, store = make_spares(tmp_path, s, [spare(3)])

    def settle(*a, **kw):
        raise OSError("store unavailable")

    monkeypatch.setattr(spares.records, "settle", settle)
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], READING)
    join_stops(spares)
    assert json.loads(store.read("fleet/nodes/b300-2/quarantine.json"))["owed"]  # the record still lists it
    assert str(ops.job_status("isl", 0)) == "FAILED" and ops.relaunch(None, "isl") == 1
    assert open_rows(tmp_path) == ["run1/isl-i1"]  # the stop went through: no owed row


def test_a_health_reading_of_a_node_back_in_its_rank_after_two_replacements_is_stale(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    rec = ops.islands["isl"]
    old = _real_reading()
    ops._health_reading(rec, 1, old)  # of b300-1 at rank 1, before it was replaced
    rec.health_pending = []
    assert ops.relaunch(None, "isl") == 1  # b300-3 at rank 1
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lambda m: None) == 0
    cluster.lose("b300-3")
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 1)
    assert ops.relaunch(None, "isl") == 2 and rec.spec.nodes[1].name == "b300-1"  # back in rank 1
    ops._node_health_lost(rec, 1, ["XID 79 on GPU 3"], old)
    assert spares.lost == {} and str(ops.job_status("isl", 2)) == "RUNNING"


# ------------------------------------------------------------------ REV-885d M1: a settle racing release-node
def test_a_settle_racing_release_node_never_writes_the_quarantine_back(tmp_path):
    """REV-885d's order: the settle has read the record; release-node removes it and run9
    leases the node; then the settle goes on. Nothing of run1's quarantine comes back, in
    the run store or in the lease store, so run9 never sees its node as quarantined."""
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch.node_lease import NodeLeases

    store, shared = DirectoryObjectStore(tmp_path / "store"), DirectoryObjectStore(tmp_path / "leases")
    clock = Clock()
    lease_objects = S3ObjectStore("s3://leases/site", client=FakeS3())
    records = NodeRecords(store, run_id="run1", log=lambda m: None, shared=shared)
    records.quarantine("b300-1", island="isl", reason="lost",
                       owed=[{"run_id": "run1", "island": "isl", "incarnation": 0, "rank": 1, "container": None}])
    cli = NodeRecords(store, run_id="", log=lambda m: None, shared=shared)
    run9 = NodeLeases("s3://leases/site", "run9/trainer", objects=lease_objects, clock=clock, monotonic=clock)
    real_read, armed = records._read, [True]

    def racing(key):
        doc = real_read(key)
        if key == "fleet/nodes/b300-1/quarantine.json" and armed and armed.pop():
            cli.release("b300-1")  # release-node, between the settle's read and its write
            assert run9.acquire("b300-1").ok
        return doc

    records._read = racing
    records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None
    assert not store.list("fleet/nodes/b300-1/settled/")  # the tombstone is newer: no marker either
    assert shared.read("nodes/b300-1/quarantine.json") is None
    run9_records = NodeRecords(DirectoryObjectStore(tmp_path / "store9"), run_id="run9", log=lambda m: None,
                               shared=shared)
    assert run9_records.quarantined("b300-1") is None  # run9's own node is not "quarantined by run1"
    assert run9.held() == ["b300-1"]


def test_an_owed_stop_retry_that_races_release_node_leaves_the_node_released(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])  # no spare: the island waits
    shared = DirectoryObjectStore(tmp_path / "leases")
    spares.records.shared = shared
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    cluster.unreachable.discard("b300-1")
    cli = NodeRecords(store, run_id="", log=lambda m: None, shared=shared)
    real = cluster.run

    def run(node, script, *, timeout=300.0):
        if node.name == "b300-1" and "kill -TERM" in script:  # release-node runs while the retry kills
            cli.release("b300-1")
        return real(node, script, timeout=timeout)

    cluster.run = run
    clock.t += 61
    ops.relaunch(None, "isl")  # the retry runs in the background: b300-1 may be picked again at once
    join_stops(spares)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None
    assert shared.read("nodes/b300-1/quarantine.json") is None
    assert not store.list("fleet/nodes/b300-1/settled/")


@pytest.mark.parametrize("skew_s", [-3600.0, 3600.0])
def test_a_settle_compares_no_clock(tmp_path, skew_s):
    """REV-885e L1: release-node's clock is an hour ahead of or behind the launcher's. The
    node is released, picked again and lost again (incarnation 1); its confirmed stop is
    settled either way, so it owes no row (no second bill) and nothing is stopped twice."""
    store, clock = DirectoryObjectStore(tmp_path / "store"), Clock()
    clock.t = 1_000_000.0
    launcher = NodeRecords(store, run_id="run1", log=lambda m: None, wall=clock)
    cli = NodeRecords(store, run_id="", log=lambda m: None, wall=lambda: clock.t + skew_s)
    launcher.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    clock.t += 10
    cli.release("b300-1")
    clock.t += 10
    launcher.quarantine("b300-1", island="isl", reason="lost again", owed=[isp.owed_stop("run1", "isl", 1, 1, False)])
    launcher.settle("b300-1", run_id="run1", island="isl", incarnation=1)
    assert launcher.own_quarantine("b300-1")["owed"] == []
    assert len(store.list("fleet/nodes/b300-1/settled/")) == 1


def test_a_release_between_the_marker_and_the_second_read_leaves_no_marker(tmp_path):
    """REV-885e L2: release-node runs after the settle created its marker and before it read
    the record again (its listing of the markers missed this one): the settle removes it."""
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    cli = NodeRecords(store, run_id="", log=lambda m: None)
    real = records._create

    def create(key, doc):
        made = real(key, doc)
        cli.release("b300-1")  # its marker listing ran before this create; its record delete after
        return made

    records._create = create
    records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is None
    assert not store.list("fleet/nodes/b300-1/settled/")


def test_a_settle_keeps_its_marker_when_the_record_only_gained_a_stop(tmp_path):
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    real = records._create

    def create(key, doc):
        made = real(key, doc)
        records.quarantine("b300-1", island="isl", reason="lost again", owed=[isp.owed_stop("run1", "isl", 1, 1, False)])
        return made

    records._create = create
    records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    assert [o["incarnation"] for o in records.own_quarantine("b300-1")["owed"]] == [1]


def test_settle_markers_of_different_stops_never_share_a_name():
    keys = {NodeRecords.settled_key("b300-1", run, island, k)
            for run, island, k in [("a-b", "c", 1), ("a", "b-c", 1), ("a.b", "c", 1), ("a", "b.c", 1),
                                   ("a", "b", 11), ("a-b-c", "i1", 1)]}
    assert len(keys) == 6
    assert all(k.startswith("fleet/nodes/b300-1/settled/") and k.endswith(".json") for k in keys)


# ------------------------------------------------------------------ REV-885d M2: a launcher gone mid-replacement
def _died_mid_replacement(tmp_path, monkeypatch):
    """b300-1 loses power; the launcher goes while it starts the spare (its preflight), after
    the spares journal named the new nodes; b300-1 is then repaired and released by hand."""
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    args = SimpleNamespace(cluster_prefix="run1", rl_node_inventory_nodes=[*islands_spec().nodes, spare(3)])

    def leases_of(run="run1/trainer", identity="aaaa"):
        return NodeLeases(LEASE_STORE, run, identity=identity, objects=objects, clock=clock, monotonic=clock)

    def island(leases):
        nodes = sti.lease_trainer_nodes(args, islands_spec().nodes, 3, leases, name="isl", wait_s=0)
        return dataclasses.replace(islands_spec(), nodes=tuple(nodes), leases=leases)

    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    spares, store = make_spares(tmp_path, islands_spec(), [spare(3)])
    ops.add("isl", island(leases_of()), spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    cluster.lose("b300-1")
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)
    real_launch = sti.SshTrainerIslands.launch

    def died(self, incarnation, *a, **kw):
        raise SystemExit("the launcher went during the spare's preflight")

    monkeypatch.setattr(sti.SshTrainerIslands, "launch", died)
    with pytest.raises(SystemExit):
        ops.relaunch(None, "isl")
    monkeypatch.setattr(sti.SshTrainerIslands, "launch", real_launch)
    rec.lock.close()
    for proc in rec.job.procs:
        proc.terminate()
    cluster.unreachable.discard("b300-1")
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3)])
    cli = NodeLeases(LEASE_STORE, isp.RELEASE_NODE_RUN, objects=objects, clock=clock, monotonic=clock)
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lambda m: None, leases=cli) == 0

    def restart():
        leases = leases_of()
        again = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
        fresh, _ = make_spares(tmp_path, islands_spec(), [spare(3)])
        again.add("isl", island(leases), spares=fresh)
        again.start("isl")
        return again, leases

    return SimpleNamespace(args=args, leases_of=leases_of, restart=restart, cluster=cluster)


def test_a_restart_after_a_launcher_gone_mid_replacement_does_not_hold_the_released_old_node(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti

    run = _died_mid_replacement(tmp_path, monkeypatch)
    assert sti._lease_record(run.args)["islands"]["isl"] == ["b300-0", "b300-3", "b300-2"]  # written before the start
    again, leases = run.restart()
    assert "b300-1" not in leases.held() and sorted(leases.held()) == ["b300-0", "b300-2", "b300-3"]
    assert run.leases_of("run9/trainer").acquire("b300-1").ok  # free for another run


def test_a_restart_after_a_launcher_gone_mid_replacement_is_not_refused_by_another_runs_old_node(
        tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti

    run = _died_mid_replacement(tmp_path, monkeypatch)
    sti._record_island_nodes(run.args, "isl", islands_spec().nodes)  # an older record, as before this fix
    assert run.leases_of("run9/trainer").acquire("b300-1").ok
    again, leases = run.restart()  # the spares journal names the island's nodes: not "not moving the island"
    assert sorted(leases.held()) == ["b300-0", "b300-2", "b300-3"]
    assert [n.name for n in again.islands["isl"].spec.nodes] == ["b300-0", "b300-3", "b300-2"]


# ------------------------------------------------------------------ REV-885d L4: a row closes after its stop
def _picked_again_after(tmp_path, release):
    """b300-1 lost (owed row open), b300-3 replaces it; ``release(...)`` frees b300-1 without
    its owed stop; then b300-3 is lost and b300-1 picked again."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    release(cluster, spares, store)
    assert open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1", "run1/isl-i1"]
    kills_before = cluster.count("kill", 0)
    rec = ops.islands["isl"]
    cluster.lose("b300-3")
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 1)
    return cluster, ops, kills_before


def _force_a_broken_record(tmp_path):
    def release(cluster, spares, store):
        store.write("fleet/nodes/b300-1/quarantine.json", b"{broken")
        inv = _inventory(tmp_path, [*spec(3).nodes, spare(3)])
        assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                     objects=store, kill_grace_s=1, say=lambda m: None, force=True) == 0
    return release


def test_after_force_over_a_broken_record_the_launcher_stops_the_old_incarnation_before_it_closes_the_row(tmp_path):
    cluster, ops, kills_before = _picked_again_after(tmp_path, _force_a_broken_record(tmp_path))
    assert ops.relaunch(None, "isl") == 2 and started(cluster, 2) == ["b300-0", "b300-1", "b300-2"]
    assert cluster.count("kill", 0) > kills_before  # i0's owed stop ran on b300-1 first
    assert open_rows(tmp_path) == ["run1/isl-i1-owed-b300-3", "run1/isl-i2"]


def test_a_row_whose_stop_fails_stays_open(tmp_path):
    def release(cluster, spares, store):
        _force_a_broken_record(tmp_path)(cluster, spares, store)
        cluster.kill_rc["b300-1"] = 1  # its old container cannot be removed

    cluster, ops, kills_before = _picked_again_after(tmp_path, release)
    ops.relaunch(None, "isl")
    assert "run1/isl-i0-owed-b300-1" in open_rows(tmp_path)  # still billed: cleanup tries it again


def test_records_kept_in_memory_a_restart_stops_the_old_incarnation_before_it_closes_the_row(tmp_path):
    def release(cluster, spares, store):
        spares.records = NodeRecords(None, run_id="run1", log=lambda m: None)  # a restart without a run store

    cluster, ops, kills_before = _picked_again_after(tmp_path, release)
    assert ops.relaunch(None, "isl") == 2
    assert cluster.count("kill", 0) > kills_before
    assert "run1/isl-i0-owed-b300-1" not in open_rows(tmp_path)


# ------------------------------------------------------------------ REV-885d: the guards of the surviving mutants
def test_a_stale_lease_is_given_back_but_not_one_another_island_picked_or_claims(tmp_path):
    """M4/M5/M17: two islands of the run in one launcher share its NodeLeases. Island isa's
    stale pick names b300-3 (gone), b300-4 (island isb's pick) and b300-5 (claimed by isb):
    only b300-3 is given back, and only it leaves ``assigned``."""
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    leases = NodeLeases(LEASE_STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    for n in ("b300-3", "b300-4", "b300-5"):
        assert leases.acquire(n).ok
    cluster = Cluster([*NAMES, "b300-3", "b300-4", "b300-5"])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0)
    s = dataclasses.replace(islands_spec(), leases=leases)
    spares, _ = make_spares(tmp_path, s, [spare(3), spare(4), spare(5)])
    ops.add("isa", s, spares=spares)
    leases.picks.update({"isa": [*NAMES, "b300-3", "b300-4", "b300-5"], "isb": ["b300-4"]})
    leases.assigned |= {*NAMES, "b300-3", "b300-4", "b300-5"}
    spares.records.claim("b300-5", "isb")
    ops._release_stale_leases(ops.islands["isa"])
    assert sorted(leases.held()) == ["b300-4", "b300-5"]  # b300-3 given back; isb's pick and claim kept
    assert "b300-3" not in leases.assigned and {"b300-4", *NAMES} <= leases.assigned
    assert leases.picks["isa"] == NAMES


def test_an_owed_stop_is_retried_only_while_the_island_waits(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1  # replaced at the first check: no wait, no retry
    join_stops(spares)
    assert ("unanswered", "b300-1", "kill") not in cluster.calls  # the dead node is not asked to stop
    assert not ops.islands["isl"].owed_retries


def test_an_islands_retry_runs_only_its_own_owed_stops(tmp_path):
    """M10: islands isa and isb of one run; b300-1 owes isa's stop. isb waits for a spare
    with b300-1 in its pool: its retries never run isa's stop."""
    import dataclasses

    cluster = Cluster([*NAMES, "b300-3", "b300-4", "b300-5", "b300-6"])
    clock = Clock()
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    sa = islands_spec()
    spa, _ = make_spares(tmp_path, sa, [spare(3)])
    ops.add("isa", sa, spares=spa)
    sb = dataclasses.replace(islands_spec(), nodes=(b300(4), b300(5), b300(6)))
    spb, _ = make_spares(tmp_path, sb, [b300(1)])  # its only spare is isa's lost node
    ops.add("isb", sb, spares=spb)
    ops.start("isa")
    ops.start("isb")
    cluster.lose("b300-1")
    assert wait_for(lambda: ops.islands["isa"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isa", 0)
    assert ops.relaunch(None, "isa") == 1  # isa: b300-3 in b300-1's place
    cluster.lose("b300-5")
    assert wait_for(lambda: ops.islands["isb"].job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isb", 0)
    assert ops.relaunch(None, "isb") is None and ops.waiting_spare("isb")  # b300-1 is quarantined
    cluster.unreachable.discard("b300-1")
    before = len(cluster.calls)
    clock.t += 61
    assert ops.relaunch(None, "isb") is None  # isb's backoff check
    join_stops(spb)
    assert "b300-1" not in ops.islands["isb"].owed_retries  # not isb's to stop
    assert not [c for c in cluster.calls[before:] if c[0] == "kill" and c[1] == "b300-1"]
    assert "run1/isa-i0-owed-b300-1" in open_rows(tmp_path)  # isa's to stop (release-node, its own retry)


def test_a_settled_stop_is_not_owed_again_after_a_restart(tmp_path):
    """M22: b300-2's own stop went through (settled); a restarted launcher, with no memory
    of that, releases incarnation 0 again and records no owed row for it."""
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, store = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    cluster.kill_rc["b300-0"] = 1  # the release at the loss is not confirmed: the restart releases again
    ops._node_health_lost(rec, 2, ["XID 79 on GPU 3"], READING)
    join_stops(spares)
    assert ("b300-2", 0) in spares.stopped and spares.records.own_quarantine("b300-2")["owed"] == []
    rec.lock.close()
    for proc in rec.job.procs:
        proc.terminate()
    cluster.kill_rc.clear()
    again = make_ops(tmp_path, cluster, health_every_s=0)
    fresh, _ = make_spares(tmp_path, islands_spec(), [spare(3)])
    again.add("isl", islands_spec(), spares=fresh)
    again.start("isl")
    assert not [r for r in open_rows(tmp_path) if r.endswith("-owed-b300-2")]


def test_a_settle_clears_only_that_nodes_stop_of_that_incarnation(tmp_path):
    records = NodeRecords(DirectoryObjectStore(tmp_path / "store"), run_id="run1", log=lambda m: None)
    rows = [{"run_id": "run1", "island": "isl", "incarnation": k} for k in (0, 1)]
    records.quarantine("b300-1", island="isl", reason="lost", owed=rows)
    records.quarantine("b300-2", island="isl", reason="lost", owed=rows[:1])
    records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    assert [o["incarnation"] for o in records.own_quarantine("b300-1")["owed"]] == [1]
    assert [o["incarnation"] for o in records.own_quarantine("b300-2")["owed"]] == [0]
    records.settle("b300-1", run_id="run1", island="other", incarnation=1)  # another island's: nothing
    assert [o["incarnation"] for o in records.own_quarantine("b300-1")["owed"]] == [1]
    records.release("b300-1")  # the record and its markers go; the tombstone stays
    assert [k.rsplit("/", 1)[-1] for k in records.objects.list("fleet/nodes/b300-1/")] == ["released.json"]


def test_a_confirmed_owed_stop_removes_the_pid_file_and_a_second_try_signals_nothing(tmp_path):
    """The owed stop, run by bash: it stops the incarnation's process group, removes its
    pid file and leaves a marker; a second try, and the ledger's plain kill script (the
    pid file gone, the state dir there), exit 0 without a pid. Without the marker the
    plain kill script exits 3 (the island may still run)."""
    import os
    import signal
    import subprocess
    import time as _time

    from yeto.launch.ssh_trainer_island import STATE_DIR, kill_script, remote_paths

    home = tmp_path / "home"
    sdir = home / STATE_DIR / "run1" / "isl-i0"
    sdir.mkdir(parents=True)
    # the island's process group, not a child of this test (a dead child stays a zombie
    # that ``kill -0`` still finds on Linux): its leader exits, the sleep is orphaned
    out = subprocess.run(["bash", "-c", "sleep 60 >/dev/null 2>&1 & echo $!"], start_new_session=True,
                         capture_output=True, text=True, timeout=30)
    pid = int(out.stdout)
    try:
        (sdir / "pgid").write_text(f"{os.getpgid(pid)}\n")
        env = {**os.environ, "HOME": str(home)}
        script = isp.owed_kill_script("run1", "isl", 0, 5, None)
        assert subprocess.run(["bash", "-c", script], env=env, timeout=30).returncode == 0
        assert not (sdir / "pgid").exists() and (sdir / "stopped").exists()
        assert subprocess.run(["bash", "-c", script], env=env, timeout=30).returncode == 0
        plain = kill_script(remote_paths("run1", "isl", 0), 2)
        assert subprocess.run(["bash", "-c", plain], env=env, timeout=30, capture_output=True).returncode == 0
        (sdir / "stopped").unlink()
        assert subprocess.run(["bash", "-c", plain], env=env, timeout=30, capture_output=True).returncode == 3
    finally:
        deadline = _time.monotonic() + 10
        while _time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            _time.sleep(0.05)
        else:  # still there: this test's own sleep
            os.kill(pid, signal.SIGKILL)


def test_a_failed_owed_stop_keeps_the_pid_file(tmp_path):
    import os
    import subprocess

    from yeto.launch.ssh_trainer_island import STATE_DIR

    home = tmp_path / "home"
    sdir = home / STATE_DIR / "run1" / "isl-i0"
    sdir.mkdir(parents=True)
    (sdir / "pgid").write_text("")  # no pid in it: the stop is not confirmed (exit 3)
    script = isp.owed_kill_script("run1", "isl", 0, 1, None)
    done = subprocess.run(["bash", "-c", script], env={**os.environ, "HOME": str(home)}, timeout=30,
                          capture_output=True)
    assert done.returncode == 3
    assert (sdir / "pgid").exists() and not (sdir / "stopped").exists()


def test_an_owed_stop_reads_the_record_again_and_skips_a_node_released_meanwhile(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])  # no spare: the island waits
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    rec, node = ops.islands["isl"], spares.pool[1]
    rows = spares.records.own_quarantine("b300-1")["owed"]  # read by the retry
    cluster.unreachable.discard("b300-1")
    NodeRecords(store, run_id="", log=lambda m: None).release("b300-1")  # release-node, before its kill
    ops._stop_owed(rec, node, rows)
    assert ("kill", "b300-1", 0) not in cluster.calls


def test_an_owed_stop_of_another_island_is_not_run(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])
    assert ops.relaunch(None, "isl") is None
    cluster.unreachable.discard("b300-1")
    doc = json.loads(store.read("fleet/nodes/b300-1/quarantine.json"))
    doc["owed"] = [{**o, "island": "isb"} for o in doc["owed"]]  # incarnation 0 of island isb
    store.write("fleet/nodes/b300-1/quarantine.json", json.dumps(doc).encode())
    assert not ops._run_owed_stop(ops.islands["isl"], spares.pool[1], 0, None, what="quarantined node")
    assert ("kill", "b300-1", 0) not in cluster.calls


def test_an_owed_stop_closes_its_row_even_when_its_settle_fails(tmp_path, monkeypatch):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])
    assert ops.relaunch(None, "isl") is None and open_rows(tmp_path) == ["run1/isl-i0-owed-b300-1"]

    def settle(*a, **kw):
        raise OSError("store unavailable")

    monkeypatch.setattr(spares.records, "settle", settle)
    cluster.unreachable.discard("b300-1")
    clock.t += 61
    ops.relaunch(None, "isl")
    join_stops(spares)
    assert ("kill", "b300-1", 0) in cluster.calls
    assert open_rows(tmp_path) == []  # stopped: the row closes
    assert spares.records.own_quarantine("b300-1")["owed"]  # the record still lists it: release-node runs it again


@pytest.mark.parametrize("how", ["down", "close"])
def test_the_owed_stop_retries_are_joined_when_the_island_goes(tmp_path, how):
    import threading
    import time as _time

    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    retry = threading.Thread(target=_time.sleep, args=(0.5,), daemon=True)
    retry.start()
    rec.owed_retries["b300-1"] = retry
    ops.down("isl") if how == "down" else ops.close()
    assert not retry.is_alive() and rec.owed_retries == {}


def test_a_start_gives_back_a_stale_lease_of_a_node_the_island_no_longer_runs_on(tmp_path):
    """A first start whose leases picked an old node for the island (a lease record older
    than its spares journal): the start gives it back and records the island's nodes."""
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    leases = NodeLeases(LEASE_STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    for n in (*NAMES, "b300-3"):
        assert leases.acquire(n).ok
    leases.picks["isl"] = [*NAMES, "b300-3"]
    leases.assigned |= {*NAMES, "b300-3"}
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0)
    s = dataclasses.replace(islands_spec(), leases=leases)
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    assert "b300-3" not in leases.held() and leases.picks["isl"] == NAMES


def test_a_stopped_node_owes_no_row_at_a_later_release_even_while_its_record_lists_it(tmp_path):
    """The stop of b300-2 went through but its settle marker was not written (the record
    still lists the stop): a release of that incarnation afterwards records no owed row."""
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    spares.lost[2] = "XID 79 on GPU 3"
    spares.records.quarantine("b300-2", island="isl", reason="lost",
                              owed=[isp.owed_stop("run1", "isl", 0, 2, False)])
    spares.stopped.add(("b300-2", 0))
    assert ops._release(rec, 0, skip=frozenset({"b300-2"}))
    assert not [r for r in open_rows(tmp_path) if "-owed-" in r]


def test_a_reattach_holds_its_nodes_leases_when_its_pick_did_not(tmp_path):
    """A restarted launcher whose leases hold the island's old nodes but not the spare that
    runs at rank 1 (an older pick): the reattach takes the spare's lease too."""
    import dataclasses

    run = _lease_run(tmp_path)
    leases = run.leases_of()
    for n in ("b300-0", "b300-2"):
        assert leases.acquire(n).ok
    again = make_ops(tmp_path, run.cluster, clock=run.clock, health_every_s=0, unreachable_polls=2)
    fresh, _ = make_spares(tmp_path, islands_spec(), [spare(3)])
    again.add("isl", dataclasses.replace(islands_spec(), leases=leases), spares=fresh)
    assert again.start("isl") == 1  # reattached
    assert sorted(leases.held()) == ["b300-0", "b300-2", "b300-3"]


def test_a_failed_start_on_a_spare_puts_the_lease_record_back(tmp_path, monkeypatch):
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    args = SimpleNamespace(cluster_prefix="run1", rl_node_inventory_nodes=[*islands_spec().nodes, spare(3)])
    leases = NodeLeases(LEASE_STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    nodes = sti.lease_trainer_nodes(args, islands_spec().nodes, 3, leases, name="isl", wait_s=0)
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0, unreachable_polls=2)
    spares, _ = make_spares(tmp_path, islands_spec(), [spare(3)])
    ops.add("isl", dataclasses.replace(islands_spec(), nodes=tuple(nodes), leases=leases), spares=spares)
    ops.start("isl")
    rec = ops.islands["isl"]
    cluster.lose("b300-1")
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 0)

    def failed(self, incarnation, *a, **kw):
        raise RuntimeError("preflight on the spare failed")

    monkeypatch.setattr(sti.SshTrainerIslands, "launch", failed)
    assert ops.relaunch(None, "isl") is None and ops.waiting_spare("isl")
    assert sti._lease_record(args)["islands"]["isl"] == NAMES  # back to the nodes it runs on
    assert leases.picks["isl"] == NAMES and [n.name for n in rec.spec.nodes] == NAMES


# ------------------------------------------------------------------ REV-885e
def test_a_release_writes_no_tombstone_into_the_lease_store(tmp_path):
    store, shared = DirectoryObjectStore(tmp_path / "store"), DirectoryObjectStore(tmp_path / "leases")
    records = NodeRecords(store, run_id="run1", log=lambda m: None, shared=shared)
    records.quarantine("b300-1", island="isl", reason="lost")
    assert shared.read("nodes/b300-1/quarantine.json") is not None
    records.release("b300-1")
    assert not shared.list("nodes/b300-1/")  # nothing left there: no one reads a tombstone in the lease store


class _FailingDeletes(DirectoryObjectStore):
    def delete(self, key):
        raise OSError("store unavailable")


def test_release_node_reports_a_store_that_fails_with_exit_1_and_no_traceback(tmp_path):
    store = _FailingDeletes(tmp_path / "store")
    NodeRecords(store, run_id="run1", log=lambda m: None).quarantine("b300-1", island="isl", reason="lost")
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3)])
    said: list[str] = []
    code = isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"),
                                 ssh=Cluster([*NAMES, "b300-3"]), objects=store, kill_grace_s=1, say=said.append)
    assert code == 1
    assert any("stays quarantined" in line and "could not be removed" in line and "OSError" in line for line in said)


class _UnlistedStore(DirectoryObjectStore):
    def list(self, prefix):
        if "/settled/" in prefix:
            raise OSError("list refused")
        return super().list(prefix)


def test_a_spare_whose_start_failed_is_quarantined_with_the_lease_holder(tmp_path):
    leases = FakeLeases()
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)], leases=leases)
    real = cluster.run

    def run(node, script, *, timeout=300.0):
        if node.name == "b300-3" and script.startswith("docker pull"):
            import subprocess

            return subprocess.CompletedProcess([], 1, stdout="", stderr="no space left on device")
        return real(node, script, timeout=timeout)

    cluster.run = run
    assert ops.relaunch(None, "isl") is None
    assert json.loads(store.read("fleet/nodes/b300-3/quarantine.json"))["lease_holder"] == leases.holder_id


class _LeaseStoreDown(FakeLeases):
    def acquire(self, node):
        if node.name == "b300-3":
            raise OSError("lease store unavailable")
        return super().acquire(node)


def test_a_pick_whose_lease_step_raises_gives_its_claim_back_and_takes_the_next(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)], leases=_LeaseStoreDown())
    assert ops.relaunch(None, "isl") == 1
    assert [n.name for n in ops.islands["isl"].spec.nodes] == ["b300-0", "b300-4", "b300-2"]
    assert store.read("fleet/nodes/b300-3/claim.json") is None


def test_the_picks_so_far_are_given_back_when_a_later_pick_raises(tmp_path, monkeypatch):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)])
    spares.lost[2] = "lost too"
    real, calls = isp.pick_spare, []

    def pick_spare(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("the run store went away")
        return real(*a, **kw)

    monkeypatch.setattr(isp, "pick_spare", pick_spare)
    with pytest.raises(RuntimeError):
        ops._replace(ops.islands["isl"], {"b300-3": (True, "healthy"), "b300-4": (True, "healthy")})
    assert store.read("fleet/nodes/b300-3/claim.json") is None and store.read("fleet/nodes/b300-4/claim.json") is None


def test_a_stale_lease_of_a_quarantined_node_is_kept(tmp_path):
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    leases = NodeLeases(LEASE_STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    for n in (*NAMES, "b300-3"):
        assert leases.acquire(n).ok
    leases.picks["isl"] = [*NAMES, "b300-3"]
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0)
    s = dataclasses.replace(islands_spec(), leases=leases)
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    spares.records.quarantine("b300-3", island="isl", reason="lost")
    ops.add("isl", s, spares=spares)
    ops._release_stale_leases(ops.islands["isl"])
    assert "b300-3" in leases.held() or leases.holder(spare(3)) == leases.holder_id
    assert leases.holder(spare(3)) == leases.holder_id  # its lease stays this run's until release-node


def test_the_owed_stop_quotes_its_trap_for_any_island_name(tmp_path):
    import os
    import subprocess

    from yeto.launch.ssh_trainer_island import STATE_DIR

    home = tmp_path / "home"
    sdir = home / STATE_DIR / "run1" / "is'l-i0"
    sdir.mkdir(parents=True)
    out = subprocess.run(["bash", "-c", "sleep 60 >/dev/null 2>&1 & echo $!"], start_new_session=True,
                         capture_output=True, text=True, timeout=30)
    pid = int(out.stdout)
    try:
        (sdir / "pgid").write_text(f"{os.getpgid(pid)}\n")
        script = isp.owed_kill_script("run1", "is'l", 0, 5, None)
        done = subprocess.run(["bash", "-c", script], env={**os.environ, "HOME": str(home)}, timeout=30,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert not (sdir / "pgid").exists() and (sdir / "stopped").exists()
    finally:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            import signal

            os.kill(pid, signal.SIGKILL)


def test_run_store_quarantines_and_the_trainer_pick_exclusions(tmp_path):
    from yeto.launch import node_quarantine as nq

    run_store, lease_store = DirectoryObjectStore(tmp_path / "store"), DirectoryObjectStore(tmp_path / "leases")
    NodeRecords(run_store, run_id="run1", log=lambda m: None).quarantine("b300-1", island="isl", reason="lost")
    run_store.write("fleet/nodes/b300-2/quarantine.json", b"{broken")
    lease_store.write("nodes/b300-0/quarantine.json", json.dumps({"run": "run9", "island": "x", "reason": "r"}).encode())
    docs = nq.run_store_quarantines(run_store, NAMES)
    assert set(docs) == {"b300-1", "b300-2"} and docs["b300-2"]["unreadable"]
    excluded, both, note = nq.trainer_pick_excluded(NAMES + ["b300-3"], lease_objects=lease_store,
                                                    run_objects=run_store, assigned={"b300-3"})
    assert excluded == {"b300-0", "b300-1", "b300-2", "b300-3"} and set(both) == {"b300-0", "b300-1", "b300-2"}
    assert "b300-0 by run run9 island x" in note and "release-node" in note
    assert nq.trainer_pick_excluded(NAMES) == (set(), {}, None)

    class Unlisted(DirectoryObjectStore):
        def list(self, prefix):
            raise OSError("no list")

    lines: list[str] = []
    docs = nq.run_store_quarantines(Unlisted(tmp_path / "store"), NAMES, log=lines.append)
    assert set(docs) == {"b300-1", "b300-2"} and "each node's record is read" in lines[0]

    class Unread(DirectoryObjectStore):
        def read(self, key):
            raise OSError("no read")

    assert nq.run_store_quarantines(Unread(tmp_path / "store"), ["b300-1"])["b300-1"]["read_error"]


def test_a_spare_that_cannot_take_the_place_is_quarantined_with_the_lease_holder(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti

    leases = FakeLeases()
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], leases=leases)
    real = sti.rendezvous_address

    def address(node):
        if node.name == "b300-3":
            raise sti.SshIslandError("node b300-3: no rendezvous address")
        return real(node)

    monkeypatch.setattr(sti, "rendezvous_address", address)
    assert ops.relaunch(None, "isl") is None
    doc = json.loads(store.read("fleet/nodes/b300-3/quarantine.json"))
    assert "cannot take a node's place" in doc["reason"] and doc["lease_holder"] == leases.holder_id


def test_a_release_and_a_new_quarantine_between_the_marker_and_the_second_read_leave_no_marker(tmp_path):
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    cli = NodeRecords(store, run_id="", log=lambda m: None)
    real = records._create

    def create(key, doc):
        made = real(key, doc)
        cli.release("b300-1")  # released, picked again and lost again in incarnation 1
        records.quarantine("b300-1", island="isl", reason="lost again", owed=[isp.owed_stop("run1", "isl", 1, 1, False)])
        return made

    records._create = create
    records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    assert not store.list("fleet/nodes/b300-1/settled/")  # the old stop's marker is not left behind
    assert [o["incarnation"] for o in records.own_quarantine("b300-1")["owed"]] == [1]


def test_a_settle_of_a_stop_the_record_never_listed_leaves_no_marker(tmp_path):
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    records.settle("b300-1", run_id="run1", island="isl", incarnation=5)
    records.settle("b300-2", run_id="run1", island="isl", incarnation=0)  # no record at all
    assert not store.list("fleet/nodes/b300-1/settled/") and not store.list("fleet/nodes/b300-2/")


class _Unread(DirectoryObjectStore):
    def read(self, key):
        if key.endswith("quarantine.json"):
            raise OSError("read refused")
        return super().read(key)


def test_release_node_reports_a_ledger_that_fails_with_exit_1_and_no_traceback(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_ledger

    store = DirectoryObjectStore(tmp_path / "store")
    NodeRecords(store, run_id="run1", log=lambda m: None).quarantine(
        "b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])

    def broken(**kw):
        raise OSError("ledger not writable")

    monkeypatch.setattr(ssh_trainer_ledger, "release_owed", broken)
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3)])
    said: list[str] = []
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"),
                                 ssh=Cluster([*NAMES, "b300-3"]), objects=store, kill_grace_s=1, say=said.append) == 1
    assert any("stays quarantined" in line and "ledger not writable" in line for line in said)
    assert store.read("fleet/nodes/b300-1/quarantine.json") is not None


# ------------------------------------------------------------------ REV-920
def _rollback_unstopped(tmp_path, *, unstopped="b300-3", extra=None):
    """b300-1 lost; spare b300-3 takes rank 1; incarnation 1 starts on b300-0 and b300-3,
    then its start on b300-2 fails; the stop of incarnation 1 on ``unstopped`` ends rc 1."""
    leases = FakeLeases()
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=extra or [spare(3), spare(4)], leases=leases)
    real = cluster.start

    def start(node, script):
        if node.name == "b300-2" and "setsid bash -c" in script and cluster._k(script) == 1:
            raise RuntimeError("ssh start refused on rank 2")
        return real(node, script)

    cluster.start = start
    cluster.kill_rc[unstopped] = 1
    assert ops.relaunch(None, "isl") is None
    cluster.start = real
    return SimpleNamespace(cluster=cluster, ops=ops, spares=spares, store=store, clock=clock, leases=leases)


def test_a_spare_whose_rollback_stop_is_not_confirmed_is_lost_owing_that_stop(tmp_path):
    run = _rollback_unstopped(tmp_path)
    doc = run.spares.records.own_quarantine("b300-3")
    assert doc is not None and [(o["island"], o["incarnation"]) for o in doc["owed"]] == [("isl", 1)]
    assert doc["lease_holder"] == run.leases.holder_id and "b300-3" in getattr(run.leases, "kept", [])
    assert run.spares.records.holder("b300-3") is None  # M1: the record keeps every island off it, not a claim
    import dataclasses

    sb = dataclasses.replace(islands_spec(), nodes=(b300(4), b300(5), b300(6)))
    other, _ = make_spares(tmp_path, sb, [spare(3)])
    pick = isp.pick_spare(other, "isb", list(sb.nodes), {1: "lost"}, ops=run.cluster,
                          health={"b300-3": (True, "healthy")})
    assert pick.node is None and "quarantined" in pick.skipped["b300-3"]


def test_a_lost_rollback_spare_is_pickable_again_once_released(tmp_path):
    """REV-920 P1: the spare recovers, release-node runs its owed stop, the island goes down:
    another island of the run takes it."""
    import dataclasses

    run = _rollback_unstopped(tmp_path)
    run.cluster.kill_rc.clear()
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3), spare(4)])
    assert isp.release_node_main(inventory=inv, name="b300-3", store=str(tmp_path / "store"), ssh=run.cluster,
                                 objects=run.store, kill_grace_s=1, say=lambda m: None, leases=run.leases) == 0
    run.ops.down("isl")
    sb = dataclasses.replace(islands_spec(), nodes=(b300(4), b300(5), b300(6)))
    other, _ = make_spares(tmp_path, sb, [spare(3)])
    pick = isp.pick_spare(other, "isb", list(sb.nodes), {1: "lost"}, ops=run.cluster,
                          health={"b300-3": (True, "healthy")})
    assert pick.node is not None and pick.node.name == "b300-3"


def test_a_lost_rollback_spare_keeps_a_ledger_row_until_its_stop_is_confirmed(tmp_path):
    """REV-920 P11: the spare's stop of incarnation 1 is in a row of its own; the
    incarnation's row closed (its other nodes confirmed); down() leaves the spare's row open;
    release-node, its stop confirmed, closes it."""
    run = _rollback_unstopped(tmp_path)
    assert "run1/isl-i1-owed-b300-3" in open_rows(tmp_path) and "run1/isl-i1" not in open_rows(tmp_path)
    run.ops.down("isl")
    assert [r for r in open_rows(tmp_path) if "b300-3" in r] == ["run1/isl-i1-owed-b300-3"]
    run.cluster.kill_rc.clear()
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3), spare(4)])
    assert isp.release_node_main(inventory=inv, name="b300-3", store=str(tmp_path / "store"), ssh=run.cluster,
                                 objects=run.store, kill_grace_s=1, say=lambda m: None, leases=run.leases) == 0
    assert not [r for r in open_rows(tmp_path) if "b300-3" in r]


def test_a_failed_start_keeps_its_incarnation_number_and_an_old_node_stops_it_before_the_next(tmp_path):
    """REV-920 P2 (M5): the stop of incarnation 1 is not confirmed on b300-0 (it stays an old
    node of the island). The next try starts incarnation 2, after stopping 1 on b300-0."""
    run = _rollback_unstopped(tmp_path, unstopped="b300-0")
    assert run.spares.records.own_quarantine("b300-3") is None  # its stop went through: back in the pool
    run.cluster.kill_rc.clear()
    run.clock.t += 61
    assert run.ops.relaunch(None, "isl") == 2
    calls = run.cluster.calls
    assert calls.index(("kill", "b300-0", 1), calls.index(("kill", "b300-0", 1)) + 1) < calls.index(("start", "b300-0", 2))
    assert "run1/isl-i1" not in open_rows(tmp_path)  # stopped on every node now: closed


class _SettledUnlisted(DirectoryObjectStore):
    def list(self, prefix):
        if "/settled/" in prefix:
            raise OSError("list refused")
        return super().list(prefix)


def test_a_settled_listing_failure_leaves_every_stop_owed(tmp_path):
    """REV-920 P5 (M3): the markers cannot be listed: the record comes back unfiltered, each
    stop counts as owed (each is idempotent), the node stays quarantined at a start, and
    release-node asks to be run again later (not --force)."""
    store = _SettledUnlisted(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    store.write(NodeRecords.settled_key("b300-1", "run1", "isl", 0), json.dumps(
        {"run_id": "run1", "island": "isl", "incarnation": 0}).encode())  # it was settled
    doc = records.own_quarantine("b300-1")
    assert doc["read_error"] and not doc.get("unreadable") and len(doc["owed"]) == 1
    records.quarantine("b300-1", island="isl", reason="lost again", owed=[isp.owed_stop("run1", "isl", 1, 1, False)])
    raw = json.loads(DirectoryObjectStore(tmp_path / "store").read("fleet/nodes/b300-1/quarantine.json"))
    assert [o["incarnation"] for o in raw["owed"]] == [0, 1]  # the second quarantine kept the first stop
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3)])
    said: list[str] = []
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"),
                                 ssh=Cluster([*NAMES, "b300-3"]), objects=store, kill_grace_s=1, say=said.append) == 1
    assert any("again later" in line for line in said) and not any("--force" in line for line in said)


def test_a_start_takes_a_node_whose_markers_cannot_be_listed_as_quarantined(tmp_path):
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    store = _SettledUnlisted(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    spares = IslandSpares(pool=(*s.nodes, spare(3)), records=records, resume_probe=lambda: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    ops.add("isl", s, spares=spares)
    assert 1 in ops._quarantined_ranks(ops.islands["isl"])


class _QuarantineUnread(DirectoryObjectStore):
    fail = False

    def read(self, key):
        if self.fail and key.endswith("quarantine.json"):
            raise OSError("read refused")
        return super().read(key)


def test_a_record_that_cannot_be_read_owes_its_stops_and_is_not_written_over(tmp_path):
    """REV-920 P8 (M3): the store fails to read the record: an owed stop is run (not skipped),
    a new quarantine refuses to write over it, release-node asks to be run again later."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[])
    assert ops.relaunch(None, "isl") is None
    bad = _QuarantineUnread(tmp_path / "store")
    spares.records.objects = bad
    bad.fail = True
    rec, node = ops.islands["isl"], spares.pool[1]
    cluster.unreachable.discard("b300-1")
    assert ops._still_owed(rec, node, 0) and ops._owes(rec, "b300-1", 0)
    ops._run_owed_stop(rec, node, 0, None, what="quarantined node")
    assert ("kill", "b300-1", 0) in cluster.calls
    with pytest.raises(isp.SpareError, match="not written over"):
        spares.records.quarantine("b300-1", island="isl", reason="again")
    bad.fail = False
    assert json.loads(bad.read("fleet/nodes/b300-1/quarantine.json"))["owed"]  # still there
    bad.fail = True
    inv = _inventory(tmp_path, [*islands_spec().nodes])
    said: list[str] = []
    assert isp.release_node_main(inventory=inv, name="b300-1", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=bad, kill_grace_s=1, say=said.append) == 1
    assert any("again later" in line for line in said) and not any("--force" in line for line in said)


def test_a_loss_whose_quarantine_cannot_be_written_keeps_the_claim(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    rec = ops.islands["isl"]

    def refuse(*a, **kw):
        raise isp.SpareError("store unavailable")

    spares.records.quarantine = refuse
    spares.lost.pop(2, None)
    ops._mark_lost(rec, 2, "XID 79")
    assert spares.records.holder("b300-2") == "isl"  # the run's other islands still cannot take it


def test_a_failed_step_after_the_pick_gives_the_spare_back(tmp_path, monkeypatch):
    """REV-920 P9 (L2): the new health watches fail after the pick: the spare's claim and
    lease go back and the island is on its old nodes."""
    leases = FakeLeases()
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)], leases=leases)

    def broken(spec):
        raise RuntimeError("health watch setup failed")

    monkeypatch.setattr(ops, "_health_watches", broken)
    assert ops.relaunch(None, "isl") is None
    assert store.read("fleet/nodes/b300-3/claim.json") is None and "b300-3" not in leases.holders
    assert [n.name for n in ops.islands["isl"].spec.nodes] == NAMES


def test_an_owed_stop_elsewhere_ends_before_the_island_starts_on_the_node(tmp_path):
    """REV-920 P10 (M4): b300-1, released elsewhere, is a candidate again: the stop of its
    old incarnation begins and ends before incarnation 2 starts on it, outside the ops lock."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    store.delete("fleet/nodes/b300-1/quarantine.json")  # released elsewhere: no record here, its row open
    real, locked = cluster.run, []

    def run(node, script, *, timeout=300.0):
        if node.name == "b300-1" and "kill -TERM" in script and cluster._k(script) == 0:
            cluster.calls.append(("begin", "b300-1", 0))
            locked.append(ops._lock._is_owned())
            done = real(node, script, timeout=timeout)
            cluster.calls.append(("end", "b300-1", 0))
            return done
        return real(node, script, timeout=timeout)

    cluster.run = run
    rec = ops.islands["isl"]
    cluster.lose("b300-3")
    assert wait_for(lambda: rec.job.procs[1].poll() is not None)
    for _ in range(2):
        ops.job_status("isl", 1)
    assert ops.relaunch(None, "isl") == 2
    c = cluster.calls
    assert c.index(("begin", "b300-1", 0)) < c.index(("end", "b300-1", 0)) < c.index(("start", "b300-1", 2))
    assert locked == [False] and "run1/isl-i0-owed-b300-1" not in open_rows(tmp_path)


def test_two_checks_of_one_candidate_stop_its_old_incarnation_once(tmp_path):
    import time as _time

    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    store.delete("fleet/nodes/b300-1/quarantine.json")
    real = cluster.run

    def run(node, script, *, timeout=300.0):
        if "kill -TERM" in script:
            _time.sleep(0.2)
        return real(node, script, timeout=timeout)

    cluster.run = run
    node = spares.pool[1]
    results = []
    threads = [threading.Thread(target=lambda: results.append(isp.close_owed_elsewhere(spares, node, cluster)))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 2 and all(ok for ok, _why in results)
    assert cluster.calls.count(("kill", "b300-1", 0)) == 1  # the second check found its row closed


def test_a_candidate_whose_owed_stop_fails_is_passed_over_with_why(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    store.delete("fleet/nodes/b300-1/quarantine.json")
    cluster.kill_rc["b300-1"] = 1
    checked = isp.check_candidates(spares, "isl", list(ops.islands["isl"].spec.nodes), {1: "lost"}, ops=cluster)
    assert checked["b300-1"][0] is False and "owed stop" in checked["b300-1"][1]
    assert "run1/isl-i0-owed-b300-1" in open_rows(tmp_path)


def test_release_removes_the_record_before_the_markers(tmp_path):
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    order, real = [], records._delete

    def delete(key):
        order.append(key.rsplit("/", 2)[-2] if "/settled/" in key else key.rsplit("/", 1)[-1])
        real(key)

    records._delete = delete
    records.release("b300-1")
    assert order == ["quarantine.json", "settled"]


def test_a_settle_whose_second_read_fails_removes_its_marker(tmp_path):
    store = DirectoryObjectStore(tmp_path / "store")
    records = NodeRecords(store, run_id="run1", log=lambda m: None)
    records.quarantine("b300-1", island="isl", reason="lost", owed=[isp.owed_stop("run1", "isl", 0, 1, False)])
    real, reads = records._read, []

    def read(key):
        if key.endswith("quarantine.json"):
            reads.append(key)
            if len(reads) == 1:
                raise OSError("read refused")
        return real(key)

    records._read = read
    with pytest.raises(OSError):
        records.settle("b300-1", run_id="run1", island="isl", incarnation=0)
    assert not store.list("fleet/nodes/b300-1/settled/")


@pytest.mark.parametrize("island", ['is"l', "is$(touch PWNED)l", "is`touch PWNED`l", "is'l"])
def test_the_owed_stop_trap_expands_nothing_in_a_name(tmp_path, island):
    import os
    import subprocess

    from yeto.launch.ssh_trainer_island import STATE_DIR

    script = isp.owed_kill_script("run1", island, 0, 5, None)
    head = "\n".join(script.splitlines()[:2]) + "\nexit 0\n"  # the trap alone: it runs at the exit
    assert subprocess.run(["bash", "-n", "-c", head], timeout=30).returncode == 0
    home = tmp_path / "home"
    sdir = home / STATE_DIR / "run1" / f"{island}-i0"
    sdir.mkdir(parents=True)
    (sdir / "pgid").write_text("1\n")
    done = subprocess.run(["bash", "-c", head], env={**os.environ, "HOME": str(home)}, cwd=tmp_path, timeout=30,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert not (tmp_path / "PWNED").exists()
    assert not (sdir / "pgid").exists() and (sdir / "stopped").exists()


def test_a_stale_pick_not_held_by_this_launcher_is_not_released(tmp_path):
    """REV-920 L5: the stale pick b300-3's lease is not held by this object (another launcher
    of the run holds it): nothing is released, nothing said."""
    import dataclasses

    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore(LEASE_STORE, client=FakeS3()), Clock()
    leases = NodeLeases(LEASE_STORE, "run1/trainer", identity="aaaa", objects=objects, clock=clock, monotonic=clock)
    other = NodeLeases(LEASE_STORE, "run1/trainer", identity="bbbb", objects=objects, clock=clock, monotonic=clock)
    for n in NAMES:
        assert leases.acquire(n).ok
    assert other.acquire("b300-3").ok
    leases.picks["isl"] = [*NAMES, "b300-3"]
    released, lines = [], []
    real = leases.release
    leases.release = lambda node: (released.append(getattr(node, "name", node)), real(node))
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, clock=clock, health_every_s=0)
    ops.log = lines.append
    s = dataclasses.replace(islands_spec(), leases=leases)
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops._release_stale_leases(ops.islands["isl"])
    assert released == [] and not [ln for ln in lines if "given back" in ln]
    assert other.held() == ["b300-3"]


def test_the_trainer_pick_leaves_out_a_node_quarantined_in_the_run_store(tmp_path, monkeypatch):
    """REV-920 L4: a node the run's own island quarantined in its run store (the lease store
    copy missing, e.g. its write failed) is no candidate of the run's trainer pick."""
    from tests.test_rl_object_store import FakeS3
    from yeto.cloud.s3_store import S3ObjectStore
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.node_lease import NodeLeases

    objects, clock = S3ObjectStore("s3://leases/x", client=FakeS3()), Clock()
    NodeRecords(DirectoryObjectStore(tmp_path / "store"), run_id="run2", log=lambda m: None).quarantine(
        "b300-0", island="isl", reason="lost")
    monkeypatch.setattr(sti, "_lease_record", lambda args: {})
    monkeypatch.setattr(sti, "_record_island_nodes", lambda args, name, nodes: None)
    leases = NodeLeases("s3://leases/x", "run2/trainer", objects=objects, clock=clock, monotonic=clock)
    args = SimpleNamespace(cluster_prefix=f"oc-f-{tmp_path.name}", rl_checkpoint_store=str(tmp_path / "store"))
    picked = sti.lease_trainer_nodes(args, [b300(i) for i in range(3)], 2, leases, name="isl", wait_s=0)
    assert [n.name for n in picked] == ["b300-1", "b300-2"]


# ------------------------------------------------------------------ REV-920b
def test_down_stops_an_earlier_incarnation_an_old_node_did_not_confirm_or_keeps_the_island(tmp_path):
    """REV-920b M-a: after a rollback whose stop of incarnation 1 b300-0 did not confirm,
    down() stops 1 there first; while it is not confirmed, down() is False and the island keeps
    its leases, claims and the row of 1; once confirmed, all of it goes."""
    run = _rollback_unstopped(tmp_path, unstopped="b300-0")
    assert 1 in run.ops.islands["isl"].islands.unreleased
    before = run.cluster.calls.count(("kill", "b300-0", 1))
    assert run.ops.down("isl") is False
    assert run.cluster.calls.count(("kill", "b300-0", 1)) == before + 1
    rec = run.ops.islands["isl"]
    assert run.spares.records.holder("b300-0") == "isl" and rec.lock is not None
    assert "run1/isl-i1" in open_rows(tmp_path)
    run.cluster.kill_rc.clear()
    assert run.ops.down("isl") is True
    assert run.spares.records.holder("b300-0") is None and rec.lock is None
    assert "run1/isl-i1" not in open_rows(tmp_path)


def test_a_rollback_with_an_unconfirmed_old_node_keeps_the_incarnation_row_open(tmp_path):
    """REV-920b L-c (m01): b300-0 and the spare both leave incarnation 1 unconfirmed: the
    spare owes its own row, and the row of 1 stays open for b300-0."""
    leases = FakeLeases()
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)], leases=leases)
    real = cluster.start

    def start(node, script):
        if node.name == "b300-2" and "setsid bash -c" in script and cluster._k(script) == 1:
            raise RuntimeError("ssh start refused on rank 2")
        return real(node, script)

    cluster.start = start
    cluster.kill_rc.update({"b300-0": 1, "b300-3": 1})
    assert ops.relaunch(None, "isl") is None
    rows = open_rows(tmp_path)
    assert "run1/isl-i1" in rows and "run1/isl-i1-owed-b300-3" in rows
    assert 1 in ops.islands["isl"].islands.unreleased


def _refusing(records, *names):
    real = records.quarantine
    state = {"down": True}

    def quarantine(name, **kw):
        if state["down"] and name in names:
            raise isp.SpareError("store write refused (503)")
        return real(name, **kw)

    records.quarantine = quarantine
    return state


def test_a_loss_whose_quarantine_the_store_refuses_owes_a_row_and_is_written_once_the_store_is_back(tmp_path):
    """REV-920b M-b: _mark_lost with a store that refuses the write: the node owes its stop in a
    ledger row, this island does not pick it again, its claim keeps the run's other islands off
    it; once the store answers, a check writes the record and gives the claim up."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    store_state = None
    rec = ops.islands["isl"]
    lines: list[str] = []
    ops.log = lines.append
    store_state = _refusing(spares.records, "b300-2")
    spares.lost.pop(2, None)
    ops._mark_lost(rec, 2, "XID 79")
    assert "run1/isl-i0-owed-b300-2" in open_rows(tmp_path)
    assert spares.records.holder("b300-2") == "isl" and "b300-2" in spares.pending
    assert ops._owes(rec, "b300-2", 0) and ops._still_owed(rec, spares.pool[2], 0)
    assert any("CRITICAL" in ln and "holds its claim and owes its stops" in ln for ln in lines)
    _c, _g, skipped, _m, _k = isp.spare_candidates(spares, "isl", list(rec.spec.nodes), [1, 2])
    assert "not written to the store yet" in skipped["b300-2"]  # this island leaves it out too
    store_state["down"] = False
    ops._replace_checked(rec)  # a check: the record is written now
    assert spares.records.own_quarantine("b300-2") is not None and "b300-2" not in spares.pending
    assert spares.records.holder("b300-2") is None


def test_a_rollback_spare_whose_quarantine_was_refused_is_pickable_once_written_and_released(tmp_path):
    """REV-920b M-b / L-c (m04): the rollback spare's quarantine is refused: its claim stays
    with isl; the store comes back, down() writes the record and gives the claim up; once its
    stop goes through and release-node runs, isb takes it."""
    import dataclasses

    leases = FakeLeases()
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3), spare(4)], leases=leases)
    state = _refusing(spares.records, "b300-3")
    real = cluster.start

    def start(node, script):
        if node.name == "b300-2" and "setsid bash -c" in script and cluster._k(script) == 1:
            raise RuntimeError("ssh start refused on rank 2")
        return real(node, script)

    cluster.start, cluster.kill_rc["b300-3"] = start, 1
    assert ops.relaunch(None, "isl") is None
    cluster.start = real
    assert spares.records.own_quarantine("b300-3") is None and spares.records.holder("b300-3") == "isl"
    assert "run1/isl-i1-owed-b300-3" in open_rows(tmp_path)
    sb = dataclasses.replace(islands_spec(), nodes=(b300(5), b300(6), b300(7)))
    other, _ = make_spares(tmp_path, sb, [spare(3)])
    pick = isp.pick_spare(other, "isb", list(sb.nodes), {1: "lost"}, ops=cluster, health={"b300-3": (True, "healthy")})
    assert pick.node is None and "held by island isl" in pick.skipped["b300-3"]
    state["down"] = False
    assert ops.down("isl")
    assert spares.records.own_quarantine("b300-3") is not None and spares.records.holder("b300-3") is None
    cluster.kill_rc.clear()
    inv = _inventory(tmp_path, [*islands_spec().nodes, spare(3), spare(4)])
    assert isp.release_node_main(inventory=inv, name="b300-3", store=str(tmp_path / "store"), ssh=cluster,
                                 objects=store, kill_grace_s=1, say=lambda m: None, leases=leases) == 0
    pick = isp.pick_spare(other, "isb", list(sb.nodes), {1: "lost"}, ops=cluster, health={"b300-3": (True, "healthy")})
    assert pick.node is not None and pick.node.name == "b300-3"
    assert not [r for r in open_rows(tmp_path) if "b300-3" in r]


def test_a_confirmed_stop_of_a_node_whose_quarantine_was_refused_owes_nothing_more(tmp_path):
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    rec = ops.islands["isl"]
    _refusing(spares.records, "b300-2")
    spares.lost.pop(2, None)
    ops._mark_lost(rec, 2, "XID 79", answers=True)  # still answers: stopped in the background
    join_stops(spares)
    assert ("kill", "b300-2", 0) in cluster.calls
    assert spares.pending["b300-2"]["owed"] == [] and not ops._owes(rec, "b300-2", 0)
    assert "run1/isl-i0-owed-b300-2" not in open_rows(tmp_path)


def test_a_candidate_whose_ledger_cannot_be_read_is_passed_over(tmp_path, monkeypatch):
    """REV-920b L-a (m09): the run's ledger cannot be read: the candidate's owed rows are not
    known, so it is not a candidate this round."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    store.delete("fleet/nodes/b300-1/quarantine.json")  # released elsewhere, its owed row open

    def broken(self):
        raise OSError("ledger unreadable")

    monkeypatch.setattr(ResourceLedger, "open_resources", broken)
    ok, why = isp.close_owed_elsewhere(spares, spares.pool[1], cluster)
    assert not ok and "the ledger could not be read" in why
    assert not [c for c in cluster.calls if c[0] == "kill" and c[1] == "b300-1" and c[2] == 0]


def test_a_rollback_whose_only_unconfirmed_stop_is_a_lost_spare_owes_its_incarnation_nowhere_else(tmp_path):
    """REV-920b (m02): only the lost spare b300-3 did not confirm the stop of incarnation 1: its
    own row and record owe that stop, so the island does not carry 1 as unreleased and down()
    stops it on no old node again."""
    run = _rollback_unstopped(tmp_path)
    rec = run.ops.islands["isl"]
    assert 1 not in rec.islands.unreleased
    before = [c for c in run.cluster.calls if c[0] == "kill" and c[2] == 1]
    assert run.ops.down("isl")
    after = [c for c in run.cluster.calls if c[0] == "kill" and c[2] == 1]
    assert after == before


def test_an_unhealthy_candidate_does_not_run_its_owed_stops(tmp_path):
    """REV-920b (m08): an unhealthy candidate is no candidate: its owed stop is not run in
    the checks and its row stays open."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    store.delete("fleet/nodes/b300-1/quarantine.json")  # released elsewhere, its owed row open
    cluster.sick.add("b300-1")
    checked = isp.check_candidates(spares, "isl", list(ops.islands["isl"].spec.nodes), {1: "lost"}, ops=cluster)
    assert checked["b300-1"][0] is False and checked["b300-1"][1].startswith("unhealthy")
    assert ("kill", "b300-1", 0) not in cluster.calls
    assert "run1/isl-i0-owed-b300-1" in open_rows(tmp_path)


def test_a_candidate_whose_owed_stop_raises_is_passed_over(tmp_path, monkeypatch):
    """REV-920b (m09): the owed stop raises (not a ledger read error): not known whether the
    old incarnation still runs there, so the candidate is passed over and its row stays open."""
    from yeto.launch import ssh_trainer_ledger

    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    assert ops.relaunch(None, "isl") == 1
    cluster.unreachable.discard("b300-1")
    store.delete("fleet/nodes/b300-1/quarantine.json")

    def boom(row, ops=None, dry_run=False, ssh=None):
        raise RuntimeError("ssh agent gone")

    monkeypatch.setattr(ssh_trainer_ledger, "cleanup_release", boom)
    ok, why = isp.close_owed_elsewhere(spares, spares.pool[1], cluster)
    assert not ok and why.startswith("its owed stops not run (RuntimeError: ssh agent gone")
    assert "run1/isl-i0-owed-b300-1" in open_rows(tmp_path)


def test_two_refused_quarantines_of_one_node_owe_both_stops(tmp_path):
    """REV-920b M-b: a second quarantine of a node whose first one the store refused adds its
    stop to the one kept in memory; once written, the record lists both."""
    cluster, ops, spares, store, clock = lost_island(tmp_path, extra=[spare(3)])
    rec = ops.islands["isl"]
    state = _refusing(spares.records, "b300-2")
    node = spares.pool[2]
    for inc in (0, 1):
        assert not ops._quarantine(rec, node, reason=f"lost at {inc}",
                                   owed=[isp.owed_stop("run1", "isl", inc, 2, False)])
    assert ops._owes(rec, "b300-2", 0) and ops._owes(rec, "b300-2", 1)
    state["down"] = False
    ops._flush_pending(rec)
    doc = spares.records.own_quarantine("b300-2")
    assert sorted(int(o["incarnation"]) for o in doc["owed"]) == [0, 1]


def test_the_compiled_plan_keeps_the_controller_poll(fleet_run, files):  # noqa: F811
    """The dashboard's unreported-wait threshold reads --controller-poll from the plan."""
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] = [*cfg["fleet"]["trainer_argv"], "--controller-poll", "1500"]
    plan = fleet_run(cfg).fleet_plan()
    assert plan.controller_poll_s == 1500.0 and plan.to_json()["roster"]["controller_poll_s"] == 1500.0
    assert "controller_poll_s" not in fleet_run(_cfg(files)).fleet_plan().to_json()["roster"]


# ------------------------------------------------------------------ M-C: a takeover reattaches first
def _taken_over_with_a_quarantined_node(tmp_path, **kw):
    """A running 3-node island with a spare; its launcher (host A) is lost, then another run
    quarantines rank 2's node; a launcher on another host starts it again from the journal."""
    shared = DirectoryObjectStore(tmp_path / "leases")
    cluster = Cluster([*NAMES, "b300-3"])

    def spares_for():
        return IslandSpares(pool=(*islands_spec().nodes, spare(3)),
                            records=NodeRecords(DirectoryObjectStore(tmp_path / "store"), run_id="run1",
                                                log=lambda m: None, shared=shared))

    first = make_ops(tmp_path, cluster, health_every_s=0)
    first.add("isl", islands_spec(), spares=spares_for())
    assert first.start("isl") == 0
    _crash(first)  # host A is lost; the nodes run on
    NodeRecords(DirectoryObjectStore(tmp_path / "store2"), run_id="run2", log=lambda m: None,
                shared=shared).quarantine("b300-2", island="x", reason="XID 79 in run2")
    again = make_ops(tmp_path, cluster, health_every_s=0, **kw)
    again.add("isl", islands_spec(), spares=spares_for())
    return cluster, again


@pytest.mark.parametrize("kind", ["taken_over", "reattach_only"])
def test_a_takeover_with_a_quarantined_node_reattaches_and_does_not_restart(tmp_path, kind):
    """The takeover rule wins over the spares rule at the start: the running incarnation is
    reattached (no release, no new incarnation); a spare replaces the node later, in the
    control loop."""
    cluster, again = _taken_over_with_a_quarantined_node(tmp_path, **{kind: {"isl"}})
    assert again.start("isl") == 0
    assert "isl" in again.reattached and again.islands["isl"].incarnation == 0
    assert cluster.count("kill") == 0 and started(cluster, 1) == []  # nothing released, nothing started
    assert started(cluster, 0) == NAMES  # only the first start


def test_without_a_takeover_the_quarantined_node_is_replaced_at_the_start(tmp_path):
    """The same start not taken over (a launcher restarted on its own host): main's spares rule."""
    cluster, again = _taken_over_with_a_quarantined_node(tmp_path)
    assert again.start("isl") == 1
    assert started(cluster, 1) == ["b300-0", "b300-1", "b300-3"]
