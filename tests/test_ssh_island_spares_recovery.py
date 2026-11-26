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

from test_rl_ssh_resident import Clock, _cards, b300, make_ops, spec, wait_for  # noqa: F401 - _cards: autouse
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
    assert json.loads(store.read("fleet/nodes/b300-2/quarantine.json"))["owed"] == []  # stopped: owes nothing
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
    assert "not replaced by a spare until that check is in" in view["note"]
    assert "not replaced by a spare until that check is in" in capsys.readouterr().err
    lines = []
    isp.spares_from_args(args, s, log=lines.append)
    assert [ln for ln in lines if "NOTE" in ln and "device family cuda has no node health check" in ln]
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
    assert pick.node is None and pick.skipped["b300-1"].startswith("quarantined (")
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
def test_a_health_reading_of_another_node_is_ignored(tmp_path):
    cluster = Cluster([*NAMES, "b300-3"])
    ops = make_ops(tmp_path, cluster, health_every_s=0)
    s = islands_spec()
    spares, _ = make_spares(tmp_path, s, [spare(3)])
    ops.add("isl", s, spares=spares)
    ops.start("isl")
    stale = SimpleNamespace(**vars(READING), node="b300-9")
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], stale)
    assert spares.lost == {} and str(ops.job_status("isl", 0)) == "RUNNING"
    ops._node_health_lost(ops.islands["isl"], 2, ["XID 79 on GPU 3"], SimpleNamespace(**vars(READING), node="b300-2"))
    assert list(spares.lost) == [2] and str(ops.job_status("isl", 0)) == "FAILED"


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
