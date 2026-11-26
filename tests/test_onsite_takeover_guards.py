"""What a takeover by an on-site control plane must never do (own-cluster-fleet-stability 5.2,
second review): start an island that ended on the lost host again, clear the STOP of a moving or
stopping one, order a Modal island twice, step around the relaunch budget, or relaunch an SSH
island on other nodes than the ones it runs on. Local store, fake launches, fake SSH nodes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from test_fleet_trainer_worker import FakeLaunches, fleet, member
from test_onsite_takeover import LATER, T0, _seed, moving_on_a, started_on_a, worker_on

from yeto.launch import island_lifecycle as lifecycle
from yeto.rl import trainer_worker as tw


def ended_there(objects, k: int, state: str, *, record: dict | None = None, name: str | None = None,
                since: float = T0 + 300, reason: str = "finished"):
    """Island ``k``'s own record, as its island launch on host A left it."""
    if name is None:
        name = tw.island_name(record or tw.TrainerWorker(None, objects, None, None).load(k))
    states = lifecycle.IslandStateStore(objects, log=lambda m: None, clock=lambda: since)
    states.write(k, lifecycle.ORDERING, name=name, source="ssh", role="trainer", reason="ordering")
    if state != lifecycle.ORDERING:
        states.write(k, lifecycle.RUNNING, name=name, source="ssh", role="trainer", reason="up")
        states.write(k, state, name=name, source="ssh", role="trainer", reason=reason)


# -- H2: a launch that ended on the lost host is closed there, never started again ------------------

@pytest.mark.parametrize("state", [lifecycle.DONE, lifecycle.FAILED, lifecycle.ABANDONED])
def test_an_island_that_ended_on_the_lost_host_is_closed_and_never_started(tmp_path, state):
    objects, store, _a, _locks = started_on_a(tmp_path)
    ended_there(objects, 0, state)
    given_up: list = []
    b, launches = worker_on(objects, store, "host-b", LATER, on_given_up=lambda k, m, why: given_up.append(k))
    b.step()
    assert launches.started("run-i0") == 0 and launches.started("run-i1") == 1  # island 1 runs: taken over
    record = b.load(0)
    assert record["state"] == state and record["host"] == "host-b" and record["incarnation"] == 0
    assert record["taken_over"]["ended"]["state"] == state and "nothing started" not in record["ended"]["detail"]
    assert given_up == ([0] if state in (lifecycle.FAILED, lifecycle.ABANDONED) else [])
    # billed up to its end on host A (its island record's time), not up to the takeover
    assert record["billed_s"] == pytest.approx(300.0) and record["closed_at"] is not None
    for _ in range(3):
        b.step()
    assert launches.started("run-i0") == 0 and b.load(0)["state"] == state  # and never later


def test_an_ended_island_with_its_stop_gives_no_pools_up(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    ended_there(objects, 0, lifecycle.ABANDONED)
    lifecycle.IslandStateStore(objects).request_stop(0, "operator")
    given_up: list = []
    b, launches = worker_on(objects, store, "host-b", LATER, on_given_up=lambda k, m, why: given_up.append(k))
    b.step()
    assert b.load(0)["state"] == lifecycle.ABANDONED and given_up == [] and launches.started("run-i0") == 0


def test_an_earlier_incarnation_s_ended_record_is_not_this_launch_s_end(tmp_path):
    """A moved island whose new launch had not written its own island record yet: the record
    there is the old incarnation's (another cluster name). The launch is taken over."""
    objects, store, _a, _locks = started_on_a(tmp_path)
    ended_there(objects, 0, lifecycle.DONE, name="an-earlier-incarnation-l0-ssh")
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    assert launches.started("run-i0") == 1 and b.load(0)["state"] == tw.STARTING
    assert b.load(0)["taken_over"]["ended"] is None


@pytest.mark.parametrize("remove", [False, True])
def test_a_moving_or_stopping_island_that_ended_goes_on_with_its_move_or_removal(tmp_path, remove):
    objects, store = moving_on_a(tmp_path, remove=remove)
    ended_there(objects, 0, lifecycle.DONE, reason="island STOP")
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    assert launches.started("run-i0") == 0 and not b.load(0).get("reattach")  # nothing reattached
    assert lifecycle.IslandStateStore(objects).stop_requested(0) is not None  # its STOP kept so far
    b.step()
    record = b.load(0)
    if remove:
        assert record["state"] == tw.REMOVED and launches.started("run-i0-m1") == 0
    else:  # the move: its next incarnation on the new cards, as host A would have started it
        assert getattr(launches, "torn_down", []) == ["run-i0"]  # its old prefix torn down here first
        b.step()
        assert b.load(0)["incarnation"] == 1 and launches.started("run-i0-m1") == 1
        assert tw.TAKEOVER_FLAG not in launches.starts[-1][1]  # a launch of its own, not a takeover


# -- H2: a takeover launch is flagged, counted, and never clears the STOP ----------------------------

def test_a_takeover_launch_keeps_the_stop_and_counts_against_the_budget(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    states.request_stop(0, "operator")
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    prefix, argv = launches.starts[0]
    assert prefix == "run-i0" and argv[-1] == tw.TAKEOVER_FLAG and argv.count(tw.TAKEOVER_FLAG) == 1
    assert states.stop_requested(0) is not None
    assert b.load(0)["relaunches"] == 1 and b.load(0)["takeover_launch"] is True


def test_a_takeover_with_the_relaunch_budget_spent_gives_the_island_up(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    w = tw.TrainerWorker(store, objects, None, lifecycle.IslandStateStore(objects))
    w.save(0, {**w.load(0), "relaunches": tw.DEFAULT_RELAUNCH_BUDGET})
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", LATER, logs)
    b.step()
    assert launches.started("run-i0") == 0 and launches.started("run-i1") == 1
    record = b.load(0)
    assert record["state"] == tw.ABANDONED and record["billing_open"]
    assert lifecycle.IslandStateStore(objects).stop_requested(0) is not None
    assert any("relaunch budget is spent" in m for m in logs)


def test_a_reattach_starts_its_stop_wait_again_here(tmp_path):
    objects, store = moving_on_a(tmp_path)
    w = tw.TrainerWorker(store, objects, None, lifecycle.IslandStateStore(objects))
    w.save(0, {**w.load(0), "stop_at": T0 + 10, "torn_down_at": T0 + 20})  # host A's wait, and its teardown
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    record = b.load(0)
    assert record["stop_at"] == LATER and record["torn_down_at"] is None  # host A's wait is not this launch's
    assert record["relaunches"] == 1 and record["takeover_launch"] is True
    (argv,) = [a for p, a in launches.starts if p == "run-i0"]
    assert argv[-1] == tw.TAKEOVER_FLAG and tw.TAKEOVER_FLAG not in record["argv"]
    b.step()  # its launch runs: no teardown for host A's old STOP time
    assert getattr(launches, "torn_down", []) == []


def test_a_takeover_launch_that_is_refused_blocks_the_island(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    seen: list[dict] = []
    b, launches = worker_on(objects, store, "host-b", LATER, on_event=seen.append)
    b.step()
    launches.end("run-i0", 13, f"[launcher] ERROR: {tw.TAKEOVER_BLOCKED_MARKER} island x: its record says it ended")
    b.step()
    b.step()
    record = b.load(0)
    assert record["takeover_blocked"] and launches.started("run-i0") == 1  # no hand-back restart
    assert [e["event"] for e in seen] == ["island_takeover_blocked"]


def test_a_takeover_launch_that_hands_off_is_started_again_as_a_takeover(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    launches.end("run-i0", 12, "another launcher holds the island records")
    b.step()
    assert launches.started("run-i0") == 2 and launches.starts[-1][1][-1] == tw.TAKEOVER_FLAG


def test_the_lost_launch_s_time_is_billed_at_the_takeover(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    b, _launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    record = b.load(0)
    assert record["billed_s"] == pytest.approx(LATER - T0)  # host A's time is not lost
    assert record["started_at"] == LATER and record["closed_at"] is None  # host B's launch bills on


def test_a_reattach_gone_without_a_run_record_blocks_the_island(tmp_path):
    """No marker, no run record (the launch never registered): blocked too (Mi)."""
    objects, store = moving_on_a(tmp_path)
    clock = {"t": LATER}
    b, launches = worker_on(objects, store, "host-b", lambda: clock["t"])
    b.step()
    launches.live.pop("run-i0")  # gone, no exit recorded
    clock["t"] = LATER + tw.START_GRACE_S + 1
    b.step()
    record = b.load(0)
    assert record["takeover_blocked"] and record["state"] == tw.MOVING and record["incarnation"] == 0
    assert launches.started("run-i0-m1") == 0


def test_a_reattached_launch_gets_the_fleet_stop_when_the_store_does_not_list(tmp_path):
    """The fleet's STOP reaches a reattached launch through the launches this worker knows,
    even when the store's island records do not list (Mt)."""
    objects, store = moving_on_a(tmp_path)
    stop = {"why": None}
    b, launches = worker_on(objects, store, "host-b", LATER, fleet_stop=lambda: stop["why"])
    b.step()
    assert launches.started("run-i0") == 1
    real = objects.list
    objects.list = lambda prefix, **kw: (_ for _ in ()).throw(OSError("store down")) if prefix.startswith(
        lifecycle.STATE_DIR) else real(prefix, **kw)
    stop["why"] = "run STOP"
    b.step()
    assert "run-i0" in launches.stops


# -- the takeover launch itself (yeto launch --rl-fleet-takeover) ------------------------------------

def _plan(tmp_path, monkeypatch, *, takeover: bool, state: str, name: str = "isl"):
    from yeto import launcher
    from yeto.launch import island_adopt
    from yeto.rl.engine.object_store import DirectoryObjectStore

    monkeypatch.setattr(launcher, "_island_refs", lambda *a, **kw: {"isl": lifecycle.IslandRef(0, "ssh")})
    monkeypatch.setattr(launcher, "_record_id", lambda args, m: m)
    monkeypatch.setattr(launcher, "_adopt_probes", lambda args: None)
    monkeypatch.setattr(launcher, "_seed_ssh_journals", lambda *a, **kw: [])
    monkeypatch.setattr(island_adopt, "plan", lambda *a, **kw: {})
    objects = DirectoryObjectStore(tmp_path / "store")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    states.write(0, state, name=name, source="ssh", role="trainer", reason="it ended")
    states.request_stop(0, "moving")
    args = SimpleNamespace(rl_fleet_takeover=takeover)
    kw = dict(relaunch=False, ssh_names={"isl"}, events_dir=None, echo_names=(), held_name=None, syncer_cluster=None)
    call = lambda: launcher._plan_island_records(args, [SimpleNamespace(cloud="ssh")], ["isl"],  # noqa: E731
                                                 lifecycle.IslandStateStore(objects, log=lambda m: None), object(), **kw)
    return call, lifecycle.IslandStateStore(objects, log=lambda m: None)


def test_a_takeover_launch_of_an_island_that_ended_starts_nothing(tmp_path, monkeypatch):
    from yeto.launch.island_adopt import AdoptionUnknown

    call, states = _plan(tmp_path, monkeypatch, takeover=True, state=lifecycle.DONE)
    with pytest.raises(AdoptionUnknown, match=r"\[yeto-takeover-blocked\].*ended \(done"):
        call()
    assert states.stop_requested(0) is not None  # never cleared


def test_a_takeover_launch_never_clears_the_stop(tmp_path, monkeypatch):
    """Its island's record is not adoptable (another incarnation's, ended): a plain launch would
    take the run as new and clear the STOP; a takeover launch is a restart."""
    call, states = _plan(tmp_path, monkeypatch, takeover=True, state=lifecycle.DONE, name="older-name")
    call()
    assert states.stop_requested(0) is not None
    call, states = _plan(tmp_path / "plain", monkeypatch, takeover=False, state=lifecycle.DONE, name="older-name")
    call()
    assert states.stop_requested(0) is None  # as before without the flag: a new run's STOP is void


def test_the_takeover_flag_belongs_to_an_island_launch():
    from yeto.rl.engine import sample_roles

    with pytest.raises(ValueError, match="--rl-fleet-takeover belongs to an island launch"):
        sample_roles.check_fleet_member(SimpleNamespace(rl_fleet_takeover=True, rl_fleet_role=None))
    from yeto.cli import parse_args

    base = ["--model", "m", "--data", "d.jsonl", "--gpu", "ssh:1x8xb300"]
    assert parse_args([*base, "--rl-fleet-takeover"]).rl_fleet_takeover is True
    assert parse_args(base).rl_fleet_takeover is False


# -- M1: the store's incarnation wins over an older local journal ------------------------------------

def _journal(path):
    from yeto.launch.trainer_island import read_jobs

    return sorted(int(k) for k in read_jobs(path))


def test_a_journal_older_than_the_store_record_is_seeded_again(tmp_path, monkeypatch):
    from yeto.launch.trainer_island import save_job

    seeded, journal, _ = _seed(tmp_path, monkeypatch, incarnation=0, env=None)
    path = journal / "isl-jobs.json"
    save_job(path, 0, "ssh:isl-i0")  # this host ran incarnation 0
    seeded, journal, _ = _seed(tmp_path, monkeypatch, incarnation=1)  # another host started 1 since
    assert seeded == ["isl"] and _journal(path) == [0, 1]
    seeded, _journal_dir, _ = _seed(tmp_path, monkeypatch, incarnation=1)  # as new as the store: kept (Mc)
    assert seeded == [] and _journal(path) == [0, 1]
    save_job(path, 2, "ssh:isl-i2")
    seeded, _journal_dir, _ = _seed(tmp_path, monkeypatch, incarnation=1)  # newer here: kept
    assert seeded == [] and _journal(path) == [0, 1, 2]


def test_a_to_b_to_a_to_b_each_host_follows_the_newest_incarnation(tmp_path, monkeypatch):
    from yeto import launcher, runs
    from yeto.launch.trainer_island import read_jobs, save_job

    def on(host: str, incarnation: int) -> list[int]:
        monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / host / "runs")
        records = {0: {"state": "running", "handle": {"cluster": "isl", "incarnation": incarnation},
                       "incarnation": incarnation}}
        launcher._seed_ssh_journals(SimpleNamespace(cluster_prefix="run1"), records,
                                    {"isl": lifecycle.IslandRef(0, "ssh", "trainer", None)}, {"isl"},
                                    environ={launcher.JOURNAL_FROM_STORE_ENV: "1"},
                                    states=SimpleNamespace(stop_requested=lambda k: None))
        path = tmp_path / host / "runs" / "run1" / "ssh-islands" / "isl-jobs.json"
        newest = max(int(k) for k in read_jobs(path))
        save_job(path, newest, f"ssh:isl-i{newest}")
        return _journal(path)

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "a" / "runs")
    save_job(tmp_path / "a" / "runs" / "run1" / "ssh-islands" / "isl-jobs.json", 0, "ssh:isl-i0")  # A starts 0
    assert on("b", 0) == [0]  # B takes over: 0 (no journal on B)
    save_job(tmp_path / "b" / "runs" / "run1" / "ssh-islands" / "isl-jobs.json", 1, "ssh:isl-i1")  # B relaunches 1
    assert on("a", 1) == [0, 1]  # A takes it back: its own journal (0) is older than the store's 1
    save_job(tmp_path / "a" / "runs" / "run1" / "ssh-islands" / "isl-jobs.json", 2, "ssh:isl-i2")  # A relaunches 2
    assert on("b", 2) == [0, 1, 2]  # and B again: 1 here, 2 in the store


# -- Ms: a moving island whose incarnation finished may still be reattached -------------------------

def test_a_reattach_only_island_that_finished_is_accepted(tmp_path, monkeypatch):
    from test_onsite_takeover import _again, _ssh_fixture

    cluster, spec = _ssh_fixture(tmp_path, monkeypatch)
    for node in cluster.runs:
        cluster.exit(node, 0, 0)  # it read its STOP and every node ended 0
    _seeded, journal, args = _seed(tmp_path, monkeypatch, stop=True)
    again = _again(cluster, spec, journal, tmp_path, reattach_only=args._reattach_only)
    again.start("isl")  # not refused: the island ended cleanly, nothing to block on
    assert cluster.count("start") == 3  # and nothing started


# -- H3: the island's nodes come back with it, and only those ---------------------------------------

def _lease_args(tmp_path, monkeypatch, host: str, nodes):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / host / "runs")
    inv = type("Nodes", (list,), {"lease_store": str(tmp_path / "leases")})(nodes)
    return SimpleNamespace(rl_fleet_role="island", rl_fleet_learner_id=0, cluster_prefix="run1-i0",
                           rl_node_inventory_nodes=inv, dry_run=False)


def test_the_placement_goes_to_the_store_and_comes_back_on_the_new_host(tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.launch.ssh_trainer_island import _lease_record, _record_island_nodes
    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path / "store")
    env = {launcher.JOURNAL_FROM_STORE_ENV: "1"}
    trainers = [SimpleNamespace(name=f"b300-{i}") for i in range(4)]
    a = _lease_args(tmp_path, monkeypatch, "host-a", trainers)
    launcher._restore_lease_identity(a, environ=env, objects=objects)
    assert not getattr(a, "_lease_pinned", None)  # nothing in the store yet
    _record_island_nodes(a, "isl", [trainers[2], trainers[3]])  # host A's pick, rank order
    assert launcher._share_lease_placement(a, environ=env, objects=objects) == {"isl": ["b300-2", "b300-3"]}
    assert launcher._share_lease_placement(a, environ=env, objects=objects) == {}  # once
    b = _lease_args(tmp_path, monkeypatch, "host-b", trainers)
    _record_island_nodes(b, "isl", [trainers[0], trainers[1]])  # a stale pick of this host
    launcher._restore_lease_identity(b, environ=env, objects=objects)
    assert _lease_record(b)["islands"]["isl"] == ["b300-2", "b300-3"] and b._lease_pinned == {"isl"}
    _record_island_nodes(b, "isl", [trainers[0], trainers[1]])
    launcher._share_lease_placement(b, environ=env, objects=objects)  # never changes the store's
    doc = json.loads(objects.read(launcher.node_lease_identity_key(0)))
    assert doc["islands"] == {"isl": ["b300-2", "b300-3"]} and doc["identity"]


def test_a_placement_written_meanwhile_is_kept(tmp_path, monkeypatch):
    """The conditional write: another write of the record between the read and this one is read
    again and kept."""
    from yeto import launcher
    from yeto.launch.ssh_trainer_island import _record_island_nodes
    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path / "store")
    env = {launcher.JOURNAL_FROM_STORE_ENV: "1"}
    trainers = [SimpleNamespace(name=f"b300-{i}") for i in range(4)]
    a = _lease_args(tmp_path, monkeypatch, "host-a", trainers)
    launcher._restore_lease_identity(a, environ=env, objects=objects)
    _record_island_nodes(a, "isl-m1", [trainers[0]])
    key = launcher.node_lease_identity_key(0)
    real = objects.replace_if
    raced = []

    def replace_if(k, data, expected):
        if not raced:
            raced.append(1)
            doc = json.loads(objects.read(k))
            objects.write(k, json.dumps({**doc, "islands": {"isl": ["b300-3"]}}).encode())
        return real(k, data, expected)

    objects.replace_if = replace_if
    assert launcher._share_lease_placement(a, environ=env, objects=objects) == {"isl-m1": ["b300-0"]}
    assert json.loads(objects.read(key))["islands"] == {"isl": ["b300-3"], "isl-m1": ["b300-0"]}


def _leases(tmp_path, clock, identity="ident-a", run="run1-i0/trainer"):
    from yeto.launch.node_lease import NodeLeases
    from yeto.rl.engine.object_store import DirectoryObjectStore

    return NodeLeases(str(tmp_path / "leases"), run, identity=identity,
                      objects=DirectoryObjectStore(tmp_path / "lease-store"), clock=clock, monotonic=clock)


def test_after_the_leases_ran_out_the_new_host_takes_the_same_nodes(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti

    clock = {"t": 1000.0}
    now = lambda: clock["t"]  # noqa: E731
    trainers = [SimpleNamespace(name=f"b300-{i}") for i in range(4)]
    a = _leases(tmp_path, now)
    assert a.acquire("b300-2").ok and a.acquire("b300-3").ok  # host A's island runs on 2 and 3
    clock["t"] += 600 + 60  # host A is lost: its leases ran out (TTL 600 s) long ago
    args = SimpleNamespace(cluster_prefix="run1-i0", _lease_pinned={"isl"})
    monkeypatch.setattr(sti, "_lease_record", lambda args: {"islands": {"isl": ["b300-2", "b300-3"]}})
    monkeypatch.setattr(sti, "_journal_incarnation", lambda args, name: False)  # not seeded yet at the check
    monkeypatch.setattr(sti, "_record_island_nodes", lambda *a: None)
    b = _leases(tmp_path, now)  # the same identity (from the store) on host B
    picked = sti.lease_trainer_nodes(args, trainers, 2, b, name="isl", wait_s=0)
    assert [n.name for n in picked] == ["b300-2", "b300-3"]  # not b300-0 / b300-1, free and first in order


def test_a_node_of_the_placement_held_by_another_run_blocks_the_island(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.ssh_trainer_island import SshIslandError

    clock = {"t": 1000.0}
    now = lambda: clock["t"]  # noqa: E731
    trainers = [SimpleNamespace(name=f"b300-{i}") for i in range(4)]
    clock["t"] += 700
    other = _leases(tmp_path, now, identity="x", run="another-run/trainer")
    assert other.acquire("b300-3").ok  # another run took a node of the island meanwhile
    args = SimpleNamespace(cluster_prefix="run1-i0", _lease_pinned={"isl"})
    monkeypatch.setattr(sti, "_lease_record", lambda args: {"islands": {"isl": ["b300-2", "b300-3"]}})
    monkeypatch.setattr(sti, "_journal_incarnation", lambda args, name: False)
    b = _leases(tmp_path, now)
    with pytest.raises(SshIslandError, match=r"\[yeto-takeover-blocked\].*b300-3 is held by"):
        sti.lease_trainer_nodes(args, trainers, 2, b, name="isl", wait_s=0)
    assert b.holder("b300-2") is None  # what it took is given back; nothing taken elsewhere
    assert all(b.holder(n) is None for n in ("b300-0", "b300-1"))


def test_a_placement_the_inventory_does_not_hold_blocks_the_island(tmp_path, monkeypatch):
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.ssh_trainer_island import SshIslandError

    trainers = [SimpleNamespace(name=f"b300-{i}") for i in range(3)]
    monkeypatch.setattr(sti, "_lease_record", lambda args: {"islands": {"isl": ["b300-2", "b300-9"]}})
    b = _leases(tmp_path, lambda: 1000.0)
    args = SimpleNamespace(cluster_prefix="run1-i0", _lease_pinned={"isl"})
    with pytest.raises(SshIslandError, match=r"\[yeto-takeover-blocked\].*do not list \['b300-9'\]"):
        sti.lease_trainer_nodes(args, trainers, 2, b, name="isl", wait_s=0)
    assert all(b.holder(n.name) is None for n in trainers)


# -- L3: the takeover host's ledger holds the Modal call it adopted ----------------------------------

def test_an_adopted_modal_call_is_in_the_takeover_host_s_ledger(tmp_path):
    from yeto.cost.ledger import ResourceLedger
    from yeto.launch import trainer_provision as tp

    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    class Ops:
        app_name = "yeto-run1-i0"
        _app = SimpleNamespace(app_id="ap-1")

        def __init__(self):
            self.ledger = ledger

        def deploy(self):
            pass

    for takeover, expect in ((False, []), (True, ["fc-1"])):
        prov = SimpleNamespace(args=SimpleNamespace(rl_fleet_takeover=takeover), guard_order=None, on_call=None,
                               results={}, errors={}, adopted=lambda name: ("fc-1", None),
                               batch=lambda kind: {"isl": SimpleNamespace(task=SimpleNamespace(
                                   learner_id=0, function_name="island_0"))})
        tp._adapter("modal").spawn(prov, Ops())
        calls = [r.id for r in ledger.open_resources() if r.kind == "modal_call"]
        assert calls == expect and prov.results["isl"] == ("fc-1", None)
