"""An on-site control plane moved to another host (own-cluster-fleet-stability 5.2): the
trainer worker of the new host takes the island launches of the lost host over, only once the
fleet's syncer is known, their records lock's lease ended ``takeover_margin_s`` ago and their
launch started at least the start grace ago, and only by claiming that lock (its token fences
the old launch), never a sky island. The launch it starts is the same launch (same prefix and
incarnation), which adopts the running island; a moving or stopping island keeps its STOP and
finishes its move or removal, or is blocked when it cannot reattach. Local store, fake launches."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from test_fleet_trainer_worker import FakeLaunches, fleet, member

from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool.store_lock import LockLost, StoreLock
from yeto.rl import island_launches as il
from yeto.rl import trainer_worker as tw

T0 = 1_800_000_000.0
LATER = T0 + tw.START_GRACE_S + 1  # past the start grace of a launch started at T0


def started_on_a(tmp_path, gpu: str = "ssh:2x8xb300", clock=None):
    """Islands 0 and 1 launched by host A; each launch holds its records lock (lease 120 s)."""
    objects, store = fleet(tmp_path, [member(0, gpu=gpu), member(1, gpu=gpu)])
    a = FakeLaunches()
    tw.TrainerWorker(store, objects, a, lifecycle.IslandStateStore(objects, log=lambda m: None),
                     log=lambda m: None, host="host-a", clock=lambda: T0).step()
    locks = {k: StoreLock.acquire(objects, lifecycle.lock_prefix(k), holder=f"launch run-i{k} on host-a",
                                  ttl_s=120.0, clock=clock or (lambda: T0 + 5)) for k in (0, 1)}
    assert all(locks.values())
    return objects, store, a, locks


def worker_on(objects, store, host: str, at, logs: list | None = None, launches=None, **kw):
    launches = launches or FakeLaunches()
    w = tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=(logs.append if logs is not None else (lambda m: None)), host=host,
                         clock=(at if callable(at) else (lambda: at)), takeover=True, **kw)
    return w, launches


def test_a_launch_whose_lease_still_runs_is_never_taken_over(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", T0 + 60, logs)
    b.step()
    b.step()
    assert launches.starts == []
    assert b.load(0)["host"] == "host-a" and "taken_over" not in b.load(0)
    assert sum("still holds its records lock" in m for m in logs) == 2  # once per island
    assert not objects.list("fleet/trainers/0/takeover/")


def test_a_lost_host_s_launch_is_taken_over_as_the_same_launch(tmp_path):
    objects, store, a, _locks = started_on_a(tmp_path)
    lifecycle.IslandStateStore(objects, log=lambda m: None).request_stop(1, "operator")  # a STOP stays
    before = {k: tw.TrainerWorker(store, objects, a, lifecycle.IslandStateStore(objects)).load(k) for k in (0, 1)}
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", LATER, logs)
    b.step()
    assert sorted(p for p, _ in launches.starts) == ["run-i0", "run-i1"]  # the same prefixes: no -m1, no new island
    for k in (0, 1):
        record = b.load(k)
        assert record["host"] == "host-b" and record["incarnation"] == before[k]["incarnation"] == 0
        assert record["relaunches"] == before[k]["relaunches"]  # not an infrastructure relaunch
        assert record["taken_over"]["from"] == "host-a" and record["taken_over"]["token"] == 2
        assert record["why"] == "taken over from host host-a"
        assert record["state"] == tw.STARTING  # nothing more for it in the step that took it over
        assert json.loads(objects.read(lifecycle.takeover_key(k, 2)))["host"] == "host-b"  # a note
        # the takeover's claim is released at once: the launch started here claims the next token
        assert StoreLock.holder_of(objects, lifecycle.lock_prefix(k), now=LATER) is None
        assert StoreLock.tokens(objects, lifecycle.lock_prefix(k)) == [1, 2]
    assert lifecycle.IslandStateStore(objects).stop_requested(1) is not None  # not cleared by the takeover
    b.step()
    assert len(launches.starts) == 2 and b.load(0)["state"] == tw.RUNNING  # its launch runs: adopted
    assert any("taken over from host host-a" in m for m in logs)


def test_the_lost_launch_s_lock_is_fenced_by_the_takeover(tmp_path):
    """Host A's launch was only cut off: once host B took its island over, its next renewal
    finds the higher token and fails (LockLost); it orders nothing more."""
    clock = {"t": T0 + 5}
    objects, store, _a, locks = started_on_a(tmp_path, clock=lambda: clock["t"])
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    assert [p for p, _ in launches.starts] == ["run-i0", "run-i1"]
    clock["t"] = LATER + 1  # host A is reachable again
    with pytest.raises(LockLost):
        locks[0].renew()


def test_a_lease_that_comes_back_during_the_takeover_makes_it_fail(tmp_path):
    clock = {"t": T0 + 5}
    objects, store, _a, locks = started_on_a(tmp_path, clock=lambda: clock["t"])

    class Racing:
        """The run store as host B sees it: host A's launch renews its lease between B's look at
        the lock and B's claim of it (the claim's own listing of the tokens)."""

        def __init__(self, inner):
            self.inner, self.lists = inner, 0

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def list(self, prefix, **kw):
            if prefix.startswith(lifecycle.lock_prefix(0)):
                self.lists += 1
                if self.lists == 2:  # the acquire's listing: host A renews right before it
                    clock["t"] = LATER - 1
                    locks[0].renew()
            return self.inner.list(prefix, **kw)

    logs: list[str] = []
    racing = Racing(objects)
    b, launches = worker_on(racing, store, "host-b", LATER, logs)
    b.step()
    assert [p for p, _ in launches.starts] == ["run-i1"]  # island 0 is not taken over
    assert b.load(0)["host"] == "host-a" and any("is held again" in m for m in logs)
    assert StoreLock.tokens(objects, lifecycle.lock_prefix(0)) == [1]  # no claim made over the live lease
    locks[0].renew()  # host A still holds it


def test_of_two_hosts_that_take_over_at_once_one_does(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    workers = [worker_on(objects, store, host, LATER) for host in ("host-b", "host-c")]
    gate = threading.Barrier(2)

    def take(w):
        gate.wait()
        w.step()

    threads = [threading.Thread(target=take, args=(w,)) for w, _ in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    starts = [p for _, launches in workers for p, _ in launches.starts]
    assert sorted(starts) == ["run-i0", "run-i1"]  # each island once, on one host
    for k in (0, 1):
        winner = workers[0][0].load(k)["host"]
        assert winner in ("host-b", "host-c")
        (launches,) = [lc for w, lc in workers if w.host == winner]
        assert launches.started(f"run-i{k}") == 1
    for w, _ in workers:  # and later steps change nothing
        w.step()
    assert sorted(p for _, launches in workers for p, _ in launches.starts) == ["run-i0", "run-i1"]


def test_another_host_s_claim_in_flight_leaves_the_island_alone(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    assert StoreLock.acquire(objects, lifecycle.lock_prefix(0), holder="takeover of island 0 by host-c",
                             ttl_s=120.0, clock=lambda: LATER)  # host C is taking island 0 over right now
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", LATER, logs)
    b.step()
    assert [p for p, _ in launches.starts] == ["run-i1"]
    assert b.load(0)["host"] == "host-a" and any("still holds its records lock" in m for m in logs)
    assert StoreLock.tokens(objects, lifecycle.lock_prefix(0)) == [1, 2]  # host C's claim only


@pytest.mark.parametrize("gpu", ["aws:2xh100@us-east-2", "nebius:8xh200"])
def test_a_sky_island_is_never_taken_over(tmp_path, gpu):
    objects, store, _a, _locks = started_on_a(tmp_path, gpu=gpu)
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", T0 + 6000, logs)
    b.step()
    assert launches.starts == [] and b.load(0)["host"] == "host-a"
    assert any("is never taken over from another host" in m for m in logs)


def test_no_island_is_taken_over_before_the_fleet_s_syncer_is_known(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    objects.delete(il.SYNCER_RECORD)  # the syncer service is not recorded (yet)
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    assert launches.starts == [] and b.load(0)["host"] == "host-a"
    assert StoreLock.tokens(objects, lifecycle.lock_prefix(0)) == [1]  # not even claimed


def test_a_launch_that_has_not_claimed_its_lock_yet_is_waited_for(tmp_path):
    objects, store = fleet(tmp_path, [member(0, gpu="modal:1x2xh100")])
    tw.TrainerWorker(store, objects, FakeLaunches(), lifecycle.IslandStateStore(objects, log=lambda m: None),
                     log=lambda m: None, host="host-a", clock=lambda: T0).step()
    b, launches = worker_on(objects, store, "host-b", T0 + 60)
    b.step()
    assert launches.starts == []  # just started on host A: its lock may come any moment
    later, launches = worker_on(objects, store, "host-b", LATER)
    later.step()
    assert [p for p, _ in launches.starts] == ["run-i0"]  # it never claimed its lock: gone
    assert later.load(0)["taken_over"]["token"] == 1


def test_a_host_that_takes_over_and_dies_is_taken_over_in_turn_after_the_grace(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    b, launches_b = worker_on(objects, store, "host-b", LATER)
    b.step()
    assert b.load(0)["host"] == "host-b"
    # host B's launch claims the records lock, then host B dies (its lease ends 120 s later)
    assert StoreLock.acquire(objects, lifecycle.lock_prefix(0), holder="launch run-i0 on host-b", ttl_s=120.0,
                             clock=lambda: LATER + 10)
    c, launches_c = worker_on(objects, store, "host-c", LATER + 400)
    c.step()
    assert launches_c.starts == []  # its lease ended, but host B's launch started (taken over) < the grace ago
    c, launches_c = worker_on(objects, store, "host-c", LATER + tw.START_GRACE_S + 1)
    c.step()
    assert launches_c.started("run-i0") == 1
    record = c.load(0)
    assert record["host"] == "host-c" and record["taken_over"]["from"] == "host-b"


def test_a_host_that_takes_a_moving_island_over_and_dies_is_waited_for_from_its_takeover(tmp_path):
    objects, store = moving_on_a(tmp_path)
    b, _launches_b = worker_on(objects, store, "host-b", LATER)
    b.step()
    assert b.load(0)["host"] == "host-b" and b.load(0)["started_at"] == LATER  # the takeover starts its grace
    c, launches_c = worker_on(objects, store, "host-c", LATER + 400)
    c.step()
    assert launches_c.started("run-i0") == 0  # host B's launch may not have claimed its lock yet
    c, launches_c = worker_on(objects, store, "host-c", LATER + tw.START_GRACE_S + 1)
    c.step()
    assert launches_c.started("run-i0") == 1 and c.load(0)["taken_over"]["from"] == "host-b"


@pytest.mark.parametrize("margin, taken", [(60.0, False), (10.0, True)])
def test_a_lease_is_taken_over_only_its_margin_after_it_ended(tmp_path, margin, taken):
    objects, store, _a, _locks = started_on_a(tmp_path, clock=lambda: LATER - 140)  # lease ends LATER - 20
    b, launches = worker_on(objects, store, "host-b", LATER, takeover_margin_s=margin)
    b.step()
    assert (launches.started("run-i0") == 1) is taken


def test_without_an_on_site_control_plane_nothing_is_taken_over(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)
    b, launches = worker_on(objects, store, "host-b", T0 + 6000)
    b.takeover = False
    b.step()
    assert launches.starts == [] and b.load(0)["host"] == "host-a"
    assert not objects.list("fleet/trainers/0/takeover/")


@pytest.mark.parametrize("state, host, lost", [
    (tw.RUNNING, "host-a", True), (tw.MOVING, "host-a", True), (tw.STOPPING, "host-a", True),
    (tw.RUNNING, "host-b", False), (tw.RUNNING, None, False), (tw.DOWN, "host-a", False),
    *[(s, "host-a", False) for s in sorted(tw.TERMINAL)]])
def test_only_a_live_launch_of_another_host_is_lost(tmp_path, state, host, lost):
    objects, store = fleet(tmp_path, [member(0)])
    b, _ = worker_on(objects, store, "host-b", LATER)
    assert b._lost_host(0, {"state": state, "host": host}) is lost


def moving_on_a(tmp_path, *, remove: bool = False):
    """Island 0 moving to other cards (or removed) on host A: its STOP written, then host A lost."""
    objects, store, a, _locks = started_on_a(tmp_path)
    a_worker = tw.TrainerWorker(store, objects, a, lifecycle.IslandStateStore(objects, log=lambda m: None),
                                log=lambda m: None, host="host-a", clock=lambda: T0 + 10)
    a_worker.step()  # adopted (running)

    def change(doc):
        if remove:
            doc["members"] = [m for m in doc["members"] if m.get("learner_id") != 0]
        else:
            for m in doc["members"]:
                if m.get("learner_id") == 0:
                    m["gpu"] = "ssh:2x8xh200"

    store.update(change)
    a_worker.step()
    assert a_worker.load(0)["state"] == (tw.STOPPING if remove else tw.MOVING)
    assert lifecycle.IslandStateStore(objects).stop_requested(0) is not None
    return objects, store


@pytest.mark.parametrize("remove", [False, True])
def test_a_moving_or_stopping_island_is_reattached_with_its_stop_and_finishes_after_its_release(tmp_path, remove):
    objects, store = moving_on_a(tmp_path, remove=remove)
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    record = b.load(0)
    assert launches.started("run-i0") == 1  # the same launch: same prefix and incarnation, its own argv
    assert record["host"] == "host-b" and record["incarnation"] == 0 and record["reattach"]
    assert record["state"] == (tw.STOPPING if remove else tw.MOVING)
    assert lifecycle.IslandStateStore(objects).stop_requested(0) is not None  # its STOP is kept
    b.step()
    assert len(launches.starts) == 2  # island 1 too; nothing new for island 0 while its launch runs
    launches.end("run-i0", 0, "island STOP: released")  # it read its STOP and released the island
    b.step()
    record = b.load(0)
    if remove:
        assert record["state"] == tw.REMOVED
    else:  # the move goes on: the next incarnation on the new cards
        assert record["incarnation"] == 1 and record["gpu"] == "ssh:2x8xh200" and not record.get("reattach")


@pytest.mark.parametrize("remove", [False, True])
def test_a_reattach_that_fails_blocks_the_island_and_starts_nothing_new(tmp_path, remove):
    objects, store = moving_on_a(tmp_path, remove=remove)
    seen: list[dict] = []
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", LATER, logs, on_event=seen.append)
    b.step()
    launches.end("run-i0", 1, f"{tw.TAKEOVER_BLOCKED_MARKER} ssh: island isl0: not running on every node")
    b.step()
    b.step()
    record = b.load(0)
    assert record["takeover_blocked"] and record["incarnation"] == 0
    assert record["state"] == (tw.STOPPING if remove else tw.MOVING)  # its records as they were
    assert launches.started("run-i0") == 1 and not any(p.startswith("run-i0-") for p, _ in launches.starts)
    assert [e["event"] for e in seen] == ["island_takeover_blocked"]  # once
    assert any("island_takeover_blocked" in m for m in logs)


def test_a_removed_island_of_a_lost_host_is_taken_over_then_removed(tmp_path):
    objects, store, _a, _locks = started_on_a(tmp_path)

    def drop(doc):
        doc["members"] = [m for m in doc["members"] if m.get("learner_id") != 1]

    store.update(drop)
    b, launches = worker_on(objects, store, "host-b", LATER)
    b.step()
    record = b.load(1)
    assert record["host"] == "host-b" and record["reattach"] and launches.started("run-i1") == 1
    b.step()
    assert b.load(1)["state"] == tw.STOPPING  # its STOP written, its launch stops
    launches.end("run-i1", 0, "island STOP")
    b.step()
    assert b.load(1)["state"] == tw.REMOVED


def test_a_fenced_worker_of_an_on_site_control_plane_forwards_no_fleet_stop(tmp_path):
    objects, store, a, _locks = started_on_a(tmp_path)
    fenced = {"why": "run up's state in the run store is written by host host-b now"}
    w = tw.TrainerWorker(store, objects, a, lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=lambda m: None, host="host-a", clock=lambda: T0 + 10, takeover=True,
                         fleet_stop=lambda: "run STOP", fenced=lambda: fenced["why"])
    w.step()
    assert a.stops == [] and lifecycle.IslandStateStore(objects).stop_requested(0) is None
    w.takeover = False  # without an on-site control plane: the STOP first, as before
    w.step()
    assert a.stops


def test_the_takeover_margin_is_a_fleet_key_of_an_on_site_control_plane_only():
    from yeto.rl import fleet

    assert fleet.takeover_margin_s({"syncer_service": "h"}) == 60.0
    assert fleet.takeover_margin_s({"syncer_service": "h", "takeover_margin_s": 5}) == 5.0
    for bad in ({"takeover_margin_s": 5}, {"syncer_service": "h", "takeover_margin_s": -1},
                {"syncer_service": "h", "takeover_margin_s": True}):
        with pytest.raises(Exception, match="takeover_margin_s"):
            fleet.takeover_margin_s(bad)


# -- the island launch on the new host: its SSH journal from the store record -------------------------

def _seed(tmp_path, monkeypatch, *, state="running", incarnation=0, env="1", stop=False):
    from yeto import launcher, runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "host-b" / "runs")  # the new host: no run directory
    records = {0: {"state": state, "handle": {"cluster": "isl", "incarnation": incarnation},
                   "incarnation": incarnation}}
    refs = {"isl": lifecycle.IslandRef(0, "ssh", "trainer", None)}
    states = SimpleNamespace(stop_requested=lambda k: {"reason": "moving"} if stop else None)
    args = SimpleNamespace(cluster_prefix="run1")
    seeded = launcher._seed_ssh_journals(args, records, refs, {"isl"},
                                         environ={launcher.JOURNAL_FROM_STORE_ENV: env} if env else {},
                                         states=states)
    return seeded, tmp_path / "host-b" / "runs" / "run1" / "ssh-islands", args


def _ssh_fixture(tmp_path, monkeypatch):
    from test_rl_ssh_resident import NAMES, FakeCluster, _crash, make_ops, spec
    from yeto.hw import card_tools

    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path / "host-a", cluster)  # host A: its journal in its own run directory
    first.add("isl", spec())
    assert first.start("isl") == 0
    _crash(first)  # host A is lost; the nodes run on
    return cluster, spec


def _again(cluster, spec, journal, tmp_path, **kw):
    from yeto.launch.ssh_resident import SshIslandOps

    again = SshIslandOps(run_id="run1", journal_dir=journal, events_dir=tmp_path / "events", ops=cluster,
                         sink_for=lambda name: (lambda line: None), log=lambda m: None, **kw)
    again.add("isl", spec())
    return again


def test_the_new_host_reattaches_to_the_running_ssh_incarnation_from_its_store_record(tmp_path, monkeypatch):
    cluster, spec = _ssh_fixture(tmp_path, monkeypatch)
    seeded, journal, args = _seed(tmp_path, monkeypatch)
    assert seeded == ["isl"] and not getattr(args, "_reattach_only", None)
    again = _again(cluster, spec, journal, tmp_path)
    assert again.start("isl") == 0
    assert cluster.count("start") == 3 and cluster.count("follow", 0) == 3  # asked by ssh, followed: no 2nd copy
    assert "isl" in again.reattached and str(again.job_status("isl", 0)) == "RUNNING"


def test_without_the_store_record_the_new_host_would_start_a_second_copy(tmp_path, monkeypatch):
    """What the seeding prevents (and what stays as on main without an on-site control plane)."""
    seeded, journal, _args = _seed(tmp_path, monkeypatch, env=None)
    assert seeded == [] and not journal.exists()
    for state in ("done", "failed", "down"):
        assert _seed(tmp_path, monkeypatch, state=state)[0] == []  # nothing that may run is seeded


def test_a_seeded_journal_still_refuses_a_node_that_does_not_answer(tmp_path, monkeypatch):
    from yeto.launch.ssh_trainer_island import SshIslandError

    cluster, spec = _ssh_fixture(tmp_path, monkeypatch)
    cluster.unreachable.add("b300-1")
    _seeded, journal, _args = _seed(tmp_path, monkeypatch)
    again = _again(cluster, spec, journal, tmp_path)
    with pytest.raises(SshIslandError, match="refusing to start another copy"):
        again.start("isl")
    assert cluster.count("start") == 3 and cluster.count("kill") == 0


def test_a_moving_island_s_launch_may_only_reattach(tmp_path, monkeypatch):
    """Its STOP is set (it was moving or stopping): the launch reattaches when its incarnation
    runs on every node; not whole, it neither releases nor starts anything (the marker the
    trainer worker blocks the island on)."""
    from yeto.launch.ssh_resident import TAKEOVER_BLOCKED_MARKER
    from yeto.launch.ssh_trainer_island import SshIslandError

    cluster, spec = _ssh_fixture(tmp_path, monkeypatch)
    _seeded, journal, args = _seed(tmp_path, monkeypatch, stop=True)
    assert args._reattach_only == {"isl"}
    again = _again(cluster, spec, journal, tmp_path, reattach_only=args._reattach_only)
    assert again.start("isl") == 0 and "isl" in again.reattached  # whole: reattached
    assert TAKEOVER_BLOCKED_MARKER == tw.TAKEOVER_BLOCKED_MARKER

    cluster2, spec2 = _ssh_fixture(tmp_path / "2", monkeypatch)
    cluster2.runs["b300-1"][0].update(state="exited", rc=1)  # not whole any more
    _seeded, journal2, args2 = _seed(tmp_path / "2", monkeypatch, stop=True)
    again2 = _again(cluster2, spec2, journal2, tmp_path / "2", reattach_only=args2._reattach_only)
    with pytest.raises(SshIslandError, match=r"\[yeto-takeover-blocked\]"):
        again2.start("isl")
    assert cluster2.count("start") == 3 and cluster2.count("kill") == 0  # nothing released, nothing started


def test_the_journal_is_seeded_only_under_the_records_lock(tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.launch import island_adopt

    calls = []
    monkeypatch.setattr(launcher, "_seed_ssh_journals", lambda *a, **kw: calls.append(a) or [])
    monkeypatch.setattr(launcher, "_island_refs", lambda *a, **kw: {"isl": lifecycle.IslandRef(0, "ssh")})
    monkeypatch.setattr(launcher, "_record_id", lambda args, m: m)
    monkeypatch.setattr(launcher, "_adopt_probes", lambda args: None)
    monkeypatch.setattr(island_adopt, "plan", lambda *a, **kw: {})
    from yeto.rl.engine.object_store import DirectoryObjectStore

    states = lifecycle.IslandStateStore(DirectoryObjectStore(tmp_path / "store"), log=lambda m: None)
    kw = dict(relaunch=True, ssh_names={"isl"}, events_dir=None, echo_names=(), held_name=None, syncer_cluster=None)
    specs = [SimpleNamespace(cloud="ssh")]
    launcher._plan_island_records(SimpleNamespace(), specs, ["isl"], states, None, **kw)
    assert calls == []  # no lock (a store without an exclusive create): nothing seeded
    launcher._plan_island_records(SimpleNamespace(), specs, ["isl"], states, object(), **kw)
    assert len(calls) == 1


def test_the_island_s_node_lease_identity_is_the_same_on_the_new_host(tmp_path, monkeypatch):
    from yeto import launcher, runs
    from yeto.launch.node_lease import local_identity
    from yeto.launch.ssh_trainer_island import _lease_record_path
    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path / "store")
    nodes = type("Nodes", (list,), {"lease_store": str(tmp_path / "leases")})()
    args = SimpleNamespace(rl_fleet_role="island", rl_fleet_learner_id=0, cluster_prefix="run1-i0",
                           rl_node_inventory_nodes=nodes, dry_run=False)
    env = {launcher.JOURNAL_FROM_STORE_ENV: "1"}
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "host-a" / "runs")
    path_a = _lease_record_path(args)
    path_a.parent.mkdir(parents=True)
    path_a.write_text(json.dumps({"identity": "host-a-own"}))  # the identity its nodes are leased under
    on_a = launcher._restore_lease_identity(args, environ=env, objects=objects)
    assert on_a == "host-a-own" == local_identity(path_a)  # host A's own, now in the run store
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "host-b" / "runs")  # the new host: another run dir
    path_b = _lease_record_path(args)
    path_b.parent.mkdir(parents=True)
    path_b.write_text(json.dumps({"identity": "host-b-own", "islands": {"isl": ["n1"]}}))
    on_b = launcher._restore_lease_identity(args, environ=env, objects=objects)
    assert on_b == on_a == local_identity(path_b)  # the same holder of the island's node leases
    assert json.loads(path_b.read_text())["islands"] == {"isl": ["n1"]}  # the rest of the file kept
    # not an on-site control plane, not an island launch, or no lease_store: nothing changes
    assert launcher._restore_lease_identity(args, environ={}, objects=objects) is None
    assert launcher._restore_lease_identity(SimpleNamespace(**{**vars(args), "rl_fleet_role": None}), environ=env,
                                            objects=objects) is None
    assert launcher._restore_lease_identity(SimpleNamespace(**{**vars(args), "rl_node_inventory_nodes": []}),
                                            environ=env, objects=objects) is None
