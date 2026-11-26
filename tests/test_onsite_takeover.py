"""An on-site control plane moved to another host (own-cluster-fleet-stability 5.2): the
trainer worker of the new host takes the island launches of the lost host over, only once
their records lock's lease ran out and only through an exclusive claim (one of two hosts),
never a sky island; the launch it starts is the same launch (same prefix and incarnation),
which adopts the running island. Local store, fake launches."""

from __future__ import annotations

import threading

import pytest

from test_fleet_trainer_worker import FakeLaunches, fleet, member

from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool.store_lock import StoreLock
from yeto.rl import trainer_worker as tw

T0 = 1_800_000_000.0


def started_on_a(tmp_path, gpu: str = "ssh:2x8xb300"):
    """Islands 0 and 1 launched by host A; each launch holds its records lock (lease 120 s)."""
    objects, store = fleet(tmp_path, [member(0, gpu=gpu), member(1, gpu=gpu)])
    a = FakeLaunches()
    tw.TrainerWorker(store, objects, a, lifecycle.IslandStateStore(objects, log=lambda m: None),
                     log=lambda m: None, host="host-a", clock=lambda: T0).step()
    for k in (0, 1):
        assert StoreLock.acquire(objects, lifecycle.lock_prefix(k), holder=f"launch run-i{k} on host-a",
                                 ttl_s=120.0, clock=lambda: T0 + 5) is not None
    return objects, store, a


def worker_on(objects, store, host: str, at: float, logs: list | None = None, **kw):
    launches = FakeLaunches()
    w = tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=(logs.append if logs is not None else (lambda m: None)), host=host,
                         clock=lambda: at, takeover=True, **kw)
    return w, launches


def test_a_launch_whose_lease_still_runs_is_never_taken_over(tmp_path):
    objects, store, _a = started_on_a(tmp_path)
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", T0 + 60, logs)
    b.step()
    b.step()
    assert launches.starts == []
    assert b.load(0)["host"] == "host-a" and "taken_over" not in b.load(0)
    assert sum("still holds its records lock" in m for m in logs) == 2  # once per island
    assert not objects.list("fleet/trainers/0/takeover/")


def test_a_lost_host_s_launch_is_taken_over_as_the_same_launch(tmp_path):
    objects, store, a = started_on_a(tmp_path)
    lifecycle.IslandStateStore(objects, log=lambda m: None).request_stop(1, "operator")  # a STOP stays
    before = {k: tw.TrainerWorker(store, objects, a, lifecycle.IslandStateStore(objects)).load(k) for k in (0, 1)}
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", T0 + 600, logs)
    b.step()
    assert sorted(p for p, _ in launches.starts) == ["run-i0", "run-i1"]  # the same prefixes: no -m1, no new island
    for k in (0, 1):
        record = b.load(k)
        assert record["host"] == "host-b" and record["incarnation"] == before[k]["incarnation"] == 0
        assert record["relaunches"] == before[k]["relaunches"]  # not an infrastructure relaunch
        assert record["taken_over"]["from"] == "host-a" and record["taken_over"]["lease_token"] == 1
        assert record["why"] == "taken over from host host-a"
        assert objects.read(lifecycle.takeover_key(k, 1)) is not None
    assert lifecycle.IslandStateStore(objects).stop_requested(1) is not None  # not cleared by the takeover
    b.step()
    assert len(launches.starts) == 2 and b.load(0)["state"] == tw.RUNNING  # its launch runs: adopted
    assert any("taken over from host host-a" in m for m in logs)


def test_of_two_hosts_that_take_over_at_once_one_does(tmp_path):
    objects, store, _a = started_on_a(tmp_path)
    workers = [worker_on(objects, store, host, T0 + 600) for host in ("host-b", "host-c")]
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


def test_the_loser_of_a_claim_leaves_the_island_alone(tmp_path):
    objects, store, _a = started_on_a(tmp_path)
    objects.create(lifecycle.takeover_key(0, 1), b'{"host": "host-c"}')  # host C claimed the lapsed lease
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", T0 + 600, logs)
    b.step()
    assert [p for p, _ in launches.starts] == ["run-i1"]
    assert b.load(0)["host"] == "host-a" and any("taken over by host host-c" in m for m in logs)


@pytest.mark.parametrize("gpu", ["aws:2xh100@us-east-2", "nebius:8xh200"])
def test_a_sky_island_is_never_taken_over(tmp_path, gpu):
    objects, store, _a = started_on_a(tmp_path, gpu=gpu)
    logs: list[str] = []
    b, launches = worker_on(objects, store, "host-b", T0 + 6000, logs)
    b.step()
    assert launches.starts == [] and b.load(0)["host"] == "host-a"
    assert any("is never taken over from another host" in m for m in logs)


def test_a_launch_that_has_not_claimed_its_lock_yet_is_waited_for(tmp_path):
    objects, store = fleet(tmp_path, [member(0, gpu="modal:1x2xh100")])
    tw.TrainerWorker(store, objects, FakeLaunches(), lifecycle.IslandStateStore(objects, log=lambda m: None),
                     log=lambda m: None, host="host-a", clock=lambda: T0).step()
    b, launches = worker_on(objects, store, "host-b", T0 + 60)
    b.step()
    assert launches.starts == []  # just started on host A: its lock may come any moment
    later, launches = worker_on(objects, store, "host-b", T0 + tw.START_GRACE_S + 1)
    later.step()
    assert [p for p, _ in launches.starts] == ["run-i0"]  # it never claimed its lock: gone
    assert later.load(0)["taken_over"]["lease_token"] == 0


def test_without_an_on_site_control_plane_nothing_is_taken_over(tmp_path):
    objects, store, _a = started_on_a(tmp_path)
    b, launches = worker_on(objects, store, "host-b", T0 + 6000)
    b.takeover = False
    b.step()
    assert launches.starts == [] and b.load(0)["host"] == "host-a"
    assert not objects.list("fleet/trainers/0/takeover/")


def test_a_removed_island_of_a_lost_host_is_taken_over_then_removed(tmp_path):
    objects, store, _a = started_on_a(tmp_path)
    def drop(doc):
        doc["members"] = [m for m in doc["members"] if m.get("learner_id") != 1]

    store.update(drop)
    b, launches = worker_on(objects, store, "host-b", T0 + 600)
    b.step()
    record = b.load(1)
    assert record["host"] == "host-b" and record["state"] == tw.REMOVED
    assert "run-i1" not in [p for p, _ in launches.starts]


# -- the island launch on the new host: its SSH journal from the store record -------------------------

def _seed(tmp_path, monkeypatch, *, state="running", incarnation=0, env="1"):
    from types import SimpleNamespace

    from yeto import launcher, runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "host-b" / "runs")  # the new host: no run directory
    records = {0: {"state": state, "handle": {"cluster": "isl", "incarnation": incarnation},
                   "incarnation": incarnation}}
    refs = {"isl": lifecycle.IslandRef(0, "ssh", "trainer", None)}
    seeded = launcher._seed_ssh_journals(SimpleNamespace(cluster_prefix="run1"), records, refs, {"isl"},
                                         environ={launcher.JOURNAL_FROM_STORE_ENV: env} if env else {})
    return seeded, tmp_path / "host-b" / "runs" / "run1" / "ssh-islands"


def test_the_new_host_reattaches_to_the_running_ssh_incarnation_from_its_store_record(tmp_path, monkeypatch):
    from test_rl_ssh_resident import NAMES, FakeCluster, _crash, make_ops, spec

    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    from yeto.hw import card_tools

    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path / "host-a", cluster)  # host A: its journal in its own run directory
    first.add("isl", spec())
    assert first.start("isl") == 0
    _crash(first)  # host A is lost; the nodes run on
    seeded, journal = _seed(tmp_path, monkeypatch)
    assert seeded == ["isl"]
    from yeto.launch.ssh_resident import SshIslandOps

    again = SshIslandOps(run_id="run1", journal_dir=journal, events_dir=tmp_path / "events", ops=cluster,
                         sink_for=lambda name: (lambda line: None), log=lambda m: None)
    again.add("isl", spec())
    assert again.start("isl") == 0
    assert cluster.count("start") == 3 and cluster.count("follow", 0) == 3  # asked by ssh, followed: no 2nd copy
    assert "isl" in again.reattached and str(again.job_status("isl", 0)) == "RUNNING"


def test_without_the_store_record_the_new_host_would_start_a_second_copy(tmp_path, monkeypatch):
    """What the seeding prevents (and what stays as on main without an on-site control plane)."""
    seeded, journal = _seed(tmp_path, monkeypatch, env=None)
    assert seeded == [] and not journal.exists()
    for state in ("done", "failed", "down"):
        assert _seed(tmp_path, monkeypatch, state=state)[0] == []  # nothing that may run is seeded


def test_a_seeded_journal_still_refuses_a_node_that_does_not_answer(tmp_path, monkeypatch):
    from test_rl_ssh_resident import NAMES, FakeCluster, _crash, make_ops, spec
    from yeto.hw import card_tools
    from yeto.launch.ssh_resident import SshIslandOps
    from yeto.launch.ssh_trainer_island import SshIslandError

    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(card_tools, "check_cards", lambda family, text, **kw: [])
    cluster = FakeCluster(NAMES)
    first = make_ops(tmp_path / "host-a", cluster)
    first.add("isl", spec())
    first.start("isl")
    _crash(first)
    cluster.unreachable.add("b300-1")
    _seeded, journal = _seed(tmp_path, monkeypatch)
    again = SshIslandOps(run_id="run1", journal_dir=journal, events_dir=tmp_path / "events", ops=cluster,
                         sink_for=lambda name: (lambda line: None), log=lambda m: None)
    again.add("isl", spec())
    with pytest.raises(SshIslandError, match="refusing to start another copy"):
        again.start("isl")
    assert cluster.count("start") == 3 and cluster.count("kill") == 0
