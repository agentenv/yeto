"""Adopting the running trainer islands of a dead launcher (``yeto.launch.island_adopt``): the
plan per source (sky by cluster-name prefix and a refreshed status, Modal by function call
id, SSH by its journal), and ``launcher.run`` of a two-island Modal fleet restarted on
the same run: no new order and no new incarnation for an island that still runs, one
order for an island whose machine is gone. No cloud."""

from __future__ import annotations

import types

import pytest

import yeto.launcher as launcher
import yeto.modal_runner as modal_runner
from test_rl_algorithm_provenance import _fake_sky, _launcher_args  # noqa: F401  (fixture)
from test_rl_launcher_island_failure import Clock, _setup
from yeto.launch import island_adopt
from yeto.launch import island_lifecycle as lc
from yeto.rl.engine.object_store import DirectoryObjectStore


@pytest.fixture(autouse=True)
def _no_modal_listing(monkeypatch):
    monkeypatch.setattr("yeto.launch.modal_islands._list_modal_apps", lambda: [])
    monkeypatch.setattr("yeto.launch.modal_islands.MODAL_STOP_VERIFY_DELAY_S", 0.0)


REFS = {"r-l0-aws": lc.IslandRef(0, "sky"), "r-l1-modal": lc.IslandRef(1, "modal"),
        "r-l2-ssh": lc.IslandRef(2, "ssh")}


def _records():
    return {0: {"state": "running", "name": "r-l0-aws", "source": "sky", "incarnation": 2, "relaunches": 2,
                "handle": {"cluster": "r-l0-aws", "job_id": 4}},
            1: {"state": "running", "name": "r-l1-modal", "source": "modal", "incarnation": 0,
                "handle": {"cluster": "r-l1-modal", "call_id": "fc-abc"}},
            2: {"state": "running", "name": "r-l2-ssh", "source": "ssh", "incarnation": 3,
                "handle": {"cluster": "r-l2-ssh", "incarnation": 3}}}


def _probes(*, sky_status=None, alive=True, modal="RUNNING"):
    calls = {"status": 0}

    def status():
        calls["status"] += 1
        return sky_status if sky_status is not None else [
            {"name": "r-l0-aws", "status": types.SimpleNamespace(value="UP")},
            {"name": "other-run-l0-aws", "status": "UP"}]

    sky = island_adopt.SkyProbe("r-", status=status, job_alive=lambda c, j: alive)
    return {"sky": sky, "modal": island_adopt.ModalProbe(lambda call: modal)}, calls


def test_running_islands_are_adopted_with_their_incarnation():
    probes, calls = _probes()
    plan = island_adopt.plan(_records(), REFS, probes=probes)
    assert plan["r-l0-aws"].adopt and plan["r-l0-aws"].incarnation == 2 and plan["r-l0-aws"].job_id == 4
    assert plan["r-l1-modal"].adopt and plan["r-l1-modal"].job_id == "fc-abc"
    # ssh: its journal decides at its start (ssh_resident reattach rule); same incarnation number
    assert not plan["r-l2-ssh"].adopt and plan["r-l2-ssh"].incarnation == 3
    assert calls["status"] == 1  # one refreshed listing for every sky island


def test_a_gone_cluster_is_launched_again_as_the_next_incarnation():
    probes, _ = _probes(sky_status=[{"name": "other-run-l0-aws", "status": "UP"}])
    plan = island_adopt.plan(_records(), REFS, probes=probes)
    a = plan["r-l0-aws"]
    assert not a.adopt and a.incarnation == 3 and a.relaunches == 3 and "gone" in a.why


def test_a_dead_job_or_call_is_not_adopted():
    probes, _ = _probes(alive=False, modal="FAILED")
    plan = island_adopt.plan(_records(), REFS, probes=probes)
    assert not plan["r-l0-aws"].adopt and "not running" in plan["r-l0-aws"].why
    assert not plan["r-l1-modal"].adopt and plan["r-l1-modal"].incarnation == 1


def test_terminal_records_and_probe_errors_are_not_adopted():
    records = _records()
    records[1]["state"] = "done"

    def boom(handle):
        raise RuntimeError("sky api down")

    plan = island_adopt.plan(records, REFS, probes={"sky": boom, "modal": lambda h: (True, "")})
    assert not plan["r-l1-modal"].adopt and "done" in plan["r-l1-modal"].why
    assert not plan["r-l0-aws"].adopt and "probe failed" in plan["r-l0-aws"].why
    assert island_adopt.plan({}, REFS, probes={})["r-l0-aws"].incarnation == 0


def test_a_record_of_another_island_shape_is_not_adopted():
    records = _records()
    records[1]["source"] = "sky"  # island 1 moved from sky to Modal between launches
    probes, _ = _probes()
    assert not island_adopt.plan(records, REFS, probes=probes)["r-l1-modal"].adopt


# -- launcher.run of a restarted two-island Modal fleet ---------------------------------------------

def _fleet_args(tmp_path):
    args = _launcher_args("ports", ("--controller", "local", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64), gpu="modal:1xa100,modal:1xa100")
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    return args


def _store(monkeypatch, tmp_path):
    """The run store's island records on a local directory (the launch args keep no store:
    a synced test launch refuses one)."""
    monkeypatch.setattr(launcher, "_island_state_store",
                        lambda args: lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store")))


def _run(monkeypatch, tmp_path, *, seeded, alive_calls):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    fake = modal_runner.ModalOps
    asked: dict[str, int] = {}

    class AdoptingModalOps(fake):
        def __init__(self, app_name):
            super().__init__(app_name)
            for call, island in seeded.items():
                self.owner[call] = island

        def status(self, call_id):  # a seeded call runs until the controller has seen it once
            if call_id in alive_calls:
                asked[call_id] = asked.get(call_id, 0) + 1
                if asked[call_id] <= 1:
                    return "RUNNING"
            return super().status(call_id)

    monkeypatch.setattr(modal_runner, "ModalOps", AdoptingModalOps)
    monkeypatch.setattr(launcher, "_relaunch_of_same_run", lambda args: True)
    _store(monkeypatch, tmp_path)
    args = _fleet_args(tmp_path)
    names = launcher.learner_cluster_names(args.cluster_prefix, launcher.parse_gpu_spec(args.gpu))
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    for k, (call, island) in enumerate(seeded.items()):
        states.write(island, "running", name=names[island], source="modal", incarnation=4, relaunches=1,
                     handle={"cluster": names[island], "call_id": call})
    code = launcher.run(args)
    return code, record, states, names


def test_restarted_launcher_adopts_running_modal_islands_without_a_new_order(monkeypatch, tmp_path):
    seeded = {"fc-old-0": 0, "fc-old-1": 1}
    code, record, states, names = _run(monkeypatch, tmp_path, seeded=seeded, alive_calls=set(seeded))
    assert code in (0, 2, 3), code  # 2: Modal artifact not fetchable; 3: the fake tails echo no tape
    assert [r for r in record if r[0] == "spawn"] == []  # no new order
    records = states.read_all([0, 1])
    for k in (0, 1):
        assert records[k]["incarnation"] == 4 and records[k]["relaunches"] == 1  # no new incarnation
        assert records[k]["handle"]["call_id"] == f"fc-old-{k}"
        assert records[k]["adopted"] is True


def test_restarted_launcher_orders_once_for_the_island_whose_call_ended(monkeypatch, tmp_path):
    seeded = {"fc-old-0": 0, "fc-old-1": 1}
    code, record, states, names = _run(monkeypatch, tmp_path, seeded=seeded, alive_calls={"fc-old-0"})
    assert code in (0, 2, 3), code
    assert [r for r in record if r[0] == "spawn"] == [("spawn", 1)]  # one order, island 1 only
    records = states.read_all([0, 1])
    assert records[0]["incarnation"] == 4 and records[0]["handle"]["call_id"] == "fc-old-0"
    assert records[0]["adopted"] is True and records[1]["adopted"] is False
    assert records[1]["incarnation"] == 5 and records[1]["relaunches"] == 2
    assert records[1]["handle"]["call_id"] != "fc-old-1"
    assert any(h["to"] == "ordering" for h in records[1]["history"])


def test_a_single_island_launch_never_touches_the_store(monkeypatch, tmp_path):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0}, clock=clock, record=record)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64), gpu="modal:1xa100")
    args.keep, args.recover_timeout = False, 0.0
    _store(monkeypatch, tmp_path)
    launcher.run(args)
    assert not (tmp_path / "store" / "fleet").exists()


def test_a_single_island_launch_opted_in_records_and_adopts(monkeypatch, tmp_path):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0}, clock=clock, record=record)
    fake = modal_runner.ModalOps
    asked = {}

    class Adopting(fake):
        def __init__(self, app_name):
            super().__init__(app_name)
            self.owner["fc-old"] = 0

        def status(self, call_id):
            if call_id == "fc-old":
                asked[call_id] = asked.get(call_id, 0) + 1
                if asked[call_id] <= 1:
                    return "RUNNING"
            return super().status(call_id)

    monkeypatch.setattr(modal_runner, "ModalOps", Adopting)
    monkeypatch.setattr(launcher, "_relaunch_of_same_run", lambda args: True)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64, "--island-lifecycle", "on"),
                          gpu="modal:1xa100")
    args.keep, args.recover_timeout = False, 0.0
    _store(monkeypatch, tmp_path)
    name = launcher.learner_cluster_names(args.cluster_prefix, launcher.parse_gpu_spec(args.gpu))[0]
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    states.write(0, "running", name=name, source="modal", incarnation=2, handle={"cluster": name, "call_id": "fc-old"})
    launcher.run(args)
    assert [r for r in record if r[0] == "spawn"] == []
    assert states.read(0)["incarnation"] == 2 and states.read(0)["state"] == "done"


def test_restarted_launcher_adopts_sky_islands_and_their_syncer(monkeypatch, tmp_path):
    import sys

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    sky = sys.modules["sky"]
    launched = []
    real_launch = sky.launch
    sky.launch = lambda task, **kw: launched.append(kw.get("cluster_name")) or real_launch(task, **kw)
    monkeypatch.setattr(launcher, "_relaunch_of_same_run", lambda args: True)
    _store(monkeypatch, tmp_path)
    gpu = "aws:1xa100@us-east-1,aws:1xa100@us-east-1"
    args = _launcher_args("ports", ("--controller", "local", "--rl-image",
                                    "docker:ghcr.io/x/y@sha256:" + "a" * 64), gpu=gpu)
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    names = launcher.learner_cluster_names(args.cluster_prefix, launcher.parse_gpu_spec(gpu))
    syncer = launcher.sky_cluster_name(f"{args.cluster_prefix}-syncer")
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    for k, name in enumerate(names):
        states.write(k, "running", name=name, source="sky", incarnation=1, handle={"cluster": name, "job_id": 9})
    states.write_syncer(cluster=syncer, job_id=3, head_ip="10.0.0.9")
    up = {*names, syncer}
    monkeypatch.setattr(island_adopt.SkyProbe, "_sky_status",
                        staticmethod(lambda: [{"name": n, "status": "UP"} for n in up]))
    monkeypatch.setattr(island_adopt.SkyProbe, "_sky_job_alive", staticmethod(lambda c, j: True))
    code = launcher.run(args)
    assert code in (0, 1, 2, 3), code
    assert launched == []  # neither an island nor the syncer is ordered again
    records = states.read_all([0, 1])
    assert all(records[k]["incarnation"] == 1 and records[k]["adopted"] for k in (0, 1))
    assert all(records[k]["handle"]["job_id"] == 9 for k in (0, 1))


# -- one launcher adopts: the run store's single-writer lock, and the store is the basis -------------

def test_a_launcher_restarted_on_another_machine_adopts_from_the_store(monkeypatch, tmp_path):
    """No local run manifest (another control machine): the records in the run store decide."""
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    seeded = {"fc-old-0": 0, "fc-old-1": 1}
    fake = modal_runner.ModalOps
    asked: dict[str, int] = {}

    class AdoptingModalOps(fake):
        def __init__(self, app_name):
            super().__init__(app_name)
            self.owner.update(seeded)

        def status(self, call_id):
            if call_id in seeded:
                asked[call_id] = asked.get(call_id, 0) + 1
                if asked[call_id] <= 1:
                    return "RUNNING"
            return super().status(call_id)

    monkeypatch.setattr(modal_runner, "ModalOps", AdoptingModalOps)
    monkeypatch.setattr(launcher, "_relaunch_of_same_run", lambda args: False)  # nothing local
    _store(monkeypatch, tmp_path)
    args = _fleet_args(tmp_path)
    names = launcher.learner_cluster_names(args.cluster_prefix, launcher.parse_gpu_spec(args.gpu))
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    for call, island in seeded.items():
        states.write(island, "running", name=names[island], source="modal", incarnation=2, relaunches=0,
                     handle={"cluster": names[island], "call_id": call})
    launcher.run(args)
    assert [r for r in record if r[0] == "spawn"] == []  # no new order
    records = states.read_all([0, 1])
    assert all(records[k]["adopted"] and records[k]["incarnation"] == 2 for k in (0, 1))
    assert all(records[k]["fence_token"] >= 1 for k in (0, 1))  # written under the lock


def test_a_second_launcher_while_one_holds_the_records_launches_nothing(monkeypatch, tmp_path):
    from yeto.launch.pool.store_lock import StoreLock

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    _store(monkeypatch, tmp_path)
    objects = DirectoryObjectStore(tmp_path / "store")
    held = StoreLock.acquire(objects, launcher.ISLAND_RECORDS_LOCK, ttl_s=3600)  # a live first launcher
    assert held is not None
    real = launcher._claim_island_records
    monkeypatch.setattr(launcher, "_claim_island_records", lambda states: real(states, wait_s=0))
    code = launcher.run(_fleet_args(tmp_path))
    assert code == launcher.ISLAND_RECORDS_LOCKED_EXIT
    assert [r for r in record if r[0] in ("spawn", "stop_app")] == []  # nothing ordered, nothing touched
    assert lc.IslandStateStore(objects).read_all([0, 1]) == {}  # and no record written


def test_two_launchers_claiming_at_once_one_wins(tmp_path):
    import threading

    objects = DirectoryObjectStore(tmp_path / "store")
    results, start = [], threading.Barrier(2)

    def claim():
        states = lc.IslandStateStore(objects)
        start.wait()
        results.append(launcher._claim_island_records(states, wait_s=0))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    won = [lock for lock, _why in results if lock is not None]
    refused = [why for lock, why in results if lock is None]
    assert len(won) == 1 and len(refused) == 1 and "another launcher holds" in refused[0]


def test_a_dead_launchers_lease_is_waited_out(tmp_path):
    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    t = [1000.0]
    dead = StoreLock.acquire(objects, launcher.ISLAND_RECORDS_LOCK, ttl_s=120, clock=lambda: t[0])
    assert dead is not None  # it never renews again
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += s

    states = lc.IslandStateStore(objects)
    lock, why = launcher._claim_island_records(states, wait_s=300, sleep=sleep, clock=lambda: t[0])
    assert lock is not None and why is None and lock.token == dead.token + 1
    assert states.fence is lock and 115 <= sum(slept) <= 135


def test_a_fenced_controller_orders_nothing():
    from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread

    from yeto.launch.fleet import FleetController

    class Store(lc.IslandStateStore):
        pass

    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [FAILED]
    ops.status_seq["r-l1-modal"] = [RUNNING] * 3 + [SUCCEEDED]
    ops.status_seq[SYNCER] = [RUNNING]
    store = Store(DirectoryObjectStore("/nonexistent-t1"), log=lambda m: None)
    store.fenced = RuntimeError("lock token 1 is fenced by token 2")
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    ctl = FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                          elastic=True, recorder=lc.FleetRecorder(store, refs))
    ctl.run()
    assert ops.relaunch_calls == []  # the launcher that took the records over relaunches, not this one



def test_an_adopted_islands_tape_continues_without_duplicates(tmp_path):
    """The restarted launcher's collector of an adopted island appends to the island's tape;
    a log stream that replays from the start writes no record twice."""
    from yeto.rl.event_echo import TapeCollector, format_record

    tape = tmp_path / "events" / "r-l1-modal.jsonl"
    first = TapeCollector(tape, fresh=False)
    lines = [format_record({"event": "rl_round_trained", "island_id": 1, "round": r, "time_unix": 1.0 + r}) for r in range(3)]
    for line in lines:
        first.feed(line + "\n")
    assert len(tape.read_text().splitlines()) == 3
    again = TapeCollector(tape, fresh=False)  # the next launcher, same machine
    for line in lines:  # the adopted call's log stream replays
        again.feed(line + "\n")
    again.feed(format_record({"event": "rl_round_trained", "island_id": 1, "round": 3, "time_unix": 4.0}) + "\n")
    rows = tape.read_text().splitlines()
    assert len(rows) == 4 and again.count == 1
