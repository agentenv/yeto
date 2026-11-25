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


def test_terminal_records_are_not_adopted():
    records = _records()
    records[1]["state"] = "done"
    plan = island_adopt.plan(records, REFS, probes={"sky": lambda h: (True, "up"), "modal": lambda h: (True, "")})
    assert not plan["r-l1-modal"].adopt and "done" in plan["r-l1-modal"].why
    assert plan["r-l0-aws"].adopt
    assert island_adopt.plan({}, REFS, probes={})["r-l0-aws"].incarnation == 0


def test_a_probe_that_cannot_tell_orders_nothing():
    """Unknown is not gone: an order then could start the island twice."""
    def boom(handle):
        raise RuntimeError("sky api down")

    with pytest.raises(island_adopt.AdoptionUnknown, match="r-l0-aws.*unknown.*sky api down"):
        island_adopt.plan(_records(), REFS, probes={"sky": boom, "modal": lambda h: (True, "")})
    probe = island_adopt.SkyProbe("r-", status=lambda: (_ for _ in ()).throw(RuntimeError("no listing")),
                                  job_alive=lambda c, j: True)
    with pytest.raises(island_adopt.AdoptionUnknown, match="no listing"):
        island_adopt.plan(_records(), REFS, probes={"sky": probe, "modal": lambda h: (True, "")})


@pytest.mark.parametrize("state", ["ordering", "recovering"])
def test_an_ordering_or_recovering_record_is_probed_and_adopted(state):
    """A launcher that died mid-order or mid-relaunch: the machine its record names may run."""
    records = _records()
    records[1]["state"] = state
    plan = island_adopt.plan(records, REFS, probes={"sky": lambda h: (True, ""), "modal": lambda h: (True, "up")})
    assert plan["r-l1-modal"].adopt and plan["r-l1-modal"].incarnation == 0
    records[1]["handle"] = {"cluster": "r-l1-modal"}  # ordered, its call not recorded yet
    plan = island_adopt.plan(records, REFS, probes={"sky": lambda h: (True, ""),
                                                    "modal": island_adopt.ModalProbe(lambda c: "RUNNING")})
    assert not plan["r-l1-modal"].adopt and plan["r-l1-modal"].incarnation == 1


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


def _run(monkeypatch, tmp_path, *, seeded, alive_calls, before=None):
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
    if before is not None:
        before(args, names)
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
    monkeypatch.setattr(launcher, "_claim_island_records", lambda states, **kw: real(states, **kw, wait_s=0))
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


def test_a_finished_launcher_hands_the_records_on_at_once(monkeypatch, tmp_path):
    """The launcher writes its last records, then releases the lock: the next launcher of
    the run claims it without waiting out the lease, under a higher token."""
    seeded = {"fc-old-0": 0, "fc-old-1": 1}
    _run(monkeypatch, tmp_path, seeded=seeded, alive_calls=set(seeded))
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    lock, why = launcher._claim_island_records(states, wait_s=0)
    assert lock is not None and why is None and lock.token >= 2


def test_a_released_or_lost_fence_stops_every_later_write(tmp_path):
    """A lock error that names a holder (``LockLost`` and its ``LockReleased``) is sticky:
    nothing is written after it, also not a record queued before it."""

    class Released(BaseException):
        def __init__(self):
            super().__init__("lock token 3: the lock was released")
            self.token = self.by = 3

    class Fence:
        token, released = 3, False

        def guard(self):
            if self.released:
                raise Released()

    objects = DirectoryObjectStore(tmp_path / "store")
    fence = Fence()
    states = lc.IslandStateStore(objects, fence=fence, background=True)
    states.write(0, "running", name="r-l0", source="modal", incarnation=0)
    assert states.flush(5)
    fence.released = True
    states.write(0, "done", name="r-l0", source="modal", incarnation=0)
    states.close(5)
    assert lc.IslandStateStore(objects).read_all([0])[0]["state"] == "running"
    assert states.fenced is not None


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
    with pytest.raises(lc.IslandRecordsFenced):  # supervision ends at once
        ctl.run()
    assert ops.relaunch_calls == []  # the launcher that took the records over relaunches, not this one
    assert ops.down_calls == []  # ... and tears down, not this one


def test_a_controller_fenced_while_supervising_tears_nothing_down():
    """The fence is lost mid-run (a recovering island past its timeout would be abandoned and
    torn down): this controller tears nothing down and stops."""
    from test_controller import FAILED, RUNNING, SYNCER, FakeOps, ImmediateThread

    from yeto.launch.fleet import FleetController

    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [FAILED]
    ops.status_seq["r-l1-modal"] = [RUNNING] * 50
    ops.status_seq[SYNCER] = [RUNNING] * 50
    store = lc.IslandStateStore(DirectoryObjectStore("/nonexistent-t1"), log=lambda m: None)
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    ctl = FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=0, thread_cls=ImmediateThread,
                          elastic=True, recorder=lc.FleetRecorder(store, refs))
    rec = ctl.learners["r-l0-modal"]
    rec.update(state="recovering", failed_at=ops.now() - 10, attempt=None)
    store.fenced = RuntimeError("lock token 1 is fenced by token 2")
    ctl._drive_recovery(rec, is_syncer=False)  # past its timeout: abandoned ...
    from yeto.launch.fleet import ABANDONED

    assert rec["state"] == ABANDONED
    assert ops.down_calls == [] and ops.relaunch_calls == []  # ... but not torn down by this controller



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


# -- REV #750: the lease from the claim on, unknown is not gone, hands off once fenced --------

def test_a_record_that_does_not_read_orders_nothing(monkeypatch, tmp_path):
    """A store read error is not "no record": the launcher orders nothing, keeps every island
    STOP and exits 13 (an order then could start a running island twice)."""
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)

    class Flaky(DirectoryObjectStore):
        def read(self, key):
            if key.endswith("state.json"):
                raise OSError("store timeout")
            return super().read(key)

    objects = Flaky(tmp_path / "store")
    objects.write(lc.stop_key(1), b'{"reason": "by hand"}')
    monkeypatch.setattr(launcher, "_island_state_store", lambda args: lc.IslandStateStore(objects))
    code = launcher.run(_fleet_args(tmp_path))
    assert code == launcher.ISLAND_RECORDS_UNKNOWN_EXIT
    assert [r for r in record if r[0] in ("spawn", "stop_app", "down", "terminate")] == []
    assert DirectoryObjectStore(tmp_path / "store").read(lc.stop_key(1)) is not None  # not cleared
    from yeto.launch.pool.store_lock import StoreLock

    assert StoreLock.acquire(objects, launcher.ISLAND_RECORDS_LOCK, ttl_s=10) is not None  # released


def test_the_lease_is_renewed_from_the_claim_on_while_nothing_is_written(tmp_path):
    """A syncer launch or a Modal deploy writes no record for minutes: the lease must not run
    out meanwhile (a second launcher would then order the same islands)."""
    import threading
    import time as _time

    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    t = [1000.0]
    states = lc.IslandStateStore(objects, background=True, stop_poll_s=0.02, log=lambda m: None)
    lock, why = launcher._claim_island_records(states, wait_s=0, clock=lambda: t[0])
    assert lock is not None
    states.keep_lease()
    for _ in range(4):  # 4 x 70 s idle (renewed once half the lease is gone): over twice the 120 s lease
        t[0] += 70.0
        deadline = _time.monotonic() + 5
        while lock.renewed_at != t[0] and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert lock.renewed_at == t[0]
    rival = lc.IslandStateStore(objects, log=lambda m: None)
    other, refusal = launcher._claim_island_records(rival, wait_s=0, clock=lambda: t[0])
    assert other is None and "another launcher holds" in refusal
    states.close(2)
    lock.release()
    t[0] += 70.0
    assert states._fence_holds() is False and lock.renewed_at == t[0] - 70.0  # closed: no renewal after the release
    with pytest.raises(lc.IslandRecordsFenced):
        states.guard_order("late")
    assert threading.active_count() >= 1
    assert StoreLock.acquire(objects, launcher.ISLAND_RECORDS_LOCK, ttl_s=10, clock=lambda: t[0]) is not None


def test_a_launcher_that_lost_the_lock_orders_nothing_and_tears_nothing_down(monkeypatch, tmp_path):
    """The lock is lost between the claim and the orders: no order, no cancel, no app stop,
    exit 12 (the launcher holding the records supervises the fleet)."""
    from yeto.launch.pool.store_lock import LockLost

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    _store(monkeypatch, tmp_path)
    real = launcher._claim_island_records

    def claim_then_lose(states, **kw):
        lock, why = real(states, **kw)
        lock.lost = LockLost(launcher.ISLAND_RECORDS_LOCK, lock.token, lock.token + 1)
        return lock, why

    monkeypatch.setattr(launcher, "_claim_island_records", claim_then_lose)
    code = launcher.run(_fleet_args(tmp_path))
    assert code == launcher.ISLAND_RECORDS_LOCKED_EXIT
    assert [r for r in record if r[0] in ("spawn", "cancel", "stop_app", "down", "terminate")] == []


def test_a_restart_on_the_same_machine_keeps_the_adopted_islands_tapes(monkeypatch, tmp_path):
    """Only the tapes of islands launched again move to previous/: an adopted island's tape is
    continued where it is."""
    from yeto.launch.manifest import _no_sync_events_dir

    seen = {}

    def tapes(args, names):
        events = seen["events"] = _no_sync_events_dir(args)
        events.mkdir(parents=True, exist_ok=True)
        for name in names:
            (events / f"{name}.jsonl").write_text('{"event": "rl_round_trained", "round": 0}\n')

    seeded = {"fc-old-0": 0, "fc-old-1": 1}
    code, record, states, names = _run(monkeypatch, tmp_path, seeded=seeded, alive_calls={"fc-old-0"}, before=tapes)
    events = seen["events"]
    kept = events / f"{names[0]}.jsonl"
    assert kept.exists() and '"round": 0' in kept.read_text().splitlines()[0]  # adopted: in place
    moved = list((events / "previous").glob(f"*/{names[1]}.jsonl"))
    assert len(moved) == 1  # launched again: its earlier tape moved aside
    assert not list((events / "previous").glob(f"*/{names[0]}.jsonl"))


# -- REV #750 round 3: every way out of a fenced launcher is hands off ---------------------------

def _fenced_controller_run(monkeypatch, outcome):
    """FleetController.run that loses the records lock mid-run, then ends with ``outcome``."""
    from yeto.launch.fleet import FleetController
    from yeto.launch.pool.store_lock import LockLost

    def run(self):
        store = self.recorder.store
        store.fenced = LockLost(launcher.ISLAND_RECORDS_LOCK, store.fence.token, store.fence.token + 1)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(FleetController, "run", run)


def _fenced_outcomes():
    from yeto.launch.fleet import FixedRosterIslandAbandoned, RunStalled

    return [SystemExit(143), RuntimeError("all learners abandoned; nothing left to train"),
            FixedRosterIslandAbandoned("x"), RunStalled("silent", silent_s=10, timeout_s=5),
            {"yeto-l0-modal": "SUCCEEDED", "yeto-l1-modal": "SUCCEEDED"}]


@pytest.mark.parametrize("outcome", _fenced_outcomes(), ids=["sigterm", "all_abandoned", "fixed_roster",
                                                             "stalled", "normal_end"])
def test_a_launcher_fenced_mid_run_tears_nothing_down_on_any_way_out(monkeypatch, tmp_path, outcome):
    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    _store(monkeypatch, tmp_path)
    _fenced_controller_run(monkeypatch, outcome)
    args = _fleet_args(tmp_path)
    code = launcher.run(args)
    assert code == launcher.ISLAND_RECORDS_LOCKED_EXIT
    assert [r for r in record if r[0] in ("cancel", "stop_app", "down", "terminate")] == []
    assert args._hands_off_exit == launcher.ISLAND_RECORDS_LOCKED_EXIT


def test_the_worker_runs_no_backstop_cleanup_after_a_fenced_launcher(monkeypatch, tmp_path):
    import yeto.cli as cli
    import yeto.cost.cleanup as cleanup_mod
    from yeto import runs

    monkeypatch.setenv("YETO_RUNS_DIR", str(tmp_path / "runs"))
    runs.create_run("w", {"cluster_prefix": "w"})

    def fenced_then_raise(args, **kw):
        args._hands_off_exit = launcher.ISLAND_RECORDS_LOCKED_EXIT
        raise RuntimeError("after the fence")

    monkeypatch.setattr(launcher, "run", fenced_then_raise)
    seen = []
    monkeypatch.setattr(cleanup_mod, "cleanup_run", lambda run, *a, **k: seen.append(run) or True)
    monkeypatch.setattr(cli, "_backstop_cleanup", lambda name, args: seen.append(name))
    assert cli.cmd_worker("w") == launcher.ISLAND_RECORDS_LOCKED_EXIT
    assert seen == []


def test_an_unconfirmed_lock_at_an_order_orders_and_tears_down_nothing(monkeypatch, tmp_path):
    from yeto.launch.pool.store_lock import LockUnconfirmed

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    _store(monkeypatch, tmp_path)
    real = launcher._claim_island_records

    def claim_then_unconfirmed(states, **kw):
        lock, why = real(states, **kw)

        def guard():
            raise LockUnconfirmed("store does not answer")

        lock.guard = guard
        return lock, why

    monkeypatch.setattr(launcher, "_claim_island_records", claim_then_unconfirmed)
    args = _fleet_args(tmp_path)
    code = launcher.run(args)
    assert code == launcher.ISLAND_RECORDS_UNKNOWN_EXIT
    assert [r for r in record if r[0] in ("spawn", "cancel", "stop_app", "down", "terminate")] == []


def test_the_lease_is_renewed_while_a_write_backs_off(tmp_path):
    """A store that refuses writes backs them off (up to RETRY_MAX_S); the fence is still
    guarded every stop_poll_s meanwhile, so the lease never lapses."""
    import time as _time

    class Refusing(DirectoryObjectStore):
        def write(self, key, data):
            raise OSError("store refuses")

    class Fence:
        token, calls = 1, 0

        def guard(self):
            Fence.calls += 1

    states = lc.IslandStateStore(Refusing(tmp_path / "s"), background=True, stop_poll_s=0.02,
                                 fence=Fence(), log=lambda m: None)
    states.write(0, "running", name="r-l0", source="modal", incarnation=0)
    _time.sleep(1.0)
    states.close(0.5)
    assert Fence.calls >= 20  # every pass, not only at the backed-off retries (about 6 in 1 s)
    assert lc.RETRY_MAX_S < launcher.ISLAND_RECORDS_LOCK_TTL_S


def test_a_dead_background_thread_is_started_again(tmp_path):
    import time as _time

    class Fence:
        token = 1

        def guard(self):
            pass

    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "s"), background=True, stop_poll_s=0.02,
                                 fence=Fence(), log=lambda m: None)
    states.keep_lease()
    first = states._thread
    states._closed = True  # the thread ends at its next pass, as a crashed one would
    states._wake.set()
    first.join(2)
    assert not first.is_alive()
    states._closed = False
    states.guard_order("island 0")
    assert states._thread is not first and states._thread.is_alive()
    states.close(1)
    _time.sleep(0)


def test_down_checks_the_lock_now_not_only_the_cached_view():
    """A lock lost after the last background guard: the abandon teardown asks the lock first."""
    from test_controller import FAILED, RUNNING, SYNCER, FakeOps, ImmediateThread

    from yeto.launch.fleet import ABANDONED, FleetController

    class Lost(BaseException):
        def __init__(self):
            super().__init__("lock token 1 is fenced by token 2")
            self.token, self.by = 1, 2

    class Fence:
        token = 1

        def guard(self):
            raise Lost()

    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [FAILED]
    ops.status_seq["r-l1-modal"] = [RUNNING] * 50
    ops.status_seq[SYNCER] = [RUNNING] * 50
    store = lc.IslandStateStore(DirectoryObjectStore("/nonexistent-t1"), log=lambda m: None, fence=Fence())
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    ctl = FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=0, thread_cls=ImmediateThread,
                          elastic=True, recorder=lc.FleetRecorder(store, refs))
    rec = ctl.learners["r-l0-modal"]
    rec.update(state="recovering", failed_at=ops.now() - 10, attempt=None)
    assert store.fenced is None  # not seen yet
    ctl._drive_recovery(rec, is_syncer=False)
    assert rec["state"] == ABANDONED and ops.down_calls == [] and store.fenced is not None


def test_every_order_site_of_the_launcher_checks_the_lock_first():
    """sky, Modal, verda and two-cluster islands go through _record_ordering (the lock checked
    now): the launcher's two-cluster site, and the source adapters' ``on_order`` hook
    (yeto.launch.trainer_provision) right before each order; the resident ssh start and the
    syncer launch guard explicitly."""
    import inspect

    from yeto.launch import trainer_provision as tp

    src = inspect.getsource(launcher.run)
    assert src.count("_record_ordering(recorder, name, adoption)") == 2
    assert "on_order=lambda name: _record_ordering(recorder, name, adoption)" in src
    site = "rids[name] = (m, (\"mixed\", mixed[name].submit()))"
    assert "_record_ordering(recorder, name, adoption)" in src[:src.index(site)].rstrip().splitlines()[-1]
    for adapter, site in ((tp.SkySource.order, "return submit_island_launch("),
                          (tp.VerdaSource.order, "return island.spec"),
                          (tp.ModalSource.spawn, "call_id = island_ops.relaunch(cfg, name)")):
        code = inspect.getsource(adapter)
        assert "prov.on_order(" in code[:code.index(site)].rstrip().splitlines()[-1], site
    ssh = inspect.getsource(tp.SshSource.start)
    assert ssh.index("prov.guard_order(") < ssh.index("start_resident_ssh(")
    assert "guard_order=None if island_states is None else island_states.guard_order" in src
    sky_launch = src.index("rid = sky.launch(syncer_task, cluster_name=syncer_cluster)")
    assert "guard_order(syncer_cluster)" in src[sky_launch - 300:sky_launch]


def test_record_ordering_writes_nothing_once_the_lock_is_lost(tmp_path):
    class Lost(BaseException):
        def __init__(self):
            super().__init__("lost")
            self.token, self.by = 1, 2

    class Fence:
        token = 1

        def guard(self):
            raise Lost()

    store = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "s"), log=lambda m: None, fence=Fence())
    recorder = lc.FleetRecorder(store, {"r-l0-modal": lc.IslandRef(0, "modal")})
    with pytest.raises(lc.IslandRecordsFenced):
        launcher._record_ordering(recorder, "r-l0-modal", {})
    assert store.read(0) is None


def test_a_sky_record_ordered_without_its_job(monkeypatch):
    """The launcher died between the order and the job id: an UP cluster tells by its job
    table (adopted with the live job, never a second job), a provisioning one cannot tell."""
    def probe(status, live):
        return island_adopt.SkyProbe("r-", status=lambda: [{"name": "r-l0-aws", "status": status}],
                                     job_alive=lambda c, j: True, live_job=lambda c: live)

    records = _records()
    records[0].update(state="ordering", handle={"cluster": "r-l0-aws"})
    modal = lambda h: (True, "")  # noqa: E731
    plan = island_adopt.plan(records, REFS, probes={"sky": probe("UP", 7), "modal": modal})
    assert plan["r-l0-aws"].adopt and plan["r-l0-aws"].job_id == 7
    plan = island_adopt.plan(records, REFS, probes={"sky": probe("UP", None), "modal": modal})
    assert not plan["r-l0-aws"].adopt and plan["r-l0-aws"].incarnation == 3
    with pytest.raises(island_adopt.AdoptionUnknown, match="INIT"):
        island_adopt.plan(records, REFS, probes={"sky": probe("INIT", None), "modal": modal})


def test_a_running_two_cluster_island_is_not_adopted(monkeypatch, tmp_path):
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "s"), log=lambda m: None)
    adoption = {"r-l0-aws": island_adopt.Adoption("r-l0-aws", 0, True, 2)}
    monkeypatch.setattr(island_adopt, "plan", lambda *a, **k: adoption)
    monkeypatch.setattr(launcher, "_adopt_probes", lambda args: {})
    monkeypatch.setattr(launcher, "_island_refs", lambda *a, **k: dict(REFS))
    states.write(0, "running", name="r-l0-aws", source="sky", incarnation=2)
    with pytest.raises(island_adopt.AdoptionUnknown, match="two-cluster"):
        launcher._plan_island_records(types.SimpleNamespace(), [types.SimpleNamespace(cloud="aws")], ["r-l0-aws"],
                                      states, object(),
                                      relaunch=True, ssh_names=set(), events_dir=None, echo_names=set(),
                                      held_name=None, syncer_cluster=None, two_cluster_names={"r-l0-aws"})


# -- REV #750 round 4: hands off before supervision; per-island records locks --------------------

def test_each_island_launch_holds_its_own_records_lock(tmp_path):
    """Island launches of one fleet (``--rl-fleet-role island``) share the run store: each claims
    the lock of its own record, so none waits for or fences another; a launch without a fleet
    role keeps the run's lock, and the two kinds exclude each other."""
    objects = DirectoryObjectStore(tmp_path / "store")
    island = [types.SimpleNamespace(rl_fleet_role="island", rl_fleet_learner_id=k, rl_fleet_learners=2)
              for k in (0, 1)]
    prefixes = [launcher._island_records_lock(a) for a in island]
    assert prefixes == ["fleet/trainers/0/lock", "fleet/trainers/1/lock"]
    assert launcher._island_records_lock(types.SimpleNamespace()) == launcher.ISLAND_RECORDS_LOCK
    assert launcher._island_records_lock(types.SimpleNamespace(rl_fleet_role="syncer",
                                                               rl_fleet_learner_id=0)) == launcher.ISLAND_RECORDS_LOCK
    store = lambda: lc.IslandStateStore(objects, log=lambda m: None)  # noqa: E731
    locks = []
    for prefix in prefixes:
        lock, why = launcher._claim_island_records(store(), prefix=prefix, wait_s=0,
                                                   others=[launcher.ISLAND_RECORDS_LOCK])
        assert lock is not None and why is None
        locks.append(lock)
    again, why = launcher._claim_island_records(store(), prefix=prefixes[1], wait_s=0)
    assert again is None and "another launcher" in why  # a second launcher of island 1
    whole, why = launcher._claim_island_records(store(), wait_s=0, others=prefixes)
    assert whole is None and "exclude each other" in why  # a whole-fleet launch of the same run
    from yeto.launch.pool.store_lock import StoreLock

    assert StoreLock.holder_of(objects, launcher.ISLAND_RECORDS_LOCK) is None  # it gave its claim back


def test_a_store_error_while_claiming_is_hands_off_in_the_worker(monkeypatch, tmp_path):
    """Before the launcher supervises the fleet (the claim, the plan, the cost guard), any way
    out is hands off: the worker returns 13 and runs no backstop cleanup (the ledger may list
    the islands this launcher was about to adopt). A SIGTERM while waiting for the lock too."""
    import yeto.cli as cli
    import yeto.cost.cleanup as cleanup_mod
    from yeto import runs

    for raised, name in ((OSError("store list timed out"), "w1"), (SystemExit(143), "w2")):
        record, clock = [], Clock()
        _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
        _store(monkeypatch, tmp_path)

        def claim(states, raised=raised, **kw):
            raise raised

        monkeypatch.setattr(launcher, "_claim_island_records", claim)
        runs.create_run(name, vars(_fleet_args(tmp_path)))
        seen = []
        monkeypatch.setattr(cleanup_mod, "cleanup_run", lambda run, *a, **k: seen.append(run) or True)
        monkeypatch.setattr(cli, "_backstop_cleanup", lambda n, args: seen.append(n))
        assert cli.cmd_worker(name) == launcher.ISLAND_RECORDS_UNKNOWN_EXIT
        assert seen == [] and runs.load_run(name)["exit_code"] == launcher.ISLAND_RECORDS_UNKNOWN_EXIT
        assert [r for r in record if r[0] in ("spawn", "cancel", "stop_app", "down", "terminate")] == []


def test_the_worker_returns_the_hands_off_exit_after_a_fenced_launcher_raised(monkeypatch, tmp_path):
    """run up reads the worker's exit: 12, not 1, so its trainer down runs no `yeto down`."""
    import yeto.cli as cli
    from yeto import runs

    monkeypatch.setenv("YETO_RUNS_DIR", str(tmp_path / "runs"))
    runs.create_run("w", {"cluster_prefix": "w"})

    def fenced_then_raise(args, **kw):
        args._hands_off_exit = launcher.ISLAND_RECORDS_LOCKED_EXIT
        raise RuntimeError("after the fence")

    monkeypatch.setattr(launcher, "run", fenced_then_raise)
    monkeypatch.setattr(cli, "_backstop_cleanup", lambda n, args: pytest.fail("backstop cleanup ran"))
    assert cli.cmd_worker("w") == launcher.ISLAND_RECORDS_LOCKED_EXIT
    meta = runs.load_run("w")
    assert meta["exit_code"] == launcher.ISLAND_RECORDS_LOCKED_EXIT and meta["refused_launch"] is True


def test_an_unconfirmed_lock_whose_lease_ran_out_orders_and_tears_down_nothing_until_confirmed(tmp_path):
    """Not confirmed with this holder's own lease over: another machine may hold the records
    now, so nothing is ordered or torn down. Not sticky: once the store answers and no other
    launcher claimed the records, the next check confirms and renews (real StoreLock)."""
    from yeto.launch.pool.store_lock import StoreLock

    class Flaky(DirectoryObjectStore):
        down = False

        def _check(self):
            if Flaky.down:
                raise OSError("store does not answer")

        def read(self, key):
            self._check()
            return super().read(key)

        def list(self, prefix, **kw):
            self._check()
            return super().list(prefix, **kw)

        def write(self, key, data):
            self._check()
            return super().write(key, data)

    objects = Flaky(tmp_path / "s")
    t = [1000.0]
    states = lc.IslandStateStore(objects, log=lambda m: None)
    lock, why = launcher._claim_island_records(states, wait_s=0, clock=lambda: t[0])
    assert lock is not None
    Flaky.down = True
    t[0] += 60.0  # half the lease left: an outage inside it does not stop anything
    states.guard_order("island 0")
    t[0] += 52.0  # 8 s of lease left, within the margin: orders wait, a teardown goes on (still ours)
    with pytest.raises(lc.IslandRecordsUnconfirmed) as inside:
        states.guard_order("island 0")
    assert not inside.value.lease_over and launcher._lost_island_records(states) is False
    t[0] += 60.0  # past the lease
    with pytest.raises(lc.IslandRecordsUnconfirmed) as over:
        states.guard_order("island 0")
    assert over.value.lease_over and states.fenced is None
    assert launcher._lost_island_records(states) is True  # no teardown while it cannot tell
    Flaky.down = False  # the store answers again; nobody claimed the records meanwhile
    states.guard_order("island 0")  # confirmed (and renewed): orders go on
    assert states.fenced is None and launcher._lost_island_records(states) is False
    assert lock.renewed_at == t[0]
    rival, refusal = launcher._claim_island_records(lc.IslandStateStore(objects, log=lambda m: None), wait_s=0,
                                                    clock=lambda: t[0])
    assert rival is None and "another launcher holds" in refusal
    assert StoreLock.holder_of(objects, launcher.ISLAND_RECORDS_LOCK, now=t[0])["token"] == lock.token


def test_down_skips_a_teardown_while_the_lease_ran_out_unconfirmed():
    from test_controller import FakeOps

    from yeto.launch.fleet import FleetController

    class Unconfirmed(BaseException):
        pass

    class Fence:
        token = 1

        def guard(self):
            raise Unconfirmed("store does not answer")

        def lease_left(self):
            return 0.0

    ops = FakeOps()
    store = lc.IslandStateStore(DirectoryObjectStore("/nonexistent-t1"), log=lambda m: None, fence=Fence())
    ctl = FleetController(learners={"r-l0-modal": ("t", 1)}, syncer=None, sky_ops=ops, poll_interval=30,
                          recover_timeout=600, recorder=lc.FleetRecorder(store, {"r-l0-modal": lc.IslandRef(0, "modal")}))
    ctl._down("r-l0-modal")
    assert ops.down_calls == [] and "r-l0-modal" not in ctl.downed_clusters  # the launcher's end tears it down
    Fence.lease_left = lambda self: 30.0  # within the lease: still ours
    ctl._down("r-l0-modal")
    assert ops.down_calls == ["r-l0-modal"]


# -- #750 follow-up: a teardown skipped past the lease is tried again; the end exit says why --------

def test_a_teardown_skipped_past_the_lease_is_done_once_the_lock_is_confirmed():
    """An abandoned island whose teardown was skipped (the lock not confirmed past this
    holder's lease) is torn down at the next poll that confirms the lock, not only when the
    launcher ends (it would bill meanwhile)."""
    from test_controller import FakeOps

    from yeto.launch.fleet import FleetController

    class Unconfirmed(BaseException):
        pass

    class Fence:
        token, down = 1, True

        def guard(self):
            if Fence.down:
                raise Unconfirmed("store does not answer")

        def lease_left(self):
            return 0.0

    ops = FakeOps()
    store = lc.IslandStateStore(DirectoryObjectStore("/nonexistent-t1"), log=lambda m: None, fence=Fence())
    ctl = FleetController(learners={"r-l0-modal": ("t", 1)}, syncer=None, sky_ops=ops, poll_interval=30,
                          recover_timeout=600, recorder=lc.FleetRecorder(store, {"r-l0-modal": lc.IslandRef(0, "modal")}))
    ctl._down("r-l0-modal")
    assert ops.down_calls == [] and ctl.pending_downs == {"r-l0-modal": False}
    ctl._retry_pending_downs()  # still not confirmed: kept pending
    assert ops.down_calls == [] and ctl.pending_downs == {"r-l0-modal": False}
    Fence.down = False
    ctl._retry_pending_downs()
    assert ops.down_calls == ["r-l0-modal"] and ctl.pending_downs == {}


def test_the_end_says_13_when_the_lock_is_not_confirmed_past_the_lease(monkeypatch, tmp_path):
    """At the launcher's end a lock not confirmed past this holder's lease is hands off with
    13 (not 12: nobody is known to hold the records)."""
    from yeto.launch.fleet import FleetController
    from yeto.launch.pool.store_lock import LockUnconfirmed

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    _store(monkeypatch, tmp_path)

    def run(self):
        fence = self.recorder.store.fence

        def guard():
            raise LockUnconfirmed("store does not answer")

        fence.guard = guard
        fence.lease_left = lambda: 0.0
        return {"yeto-l0-modal": "SUCCEEDED", "yeto-l1-modal": "SUCCEEDED"}

    monkeypatch.setattr(FleetController, "run", run)
    args = _fleet_args(tmp_path)
    assert launcher.run(args) == launcher.ISLAND_RECORDS_UNKNOWN_EXIT
    assert [r for r in record if r[0] in ("cancel", "stop_app", "down", "terminate")] == []


def test_pending_teardowns_ask_the_lock_once_per_poll_log_once_and_keep_force(capsys):
    """Three pending teardowns over ten unconfirmed polls: one lock check per poll (a hung
    store is asked once, not once per island), one line per island; a forced teardown (a
    relaunched cluster of the same name) is still forced when it is done."""
    from test_controller import FakeOps

    from yeto.launch.fleet import FleetController

    class Unconfirmed(BaseException):
        pass

    class Fence:
        token, down, guards = 1, True, 0

        def guard(self):
            Fence.guards += 1
            if Fence.down:
                raise Unconfirmed("store does not answer")

        def lease_left(self):
            return 0.0

    ops = FakeOps()
    store = lc.IslandStateStore(DirectoryObjectStore("/nonexistent-t1"), log=lambda m: None, fence=Fence())
    names = ["r-l0-modal", "r-l1-modal", "r-l2-modal"]
    ctl = FleetController(learners={n: ("t", k) for k, n in enumerate(names)}, syncer=None, sky_ops=ops,
                          poll_interval=30, recover_timeout=600,
                          recorder=lc.FleetRecorder(store, {n: lc.IslandRef(k, "modal") for k, n in enumerate(names)}))
    ctl.downed_clusters.add("r-l2-modal")  # torn down once; its relaunch needs a forced teardown
    ctl._down("r-l0-modal")
    ctl._down("r-l1-modal")
    ctl._down("r-l2-modal", force=True)
    capsys.readouterr()
    Fence.guards = 0
    for _ in range(10):
        ctl._retry_pending_downs()
    assert Fence.guards == 10 and ops.down_calls == []
    assert "NOT tearing down" not in capsys.readouterr().err  # logged once each, when first skipped
    Fence.down = False
    ctl._retry_pending_downs()
    assert sorted(ops.down_calls) == names and ctl.pending_downs == {}


def test_the_launchers_end_tears_down_what_the_controller_left_pending(monkeypatch, tmp_path):
    """Teardowns still pending when supervision ends are in the end's teardown list (a renamed
    relaunch target is not always in the launcher's cluster list)."""
    from yeto.launch.fleet import FleetController

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    _store(monkeypatch, tmp_path)

    def run(self):
        self.pending_downs["yeto-l1-modal-r1"] = True  # skipped, then the run ended
        return {"yeto-l0-modal": "SUCCEEDED", "yeto-l1-modal": "SUCCEEDED"}

    monkeypatch.setattr(FleetController, "run", run)
    launcher.run(_fleet_args(tmp_path))
    assert ("terminate", "yeto-l1-modal-r1") in record
