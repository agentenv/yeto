"""``yeto run up`` of a fleet launched island by island: the trainer worker joins the
controller's supervisor with the islands seeded into the desired state, the island
launches share the syncer launch's HMAC key, and the run's stops reach every island
launch (local store, fake launches; no cloud). The syncer launch records its address
and supervises the syncer alone (``yeto.launch.fleet_syncer``)."""

from __future__ import annotations

import json
import stat
import types

import pytest

from test_fleet_island_launches import example_doc
from test_fleet_trainer_worker import FakeLaunches
from yeto.launch import fleet_syncer
from yeto.launch.pool import desired as dz
from yeto.rl import island_launches as il
from yeto.launch import island_lifecycle as lifecycle
from yeto.rl import trainer_worker, up
from yeto.rl.engine.member_table import MEMBER_LANES_ENV, MEMBER_TABLE_ENV, lanes_from, lane_matches
from yeto.rl.engine.object_store import DirectoryObjectStore


@pytest.fixture(autouse=True)
def _island_env(monkeypatch, tmp_path):
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY", raising=False)
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")


class FakeController:
    def __init__(self) -> None:
        self.workers = {}

    def add_worker(self, name, step, *, interval=10.0, on_restart=None):
        self.workers[name] = (step, interval)


def build(tmp_path, monkeypatch, **kw):
    root = tmp_path / "bucket"
    monkeypatch.setattr(up.Run, "_objects", lambda self: DirectoryObjectStore(root))
    path = tmp_path / "run.json"
    path.write_text(json.dumps(example_doc(tmp_path, **kw)))
    run = up.Run(up.load_config(str(path)), state_root=tmp_path / "state")
    run.calls = []
    run._cmd = lambda argv, *, log, timeout=None: run.calls.append(list(argv)) or 0
    return run, DirectoryObjectStore(root)


def test_per_island_needs_the_store_control_plane(tmp_path, monkeypatch):
    run, _ = build(tmp_path, monkeypatch)
    assert run.per_island_fleet() and run.fleet_problem() is None
    doc = example_doc(tmp_path)
    doc["rollout"].pop("control_plane")
    path = tmp_path / "run2.json"
    path.write_text(json.dumps(doc))
    other = up.Run(up.load_config(str(path)), state_root=tmp_path / "state2")
    assert "rollout.control_plane" in other.fleet_problem()


def test_the_trainer_worker_joins_the_supervisor_with_the_islands_seeded(tmp_path, monkeypatch):
    run, objects = build(tmp_path, monkeypatch)
    controller, launches = FakeController(), FakeLaunches()
    worker = run.attach_trainer_worker(controller, objects, ops=launches)
    assert "trainers" in controller.workers
    assert worker.fenced() is None  # the controller's reconciler decides
    controller.reconciler = types.SimpleNamespace(fenced="taken over")
    assert worker.fenced() == "taken over"
    controller.reconciler = None
    doc = dz.DesiredStore(objects).read()
    trainers, pools = dz.trainers(doc), dz.sampler_pools(doc)
    assert sorted(m["learner_id"] for m in trainers.values()) == [0, 1]
    for m in trainers.values():  # each trainer follows its own island's pools only (lane i<k>)
        k = m["learner_id"]
        assert m["argv"][-4:] == ["--island-env", f"{MEMBER_TABLE_ENV}=1", "--island-env", f"{MEMBER_LANES_ENV}=i{k}"]
        lanes = lanes_from({MEMBER_LANES_ENV: f"i{k}"}, ())
        bases = {p["pool"]["shard_base"]: name in m["pools"] for name, p in pools.items()}
        assert m["pools"] and len(bases) > len(m["pools"])
        assert {b for b in bases if any(lane_matches(b, lane) for lane in lanes)} == {b for b, own in bases.items() if own}
    key = run.dir / "island_hmac.key"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600 and run.env["YETO_ISLAND_HMAC_KEY"] == key.read_text().strip()
    run.attach_trainer_worker(FakeController(), objects, ops=launches)  # a restart: seeded once
    assert dz.DesiredStore(objects).read()["generation"] == doc["generation"]
    il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=0, learners=2, max_learners=4, run="r")
    worker.step()
    assert sorted(p for p, _ in launches.starts) == ["r-i0", "r-i1"]
    assert run.island_launch_names() == ["r-i0", "r-i1"]


def test_a_restarted_run_up_keeps_the_island_key(tmp_path, monkeypatch):
    run, objects = build(tmp_path, monkeypatch)
    run.island_hmac_env()
    first = run.env["YETO_ISLAND_HMAC_KEY"]
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY", raising=False)
    again = up.Run(run.cfg, state_root=tmp_path / "state")
    again.island_hmac_env()
    assert again.env["YETO_ISLAND_HMAC_KEY"] == first


def test_trainer_down_downs_every_island_launch(tmp_path, monkeypatch):
    run, objects = build(tmp_path, monkeypatch)
    worker = run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
    il.write_syncer_record(objects, addr="h:1", syncer_epoch=0, learners=2, max_learners=4, run="r")
    worker.step()
    record = worker.load(1)
    worker.save(1, {**record, "incarnation": 1, "prefix_now": "r-i1-m1"})  # moved once
    run.trainer_down("test")
    downs = [c[c.index("down") + 2] for c in run.calls if "down" in c and "--missing-ok" in c]
    assert downs == ["r", "r-i0", "r-i1", "r-i1-m1"]


def test_the_local_island_launches_are_this_fleet_s_from_the_run_registry(tmp_path, monkeypatch):
    from yeto import runs

    run, _ = build(tmp_path, monkeypatch)
    for name in ("r-i0", "r-i1-m2", "r-i7", "r", "r-island", "other-i0", "r-i0-m1x"):
        k = name.split("-i", 1)[-1].split("-")[0]
        runs.create_run(name, {"rl_fleet_role": "island", "rl_fleet_learner_id": int(k) if k.isdigit() else 0})
    runs.create_run("r-i3", {"training_mode": "rl"})  # another run that only shares the name pattern
    runs.create_run("r-i4", {"rl_fleet_role": "island", "rl_fleet_learner_id": 5})  # not island 4
    assert run.local_island_launches() == ["r-i0", "r-i1-m2", "r-i7"]


def test_run_down_records_the_island_launches_it_confirmed_down(tmp_path, monkeypatch):
    """Only a launch registered on this host is confirmed down by its `yeto down`; the next
    run up (any host) starts those islands' next incarnation, and leaves the others."""
    from yeto import runs

    run, objects = build(tmp_path, monkeypatch)
    worker = run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
    worker.host = "host-a"
    il.write_syncer_record(objects, addr="h:1", syncer_epoch=0, learners=2, max_learners=4, run="r")
    worker.step()
    runs.create_run("r-i0", {"rl_fleet_role": "island", "rl_fleet_learner_id": 0})  # island 1's: not registered here
    monkeypatch.setattr(up.Run, "lock_for_down", lambda self: None)
    run.down()
    assert worker.load(0)["state"] == trainer_worker.DOWN and worker.load(1)["state"] == trainer_worker.STARTING
    again = FakeLaunches()
    other = trainer_worker.TrainerWorker(worker.desired, objects, again, _states(objects), log=lambda m: None,
                                         host="host-b", clock=lambda: 10 ** 10)
    other.syncer_since = None
    other.step()
    assert sorted(again.live) == ["r-i0-m1"]


def test_a_single_trainer_run_names_no_island_launch(tmp_path, monkeypatch):
    run, _ = build(tmp_path, monkeypatch, per_island=False)
    assert not run.per_island_fleet() and run.island_launch_names() == []


# -- the syncer launch ------------------------------------------------------------------------

def test_the_syncer_launch_records_its_address(tmp_path):
    import types

    objects = DirectoryObjectStore(tmp_path / "store")
    args = types.SimpleNamespace(rl_syncer_epoch=3, rl_sync_max_learners=4, cluster_prefix="r",
                                 rl_checkpoint_store=str(tmp_path / "store"))
    fleet_syncer.record(args, "10.0.0.1:29400,10.0.0.1:29401", 2, objects=objects)
    record = il.read_syncer_record(objects)
    assert record["addr"] == "10.0.0.1:29400,10.0.0.1:29401" and record["syncer_epoch"] == 3
    assert record["learners"] == 2 and record["max_learners"] == 4


def test_the_syncer_launch_ends_with_the_syncer_or_the_stop_flag(tmp_path):
    from yeto.launch.fleet import RUN_STOPPED_EXIT

    sleeps = []
    states = iter([False, False, True])
    assert fleet_syncer.supervise(finished=lambda: next(states), failed=lambda: None, sleep=sleeps.append,
                                  log=lambda m: None) == 0
    assert len(sleeps) == 2
    flag = tmp_path / "STOP"
    flag.write_text("x")
    assert fleet_syncer.supervise(finished=lambda: False, failed=lambda: None, stop_flag=flag,
                                  sleep=sleeps.append, log=lambda m: None) == RUN_STOPPED_EXIT
    assert fleet_syncer.supervise(finished=lambda: False, failed=lambda: fleet_syncer.Final("job FAILED"),
                                  sleep=sleeps.append, log=lambda m: None) == 1
    restarted = []
    downs = iter(["exit 2", None])
    done = iter([False, False, True])
    assert fleet_syncer.supervise(finished=lambda: next(done), failed=lambda: next(downs),
                                  restart=lambda why: restarted.append(why) or True, sleep=sleeps.append,
                                  log=lambda m: None) == 0
    assert restarted == ["exit 2"]


class FlakySky:
    """sky ops of the syncer cluster: ``job_status`` raises while ``errors`` lasts."""

    def __init__(self, *, errors: int, alive: bool, up: bool = True, end: str | None = None):
        self.errors, self.is_alive, self.up, self.end, self.calls = errors, alive, up, end, 0

    def job_status(self, cluster, job):
        self.calls += 1
        if self.errors > 0:
            self.errors -= 1
            raise ConnectionError("sky API server unreachable")
        return _Status(self.end or "RUNNING")

    def job_alive(self, cluster, job):
        return self.is_alive

    def cluster_up(self, cluster):
        return self.up


class _Status:
    def __init__(self, name):
        self.name = name

    def is_terminal(self):
        return self.name != "RUNNING"

    def __str__(self):
        return f"JobStatus.{self.name}"


def serve_sky(tmp_path, monkeypatch, ops, *, polls):
    """``fleet_syncer.serve`` in cluster mode on fake sky ops and a fake clock (one
    15 s poll per sleep); the run ends after ``polls`` polls (its STOP flag)."""
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(fleet_syncer, "record", lambda *a, **k: {})

    clock = [0.0]
    monkeypatch.setattr(fleet_syncer.time, "monotonic", lambda: clock[0])
    left = [polls]

    def sleep(s):
        clock[0] += s
        left[0] -= 1
        if left[0] == 0:
            runs.request_stop("r")

    args = types.SimpleNamespace(cluster_prefix="r")
    return fleet_syncer.serve(args, "h:1", 2, clusters=[], head_mode=False, syncer_cluster="r-syncer",
                              syncer_job=1, ops=ops, sleep=sleep)


def test_a_transient_sky_error_never_ends_the_fleet_syncer(tmp_path, monkeypatch):
    from yeto.launch.fleet import RUN_STOPPED_EXIT

    # the status read fails, the job table says the job runs: up (no grace spent)
    assert serve_sky(tmp_path, monkeypatch, FlakySky(errors=50, alive=True), polls=40) == RUN_STOPPED_EXIT
    # it fails and the job table cannot tell, for less than the grace: still up
    assert serve_sky(tmp_path / "b", monkeypatch, FlakySky(errors=3, alive=False), polls=40) == RUN_STOPPED_EXIT


def test_a_syncer_cluster_gone_for_the_grace_or_a_failed_job_ends_it(tmp_path, monkeypatch):
    gone = FlakySky(errors=10 ** 6, alive=False)
    assert serve_sky(tmp_path, monkeypatch, gone, polls=10 ** 6) == 1
    assert gone.calls * fleet_syncer.POLL_S >= fleet_syncer.FAILURE_GRACE_S  # not at the first error
    failed = FlakySky(errors=0, alive=False, end="FAILED")
    assert serve_sky(tmp_path / "b", monkeypatch, failed, polls=10 ** 6) == 1 and failed.calls == 1
    done = FlakySky(errors=0, alive=False, end="SUCCEEDED")
    assert serve_sky(tmp_path / "c", monkeypatch, done, polls=10 ** 6) == 0
    down = FlakySky(errors=0, alive=False, up=False)  # preempted: the grace too, then 1
    assert serve_sky(tmp_path / "d", monkeypatch, down, polls=10 ** 6) == 1 and down.calls > 1


def test_the_spend_bills_the_island_launches_added_and_removed(tmp_path, monkeypatch):
    run, objects = build(tmp_path, monkeypatch)
    clock = [1000.0]
    worker = run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())
    worker.clock = lambda: clock[0]
    il.write_syncer_record(objects, addr="h:1", syncer_epoch=0, learners=2, max_learners=4, run="r")
    worker.syncer_since = None
    worker.step()  # islands 0 and 1 start at t=1000
    north = dict(dz.trainers(worker.desired.read())["sglang"], name="north", learner_id=2, prefix="r-i2",
                 usd_per_h=5.0, pools=[])
    worker.desired.update(lambda doc: doc["members"].append(north))
    clock[0] = 1000.0 + 3600
    worker.step()  # north starts an hour later
    worker.desired.update(lambda doc: doc["members"].remove(
        next(m for m in doc["members"] if m["name"] == "sglang")))
    worker.step()  # island 1: STOP
    worker.ops.end("r-i1", 0)
    clock[0] = 1000.0 + 2 * 3600
    worker.step()  # island 1 removed after 2 h
    rows = {k: (round(h, 3), p) for k, h, p, _ in trainer_worker.billed(objects, now=1000.0 + 4 * 3600)}
    assert rows == {0: (4.0, 32.0), 1: (2.0, 16.0), 2: (3.0, 15.0)}
    monkeypatch.setattr(up.Run, "trainer_hours", lambda self, now=None: 4.0)
    monkeypatch.setattr(up.Run, "node_hours", lambda self, i=0, now=None: 0.0)
    spend = run.fleet_rt().spend(now=1000.0 + 4 * 3600)
    assert [round(u, 2) for u in spend.islands_usd] == [32.0, 16.0, 15.0]
    shared = trainer_worker.island_spend(trainer_worker.billed(objects, now=1000.0 + 4 * 3600), {})
    assert [round(shared[k][1], 2) for k in sorted(shared)] == [32.0, 16.0, 15.0]  # the dashboard's function
    # island 0 moves to machines without a price for 1 h: its known $32 stays, only that hour
    # bills at its declared price ($8/h)
    clock[0] = 1000.0 + 4 * 3600
    worker.desired.update(lambda doc: dz.trainers(doc)["mixed"].update(gpu="aws:2xh100@us-east-2", usd_per_h=None))
    worker.step()
    worker.ops.end("r-i0", 0)
    worker.step()
    spend = run.fleet_rt().spend(now=1000.0 + 5 * 3600)
    assert round(spend.islands_usd[0], 2) == 40.0


def test_per_island_notes_the_lease_a_removed_island_stalls_a_round(tmp_path, monkeypatch):
    """The syncer releases a torn-down island's open rounds one lease after it is gone: no
    warning about the quorum timeout any more, a note with the lease instead."""
    logs = []
    monkeypatch.setattr(up, "_log", logs.append)
    run, _ = build(tmp_path, monkeypatch)
    run.log_fleet_plan()
    (note,) = [m for m in logs if "without a LEAVE" in m]
    assert note.startswith("fleet: note:") and "one island lease (150 s, the default" in note
    assert not [m for m in logs if "WARNING" in m and "soft-deadline" in m]
    doc = example_doc(tmp_path)
    doc["fleet"]["island_lease_s"] = 300
    path = tmp_path / "own.json"
    path.write_text(json.dumps(doc))
    assert "(300 s, fleet.island_lease_s)" in up.Run(up.load_config(str(path)),
                                                    state_root=tmp_path / "own").removal_stall_note()
    # the lease set in the trainer argv: the note names where it comes from
    doc = example_doc(tmp_path)
    doc["fleet"]["trainer_argv"] += ["--rl-sync-island-lease-s", "200"]
    path = tmp_path / "argv.json"
    path.write_text(json.dumps(doc))
    note = up.Run(up.load_config(str(path)), state_root=tmp_path / "argv").removal_stall_note()
    assert "(200 s, --rl-sync-island-lease-s in fleet.trainer_argv)" in note
    # a synced group and a single trainer run print nothing more
    (tmp_path / "group").mkdir()
    group, _ = build(tmp_path / "group", monkeypatch, per_island=False)
    assert group.removal_stall_note() is None
    logs.clear()
    group.log_fleet_plan()
    assert not [m for m in logs if "without a LEAVE" in m]


def test_a_moved_island_bills_each_incarnation_at_its_own_price(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    member = dict(trainer_member_fixture(), usd_per_h=10.0)
    store.apply({"members": [member]}, expect=0)
    il.write_syncer_record(objects, addr="h:1", syncer_epoch=0, learners=1, max_learners=2, run="r")
    clock = [0.0]
    launches = FakeLaunches()
    w = trainer_worker.TrainerWorker(store, objects, launches, _states(objects), log=lambda m: None,
                                     clock=lambda: clock[0])
    w.step()  # $10/h from t=0
    clock[0] = 3600.0
    store.update(lambda doc: dz.trainers(doc)["isl0"].update(gpu="aws:2xh100@us-east-2", usd_per_h=100.0))
    w.step()  # moving
    launches.end("run-i0", 0)
    w.step()  # the next incarnation at $100/h from t=3600
    rows = trainer_worker.billed(objects, now=7200.0)
    assert [(k, round(h, 3), round(u, 3), x) for k, h, u, x in rows] == [(0, 2.0, 110.0, 0.0)]
    # an incarnation without --usd-per-h
    clock[0] = 7200.0
    store.update(lambda doc: dz.trainers(doc)["isl0"].update(gpu="modal:1x2xh100", usd_per_h=None))
    w.step()
    launches.end("run-i0-m1", 0)
    w.step()
    clock[0] = 9000.0
    (row,) = trainer_worker.billed(objects, now=9000.0)
    # no --usd-per-h, but the unified price table prices modal 2 x H100: 0.5 h at its row
    table = trainer_worker.table_usd_per_h("modal:1x2xh100")
    assert table is not None and tuple(round(v, 3) for v in row) == (0, 2.5, round(110.0 + 0.5 * table, 3), 0.0)
    # machines the table has no billing row for: their hours are counted apart, the known dollars kept
    assert trainer_worker.table_usd_per_h("aws:2xh100@us-east-2") is None
    store.update(lambda doc: dz.trainers(doc)["isl0"].update(gpu="aws:2xh100@us-east-2"))
    w.step()
    launches.end("run-i0-m2", 0)
    w.step()
    clock[0] = 10800.0
    (row,) = trainer_worker.billed(objects, now=10800.0)
    assert tuple(round(v, 3) for v in row) == (0, 3.0, round(110.0 + 0.5 * table, 3), 0.5)


def trainer_member_fixture() -> dict:
    return {"kind": "trainer", "name": "isl0", "learner_id": 0, "gpu": "modal:1x2xh100", "usd_per_h": 8.0,
            "argv": ["--training-mode", "rl", "--rl-fleet-role", "island", "--rl-fleet-learner-id", "0"],
            "prefix": "run-i0", "pools": ["i0-pool"]}


def _states(objects):
    return lifecycle.IslandStateStore(objects, log=lambda m: None)


def test_the_worker_sees_another_controller_at_its_next_step_not_at_the_next_generation(tmp_path):
    """The reconciler finds itself fenced only when it applies or writes; the trainer
    worker reads fleet/applied.json's writer at every step."""
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": []}, expect=0)
    reconciler = types.SimpleNamespace(fenced=None, _claim=False, writer="host-a:1:x", store=store)
    controller = types.SimpleNamespace(reconciler=reconciler)
    store.write_applied(1, {}, {}, writer="host-a:1:x", claim=True)
    assert up.fleet_fenced(controller) is None
    store.write_applied(1, {}, {}, writer="host-b:2:y", claim=True)  # another controller claimed it
    assert "written by host-b:2:y" in up.fleet_fenced(controller)
    assert reconciler.fenced is None  # the reconciler has not looked yet
    broken = types.SimpleNamespace(reconciler=types.SimpleNamespace(
        fenced=None, _claim=False, writer="w", store=types.SimpleNamespace(read_applied=lambda: 1 / 0)))
    assert up.fleet_fenced(broken).startswith("unknown")
    assert up.fleet_fenced(types.SimpleNamespace(reconciler=None)) is None


def test_the_syncer_grace_is_the_syncer_launch_s_own_setting(tmp_path, monkeypatch):
    from yeto.rl.engine import sample_roles

    gone = FlakySky(errors=10 ** 6, alive=False)
    monkeypatch.setattr(fleet_syncer, "FAILURE_GRACE_S", 10 ** 9)  # the flag decides, not the default

    def serve(args):
        from yeto import runs

        monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
        monkeypatch.setattr(fleet_syncer, "record", lambda *a, **k: {})
        clock = [0.0]
        monkeypatch.setattr(fleet_syncer.time, "monotonic", lambda: clock[0])
        return fleet_syncer.serve(args, "h:1", 2, clusters=[], head_mode=False, syncer_cluster="r-syncer",
                                  syncer_job=1, ops=gone, sleep=lambda s: clock.__setitem__(0, clock[0] + s))

    assert serve(types.SimpleNamespace(cluster_prefix="r", rl_fleet_syncer_grace_s=60.0)) == 1
    assert gone.calls * fleet_syncer.POLL_S >= 60 and gone.calls * fleet_syncer.POLL_S < 120
    with pytest.raises(ValueError, match="belongs to the syncer launch"):
        sample_roles.check_fleet_member(types.SimpleNamespace(rl_fleet_syncer_grace_s=60.0))
    with pytest.raises(ValueError, match="> 0"):
        sample_roles.check_fleet_member(types.SimpleNamespace(rl_fleet_role="syncer", rl_fleet_syncer_grace_s=0))


def test_a_run_budget_refuses_a_per_island_trainer_without_a_price(tmp_path, monkeypatch):
    """With a run budget, a trainer island priced neither by islands[k].trainer.usd_per_h
    nor by the price table is refused at the preflight (dry run too); without a budget it
    passes (the spend warns)."""
    doc = example_doc(tmp_path)
    doc["islands"][1]["trainer"].update(gpu="aws:2xh100@us-east-2", usd_per_h=None)
    path = tmp_path / "unpriced.json"
    path.write_text(json.dumps(doc))
    run = up.Run(up.load_config(str(path)), state_root=tmp_path / "u")
    assert run.breaker_param("run.budget", "usd") is not None
    problem = run.budget_problem()
    assert problem and "has no price" in problem and "islands[1].trainer.usd_per_h" in problem
    doc["islands"][1]["trainer"]["gpu"] = "modal:1x2xh100"  # the table prices it
    path.write_text(json.dumps(doc))
    assert up.Run(up.load_config(str(path)), state_root=tmp_path / "t").island_price_problem() is None
    doc["islands"][1]["trainer"]["gpu"] = "aws:2xh100@us-east-2"
    assert "breakers" not in doc  # the example's budget is its budget: section only
    del doc["budget"]["usd"]  # no run budget: the unpriced island passes (the spend warns)
    path.write_text(json.dumps(doc))
    free = up.Run(up.load_config(str(path)), state_root=tmp_path / "f")
    assert free.breaker_param("run.budget", "usd") is None
    assert free.island_price_problem() is not None and free.budget_problem() is None


def test_a_run_budget_refuses_an_unpriced_island_of_the_desired_state(tmp_path, monkeypatch):
    """An island added (or moved) at run time without a price while the run had no budget
    is a trainer member of the fleet's desired state only: once the run config has a
    budget, the preflight's read of the desired state refuses it as well."""
    run, objects = build(tmp_path, monkeypatch)
    run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())  # seeds the run config's islands
    store = dz.DesiredStore(objects)
    late = dict(dz.trainers(store.read())["sglang"], name="late", learner_id=2,
                gpu="aws:2xh100@us-east-2", usd_per_h=None, prefix="r-i2", pools=[])
    store.update(lambda doc: doc["members"].append(late))
    assert run.breaker_param("run.budget", "usd") is not None
    assert run.island_price_problem() is None  # the run config's islands are priced; nothing read yet
    run.adopt_desired()  # what the preflight reads first
    problem = run.budget_problem()
    assert problem and "trainer island late of the fleet's desired state" in problem
    assert "--name late --gpu aws:2xh100@us-east-2 --usd-per-h" in problem
    store.update(lambda doc: dz.trainers(doc)["late"].update(usd_per_h=6.0))
    run.adopt_desired()
    assert run.island_price_problem() is None


def test_the_preflight_refuses_a_desired_state_that_does_not_parse_and_warns_when_unread(tmp_path, monkeypatch):
    """Read but broken (the store answers, the newest generation is not a desired state):
    refused, nothing it drives could be checked. Not read at all (the store does not
    answer): a WARNING naming what goes unchecked, and the run config is checked."""
    logs = []
    monkeypatch.setattr(up, "_log", logs.append)
    run, objects = build(tmp_path, monkeypatch)
    run.attach_trainer_worker(FakeController(), objects, ops=FakeLaunches())  # a desired state, generation 1
    store = dz.DesiredStore(objects)
    objects.write(store.key(store.generation() + 1), b"{not json")
    assert run.preflight() is False
    assert any("does not parse" in m and "yeto fleet apply" in m for m in logs)
    logs.clear()

    def unreachable(*a, **k):
        raise OSError("the run store does not answer")

    monkeypatch.setattr(up.Run, "adopt_desired", unreachable)
    run.preflight()
    assert any(m.startswith("preflight: WARNING: the fleet's desired state was not read") for m in logs)
    assert not any("does not parse" in m for m in logs)
