"""The trainer worker (``yeto.rl.trainer_worker``): the trainer islands of the fleet's
desired state, one ``yeto launch`` each, reconciled under the rollout controller's
supervisor (local store, fake launches; no cloud).

* 2.3: the trainer members live in ``fleet/desired.json``; a worker whose thread dies
  is restarted by the supervisor and adopts the running launches (no second launch);
  so does a new worker (``yeto run up`` restarted).
* 5.3: one launch per island; an island whose launch ends with an infrastructure
  failure is relaunched alone: the other islands' launches are never stopped or started
  again. A configuration error is never relaunched.
"""

from __future__ import annotations

import threading
import time

import pytest

from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.pool import controller as rc
from yeto.launch.pool import desired as dz
from yeto.rl import island_launches as il
from yeto.rl import trainer_worker as tw
from yeto.rl.engine.object_store import DirectoryObjectStore


class FakeLaunches:
    """``yeto launch`` processes by prefix: started, alive, ended with an exit code."""

    def __init__(self) -> None:
        self.live: dict[str, list[str]] = {}
        self.exits: dict[str, dict] = {}
        self.starts: list[tuple[str, list[str]]] = []
        self.stops: list[str] = []
        self.lock = threading.Lock()

    def alive(self, prefix: str) -> bool:
        return prefix in self.live

    def start(self, prefix: str, argv: list[str]) -> int:
        with self.lock:
            self.starts.append((prefix, list(argv)))
            self.live[prefix] = list(argv)
            self.exits.pop(prefix, None)
            return 1000 + len(self.starts)

    def ended(self, prefix: str):
        return self.exits.get(prefix)

    def request_stop(self, prefix: str, why: str) -> None:
        self.stops.append(prefix)

    # test controls
    def end(self, prefix: str, code: int, log: str = "") -> None:
        with self.lock:
            self.live.pop(prefix, None)
            self.exits[prefix] = {"exit_code": code, "log": log, "detail": f"exit {code}"}

    def started(self, prefix: str) -> int:
        return sum(1 for p, _ in self.starts if p == prefix)


def member(k: int, gpu: str = "modal:1x2xh100", name: str | None = None) -> dict:
    return {"kind": "trainer", "name": name or f"isl{k}", "learner_id": k, "gpu": gpu, "usd_per_h": 8.0,
            "argv": ["--training-mode", "rl", "--rl-fleet-role", "island", "--rl-fleet-learner-id", str(k)],
            "prefix": f"run-i{k}", "pools": [f"i{k}-pool"]}


def fleet(tmp_path, members, *, syncer: bool = True):
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": members}, expect=0)
    if syncer:
        il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=2, learners=2, max_learners=4, run="run")
    return objects, store


def worker(objects, store, launches, **kw) -> tw.TrainerWorker:
    return tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                            log=lambda m: None, **kw)


def test_trainer_members_are_kinds_of_the_one_desired_state(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    doc = store.read()
    assert sorted(dz.trainers(doc)) == ["isl0", "isl1"]
    assert objects.read(dz.DESIRED_KEY) is not None  # the same document the pool reconcile reads


def test_islands_start_once_the_syncer_is_recorded(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)], syncer=False)
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    assert launches.starts == []
    il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=2, learners=2, max_learners=4, run="run")
    w.step()
    assert [p for p, _ in launches.starts] == ["run-i0", "run-i1"]
    argv = launches.live["run-i1"]
    assert argv[argv.index("--rl-fleet-syncer") + 1] == "10.0.0.9:29400"
    assert argv[argv.index("--rl-syncer-epoch") + 1] == "2"
    assert argv[argv.index("--gpu") + 1] == "modal:1x2xh100"
    assert argv[argv.index("--cluster-prefix") + 1] == "run-i1"
    w.step()
    assert len(launches.starts) == 2  # running: nothing again
    assert w.load(1)["state"] == tw.RUNNING


def test_an_earlier_runs_syncer_record_is_not_dialed(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    worker(objects, store, launches, syncer_since=time.time() + 3600).step()
    assert launches.starts == []


def test_one_island_relaunch_never_restarts_the_others(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1), member(2)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    w.step()
    launches.end("run-i1", 1, log="sky: cluster lost")
    w.step()
    assert launches.started("run-i1") == 2
    assert launches.started("run-i0") == launches.started("run-i2") == 1
    assert launches.stops == []
    record = w.load(1)
    assert record["relaunches"] == 1 and record["prefix_now"] == "run-i1" and record["incarnation"] == 0
    assert record["last_end"]["exit_code"] == 1 and record["ended"] is None
    for k in (0, 2):
        assert w.load(k)["relaunches"] == 0


def test_a_config_error_is_never_relaunched(tmp_path):
    from yeto.config_errors import CONFIG_ERROR_EXIT

    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    launches.end("run-i1", CONFIG_ERROR_EXIT, log="FullParameterLayoutMismatch")
    w.step()
    w.step()
    assert launches.started("run-i1") == 1
    assert w.load(1)["state"] == lifecycle.FAILED
    assert w.load(1)["decision"]["cause"] == lifecycle.CONFIG_ERROR
    assert launches.alive("run-i0")


def test_an_island_launch_that_failed_for_good_is_never_started_again(tmp_path):
    """The island launch ended with STRICT_FAILURE_EXIT (its island failed on a
    deterministic error, e.g. the syncer refused its profile), its log tail holding only
    teardown lines: failed, never started again (no new machines ordered)."""
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    launches.end("run-i1", lifecycle.STRICT_FAILURE_EXIT, log="(RayWorkerManager) teardown line")
    for _ in range(3):
        w.step()
    assert launches.started("run-i1") == 1
    assert w.load(1)["state"] == lifecycle.FAILED
    assert w.load(1)["decision"]["cause"] == lifecycle.STRICT
    assert launches.alive("run-i0")


def test_a_fleet_island_launch_ends_with_the_strict_failure_exit_when_its_islands_failed_for_good():
    """launcher.run: a fleet's island launch maps IslandsFailedForGood to the exit the
    trainer worker never relaunches; any other launch keeps the error."""
    import inspect

    from yeto import launcher

    src = inspect.getsource(launcher.run)
    branch = src[src.index("except IslandsFailedForGood as error:"):]
    branch = branch[:branch.index("except FixedRosterIslandAbandoned")]
    assert "if member_role != FLEET_ROLE_ISLAND:\n                raise" in branch
    assert "return CONTAINER_CHANGED_EXIT if container_changed() else STRICT_FAILURE_EXIT" in branch
    assert "strict_failure_probe=" in src


def test_the_relaunch_budget_ends_an_island_abandoned(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    w = worker(objects, store, launches, relaunch_budget=2)
    w.step()
    for _ in range(3):
        launches.end("run-i0", 1)
        w.step()
    assert launches.started("run-i0") == 3
    assert w.load(0)["state"] == lifecycle.ABANDONED


def test_a_normal_end_is_done(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    launches.end("run-i0", 0)
    w.step()
    w.step()
    assert launches.started("run-i0") == 1 and w.load(0)["state"] == lifecycle.DONE


def test_a_new_worker_adopts_the_running_launches(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    worker(objects, store, launches).step()
    again = worker(objects, store, launches)  # yeto run up restarted: a new worker, same store
    again.step()
    again.step()
    assert len(launches.starts) == 2
    assert again.load(0)["incarnation"] == 0 and again.load(0)["state"] == tw.RUNNING


class WorkerCrash(BaseException):
    """What kills a worker thread (a step's Exception is retried in place)."""


def test_a_worker_step_that_raises_backs_off_in_its_thread(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        if calls["n"] in (2, 3):  # a store read error, twice in a row
            raise OSError("store unreachable")
        w.step()

    logs, restarts = [], []
    controller = rc.RolloutController({}, pool_workers=True, interval=0.05, log=logs.append,
                                      restart_backoff_s=0, drain_on_exit=False)
    controller.add_worker("trainers", step, interval=0.02, on_restart=restarts.append)
    controller.start()
    try:
        deadline = time.time() + 10
        while calls["n"] < 5 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        controller.stop("test")
    assert calls["n"] >= 5 and restarts == [] and controller.workers["trainers"].restarts == 0
    assert sum("failure 1 in a row" in m for m in logs) == 1 and sum("failure 2 in a row" in m for m in logs) == 1
    assert len(launches.starts) == 2 and all(w.load(k)["state"] == tw.RUNNING for k in (0, 1))


def test_a_worker_thread_that_dies_is_restarted_by_the_supervisor_and_adopts(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    calls = {"n": 0}
    crashed = threading.Event()

    def step():
        calls["n"] += 1
        if calls["n"] == 2:  # after the first pass started both launches: the thread dies
            crashed.set()
            raise WorkerCrash("worker bug")
        w.step()

    restarts = []
    controller = rc.RolloutController({}, pool_workers=True, interval=0.05, log=lambda m: None,
                                      restart_backoff_s=0, drain_on_exit=False)
    controller.add_worker("trainers", step, interval=0.02, on_restart=restarts.append)
    controller.start()
    try:
        deadline = time.time() + 10
        while (calls["n"] < 6 or not restarts) and time.time() < deadline:
            time.sleep(0.02)
    finally:
        controller.stop("test")
    assert crashed.is_set() and len(restarts) >= 1 and "worker bug" in repr(restarts[0])
    assert controller.workers["trainers"].restarts >= 1
    assert len(launches.starts) == 2  # adopted after the restart: no second launch
    assert all(w.load(k)["state"] == tw.RUNNING for k in (0, 1))


def test_a_removed_island_stops_and_ends_removed(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    w = worker(objects, store, launches)
    w.step()
    store.update(lambda doc: doc.__setitem__("members", [m for m in doc["members"] if m["learner_id"] != 1]))
    w.step()
    assert states.stop_requested(1)["reason"] == "removed from the fleet's desired state"
    assert states.stop_requested(0) is None
    assert w.load(1)["state"] == tw.STOPPING
    launches.end("run-i1", 0)
    w.step()
    assert w.load(1)["state"] == tw.REMOVED and states.stop_requested(1) is None
    assert launches.started("run-i0") == 1 and launches.alive("run-i0")


def test_the_fleet_stop_is_forwarded_to_every_island_launch(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    why = {"v": None}
    w = worker(objects, store, launches, fleet_stop=lambda: why["v"])
    w.step()
    why["v"] = "STOP flag"
    w.step()
    w.step()
    assert sorted(launches.stops) == ["run-i0", "run-i1"]
    assert len(launches.starts) == 2


def test_seed_trainers_adds_the_members_once(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": [{"kind": "sampler_pool", "name": "i0-pool", "pool": {"provider": "nc", "target": 1}}]},
                expect=0)
    doc = tw.seed_trainers(store, [member(0), member(1)])
    assert doc["generation"] == 2 and sorted(dz.trainers(doc)) == ["isl0", "isl1"]
    assert "i0-pool" in dz.sampler_pools(doc)
    assert tw.seed_trainers(store, [member(5)])["generation"] == 2


@pytest.mark.parametrize("bad", [{"learner_id": "1"}, {"learner_id": None}])
def test_a_trainer_member_without_an_integer_learner_id_is_refused(bad):
    with pytest.raises(dz.DesiredError, match="learner_id"):
        dz.validate({"members": [{**member(0), **bad}]})


def test_a_fenced_controller_s_worker_touches_no_island(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    fenced = [None]
    w = worker(objects, store, launches, fenced=lambda: fenced[0])
    w.step()
    assert sorted(launches.live) == ["run-i0", "run-i1"]
    fenced[0] = "fleet/applied.json is written by another controller now"
    launches.end("run-i0", 1, "NCCL timeout")  # an infrastructure failure: relaunched, if not fenced
    store.update(lambda doc: doc["members"].remove(next(m for m in doc["members"] if m["name"] == "isl1")))
    for _ in range(3):
        w.step()
    assert launches.started("run-i0") == 1 and launches.alive("run-i1") and launches.stops == []
    assert w.states.stop_requested(1) is None and w.load(0)["state"] == tw.STARTING


def test_a_launch_of_another_host_is_never_started_twice(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    first = FakeLaunches()
    worker(objects, store, first, host="host-a").step()
    assert worker(objects, store, FakeLaunches(), host="host-a").load(0)["host"] == "host-a"
    # run up restarted on another host: its local registry sees no launch at all
    elsewhere, logs = FakeLaunches(), []
    other = tw.TrainerWorker(store, objects, elsewhere, lifecycle.IslandStateStore(objects, log=lambda m: None),
                             log=logs.append, host="host-b", clock=lambda: 10 ** 10)  # past the start grace
    for _ in range(2):
        other.step()
    assert elsewhere.starts == [] and other.load(0)["state"] == tw.STARTING
    assert sum("started on host host-a" in m for m in logs) == 2  # once per island
    # the same host again adopts them
    again = FakeLaunches()
    again.live = dict(first.live)
    worker(objects, store, again, host="host-a").step()
    assert again.starts == [] and worker(objects, store, again).load(1)["state"] == tw.RUNNING


def test_a_stop_cleared_by_a_starting_launch_is_written_again(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    # removed while island 1's launch is still starting: the worker writes its STOP ...
    store.update(lambda doc: doc["members"].remove(next(m for m in doc["members"] if m["name"] == "isl1")))
    w.step()
    assert w.load(1)["state"] == tw.STOPPING and w.states.stop_requested(1) is not None
    w.states.clear_stop(1)  # ... and the launcher, coming up, clears it (clear_stop at its start)
    w.step()
    assert w.states.stop_requested(1) is not None  # written again while the launch runs
    launches.end("run-i1", 0)
    w.step()
    assert w.load(1)["state"] == tw.REMOVED and w.states.stop_requested(1) is None
    # a move the same way: its STOP is written again until the old launch ended
    store.update(lambda doc: next(m for m in doc["members"] if m["name"] == "isl0").update(gpu="aws:2xh100@us-east-2"))
    w.step()
    assert w.load(0)["state"] == tw.MOVING
    w.states.clear_stop(0)
    w.step()
    assert w.states.stop_requested(0) is not None and "run-i0-m1" not in launches.live
    launches.end("run-i0", 0)
    w.step()
    assert "run-i0-m1" in launches.live and w.states.stop_requested(0) is None


def test_an_unknown_fence_skips_the_step_only(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    answer = ["unknown: fleet/applied.json not read (OSError)"]
    w = worker(objects, store, launches, fenced=lambda: answer[0])
    w.step()
    assert launches.starts == []
    answer[0] = None
    w.step()
    assert sorted(launches.live) == ["run-i0"]


def test_a_fence_unknown_for_long_is_an_error_not_a_silent_idle(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    logs = []
    w = tw.TrainerWorker(store, objects, FakeLaunches(), lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=logs.append, fenced=lambda: "unknown: fleet/applied.json not read (OSError)",
                         unknown_alarm_steps=5)
    for _ in range(10):
        w.step()
    assert sum("ERROR" in m for m in logs) == 2 and "10 step(s) in a row" in logs[-1]


class BrokenStore(DirectoryObjectStore):
    """The run store, failing while ``broken`` (an outage)."""

    broken = False

    def _check(self):
        if self.broken:
            raise OSError("store unreachable")

    def read(self, key):
        self._check()
        return super().read(key)

    def write(self, key, data):
        self._check()
        return super().write(key, data)

    def list(self, prefix=""):
        self._check()
        return super().list(prefix)


def test_the_fleet_stop_is_forwarded_while_the_store_does_not_read(tmp_path):
    """The fleet STOP (a local flag) is read before the fence and the store: this host's
    island launches are stopped through the launch ops, and the islands' STOPs in the
    store are written once it answers again; the failure is a warning, then an error."""
    objects = BrokenStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": [member(0), member(1)]}, expect=0)
    il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=2, learners=2, max_learners=4, run="run")
    launches, logs, why = FakeLaunches(), [], [None]
    w = tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=logs.append, fleet_stop=lambda: why[0], unknown_alarm_steps=3,
                         fenced=lambda: "unknown: fleet/applied.json not read (OSError)" if objects.broken else None)
    w.step()
    assert sorted(launches.live) == ["run-i0", "run-i1"]
    objects.broken, why[0] = True, "STOP flag"
    for _ in range(3):
        w.step()
    assert sorted(launches.stops) == ["run-i0", "run-i1"]  # stopped directly, without the store
    assert [m.split(":")[1].strip() for m in logs if "not written to every island's STOP" in m] == ["WARNING", "ERROR"]
    objects.broken = False
    w.step()
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    assert states.stop_requested(0) is not None and states.stop_requested(1) is not None
    assert sorted(launches.stops) == ["run-i0", "run-i1"] and len(launches.starts) == 2  # once each, nothing started


def test_a_restarted_worker_stops_this_host_s_island_launches_from_the_run_registry(tmp_path):
    """A worker that restarted knows no launch yet; with the store down it still stops
    the island launches this host's run registry knows under the fleet's prefixes."""
    objects = BrokenStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": [member(0)]}, expect=0)
    launches = FakeLaunches()
    launches.live = {"run-i0-m2": [], "run-i3": [], "run-i9": []}
    objects.broken = True
    w = tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                         log=lambda m: None, fleet_stop=lambda: "run STOP",
                         local_launches=lambda: ["run-i0-m2", "run-i3", "run-i5"])  # run-i5 ended already
    w.step()
    w.step()
    assert sorted(launches.stops) == ["run-i0-m2", "run-i3"]
    # a registry that does not read either: the launches the worker knows still stop
    def broken():
        raise OSError("registry")
    w2 = tw.TrainerWorker(store, objects, launches, lifecycle.IslandStateStore(objects, log=lambda m: None),
                          log=lambda m: None, fleet_stop=lambda: "run STOP", local_launches=broken)
    w2._seen.add("run-i9")
    w2.step()
    assert "run-i9" in launches.stops


def test_an_incarnation_without_a_price_keeps_the_others_dollars(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches, clock = FakeLaunches(), [0.0]
    w = worker(objects, store, launches, clock=lambda: clock[0])
    w.step()
    clock[0] = 3600.0 * 10  # 10 h at $8 = $80
    store.update(lambda doc: dz.trainers(doc)["isl0"].update(gpu="aws:2xh100@us-east-2", usd_per_h=None))
    w.step()
    launches.end("run-i0", 0)
    w.step()
    clock[0] = 3600.0 * 12  # then 2 h without a price
    ((k, hours, usd, unpriced),) = tw.billed(objects, now=clock[0])
    assert (k, round(hours, 6), round(usd, 6), round(unpriced, 6)) == (0, 12.0, 80.0, 2.0)


def test_a_launch_torn_down_by_run_down_starts_again_on_any_host(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    worker(objects, store, FakeLaunches(), host="host-a").step()
    assert tw.mark_down(objects, ["run-i0"], log=lambda m: None) == [0]  # island 1 was not confirmed down
    here = FakeLaunches()
    w = worker(objects, store, here, host="host-b", clock=lambda: 10 ** 10)
    w.step()
    assert sorted(here.live) == ["run-i0-m1"]  # island 1 still host-a's: left alone
    assert w.load(0)["host"] == "host-b" and w.load(0)["incarnation"] == 1 and w.load(1)["host"] == "host-a"
    assert w.load(0)["billed_s"] is not None and w.load(0)["closed_at"] is None



class StopWritesFail(DirectoryObjectStore):
    """The run store, reading fine, but refusing every island STOP write while ``failing``."""

    failing = False

    def write(self, key, data):
        if self.failing and str(key).endswith("/STOP"):
            raise OSError("STOP write refused")
        return super().write(key, data)


def _stop_writes_fail(tmp_path, members):
    objects = StopWritesFail(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": members}, expect=0)
    il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=2, learners=2, max_learners=4, run="run")
    return objects, store


def test_a_stop_write_that_fails_alone_is_a_failure_not_written(tmp_path):
    """The store reads, only the STOP writes fail (the island state store swallows the
    error and returns False): the fleet stop is not logged written, it escalates and is
    written once the store takes it; a move or a remove waits for its STOP."""
    objects, store = _stop_writes_fail(tmp_path, [member(0), member(1)])
    launches, logs, why = FakeLaunches(), [], [None]
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    w = tw.TrainerWorker(store, objects, launches, states, log=logs.append, fleet_stop=lambda: why[0],
                         unknown_alarm_steps=3)
    w.step()
    # a move and a remove while STOP writes fail: neither is recorded as under way
    objects.failing = True
    store.update(lambda doc: next(m for m in doc["members"] if m["name"] == "isl0").update(gpu="aws:2xh100@us-east-2"))
    store.update(lambda doc: doc["members"].remove(next(m for m in doc["members"] if m["name"] == "isl1")))
    w.step()
    assert w.load(0)["state"] == tw.STARTING and w.load(1)["state"] == tw.STARTING
    assert any("WARNING: trainer island 0: its STOP" in m for m in logs)
    assert any("WARNING: trainer island 1: its STOP" in m for m in logs)
    # the fleet stop: this host's launches stopped directly, the STOP writes escalate
    why[0] = "STOP flag"
    for _ in range(3):
        w.step()
    assert sorted(launches.stops) == ["run-i0", "run-i1"]
    assert not any("fleet stop written" in m for m in logs)
    alarms = [m.split(":")[1].strip() for m in logs if "not written to every island's STOP" in m]
    assert alarms == ["WARNING", "ERROR"]
    objects.failing = False
    w.step()
    assert states.stop_requested(0) is not None and states.stop_requested(1) is not None
    assert any("fleet stop written" in m for m in logs)


def test_a_launch_that_keeps_handing_off_is_given_up_stopped_and_still_billed(tmp_path):
    """After HANDOFF_LIMIT hand-offs in a row the island is abandoned, but only once its
    island STOP is written (the launcher that holds it stops), and it stays billed until
    it is removed or run down; a fleet stop still reaches an abandoned island."""
    objects, store = _stop_writes_fail(tmp_path, [member(0)])
    launches, logs, clock = FakeLaunches(), [], [0.0]
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    w = tw.TrainerWorker(store, objects, launches, states, log=logs.append, clock=lambda: clock[0])
    w.step()

    def hand_off():
        launches.live.pop("run-i0", None)
        launches.exits["run-i0"] = {"exit_code": 12, "log": "", "detail": "another launcher holds the records"}
        w.step()

    for _ in range(tw.HANDOFF_LIMIT):
        hand_off()
    assert launches.started("run-i0") == 1 + tw.HANDOFF_LIMIT  # started again at once, a bounded number of times
    objects.failing = True  # the STOP does not write: not given up yet, not started again either
    hand_off()
    assert w.load(0)["state"] == tw.STARTING and launches.started("run-i0") == 1 + tw.HANDOFF_LIMIT
    objects.failing = False
    hand_off()
    record = w.load(0)
    assert record["state"] == tw.ABANDONED and states.stop_requested(0) is not None
    assert sum("ERROR" in m and "handed off" in m for m in logs) == 1
    assert launches.started("run-i0") == 1 + tw.HANDOFF_LIMIT
    clock[0] = 7200.0  # still billed: 2 h at $8
    ((_, hours, usd, _),) = tw.billed(objects, now=clock[0])
    assert round(hours, 6) == 2.0 and round(usd, 6) == 16.0
    # a fleet stop goes to the abandoned island too (its STOP cleared meanwhile)
    states.clear_stop(0)
    w.fleet_stop = lambda: "STOP flag"
    w.step()
    assert states.stop_requested(0) is not None
    # run down confirms it down: the bill closes, and it stays given up (never restarted)
    assert tw.mark_down(objects, ["run-i0"], log=lambda m: None, clock=lambda: 9000.0) == []
    ((_, hours, usd, _),) = tw.billed(objects, now=99999.0)
    assert round(hours, 6) == 2.5 and round(usd, 6) == 20.0
    assert w.load(0)["state"] == tw.ABANDONED and not w.load(0).get("billing_open")
    again = FakeLaunches()
    worker(objects, store, again, clock=lambda: 10 ** 10).step()
    assert again.starts == []


def test_a_launch_that_comes_up_resets_its_hand_offs(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    for _ in range(tw.HANDOFF_LIMIT):
        launches.live.pop("run-i0", None)
        launches.exits["run-i0"] = {"exit_code": 12, "log": "", "detail": "held"}
        w.step()
    launches.live["run-i0"] = []  # up at last
    w.step()
    assert w.load(0)["state"] == tw.RUNNING and w.load(0)["handoffs"] == 0
    launches.live.pop("run-i0")
    launches.exits["run-i0"] = {"exit_code": 12, "log": "", "detail": "held"}
    w.step()
    assert w.load(0)["state"] == tw.STARTING and w.load(0)["handoffs"] == 1  # a fresh count, not given up


def test_a_move_starts_with_no_hand_offs(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    launches.live.pop("run-i0")
    launches.exits["run-i0"] = {"exit_code": 12, "log": "", "detail": "held"}
    w.step()
    assert w.load(0)["handoffs"] == 1
    store.update(lambda doc: next(m for m in doc["members"] if m["name"] == "isl0").update(gpu="aws:2xh100@us-east-2"))
    w.step()  # moving (the relaunched launch runs)
    launches.end("run-i0", 0)
    w.step()
    assert w.load(0)["prefix_now"] == "run-i0-m1" and w.load(0)["handoffs"] == 0


def test_one_island_s_stop_write_failing_never_keeps_the_next_from_its_stop(tmp_path):
    class RefuseIsland0(DirectoryObjectStore):
        def write(self, key, data):
            if str(key) == lifecycle.stop_key(0):
                raise OSError("refused")
            return super().write(key, data)

    objects = RefuseIsland0(tmp_path / "store")
    store = dz.DesiredStore(objects)
    store.apply({"members": [member(0), member(1), member(2)]}, expect=0)
    il.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=2, learners=3, max_learners=4, run="run")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    logs = []
    w = tw.TrainerWorker(store, objects, FakeLaunches(), states, log=logs.append, fleet_stop=lambda: None)
    w.step()
    w.fleet_stop = lambda: "STOP flag"
    w.step()
    assert states.stop_requested(0) is None
    assert states.stop_requested(1) is not None and states.stop_requested(2) is not None
    assert any("island 0" in m and "not written to every island's STOP" in m for m in logs)


def _abandon(objects, store, launches, w):
    for _ in range(tw.HANDOFF_LIMIT + 1):
        launches.live.pop("run-i0", None)
        launches.exits["run-i0"] = {"exit_code": 12, "log": "", "detail": "held"}
        w.step()
    assert w.load(0)["state"] == tw.ABANDONED and w.load(0)["billing_open"]


def test_a_hand_off_keeps_the_time_billed_so_far(tmp_path):
    """A launcher that lost the island's lock while it ran exits 13 and tears nothing down:
    the hour billed so far stays billed ($8), the relaunch's time adds to it."""
    objects, store = fleet(tmp_path, [member(0)])
    launches, clock = FakeLaunches(), [0.0]
    w = worker(objects, store, launches, clock=lambda: clock[0])
    w.step()
    clock[0] = 3600.0
    launches.live.pop("run-i0")
    launches.exits["run-i0"] = {"exit_code": 13, "log": "", "detail": "lock lost"}
    w.step()
    ((_, hours, usd, _),) = tw.billed(objects, now=3600.0)
    assert (round(hours, 6), round(usd, 6)) == (1.0, 8.0)
    ((_, hours, usd, _),) = tw.billed(objects, now=5400.0)
    assert (round(hours, 6), round(usd, 6)) == (1.5, 12.0)


def test_a_given_up_island_moved_again_carries_no_billing_open_and_run_down_never_restarts_it(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    _abandon(objects, store, launches, w)
    store.update(lambda doc: dz.trainers(doc)["isl0"].update(gpu="aws:2xh100@us-east-2"))
    w.step()  # changed after it ended: its next incarnation
    record = w.load(0)
    assert record["prefix_now"] == "run-i0-m1" and record.get("billing_open") is None
    launches.end("run-i0-m1", 0)
    w.step()
    assert w.load(0)["state"] == tw.DONE and w.load(0).get("billing_open") is None
    assert tw.mark_down(objects, ["run-i0-m1"], log=lambda m: None) == []  # done: never recorded down
    again = FakeLaunches()
    worker(objects, store, again, clock=lambda: 10 ** 10).step()
    assert again.starts == [] and w.load(0)["state"] == tw.DONE


def test_an_adopted_launch_carries_no_billing_open(tmp_path):
    objects, store = fleet(tmp_path, [member(0)])
    launches = FakeLaunches()
    w = worker(objects, store, launches)
    w.step()
    w.save(0, {**w.load(0), "billing_open": "stale"})
    w.step()  # alive: adopted as running
    assert w.load(0)["state"] == tw.RUNNING and w.load(0).get("billing_open") is None


def test_removing_a_given_up_island_keeps_its_stop_and_closes_its_bill(tmp_path):
    objects, store = fleet(tmp_path, [member(0), member(1)])
    launches = FakeLaunches()
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    w = tw.TrainerWorker(store, objects, launches, states, log=lambda m: None)
    w.step()
    _abandon(objects, store, launches, w)
    store.update(lambda doc: doc["members"].remove(next(m for m in doc["members"] if m["name"] == "isl0")))
    w.step()
    record = w.load(0)
    assert record["state"] == tw.REMOVED and record.get("billing_open") is None and record["closed_at"] is not None
    assert states.stop_requested(0) is not None  # the launcher that may still hold it keeps its STOP
