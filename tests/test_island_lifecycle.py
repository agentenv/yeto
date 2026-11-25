"""The unified trainer island lifecycle (``yeto.launch.island_lifecycle``): the state
machine, the exit decision table, the island records in the run store and each island's
own STOP, with the fleet controller writing and honoring them. No cloud."""

from __future__ import annotations

import json
import sys

import pytest

from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
from test_rl_algorithm_provenance import _fake_sky  # noqa: F401  (fixture)
from tests.test_rl_object_store import FakeS3
from yeto.cloud import s3_store
from yeto.config_errors import CONFIG_ERROR_EXIT, ConfigError, config_error_line
from yeto.launch import island_lifecycle as lc
from yeto.launch.fleet import FixedRosterIslandAbandoned, FleetController
from yeto.rl.leave_code import LeaveReason


# -- the state machine ----------------------------------------------------------------------------

def test_states_and_moves():
    assert lc.STATES == ("ordering", "starting", "running", "recovering", "draining", "done", "abandoned",
                         "failed")
    assert lc.TERMINAL == {"done", "abandoned", "failed"}
    assert lc.allowed(None, "ordering") and lc.allowed("ordering", "starting")
    assert lc.allowed("starting", "running") and lc.allowed("running", "recovering")
    assert lc.allowed("recovering", "running") and lc.allowed("running", "draining")
    assert lc.allowed("draining", "done")
    assert not lc.allowed("draining", "running")  # a draining island is never started again in place
    assert not lc.allowed("done", "running") and lc.allowed("done", "ordering")  # only an explicit restart
    assert not lc.allowed("running", "bogus")


# -- the decision table: the five classes of 1.1 ---------------------------------------------------

def test_configuration_error_is_never_relaunched():
    for d in (lc.classify(exit_code=CONFIG_ERROR_EXIT),
              lc.classify(log_text="noise\n" + config_error_line(ConfigError("bad --fragments"))),
              lc.classify(leave_reason=LeaveReason.CONFIG_MISMATCH),
              lc.classify(leave_reason="CONFIG_MISMATCH")):
        assert (d.cause, d.action, d.state) == ("config_error", "fail", "failed")
        assert not d.relaunch


def test_layout_mismatch_exit_10_is_a_configuration_error():
    from yeto.rl.adapters.miles.models.chunked_full_parameter import FullParameterLayoutMismatch

    error = FullParameterLayoutMismatch("expected_fragments=4, actual_fragments=10")
    d = lc.classify_error(error)
    assert (d.cause, d.state) == ("config_error", "failed")
    assert "actual_fragments=10" in d.detail
    assert lc.classify(exit_code=10).state == "failed"


def test_strict_failure_is_never_relaunched():
    for d in (lc.classify(log_text="x\n[yeto-rl-strict-failure] mixed_version_group_count"),
              lc.classify(log_text="session mismatch (HELLO refused: profile)"),
              lc.classify(leave_reason=LeaveReason.TRAINING_FAILED)):
        assert (d.cause, d.action, d.state) == ("strict", "fail", "failed")


def test_infrastructure_failure_and_lost_nodes_are_relaunched():
    for d, cause in ((lc.classify(exit_code=1), "infra"),
                     (lc.classify(leave_reason=LeaveReason.INFRA_FAILED), "infra"),
                     (lc.classify(), "infra"),
                     (lc.classify(lost=True), "lost"),
                     (lc.classify(leave_reason=LeaveReason.LEASE_EXPIRED), "lost")):
        assert (d.cause, d.action, d.state) == (cause, "relaunch", "recovering")
    assert lc.classify_error(RuntimeError("NCCL timeout")).relaunch


def test_preemption_is_relaunched():
    for d in (lc.classify(preempted=True), lc.classify(leave_reason=LeaveReason.PREEMPTED)):
        assert (d.cause, d.action, d.state) == ("preempted", "relaunch", "recovering")


def test_normal_end_and_stop():
    for d in (lc.classify(succeeded=True), lc.classify(exit_code=0), lc.classify(leave_reason=LeaveReason.FINISHED),
              lc.classify_error(None)):
        assert (d.cause, d.action, d.state) == ("finished", "finish", "done")
    for d in (lc.classify(stopped=True), lc.classify(leave_reason=LeaveReason.STOPPED),
              lc.classify_error(KeyboardInterrupt())):
        assert (d.cause, d.action, d.state) == ("stopped", "stop", "done")
        assert not d.relaunch


def test_the_first_matching_row_wins():
    # a configuration error line beats a non-zero exit, a strict line beats a stop
    assert lc.classify(exit_code=1, log_text=config_error_line(ConfigError("x"))).cause == "config_error"
    assert lc.classify(stopped=True, log_text="[yeto-rl-strict-failure] y").cause == "strict"
    assert lc.classify(exit_code=0, preempted=True).cause == "finished"


# -- the island records in the store ----------------------------------------------------------------

def _s3(client=None):
    return s3_store.S3ObjectStore("s3://bkt/runs/r", client=client or FakeS3())


def test_records_survive_the_controller(tmp_path):
    client = FakeS3()
    states = lc.IslandStateStore(_s3(client), clock=lambda: 100.0)
    states.write(0, "ordering", name="r-l0-modal", source="modal")
    states.write(0, "running", handle={"cluster": "r-l0-modal", "call_id": "fc-1"}, incarnation=0)
    states.write(1, "running", source="sky", handle={"cluster": "r-l1-aws", "job_id": 3})
    # another machine reads the same store after the launcher's machine is gone
    other = lc.IslandStateStore(_s3(client))
    records = other.read_all()
    assert sorted(records) == [0, 1]
    assert records[0]["state"] == "running" and records[0]["handle"]["call_id"] == "fc-1"
    assert records[0]["source"] == "modal" and records[0]["name"] == "r-l0-modal"
    assert [h["to"] for h in records[0]["history"]] == ["ordering", "running"]
    assert records[1]["handle"] == {"cluster": "r-l1-aws", "job_id": 3}
    assert records[0]["controller"]["pid"] and records[0]["controller"]["token"]
    raw = json.loads(client.objects[("bkt", "runs/r/fleet/trainers/0/state.json")])
    assert raw["schema"] == lc.SCHEMA and raw["island"] == 0


def test_a_store_that_does_not_answer_never_stops_supervision():
    class Broken:
        def read(self, key):
            raise OSError("store down")

        def write(self, key, data):
            raise OSError("store down")

    logged = []
    states = lc.IslandStateStore(Broken(), log=logged.append)
    states.write(0, "running")
    states.write(0, "recovering")
    assert states.read(0) is None and states.stop_requested(0) is None
    assert len([m for m in logged if "write failed" in m]) == 1  # logged once per kind


def test_island_stop_channel_round_trip(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path))
    assert states.stop_requested(1) is None
    assert states.request_stop(1, "breaker:run.starvation", by="run up")
    assert states.stop_requested(1)["reason"] == "breaker:run.starvation"
    assert (tmp_path / "fleet/trainers/1/STOP").is_file()
    assert states.stop_requested(0) is None
    assert states.clear_stop(1) and states.stop_requested(1) is None


def test_unknown_state_is_refused():
    with pytest.raises(ValueError):
        lc.IslandStateStore(_s3()).write(0, "sleeping")


# -- the fleet controller writes the records and honors each island's STOP ---------------------------

def _ctl(ops, client, *, learners=None, **kw):
    learners = learners or {"r-l0-modal": 1, "r-l1-modal": 2}
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    refs = {name: lc.IslandRef(k, "modal") for k, name in enumerate(learners)}
    recorder = lc.FleetRecorder(lc.IslandStateStore(_s3(client)), refs)
    return FleetController(learners={n: (f"task-{n}", j) for n, j in learners.items()},
                           syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30,
                           recover_timeout=600, thread_cls=ImmediateThread, recorder=recorder, **kw)


def _records(client):
    return lc.IslandStateStore(_s3(client)).read_all([0, 1])


def test_controller_records_each_island_and_its_relaunch():
    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l0-modal"] = [RUNNING, RUNNING, RUNNING, SUCCEEDED]
    ops.status_seq["r-l1-modal"] = [FAILED]
    ops.relaunch_results["r-l1-modal"] = ["fc-new"]
    ops.after_relaunch["r-l1-modal"] = [RUNNING, SUCCEEDED]
    ops.up["r-l1-modal"] = False
    ctl = _ctl(ops, client)
    ctl.run()
    records = _records(client)
    assert records[0]["state"] == "done" and records[0]["handle"] == {"cluster": "r-l0-modal", "call_id": "1"}
    r1 = records[1]
    assert r1["state"] == "done" and r1["relaunches"] == 1 and r1["incarnation"] == 1
    assert r1["handle"]["call_id"] == "fc-new"
    assert [h["to"] for h in r1["history"]] == ["running", "recovering", "running", "done"]
    assert r1["decision"]["cause"] == "finished"
    assert "FAILED" in r1["history"][1]["reason"]  # why it went recovering


def test_island_1_stop_stops_island_1_only():
    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l0-modal"] = [RUNNING] * 6 + [SUCCEEDED]
    ops.status_seq["r-l1-modal"] = [RUNNING]
    ctl = _ctl(ops, client, elastic=True)
    lc.IslandStateStore(_s3(client)).request_stop(1, "breaker:run.starvation", by="run up")
    exits = ctl.run()
    assert ops.down_calls == ["r-l1-modal"]  # island 0 is never touched
    assert ops.relaunch_calls == []
    assert exits["r-l0-modal"].endswith("SUCCEEDED")
    assert exits["r-l1-modal"].startswith("STOPPED") and "run.starvation" in exits["r-l1-modal"]
    records = _records(client)
    assert records[1]["state"] == "done" and records[1]["decision"]["cause"] == "stopped"
    assert [h["to"] for h in records[1]["history"]][-2:] == ["draining", "done"]
    assert records[1]["stop"]["by"] == "run up"
    assert records[0]["state"] == "done"


def test_island_stop_of_a_fixed_roster_ends_the_fleet_naming_the_roster():
    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l0-modal"] = [RUNNING]
    ops.status_seq["r-l1-modal"] = [RUNNING]
    ctl = _ctl(ops, client, fixed_roster=True)
    lc.IslandStateStore(_s3(client)).request_stop(1, "operator")
    with pytest.raises(FixedRosterIslandAbandoned, match="fixed roster"):
        ctl.run()
    assert ops.down_calls == ["r-l1-modal"]
    assert _records(client)[1]["state"] == "done"


def test_configuration_error_island_is_failed_in_its_record():
    class ConfigOps(FakeOps):
        def rl_strict_failure(self, cluster, job_id):
            return config_error_line(ConfigError("layout")) if cluster == "r-l1-modal" else None

    ops, client = ConfigOps(), FakeS3()
    ops.status_seq["r-l0-modal"] = [RUNNING]
    ops.status_seq["r-l1-modal"] = [FAILED]
    ctl = _ctl(ops, client, elastic=True, learners={"r-l0-modal": 1, "r-l1-modal": 2})
    ops.status_seq["r-l0-modal"] = [RUNNING, RUNNING, SUCCEEDED]
    ctl.run()
    records = _records(client)
    assert records[1]["state"] == "failed" and records[1]["decision"]["cause"] == "config_error"
    assert ops.relaunch_calls == []
    assert records[0]["state"] == "done"


def test_fixed_roster_abandon_is_recorded_before_the_run_ends():
    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l0-modal"] = [RUNNING]
    ops.status_seq["r-l1-modal"] = [FAILED]
    ctl = _ctl(ops, client, fixed_roster=True)
    ctl.recover_timeout = 0
    with pytest.raises(FixedRosterIslandAbandoned):
        ctl.run()
    assert _records(client)[1]["state"] == "abandoned"


def test_without_a_recorder_nothing_is_read_or_written():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, SUCCEEDED]
    ops.status_seq[SYNCER] = [RUNNING]
    ctl = FleetController(learners={"l0": ("t", 1)}, syncer=(SYNCER, "t", 1), sky_ops=ops, poll_interval=30,
                          recover_timeout=600, thread_cls=ImmediateThread)
    assert ctl.recorder is None
    assert ctl.run() == {"l0": "JobStatus.SUCCEEDED"}


# -- the launcher switch: a single trainer island keeps every behavior --------------------------------

def _args(**kw):
    from types import SimpleNamespace

    base = dict(training_mode="rl", island_lifecycle="auto", dry_run=False, gpu="modal:1xl40s")
    base.update(kw)
    return SimpleNamespace(**base)


def test_switch_auto_is_off_for_one_trainer_island(monkeypatch):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import island_lifecycle_enabled

    one = parse_gpu_spec("modal:1xl40s")
    two = parse_gpu_spec("modal:1xl40s,modal:1xl40s")
    assert not island_lifecycle_enabled(_args(), one)
    assert island_lifecycle_enabled(_args(), two)
    assert not island_lifecycle_enabled(_args(training_mode="sft"), two)
    assert not island_lifecycle_enabled(_args(island_lifecycle="off"), two)
    assert not island_lifecycle_enabled(_args(dry_run=True), two)
    assert island_lifecycle_enabled(_args(island_lifecycle="on"), one)  # the opt-in of a single island


def test_switch_counts_trainers_not_sampler_islands(monkeypatch):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import island_lifecycle_enabled
    from yeto.rl.engine import sample_roles

    monkeypatch.setattr(sample_roles, "sampler_islands", lambda args, n: [1])
    assert not island_lifecycle_enabled(_args(), parse_gpu_spec("modal:1xl40s,modal:1xl4"))


def test_cli_default_is_auto():
    from yeto.cli import parse_args

    args = parse_args(["--training-mode", "rl", "--gpu", "modal:1xl40s", "--model", "m", "--data", "d"])
    assert args.island_lifecycle == "auto"


# -- review follow-ups: the launcher's exit, a store that hangs, the writer fence ---------------------

def test_an_island_stop_is_neither_a_success_nor_a_failure():
    from yeto.launcher import island_outcome

    exits = {"r-l0": "JobStatus.SUCCEEDED", "r-l1": "STOPPED (island STOP: breaker:run.starvation)"}
    assert island_outcome(exits) == (["r-l0"], ["r-l1"], [])
    exits["r-l2"] = "ABANDONED after 1500s"
    assert island_outcome(exits) == (["r-l0"], ["r-l1"], ["r-l2"])
    assert island_outcome({"r-l0": "STOPPED (island STOP: x)", "r-l1": "STOPPED (island STOP: y)"}) == \
        ([], ["r-l0", "r-l1"], [])
    assert island_outcome({"r-l0": "STOPPED"}) == ([], [], ["r-l0"])  # not an island STOP: a failure


def _sky_fleet(monkeypatch, tmp_path, stops):
    """``launcher.run`` of a two-island sky fleet with no fixed roster (what ``yeto run up``
    compiles), both islands succeeding unless an island STOP of ``stops`` ends them."""
    import types as _types

    import yeto.launcher as launcher
    from test_rl_algorithm_provenance import _launcher_args
    from test_rl_launcher_island_failure import Clock, _setup
    from yeto.rl.engine.object_store import DirectoryObjectStore

    record, clock = [], Clock()
    _setup(monkeypatch, tmp_path, failing=set(), succeeding={0, 1}, clock=clock, record=record)
    monkeypatch.setattr(launcher.subprocess, "run",  # the syncer's island-pool status: not there
                        lambda *a, **k: _types.SimpleNamespace(returncode=1, stdout="", stderr=""))
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path / "store"))
    monkeypatch.setattr(launcher, "_island_state_store", lambda args: states)
    monkeypatch.setattr(launcher, "_relaunch_of_same_run", lambda args: True)  # keep the STOPs below
    for k in stops:
        states.request_stop(k, "breaker:run.trainer_silence", by="yeto run up")
    gpu = "aws:1xa100@us-east-1,aws:1xa100@us-east-1"
    args = _launcher_args("ports", ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64,
                                    "--rl-sync-island-lease-s", "150", "--rl-sync-quorum", "1",
                                    "--rl-syncer-epoch", "0"), gpu=gpu)
    args.keep, args.recover_timeout, args.controller_poll = False, 600.0, 30.0
    return launcher.run(args), states


def test_one_island_stopped_does_not_fail_the_launch(monkeypatch, tmp_path, _fake_sky):  # noqa: F811
    base, _ = _sky_fleet(monkeypatch, tmp_path / "a", stops=())
    code, states = _sky_fleet(monkeypatch, tmp_path / "b", stops=(1,))
    assert code == base != 1  # island 1's own STOP changes nothing in the launch's exit
    records = states.read_all([0, 1])
    assert records[1]["state"] == "done" and records[1]["decision"]["cause"] == "stopped"
    assert records[0]["state"] == "done" and records[0]["decision"]["cause"] == "finished"


def test_every_island_stopped_is_the_fleet_stop_exit(monkeypatch, tmp_path, _fake_sky):  # noqa: F811
    from yeto.launch.fleet import RUN_STOPPED_EXIT

    code, states = _sky_fleet(monkeypatch, tmp_path, stops=(0, 1))
    assert code == RUN_STOPPED_EXIT  # the fleet stopped, as on the run's STOP flag; not a failure (1)
    assert all(r["decision"]["cause"] == "stopped" for r in states.read_all([0, 1]).values())


class _Hanging:
    """A store whose every call blocks until ``release`` is set."""

    def __init__(self):
        import threading

        self.release = threading.Event()
        self.objects = {}

    def read(self, key):
        self.release.wait()
        return self.objects.get(key)

    def write(self, key, data):
        self.release.wait()
        self.objects[key] = data


def test_a_background_store_never_blocks_the_poll_loop():
    import time as _time

    store = _Hanging()
    states = lc.IslandStateStore(store, background=True, stop_poll_s=0.05)
    t0 = _time.monotonic()
    for _ in range(20):  # a poll loop: records and STOP checks while the store hangs
        states.write(1, "running", name="r-l1")
        assert states.stop_requested(1) is None
    assert _time.monotonic() - t0 < 1.0
    assert not states.flush(0.1)  # still queued
    store.objects[lc.stop_key(1)] = json.dumps({"reason": "operator"}).encode()
    store.release.set()
    assert states.flush(5.0)
    assert json.loads(store.objects[lc.state_key(1)])["state"] == "running"
    deadline = _time.monotonic() + 5.0
    while states.stop_requested(1) is None and _time.monotonic() < deadline:
        _time.sleep(0.02)
    assert states.stop_requested(1)["reason"] == "operator"  # the cache follows the store
    assert states.close(5.0)


def test_the_fleet_controller_reads_island_stops_from_the_cache():
    store = _Hanging()
    states = lc.IslandStateStore(store, background=True, stop_poll_s=0.05)
    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [RUNNING, SUCCEEDED]
    ops.status_seq["r-l1-modal"] = [RUNNING, SUCCEEDED]
    ops.status_seq[SYNCER] = [RUNNING]
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    ctl = FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                          elastic=True, recorder=lc.FleetRecorder(states, refs))
    exits = ctl.run()  # the store hangs the whole time: the fleet still runs to its end
    assert all(s.endswith("SUCCEEDED") for s in exits.values())
    store.release.set()
    assert states.close(5.0)


class _Lost(BaseException):  # the shape of store_lock.LockLost
    def __init__(self, token, by):
        super().__init__(f"lock token {token} is fenced by token {by}")
        self.token, self.by = token, by


class _Unconfirmed(BaseException):  # the shape of store_lock.LockUnconfirmed
    pass


class _Lock:
    def __init__(self, token):
        self.token, self.raises = token, None

    def guard(self):
        if self.raises is not None:
            raise self.raises


def test_a_fenced_controller_writes_no_record(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path)
    old_lock, logged = _Lock(1), []
    old = lc.IslandStateStore(objects, fence=old_lock, log=logged.append)
    old.write(0, "running", name="r-l0")
    assert lc.IslandStateStore(objects).read(0)["fence_token"] == 1
    old_lock.raises = _Unconfirmed("store did not answer")
    old.write(0, "recovering")  # not confirmed: skipped, not fenced
    assert lc.IslandStateStore(objects).read(0)["state"] == "running" and old.fenced is None
    old_lock.raises = None
    old.write(0, "recovering")
    assert lc.IslandStateStore(objects).read(0)["state"] == "recovering"
    new = lc.IslandStateStore(objects, fence=_Lock(2))  # a restarted launcher took the lock over
    new.write(0, "running", name="r-l0", adopted=True)
    old_lock.raises = _Lost(1, 2)
    old.write(0, "abandoned")
    old_lock.raises = None  # sticky: a fenced controller never writes again
    old.write(0, "done")
    record = lc.IslandStateStore(objects).read(0)
    assert record["state"] == "running" and record["fence_token"] == 2
    assert old.fenced is not None and any("fenced" in m for m in logged)
    assert old.request_stop(0, "operator")  # the STOP channel is not fenced


def test_a_background_write_the_fence_or_the_store_did_not_take_is_tried_again(tmp_path):
    """A terminal record (done / failed / abandoned) usually has no later write: a body the
    fence could not confirm, or the store did not take, is queued again (unless a newer one
    is queued) until it lands."""
    import time as _time

    from yeto.rl.engine.object_store import DirectoryObjectStore

    objects = DirectoryObjectStore(tmp_path)
    lock = _Lock(1)
    states = lc.IslandStateStore(objects, fence=lock, background=True, stop_poll_s=0.05)
    states.write(0, "running", name="r-l0")
    assert states.flush(5.0)
    lock.raises = _Unconfirmed("store did not answer")
    states.write(0, "done", exit="SUCCEEDED")  # the last write of the island
    _time.sleep(0.3)
    assert lc.IslandStateStore(objects).read(0)["state"] == "running"  # not confirmed yet: not written
    assert not states.flush(0.1)  # still queued
    lock.raises = None
    assert states.flush(5.0)
    assert lc.IslandStateStore(objects).read(0)["state"] == "done"

    class Flaky:
        def __init__(self):
            self.fail, self.objects = 2, {}

        def read(self, key):
            return self.objects.get(key)

        def write(self, key, data):
            if self.fail:
                self.fail -= 1
                raise OSError("store down")
            self.objects[key] = data

    flaky = Flaky()
    states = lc.IslandStateStore(flaky, background=True, stop_poll_s=0.05, log=lambda m: None)
    states.write(1, "failed", exit="config")
    assert states.flush(5.0)
    assert json.loads(flaky.objects[lc.state_key(1)])["state"] == "failed"
    assert states.close(5.0)


def test_a_write_left_behind_at_the_end_is_an_explicit_error(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    lock, logged = _Lock(1), []
    lock.raises = _Unconfirmed("store did not answer")
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path), fence=lock, background=True, stop_poll_s=0.05,
                                 log=logged.append)
    states.write(1, "abandoned", exit="ABANDONED")
    assert states.close(1.0) is False
    error = [m for m in logged if "ERROR" in m]
    assert error and "fleet/trainers/1/state.json (abandoned)" in error[0] and "did not land" in error[0]


def test_a_fence_interrupt_never_ends_the_background_thread(tmp_path):
    from yeto.rl.engine.object_store import DirectoryObjectStore

    lock = _Lock(1)
    lock.raises = KeyboardInterrupt()
    states = lc.IslandStateStore(DirectoryObjectStore(tmp_path), fence=lock, background=True, stop_poll_s=0.05,
                                 log=lambda m: None)
    states.write(0, "running")
    assert not states.flush(0.3)
    lock.raises = None
    assert states.flush(5.0) and states._thread.is_alive()
    assert lc.IslandStateStore(DirectoryObjectStore(tmp_path)).read(0)["state"] == "running"
    assert states.close(5.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
