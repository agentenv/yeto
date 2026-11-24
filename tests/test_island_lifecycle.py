"""The unified trainer island lifecycle (``yeto.launch.island_lifecycle``): the state
machine, the exit decision table, the island records in the run store and each island's
own STOP, with the fleet controller writing and honoring them. No cloud."""

from __future__ import annotations

import json
import sys

import pytest

from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
