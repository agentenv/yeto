"""``yeto island {status,stop,start}`` (task 2.4 of unified-trainer-islands): the requests it
writes in the run store, its dry run, the fleet controller ordering a stopped island again
on its START, and ``yeto run up`` stopping the pools of an island stopped by hand. No cloud."""

from __future__ import annotations

import json

from test_controller import RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
from tests.test_run_up_fleet import _cfg, files  # noqa: F401
from tests.test_run_up_fleet_runtime import Clock, _events, fleet_run  # noqa: F401
from yeto.cli import main as yeto_main
from yeto.launch import island_lifecycle as lc
from yeto.launch.fleet import FleetController
from yeto.rl.engine.object_store import DirectoryObjectStore


def _states(path):
    return lc.IslandStateStore(DirectoryObjectStore(path))


def _seed(path):
    states = _states(path)
    states.write(0, "running", name="r-l0-modal", source="modal", incarnation=0, handle={"call_id": "fc-0"})
    states.write(1, "running", name="r-l1-modal", source="modal", incarnation=2, relaunches=2,
                 handle={"call_id": "fc-9"})
    return states


def test_dry_run_writes_nothing(tmp_path, capsys):
    _seed(tmp_path)
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path), "--dry-run", "--reason", "maintenance"]) == 0
    out = capsys.readouterr().out
    assert "dry run: would write" in out and "fleet/trainers/1/STOP" in out and "maintenance" in out
    assert "drains and releases this island only" in out
    assert not (tmp_path / "fleet/trainers/1/STOP").exists()
    assert yeto_main(["island", "start", "1", "--store", str(tmp_path), "--dry-run"]) == 2  # it runs: no start
    _states(tmp_path).write(1, "done", decision={"cause": "stopped", "action": "stop", "state": "done"})
    assert yeto_main(["island", "start", "1", "--store", str(tmp_path), "--dry-run"]) == 0
    assert not (tmp_path / "fleet/trainers/1/START").exists()


def test_stop_then_start_write_the_requests(tmp_path, capsys):
    states = _seed(tmp_path)
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path), "--reason", "bad node"]) == 0
    stop = states.stop_requested(1)
    assert stop["reason"] == "bad node" and stop["by"] == "yeto island stop"
    assert states.stop_requested(0) is None
    assert yeto_main(["island", "start", "1", "--store", str(tmp_path)]) == 0
    assert states.stop_requested(1) is None
    assert states.start_requested(1)["by"] == "yeto island start"
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path)]) == 0
    assert states.start_requested(1) is None  # a stop drops a pending start


def test_status_lists_every_island(tmp_path, capsys):
    states = _seed(tmp_path)
    states.request_stop(1, "operator", by="yeto island stop")
    assert yeto_main(["island", "status", "--store", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "island 0: running" in out and "island 1: running" in out and "[fc-9]" in out
    assert "incarnation 2, relaunches 2" in out and "STOP pending (operator, by yeto island stop)" in out
    assert yeto_main(["island", "status", "--store", str(tmp_path), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["1"]["stop_request"]["reason"] == "operator" and doc["0"]["stop_request"] is None


def test_a_run_config_names_the_store(tmp_path, capsys):
    _seed(tmp_path / "store")
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"run": "r", "store": str(tmp_path / "store"), "env": {}}))
    assert yeto_main(["island", "stop", "0", "--config", str(cfg)]) == 0
    assert _states(tmp_path / "store").stop_requested(0)["by"] == "yeto island stop"


def test_bad_input(tmp_path, capsys):
    _seed(tmp_path)
    assert yeto_main(["island", "stop", "7", "--store", str(tmp_path)]) == 2  # no such island
    assert yeto_main(["island", "stop", "-1", "--store", str(tmp_path)]) == 2  # no negative id
    assert not (tmp_path / "fleet/trainers/7/STOP").exists() and not (tmp_path / "fleet/trainers/-1").exists()
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"run": "r"}))
    assert yeto_main(["island", "status", "--config", str(cfg)]) == 2  # no store
    assert yeto_main(["island", "status", "--store", "modal-volume://v/x"]) == 2  # no object API here


def test_an_island_without_a_record_is_refused_in_an_empty_store(tmp_path, capsys):
    assert yeto_main(["island", "stop", "99", "--store", str(tmp_path)]) == 2
    assert yeto_main(["island", "stop", "0", "--store", str(tmp_path)]) == 2
    assert not (tmp_path / "fleet").exists()
    assert "no record" in capsys.readouterr().err


def test_only_a_fleet_trainer_island_is_driven(tmp_path, capsys):
    states = _states(tmp_path)
    states.write(0, "running", name="r-l0", source="modal", role="sampler")
    states.write(1, "running", name="r-l1", source="modal", role="trainer", mode="s1")
    assert yeto_main(["island", "stop", "0", "--store", str(tmp_path)]) == 2
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "sampler island" in err and "trainer scheduler's (s1)" in err
    assert states.stop_requested(0) is None and states.stop_requested(1) is None


def test_stop_names_the_island_and_its_controller(tmp_path, capsys):
    _seed(tmp_path)
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "island 1 = r-l1-modal on modal, running, supervised by" in out and "PID" in out


def test_a_fixed_roster_stop_is_refused_without_end_fleet(tmp_path, capsys):
    states = _seed(tmp_path)
    states.write(1, fixed_roster=True)
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path)]) == 2
    assert "fixed roster" in capsys.readouterr().err and states.stop_requested(1) is None
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path), "--end-fleet"]) == 0
    assert "WARNING: a fixed roster" in capsys.readouterr().err and states.stop_requested(1) is not None


def test_a_legacy_plan_is_a_fixed_roster(tmp_path, capsys):
    _seed(tmp_path)
    (tmp_path / "fleet/plan.json").write_text(json.dumps({"run": "r", "store": str(tmp_path), "dispense": "fleet",
                                                          "islands": []}))
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path)]) == 2
    assert "fixed roster" in capsys.readouterr().err


def test_a_record_that_does_not_read_writes_nothing(tmp_path, monkeypatch, capsys):
    _seed(tmp_path)

    def broken(self, islands=None, *, strict=False):
        raise lc.IslandRecordsUnreadable("island 1: its record did not read (OSError: timeout)")

    monkeypatch.setattr(lc.IslandStateStore, "read_all", broken)
    assert yeto_main(["island", "stop", "1", "--store", str(tmp_path)]) == 1
    assert not (tmp_path / "fleet/trainers/1/STOP").exists()


def test_status_shows_an_idle_s1_island(tmp_path, capsys):
    _states(tmp_path).write(0, "done", name="r-l0", source="modal", mode="s1", idle=True)
    assert yeto_main(["island", "status", "--store", str(tmp_path)]) == 0
    assert "island 0: idle (s1" in capsys.readouterr().out


# -- the fleet controller orders a stopped island again on its START -----------------------------------

def test_a_stopped_island_is_ordered_again_on_its_start(tmp_path):
    states = _states(tmp_path)
    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [RUNNING] * 12 + [SUCCEEDED]
    ops.status_seq["r-l1-modal"] = [RUNNING]
    ops.status_seq[SYNCER] = [RUNNING]
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    ctl = FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                          elastic=True, recorder=lc.FleetRecorder(states, refs))
    ops.relaunch_results["r-l1-modal"] = [7]
    states.request_stop(1, "maintenance", by="yeto island stop")
    real_sleep, polls = ops.sleep, []

    def sleep(seconds):  # the operator starts island 1 again after a few polls
        polls.append(seconds)
        if len(polls) == 3:
            states.clear_stop(1)
            states.request_start(1, "back", by="yeto island start")
        if len(polls) == 5:
            ops.status_seq["r-l1-modal"] = [RUNNING] * 3 + [SUCCEEDED]
        real_sleep(seconds)

    ops.sleep = sleep
    exits = ctl.run()
    assert ops.down_calls[:1] == ["r-l1-modal"]  # stopped first
    assert ops.relaunch_calls == ["r-l1-modal"]  # then ordered again, once
    assert exits["r-l1-modal"].endswith("SUCCEEDED") and exits["r-l0-modal"].endswith("SUCCEEDED")
    record = states.read(1)
    moves = [h["to"] for h in record["history"]]
    assert moves[:4] == ["running", "draining", "done", "ordering"] and moves[-1] == "done"
    assert record["incarnation"] == 1 and record["start"]["reason"] == "back"


def test_a_start_that_reaches_a_draining_island_is_kept_until_it_is_done(tmp_path, capsys):
    """The race ``yeto island start`` cannot see: the controller acts on the STOP (the island drains)
    before its record says draining, the CLI then swaps the STOP for a START. The START is kept, not
    consumed as ignored, and the island is ordered again once its drain is done (it is not left done)."""
    states = _states(tmp_path)
    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [RUNNING] * 14 + [SUCCEEDED]
    ops.status_seq["r-l1-modal"] = [RUNNING]  # still training: the drain takes its whole grace
    ops.status_seq[SYNCER] = [RUNNING]
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    ctl = FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                          sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                          elastic=True, recorder=lc.FleetRecorder(states, refs))
    ops.relaunch_results["r-l1-modal"] = [7]
    states.request_stop(1, "maintenance", by="yeto island stop")
    real_sleep, polls, seen = ops.sleep, [], {}

    def sleep(seconds):
        polls.append(seconds)
        if len(polls) == 1:  # the STOP is acted on: island 1 drains
            seen["state"] = ctl.learners["r-l1-modal"].get("draining")
            states.clear_stop(1)
            states.request_start(1, "back", by="yeto island start")
        if len(polls) == 2:
            seen["kept"] = states.start_requested(1) is not None
        if len(polls) == 7:
            ops.status_seq["r-l1-modal"] = [RUNNING] * 3 + [SUCCEEDED]
        real_sleep(seconds)

    ops.sleep = sleep
    exits = ctl.run()
    assert seen == {"state": True, "kept": True}  # draining when the START came, and it was kept
    assert ops.down_calls[:1] == ["r-l1-modal"]  # the drain ended: released first
    assert ops.relaunch_calls == ["r-l1-modal"]  # then ordered again, once
    assert exits["r-l1-modal"].endswith("SUCCEEDED")
    record = states.read(1)
    moves = [h["to"] for h in record["history"]]
    assert moves[:4] == ["running", "draining", "done", "ordering"] and moves[-1] == "done"
    assert record["start"]["reason"] == "back" and record["start"]["outcome"] == "ordered"
    assert states.start_requested(1) is None  # consumed once acted on
    err = capsys.readouterr().err
    assert err.count("island START (back) kept: the island drains") == 1 and "ignored" not in err


def test_start_is_refused_while_the_record_drains(tmp_path, capsys):
    """The controller acts on the STOP already: ``yeto island start`` refuses (exit 2) rather than
    write a START the drain would lose; once the island is done it is started."""
    states = _seed(tmp_path)
    states.request_stop(1, "maintenance", by="yeto island stop")
    states.write(1, "draining", name="r-l1-modal", source="modal")
    assert yeto_main(["island", "start", "1", "--store", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "is draining on its island STOP" in err and "nothing written" in err
    assert states.start_requested(1) is None and states.stop_requested(1) is not None  # untouched
    _stopped_record(states)
    assert yeto_main(["island", "start", "1", "--store", str(tmp_path)]) == 0
    assert states.start_requested(1)["by"] == "yeto island start"


# -- yeto run up stops the pools of an island stopped by hand -------------------------------------------

def test_run_up_stops_the_pools_of_an_island_stopped_by_hand(fleet_run):  # noqa: F811
    run = fleet_run()
    states = lc.IslandStateStore(run._objects())
    states.write(0, "running", name="r-l0", source="modal")
    states.request_stop(1, "maintenance", by="yeto island stop")
    states.write(1, "done", name="r-l1", source="modal", stop={"reason": "maintenance", "by": "yeto island stop"},
                 decision={"cause": "stopped", "action": "stop", "state": "done", "detail": "maintenance"})
    run.fleet_rt().trainer_check(1000.0)
    assert run.fleet_rt().active() == [0]
    assert [c for c in run.calls if "pool-stop" in c] and "i1-sglang" in " ".join(
        [c for c in run.calls if "pool-stop" in c][0])
    assert states.stop_requested(1)["by"] == "yeto island stop"  # the request is kept as written
    # its own event, not a breaker trip (whatever run.trainer_exit's action is, its pools stop)
    assert not [e for e in _events(run) if e.get("event") == "rl_breaker"]
    stops = [e for e in _events(run) if e.get("event") == "fleet_island_operator_stop"]
    assert len(stops) == 1 and stops[0]["island"] == 1 and stops[0]["by"] == "yeto island stop"
    assert run.fleet_all_stopped() is None  # island 0 goes on
    run.fleet_rt().trainer_check(1100.0)  # reported once
    assert len([e for e in _events(run) if e.get("event") == "fleet_island_operator_stop"]) == 1
    # a restarted run up keeps it stopped, as an operator stop
    run._fleet_rt = None
    assert run.fleet_rt().island_stopped(1) == "operator:maintenance" and run.fleet_rt().active() == [0]


def _ctl(states, ops, **kw):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    refs = {"r-l0-modal": lc.IslandRef(0, "modal"), "r-l1-modal": lc.IslandRef(1, "modal")}
    return FleetController(learners={"r-l0-modal": ("t", 1), "r-l1-modal": ("t", 2)}, syncer=(SYNCER, "t", 1),
                           sky_ops=ops, poll_interval=30, recover_timeout=600, thread_cls=ImmediateThread,
                           elastic=True, recorder=lc.FleetRecorder(states, refs), **kw)


def _stopped_record(states, k=1):
    states.write(k, "done", name=f"r-l{k}-modal", source="modal", exit="STOPPED (island STOP: x)",
                 decision={"cause": "stopped", "action": "stop", "state": "done", "detail": "x"})


def test_a_fenced_controller_acts_on_no_start(tmp_path):
    import pytest

    from yeto.launch.island_lifecycle import IslandRecordsFenced

    states = _states(tmp_path)
    _stopped_record(states)
    ops = FakeOps()
    ctl = _ctl(states, ops)
    rec = ctl.learners["r-l1-modal"]
    rec["state"] = "done"
    states.request_start(1, "back", by="yeto island start")
    states.fenced = RuntimeError("lock taken by token 9")
    with pytest.raises(IslandRecordsFenced):
        ctl._island_start(rec)
    assert rec["state"] == "done" and ops.relaunch_calls == []
    assert states.start_requested(1) is not None  # left for the launcher that holds the records


def test_start_is_judged_by_the_record_after_a_restart(tmp_path):
    """A restarted (or adopting) controller never stopped the island itself: the record decides."""
    states = _states(tmp_path)
    _stopped_record(states)
    ops = FakeOps()
    ctl = _ctl(states, ops)
    states.read_all([0, 1])  # what the launcher read at its start
    rec = ctl.learners["r-l1-modal"]
    rec["state"] = "done"  # no in-memory "stopped" mark
    states.request_start(1, "back", by="yeto island start")
    assert ctl._island_start(rec)
    assert rec["state"] == "recovering"
    record = states.read(1)
    assert record["state"] == "ordering" and record["start"]["outcome"] == "ordered"
    assert states.start_requested(1) is None  # consumed
    assert not ctl._island_start(rec)  # once


def test_a_start_whose_delete_did_not_land_is_not_acted_on_twice(tmp_path, monkeypatch):
    states = _states(tmp_path)
    _stopped_record(states)
    ctl = _ctl(states, FakeOps())
    rec = ctl.learners["r-l1-modal"]
    rec["state"] = "done"
    states.read_all([1])
    states.request_start(1, "back", by="yeto island start")
    monkeypatch.setattr(lc.IslandStateStore, "clear_start", lambda self, k: False)
    assert ctl._island_start(rec)
    rec["state"] = "done"  # stopped again meanwhile; the same request still there
    _stopped_record(states)
    assert not ctl._island_start(rec)


def test_a_start_of_a_running_island_orders_nothing_and_is_consumed(tmp_path, capsys):
    states = _states(tmp_path)
    states.write(1, "running", name="r-l1-modal", source="modal")
    ctl = _ctl(states, FakeOps())
    states.request_start(1, "back", by="yeto island start")
    assert not ctl._island_start(ctl.learners["r-l1-modal"])
    assert states.start_requested(1) is None and states.read(1)["start"]["outcome"] == "ignored"
    assert "ignored" in capsys.readouterr().err


def test_the_controller_records_its_roster(tmp_path):
    states = _states(tmp_path)
    ops = FakeOps()
    ops.status_seq["r-l0-modal"] = [SUCCEEDED]
    ops.status_seq["r-l1-modal"] = [SUCCEEDED]
    _ctl(states, ops).run()
    assert states.read(0)["fixed_roster"] is False


def test_a_start_waits_while_the_lock_is_not_confirmed(tmp_path):
    class Unconfirmed:
        token = 3

        def guard(self):
            raise RuntimeError("store did not answer")  # no ``by``: not lost, not confirmed

    states = _states(tmp_path)
    _stopped_record(states)
    states.read_all([1])
    ctl = _ctl(states, FakeOps())
    rec = ctl.learners["r-l1-modal"]
    rec["state"] = "done"
    states.request_start(1, "back", by="yeto island start")
    states.fence = Unconfirmed()
    assert not ctl._island_start(rec)
    assert rec["state"] == "done" and states.start_requested(1) is not None  # tried again next poll
