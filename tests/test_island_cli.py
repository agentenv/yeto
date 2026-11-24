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
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"run": "r"}))
    assert yeto_main(["island", "status", "--config", str(cfg)]) == 2  # no store
    assert yeto_main(["island", "status", "--store", "modal-volume://v/x"]) == 2  # no object API here


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
    trips = [e for e in _events(run) if e.get("event") == "rl_breaker" and e.get("name") == "run.trainer_exit"]
    assert len(trips) == 1 and "yeto island stop" in trips[0]["detail"]
    assert run.fleet_all_stopped() is None  # island 0 goes on
