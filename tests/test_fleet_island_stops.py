"""Per-island stops of an island fleet (tasks 1.4 and 1.5 of unified-trainer-islands):

* ``yeto run up``: an island-scoped trip writes that island's STOP (``fleet/trainers/<k>/STOP``)
  and stops its pools; each island's trainer is judged by its own record and tape
  (``run.trainer_exit``, ``run.trainer_silence``); the run's STOP flag is kept for the
  fleet-wide stops (budget, wall clock, fewer islands than ``fleet.min_islands``);
* the compile: no fixed roster by default (the syncer gets membership options: a lease and
  quorum 1, so the launch's controller drops a given-up island and the rest go on);
  ``fleet.fixed_roster: true`` keeps the old fixed roster;
* the launch's controller: island 1 given up, island 0 goes on (no exit 4) without a fixed
  roster; with one, the fleet ends (exit 4)."""

from __future__ import annotations

import json
import os

import pytest

from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
from tests.test_run_up_fleet import TRAINER_ARGV, _cfg, _write, files  # noqa: F401
from tests.test_run_up_fleet_runtime import Clock, _events, fleet_run  # noqa: F401
from yeto.launch import island_lifecycle as lc
from yeto.launch.fleet import FixedRosterIslandAbandoned, FleetController
from yeto.rl import fleet, up


def _states(run):
    return lc.IslandStateStore(run._objects())


def _pool_stops(run):
    return [c for c in run.calls if "pool-stop" in c]


# -- run up: an island's trip stops that island only ------------------------------------------------

def test_an_island_trip_writes_its_island_stop_and_the_fleet_goes_on(fleet_run):  # noqa: F811
    run = fleet_run()
    assert run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="no group")
    states = _states(run)
    assert states.stop_requested(1)["reason"] == "breaker:run.starvation"
    assert states.stop_requested(1)["by"] == "yeto run up"
    assert states.stop_requested(0) is None
    assert len(_pool_stops(run)) == 1 and "i1-sglang" in " ".join(_pool_stops(run)[0])
    assert run.fleet_rt().active() == [0]
    assert run.fleet_all_stopped() is None  # island 0 runs: the run's STOP flag is not written


def test_every_island_stopped_is_the_fleet_stop(fleet_run):  # noqa: F811
    run = fleet_run()
    run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="x")
    assert run.fleet_rt().trip_island(0, "run.starvation", value=1, limit=1, why="y")
    assert run.fleet_all_stopped() == "breaker:run.starvation"


def test_min_islands_stops_the_fleet_below_it(fleet_run, files):  # noqa: F811
    cfg = _cfg(files)
    cfg["fleet"]["min_islands"] = 2
    run = fleet_run(cfg)
    assert run.fleet_plan().min_islands == 2
    assert run.fleet_all_stopped() is None
    run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="x")
    assert run.fleet_all_stopped() == "breaker:run.starvation"


def test_trainer_exit_is_judged_per_island_from_its_record(fleet_run):  # noqa: F811
    run = fleet_run()
    states = _states(run)
    states.write(0, "running", name="r-l0", source="modal")
    states.write(1, "abandoned", name="r-l1", source="modal", exit="ABANDONED after 1500s",
                 decision={"cause": "infra", "detail": "cluster lost"})
    run.fleet_rt().trainer_check(1000.0)
    assert run.fleet_rt().active() == [0]
    assert states.stop_requested(1)["reason"] == "breaker:run.trainer_exit"
    trips = [e for e in _events(run) if e.get("event") == "rl_breaker" and e.get("name") == "run.trainer_exit"]
    assert len(trips) == 1 and trips[0]["scope"]["island"] == "1"
    assert "abandoned" in trips[0]["detail"]
    run.fleet_rt().trainer_check(1060.0)  # reported once
    assert len([e for e in _events(run) if e.get("name") == "run.trainer_exit"]) == 1
    assert run.fleet_all_stopped() is None


def test_trainer_silence_is_judged_per_island_from_its_tape(fleet_run, tmp_path):  # noqa: F811
    run = fleet_run()
    states = _states(run)
    tapes = {k: tmp_path / f"tape{k}.jsonl" for k in (0, 1)}
    for k, tape in tapes.items():
        tape.write_text("{}\n")
        states.write(k, "running", name=f"r-l{k}", source="modal", tape=str(tape))
    limit = run.fleet_rt().silence_limit_s(1)
    t0 = 1_000_000.0
    run.fleet_rt().trainer_check(t0)
    with open(tapes[0], "a") as fh:  # island 0 keeps writing
        fh.write("{}\n")
    os.utime(tapes[0], (t0 + limit + 5, t0 + limit + 5))
    run.fleet_rt().trainer_check(t0 + limit + 10)
    assert run.fleet_rt().active() == [0]
    assert states.stop_requested(1)["reason"] == "breaker:run.trainer_silence"
    assert states.stop_requested(0) is None


def test_the_global_silence_is_off_for_a_fleet(fleet_run, monkeypatch):  # noqa: F811
    run = fleet_run()
    run.state["procs"] = {"trainer": {"pid": os.getpid(), "log": "/nonexistent"}}
    monkeypatch.setattr(up.Run, "silence", lambda self, now: "trainer: log / tape silent 99 min (limit 1 min)")
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    assert run.breach() is None  # a fleet's islands are judged one by one


# -- the compile: no fixed roster by default ---------------------------------------------------------

def _argv(cfg):
    return cfg["trainer"]["argv"]


def test_a_fleet_has_no_fixed_roster_by_default(files):  # noqa: F811
    cfg = up.load_config(_write(files, _cfg(files)))
    argv = _argv(cfg)
    assert argv[argv.index("--rl-sync-island-lease-s") + 1] == "150"
    assert argv[argv.index("--rl-sync-quorum") + 1] == "1"
    plan = fleet.plan_of(cfg)
    assert not plan.fixed_roster and plan.min_islands == 1
    assert any("no fixed roster" in line for line in plan.lines())


def test_fixed_roster_true_keeps_the_old_launch(files):  # noqa: F811
    raw = _cfg(files)
    raw["fleet"]["fixed_roster"] = True
    cfg = up.load_config(_write(files, raw))
    assert not any(a.startswith("--rl-sync-") for a in _argv(cfg) if a not in TRAINER_ARGV)
    assert fleet.plan_of(cfg).fixed_roster


def test_the_trainers_own_membership_options_are_kept(files):  # noqa: F811
    raw = _cfg(files)
    raw["fleet"]["trainer_argv"] = [*raw["fleet"]["trainer_argv"], "--rl-sync-quorum-theta", "0.5"]
    argv = _argv(up.load_config(_write(files, raw)))
    assert "--rl-sync-island-lease-s" not in argv and "--rl-sync-quorum" not in argv


@pytest.mark.parametrize("edit, match", [
    ({"fixed_roster": "yes"}, "fleet.fixed_roster must be true or false"),
    ({"min_islands": 0}, "fleet.min_islands must be an integer >= 1"),
    ({"min_islands": 3}, r"fleet.min_islands 3 > the fleet's 2 island"),
])
def test_bad_roster_options_are_refused(files, edit, match):  # noqa: F811
    raw = _cfg(files)
    raw["fleet"].update(edit)
    with pytest.raises(ValueError, match=match):
        up.load_config(_write(files, raw))


def test_a_plan_written_before_the_option_was_a_fixed_roster(files):  # noqa: F811
    doc = fleet.plan_of(up.load_config(_write(files, _cfg(files)))).to_json()
    assert doc["roster"] == {"fixed": False, "min_islands": 1}
    doc.pop("roster")
    assert fleet.FleetPlan.from_json(doc).fixed_roster


def test_the_compiled_launch_runs_an_elastic_controller(files, tmp_path, monkeypatch):  # noqa: F811
    """What the launch makes of the compiled argv: an island pool (no fixed roster) by default."""
    from yeto.cli import parse_args
    from yeto.launch.fleet import controller_fixed_roster
    from yeto.launch.syncer import island_pool

    for fixed in (False, True):
        raw = _cfg(files)
        raw["fleet"]["fixed_roster"] = fixed
        args = parse_args(_argv(up.load_config(_write(files, raw))))
        assert island_pool(args) is not fixed
        assert controller_fixed_roster(args) is fixed


# -- the launch's controller: one island given up --------------------------------------------------

def _ctl(ops, **kw):
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(learners={"l0": ("t0", 1), "l1": ("t1", 2)}, syncer=(SYNCER, "ts", 1), sky_ops=ops,
                           poll_interval=30, recover_timeout=0, thread_cls=ImmediateThread, **kw)


def test_without_a_fixed_roster_the_other_islands_go_on():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING, RUNNING, RUNNING, SUCCEEDED]
    ops.status_seq["l1"] = [FAILED]
    exits = _ctl(ops, fixed_roster=False, elastic=True).run()  # what the default compile launches
    assert exits["l0"].endswith("SUCCEEDED") and exits["l1"].startswith("ABANDONED")


def test_with_a_fixed_roster_one_island_given_up_ends_the_fleet():
    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.status_seq["l1"] = [FAILED]
    with pytest.raises(FixedRosterIslandAbandoned, match="fixed-roster"):
        _ctl(ops, fixed_roster=True).run()
