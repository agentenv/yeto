"""Per-island stops of an island fleet (tasks 1.4 and 1.5 of unified-trainer-islands):

* ``yeto run up``: an island-scoped trip writes that island's STOP (``fleet/trainers/<k>/STOP``)
  and stops its pools; each island's trainer is judged by its own record and tape
  (``run.trainer_exit``, ``run.trainer_silence``); the run's STOP flag is kept for the
  fleet-wide stops (budget, wall clock, fewer islands than ``fleet.min_islands``);
* the compile: no fixed roster by default (the syncer gets one membership option, the island
  lease ``--rl-sync-island-lease-s``, so the launch's controller drops a given-up island and the
  rest go on); ``fleet.fixed_roster: true`` keeps the old fixed roster;
* the launch's controller: island 1 given up, island 0 goes on (no exit 4) without a fixed
  roster; with one, the fleet ends (exit 4)."""

from __future__ import annotations

import json
import os

import pytest

from test_controller import FAILED, RUNNING, SUCCEEDED, SYNCER, FakeOps, ImmediateThread
from tests.test_run_up_fleet import TRAINER_ARGV, _cfg, _write, files  # noqa: F401
from tests.test_run_up_fleet_runtime import Clock, _events, fleet_run  # noqa: F401
from yeto.launch import fleet as lf
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


def _silence_breach(run, monkeypatch):
    run.state["procs"] = {"trainer": {"pid": os.getpid(), "log": "/nonexistent"}}
    monkeypatch.setattr(up.Run, "silence", lambda self, now: "trainer: log / tape silent 99 min (limit 1 min)")
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    run._quiet_s = 99 * 60
    return run.breach()


def test_the_global_silence_is_off_once_every_island_has_a_tape(fleet_run, monkeypatch, tmp_path):  # noqa: F811
    run = fleet_run()
    for k in (0, 1):
        tape = tmp_path / f"t{k}.jsonl"
        tape.write_text("{}\n")
        _states(run).write(k, "running", name=f"r-l{k}", source="modal", tape=str(tape))
    assert run.fleet_rt().judges_every_trainer(1000.0)
    assert _silence_breach(run, monkeypatch) is None  # each island is judged by its own tape


def test_a_recorded_tape_that_never_stats_keeps_the_global_silence(fleet_run, monkeypatch, tmp_path):  # noqa: F811
    """A tape path in the record is not enough: while one does not stat (not written yet, on another
    machine), the island's own silence cannot judge it, so the run's silence still covers the fleet."""
    run = fleet_run()
    present = tmp_path / "t0.jsonl"
    present.write_text("{}\n")
    _states(run).write(0, "running", name="r-l0", source="modal", tape=str(present))
    _states(run).write(1, "running", name="r-l1", source="modal", tape=str(tmp_path / "never.jsonl"))
    assert not run.fleet_rt().judges_every_trainer(1000.0)
    (tmp_path / "never.jsonl").write_text("{}\n")  # once it stats, the islands judge themselves
    run.fleet_rt()._records_read = None
    assert run.fleet_rt().judges_every_trainer(1000.0)
    (tmp_path / "never.jsonl").unlink()  # gone again: back to the run's silence
    run.fleet_rt()._records_read = None
    assert not run.fleet_rt().judges_every_trainer(1000.0)
    assert _silence_breach(run, monkeypatch).startswith("trainer: log / tape silent")


def test_an_island_without_a_tape_keeps_the_global_silence(fleet_run, monkeypatch, tmp_path):  # noqa: F811
    """A sky / Verda / fetchable SSH island has no echoed tape: the run's silence over the launcher log
    still covers the fleet (and so does a launch without island records)."""
    run = fleet_run()
    assert not run.fleet_rt().judges_every_trainer(1000.0)  # no records at all
    _states(run).write(0, "running", name="r-l0", source="modal", tape=str(tmp_path / "t0.jsonl"))
    _states(run).write(1, "running", name="r-l1", source="sky")  # no tape
    run.fleet_rt()._records_read = None
    assert not run.fleet_rt().judges_every_trainer(1000.0)
    assert _silence_breach(run, monkeypatch).startswith("trainer: log / tape silent")


def test_a_tape_that_does_not_stat_is_unknown_not_quiet(fleet_run, tmp_path):  # noqa: F811
    run = fleet_run()
    states = _states(run)
    missing = tmp_path / "elsewhere.jsonl"  # on another machine, or not written yet
    states.write(1, "running", name="r-l1", source="modal", tape=str(missing))
    limit = run.fleet_rt().silence_limit_s(1)
    t0 = 1_000_000.0
    for step in range(6):
        run.fleet_rt().trainer_check(t0 + step * (limit / 2 + 10))
    assert states.stop_requested(1) is None and run.fleet_rt().active() == [0, 1]
    missing.write_text("{}\n")  # readable from now on: its quiet time starts here
    now = t0 + 6 * (limit / 2 + 10)
    run.fleet_rt()._records_read = None
    run.fleet_rt().trainer_check(now)
    run.fleet_rt()._records_read = None
    run.fleet_rt().trainer_check(now + limit / 2)
    assert states.stop_requested(1) is None
    run.fleet_rt()._records_read = None
    run.fleet_rt().trainer_check(now + limit + 1)
    assert states.stop_requested(1)["reason"] == "breaker:run.trainer_silence"


def _fixed_run(fleet_run, files):  # noqa: F811
    raw = _cfg(files)
    raw["fleet"]["fixed_roster"] = True
    return fleet_run(raw)


def test_a_fixed_roster_writes_no_island_stop(fleet_run, files, capsys):  # noqa: F811
    """fleet.fixed_roster: true is the old path: the trip stops the island's pools and its trainer
    starves; an island STOP would end the whole fixed-roster launch (exit 4)."""
    run = _fixed_run(fleet_run, files)
    assert run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="no group")
    assert _states(run).stop_requested(1) is None
    assert len(_pool_stops(run)) == 1
    assert any("a fixed roster (fleet.fixed_roster)" in line and "starves" in line
               for line in capsys.readouterr().out.splitlines())
    assert run.fleet_all_stopped() is None


def test_a_fixed_roster_is_not_judged_per_island(fleet_run, files, monkeypatch, tmp_path):  # noqa: F811
    run = _fixed_run(fleet_run, files)
    states = _states(run)
    for k in (0, 1):
        states.write(k, "running", name=f"r-l{k}", source="modal", tape=str(tmp_path / f"t{k}.jsonl"))
    states.write(1, "abandoned", name="r-l1", source="modal")
    run.fleet_rt().trainer_check(1000.0)
    assert states.stop_requested(1) is None and run.fleet_rt().active() == [0, 1]
    assert not run.fleet_rt().judges_every_trainer(1000.0)
    assert _silence_breach(run, monkeypatch).startswith("trainer: log / tape silent")  # the old silence


def test_min_islands_stop_has_its_own_event(fleet_run, files, monkeypatch):  # noqa: F811
    cfg = _cfg(files)
    cfg["fleet"]["min_islands"] = 2
    run = fleet_run(cfg)
    run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="x")
    flagged = []
    monkeypatch.setattr(run, "flag_stop", lambda name, key, event, why, **kw: flagged.append((event, why)))
    run.fleet_stop(run.fleet_all_stopped())
    assert flagged == [("fleet_min_islands_stopped", "1 island(s) run, fewer than fleet.min_islands 2 "
                                                     "(island 1 (west): breaker:run.starvation)")]


# -- re-up: the stored plan's roster -----------------------------------------------------------------

# -- an island's pool stop on the store control plane names the run store and the pool ------------

def _stop_island_1(run, how):
    if how == "trip":
        assert run.fleet_rt().trip_island(1, "run.starvation", value=1, limit=1, why="no group")
    else:  # stopped by hand: its record is done on an island STOP that run up did not write
        states = _states(run)
        states.write(0, "running", name="r-l0", source="modal")
        states.request_stop(1, "maintenance", by="yeto island stop")
        states.write(1, "done", name="r-l1", source="modal", stop={"reason": "maintenance", "by": "yeto island stop"},
                     decision={"cause": "stopped", "action": "stop", "state": "done", "detail": "maintenance"})
        run.fleet_rt().trainer_check(1000.0)


@pytest.mark.parametrize("how,fixed", [("trip", False), ("operator", False), ("trip", True)])
def test_an_island_pool_stop_on_the_store_control_plane_names_the_store_and_the_pool(
        fleet_run, files, how, fixed):  # noqa: F811
    """Run.pool_stop: a pool whose controller is gone is still released from the store's state."""
    raw = _cfg(files)
    raw["fleet"]["fixed_roster"] = fixed
    raw.update(controller="in_process", rollout={"control_plane": "store"})
    run = fleet_run(raw)
    _stop_island_1(run, how)
    stops = _pool_stops(run)
    (i,) = [j for j, ref in enumerate(run.pools) if ref.name == "i1-sglang"]
    assert stops == [run.nc("pool-stop", "--config", run.pools[i].config,
                            "--control-store", str(run.cfg["store"]), "--pool-name", "i1-sglang")]


@pytest.mark.parametrize("how,fixed", [("trip", False), ("operator", False), ("trip", True)])
def test_an_island_pool_stop_on_the_local_control_plane_is_as_before(fleet_run, files, how, fixed):  # noqa: F811
    raw = _cfg(files)
    raw["fleet"]["fixed_roster"] = fixed
    run = fleet_run(raw)
    _stop_island_1(run, how)
    (i,) = [j for j, ref in enumerate(run.pools) if ref.name == "i1-sglang"]
    assert _pool_stops(run) == [run.nc("pool-stop", "--config", run.pools[i].config)]


def _stored_plan(run, **roster):
    doc = run.fleet_plan().to_json()
    if roster.get("legacy"):
        doc.pop("roster")
    run._objects().write(fleet.PLAN_OBJECT, json.dumps(doc).encode())


def test_re_up_of_a_legacy_fixed_roster_fleet_with_running_islands_is_refused(fleet_run, capsys):  # noqa: F811
    run = fleet_run()
    _stored_plan(run, legacy=True)
    _states(run).write(0, "running", name="r-l0", source="modal")
    assert not run.write_fleet_plan()
    out = capsys.readouterr().out
    assert "a fixed roster (a plan written before fleet.fixed_roster)" in out
    assert "fleet.fixed_roster: true" in out and "Nothing ordered" in out
    assert "roster" not in json.loads(run._objects().read(fleet.PLAN_OBJECT))  # not written over


def test_re_up_keeping_the_stored_roster_goes_on(fleet_run, files):  # noqa: F811
    run = _fixed_run(fleet_run, files)
    _stored_plan(run, legacy=True)
    _states(run).write(0, "running", name="r-l0", source="modal")
    assert run.write_fleet_plan()
    assert json.loads(run._objects().read(fleet.PLAN_OBJECT))["roster"]["fixed"] is True


def test_re_up_with_another_roster_once_nothing_runs_goes_on(fleet_run):  # noqa: F811
    run = fleet_run()
    _stored_plan(run, legacy=True)
    _states(run).write(0, "done", name="r-l0", source="modal")
    assert run.write_fleet_plan()
    assert json.loads(run._objects().read(fleet.PLAN_OBJECT))["roster"]["fixed"] is False


def _stored_pool_plan(run, lease_s):
    doc = run.fleet_plan().to_json()
    doc["roster"]["lease_s"] = lease_s
    run._objects().write(fleet.PLAN_OBJECT, json.dumps(doc).encode())


def test_re_up_with_another_island_lease_while_islands_run_is_refused(fleet_run, capsys):  # noqa: F811
    """Adopted trainers still heartbeat at the stored lease's pace: a shorter lease could drop them."""
    run = fleet_run()
    assert run.fleet_plan().island_lease_s == 150
    _stored_pool_plan(run, 600)
    _states(run).write(0, "running", name="r-l0", source="modal")
    assert not run.write_fleet_plan()
    out = capsys.readouterr().out
    assert "the syncer island pool (lease 600 s)" in out and "the syncer island pool (lease 150 s)" in out
    assert "the syncer lease kept at 600 s where it is set" in out and "Nothing ordered" in out
    assert "--rl-sync-island-lease-s in fleet.trainer_argv; not both" in out  # a lease from the argv: keep it there
    assert json.loads(run._objects().read(fleet.PLAN_OBJECT))["roster"]["lease_s"] == 600  # not written over


def test_re_up_with_the_stored_island_lease_goes_on(fleet_run):  # noqa: F811
    run = fleet_run()
    _stored_pool_plan(run, 150)
    _states(run).write(0, "running", name="r-l0", source="modal")
    assert run.write_fleet_plan()


def test_re_up_with_another_island_lease_once_nothing_runs_goes_on(fleet_run):  # noqa: F811
    run = fleet_run()
    _stored_pool_plan(run, 600)
    _states(run).write(0, "done", name="r-l0", source="modal")
    assert run.write_fleet_plan()
    assert json.loads(run._objects().read(fleet.PLAN_OBJECT))["roster"]["lease_s"] == 150


def test_re_up_refuses_when_the_stored_plan_does_not_read(fleet_run, monkeypatch, capsys):  # noqa: F811
    run = fleet_run()

    class Broken:
        def read(self, key):
            raise OSError("store down")

    monkeypatch.setattr(up.Run, "_objects", lambda self: Broken())
    assert not run.write_fleet_plan()
    assert "did not read" in capsys.readouterr().out


# -- the compile: no fixed roster by default ---------------------------------------------------------

def _argv(cfg):
    return cfg["trainer"]["argv"]


def test_a_fleet_has_no_fixed_roster_by_default(files):  # noqa: F811
    cfg = up.load_config(_write(files, _cfg(files)))
    argv = _argv(cfg)
    assert argv[argv.index("--rl-sync-island-lease-s") + 1] == "150"
    assert "--rl-sync-quorum" not in argv  # the async rounds' quorum 1 stays as resolved
    plan = fleet.plan_of(cfg)
    assert not plan.fixed_roster and plan.min_islands == 1 and plan.island_lease_s == 150
    assert any("no fixed roster (syncer island pool, lease 150 s)" in line for line in plan.lines())


def test_the_default_launch_passes_the_launchers_island_pool_check(files):  # noqa: F811
    from yeto.launch import rl_args
    from yeto.launch.syncer import island_pool
    from yeto.rl.engine.sync_route import apply_sync_route
    from yeto.rl.up_consistency import trainer_args

    args = trainer_args(_argv(up.load_config(_write(files, _cfg(files)))))
    apply_sync_route(args)
    rl_args._apply_rl_sync_mode(args)
    rl_args._check_island_pool(args, num_islands=2)  # raises when the syncer would refuse the pool
    assert island_pool(args) and args.rl_sync_quorum == 1 and args.rl_sync_island_lease_s == 150


def _compiled(files, extra=(), **fleet_keys):  # noqa: F811
    raw = _cfg(files)
    raw["fleet"]["trainer_argv"] = [*raw["fleet"]["trainer_argv"], *extra]
    raw["fleet"].update(fleet_keys)
    return up.load_config(_write(files, raw))


@pytest.mark.parametrize("extra, reason", [
    (["--rl-sync-mode", "strict", "--rl-sync-full-parameter-rounds", "async"],
     "--rl-sync-mode strict: every round waits for every island"),
    (["--syncer-shards", "2"], "--syncer-shards > 1"),
    (["--rl-sync-bcast-delta-full-every", "4"], "--rl-sync-bcast-delta-full-every"),
    (["--rl-sync-quorum", "2", "--rl-sync-grace-ms", "0", "--rl-sync-max-base-lag", "0"],
     "rounds that wait for every island (--rl-sync-quorum 2, no grace window, no base lag)"),
])
def test_a_fleet_the_island_pool_does_not_fit_keeps_its_fixed_roster(files, capsys, extra, reason):  # noqa: F811
    """A strict roster is never turned into quorum 1, and a fleet that launched before (syncer shards,
    broadcast deltas, an explicit full quorum) still launches: a fixed roster, with a WARNING."""
    cfg = _compiled(files, extra)
    argv = _argv(cfg)
    assert not any(a in argv[len(TRAINER_ARGV) + len(extra):] for a in ("--rl-sync-island-lease-s", "--rl-sync-quorum"))
    plan = fleet.plan_of(cfg)
    assert plan.fixed_roster and plan.roster_reason == reason and plan.island_lease_s is None
    assert f"WARNING: fleet: the islands keep a fixed roster ({reason})" in capsys.readouterr().err
    assert any(f"fixed roster ({reason})" in line for line in plan.lines())


def test_a_fleet_with_a_critic_keeps_its_fixed_roster(files, monkeypatch):  # noqa: F811
    from yeto.launch import rl_args

    monkeypatch.setattr(rl_args, "rl_needs_critic", lambda args: True)
    plan = fleet.plan_of(_compiled(files))
    assert plan.fixed_roster and plan.roster_reason == "a critic"


def test_an_explicit_quorum_with_a_grace_window_takes_the_pool_as_given(files):  # noqa: F811
    argv = _argv(_compiled(files, ["--rl-sync-quorum", "2"]))
    assert argv.count("--rl-sync-quorum") == 1 and argv[argv.index("--rl-sync-quorum") + 1] == "2"
    assert argv[argv.index("--rl-sync-island-lease-s") + 1] == "150"


def test_the_island_lease_is_set_in_the_fleet_section(files):  # noqa: F811
    cfg = _compiled(files, island_lease_s=600)
    argv = _argv(cfg)
    assert argv[argv.index("--rl-sync-island-lease-s") + 1] == "600"
    assert fleet.plan_of(cfg).island_lease_s == 600
    # with the trainer's own membership options the section's lease is added once
    argv = _argv(_compiled(files, ["--rl-sync-quorum-theta", "0.5"], island_lease_s=300))
    assert argv.count("--rl-sync-island-lease-s") == 1 and argv[argv.index("--rl-sync-island-lease-s") + 1] == "300"


@pytest.mark.parametrize("extra, keys, match", [
    ([], {"island_lease_s": 0}, "fleet.island_lease_s must be a number of seconds > 0"),
    ([], {"island_lease_s": -30}, r"fleet.island_lease_s must be a number of seconds > 0 \(got -30\)"),
    ([], {"island_lease_s": "200"}, r"fleet.island_lease_s must be a number of seconds > 0 \(got '200'\)"),
    ([], {"island_lease_s": "soon"}, "fleet.island_lease_s must be a number of seconds > 0"),
    ([], {"island_lease_s": True}, r"fleet.island_lease_s must be a number of seconds > 0 \(got True\)"),
    ([], {"island_lease_s": float("nan")}, "fleet.island_lease_s must be a number of seconds > 0"),
    ([], {"island_lease_s": 300, "fixed_roster": True}, "a fixed roster .* has none"),
    (["--rl-sync-island-lease-s", "200"], {"island_lease_s": 300}, "both set the syncer lease"),
])
def test_bad_lease_options_are_refused(files, extra, keys, match):  # noqa: F811
    with pytest.raises(ValueError, match=match):
        _compiled(files, extra, **keys)


def test_the_membership_flags_are_the_syncers():
    from yeto.launch.syncer import _ISLAND_POOL_PARAMS

    assert fleet.membership_flags() == {option for _dest, option in _ISLAND_POOL_PARAMS}


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
    assert doc["roster"] == {"fixed": False, "min_islands": 1, "lease_s": 150, "reason": None}
    doc.pop("roster")
    legacy = fleet.FleetPlan.from_json(doc)
    assert legacy.fixed_roster and legacy.roster_reason == fleet.LEGACY_PLAN


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


# -- the launch's controller: an island STOP drains, then releases ----------------------------------

def _recorded_ctl(ops, client, **kw):
    from tests.test_island_lifecycle import _s3

    learners = {"r-l0": 1, "r-l1": 2}
    ops.status_seq.setdefault(SYNCER, [RUNNING])
    refs = {name: lc.IslandRef(k, "modal") for k, name in enumerate(learners)}
    store = lc.IslandStateStore(_s3(client))
    ctl = FleetController(learners={n: (f"task-{n}", j) for n, j in learners.items()},
                          syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30, recover_timeout=600,
                          thread_cls=ImmediateThread, recorder=lc.FleetRecorder(store, refs), elastic=True, **kw)
    store.request_stop(1, "operator", by="yeto island stop")
    return ctl, store


def test_an_island_stop_gives_its_trainer_the_grace_then_releases_it():
    from tests.test_rl_object_store import FakeS3
    from yeto.launch.fleet import ISLAND_STOP_GRACE_S

    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l1"] = [RUNNING]  # never ends on its own (no command channel)
    ctl, store = _recorded_ctl(ops, client)
    rec = ctl.learners["r-l1"]
    assert ctl._island_stop(rec) and rec["state"] == lf.RUNNING and rec["draining"]
    assert ops.down_calls == [] and store.read(1)["state"] == "draining"
    ops.t += ISLAND_STOP_GRACE_S - 1
    assert ctl._island_stop(rec) and ops.down_calls == []  # still within the grace
    ops.t += 2
    assert ctl._island_stop(rec) and ops.down_calls == ["r-l1"]
    assert rec["state"] == "done" and rec["exit"].startswith("STOPPED")
    assert store.read(1)["state"] == "done" and ops.relaunch_calls == []


def test_a_trainer_that_ends_within_the_grace_is_released_at_once_and_not_relaunched():
    from tests.test_rl_object_store import FakeS3

    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l1"] = [RUNNING, FAILED]
    ctl, store = _recorded_ctl(ops, client)
    rec = ctl.learners["r-l1"]
    assert ctl._island_stop(rec) and ops.down_calls == []  # probed: RUNNING
    ops.t += 30
    assert ctl._island_stop(rec)  # probed: FAILED within the grace
    assert ops.down_calls == ["r-l1"] and ops.relaunch_calls == []
    assert store.read(1)["decision"]["cause"] == "stopped"


def test_a_fenced_controller_tears_nothing_down_on_an_island_stop():
    from tests.test_rl_object_store import FakeS3
    from yeto.launch.island_lifecycle import IslandRecordsFenced

    ops, client = FakeOps(), FakeS3()
    ops.status_seq["r-l1"] = [RUNNING]
    ctl, store = _recorded_ctl(ops, client)
    ctl.recorder.store.fenced = RuntimeError("lock taken by token 9")
    with pytest.raises(IslandRecordsFenced):
        ctl._island_stop(ctl.learners["r-l1"])
    assert ops.down_calls == [] and store.read(1) is None  # no record, no teardown
    assert ctl.learners["r-l1"]["state"] == lf.RUNNING
