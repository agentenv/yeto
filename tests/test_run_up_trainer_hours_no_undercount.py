"""Trainer hours of ``yeto run up`` / ``run status``: the stopped gap of a run upgraded from
34c08526 is not billed before its first ``run up``, and no case bills less than the
trainer really ran (alive, crashed, several ups, an open period closed while the trainer
runs, a finished trainer, a read-only state disk, the legacy upgrade)."""

from __future__ import annotations

import errno
import json
import os
import types

from tests.test_run_up import make_run  # noqa: F401 - the fixture
from tests.test_run_up_upgrade_compat import STATE_34C08526, T0
from yeto.rl import up

H = 3600.0
DOWN = T0 + 600 + 5 * H  # the old `run down` of STATE_34C08526 (trainer launched at T0 + 600)
DEAD = 4195303  # a pid that does not run


def _clock(monkeypatch, start):
    now = [start]
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=lambda: now[0], sleep=lambda s: None, gmtime=up.time.gmtime, strftime=up.time.strftime))
    return now


def _run(make_run, monkeypatch, state, *, worker=False):
    run = make_run(usd=10_000, wall_h=1000)
    run.state_path.write_text(json.dumps(state))
    monkeypatch.setattr(up, "_worker_alive", lambda name: worker)
    run = make_run(usd=10_000, wall_h=1000)  # a fresh process (run status / the next run up) reads it
    monkeypatch.setattr(run, "_members", lambda i: [])  # the spend is the trainer's
    return run


def _with_trainer(state, pid):
    return {**state, "procs": {**state["procs"], "trainer": {**state["procs"]["trainer"], "pid": pid}}}


# -- the overcount fixed: run status of an upgraded run before its first run up


def test_status_before_the_first_up_does_not_bill_the_stopped_gap(make_run, monkeypatch):
    _clock(monkeypatch, DOWN + 24 * H)  # a day after the old down, the upgraded yeto not run up yet
    run = _run(make_run, monkeypatch, STATE_34C08526)
    assert abs(run.trainer_hours() - 5.0) < 1e-9  # was 29 h: now - trainer_t0
    assert abs(run.spend_usd() - 500.0) < 1e-6
    assert "ups" not in json.loads(run.state_path.read_text())  # nothing written


def test_status_before_the_first_up_equals_the_spend_the_first_up_records(make_run, monkeypatch):
    for extra in ({}, {"trainer_finished": "2023-11-15T01:23:20Z"}, {"trainer_exit_seen": "2023-11-15T02:00:00Z"}):
        _clock(monkeypatch, DOWN + 24 * H)
        state = {**STATE_34C08526, **extra}
        before = _run(make_run, monkeypatch, state).trainer_hours()
        run = _run(make_run, monkeypatch, state)
        run.begin_up()
        assert abs(run.trainer_hours() - before) < 1e-9


# -- no undercount: the legacy cases where the last stamp may be short keep now - trainer_t0


def test_a_legacy_trainer_launcher_still_running_bills_to_now(make_run, monkeypatch):
    now = _clock(monkeypatch, DOWN + 24 * H)
    run = _run(make_run, monkeypatch, _with_trainer(STATE_34C08526, os.getpid()))
    assert abs(run.trainer_hours() - (now[0] - (T0 + 600)) / H) < 1e-9


def test_a_legacy_worker_still_running_bills_to_now(make_run, monkeypatch):
    """The launcher gone (a killed log streamer) but its ``yeto launch`` worker runs."""
    now = _clock(monkeypatch, DOWN + 24 * H)
    run = _run(make_run, monkeypatch, STATE_34C08526, worker=True)
    assert abs(run.trainer_hours() - (now[0] - (T0 + 600)) / H) < 1e-9


def test_a_legacy_state_without_stamps_bills_to_now(make_run, monkeypatch):
    now = _clock(monkeypatch, T0 + 30 * H)
    run = _run(make_run, monkeypatch, {"t0": T0, "trainer_t0": T0 + 600})
    assert abs(run.trainer_hours() - (now[0] - (T0 + 600)) / H) < 1e-9


def test_a_legacy_crash_without_down_bills_to_its_last_stamp_at_least(make_run, monkeypatch):
    """The old up crashed (no down, no finished): the trainer's exit seen is the last stamp."""
    _clock(monkeypatch, DOWN + 24 * H)
    steps = {k: v for k, v in STATE_34C08526["steps"].items() if k != "down"}
    state = {**STATE_34C08526, "steps": steps, "trainer_exit_seen": "2023-11-15T01:23:20Z"}
    run = _run(make_run, monkeypatch, state)
    assert run.trainer_hours() >= 3.0 - 1e-9  # launched 22:23:20, its exit seen 01:23:20


def test_a_legacy_trainer_finished_bills_to_its_finish(make_run, monkeypatch):  # #444 on a legacy state
    _clock(monkeypatch, DOWN + 24 * H)
    run = _run(make_run, monkeypatch, {**STATE_34C08526, "trainer_finished": "2023-11-15T01:23:20Z"})
    assert abs(run.trainer_hours() - 3.0) < 1e-9


def test_a_legacy_stamp_past_now_bills_no_more_than_now(make_run, monkeypatch):
    now = _clock(monkeypatch, T0 + 2 * H)
    state = {"t0": T0, "trainer_t0": T0, "steps": {"down": {"at": "2023-11-20T00:00:00Z", "ok": True}}}
    run = _run(make_run, monkeypatch, state)
    assert abs(run.trainer_hours() - (now[0] - T0) / H) < 1e-9


# -- no undercount: the runs with periods are billed as before


def _periods_state(ups, *, trainer_pid=DEAD, **extra):
    return {"t0": ups[-1]["start"], "run_t0": T0, "trainer_t0": T0 + 600, "ups": ups,
            "procs": {"trainer": {"log": "x", "pid": trainer_pid, "started": "2023-11-14T22:23:20Z"}}, **extra}


def test_a_trainer_alive_in_the_current_up_bills_to_now(make_run, monkeypatch):
    now = _clock(monkeypatch, T0 + 10 * H)
    ups = [{"pid": os.getpid(), "start": T0, "seen": T0 + 9 * H, "end": None}]
    run = _run(make_run, monkeypatch, _periods_state(ups, trainer_pid=os.getpid()))
    assert abs(run.trainer_hours() - (now[0] - T0 - 600) / H) < 1e-9


def test_a_trainer_alive_after_its_up_died_bills_to_now(make_run, monkeypatch):
    now = _clock(monkeypatch, T0 + 10 * H)
    ups = [{"pid": DEAD, "start": T0, "seen": T0 + H, "end": None}]  # a killed --no-monitor up
    run = _run(make_run, monkeypatch, _periods_state(ups, trainer_pid=os.getpid()))
    assert abs(run.trainer_hours() - (now[0] - T0 - 600) / H) < 1e-9


def test_a_crashed_trainer_without_finished_bills_its_up_to_the_last_poll(make_run, monkeypatch):
    _clock(monkeypatch, T0 + 10 * H)
    ups = [{"pid": DEAD, "start": T0, "seen": T0 + 4 * H, "end": None}]
    run = _run(make_run, monkeypatch, _periods_state(ups, trainer_exit_seen="2023-11-15T00:13:20Z"))
    assert abs(run.trainer_hours() - (4 * H - 600) / H) < 1e-9


def test_several_ups_bill_every_period_from_the_trainer_launch(make_run, monkeypatch):
    now = _clock(monkeypatch, T0 + 20 * H)
    ups = [{"pid": None, "start": T0, "seen": T0 + 3 * H, "end": T0 + 3 * H},
           {"pid": None, "start": T0 + 5 * H, "seen": T0 + 8 * H, "end": T0 + 8 * H},
           {"pid": os.getpid(), "start": T0 + 12 * H, "seen": T0 + 19 * H, "end": None}]
    run = _run(make_run, monkeypatch, _periods_state(ups, trainer_pid=os.getpid()))
    assert abs(run.trainer_hours() - ((3 * H - 600) + 3 * H + (now[0] - T0 - 12 * H)) / H) < 1e-9


def test_an_open_period_closed_while_its_trainer_runs_bills_the_gap(make_run, monkeypatch):  # #419
    now = _clock(monkeypatch, T0 + 10 * H)
    ups = [{"pid": DEAD, "start": T0, "seen": T0 + H, "end": None}]
    run = _run(make_run, monkeypatch, _periods_state(ups, trainer_pid=os.getpid()))
    run.begin_up()  # closes the killed up's period at now: its detached trainer ran the whole gap
    assert run.state["ups"][0]["end"] == now[0]
    now[0] += H
    assert abs(run.trainer_hours() - (now[0] - T0 - 600) / H) < 1e-9


def test_a_finished_trainer_bills_to_its_finish(make_run, monkeypatch):  # #444
    _clock(monkeypatch, T0 + 10 * H)
    ups = [{"pid": None, "start": T0, "seen": T0 + 2 * H, "end": T0 + 2 * H},
           {"pid": os.getpid(), "start": T0 + 4 * H, "seen": T0 + 9 * H, "end": None}]
    run = _run(make_run, monkeypatch, _periods_state(ups, trainer_finished="2023-11-15T04:13:20Z"))  # T0 + 6 h
    assert abs(run.trainer_hours() - ((2 * H - 600) + 2 * H) / H) < 1e-9


def test_a_read_only_state_disk_keeps_billing_in_memory(make_run, monkeypatch):  # #469
    now = _clock(monkeypatch, DOWN + 24 * H)
    run = _run(make_run, monkeypatch, STATE_34C08526)
    run.soft_save = True

    def erofs(*a, **k):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(up.os, "replace", erofs)
    run.begin_up()  # the period lives in memory only
    assert "ups" not in json.loads(run.state_path.read_text())
    assert abs(run.trainer_hours() - 5.0) < 1e-9
    run.state["procs"]["trainer"]["pid"] = os.getpid()  # the trainer this up started runs
    now[0] += 3 * H
    assert abs(run.trainer_hours() - 8.0) < 1e-9


# -- no undercount: a legacy state without a successful down after the trainer's launch


def _no_down(**extra):
    steps = {k: v for k, v in STATE_34C08526["steps"].items() if k != "down"}
    return {**STATE_34C08526, "steps": steps, **extra}


def test_a_legacy_state_without_down_bills_to_now_though_its_pid_looks_dead(make_run, monkeypatch):
    """Another control machine, a killed launcher whose app still runs: the pid looks dead here."""
    now = _clock(monkeypatch, DOWN + 24 * H)
    for state in (_no_down(), _no_down(trainer_exit_seen="2023-11-15T01:23:20Z"),
                  {**STATE_34C08526, "steps": {**STATE_34C08526["steps"], "down": {"at": "2023-11-15T03:23:20Z",
                                                                                    "ok": False}}},
                  {**STATE_34C08526, "steps": {**STATE_34C08526["steps"], "down": {"at": "2023-11-14T22:00:00Z",
                                                                                    "ok": True}}}):
        run = _run(make_run, monkeypatch, state)
        assert abs(run.trainer_hours() - (now[0] - (T0 + 600)) / H) < 1e-9


def test_the_first_up_of_a_legacy_state_without_down_bills_the_gap(make_run, monkeypatch):
    now = _clock(monkeypatch, DOWN + 24 * H)
    run = _run(make_run, monkeypatch, _no_down())
    run.begin_up()
    assert run.state["ups"][0]["seen"] == now[0]
    assert abs(run.trainer_hours() - (now[0] - (T0 + 600)) / H) < 1e-9
