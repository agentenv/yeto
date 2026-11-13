"""``yeto run down`` then ``yeto run up``: the per-up clocks (``budget.wall_h``) start
again; the run's cumulative runtime and spend span every up."""

from __future__ import annotations

import json
import os
import types

from tests.test_run_up import make_run  # noqa: F401 - the fixture
from yeto.dashboard import cost as dash_cost
from yeto.rl import up, up_segments

H = 3600.0


def _clock(monkeypatch, start=1_000_000.0):
    now = [start]
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=lambda: now[0], sleep=lambda s: None, gmtime=up.time.gmtime, strftime=up.time.strftime))
    return now


def _no_pool_state(run, monkeypatch):
    monkeypatch.setattr(run, "_members", lambda i: [])  # no NC member bills: the spend is the trainer's


def test_down_then_up_restarts_the_wall_clock(make_run, monkeypatch):
    now = _clock(monkeypatch)
    first = make_run(wall_h=3, usd=10_000)
    _no_pool_state(first, monkeypatch)
    first.begin_up()
    now[0] += 2.5 * H
    first.up_heartbeat()
    assert first.breach(trainer=False) is None
    first.down()
    now[0] += 10 * H  # stopped: the gap is neither wall clock nor runtime
    second = make_run(wall_h=3, usd=10_000)  # the next `yeto run up` process
    _no_pool_state(second, monkeypatch)
    second.begin_up()
    assert second.state["t0"] == now[0]
    assert second.state["run_t0"] == 1_000_000.0
    assert second.breach(trainer=False) is None  # not 12.5 h since the first start
    now[0] += 2 * H
    assert second.breach(trainer=False) is None
    now[0] += 1.5 * H
    assert second.breach(trainer=False) == "wall clock"  # 3.5 h of this up


def test_a_restart_inside_the_same_up_keeps_the_clock(make_run, monkeypatch):
    now = _clock(monkeypatch)
    run = make_run(wall_h=3)
    run.cfg["controller"] = "in_process"
    monkeypatch.setattr(run, "start_controller", lambda: None)
    run.pool()
    now[0] += 2 * H
    run.pool()  # e.g. the pools / trainer started again by this same process
    run.begin_up()
    assert run.state["t0"] == 1_000_000.0
    assert len(run.state["ups"]) == 1
    now[0] += 1.5 * H
    assert run.breach(trainer=False) == "wall clock"


def test_cumulative_runtime_sums_the_up_periods(make_run, monkeypatch):
    now = _clock(monkeypatch)
    run = make_run()
    run.begin_up()
    now[0] += 2 * H
    run.down()
    now[0] += 5 * H
    again = make_run()
    again.begin_up()
    now[0] += 1 * H
    u = again.uptime()
    assert u["ups"] == 2
    assert u["cumulative_runtime_h"] == 3.0
    assert u["current_up_runtime_h"] == 1.0 and u["current_up_open"]
    assert u["wall_h_left"] == 2.0


def test_a_crashed_up_counts_until_its_last_poll(make_run, monkeypatch):
    state = {"ups": [{"pid": 999_999_999, "start": 0.0, "seen": 1800.0, "end": None},
                     {"pid": os.getpid(), "start": 5000.0, "seen": 5000.0, "end": None}]}
    alive = lambda pid: pid == os.getpid()  # noqa: E731
    u = up_segments.uptime_summary(state, 5000.0 + H, alive=alive)
    assert u["cumulative_runtime_h"] == 1.5 and u["ups"] == 2


def test_the_spend_stays_cumulative_over_ups(make_run, monkeypatch):
    now = _clock(monkeypatch)
    run = make_run(usd=150)
    _no_pool_state(run, monkeypatch)
    run.begin_up()
    run.trainer = lambda: run.state.setdefault("trainer_t0", now[0])
    run.trainer()
    run.save()
    now[0] += 1 * H
    run.down()  # $100 of trainer time
    again = make_run(usd=150)
    _no_pool_state(again, monkeypatch)
    again.begin_up()
    now[0] += 0.6 * H
    assert again.spend_usd() > 150
    assert again.breach(trainer=False).startswith("budget")  # not reset by the new up


def test_a_run_from_before_the_periods_keeps_its_first_start(make_run, monkeypatch):
    now = _clock(monkeypatch)
    run = make_run()
    run.state["t0"] = now[0] - 4 * H
    run.save()
    now[0] += 0.0
    run.begin_up()
    assert run.state["run_t0"] == now[0] - 4 * H
    assert run.state["t0"] == now[0]
    assert run.state["ups"][0]["legacy"] and len(run.state["ups"]) == 2


def test_status_and_dashboard_show_the_cumulative_runtime(make_run, monkeypatch):
    now = _clock(monkeypatch)
    run = make_run()
    run.begin_up()
    now[0] += 2 * H
    run.down()
    now[0] += 3 * H
    run = make_run()
    run.begin_up()
    now[0] += 1 * H
    run.up_heartbeat()
    monkeypatch.setattr(run, "buffer_objects", lambda: 0)
    monkeypatch.setattr(run, "run_containers", lambda: 0)
    monkeypatch.setattr(up.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="{}", stderr=""))
    st = json.loads(json.dumps(run.status(), default=str))
    assert st["uptime"]["cumulative_runtime_h"] == 3.0 and st["uptime"]["ups"] == 2
    assert st["uptime"]["spend_usd_estimate"] == st["spend_usd_estimate"]
    argv = run.dashboard_args()
    assert argv[argv.index("--up-state") + 1] == str(run.state_path)
    view = dash_cost.run_up_view(str(run.state_path), now[0])
    assert view["ups"] == 2 and view["cumulative_runtime_h"] == 3.0
    assert view["spend_usd_estimate"] is not None
    assert dash_cost.run_up_view(None, now[0]) is None
