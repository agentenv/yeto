"""``yeto run up`` on a ``state.json`` written by an older yeto (upgrade mid-run).

The fixture below is the ``state.json`` the run-up code of 34c08526 (before the ``ups``
periods were recorded) wrote for: ``run up`` at t0, the trainer 10 min later, ``run down``
5 h after that. Its spend (5 h of trainer at $100/h) must survive the upgrade."""

from __future__ import annotations

import json
import types

from tests.test_run_up import make_run  # noqa: F401 - the fixture
from yeto.rl import up, up_segments

H = 3600.0
T0 = 1_700_000_000.0
STATE_34C08526 = {
    "procs": {"pool": {"log": "x", "pid": 4195303, "started": "2023-11-14T22:13:20Z"},
              "trainer": {"log": "x", "pid": 4195303, "started": "2023-11-14T22:23:20Z"}},
    "steps": {"down": {"at": "2023-11-15T03:23:20Z", "ok": True},
              "pool": {"at": "2023-11-14T22:13:20Z"},
              "trainer": {"at": "2023-11-14T22:23:20Z"}},
    "t0": T0,
    "trainer_t0": T0 + 600,
}


def _clock(monkeypatch, start):
    now = [start]
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=lambda: now[0], sleep=lambda s: None, gmtime=up.time.gmtime, strftime=up.time.strftime))
    return now


def _legacy_run(make_run, monkeypatch, state, usd=1000):
    run = make_run(usd=usd, wall_h=100)
    run.state_path.write_text(json.dumps(state))
    run = make_run(usd=usd, wall_h=100)  # the upgraded `yeto run up` process reads it
    monkeypatch.setattr(run, "_members", lambda i: [])  # the spend is the trainer's
    return run


def test_an_upgraded_run_keeps_the_spend_of_its_pre_period_ups(make_run, monkeypatch):
    now = _clock(monkeypatch, T0 + 600 + 5 * H + 24 * H)  # a day after the old `run down`
    run = _legacy_run(make_run, monkeypatch, STATE_34C08526)
    run.begin_up()
    legacy = run.state["ups"][0]
    assert legacy["legacy"] and legacy["end"] == T0 + 600 + 5 * H  # the old `run down`
    assert abs(run.trainer_hours() - 5.0) < 1e-9
    assert abs(run.spend_usd() - 500.0) < 1e-6
    summary = up_segments.uptime_summary(run.state, now[0])
    assert abs(summary["cumulative_runtime_h"] - (5 * H + 600) / H) < 1e-3
    now[0] += 5.1 * H
    assert run.breach(trainer=False).startswith("budget")  # $500 before + $510 now > $1000


def test_an_upgraded_run_without_stamps_still_starts(make_run, monkeypatch):
    _clock(monkeypatch, T0 + 2 * H)
    run = _legacy_run(make_run, monkeypatch, {"t0": T0})
    run.begin_up()
    assert run.state["ups"][0]["seen"] == T0 and run.trainer_hours() == 0.0
