"""``yeto run up``: the budget / wall clock survive corrupt pool-state stamps and a wall
clock stepped back, and a forced pool-stop that keeps failing is retried and alerted."""

from __future__ import annotations

import types

import pytest

from tests.test_run_up import _pool_state, make_run  # noqa: F401 - the fixture
from yeto.rl import up

H = 3600.0


def test_a_corrupt_member_stamp_does_not_raise_out_of_the_breaker(make_run, tmp_path, monkeypatch):
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=100)
    run.cfg["trainer"]["usd_per_h"] = 0
    now = 100_000.0
    monkeypatch.setattr(up.time, "time", lambda: now)
    run.state["t0"] = now - 11 * H
    _pool_state(tmp_path, monkeypatch, run, [
        {"state": "ready", "granted": "garbage"},                    # bills from t0: 11 h
        {"state": "gone", "granted": now - 2 * H, "charged_to": "x"},  # end unknown: to now, 2 h
    ])
    assert run.nc_node_hours(now) == pytest.approx(13.0)
    assert run.breach(trainer=False).startswith("budget")


def test_the_wall_clock_survives_a_clock_stepped_back(make_run, monkeypatch):
    wall, mono = [1_000_000.0], [50.0]
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=lambda: wall[0], monotonic=lambda: mono[0], sleep=lambda s: None,
        gmtime=up.time.gmtime, strftime=up.time.strftime))
    run = make_run(wall_h=3, usd=10_000)
    monkeypatch.setattr(run, "_members", lambda i: [])
    run.begin_up()
    assert run.breach(trainer=False) is None
    wall[0] += 2 * H
    mono[0] += 2 * H
    assert run.breach(trainer=False) is None
    wall[0] -= 5 * H  # NTP step back
    mono[0] += 1.5 * H
    assert run.breach(trainer=False) == "wall clock"  # 3.5 h really passed


def test_a_forced_pool_stop_that_keeps_failing_is_retried_and_alerted(make_run, tmp_path, monkeypatch, capsys):
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=100)
    run.cfg["trainer"]["usd_per_h"] = 0
    clock = {"t": 100_000.0, "polls": 0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])

    def sleep(s):
        clock["t"] += 3600
        clock["polls"] += 1
        if clock["polls"] == 4:  # the machines go away at last
            _pool_state(tmp_path, monkeypatch, run, [{"state": "gone", "granted": 1.0, "charged_to": 2.0}])

    monkeypatch.setattr(up.time, "sleep", sleep)
    run.state["t0"] = clock["t"] - 13 * 3600
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": clock["t"] - 13 * 3600}])
    monkeypatch.setattr(run, "_cmd", lambda argv, **k: run.calls.append(argv) or 124)
    run.watch_nc_after_breach("budget $130", poll_s=60)
    stops = [argv for argv in run.calls if "pool-stop" in argv]
    assert len(stops) == 4
    assert run.state["steps"]["nc_forced_stop_failed"]["attempts"] >= 3


def test_a_spend_error_in_the_watch_does_not_end_it(make_run, tmp_path, monkeypatch):
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=100)
    run.cfg["trainer"]["usd_per_h"] = 0
    clock = {"t": 100_000.0, "n": 0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])
    monkeypatch.setattr(up.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 3600))
    run.state["t0"] = clock["t"] - 13 * 3600
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": clock["t"] - 13 * 3600}])
    real = up.Run.spend_usd

    def flaky(self):
        clock["n"] += 1
        if clock["n"] == 1:
            raise OSError("EIO")
        return real(self)

    monkeypatch.setattr(up.Run, "spend_usd", flaky)
    run.watch_nc_after_breach("budget $130", poll_s=60)
    assert any("pool-stop" in argv for argv in run.calls)
