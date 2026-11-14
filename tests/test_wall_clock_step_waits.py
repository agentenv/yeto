"""Local waits (first-group timeout, starvation stop grace, ``yeto down`` worker grace)
measure their interval on the monotonic clock: an NTP step of the wall clock neither
ends them early nor stretches them forever."""

from __future__ import annotations

import json
import signal
import time
import types
from types import SimpleNamespace

import pytest

from yeto import cli, runs
from yeto.rl import up


class SteppingClock:
    """time.time steps by ``wall_step`` on every read; time.monotonic advances only by sleeps."""

    def __init__(self, wall_step: float, max_sleeps: int = 10_000) -> None:
        self.wall = 1_800_000_000.0
        self.mono = 1000.0
        self.wall_step = wall_step
        self.sleeps = 0
        self.max_sleeps = max_sleeps

    def time(self) -> float:
        self.wall += self.wall_step
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        if self.sleeps > self.max_sleeps:
            raise AssertionError("wait never ended (wall clock stepped back)")
        self.mono += float(seconds)
        self.wall += float(seconds)

    def namespace(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(time=self.time, monotonic=self.monotonic, sleep=self.sleep,
                                     gmtime=time.gmtime, strftime=time.strftime)


def _run(tmp_path, monkeypatch, clock, first_group):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up, "time", clock.namespace())
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))
    cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
           "trainer": {"argv": ["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"]},
           "first_group": first_group, "budget": {}}
    run = up.Run(cfg)
    run._cmd = lambda argv, *, log, timeout=None: 0
    monkeypatch.setattr(up.Run, "breach", lambda self, trainer=False: None)
    return run


def test_first_group_wait_ends_when_the_wall_clock_steps_back(tmp_path, monkeypatch):
    clock = SteppingClock(wall_step=-3600.0)
    run = _run(tmp_path, monkeypatch, clock, {"timeout_s": 600, "poll_s": 60})
    monkeypatch.setattr(up.Run, "buffer_objects", lambda self: 0)
    assert run.wait_first_group() == "first group timed out"
    assert clock.sleeps <= 11


def test_first_group_wait_is_not_cut_by_a_wall_clock_jump_forward(tmp_path, monkeypatch):
    clock = SteppingClock(wall_step=86400.0)
    run = _run(tmp_path, monkeypatch, clock, {"timeout_s": 600, "poll_s": 60})
    monkeypatch.setattr(up.Run, "buffer_objects", lambda self: 1 if clock.sleeps >= 3 else 0)
    assert run.wait_first_group() is None


def test_starvation_stop_grace_survives_a_wall_clock_jump_forward(tmp_path, monkeypatch):
    clock = SteppingClock(wall_step=86400.0)
    run = _run(tmp_path, monkeypatch, clock, {"timeout_s": 0, "poll_s": 0})
    run.state.setdefault("procs", {})["trainer"] = {"pid": 4242}
    monkeypatch.setattr(up.Run, "starvation_config", property(lambda self: SimpleNamespace(stop_grace_s=300.0)))
    monkeypatch.setattr(runs, "request_stop", lambda name: "flag")
    down_at = []
    monkeypatch.setattr(up.Run, "trainer_down", lambda self, why: down_at.append(clock.mono))
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    run.starvation_stop("starved", poll_s=30.0)
    assert down_at and down_at[0] - 1000.0 >= 300.0


def _down_worker(monkeypatch, clock):
    sent = []
    monkeypatch.setattr(cli, "time", clock.namespace())
    monkeypatch.setattr(runs, "load_run", lambda name: {"pid": None, "clusters": []})
    monkeypatch.setattr(runs, "worker_pids", lambda name: [4242])
    monkeypatch.setattr(runs, "is_alive", lambda pid: bool(pid))
    monkeypatch.setattr(runs, "update_run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_signal_worker", lambda pid, sig: sent.append((sig, clock.mono)))
    cli._cmd_down_clusters(SimpleNamespace(run="r1"))
    return sent


def test_down_worker_grace_survives_a_wall_clock_jump_forward(monkeypatch):
    clock = SteppingClock(wall_step=86400.0)
    sent = _down_worker(monkeypatch, clock)
    kill = [at for sig, at in sent if sig == signal.SIGKILL]
    assert kill and kill[0] - 1000.0 >= 10.0


def test_down_worker_grace_ends_when_the_wall_clock_steps_back(monkeypatch):
    clock = SteppingClock(wall_step=-3600.0, max_sleeps=200)
    sent = _down_worker(monkeypatch, clock)
    assert [sig for sig, _ in sent] == [signal.SIGTERM, signal.SIGKILL]
