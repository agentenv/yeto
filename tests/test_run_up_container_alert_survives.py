"""The sandbox container alert of the ``run up`` monitor is a view only: an envs file
or a pool config that does not read must not raise out of ``breach()`` (it ended the
monitor, and with it the budget / wall-clock breaker, while the nodes billed)."""

from __future__ import annotations

import json
import time
import types

import pytest

from yeto.rl import up


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=time.time, sleep=lambda s: None, gmtime=time.gmtime, strftime=time.strftime))
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))
    cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
           "trainer": {"argv": ["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"], "usd_per_h": 100},
           "first_group": {"timeout_s": 0, "poll_s": 0}, "budget": {"usd": 500, "wall_h": 1}}
    r = up.Run(cfg)
    r.calls = []

    def cmd(argv, *, log, timeout=None):
        r.calls.append(argv)
        return 0

    r._cmd = cmd
    r.pool_cfg_path = pool_cfg
    return r


def test_an_unreadable_envs_file_does_not_raise_out_of_the_breaker(run, tmp_path, monkeypatch):
    run.cfg["sandbox"] = {"envs": str(tmp_path / "gone.json"), "docker_host": "envs", "max_run_containers": 40}
    monkeypatch.setattr(up.subprocess, "run", lambda *a, **k: pytest.fail("no host is known"))
    assert run.breach() is None
    assert "sandbox_over" not in run.state


def test_an_unreadable_pool_config_does_not_raise_out_of_the_breaker(run, monkeypatch):
    run.cfg["sandbox"] = {"docker_host": "ssh://h", "max_containers_per_sampler": 4}
    run.pool_cfg_path.write_text("{ torn")  # the pool state is missing too: the target is read from it
    monkeypatch.setattr(up.subprocess, "run", lambda argv, **k: up.subprocess.CompletedProcess(argv, 0, "a\n", ""))
    assert run.breach() is None


def test_the_monitor_keeps_enforcing_the_wall_clock_after_a_bad_container_poll(run, tmp_path, monkeypatch):
    run.cfg["sandbox"] = {"envs": str(tmp_path / "gone.json"), "docker_host": "envs", "max_run_containers": 40}
    run.state["t0"] = time.time()
    polls = []
    clock = [time.time()]

    def fake_time():
        return clock[0]

    def fake_sleep(s):
        polls.append(s)
        clock[0] += 3600  # the next poll is past wall_h
        if len(polls) > 5:
            pytest.fail("the monitor did not stop on the wall clock")

    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=fake_time, sleep=fake_sleep, gmtime=time.gmtime, strftime=time.strftime))
    monkeypatch.setattr(run, "rollout_stall_alert", lambda: None)
    monkeypatch.setattr(run, "seal_data_end", lambda: False)
    monkeypatch.setattr(run, "watch_nc_after_breach", lambda why, poll_s=60.0: None)
    run.monitor(poll_s=1)
    assert run.state["breaker"]["why"] == "wall clock"
    assert any(argv[-2:] == ["down", "r1-trainer"] for argv in run.calls)
