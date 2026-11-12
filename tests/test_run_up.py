"""``yeto run up``: no automatic decision releases the NC sampler pool."""

from __future__ import annotations

import json
import time
import types

import pytest

from yeto import launch_preflight
from yeto.rl import up


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    # up.py's own view of the time module: patching the real time.time / time.sleep
    # leaks the fake clock into every thread of the test process (a background
    # thread's sleep then advances it, which made the 1.2x breaker test flaky)
    monkeypatch.setattr(up, "time", types.SimpleNamespace(
        time=time.time, sleep=lambda s: None, gmtime=time.gmtime, strftime=time.strftime))
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))

    def build(**budget):
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
               "trainer": {"argv": ["--gpu", "modal:2x8xb300"], "usd_per_h": 100},
               "first_group": {"timeout_s": 0, "poll_s": 0}, "budget": budget}
        run = up.Run(cfg)
        run.calls = []

        def cmd(argv, *, log, timeout=None):
            run.calls.append(argv)
            return 0

        run._cmd = cmd
        run.down_called = False
        real_down = run.down

        def down():
            run.down_called = True
            return real_down()

        run.down = down
        return run

    return build


def _stops_pool(run) -> bool:
    return any("pool-stop" in argv for argv in run.calls)


def _stops_trainer(run) -> bool:
    return any(argv[-2:] == ["down", "r1"] for argv in run.calls)


@pytest.mark.parametrize("why", ["budget $600", "wall clock", "trainer: Traceback / OOM",
                                 "first group timed out", "41 sandbox containers of this run > 40"])
def test_a_breach_stops_the_trainer_only_by_default(make_run, why):
    run = make_run(usd=500)
    run.on_breach(why)
    assert _stops_trainer(run) and not _stops_pool(run) and not run.down_called
    assert run.state["steps"]["trainer_down"]["why"] == why


def test_budget_and_wall_clock_release_nc_only_when_opted_in(make_run):
    run = make_run(usd=500, release_nc_on_breach=True)
    run.on_breach("wall clock")
    assert run.down_called and _stops_pool(run)
    run = make_run(usd=500, release_nc_on_breach=True)
    run.on_breach("trainer: silent 40 min")  # a trainer failure never takes the pool
    assert not _stops_pool(run)


def test_first_group_timeout_keeps_the_pool(make_run, monkeypatch):
    run = make_run()
    monkeypatch.setattr(up.Run, "buffer_objects", lambda self: 0)
    assert run.wait_first_group() == "first group timed out"


def _args(**kw):
    base = dict(config="x", rl_command="up", skip_preflight=True, dry_run=False, no_monitor=True,
                trainer_now=False)
    return types.SimpleNamespace(**{**base, **kw})


def test_trainer_now_skips_the_first_group_wait(make_run, monkeypatch):
    run = make_run()
    order = []
    monkeypatch.setattr(up, "Run", lambda cfg: run)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    monkeypatch.setattr(run, "sandbox_prepare", lambda: True)
    monkeypatch.setattr(run, "pool", lambda: order.append("pool"))
    monkeypatch.setattr(run, "wait_first_group", lambda: order.append("wait") or None)
    monkeypatch.setattr(run, "trainer", lambda: order.append("trainer"))
    monkeypatch.setattr(run, "dashboard", lambda: order.append("dashboard"))
    assert up.main(_args(trainer_now=True)) == 0
    assert order == ["pool", "trainer", "dashboard"]
    order.clear()
    assert up.main(_args()) == 0
    assert order == ["pool", "wait", "trainer", "dashboard"]


def test_first_group_failure_in_main_does_not_stop_the_pool(make_run, monkeypatch):
    run = make_run()
    monkeypatch.setattr(up, "Run", lambda cfg: run)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    monkeypatch.setattr(run, "sandbox_prepare", lambda: True)
    monkeypatch.setattr(run, "pool", lambda: None)
    monkeypatch.setattr(run, "wait_first_group", lambda: "first group timed out")
    assert up.main(_args()) == 1
    assert _stops_trainer(run) and not _stops_pool(run)


def test_parser_has_trainer_now():
    import argparse

    parser = argparse.ArgumentParser()
    up.add_parser(parser.add_subparsers(dest="command"))
    assert parser.parse_args(["run", "up", "--config", "c", "--trainer-now"]).trainer_now


def _finder(present):
    return lambda name: object() if name in present else None


def test_modal_sdk_problem_checks_the_proxy_extra():
    find = _finder({"modal"})
    assert launch_preflight.modal_sdk_problem({}, find) is None
    msg = launch_preflight.modal_sdk_problem({"HTTPS_PROXY": "http://127.0.0.1:7897"}, find)
    assert msg and "aiohttp_socks" in msg and "modal[api-proxy-support]==1.6.3" in msg
    assert "ALL_PROXY" in launch_preflight.modal_sdk_problem({"ALL_PROXY": "socks5://h:1"}, find)
    assert launch_preflight.modal_sdk_problem({"HTTPS_PROXY": "x"}, _finder({"modal", "aiohttp_socks"})) is None
    assert "not importable" in launch_preflight.modal_sdk_problem({}, _finder(set()))


def test_run_up_preflight_refuses_without_the_proxy_extra(make_run, monkeypatch):
    run = make_run()
    run.env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "aiohttp_socks" else object())
    assert run.preflight() is False and run.calls == []


def _trainer(run, tmp_path, text, *, alive=False):
    log = tmp_path / "trainer.log"
    log.write_text(text)
    pid = up.os.getpid() if alive else 2 ** 22 + 12345
    run.state["procs"] = {"trainer": {"pid": pid, "log": str(log), "offset": 0}}


def test_log_content_is_no_breach_while_the_launcher_runs(make_run, tmp_path):
    run = make_run()
    _trainer(run, tmp_path, "Traceback (most recent call last)\ntorch.OutOfMemoryError: CUDA out of memory\n"
             "RuntimeError: island 0 rank 1 setup exited with 128\n", alive=True)
    assert run.breach() is None


@pytest.mark.parametrize("tail", ["[yeto] run 'r1' finished: FAILED (exit code 1)\n",
                                  "[yeto] run 'r1' finished: SUCCEEDED (exit code 2)\n", "killed\n"])
def test_an_exited_launcher_without_success_is_a_breach(make_run, tmp_path, tail, monkeypatch):
    run = make_run()
    monkeypatch.setattr(up, "_worker_alive", lambda name: False)
    _trainer(run, tmp_path, tail)
    assert run.breach().startswith("trainer: launcher")


def test_a_killed_log_streamer_of_a_running_worker_is_no_breach(make_run, tmp_path, monkeypatch):
    run = make_run()
    _trainer(run, tmp_path, "streaming\n")
    monkeypatch.setattr(up, "_worker_alive", lambda name: True)
    assert run.breach() is None


def test_a_launcher_that_finished_ok_is_no_breach(make_run, tmp_path):
    run = make_run()
    _trainer(run, tmp_path, "Traceback (most recent call last)\n[yeto] run 'r1' finished: SUCCEEDED (exit code 0)\n")
    assert run.breach() is None


def test_sandbox_containers_over_the_limit_warn_without_a_breach(make_run, monkeypatch):
    run = make_run()
    run.cfg["sandbox"] = {"docker_host": "ssh://h", "max_run_containers": 40}
    monkeypatch.setattr(run, "run_containers", lambda: 41)
    assert run.breach() is None
    assert run.state["sandbox_over"]["containers"] == 41


def test_a_hung_docker_ps_does_not_end_the_monitor(make_run, monkeypatch):
    run = make_run()
    run.cfg["sandbox"] = {"docker_host": "ssh://h"}

    def hang(*a, **k):
        raise up.subprocess.TimeoutExpired("docker", 60)

    monkeypatch.setattr(up.subprocess, "run", hang)
    assert run.run_containers() is None


def test_run_containers_are_counted_on_every_sandbox_host(make_run, monkeypatch, tmp_path):
    import json

    run = make_run()
    envs = tmp_path / "envs.json"
    envs.write_text(json.dumps({"run": "r1", "hosts": [{"name": f"h{i}", "docker_host": f"ssh://u@h{i}"}
                                                      for i in range(4)]}))
    run.cfg["sandbox"] = {"envs": str(envs), "docker_host": "envs"}
    asked = []

    def ps(argv, **k):
        asked.append(argv[2])
        if argv[2] == "ssh://u@h3":
            raise up.subprocess.TimeoutExpired("docker", 60)
        return up.subprocess.CompletedProcess(argv, 0, "a\nb\n" if argv[2] == "ssh://u@h0" else "c\n", "")

    monkeypatch.setattr(up.subprocess, "run", ps)
    assert run.sandbox_hosts() == [f"ssh://u@h{i}" for i in range(4)]
    assert run.run_containers() == 4 and sorted(asked) == sorted(run.sandbox_hosts())  # h3 did not answer
    run.cfg["sandbox"] = {"docker_host": ["ssh://u@h0", "ssh://u@h1"]}
    assert run.run_containers() == 3


def test_the_first_group_wait_ignores_the_previous_launchers_log(make_run, tmp_path):
    run = make_run()
    log = tmp_path / "old-trainer.log"
    log.write_text("Traceback (most recent call last)\n[yeto] run 'r1' finished: FAILED (exit code 1)\n")
    run.state["procs"] = {"trainer": {"pid": 2 ** 22 + 12345, "log": str(log), "offset": 0}}
    assert run.breach() is not None
    run.state.pop("breaker", None)
    assert run.breach(trainer=False) is None


def _pool_state(tmp_path, monkeypatch, run, members):
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir(exist_ok=True)
    run.env["YETO_NC_STATE_DIR"] = str(nc_dir)
    (nc_dir / "pool-s.json").write_text(json.dumps({"members": members}))


def test_nc_spend_counts_every_billed_member(make_run, tmp_path, monkeypatch):
    run = make_run(nc_usd_per_h=10)
    run.cfg["trainer"]["usd_per_h"] = 0
    now = 100_000.0
    monkeypatch.setattr(up.time, "time", lambda: now)
    run.state["t0"] = now - 7200
    _pool_state(tmp_path, monkeypatch, run, [
        {"state": "ready", "granted": now - 3600},                       # 1 h, still billed
        {"state": "ready", "granted": now - 3600},                       # 1 h, still billed
        {"state": "gone", "granted": now - 7200, "charged_to": now - 5400},  # 0.5 h, released
        {"state": "provisioning", "granted": None},                      # never granted: free
        {"state": "ready", "granted": now - 99_999},                     # before t0: clipped to 2 h
    ])
    assert run.nc_node_hours(now) == pytest.approx(4.5)
    assert run.spend_usd() == pytest.approx(45.0)


def test_nc_spend_without_pool_state_assumes_the_target(make_run, tmp_path, monkeypatch):
    run = make_run(nc_usd_per_h=10)
    run.cfg["trainer"]["usd_per_h"] = 0
    run.env["YETO_NC_STATE_DIR"] = str(tmp_path / "empty")
    up.Path(run.pool_config).write_text(json.dumps({"shard_base": "s", "target": 3}))
    now = 100_000.0
    monkeypatch.setattr(up.time, "time", lambda: now)
    run.state["t0"] = now - 3600
    assert run.spend_usd() == pytest.approx(30.0)


def test_a_budget_breach_keeps_watching_nc_and_forces_pool_stop_past_1_2x(make_run, tmp_path, monkeypatch):
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=100)
    run.cfg["trainer"]["usd_per_h"] = 0
    clock = {"t": 100_000.0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])
    monkeypatch.setattr(up.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 3600))
    run.state["t0"] = clock["t"] - 11 * 3600  # one node 11 h: $110 > $100
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": clock["t"] - 11 * 3600}])
    monkeypatch.setattr(up.Run, "breach", lambda self, trainer=True: "budget $110")
    run.monitor(poll_s=60)
    assert _stops_trainer(run) and not run.down_called
    stops = [i for i, argv in enumerate(run.calls) if "pool-stop" in argv]
    assert len(stops) == 1  # forced once 1.2x ($120) is passed: at 13 h
    assert run.state["steps"]["nc_forced_stop"]["why"].startswith("spend $130")


def test_the_watch_ends_when_no_node_bills(make_run, tmp_path, monkeypatch):
    run = make_run(usd=100)
    _pool_state(tmp_path, monkeypatch, run, [{"state": "gone", "granted": 1.0, "charged_to": 2.0}])
    run.watch_nc_after_breach("wall clock", poll_s=0)
    assert not _stops_pool(run)


def test_a_trainer_failure_or_opt_in_release_ends_the_monitor(make_run, monkeypatch):
    run = make_run(usd=100, release_nc_on_breach=True)
    monkeypatch.setattr(up.Run, "breach", lambda self, trainer=True: "budget $500")
    monkeypatch.setattr(up.Run, "watch_nc_after_breach", lambda *a, **k: pytest.fail("no watch"))
    run.monitor(poll_s=0)
    assert run.down_called
    run = make_run(usd=100)
    monkeypatch.setattr(up.Run, "breach", lambda self, trainer=True: "trainer: launcher exited without a result")
    run.monitor(poll_s=0)
    assert _stops_trainer(run) and not _stops_pool(run)


def test_a_silent_trainer_log_is_a_breach_past_4x_the_stall_timeout(make_run, tmp_path, monkeypatch):
    run = make_run(silence_min=30)
    run.cfg["trainer"]["argv"] += ["--rl-stall-timeout", "1200"]  # 4x = 80 min > 30 min
    assert run.silence_limit_s() == 4800
    _trainer(run, tmp_path, "step 1\n", alive=True)
    clock = {"t": 1000.0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])
    assert run.breach() is None  # first look
    clock["t"] += 4700
    assert run.breach() is None
    (tmp_path / "trainer.log").write_text("step 1\nstep 2\n")  # it grew: the clock restarts
    clock["t"] += 200
    assert run.breach() is None
    clock["t"] += 4801
    assert run.breach().startswith("trainer: log / tape silent 80 min")


def test_the_silence_limit_is_never_below_silence_min(make_run):
    run = make_run(silence_min=45)
    run.cfg["trainer"]["argv"] += ["--rl-stall-timeout=60"]
    assert run.silence_limit_s() == 45 * 60
    assert make_run().silence_limit_s() == 4 * 900  # the launcher's default stall timeout


@pytest.mark.parametrize("flag", [["--rl-stall-timeout", "0"], ["--rl-stall-timeout=0"]])
def test_preflight_refuses_a_disabled_stall_timeout(make_run, flag, monkeypatch):
    monkeypatch.setattr(launch_preflight, "modal_sdk_problem", lambda env: None)
    run = make_run()
    assert run.preflight() is True and run.calls
    run = make_run()
    run.cfg["trainer"]["argv"] += flag
    assert run.preflight() is False and run.calls == []


def test_a_trainer_restart_sets_the_earlier_ledger_aside(make_run, monkeypatch):
    monkeypatch.setattr(launch_preflight, "modal_sdk_problem", lambda env: None)
    run = make_run()
    assert "--rl-ledger-set-aside-before" not in run.trainer_argv()  # the first launch
    run.preflight()
    assert not any("--rl-ledger-set-aside-before" in a for a in run.calls[1])
    run.state["procs"] = {"trainer": {"pid": 2 ** 22 + 1, "log": "x"}}
    argv = run.trainer_argv(now=1234.0)
    assert argv[-2:] == ["--rl-ledger-set-aside-before", "1234"]
    run.calls = []
    run.preflight()  # the dry run checks the argv the restart uses
    assert any("--rl-ledger-set-aside-before" in argv for argv in run.calls)


def test_preflight_refuses_a_zero_budget_with_prices_set(make_run, monkeypatch):
    """The guide's template has budget.usd 0; a customer who fills in the prices and
    keeps it would see the trainer stopped at the first estimated dollar."""
    monkeypatch.setattr(launch_preflight, "modal_sdk_problem", lambda env: None)
    run = make_run(usd=0)
    assert run.preflight() is False and run.calls == []
    run = make_run(usd=0, nc_usd_per_h=0)
    run.cfg["trainer"]["usd_per_h"] = 0  # the template as is: nothing priced, nothing to cap
    assert run.preflight() is True
    assert make_run(usd=450).preflight() is True


def test_children_do_not_import_from_the_start_directory(make_run, tmp_path, monkeypatch):
    """A stray module in the directory ``up`` starts from (a ``h2.py`` shadowed the
    HTTP/2 package of every ``python -m modal`` child) must not reach the children."""
    import os
    import subprocess
    import sys

    (tmp_path / "h2.py").write_text("raise SystemExit('shadowed')\n")
    monkeypatch.chdir(tmp_path)
    # CI runs pytest with ``PYTHONPATH=.``: that entry must not re-add the start dir
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([".", "", "rel/lib", "/abs/lib"]))
    run = make_run()
    assert run.env["PYTHONSAFEPATH"] == "1"
    out = subprocess.run([sys.executable, "-c", "import h2"], env=run.env, cwd=tmp_path,
                         capture_output=True, text=True)
    assert "shadowed" not in (out.stdout + out.stderr)
    assert run.env["PYTHONPATH"].split(os.pathsep) == [str((tmp_path / "rel/lib").resolve()), "/abs/lib"]


def test_dashboard_gets_the_run_name_budget_and_prices(make_run, tmp_path):
    """The dashboard of ``up`` names the run, prices the trainer from the launcher's
    fleet record (``--run``) and draws the cost bar against ``budget.usd``."""
    run = make_run(usd=60)
    assert run.dashboard_args() == ["--store", "s3://b/runs/r1", "--run", "r1", "--budget", "60"]
    run.cfg["dashboard"] = {"port": 1, "prices": str(tmp_path / "prices.json")}
    assert run.dashboard_args()[-2:] == ["--prices", str(tmp_path / "prices.json")]
    run.cfg["budget"] = {}
    assert "--budget" not in run.dashboard_args()


def test_a_finished_trainer_says_the_pool_still_bills_or_releases_it(make_run, tmp_path, capsys):
    """After SUCCEEDED the samplers have nothing to feed: by default a banner names the
    release command (the monitor keeps its guards); release_nc_on_finish runs down."""
    ok = "[yeto] run 'r1' finished: SUCCEEDED (exit code 0)\n"
    run = make_run()
    _trainer(run, tmp_path, ok)
    run.state["trainer_exit_seen"] = "earlier"
    assert run.breach() is None and run.state.get("trainer_finished")
    assert run.on_trainer_finished() is False and not run.down_called
    assert run.on_trainer_finished() is False
    assert capsys.readouterr().out.count("still bill") == 1
    run = make_run(release_nc_on_finish=True)
    _trainer(run, tmp_path, ok)
    run.breach()
    assert run.on_trainer_finished() is True and run.down_called and _stops_pool(run)


def test_a_new_trainer_launch_forgets_the_exit_of_the_previous_one(make_run):
    run = make_run()
    run.state.update(trainer_exit_seen="2026-01-01T00:00:00Z", trainer_finished="2026-01-01T00:00:00Z")
    run._spawn = lambda name, argv: None
    run.trainer()
    assert "trainer_exit_seen" not in run.state and "trainer_finished" not in run.state


def test_status_counts_what_the_trainer_reader_passed_over(make_run, tmp_path):
    """``run status`` sums the trainer log's reader events: groups without a policy
    identity after a resume, stale groups, and the last starvation warning."""
    from yeto.rl.event_echo import format_record

    run = make_run()
    assert run.trainer_buffer_counts() == {"unidentified_policy_groups": 0, "stale_groups": 0,
                                           "unidentified_starved": None}
    log = tmp_path / "trainer.log"
    records = [{"event": "rl_buffer_unidentified_policy_skipped", "unidentified_policy_groups": 4, "rollout_id": 7},
               {"event": "rl_buffer_stale_skipped", "stale_groups": 2, "rollout_id": 7},
               {"event": "rl_buffer_unidentified_policy_skipped", "unidentified_policy_groups": 1, "rollout_id": 8},
               {"event": "rl_buffer_unidentified_policy_starved", "rollout_id": 9, "selections": 3,
                "unidentified_policy_groups_total": 5},
               {"event": "rl_round_trained", "rollout_id": 9}]
    log.write_text("noise\n" + "".join(
        "[trainer] " + format_record({"island_id": 0, "time_unix": 10.0 + i, **r}) + "\n"
        for i, r in enumerate(records)) + "YETO_RL_EVENT {broken\n")
    run.state["procs"] = {"trainer": {"pid": 1, "log": str(log)}}
    counts = run.trainer_buffer_counts()
    assert (counts["unidentified_policy_groups"], counts["stale_groups"]) == (5, 2)
    assert counts["unidentified_starved"] == {"time_unix": 13.0, "rollout_id": 9, "selections": 3,
                                              "unidentified_policy_groups_total": 5}
