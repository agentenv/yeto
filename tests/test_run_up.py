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
        time=time.time, monotonic=time.monotonic, sleep=lambda s: None, gmtime=time.gmtime, strftime=time.strftime))
    pool_cfg = tmp_path / "pool.json"
    pool_cfg.write_text(json.dumps({"shard_base": "s"}))

    def build(**budget):
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "pool": {"config": str(pool_cfg)},
               "trainer": {"argv": ["--gpu", "modal:2x8xb300", "--cluster-prefix", "r1-trainer"], "usd_per_h": 100},
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
    return any(argv[-2:] == ["down", "r1-trainer"] for argv in run.calls)


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
    assert run.dashboard_args() == ["--store", "s3://b/runs/r1", "--run", "r1-trainer",
                                    "--up-state", str(run.state_path), "--budget", "60"]
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


def test_a_modal_pools_spend_uses_its_provider_price_not_the_nc_rate(make_run, tmp_path, monkeypatch):
    run = make_run(nc_usd_per_h=24)
    run.cfg["trainer"]["usd_per_h"] = 0
    up.Path(run.pool_config).write_text(json.dumps(
        {"shard_base": "s", "provider": {"kind": "modal", "gpu": "B300", "gpus": 8, "price_usd_h": 68}}))
    now = 100_000.0
    monkeypatch.setattr(up.time, "time", lambda: now)
    run.state["t0"] = now - 7200
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": now - 3600}])
    assert run.node_usd_per_h() == pytest.approx(68.0)
    assert run.spend_usd() == pytest.approx(68.0)


def test_an_nc_pool_keeps_the_nc_rate_even_with_a_price_in_its_provider(make_run, tmp_path, monkeypatch):
    run = make_run(nc_usd_per_h=24)
    up.Path(run.pool_config).write_text(json.dumps({"shard_base": "s", "provider": {"kind": "nc", "price_usd_h": 3}}))
    assert run.node_usd_per_h() == pytest.approx(24.0)


def test_null_prices_and_silence_take_their_defaults_instead_of_crashing_the_monitor(make_run, monkeypatch):
    # trainer.usd_per_h: null passed preflight (``or 0``) but crashed spend_usd mid-run
    run = make_run(usd=100, nc_usd_per_h=None, silence_min=None)
    run.cfg["trainer"]["usd_per_h"] = None
    assert run.budget_problem() is None
    assert run.node_usd_per_h(0) == pytest.approx(24.0)
    assert run.spend_usd() >= 0.0
    assert run.silence_limit_s() >= 30 * 60


@pytest.mark.parametrize("wall_h, why", [(None, None), (3, "wall clock"), (5, None)])
def test_the_wall_clock_breach_follows_budget_wall_h(make_run, monkeypatch, wall_h, why):
    run = make_run(usd=None, wall_h=wall_h)
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    run.state["t0"] = up.time.time() - 4 * 3600
    assert run.breach(trainer=False) == why


def test_an_absent_wall_h_keeps_the_3_h_default(make_run, monkeypatch):
    run = make_run(usd=None)
    monkeypatch.setattr(up.Run, "spend_usd", lambda self: 0.0)
    run.state["t0"] = up.time.time() - 4 * 3600
    assert run.breach(trainer=False) == "wall clock"


@pytest.mark.parametrize("wall_h, forced", [(None, "spend $"), (3, "wall ")])
def test_the_1_2x_breaker_skips_the_wall_cap_when_wall_h_is_null(make_run, tmp_path, monkeypatch, wall_h, forced):
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=wall_h)
    run.cfg["trainer"]["usd_per_h"] = 0
    clock = {"t": 100_000.0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])
    monkeypatch.setattr(up.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 3600))
    run.state["t0"] = clock["t"] - 3600
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": clock["t"] - 3600}])
    run.watch_nc_after_breach("budget", poll_s=60)
    assert run.state["steps"]["nc_forced_stop"]["why"].startswith(forced)


@pytest.mark.parametrize("budget, warned", [({"usd": 450}, True), ({"usd": 450, "wall_h": 80}, False),
                                            ({"usd": 450, "wall_h": None}, False)])
def test_preflight_warns_when_wall_h_is_not_set(make_run, monkeypatch, capsys, budget, warned):
    monkeypatch.setattr("yeto.launch_preflight.modal_sdk_problem", lambda env: None)
    run = make_run(**budget)
    assert run.preflight() is True
    assert ("WARNING: budget.wall_h is not set" in capsys.readouterr().out) is warned


@pytest.mark.parametrize("wall_h", ["3h", "", 0, -1, True, [3]])
def test_a_wall_h_that_is_not_a_positive_number_fails_preflight(make_run, monkeypatch, capsys, wall_h):
    monkeypatch.setattr("yeto.launch_preflight.modal_sdk_problem", lambda env: None)
    run = make_run(usd=450, wall_h=wall_h)
    assert run.preflight() is False and run.calls == []
    assert "budget.wall_h must be a positive number of hours or null" in capsys.readouterr().out


def test_a_forced_pool_stop_that_times_out_is_retried_until_no_node_bills(make_run, tmp_path, monkeypatch):
    """The breaker's pool-stop hits its timeout (rc 124): the machines may still bill, so
    the watch goes on and stops again at the next poll instead of returning as if done."""
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=100)
    run.cfg["trainer"]["usd_per_h"] = 0
    clock = {"t": 100_000.0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])
    monkeypatch.setattr(up.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 3600))
    run.state["t0"] = clock["t"] - 13 * 3600
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": clock["t"] - 13 * 3600}])
    path = tmp_path / "nc" / "pool-s.json"
    rcs = iter([124, 0])

    def cmd(argv, *, log, timeout=None):
        run.calls.append(argv)
        rc = next(rcs)
        if rc == 0:  # this stop released the node
            doc = json.loads(path.read_text())
            doc["members"][0]["state"] = "gone"
            path.write_text(json.dumps(doc))
        return rc

    run._cmd = cmd
    run.watch_nc_after_breach("budget", poll_s=60)
    assert sum("pool-stop" in argv for argv in run.calls) == 2
    assert run.state["steps"]["nc_forced_stop"]["rc"] == 124  # the first forced stop is the one recorded


def test_status_reports_unknown_buffer_count_when_the_store_listing_fails(make_run, monkeypatch):
    """A control host without store credentials: ``run status`` must not claim an
    empty buffer (0 objects) when it could not list the store at all."""
    run = make_run()

    def no_creds():
        raise RuntimeError("Unable to locate credentials")

    monkeypatch.setattr(run, "_objects", no_creds)
    monkeypatch.setattr(run, "run_containers", lambda: None)
    monkeypatch.setattr(up.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="", stderr="", returncode=1))
    assert run.status()["buffer_objects"] is None


def _leaky_stderr():
    """A failing subprocess's stderr naming a presigned URL, a key=value secret and URL
    userinfo, around readable diagnostics (the fake values are split so no scanner
    takes them for real ones)."""
    sig = "Zm9vYmFy" + "c2lnbmF0dXJl"
    secret = "wJalr" + "XUtnFEMI" + "K7MDENG"
    userinfo = "admin:" + "hunt" + "er2pass"
    text = (f"Traceback (most recent call last):\n  pool-status: no node answered\n"
            f"GET https://bucket.r2.example/obj?X-Amz-Signature={sig}&X-Amz-Expires=600 failed\n"
            f"aws_secret_access_key={secret}\n"
            f"mirror https://{userinfo}@store.example/run refused\n")
    return text, (sig, secret, userinfo)


def test_status_redacts_credentials_in_subprocess_stderr(make_run, monkeypatch):
    """``run status`` (and the dashboard reading it) shows the tail of a failing
    ``pool-status`` / dispense stderr: no credential reaches it, the diagnostics do."""
    run = make_run()
    text, secrets = _leaky_stderr()
    monkeypatch.setattr(run, "_objects", lambda: [])
    monkeypatch.setattr(run, "run_containers", lambda: None)
    monkeypatch.setattr(up.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="", stderr=text, returncode=1))
    out = run.status()
    shown = json.dumps(out)
    for value in secrets:
        assert value not in shown
    for field in (out["pool"], out["dispense"]):
        assert "pool-status: no node answered" in field and "\n" in field
        assert "https://bucket.r2.example/obj?" in field and "store.example/run refused" in field


@pytest.fixture
def read_only(tmp_path):
    """A directory made read-only (the monitor's state dir on an EROFS / full disk);
    writable again after the test so tmp_path cleans up."""
    made = []

    def make(path):
        path.chmod(0o500)
        made.append(path)

    yield make
    for path in made:
        path.chmod(0o700)


def test_a_state_write_failure_does_not_end_the_breaker_or_the_forced_pool_stop(make_run, tmp_path, monkeypatch,
                                                                                read_only, capsys):
    """state.json not writable (ENOSPC / EROFS) while the run bills: the monitor still
    detects the budget breach, stops the trainer and forces the pool-stop past 1.2x."""
    run = make_run(usd=100, nc_usd_per_h=10, wall_h=100)
    run.cfg["trainer"]["usd_per_h"] = 0
    clock = {"t": 100_000.0}
    monkeypatch.setattr(up.time, "time", lambda: clock["t"])
    monkeypatch.setattr(up.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 3600))
    run.state["t0"] = clock["t"] - 11 * 3600
    _pool_state(tmp_path, monkeypatch, run, [{"state": "ready", "granted": clock["t"] - 11 * 3600}])
    run.save()
    read_only(run.dir)
    run.monitor(poll_s=60)
    assert _stops_trainer(run) and _stops_pool(run)
    assert run.state["breaker"]["why"].startswith("budget")  # kept in memory
    assert "CRITICAL state_write_failed" in capsys.readouterr().out
    run.soft_save = False
    with pytest.raises(OSError):  # outside the monitor (run down, run scale) a failed write still raises
        run.save()


def test_a_command_runs_when_its_log_cannot_be_written(make_run, tmp_path, read_only):
    """A pool-stop / down whose log file cannot be opened still runs (and returns its rc)."""
    import sys

    run = make_run(usd=100)
    read_only(run.dir)
    marker = tmp_path / "ran"
    rc = up.Run._cmd(run, [sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x'); print('out')"],
                     log="down.log", timeout=60)
    assert rc == 0 and marker.exists()
    assert not (run.dir / "down.log").exists()
