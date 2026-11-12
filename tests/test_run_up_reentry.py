"""``yeto run up`` run again on the same run config (after a stop, a crash, a down)."""

from __future__ import annotations

import os


from tests.test_run_up import _args, make_run  # noqa: F401 - the fixture
from yeto.rl import up


def _stub_steps(run, monkeypatch, order):
    monkeypatch.setattr(run, "sandbox_prepare", lambda: True)
    monkeypatch.setattr(run, "pool", lambda: order.append("pool"))
    monkeypatch.setattr(run, "wait_first_group", lambda: None)
    monkeypatch.setattr(run, "trainer", lambda: order.append("trainer"))
    monkeypatch.setattr(run, "dashboard", lambda: order.append("dashboard"))


def test_a_second_run_up_of_a_running_run_starts_nothing(make_run, monkeypatch):
    first, second = make_run(), make_run()
    assert first.lock_up() is None  # the first run up (its monitor still runs)
    order = []
    _stub_steps(second, monkeypatch, order)
    monkeypatch.setattr(up, "Run", lambda cfg: second)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    assert up.main(_args()) == 2
    assert order == [] and second.calls == []  # no second monitor, no pool / trainer step, no preflight
    assert first.lock_up() is None  # still ours


def test_a_crashed_run_up_leaves_no_stale_lock(make_run, monkeypatch):
    first = make_run()
    assert first.lock_up() is None
    first._up_lock.close()  # the process died: the kernel drops its flock
    (first.dir / "up.lock").write_text("999999")  # its PID stays in the file
    again = make_run()
    order = []
    _stub_steps(again, monkeypatch, order)
    monkeypatch.setattr(up, "Run", lambda cfg: again)
    monkeypatch.setattr(up, "load_config", lambda path: {})
    assert up.main(_args()) == 0
    assert order == ["pool", "trainer", "dashboard"]
    assert (again.dir / "up.lock").read_text() == str(os.getpid())


def test_a_left_stop_flag_does_not_drain_the_pools_of_the_next_in_process_run_up(make_run, monkeypatch):
    run = make_run()
    run.cfg["controller"] = "in_process"
    (run.dir / "STOP").write_text("stop\n")  # an earlier stop of the in-process pools
    seen = []
    monkeypatch.setattr(run, "start_controller", lambda: seen.append((run.dir / "STOP").exists()))
    run.pool()
    assert seen == [False]


def test_trainer_stop_paths_use_the_trainer_launch_name(make_run, monkeypatch):
    """``yeto launch`` names its run by ``--cluster-prefix``: ``yeto down`` and the
    worker check must use that name, not the run config's ``run``."""
    run = make_run()
    assert run.trainer_launch_name() == "r1-trainer"
    run.trainer_down("budget $600")
    assert ["down", "r1-trainer"] == run.calls[-1][-2:]
    run.calls.clear()
    run.down()
    assert any(argv[-2:] == ["down", "r1-trainer"] for argv in run.calls)
    assert not any(argv[-2:] == ["down", "r1"] for argv in run.calls)
    asked = []
    monkeypatch.setattr(up, "_worker_alive", lambda name: asked.append(name) or True)
    log = run.dir / "trainer.log"
    log.write_text("streamer killed\n")
    run.state["procs"] = {"trainer": {"pid": 0, "log": str(log)}}
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    assert run.breach() is None and asked == ["r1-trainer"]
    bare = make_run()
    bare.cfg["trainer"] = {"argv": ["--gpu", "modal:2x8xb300"]}
    assert bare.trainer_launch_name() == "yeto"  # the launch default


