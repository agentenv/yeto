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


def test_the_last_cluster_prefix_names_the_launch(make_run):
    run = make_run()
    run.cfg["trainer"]["argv"] = ["--cluster-prefix", "a", "--cluster-prefix=b"]
    assert run.trainer_launch_name() == "b"  # argparse keeps the last


def test_the_dashboard_follows_the_trainer_launch_records(make_run):
    """The launcher writes ``~/.yeto/runs/<--cluster-prefix>/fleet.jsonl``: the dashboard's
    ``--run`` must be that name, not the run config's ``run``."""
    argv = make_run().dashboard_args()
    assert argv[argv.index("--run") + 1] == "r1-trainer"


_HOLD = """
import subprocess, sys, time
sys.path.insert(0, {root!r})
from yeto.rl import up
up.STATE_ROOT = __import__("pathlib").Path({state!r})
run = up.Run.__new__(up.Run)
run.dir = up.STATE_ROOT / "r1"
run.dir.mkdir(parents=True, exist_ok=True)
assert run.lock_up() is None
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], close_fds=False,
                         start_new_session=True)
print(child.pid, flush=True)
time.sleep(60)
"""


def test_a_sigkilled_run_up_leaves_no_lock_held_by_its_children(make_run, tmp_path):
    """The lock fd is not inherited (O_CLOEXEC, even with close_fds=False): a detached
    child (pool / trainer launcher) outliving a SIGKILLed ``up`` does not hold the lock."""
    import signal
    import subprocess
    import sys
    import time
    from pathlib import Path

    run = make_run()
    root = str(Path(up.__file__).resolve().parents[2])
    holder = subprocess.Popen([sys.executable, "-c", _HOLD.format(root=root, state=str(run.dir.parent))],
                              stdout=subprocess.PIPE, text=True)
    child = int(holder.stdout.readline())
    try:
        assert run.lock_up() == holder.pid  # held by the live up
        run._up_lock = None
        holder.send_signal(signal.SIGKILL)
        holder.wait()
        time.sleep(0.2)
        assert os.kill(child, 0) is None  # the grandchild still runs
        assert run.lock_up() is None
    finally:
        holder.kill()
        os.kill(child, signal.SIGKILL)
        holder.stdout.close()


def test_run_scale_and_the_run_up_monitor_do_not_drop_each_others_state(make_run, monkeypatch):
    """``run scale`` (its own process) appends to ``scales`` while ``run up``'s monitor
    keeps writing from its older copy: neither write loses the other's change."""
    import json

    monitor, scaler = make_run(), make_run()
    monitor.mark("pool")
    scaler2 = make_run()  # read after the pool step
    monkeypatch.setattr(up.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 0})())
    monkeypatch.setattr(scaler2, "scale_refusal", lambda wanted, i=0: None)
    assert scaler2.scale(3) == 0
    monitor.state["breaker"] = {"at": "t", "why": "w"}
    monitor.save()  # the monitor's copy has no scales
    monitor.mark("first_group")
    assert scaler2.scale(4) == 0  # its copy has no breaker / first_group
    disk = json.loads(monitor.state_path.read_text())
    assert [s["samplers"] for s in disk["scales"]] == [3, 4]
    assert disk["breaker"]["why"] == "w"
    assert set(disk["steps"]) == {"pool", "first_group"}
    scaler.state.setdefault("procs", {})["trainer"] = {"pid": 1}
    scaler.save()
    disk = json.loads(monitor.state_path.read_text())
    assert disk["procs"]["trainer"] == {"pid": 1} and len(disk["scales"]) == 2
    monitor.state.pop("breaker")
    monitor.save()  # a key this process removes is removed
    assert "breaker" not in json.loads(monitor.state_path.read_text())
    assert not list(monitor.dir.glob(".state.json.*.tmp"))
