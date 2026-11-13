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
    assert up._parse_lock_holder((again.dir / "up.lock").read_text())["pid"] == os.getpid()


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


_HOLD_UNTIL_TERM = """
import json, sys, time
sys.path.insert(0, {root!r})
from yeto.rl import up
up.STATE_ROOT = __import__("pathlib").Path({state!r})
run = up.Run.__new__(up.Run)
run.dir = up.STATE_ROOT / "r1"
run.dir.mkdir(parents=True, exist_ok=True)
assert run.lock_up() is None
(run.dir / "state.json").write_text(json.dumps({{"procs": {{"trainer": {{"pid": 0, "log": "t.log"}}}}}}))
print("held", flush=True)
time.sleep({hold})
"""


def _holder(run, hold):
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(up.__file__).resolve().parents[2])
    proc = subprocess.Popen([sys.executable, "-c", _HOLD_UNTIL_TERM.format(root=root, state=str(run.dir.parent),
                                                                           hold=hold)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def test_run_down_stops_a_running_run_up_and_takes_its_lock(make_run, monkeypatch):
    """``run down`` while ``run up`` runs: the up gets SIGTERM, down holds ``up.lock`` (no
    up of the run starts during down) and reads the state the up wrote since."""
    import time as real_time

    run = make_run()
    monkeypatch.setattr(up.time, "sleep", real_time.sleep)  # the fixture's sleep is a no-op
    holder = _holder(run, 60)
    try:
        assert run.lock_for_down(grace_s=10, poll_s=0.05) is True
        assert holder.wait(timeout=5) == -15  # SIGTERM
        assert run.state["procs"]["trainer"]["log"] == "t.log"  # reloaded
        other = up.Run.__new__(up.Run)
        other.dir = run.dir
        assert other.lock_up() == os.getpid()  # a new run up sees down holding the lock
    finally:
        holder.kill()
        holder.stdout.close()


def test_run_down_does_not_wait_forever_for_a_run_up_that_ignores_sigterm(make_run, monkeypatch):
    """Shutdown first: an up that does not let go of the lock delays down by the grace
    period only; down still stops the trainer, the pools and the sandboxes."""
    import signal

    run = make_run()
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    sent = []
    monkeypatch.setattr(up.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    held = up.Run.__new__(up.Run)
    held.dir = run.dir
    assert held.lock_up() is None  # stands for an up that ignores SIGTERM
    (run.dir / "up.lock").write_text("424242")
    try:
        assert run.lock_for_down(grace_s=3, poll_s=1) is False
        assert (424242, signal.SIGTERM) in sent
        run.calls.clear()
        run.down()
        assert any(argv[-2:] == ["down", "r1-trainer"] for argv in run.calls)
        assert any("pool-stop" in argv for argv in run.calls)
    finally:
        held._up_lock.close()


def test_run_down_without_a_running_up_takes_the_lock_at_once(make_run, monkeypatch):
    run = make_run()
    monkeypatch.setattr(up.os, "kill", lambda pid, sig: (_ for _ in ()).throw(AssertionError("no kill")))
    assert run.lock_for_down(grace_s=3, poll_s=1) is True
    holder = up._parse_lock_holder((run.dir / "up.lock").read_text())
    assert holder["role"] == "down" and holder["pid"] == os.getpid() and holder["start"] == up._proc_start(os.getpid())


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


def test_run_down_reload_resets_the_merge_base_so_scales_are_not_appended_twice(make_run, monkeypatch):
    """Disk [n1]; a ``run scale`` appends n2 while down's copy is older; down reloads in
    ``lock_for_down`` and saves at the end: [n1, n2], not [n1, n2, n2]."""
    import json

    first = make_run()
    first.state["scales"] = [{"samplers": 1}]
    first.save()
    down = make_run()  # read [n1]
    scaler = make_run()
    scaler.state["scales"].append({"samplers": 2})
    scaler.save()
    assert down.lock_for_down(grace_s=1, poll_s=1) is True
    down.state["procs"] = {}
    down.save()
    assert [s["samplers"] for s in json.loads(down.state_path.read_text())["scales"]] == [1, 2]


def test_run_down_shuts_down_with_a_truncated_state_json(make_run, monkeypatch):
    run = make_run()
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    run.state["procs"] = {"trainer": {"pid": 999999, "log": "t.log"}}
    run.save()
    run.state_path.write_text('{"procs": {"tra')  # truncated
    assert run.lock_for_down(grace_s=1, poll_s=1) is True
    assert run.state["procs"]["trainer"]["log"] == "t.log"  # the copy in memory
    run.calls.clear()
    run.down()
    assert any(argv[-2:] == ["down", "r1-trainer"] for argv in run.calls)


def test_run_down_waits_for_another_run_down_without_sigterm(make_run, monkeypatch):
    import json

    run = make_run()
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    monkeypatch.setattr(up.os, "kill", lambda pid, sig: (_ for _ in ()).throw(AssertionError("no kill")))
    held = up.Run.__new__(up.Run)
    held.dir = run.dir
    assert held.lock_up(role="down") is None
    try:
        assert run.lock_for_down(grace_s=2, poll_s=1) is False
        assert run.lock_holder["role"] == "down"
    finally:
        held._up_lock.close()
    (run.dir / "up.lock").write_text(json.dumps({"role": "down", "pid": 1, "start": None}))
    assert run.lock_for_down(grace_s=1, poll_s=1) is True  # the other down exited


def test_run_down_does_not_sigterm_a_reused_pid(make_run, monkeypatch):
    import json

    run = make_run()
    monkeypatch.setattr(up, "_alive", lambda pid: True)
    monkeypatch.setattr(up.os, "kill", lambda pid, sig: (_ for _ in ()).throw(AssertionError("no kill")))
    held = up.Run.__new__(up.Run)
    held.dir = run.dir
    assert held.lock_up() is None
    (run.dir / "up.lock").write_text(json.dumps({"role": "up", "pid": 424242, "start": "not-its-start"}))
    monkeypatch.setattr(up, "_proc_start", lambda pid: "another-process")
    try:
        assert run.lock_for_down(grace_s=1, poll_s=1) is False
    finally:
        held._up_lock.close()


def test_run_down_grace_from_the_environment(make_run, monkeypatch):
    run = make_run()
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    slept = []
    monkeypatch.setattr(up.time, "sleep", slept.append)
    held = up.Run.__new__(up.Run)
    held.dir = run.dir
    assert held.lock_up() is None
    monkeypatch.setenv("YETO_RUN_DOWN_GRACE_S", "5")
    try:
        assert run.lock_for_down(poll_s=1) is False
        assert len(slept) == 5
        slept.clear()
        monkeypatch.delenv("YETO_RUN_DOWN_GRACE_S")
        assert run.lock_for_down(poll_s=1) is False
        assert len(slept) == up.DOWN_GRACE_S == 60
    finally:
        held._up_lock.close()


def test_run_down_names_the_check_that_did_not_verify(make_run, capsys):
    """NOT verified says which check failed (q06: the trainer's Modal app had not
    converged yet), not only "see down.log"."""
    run = make_run()
    run._cmd = lambda argv, *, log, timeout=None: 1 if argv[-2:] == ["down", "r1-trainer"] else 0
    assert run.down() is False
    out = capsys.readouterr()
    text = out.out + out.err
    assert "NOT verified (trainer (yeto down: cloud/Modal teardown not confirmed))" in text
    assert "pool-status" not in text.split("NOT verified", 1)[1]
    run._cmd = lambda argv, *, log, timeout=None: 0
    assert run.down() is True


def test_run_down_goes_on_past_a_step_that_hangs(make_run, monkeypatch):
    """A step of run down that hits its timeout (a hung dashboard export or Modal
    teardown) must not abort the down: the sampler pools are still stopped (else they
    keep billing), and the down is recorded as not verified."""
    import subprocess  # noqa: PLC0415

    run = make_run()
    del run._cmd  # the real _cmd, over a fake subprocess.run
    monkeypatch.setattr(up, "_alive", lambda pid: False)
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        if "export" in argv or argv[-2:] == ["down", "r1-trainer"]:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(up.subprocess, "run", fake_run)
    assert run.down() is False
    assert any("pool-stop" in argv for argv in seen)
    assert run.state["steps"]["down"]["ok"] is False
