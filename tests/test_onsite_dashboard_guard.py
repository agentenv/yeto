"""The on-site control unit (KillMode=process) stops only run up: its dashboard must end with it."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from yeto.rl import up

ROOT = Path(__file__).resolve().parents[1]


def _run(onsite: bool) -> up.Run:
    run = up.Run.__new__(up.Run)
    run.py, run.cfg = "/opt/v/bin/python", {}
    run.yeto = lambda *a: ["/opt/v/bin/python", "-m", "yeto.cli", *a]
    run.dashboard_args = lambda: ["--store", "s3://b/r"]
    if onsite:
        run.onsite_service = True
    return run


def test_a_run_up_off_site_starts_the_dashboard_as_ever():
    want = ["/opt/v/bin/python", "-m", "yeto.cli", "dashboard", "serve", "--store", "s3://b/r", "--port", "8795"]
    assert _run(False).dashboard_argv("8795") == want
    run = _run(False)
    run.onsite_service = False
    assert run.dashboard_argv("8795") == want


def test_the_on_site_control_unit_starts_the_dashboard_under_a_guard_of_this_process():
    argv = _run(True).dashboard_argv("8795")
    assert argv[:2] == ["/opt/v/bin/python", "-I"] and argv[2].endswith("sampler_engines/parent_guard.py")
    assert argv[3:6] == ["--parent", str(os.getpid()), "--"]
    assert argv[6:] == _run(False).dashboard_argv("8795")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # a zombie is gone too
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                               timeout=10).stdout.strip()
    except subprocess.SubprocessError:
        return True
    return bool(state) and not state.startswith("Z")


def test_the_dashboard_ends_when_the_unit_kills_run_up_alone(tmp_path):
    """run up as the unit's main process spawns the dashboard (its own session, as _spawn does);
    SIGKILL of run up alone (KillMode=process after the stop timeout) ends the dashboard too."""
    pidfile = tmp_path / "dash.pid"
    parent = textwrap.dedent(f"""
        import subprocess, sys, time
        from yeto.rl import up
        run = up.Run.__new__(up.Run)
        run.py, run.cfg, run.onsite_service = sys.executable, {{}}, True
        run.yeto = lambda *a: [sys.executable, "-c",
                               "import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(600)"]
        run.dashboard_args = lambda: []
        p = subprocess.Popen(run.dashboard_argv("0"), start_new_session=True)
        print(p.pid, flush=True)
        time.sleep(600)
    """)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    proc = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE, text=True, env=env)
    guard = 0
    try:
        guard = int(proc.stdout.readline())
        deadline = time.monotonic() + 60
        while not pidfile.exists() or not pidfile.read_text():
            assert time.monotonic() < deadline, "the dashboard did not start"
            time.sleep(0.1)
        dash = int(pidfile.read_text())
        assert _alive(dash) and _alive(guard)
        proc.kill()  # the unit's stop of its main process only
        proc.wait(timeout=30)
        deadline = time.monotonic() + 40
        while _alive(dash) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(dash), "the dashboard outlived run up"
    finally:
        if proc.poll() is None:
            proc.kill()
        for pid in (guard, int(pidfile.read_text() or 0) if pidfile.exists() else 0):
            if pid and _alive(pid):
                os.kill(pid, signal.SIGKILL)
