"""A server engine that is killed (start timeout, stop past its grace) ends its
whole process group: tensor-parallel workers / schedulers the leader forked must
not outlive it holding GPU memory (CPU test with real processes)."""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

import pytest

from yeto.rl.adapters.sampler_engines.server import ServerConfig, ServerEngine

pytestmark = pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")

LEADER = """
import os, signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN if sys.argv[2] == "stubborn" else signal.SIG_DFL)
child = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(600)"])
open(sys.argv[1], "w").write(str(child.pid))
if sys.argv[2] == "exits":
    time.sleep(0.3)
    sys.exit(3)  # the leader dies while starting (or serving) and leaves its worker
time.sleep(600)
"""


class _Engine(ServerEngine):
    ENGINE = "groupkill"

    def __init__(self, config: ServerConfig, script: Path, pidfile: Path, mode: str) -> None:
        super().__init__(config, http=lambda *a: (0, None), sleep=lambda s: time.sleep(0.05))
        self._argv = [sys.executable, str(script), str(pidfile), mode]

    def argv(self, model_dir: Path) -> list[str]:
        return list(self._argv)

    def preflight(self, model_dir: Path) -> None:
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie (reparented, not yet reaped) counts as gone
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        import subprocess
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
        return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _wait_pid(pidfile: Path) -> int:
    for _ in range(200):
        if pidfile.exists() and pidfile.read_text().strip():
            return int(pidfile.read_text())
        time.sleep(0.05)
    raise AssertionError("worker never started")


def _gone(pid: int) -> bool:
    for _ in range(100):
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _config(tmp_path: Path, port: int, **kw) -> ServerConfig:
    return ServerConfig(base_dir=tmp_path, card_model="cpu", device_family="cpu", engine_version="0",
                        precision="bf16", quant_scheme="bf16", topology="tp1", max_model_len=16,
                        port=port, **kw)


def test_start_timeout_kill_ends_the_workers_too(tmp_path):
    script = tmp_path / "leader.py"
    script.write_text(LEADER)
    pidfile = tmp_path / "worker.pid"
    engine = _Engine(_config(tmp_path, 47311, start_timeout_s=2.0), script, pidfile, "plain")
    worker = None
    try:
        with pytest.raises(Exception, match="not ready"):
            engine.start(None, 1)
        worker = _wait_pid(pidfile)
        assert _gone(worker), "the leader's worker survived the start-timeout kill"
    finally:
        if worker is not None and _alive(worker):
            os.kill(worker, signal.SIGKILL)


def test_stop_past_grace_ends_the_workers_too(tmp_path):
    script = tmp_path / "leader.py"
    script.write_text(LEADER)
    pidfile = tmp_path / "worker.pid"
    engine = _Engine(_config(tmp_path, 47312, stop_grace_s=0.5), script, pidfile, "stubborn")
    engine._process = engine._spawn(engine.argv(tmp_path), {})
    worker = _wait_pid(pidfile)
    try:
        engine.stop()
        assert _gone(worker), "the leader's worker survived stop()"
    finally:
        if _alive(worker):
            os.kill(worker, signal.SIGKILL)


def test_leader_exit_during_start_ends_the_workers_too(tmp_path):
    script = tmp_path / "leader.py"
    script.write_text(LEADER)
    pidfile = tmp_path / "worker.pid"
    engine = _Engine(_config(tmp_path, 47313, start_timeout_s=30.0), script, pidfile, "exits")
    worker = None
    try:
        with pytest.raises(Exception, match="exited with code 3 during start"):
            engine.start(None, 1)
        worker = _wait_pid(pidfile)
        assert _gone(worker), "the worker of a leader that died while starting survived"
        assert engine.last_exit == {"code": 3, "during": "start", "normal": False}
    finally:
        if worker is not None and _alive(worker):
            os.kill(worker, signal.SIGKILL)


def test_leader_exit_while_serving_ends_the_workers_too(tmp_path):
    script = tmp_path / "leader.py"
    script.write_text(LEADER)
    pidfile = tmp_path / "worker.pid"
    engine = _Engine(_config(tmp_path, 47314), script, pidfile, "exits")
    engine._process = engine._spawn(engine.argv(tmp_path), {})
    worker = _wait_pid(pidfile)
    try:
        engine._process.wait(timeout=30)
        with pytest.raises(Exception, match="while serving"):
            engine.call("GET", "/health")
        assert _gone(worker), "the worker of a leader that died while serving survived"
    finally:
        if _alive(worker):
            os.kill(worker, signal.SIGKILL)
