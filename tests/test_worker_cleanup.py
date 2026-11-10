"""`yeto down` stops every `_worker <run>` process by PID, also one the run record lost,
and `yeto launch` names `yeto down <run>` when a worker is still alive."""

from __future__ import annotations

import subprocess
import sys
import time
from types import SimpleNamespace

from yeto import cli, runs


def _fake_worker(name: str) -> subprocess.Popen:
    # argv carries the worker's tokens; the process just sleeps
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)",
                             "-m", "yeto.cli", "_worker", name], start_new_session=True)
    deadline = time.time() + 10
    while proc.pid not in runs.worker_pids(name) and time.time() < deadline:
        time.sleep(0.1)
    return proc


def test_down_stops_a_worker_the_record_lost(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    name = f"wk-{time.time_ns()}"
    runs.create_run(name, {"cluster_prefix": name})
    worker = _fake_worker(name)
    try:
        assert worker.pid in runs.worker_pids(name)
        monkeypatch.setattr(cli, "_fleet_args_error", lambda args: None)
        code = cli.cmd_launch(SimpleNamespace(cluster_prefix=name, gpu="modal:8xh100"))
        assert code == 1
        assert f"`yeto down {name}`" in capsys.readouterr().err
        assert cli.cmd_down(SimpleNamespace(run=name)) == 0
        worker.wait(timeout=20)
        assert runs.worker_pids(name) == []
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_a_reaped_or_foreign_pid_is_not_a_live_worker():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert not runs.is_alive(proc.pid)
    assert runs.worker_pids("no-such-run-name") == []
