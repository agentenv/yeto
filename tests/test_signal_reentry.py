"""A second SIGTERM must not deadlock or cut short the shutdown work of the first."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.Popen([sys.executable, "-c", textwrap.dedent(code)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.kill(proc.pid, signal.SIGKILL)  # exact pid
        out, err = proc.communicate()
        raise AssertionError(f"child hung (deadlock?)\nstdout={out}\nstderr={err}") from None
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def test_modal_exit_handler_second_signal_during_save_does_not_deadlock():
    res = _run("""
        import os, signal
        from yeto.cloud import preemption as p
        saves, leaves = [], []
        def save(budget):
            saves.append(budget)
            return {"ok": True}
        def leave():
            leaves.append(1)
            os.kill(os.getpid(), signal.SIGTERM)  # the second reclaim signal lands mid-leave
        h = p.ModalExitHandler("0", save=save, leave=leave, emit=lambda e, **f: None,
                               last_save_s=1.0)
        assert h.install(("SIGTERM",)) == ["SIGTERM"]
        os.kill(os.getpid(), signal.SIGTERM)
        assert h.stop.is_set() and h.event is not None
        assert h.handle() is h.event  # the exit hook later: same event, no second save
        print("saves", len(saves), "leaves", len(leaves))
    """)
    assert res.returncode == 0, res.stderr
    assert "saves 1 leaves 1" in res.stdout


def test_cmd_worker_second_sigterm_does_not_cut_backstop_cleanup(tmp_path):
    res = _run(f"""
        import os, signal, sys, time
        os.environ["HOME"] = {str(tmp_path)!r}
        from yeto import cli, launcher, runs
        done = []
        def fake_run(args, on_clusters=None, **kw):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)
        def backstop(name, args):
            os.kill(os.getpid(), signal.SIGTERM)  # `yeto down` retried while cleaning up
            done.append("cleanup")
        launcher.run = fake_run
        cli._backstop_cleanup = backstop
        runs.load_run = lambda name: {{"args": {{}}}}
        runs.update_run = lambda name, **kw: done.append(kw.get("state"))
        rc = cli.cmd_worker("r")
        print("rc", rc, done)
    """)
    assert res.returncode == 0, res.stderr
    assert "cleanup" in res.stdout and "rc 1" in res.stdout
