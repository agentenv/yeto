"""Children that write more than a pipe buffer to stderr must not wedge the caller.

Each case runs a small Python child that writes ~1 MiB to stderr before it
exits. A caller that pipes stderr without draining it blocks the child on a
full pipe forever; the caller here is run in a worker thread with a deadline
and the child is killed by its exact PID if the deadline passes.
"""

import os
import signal
import subprocess
import sys
import threading

from yeto.dashboard.mirror import SshTapeMirror
from yeto.launch import agentic_node_agent as agent

NOISY = [sys.executable, "-c",
         "import sys; sys.stdout.write('ok\\n'); sys.stdout.flush(); "
         "sys.stderr.write('e' * (1 << 20) + 'TAIL'); sys.stderr.flush()"]


def run_bounded(fn, timeout=20.0):
    pids, result = [], {}
    real = subprocess.Popen

    def popen(*a, **kw):
        proc = real(*a, **kw)
        pids.append(proc.pid)
        return proc

    def body():
        result["value"] = fn(popen)

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(timeout)
    hung = t.is_alive()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    t.join(5)
    assert not hung, "caller blocked on a child whose stderr filled the pipe"
    return result.get("value")


def test_tunnel_loop_survives_a_noisy_ssh(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    run_bounded(lambda popen: agent.tunnel_loop(NOISY, events, settle_s=0.05, max_attempts=1,
                                                popen=popen, sleep=lambda s: None))
    import json
    down = [json.loads(x) for x in events.read_text().splitlines()][-1]
    assert down["state"] == "down" and down["exit"] == 0 and down["stderr"].endswith("TAIL")


def test_mirror_survives_a_noisy_stream(tmp_path, monkeypatch):
    local = tmp_path / "mirror" / "tape.jsonl"
    m = SshTapeMirror("host", "/remote", local, command=lambda off: NOISY, sleep=lambda s: None)

    def go(popen):
        monkeypatch.setattr("yeto.dashboard.mirror.subprocess.Popen", popen)
        return m.run_once()

    assert run_bounded(go) == 3
    assert local.read_text() == "ok\n"
    import json
    lost = [json.loads(x) for x in m.sidecar.read_text().splitlines()][-1]
    assert lost["event"] == "dashboard_source_lost" and lost["error"].startswith("e")


def test_upload_code_does_not_wait_on_an_archive_nobody_reads(monkeypatch):
    """ssh refusing at once (auth, host down) must not leave ``git archive``
    blocked on a pipe whose read end the launcher still holds."""
    from yeto.launch.nc import ssh_sampler

    def go(popen):
        monkeypatch.setattr("yeto.launch.nc.ssh_sampler.subprocess.Popen", popen)

        def run(argv, **kw):
            if argv[0] == "ssh":
                return subprocess.run([sys.executable, "-c", "pass"], stdin=kw.get("stdin"), check=False)
            return subprocess.run(argv, **kw)

        return ssh_sampler.upload_code("alias", run=run)

    assert "/yeto/" in run_bounded(go)
