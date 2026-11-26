"""LocalSyncer must reap the whole syncer process group, even after the
foreground (actor) syncer exited: RL backgrounds the other role syncers in
the same shell, and they keep holding their ports otherwise."""

import os
import signal
import subprocess
import time

from yeto.launch.local_syncer import LocalSyncer


def _make(tmp_path, command):
    syncer = LocalSyncer.__new__(LocalSyncer)
    syncer.command = command
    syncer.binary = str(tmp_path / "missing-binary")
    syncer.log_file = str(tmp_path / "syncer.log")
    syncer.event_tape = str(tmp_path / "tape.jsonl")
    syncer.proc = None
    syncer.tape_forwarder = None
    syncer.tape_run = None
    syncer._log_offset = 0
    syncer._event_offset = 0
    syncer._reaped = False
    return syncer


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_pidfile(path):
    deadline = time.time() + 10
    while time.time() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text().strip())
        time.sleep(0.05)
    raise AssertionError("background role syncer never started")


def _wait_gone(pid, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def _run_case(tmp_path, action):
    pidfile = tmp_path / "critic.pid"
    # Background "critic" syncer (records its own pid), then the foreground
    # "actor" syncer crashes with a non-zero exit.
    command = f"{{ sh -c 'echo $$ > {pidfile}; exec sleep 300' & }} && sleep 0.5 && exit 3"
    syncer = _make(tmp_path, command)
    syncer.start()
    critic = _wait_pidfile(pidfile)
    try:
        syncer.proc.wait(timeout=10)
        assert syncer.probe() == "syncer subprocess exited with code 3"
        assert _alive(critic)
        old_proc = syncer.proc
        action(syncer)
        assert _wait_gone(critic), "orphaned role syncer survived"
        return syncer, old_proc
    finally:
        if _alive(critic):
            os.kill(critic, signal.SIGKILL)


def test_stop_reaps_group_after_foreground_exit(tmp_path):
    _run_case(tmp_path, lambda s: s.stop())


def test_restart_reaps_old_group_first(tmp_path):
    def restart(s):
        s.command = "sleep 300"
        s.restart()

    syncer, old_proc = _run_case(tmp_path, restart)
    try:
        assert syncer.proc is not old_proc
        assert syncer.proc.poll() is None
    finally:
        syncer.stop()
        assert syncer.proc.poll() is not None


def test_second_stop_never_signals_reused_pgid(tmp_path, monkeypatch):
    syncer = _make(tmp_path, "sleep 300")
    syncer.start()
    syncer.stop()
    assert syncer.proc.poll() is not None
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *a: calls.append(a))
    syncer.stop()
    syncer.stop()
    assert calls == []


def test_signal_tree_tolerates_vanished_group(tmp_path):
    syncer = _make(tmp_path, "true")
    syncer.start()
    syncer.proc.wait(timeout=10)
    syncer._signal_tree(signal.SIGTERM)  # must not raise
    assert not syncer._group_alive()


def test_restart_rearms_reaping(tmp_path):
    syncer = _make(tmp_path, "sleep 300")
    syncer.start()
    syncer.stop()
    assert syncer._reaped
    syncer.start()
    try:
        assert not syncer._reaped
    finally:
        syncer.stop()
    assert syncer.proc.poll() is not None


def test_a_stop_while_a_start_runs_never_signals_the_reaped_group(tmp_path, monkeypatch):
    """L5: the old process is reaped (its process group id may be reused). A stop from another
    thread (an on-site service's fence) while start() is inside its Popen signals nothing of
    the old group, and the new process is not taken as reaped."""
    import threading

    from yeto.launch import local_syncer as ls

    syncer = _make(tmp_path, "exec sleep 30")
    syncer.stop_grace_s, syncer.env = 1.0, None
    old = subprocess.Popen(["true"], start_new_session=True)
    old.wait()
    syncer.proc, syncer._reaped = old, True
    entered, go = threading.Event(), threading.Event()
    real_popen, real_killpg = subprocess.Popen, os.killpg
    signalled = []

    def popen(*args, **kwargs):
        entered.set()
        go.wait(10)
        return real_popen(*args, **kwargs)

    def killpg(pid, sig):
        signalled.append((pid, sig))
        return real_killpg(pid, sig)

    monkeypatch.setattr(ls.subprocess, "Popen", popen)
    monkeypatch.setattr(ls.os, "killpg", killpg)
    starting = threading.Thread(target=syncer.start)
    starting.start()
    try:
        assert entered.wait(10)
        syncer._stop_tree()  # another thread's stop, while the start is inside Popen
        assert all(pid != old.pid for pid, _ in signalled), signalled
    finally:
        go.set()
        starting.join(10)
    try:
        assert syncer.proc is not old and syncer._reaped is False
    finally:
        syncer._stop_tree()
    assert syncer._reaped is True and syncer.proc.poll() is not None
