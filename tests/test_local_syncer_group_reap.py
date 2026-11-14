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
