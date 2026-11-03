"""Syncer subprocesses that a test can always reap and whose output never
blocks them.

A launcher syncer command runs under ``bash -c``; with several shards bash
backgrounds one syncer per shard and supervises them. Signalling bash alone
leaves those syncers running with ppid 1, holding their ports and the pipe
``communicate()`` reads. ``group_popen`` starts the command in a new session
so ``kill_group`` stops bash and every syncer it started; the autouse fixture
in conftest reaps whatever a failed or timed-out test left behind.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import tempfile

_LIVE: list[subprocess.Popen] = []


class _FileOutputPopen(subprocess.Popen):
    """A Popen whose stdout is a file: ``communicate`` waits, then returns
    everything the process wrote; it may be called again (a fixture's
    teardown after the test's own cleanup)."""

    def communicate(self, input=None, timeout=None):  # noqa: A002 - Popen's signature
        if input is not None:
            raise ValueError("a syncer process takes no input")
        self.wait(timeout)
        self.stdout.seek(0)
        return self.stdout.read(), None


def group_popen(argv, **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` in a new session. ``stdout=subprocess.PIPE`` is
    served from a temporary file instead of a pipe: the tests read a syncer's
    output only after it exits, and a syncer whose pipe buffer is full blocks
    in its log write and stalls its rounds. ``process.stdout.read()`` and
    ``communicate()`` return what the process wrote so far."""
    log = None
    if kwargs.get("stdout") is subprocess.PIPE:
        log = tempfile.NamedTemporaryFile(prefix="syncer-", suffix=".log", delete=False)
        kwargs["stdout"] = log
    try:
        popen = subprocess.Popen if log is None else _FileOutputPopen
        process = popen(argv, start_new_session=True, **kwargs)
    finally:
        if log is not None:
            log.close()
    if log is not None:
        text = kwargs.get("text") or kwargs.get("universal_newlines") or kwargs.get("encoding")
        process.stdout = open(log.name, "r" if text else "rb")
        os.unlink(log.name)
    _LIVE.append(process)
    return process


def kill_group(process: subprocess.Popen, sig: int = signal.SIGKILL) -> None:
    """Signal the process group even when its leader already exited: the
    leader's children keep the group alive."""
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def reap_groups() -> None:
    while _LIVE:
        kill_group(_LIVE.pop())


atexit.register(reap_groups)
