"""A test syncer on the port it bound itself.

``--port 0`` lets the OS pick a free port at the bind, and the syncer logs the port it
took (``syncer listening port=N``). A port picked before the start (``free_port``) can
be taken by another test process before the syncer binds it: that syncer exits with
"bind port", and its test connects to the other process's syncer instead (a HELLO HMAC
mismatch, a member fenced by another test's retire). A test reading the port from its
own syncer's log never reaches another one.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

LISTENING = re.compile(r"syncer listening port=(\d+)")
_PORT_FLAG = re.compile(r"(--port[ =])\d+")


def on_port_zero(command: str) -> str:
    """``command`` (one syncer invocation) with its ``--port`` set to 0."""
    out, count = _PORT_FLAG.subn(r"\g<1>0", command)
    if count != 1:
        raise ValueError(f"expected one --port in the syncer command, found {count}: {command[:200]}")
    return out


def _output(process: subprocess.Popen, log_file: Path | None) -> str:
    if log_file is not None:
        path = Path(log_file)
        return path.read_text(errors="replace") if path.exists() else ""
    # group_popen serves stdout from a file: read it without moving the caller's position
    data = os.pread(process.stdout.fileno(), 1 << 22, 0)
    return data.decode(errors="replace")


def bound_port(process: subprocess.Popen, *, log_file: Path | None = None, timeout: float = 30.0) -> int:
    """The port ``process`` (a syncer started on ``--port 0``) listens on, read from its
    output (``log_file``, or its stdout as :func:`process_groups.group_popen` serves it).
    Raises when it exits or does not listen within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        text = _output(process, log_file)
        found = LISTENING.search(text)
        if found:
            return int(found.group(1))
        if process.poll() is not None:
            raise RuntimeError(f"syncer exited {process.returncode} before it listened: {text[-2000:]}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"syncer did not listen within {timeout:.0f}s: {text[-2000:]}")
        time.sleep(0.02)
