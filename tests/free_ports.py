"""Ports for test servers that bind after the picker returns.

Binding port 0 and closing returns an ephemeral port: until the syncer binds
it, any outgoing connection on the host (another test's client, a pip
download) can be given the same port as its local end, and the syncer exits
with "bind port". Ports here come from below the ephemeral ranges of Linux
(32768+) and macOS (49152+), so only an explicit bind can take them, and one
process never hands out the same port twice. Test processes on one host (parallel
pytest runs) never hand out the same port either: a port picked here holds a lock file
under ``$TMPDIR/yeto-test-ports`` until the process exits. A syncer started on
``--port 0`` (:mod:`syncer_port`) needs no port from here.
"""

from __future__ import annotations

import fcntl
import os
import random
import socket
import tempfile
import threading

_LOW, _HIGH = 20000, 32000
_lock = threading.Lock()
_given: set[int] = set()
_held: dict[int, int] = {}  # port -> the fd of its lock file (held until the process exits)
_LOCKS = os.path.join(tempfile.gettempdir(), "yeto-test-ports")


def _bindable(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _reserve(port: int) -> bool:
    """Hold ``port``'s lock file; False when another test process on this host holds it."""
    if port in _held:
        return True
    os.makedirs(_LOCKS, exist_ok=True)
    fd = os.open(os.path.join(_LOCKS, f"{port}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    _held[port] = fd
    return True


def free_ports(count: int) -> list[int]:
    picked: list[int] = []
    with _lock:
        while len(picked) < count:
            if len(_given) >= _HIGH - _LOW:
                _given.clear()
            port = random.randrange(_LOW, _HIGH)
            if port not in _given and _reserve(port) and _bindable(port):
                _given.add(port)
                picked.append(port)
    return picked


def free_port() -> int:
    return free_ports(1)[0]
