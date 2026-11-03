"""Ports for test servers that bind after the picker returns.

Binding port 0 and closing returns an ephemeral port: until the syncer binds
it, any outgoing connection on the host (another test's client, a pip
download) can be given the same port as its local end, and the syncer exits
with "bind port". Ports here come from below the ephemeral ranges of Linux
(32768+) and macOS (49152+), so only an explicit bind can take them, and one
process never hands out the same port twice.
"""

from __future__ import annotations

import random
import socket
import threading

_LOW, _HIGH = 20000, 32000
_lock = threading.Lock()
_given: set[int] = set()


def _bindable(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def free_ports(count: int) -> list[int]:
    picked: list[int] = []
    with _lock:
        while len(picked) < count:
            if len(_given) >= _HIGH - _LOW:
                _given.clear()
            port = random.randrange(_LOW, _HIGH)
            if port not in _given and _bindable(port):
                _given.add(port)
                picked.append(port)
    return picked


def free_port() -> int:
    return free_ports(1)[0]
