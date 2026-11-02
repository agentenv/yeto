"""Wall-clock limits of the S25 G1 rehearsal driver (``modal_app.orchestrate``).

No card may wait for nothing (T-G1B g1b-1: the sampler held 1 x H100 for about
30 min after both trainers had finished):

* :class:`Deadline`: the whole run's wall-clock limit. A lost container that
  Modal runs again (or one that hangs) can not keep the cards past it.
* :func:`wait_until`: wait for a call to finish, at most ``limit_s``; the driver
  uses it for the sampler after every trainer returned rc 0 (the sampler then
  has nothing to wait for but its backlog).

Pure functions over ``done()``, ``clock()`` and ``sleep()``: CPU-tested in
``tests/test_s25_g1c_sampler_resume.py``.
"""
from __future__ import annotations

import time
from collections.abc import Callable


class Deadline:
    def __init__(self, limit_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        if float(limit_s) <= 0:
            raise ValueError("limit_s must be > 0")
        self.clock = clock
        self.limit_s = float(limit_s)
        self.started = clock()

    def left(self) -> float:
        return self.limit_s - (self.clock() - self.started)

    def passed(self) -> bool:
        return self.left() <= 0


def wait_until(done: Callable[[], bool], limit_s: float, *, poll_s: float = 15.0,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep,
               deadline: Deadline | None = None) -> bool:
    """True once ``done()``; False when ``limit_s`` (or ``deadline``) passed first."""
    started = clock()
    while True:
        if done():
            return True
        if clock() - started >= float(limit_s) or (deadline is not None and deadline.passed()):
            return False
        sleep(poll_s)
