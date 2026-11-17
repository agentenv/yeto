"""Fake sleeps that only the code under test sees.

``time`` and ``asyncio`` are one module object for the whole test process, so
``monkeypatch.setattr(<module>.time, "sleep", fake)`` replaces ``time.sleep`` for
every thread: a thread an earlier test left running (a phase profiler, a client
redialing, a prefetcher) then has its sleeps counted, advances the fake clock or
gets the test's stop exception. These helpers keep the fake to the module under
test; every other thread keeps the real sleep.

A module that did ``from time import sleep`` holds its own name: patch that
directly, ``monkeypatch.setattr(module, "sleep", fake)``.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable


class ModuleView:
    """``real`` (a module) with some attributes replaced; every other attribute is
    the real module's, so ``time.time`` / ``asyncio.Event`` keep working."""

    def __init__(self, real: Any, **overrides: Any) -> None:
        self.__dict__.update(overrides)
        self._real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def fake_module_sleep(monkeypatch, module, sleep: Callable, *, name: str = "time") -> ModuleView:
    """Give ``module`` its own view of ``module.<name>`` (``time`` or ``asyncio``)
    whose ``sleep`` is ``sleep``. Returns the view, which a test may patch further
    (``monkeypatch.setattr(view, "time", ...)``) without touching the real module."""
    view = ModuleView(getattr(module, name), sleep=sleep)
    monkeypatch.setattr(module, name, view)
    return view


def fake_caller_sleep(monkeypatch, sleep: Callable, callers: tuple[str, ...]) -> None:
    """For code that does ``import time`` inside a function (no module attribute to
    patch): replace ``time.sleep`` but hand only the calls made from a module in
    ``callers`` (or a submodule) to ``sleep``; every other caller really sleeps."""
    real = time.sleep

    def caller_sleep(seconds):
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if any(caller == c or caller.startswith(c + ".") for c in callers):
            return sleep(seconds)
        return real(seconds)

    monkeypatch.setattr(time, "sleep", caller_sleep)
