"""One object per running asyncio event loop (S25 N10.8c).

The agentic path keeps process-wide guards (the admission gate, the boat
quota) that hold asyncio primitives.  An ``asyncio.Semaphore`` or ``Lock``
binds to the first loop that waits on it; used later from another loop it
raises ``RuntimeError: ... is bound to a different event loop``.

verl runs each ``AgentLoopWorker`` as a Ray async actor, and Miles runs its
rollout on one background loop, so in a training worker there is one loop
per process and :class:`LoopLocal` gives exactly one object per process.
A process that starts several loops (``asyncio.run`` per call in scripts and
tests, verl's ``auto_await`` outside a running loop) gets one object per
loop instead of a crash.  Limits held by such an object are then per loop.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class LoopLocal(Generic[T]):
    """Lazily builds ``factory()`` once for each running event loop."""

    def __init__(self, factory: Callable[[], T]) -> None:
        self._factory = factory
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()  # loops of different threads may ask at once

    def get(self) -> T:
        """The object of the running loop; raises ``RuntimeError`` outside a running loop."""
        loop = asyncio.get_running_loop()
        with self._lock:
            value = self._by_loop.get(loop)
            if value is None:
                value = self._by_loop[loop] = self._factory()
            return value

    def clear(self) -> None:
        """Forget every object (tests, or after the configuration changed)."""
        with self._lock:
            self._by_loop.clear()
