"""One persistent event loop that drives an engine's async handles from
yeto's synchronous ports (moved from ``adapters/miles/__init__.py``; the verl
full-parameter island uses the same runner, S27 G-DV)."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

__all__ = ["LoopRunner"]


class LoopRunner:
    """Drive an engine's async handles from the synchronous ports.

    One persistent event loop is used for the lifetime of an island so that
    handles bound to a loop (RPC clients, Ray async refs) stay valid. The loop
    is never made the thread's current loop. Plain (non-awaitable) values are
    returned unchanged, which lets tests use synchronous fakes.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop

    def run(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return self.loop.run_until_complete(_await(value))
        return value

    def spawn(self, value: Any) -> asyncio.Future:
        """Start ``value`` on the island loop and return its task at once; a later
        ``run(task)`` waits for its result.

        Before this returns the coroutine runs to its first suspension, so a call
        into a Ray actor is already sent and the actor works while the caller does
        other things. The task then advances whenever the loop runs (every later
        ``run``). A plain value comes back as a finished future."""
        if not inspect.isawaitable(value):
            future = self.loop.create_future()
            future.set_result(value)
            return future
        task = self.loop.create_task(_await(value))
        self.loop.run_until_complete(asyncio.sleep(0))
        return task

    def close(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()


async def _await(value: Any) -> Any:
    return await value
