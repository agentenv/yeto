"""Adapter from yeto's role-based engine ports to upstream Miles (design D1).

Every call into upstream Miles is concentrated in this package so that an
upstream pin bump only touches it. Importing the package is cheap: miles, ray
and torch are imported lazily inside the functions that need them.

Modules:

* ``config``            RLRunConfig + AlgorithmSpec -> upstream Miles argv (3.1)
* ``rollout``           RolloutPool over InferenceController/RolloutExecutor (3.2)
* ``rollout_meta_hook`` in-rollout-process metadata extraction (3.2, D3)
* ``trainer``           TrainerGroup over the single-cell actor TrainGroup (3.3)
* ``state_plugin``      code run *inside* each Megatron rank via run_plugin (3.4)
* ``state``             PolicyState over ``run_plugin`` (3.4, D4)
* ``publish``           Publisher over upstream ``update_weights`` (3.5)
* ``placement``         read-only placement description + rewrite detection (3.6)
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

__all__ = ["LoopRunner"]


class LoopRunner:
    """Drive upstream Miles' async handles from the synchronous ports.

    One persistent event loop is used for the lifetime of an island so that
    handles bound to a loop (RPC clients, Ray async refs) stay valid. Plain
    (non-awaitable) values are returned unchanged, which lets tests use
    synchronous fakes.
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

    def close(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()


async def _await(value: Any) -> Any:
    return await value
