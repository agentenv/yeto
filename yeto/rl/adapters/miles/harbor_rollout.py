"""Miles rollout function for the Harbor agentic path: the example's, plus boat cleanup.

Loaded by Miles in the RolloutExecutor Ray actor (``--rollout-function-path``).
The boat sandbox pool registers ``atexit`` there (the actor's pool is built off
the main thread, so no signal handlers), but Ray does not run ``atexit`` when
it ends an actor: idle VMs outlived the GPU run until their TTL.  Miles calls
``RolloutExecutor.dispose()`` from the driver's ``Disposer`` at the end of
``train.py`` (normal end and exceptions), and that calls ``dispose()`` of the
rollout function; this class deletes the pool's VMs there.  A killed actor or
a lost node still relies on the VM TTL.

``generate`` is the Miles example module (``examples/swe-agent-harbor-docker``)
on the workers' ``PYTHONPATH``, as for the stock ``generate.RolloutFn``.
"""

from __future__ import annotations

import asyncio
import logging

from generate import RolloutFn as _ExampleRolloutFn

from yeto.rl.harness.harbor.boat_pool import close_pools_now

logger = logging.getLogger(__name__)


class RolloutFn(_ExampleRolloutFn):
    async def dispose(self) -> None:
        try:
            await super().dispose()
        finally:
            deleted = await asyncio.to_thread(close_pools_now)
            if deleted:
                logger.info("boat pool: deleted %d VM(s) at rollout dispose", deleted)
