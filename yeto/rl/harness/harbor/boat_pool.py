"""Reuse of boat VMs across tasks, and pre-warmed VMs (S26 E8).

A fresh boat sandbox pays for a VM create, a wait until the VM is ready,
and an image pull before the task container runs.  :class:`BoatSandboxPool`
keeps VMs whose task has finished and hands them to the next task:

* release: the VM is recycled (all containers, volumes, user networks and
  the staging directory removed, then checked empty) and kept idle.  A VM
  that fails the recycle, or that ran a task that failed in its
  infrastructure, is deleted instead.
* acquire: an idle VM of the same machine type is taken, preferring one
  that already pulled the task image; the task container starts fresh from
  the image (:meth:`~.boat_sandbox.BoatSandbox.reuse`), so no file of the
  previous task is visible.  With no idle VM a new one is started.
* :meth:`BoatSandboxPool.prewarm` starts VMs (optionally with the image
  pulled) before a round needs them.

Cost bounds: an idle VM is deleted after ``idle_ttl_s`` (checked on every
pool call and by a background reaper), at most ``max_idle`` VMs are kept,
a VM is reused at most ``max_reuses`` times, and every VM keeps its boat
TTL as a backstop.  :meth:`BoatSandboxPool.close` deletes all idle VMs.

Idle VMs keep their :class:`~.boat_client.BoatQuota` slot: on boat they are
active sandboxes.  A reuse takes no start from the per-minute budget.

Isolation: tasks that share a VM share its kernel, Docker daemon and image
cache, and see each other only through those; reuse is off by default.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from yeto.rl.agentic.loop_local import LoopLocal

from .boat_client import BoatClient, BoatQuota, process_client, process_quota
from .boat_sandbox import BoatSandbox, ContainerSpec

logger = logging.getLogger(__name__)

POOL_ENV = "YETO_BOAT_POOL"  # "1" turns reuse on for BoatEnvironment
POOL_IDLE_TTL_ENV = "YETO_BOAT_POOL_IDLE_TTL_S"
POOL_MAX_IDLE_ENV = "YETO_BOAT_POOL_MAX_IDLE"


@dataclass
class _Idle:
    sandbox: BoatSandbox
    since: float


@dataclass(frozen=True)
class PoolStats:
    created: int
    reused: int
    recycled: int
    discarded: int
    expired: int
    idle: int


class BoatSandboxPool:
    def __init__(self, client: BoatClient, quota: BoatQuota, *, idle_ttl_s: float = 300.0,
                 max_idle: int = 64, max_reuses: int = 50,
                 sandbox_factory: Callable[..., BoatSandbox] = BoatSandbox,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if idle_ttl_s <= 0 or max_idle < 0 or max_reuses < 0:
            raise ValueError("pool limits must be positive")
        self._client = client
        self._quota = quota
        self._idle_ttl_s = idle_ttl_s
        self._max_idle = max_idle
        self._max_reuses = max_reuses
        self._factory = sandbox_factory
        self._clock = clock
        self._idle: list[_Idle] = []
        self._reaper: asyncio.Task[None] | None = None
        self._closed = False
        self._counts = {"created": 0, "reused": 0, "recycled": 0, "discarded": 0, "expired": 0}

    @property
    def stats(self) -> PoolStats:
        return PoolStats(idle=len(self._idle), **self._counts)

    # -- acquire / release ---------------------------------------------------

    async def acquire(self, spec: ContainerSpec, *, name: str, machine_type: str,
                      **sandbox_kwargs: object) -> BoatSandbox:
        """A started sandbox running ``spec`` as container ``main``."""
        if self._closed:
            raise RuntimeError("boat sandbox pool is closed")
        await self._expire()
        while (idle := self._take(spec.image, machine_type)) is not None:
            try:
                await idle.reuse(spec, name=name)  # deletes the VM itself on failure
            except Exception:
                logger.warning("boat pool: reuse of %s failed; trying another VM", idle.sandbox_id,
                               exc_info=True)
                self._counts["discarded"] += 1
                continue
            self._counts["reused"] += 1
            return idle
        sandbox = self._factory(self._client, self._quota, name=name, machine_type=machine_type,
                                **sandbox_kwargs)
        await sandbox.start(spec)
        self._counts["created"] += 1
        return sandbox

    def _take(self, image: str, machine_type: str) -> BoatSandbox | None:
        candidates = [i for i, e in enumerate(self._idle) if e.sandbox.machine_type == machine_type]
        if not candidates:
            return None
        same_image = [i for i in candidates if self._idle[i].sandbox.image == image]
        # Most recently released first: its image cache and VM are warmest.
        index = (same_image or candidates)[-1]
        return self._idle.pop(index).sandbox

    async def release(self, sandbox: BoatSandbox, *, reusable: bool = True) -> None:
        """Return ``sandbox`` after its task; it is recycled or deleted, never left running."""
        if sandbox.sandbox_id is None:
            return  # already deleted (a failed start or reuse)
        if (not reusable or self._closed or sandbox.reuse_count >= self._max_reuses
                or len(self._idle) >= self._max_idle):
            await sandbox.close()
            self._counts["discarded"] += 1
            return
        try:
            await asyncio.shield(sandbox.recycle())
        except Exception:
            logger.warning("boat pool: recycle of %s failed; deleting it", sandbox.sandbox_id,
                           exc_info=True)
            await sandbox.close()
            self._counts["discarded"] += 1
            return
        except asyncio.CancelledError:
            await sandbox.close()
            raise
        self._idle.append(_Idle(sandbox, self._clock()))
        self._counts["recycled"] += 1
        self._ensure_reaper()

    # -- prewarm -------------------------------------------------------------

    async def prewarm(self, count: int, *, machine_type: str, image: str | None = None,
                      name_prefix: str = "prewarm") -> int:
        """Start ``count`` idle VMs (with ``image`` pulled when given); return how many started.

        A VM that fails to start is skipped (logged); the round falls back to
        cold starts for the missing ones.
        """
        async def one(index: int) -> bool:
            name = f"{name_prefix}-{index}"
            sandbox = self._factory(self._client, self._quota, name=name, machine_type=machine_type)
            spec = ContainerSpec(image=image) if image else None
            try:
                await sandbox.start_vm(spec)
            except Exception:
                logger.warning("boat pool: prewarm %s failed", name, exc_info=True)
                return False
            self._counts["created"] += 1
            await self.release(sandbox)
            return sandbox.sandbox_id is not None

        results = await asyncio.gather(*(one(i) for i in range(count)))
        return sum(results)

    # -- expiry and shutdown -------------------------------------------------

    async def _expire(self) -> None:
        now = self._clock()
        expired = [e for e in self._idle if now - e.since >= self._idle_ttl_s]
        if not expired:
            return
        self._idle = [e for e in self._idle if now - e.since < self._idle_ttl_s]
        self._counts["expired"] += len(expired)
        await asyncio.gather(*(e.sandbox.close() for e in expired))

    def _ensure_reaper(self) -> None:
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.get_running_loop().create_task(self._reap())

    async def _reap(self) -> None:
        while self._idle and not self._closed:
            oldest = min(e.since for e in self._idle)
            await asyncio.sleep(max(0.0, oldest + self._idle_ttl_s - self._clock()) + 0.01)
            await self._expire()

    async def close(self) -> None:
        """Delete every idle VM; later releases delete their VM."""
        self._closed = True
        idle, self._idle = self._idle, []
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        await asyncio.gather(*(e.sandbox.close() for e in idle))


def _pool_from_env() -> BoatSandboxPool:
    import os

    return BoatSandboxPool(process_client(), process_quota(),
                           idle_ttl_s=float(os.environ.get(POOL_IDLE_TTL_ENV, "") or 300.0),
                           max_idle=int(os.environ.get(POOL_MAX_IDLE_ENV, "") or 64))


_process_pool: LoopLocal[BoatSandboxPool] = LoopLocal(_pool_from_env)


def process_pool() -> BoatSandboxPool:
    """The pool shared by every boat sandbox of this process's event loop (see ``process_quota``)."""
    return _process_pool.get()
