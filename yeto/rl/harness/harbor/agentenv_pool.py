"""Reuse of AgentENV microVMs across tasks, and pre-warmed microVMs.

A fresh AgentENV sandbox pays for a microVM create (and, on a cache miss, the
OCI->rootfs conversion).  :class:`AgentenvSandboxPool` keeps microVMs whose
task has finished and hands them to the next task, mirroring
:class:`~.boat_pool.BoatSandboxPool`:

* release: the microVM is recycled (work and staging directories wiped, then
  checked empty) and kept idle.  One that fails the recycle, or that ran a
  task that failed in its infrastructure, is deleted instead.
* acquire: an idle microVM of the same template is taken; the task's workdir is
  prepared fresh (:meth:`~.agentenv_sandbox.AgentenvSandbox.reuse`).  With no
  idle microVM a new one is created.

Preferred pooling path: AgentENV exposes a snapshot ``/fork`` that clones a
clean microVM in <50 ms, which both avoids a cold create *and* gives each task
a pristine filesystem (stronger isolation than an in-place wipe).  The E2B SDK
(2.2.x) does not surface fork, so it is wired through the client's optional
``fork`` method; until that is available against a real gateway
(TODO(verify on hardware)), the pool uses recycle-based reuse with a cold-start
fallback, exactly like boat.

Cost bounds: an idle microVM is deleted after ``idle_ttl_s`` (checked on every
pool call and by a background reaper), at most ``max_idle`` are kept, each is
reused at most ``max_reuses`` times.  On release an idle microVM's remote TTL is
cut to ``idle_vm_ttl_s`` (so a microVM idle in a dead process stops on its own)
and set back to the task TTL on reuse.

Exit cleanup is shared with boat: this pool registers through
:func:`yeto.rl.harness.harbor.boat_pool.register_for_exit`, so
:func:`~.boat_pool.close_pools_now` (which Miles' ``RolloutExecutor.dispose``
calls) deletes this pool's microVMs too, with no extra wiring at the dispose
site.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from yeto.rl.agentic.loop_local import LoopLocal

from .agentenv_client import AgentenvClient, AgentenvQuota, process_client, process_quota
from .agentenv_sandbox import AgentenvSandbox, SandboxSpec
from .boat_pool import close_pools_now, register_for_exit  # shared exit cleanup

logger = logging.getLogger(__name__)

POOL_ENV = "YETO_AGENTENV_POOL"  # "1" turns reuse on for AgentenvEnvironment
POOL_IDLE_TTL_ENV = "YETO_AGENTENV_POOL_IDLE_TTL_S"
POOL_MAX_IDLE_ENV = "YETO_AGENTENV_POOL_MAX_IDLE"
POOL_IDLE_VM_TTL_ENV = "YETO_AGENTENV_POOL_IDLE_VM_TTL_S"
# "1": acquire by snapshot-forking a pristine clone off an idle microVM of the
# same template (fast, and each task gets a clean filesystem -- stronger
# isolation than an in-place wipe).  Off by default until fork is confirmed on
# real hardware (TODO(verify on hardware)); then make it the default path.
POOL_FORK_ENV = "YETO_AGENTENV_FORK"
DEFAULT_IDLE_VM_TTL_S = 10 * 60

__all__ = ["AgentenvSandboxPool", "PoolStats", "POOL_ENV", "process_pool",
           "close_pools_now", "register_for_exit"]


@dataclass
class _Idle:
    sandbox: AgentenvSandbox
    since: float


@dataclass(frozen=True)
class PoolStats:
    created: int
    reused: int
    recycled: int
    discarded: int
    expired: int
    forked: int
    idle: int


class AgentenvSandboxPool:
    def __init__(self, client: AgentenvClient, quota: AgentenvQuota, *, idle_ttl_s: float = 300.0,
                 max_idle: int = 64, max_reuses: int = 50, idle_vm_ttl_s: int = DEFAULT_IDLE_VM_TTL_S,
                 use_fork: bool = False, sandbox_factory: Callable[..., AgentenvSandbox] = AgentenvSandbox,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if idle_ttl_s <= 0 or max_idle < 0 or max_reuses < 0:
            raise ValueError("pool limits must be positive")
        if idle_vm_ttl_s <= idle_ttl_s:
            raise ValueError(f"idle microVM TTL ({idle_vm_ttl_s} s) must exceed the pool's idle TTL "
                             f"({idle_ttl_s} s)")
        self._use_fork = use_fork
        self._idle_vm_ttl_s = int(idle_vm_ttl_s)
        self._leased: set[AgentenvSandbox] = set()
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
        self._counts = {"created": 0, "reused": 0, "recycled": 0, "discarded": 0, "expired": 0,
                        "forked": 0}

    @property
    def stats(self) -> PoolStats:
        return PoolStats(idle=len(self._idle), **self._counts)

    # -- acquire / release ---------------------------------------------------

    async def acquire(self, spec: SandboxSpec, *, name: str, **sandbox_kwargs: object) -> AgentenvSandbox:
        """A started microVM running ``spec``'s template, ready for the task ``name``."""
        if self._closed:
            raise RuntimeError("agentenv sandbox pool is closed")
        await self._expire()
        if self._use_fork:
            forked = await self._acquire_by_fork(spec, name, sandbox_kwargs)
            if forked is not None:
                return forked
        while (idle := self._take(spec.template)) is not None:
            try:
                await idle.set_ttl(idle.ttl_s)  # back to the task TTL before the task runs
                await idle.reuse(spec, name=name)  # deletes the microVM itself on failure
            except Exception:
                if idle.sandbox_id is not None:
                    await idle.close()
                logger.warning("agentenv pool: reuse of %s failed; trying another microVM",
                               idle.sandbox_id, exc_info=True)
                self._counts["discarded"] += 1
                continue
            self._counts["reused"] += 1
            self._leased.add(idle)
            return idle
        sandbox = self._factory(self._client, self._quota, name=name, template=spec.template,
                                **sandbox_kwargs)
        self._leased.add(sandbox)  # before start: a microVM being created is deleted at exit too
        try:
            await sandbox.start(spec)
        except BaseException:
            self._leased.discard(sandbox)
            raise
        self._counts["created"] += 1
        return sandbox

    async def _acquire_by_fork(self, spec: SandboxSpec, name: str,
                               sandbox_kwargs: dict[str, object]) -> AgentenvSandbox | None:
        """Fork a pristine clone off an idle seed of the template; None if none exists.

        The seed stays idle (fork leaves it unchanged), so one warm microVM per
        template serves every later task as a fast, clean clone.
        """
        seed = self._peek(spec.template)
        if seed is None or seed.sandbox_id is None:
            return None
        clone = self._factory(self._client, self._quota, name=name, template=spec.template,
                              **sandbox_kwargs)
        try:
            ids = await asyncio.to_thread(self._client.fork, seed.sandbox_id, 1, ttl_seconds=clone.ttl_s)
        except Exception:
            logger.warning("agentenv pool: fork off %s failed; falling back", seed.sandbox_id,
                           exc_info=True)
            return None
        try:
            await clone.adopt(ids[0], spec, name=name)
        except Exception:
            logger.warning("agentenv pool: adopting fork %s failed; falling back", ids[0], exc_info=True)
            return None
        self._counts["forked"] += 1
        self._leased.add(clone)
        return clone

    def _peek(self, template: str) -> AgentenvSandbox | None:
        for index in reversed(range(len(self._idle))):
            if self._idle[index].sandbox.template == template:
                return self._idle[index].sandbox
        return None

    def _take(self, template: str) -> AgentenvSandbox | None:
        # Most recently released first: its rootfs cache and microVM are warmest.
        for index in reversed(range(len(self._idle))):
            if self._idle[index].sandbox.template == template:
                return self._idle.pop(index).sandbox
        return None

    async def release(self, sandbox: AgentenvSandbox, *, reusable: bool = True) -> None:
        """Return ``sandbox`` after its task; it is recycled or deleted, never left running."""
        self._leased.discard(sandbox)
        if sandbox.sandbox_id is None:
            return  # already deleted (a failed start or reuse)
        if sandbox.forked:
            # A pristine clone is ephemeral: the seed it came from stays idle, so
            # the clone is killed rather than recycled or kept.
            await sandbox.close()
            self._counts["discarded"] += 1
            return
        if (not reusable or self._closed or sandbox.reuse_count >= self._max_reuses
                or len(self._idle) >= self._max_idle):
            await sandbox.close()
            self._counts["discarded"] += 1
            return
        try:
            await asyncio.shield(sandbox.recycle())
            await asyncio.shield(sandbox.set_ttl(self._idle_vm_ttl_s))
        except Exception:
            logger.warning("agentenv pool: recycle of %s failed; deleting it", sandbox.sandbox_id,
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

    async def prewarm(self, count: int, *, template: str, name_prefix: str = "prewarm") -> int:
        """Create ``count`` idle microVMs from ``template``; return how many started.

        A microVM that fails to start is skipped (logged); the round falls back
        to cold starts for the missing ones.
        """
        async def one(index: int) -> bool:
            name = f"{name_prefix}-{index}"
            sandbox = self._factory(self._client, self._quota, name=name, template=template)
            self._leased.add(sandbox)
            try:
                await sandbox.start_vm(SandboxSpec(template=template))
            except Exception:
                self._leased.discard(sandbox)
                logger.warning("agentenv pool: prewarm %s failed", name, exc_info=True)
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
        """Delete every idle microVM; later releases delete their microVM."""
        self._closed = True
        idle, self._idle = self._idle, []
        if self._reaper is not None:
            self._reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper
        await asyncio.gather(*(e.sandbox.close() for e in idle))

    def close_now(self) -> int:
        """Delete every idle and leased microVM synchronously (process exit); return how many.

        Runs outside the event loop (atexit, a signal handler): it uses
        ``delete_now`` and does not wait for running tasks.
        """
        self._closed = True
        sandboxes = [e.sandbox for e in self._idle] + list(self._leased)
        self._idle, self._leased = [], set()
        for sandbox in sandboxes:
            sandbox.delete_now()
        return len(sandboxes)


def _pool_from_env() -> AgentenvSandboxPool:
    pool = AgentenvSandboxPool(process_client(), process_quota(),
                               idle_ttl_s=float(os.environ.get(POOL_IDLE_TTL_ENV, "") or 300.0),
                               max_idle=int(os.environ.get(POOL_MAX_IDLE_ENV, "") or 64),
                               idle_vm_ttl_s=int(os.environ.get(POOL_IDLE_VM_TTL_ENV, "")
                                                 or DEFAULT_IDLE_VM_TTL_S),
                               use_fork=os.environ.get(POOL_FORK_ENV, "").strip() == "1")
    register_for_exit(pool)  # shares boat's atexit / signal cleanup (close_pools_now)
    return pool


_process_pool: LoopLocal[AgentenvSandboxPool] = LoopLocal(_pool_from_env)


def process_pool() -> AgentenvSandboxPool:
    """The pool shared by every AgentENV sandbox of this process's event loop."""
    return _process_pool.get()
