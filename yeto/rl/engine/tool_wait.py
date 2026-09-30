"""In-flight tool-wait trajectory count (rl-infra-spec 1.7; drain probe for 3.3 X5).

``tool_wait_seconds`` (rollout metadata) is a per-round total and cannot answer
"is anybody waiting on a tool *right now*?". This module gives the instantaneous
count that the admission fence / drain needs: continue only when the router
in-flight count is 0 AND no trajectory is inside a tool call.

Pieces:

* :class:`ToolWaitBoard` - thread-safe state (enter/exit/snapshot). In
  production one instance lives in a named Ray actor per island
  (:func:`board_actor`), so the tool code in the rollout process and the
  driver/controller process see the same count.
* :func:`tool_wait_scope` - context manager (sync and async) that tool-calling
  generate code wraps around each tool call; it reports to a local board or to
  the actor handle.
* :func:`read_tool_wait` + :func:`drain_blockers` - the reader side used by the
  drain (INFRA-E1): fail closed when either count is unknown.

Pure except :func:`board_actor` (lazy ``ray`` import).
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

BOARD_ACTOR_PREFIX = "yeto-rl-tool-wait"


class ToolWaitError(RuntimeError):
    """Unbalanced enter/exit (double enter or exit of an unknown trajectory)."""


@dataclass(frozen=True)
class ToolWaitSnapshot:
    in_flight: int  # trajectories currently inside a tool call
    trajectory_ids: tuple[str, ...]
    oldest_age_s: float | None  # longest current wait, None when idle
    generation: int  # bumps on every enter/exit; lets a reader detect change
    entered_total: int
    exited_total: int


class ToolWaitBoard:
    """Instantaneous tool-wait state of one island. Thread-safe."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._active: dict[str, float] = {}
        self._generation = 0
        self._entered = 0
        self._exited = 0

    def enter(self, trajectory_id: str) -> int:
        with self._lock:
            if trajectory_id in self._active:
                raise ToolWaitError(f"trajectory {trajectory_id!r} already in a tool call")
            self._active[trajectory_id] = self._clock()
            self._generation += 1
            self._entered += 1
            return self._generation

    def exit(self, trajectory_id: str) -> int:
        with self._lock:
            if self._active.pop(trajectory_id, None) is None:
                raise ToolWaitError(f"trajectory {trajectory_id!r} is not in a tool call")
            self._generation += 1
            self._exited += 1
            return self._generation

    def snapshot(self) -> ToolWaitSnapshot:
        with self._lock:
            now = self._clock()
            starts = list(self._active.values())
            return ToolWaitSnapshot(
                in_flight=len(self._active),
                trajectory_ids=tuple(sorted(self._active)),
                oldest_age_s=(now - min(starts)) if starts else None,
                generation=self._generation,
                entered_total=self._entered,
                exited_total=self._exited,
            )


def _call(target: Any, method: str, *args: Any) -> Any:
    """``board.method(*args)`` for a local board, ``handle.method.remote(*args)`` for an actor."""
    fn = getattr(target, method)
    remote = getattr(fn, "remote", None)
    return remote(*args) if callable(remote) else fn(*args)


def _resolve(value: Any) -> Any:
    if type(value).__name__ == "ObjectRef":  # ray.ObjectRef without importing ray
        import ray

        return ray.get(value)
    return value


@contextmanager
def tool_wait_scope(board: Any, trajectory_id: str):
    """Sync scope: the trajectory counts as tool-waiting inside the ``with`` body."""
    _resolve(_call(board, "enter", trajectory_id))
    try:
        yield
    finally:
        _resolve(_call(board, "exit", trajectory_id))


@asynccontextmanager
async def async_tool_wait_scope(board: Any, trajectory_id: str):
    """Async scope for generate functions running on an event loop."""

    async def call(method: str) -> None:
        value = _call(board, method, trajectory_id)
        if inspect.isawaitable(value):  # ray ObjectRef is awaitable
            await value

    await call("enter")
    try:
        yield
    finally:
        await call("exit")


def read_tool_wait(board: Any) -> ToolWaitSnapshot:
    """Snapshot from a local board or an actor handle (blocking)."""
    return _resolve(_call(board, "snapshot"))


def drain_blockers(
    router_in_flight: int | None, tool_wait: ToolWaitSnapshot | None
) -> list[str]:
    """3.3 X5 drain condition; empty = drained. Unknown counts fail closed."""
    out = []
    if router_in_flight is None:
        out.append("router in-flight count unknown")
    elif router_in_flight > 0:
        out.append(f"{router_in_flight} engine requests in flight")
    if tool_wait is None:
        out.append("tool-wait count unknown")
    elif tool_wait.in_flight > 0:
        out.append(f"{tool_wait.in_flight} trajectories waiting on tools")
    return out


def board_actor(learner_id: int, *, namespace: str | None = None) -> Any:
    """The island's named ``ToolWaitBoard`` actor (created on first use)."""
    import ray

    cls = ray.remote(num_cpus=0, max_concurrency=64)(ToolWaitBoard)
    return cls.options(
        name=f"{BOARD_ACTOR_PREFIX}-{int(learner_id)}",
        namespace=namespace,
        get_if_exists=True,
    ).remote()
