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
* :class:`HarnessBoard` / :class:`HarnessSnapshot` (IR-2, codex-harness
  R-IR): harness sessions in flight, live sandbox leases (idle ones included,
  each with a hard deadline) and the per-island admission switch
  (``allow_new_session``) the drain closes first. ``drain_blockers`` takes the
  harness snapshot as a third input; ``None`` is unknown and fails closed.

Pure except :func:`board_actor` (lazy ``ray`` import).
"""

from __future__ import annotations

import inspect
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

BOARD_ACTOR_PREFIX = "yeto-rl-tool-wait"
HARNESS_BOARD_ACTOR_PREFIX = "yeto-rl-harness"


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


# -- IR-2: harness sessions / sandbox leases / admission --------------------

TITO_CHAIN_BREAK_REASONS = (
    "retry_fork", "history_rewrite", "template_drops_reasoning", "compaction_window",
)


@dataclass(frozen=True)
class HarnessSnapshot:
    """Instantaneous harness state of one island (R-IR-2).

    ``env_live`` counts every live sandbox lease, idle ones included; each
    lease carries a hard deadline, so a drain waiting on ``env_live`` is
    bounded by ``latest_lease_deadline`` (expired leases are force-released
    by :meth:`HarnessBoard.snapshot` and counted in ``leases_expired_total``,
    an infrastructure error, never a reward).
    """

    in_flight: int  # harness sessions currently open
    env_live: int  # sandbox leases alive (busy or idle)
    generation: int  # bumps on every change
    latest_lease_deadline: float | None = None  # monotonic clock; None = no lease
    leases_expired_total: int = 0
    tito_session_mismatch: int = 0
    tito_chain_breaks: Mapping[str, int] = field(default_factory=dict)
    policy_age_violation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tito_chain_breaks", dict(self.tito_chain_breaks))


# Explicit "this island runs no harness" value: the pool passes
# HARNESS_ZERO for stock / non-agentic profiles instead of None (unknown).
HARNESS_ZERO = HarnessSnapshot(in_flight=0, env_live=0, generation=0)


class HarnessAdmissionError(RuntimeError):
    """A session was opened while admission to its member was closed."""


class HarnessBoard:
    """Thread-safe harness state + admission switch of one island.

    Admission: ``close_admission(members)`` (drain, first step) makes
    ``allow_new_session(member)`` False for those members; the agent/gateway
    calls it before ``POST /sessions`` and, on False, re-targets a non-drained
    member or waits for ``open_admission``. ``enter_session`` enforces it
    (fail closed) so a caller that skipped the check cannot slip in.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, str | None] = {}  # session id -> member
        self._leases: dict[str, float] = {}  # lease id -> hard deadline
        self._closed: set[str] = set()
        self._generation = 0
        self._expired = 0
        self._session_mismatch = 0
        self._chain_breaks: dict[str, int] = {}
        self._policy_age = 0

    # admission ------------------------------------------------------------
    def allow_new_session(self, member: str | None = None) -> bool:
        with self._lock:
            return member not in self._closed and "*" not in self._closed

    def close_admission(self, members: Iterable[str] | None = None) -> None:
        with self._lock:
            self._closed |= {"*"} if members is None else {str(m) for m in members}
            self._generation += 1

    def open_admission(self, members: Iterable[str] | None = None) -> None:
        with self._lock:
            if members is None:
                self._closed.clear()
            else:
                self._closed -= {str(m) for m in members}
            self._generation += 1

    # sessions -------------------------------------------------------------
    def enter_session(self, session_id: str, member: str | None = None) -> int:
        with self._lock:
            if member in self._closed or "*" in self._closed:
                raise HarnessAdmissionError(
                    f"session {session_id!r}: admission to member {member!r} is closed (drain)"
                )
            if session_id in self._sessions:
                raise ToolWaitError(f"harness session {session_id!r} already open")
            self._sessions[session_id] = member
            self._generation += 1
            return self._generation

    def exit_session(self, session_id: str) -> int:
        with self._lock:
            if session_id not in self._sessions:
                raise ToolWaitError(f"harness session {session_id!r} is not open")
            del self._sessions[session_id]
            self._generation += 1
            return self._generation

    # sandbox leases (idle included; hard deadline each) --------------------
    def lease_acquired(self, lease_id: str, *, deadline: float) -> int:
        with self._lock:
            if lease_id in self._leases:
                raise ToolWaitError(f"sandbox lease {lease_id!r} already live")
            self._leases[lease_id] = float(deadline)
            self._generation += 1
            return self._generation

    def lease_released(self, lease_id: str) -> int:
        with self._lock:
            if self._leases.pop(lease_id, None) is None:
                raise ToolWaitError(f"sandbox lease {lease_id!r} is not live")
            self._generation += 1
            return self._generation

    def expire_leases(self) -> tuple[str, ...]:
        """Force-release leases past their hard deadline (infrastructure error)."""
        with self._lock:
            return self._expire_locked()

    def _expire_locked(self) -> tuple[str, ...]:
        now = self._clock()
        dead = tuple(sorted(k for k, d in self._leases.items() if d <= now))
        for k in dead:
            del self._leases[k]
        if dead:
            self._expired += len(dead)
            self._generation += 1
        return dead

    # counters (IR-4) -------------------------------------------------------
    def record_session_mismatch(self, n: int = 1) -> None:
        with self._lock:
            self._session_mismatch += int(n)
            self._generation += 1

    def record_chain_break(self, reason: str, n: int = 1) -> None:
        if reason not in TITO_CHAIN_BREAK_REASONS:
            raise ValueError(f"unknown tito chain break reason {reason!r}")
        with self._lock:
            self._chain_breaks[reason] = self._chain_breaks.get(reason, 0) + int(n)
            self._generation += 1

    def record_policy_age_violation(self, n: int = 1) -> None:
        with self._lock:
            self._policy_age += int(n)
            self._generation += 1

    def snapshot(self) -> HarnessSnapshot:
        with self._lock:
            self._expire_locked()
            return HarnessSnapshot(
                in_flight=len(self._sessions),
                env_live=len(self._leases),
                generation=self._generation,
                latest_lease_deadline=max(self._leases.values()) if self._leases else None,
                leases_expired_total=self._expired,
                tito_session_mismatch=self._session_mismatch,
                tito_chain_breaks=dict(self._chain_breaks),
                policy_age_violation=self._policy_age,
            )


def read_harness(board: Any) -> HarnessSnapshot:
    """Snapshot from a local harness board or an actor handle (blocking)."""
    return _resolve(_call(board, "snapshot"))


def harness_board_actor(learner_id: int, *, namespace: str | None = None) -> Any:
    """The island's named ``HarnessBoard`` actor (created on first use)."""
    import ray

    cls = ray.remote(num_cpus=0, max_concurrency=64)(HarnessBoard)
    return cls.options(
        name=f"{HARNESS_BOARD_ACTOR_PREFIX}-{int(learner_id)}",
        namespace=namespace,
        get_if_exists=True,
    ).remote()


def drain_blockers(
    router_in_flight: int | None,
    tool_wait: ToolWaitSnapshot | None,
    harness: HarnessSnapshot | None,
) -> list[str]:
    """3.3 X5 drain condition; empty = drained. Unknown counts fail closed.

    ``harness`` (IR-2): None = unknown -> blocker; a non-agentic island passes
    :data:`HARNESS_ZERO` explicitly. ``env_live`` counts idle sandboxes too; it
    is bounded by the leases' hard deadlines (see :class:`HarnessSnapshot`).
    """
    out = []
    if router_in_flight is None:
        out.append("router in-flight count unknown")
    elif router_in_flight > 0:
        out.append(f"{router_in_flight} engine requests in flight")
    if tool_wait is None:
        out.append("tool-wait count unknown")
    elif tool_wait.in_flight > 0:
        out.append(f"{tool_wait.in_flight} trajectories waiting on tools")
    if harness is None:
        out.append("harness counts unknown")
    else:
        if harness.in_flight > 0:
            out.append(f"{harness.in_flight} harness sessions in flight")
        if harness.env_live > 0:
            out.append(f"{harness.env_live} sandboxes live")
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
