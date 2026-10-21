"""Suspend agentic trajectories between model turns across a round boundary (S26 E9.5).

A round's cut-off can find trajectories in the middle of an episode.  Instead
of aborting them (their sandbox, agent session and recorded tokens would be
thrown away), the backend closes a :class:`TurnGate`:

* a model call that arrives while the gate is closed waits until the next
  round opens it (a running tool call is not interrupted; its result is
  appended to the conversation as usual);
* a model call the engine aborted at the cut-off is sent again after the gate
  opens (the aborted call recorded no tokens, so the retry is a new turn
  sampled by the newer policy).

Everything the trajectory holds stays alive in the process while it waits:
the Harbor trial coroutine with its sandbox, the agent process in the sandbox,
and the recorder or session-server session with the tokens recorded so far.
A process restart loses all of it, so a suspended trajectory never survives a
restart: the cut keeps only a reference (:data:`SESSION_REF_PREFIX`) and the
restore discards it as ABORTED (``agentic_session_lost``), never re-running the
task under the same trajectory id.

Time spent waiting is the backend's, not the policy's:

* :class:`SuspendLedger` counts it per trajectory (``suspended_seconds``,
  ``suspensions``, ``suspend_retried_turns`` in the sample's agent metrics);
* the outer wall-clock cap of :func:`yeto.rl.agentic.run_agent_task` is
  extended by it;
* the harness's own agent timeout cannot be paused (Harbor runs it), so a
  trajectory that was suspended and then hit a time limit is ABORTED, not a
  trainable 0 (:func:`after_suspension`).

The gate holds asyncio primitives, so a backend keeps one gate per event loop
(:class:`~yeto.rl.agentic.loop_local.LoopLocal`).  :func:`suspended_references`
reads every gate of the process under a thread lock: the cut exporter runs on
another thread than the rollout loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace

from .types import AgentTrajectory, Outcome

# Cut entries of suspended trajectories carry this session reference prefix.
SESSION_REF_PREFIX = "agentic-suspended:"
# Written by the removed Codex harness; still recognised when an old cut is restored.
LEGACY_SESSION_REF_PREFIXES = ("codex-suspended:",)
SESSION_LOST_REASON = "agentic_session_lost"
# A time limit hit after a suspension: the policy's share of the time is unknown.
TIMEOUT_AFTER_SUSPENSION_REASON = "time_limit_after_suspension"
TIME_LIMIT_EXCEPTIONS = frozenset({"AgentTimeoutError"})


def is_suspended_ref(session_ref: object) -> bool:
    text = str(session_ref or "")
    return text.startswith(SESSION_REF_PREFIX) or text.startswith(LEGACY_SESSION_REF_PREFIXES)


@dataclass
class SuspendLedger:
    """Suspension bookkeeping of one trajectory."""

    trajectory_id: str
    group_id: str = ""
    task_id: str = ""
    suspensions: int = 0  # distinct gate closures this trajectory waited through
    suspended_seconds: float = 0.0
    retried_turns: int = 0  # model calls aborted at a cut-off and sent again
    waiting: int = 0  # model calls waiting at the closed gate now
    _last_epoch: int = -1

    def metrics(self) -> dict[str, float | int]:
        if not (self.suspensions or self.retried_turns):
            return {}
        return {"suspensions": self.suspensions, "suspended_seconds": round(self.suspended_seconds, 3),
                "suspend_retried_turns": self.retried_turns}

    def mark(self, epoch: int) -> None:
        if epoch != self._last_epoch:
            self._last_epoch = epoch
            self.suspensions += 1


_GATES: "weakref.WeakSet[TurnGate]" = weakref.WeakSet()
_GATES_LOCK = threading.Lock()


class TurnGate:
    """Closed from a round's cut-off until the next round starts (module doc)."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._open = asyncio.Event()
        self._open.set()
        self.epoch = 0  # incremented by every close
        self._lock = threading.Lock()
        self._ledgers: dict[str, SuspendLedger] = {}
        with _GATES_LOCK:
            _GATES.add(self)

    @property
    def closed(self) -> bool:
        return not self._open.is_set()

    def close(self) -> None:
        if not self.closed:
            self.epoch += 1
            self._open.clear()

    def open(self) -> None:
        self._open.set()

    def register(self, trajectory_id: str, *, group_id: str = "", task_id: str = "") -> SuspendLedger:
        with self._lock:
            if trajectory_id in self._ledgers:
                raise ValueError(f"trajectory {trajectory_id!r} is already registered at the turn gate")
            ledger = self._ledgers[trajectory_id] = SuspendLedger(trajectory_id, group_id, task_id)
        return ledger

    def unregister(self, ledger: SuspendLedger) -> None:
        with self._lock:
            self._ledgers.pop(ledger.trajectory_id, None)

    def live(self) -> list[SuspendLedger]:
        with self._lock:
            return list(self._ledgers.values())

    async def wait_open(self, ledger: SuspendLedger | None = None) -> None:
        """Return when the gate is open; the wait is counted on ``ledger``."""
        if not self.closed:
            return
        if ledger is not None:
            ledger.mark(self.epoch)
        started = self._clock()
        if ledger is not None:
            ledger.waiting += 1
        try:
            await self._open.wait()
        finally:
            if ledger is not None:
                ledger.waiting -= 1
                ledger.suspended_seconds += self._clock() - started


def suspended_references() -> list[dict[str, str]]:
    """Live trajectories of every closed gate of this process, as cut references.

    A trajectory registered at a closed gate is either waiting at it or will
    wait at its next model call, so the cut must account for it.
    """
    with _GATES_LOCK:
        gates = [g for g in _GATES if g.closed]
    out = []
    for gate in gates:
        for ledger in gate.live():
            out.append({"trajectory_id": ledger.trajectory_id,
                        "group_id": ledger.group_id or f"agentic:{ledger.trajectory_id}",
                        "task_id": ledger.task_id, "session_ref": SESSION_REF_PREFIX + ledger.trajectory_id})
    return sorted(out, key=lambda e: e["trajectory_id"])


def after_suspension(trajectory: AgentTrajectory, ledger: SuspendLedger) -> AgentTrajectory:
    """Attach the suspension metrics; a time limit after a suspension is ABORTED (module doc)."""
    metrics = ledger.metrics()
    if not metrics:
        return trajectory
    verdict = trajectory.verdict
    if verdict is not None:
        verdict = replace(verdict, agent_metrics={**dict(verdict.agent_metrics), **metrics})
    updated = replace(trajectory, verdict=verdict)
    if (ledger.suspensions and trajectory.outcome is Outcome.POLICY_LIMIT
            and trajectory.reason in TIME_LIMIT_EXCEPTIONS):
        updated = replace(updated, outcome=Outcome.ABORTED, reward=0.0,
                          reason=f"{TIMEOUT_AFTER_SUSPENSION_REASON}: {trajectory.reason}")
    return updated
