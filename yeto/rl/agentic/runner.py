"""The agentic outer layer: admit, run the harness, classify.

Both backends call :func:`run_agent_task`; only the harness (Harbor) and the
backend sample conversion differ.  Order per trajectory:

1. admission (bounded sandboxes, drain, policy version);
2. ``harness.run`` under the wall-clock cap;
3. verdict -> OK / POLICY_LIMIT / ABORTED (``outcome.classify``);
4. metrics.

Errors that would fail every sample the same way (bad configuration, a
trust violation) are raised, so a run stops loudly instead of training on
all-ABORTED or all-zero batches.  Every other harness failure is ABORTED.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from .admission import AdmissionClosed, AdmissionGate
from .outcome import WALL_CLOCK_REASON, classify
from .suspend import SuspendLedger
from .trust import TrustViolation
from .types import AgentTask, AgentTrajectory, Outcome, TrialVerdict

logger = logging.getLogger(__name__)


class HarnessConfigError(ValueError):
    """The harness cannot run this task as configured (missing task dir, bad settings)."""


class SessionAlreadyOpen(HarnessConfigError):
    """A recorder session id that is still open was opened again: that one trial
    cannot be recorded, the island's configuration is fine (the caller loses the
    trial, it does not stop)."""


class AgentHarness(Protocol):
    """Runs one trial of ``task``; the agent calls the model at ``endpoint`` (OpenAI base URL)."""

    async def run(self, task: AgentTask, endpoint: str) -> TrialVerdict: ...


async def run_agent_task(task: AgentTask, harness: AgentHarness, endpoint: str,
                         gate: AdmissionGate, *, ledger: SuspendLedger | None = None) -> AgentTrajectory:
    """``ledger``: the trajectory's suspension bookkeeping (:mod:`.suspend`); the
    wall-clock cap is extended by the time it was suspended."""
    try:
        async with gate.admit() as admission:
            verdict, cap_reason = await _run_capped(task, harness, endpoint, ledger)
    except AdmissionClosed:
        trajectory = _aborted(task, "admission_draining", policy_version=None)
        gate.metrics.record_outcome(trajectory.outcome)
        return trajectory

    if verdict is None:
        trajectory = _aborted(task, cap_reason, policy_version=admission.policy_version)
    else:
        outcome, reward, reason = classify(verdict)
        trajectory = AgentTrajectory(task=task, outcome=outcome, reward=reward, reason=reason,
                                     verdict=verdict, policy_version=admission.policy_version)
    gate.metrics.record_outcome(trajectory.outcome)
    logger.info("agent task %s: outcome=%s reward=%s reason=%s", task.task_id,
                trajectory.outcome.value, trajectory.reward, trajectory.reason)
    return trajectory


async def _run_capped(task: AgentTask, harness: AgentHarness, endpoint: str,
                      ledger: SuspendLedger | None = None) -> tuple[TrialVerdict | None, str]:
    """``(verdict, "")`` or ``(None, reason)`` for an infrastructure failure."""
    try:
        return await _capped(harness.run(task, endpoint), task.budget.wall_clock_s, ledger), ""
    except asyncio.TimeoutError:
        # The harness's own agent timeout sits below this cap; reaching it means
        # the harness did not return (stuck teardown, lost sandbox), not a slow policy.
        logger.error("agent task %s exceeded the %ss wall-clock cap", task.task_id, task.budget.wall_clock_s)
        return None, WALL_CLOCK_REASON
    except (HarnessConfigError, TrustViolation):
        raise
    except Exception as exc:  # noqa: BLE001 - every other harness failure is infrastructure
        logger.error("agent task %s: harness failed: %r", task.task_id, exc, exc_info=True)
        return None, type(exc).__name__


_SUSPENDED_RECHECK_S = 1.0


async def _capped(coro, cap_s: float, ledger: SuspendLedger | None):
    """``coro`` under a ``cap_s`` budget that does not count the ledger's suspended time.

    ``coro`` runs in its own task. When this call is cancelled again while it waits
    for a cancelled run that ignores the cancel (a sandbox tool call, then the island
    closing), the run's later error is still retrieved and logged, never left to
    asyncio as "Task exception was never retrieved" (``asyncio.wait_for`` of
    Python 3.11 drops it so)."""
    run = asyncio.ensure_future(coro)
    waiting = [True]

    def take_error(run: asyncio.Future) -> None:
        # retrieved here; a caller still waiting re-reads it with result()
        error = None if run.cancelled() else run.exception()
        if error is not None and not waiting[0]:
            logger.warning("agent run ended with %r after its trial was cancelled", error)

    run.add_done_callback(take_error)
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        while True:
            suspended = ledger.suspended_seconds if ledger is not None else 0.0
            remaining = started + cap_s + suspended - loop.time()
            if remaining <= 0:
                if ledger is None or not ledger.waiting:
                    raise asyncio.TimeoutError
                remaining = _SUSPENDED_RECHECK_S  # the wait in progress is not counted yet
            done, _ = await asyncio.wait({run}, timeout=remaining)
            if done:
                return run.result()
    finally:
        if not run.done():
            run.cancel()
            try:
                await asyncio.gather(run, return_exceptions=True)
            finally:
                waiting[0] = run.done()


def _aborted(task: AgentTask, reason: str, *, policy_version: int | None) -> AgentTrajectory:
    return AgentTrajectory(task=task, outcome=Outcome.ABORTED, reward=0.0, reason=reason,
                           verdict=None, policy_version=policy_version)
