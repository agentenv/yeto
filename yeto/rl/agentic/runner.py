"""The agentic outer layer: admit, run the harness, classify (S25 N10.2).

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
from .trust import TrustViolation
from .types import AgentTask, AgentTrajectory, Outcome, TrialVerdict

logger = logging.getLogger(__name__)


class HarnessConfigError(ValueError):
    """The harness cannot run this task as configured (missing task dir, bad settings)."""


class AgentHarness(Protocol):
    """Runs one trial of ``task``; the agent calls the model at ``endpoint`` (OpenAI base URL)."""

    async def run(self, task: AgentTask, endpoint: str) -> TrialVerdict: ...


async def run_agent_task(task: AgentTask, harness: AgentHarness, endpoint: str,
                         gate: AdmissionGate) -> AgentTrajectory:
    try:
        async with gate.admit() as admission:
            verdict, cap_reason = await _run_capped(task, harness, endpoint)
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


async def _run_capped(task: AgentTask, harness: AgentHarness,
                      endpoint: str) -> tuple[TrialVerdict | None, str]:
    """``(verdict, "")`` or ``(None, reason)`` for an infrastructure failure."""
    try:
        return await asyncio.wait_for(harness.run(task, endpoint), timeout=task.budget.wall_clock_s), ""
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


def _aborted(task: AgentTask, reason: str, *, policy_version: int | None) -> AgentTrajectory:
    return AgentTrajectory(task=task, outcome=Outcome.ABORTED, reward=0.0, reason=reason,
                           verdict=None, policy_version=policy_version)
