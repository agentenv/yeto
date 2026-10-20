"""Harness verdict -> OK / POLICY_LIMIT / ABORTED (S25 N10.2; was L7 of the Codex harness).

yeto rule, kept from the Codex harness: an infrastructure failure is never a
0 reward.  Only exceptions known to be the policy's own budget become a
trainable negative; every other exception, and a missing verifier result,
is ABORTED (fail closed).  Miles' Harbor example scores those 0 instead
(``AgentError``); the yeto outer layer replaces that mapping.

Exception names are Harbor's (``trial/errors.py``, ``llms/base.py``,
``agents/installed/base.py`` at the pinned commit); a renamed class in a newer
Harbor falls through to ABORTED, which is the safe direction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .trust import verified_reward
from .types import AgentTrajectory, Outcome, TrialVerdict

# The policy ran out of its own budget: time, or sequence/context length.
POLICY_LIMIT_EXCEPTIONS = frozenset({
    "AgentTimeoutError",
    "MaxSeqLenExceededError",
    "SingleTurnMaxSeqLenExceededError",
    "ContextLengthExceededError",
    "ContextWindowExceededError",
    "OutputLengthExceededError",
    "OutputTokenExceededError",
})

# Reason given when the outer wall-clock cap fires (the harness never returned).
WALL_CLOCK_REASON = "wall_clock_cap"

# The agent process exited non-zero.  On its own this is ABORTED (could be the
# sandbox); with a recorded malformed model reply it is the policy's failure.
AGENT_EXIT_EXCEPTIONS = frozenset({"NonZeroAgentExitCodeError"})
POLICY_OUTPUT_ERROR_REASON = "policy_output_error"


def policy_output_failure(trajectory: AgentTrajectory, policy_errors: Sequence[str]) -> AgentTrajectory:
    """ABORTED by the agent's exit after the recorder refused a malformed model reply -> POLICY_LIMIT, reward 0.

    S25 N10.9 (main agent decision): a tool call that is not valid JSON is the
    policy's own failure, trained as a 0 reward, not an infrastructure abort.
    Fail closed: only the agent-exit exception, and only with a recorded
    refusal, is re-classified; every other ABORTED stays ABORTED.
    """
    verdict = trajectory.verdict
    if (trajectory.outcome is not Outcome.ABORTED or not policy_errors or verdict is None
            or verdict.exception_type not in AGENT_EXIT_EXCEPTIONS):
        return trajectory
    return replace(trajectory, outcome=Outcome.POLICY_LIMIT, reward=0.0,
                   reason=f"{POLICY_OUTPUT_ERROR_REASON}: {policy_errors[0][:200]}")


def classify(verdict: TrialVerdict, *, reward_range: tuple[float, float] = (0.0, 1.0)) -> tuple[Outcome, float, str]:
    """Return ``(outcome, reward, reason)`` for one verdict.

    A verifier result that breaks the trust rules raises
    :class:`~yeto.rl.agentic.trust.TrustViolation` (the run stops; a tampered
    verdict is not an infrastructure hiccup).
    """
    low, high = reward_range
    rewards = verdict.rewards
    exc = verdict.exception_type
    if exc in POLICY_LIMIT_EXCEPTIONS:
        reward = verified_reward(rewards, low=low, high=high) if rewards else low
        return Outcome.POLICY_LIMIT, reward, exc
    if exc is not None:
        return Outcome.ABORTED, 0.0, exc
    if not rewards:
        return Outcome.ABORTED, 0.0, "no_verifier_result"
    return Outcome.OK, verified_reward(rewards, low=low, high=high), "verified"
