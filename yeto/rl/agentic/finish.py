"""After the trial: the recorded calls -> one trainable token stream (S28 G-120).

Both callers of :func:`~yeto.rl.agentic.run_agent_task` that record tokens
through a :class:`~yeto.rl.agentic.TokenRecorder` (verl's ``HarborAgentLoop``
and the engine sampler's ``AgenticRound``) end a sample the same way:

1. close the recorder session (a recorder that cannot account for its tokens
   makes the sample ABORTED);
2. a malformed model reply that ended the agent is a 0 reward, not ABORTED
   (``outcome.policy_output_failure``);
3. join the calls into prompt / response / mask / log-probs (``join_turns``;
   a call that does not extend the stream is ABORTED);
4. a prompt longer than the trainer's prompt width is ABORTED.

Every reclassification is also recorded in the gate's metrics, so the
outcome counters match the samples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .admission import HarnessMetrics
from .outcome import policy_output_failure
from .recorder import TokenRecorder
from .tokens import join_turns
from .types import AgentTrajectory, Outcome, TokenAlignmentError, TokenSegments, TokenTurn


@dataclass(frozen=True)
class FinishedSample:
    trajectory: AgentTrajectory
    segments: TokenSegments | None  # None: ABORTED, nothing to train
    turns: Sequence[TokenTurn]


def abort(trajectory: AgentTrajectory, reason: str, metrics: HarnessMetrics) -> AgentTrajectory:
    """``trajectory`` as ABORTED with ``reason`` (reward 0), counted in ``metrics``."""
    metrics.reclassify(trajectory.outcome, Outcome.ABORTED)
    return AgentTrajectory(task=trajectory.task, outcome=Outcome.ABORTED, reward=0.0, reason=reason,
                           verdict=trajectory.verdict, policy_version=trajectory.policy_version,
                           extra_metrics=trajectory.extra_metrics)


async def finish_sample(trajectory: AgentTrajectory, recorder: TokenRecorder, session: str,
                        metrics: HarnessMetrics, *, max_response_len: int,
                        max_prompt_len: int | None = None) -> FinishedSample:
    """Steps 1-4 of the module doc for one closed trial."""
    turns: Sequence[TokenTurn] = ()
    try:
        turns = await recorder.close(session)
    except TokenAlignmentError as exc:
        if trajectory.outcome is not Outcome.ABORTED:
            trajectory = abort(trajectory, f"token_alignment: {exc}", metrics)
    else:
        updated = policy_output_failure(trajectory, [t.policy_error for t in turns if t.policy_error])
        if updated.outcome is not trajectory.outcome:
            metrics.reclassify(trajectory.outcome, updated.outcome)
        trajectory = updated

    segments: TokenSegments | None = None
    if trajectory.outcome is not Outcome.ABORTED:
        try:
            segments = join_turns(turns, max_response_len=max_response_len)
        except TokenAlignmentError as exc:
            trajectory = abort(trajectory, f"token_alignment: {exc}", metrics)
    if segments is not None and max_prompt_len is not None and len(segments.prompt_ids) > max_prompt_len:
        trajectory, segments = abort(trajectory, "prompt_exceeds_prompt_length", metrics), None
    return FinishedSample(trajectory, segments, turns)
