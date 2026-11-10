"""After the trial: the recorded calls -> one trainable token stream.

Both callers of :func:`~yeto.rl.agentic.run_agent_task` that record tokens
through a :class:`~yeto.rl.agentic.TokenRecorder` (verl's ``HarborAgentLoop``
and the engine sampler's ``AgenticRound``) end a sample the same way:

1. close the recorder session (a recorder that cannot account for its tokens
   makes the sample ABORTED);
2. a malformed model reply that ended the agent is a 0 reward, not ABORTED
   (``outcome.policy_output_failure``);
3. join the calls into prompt / response / mask / log-probs (``join_turns``
   with ``rebase``: a call that re-rendered the history restarts the stream;
   one that does not even repeat the first prompt is ABORTED);
4. a prompt longer than the trainer's prompt width is ABORTED.

Every reclassification is also recorded in the gate's metrics, so the
outcome counters match the samples.  A ``token_alignment`` abort carries a
:attr:`FinishedSample.loss` report (the full error, the call, the first
differing token and the decoded tokens around it, the trial directory) for
the round's ``rl_agentic_trial_lost`` event.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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
    loss: dict[str, Any] | None = None  # token_alignment abort report (module doc)


LOSS_CONTEXT_TOKENS = 16


def _decoded(recorder: Any, ids: Sequence[int]) -> str | None:
    decode = getattr(recorder, "decode", None)
    if decode is None:
        return None
    try:
        return str(decode(list(ids)))
    except Exception as exc:  # noqa: BLE001 - a report must not fail the sample
        return f"<decode failed: {exc}>"


def alignment_report(exc: TokenAlignmentError, trajectory: AgentTrajectory, turns: Sequence[TokenTurn],
                     recorder: Any, stage: str) -> dict[str, Any]:
    """What a ``token_alignment`` loss looked like, JSON-safe."""
    report: dict[str, Any] = {
        "stage": stage, "error": f"{type(exc).__name__}: {exc}", "turn": getattr(exc, "turn", None),
        "position": getattr(exc, "position", None), "turns": len(turns),
        "trial_dir": trajectory.verdict.trial_dir if trajectory.verdict is not None else None,
        "trajectory_id": getattr(trajectory.task, "trajectory_id", None)}
    stream, prompt, at = getattr(exc, "stream", None), getattr(exc, "prompt", None), getattr(exc, "position", None)
    if stream is not None and prompt is not None and at is not None:
        lo, k = max(0, at - LOSS_CONTEXT_TOKENS), LOSS_CONTEXT_TOKENS
        report.update(stream_len=len(stream), prompt_len=len(prompt),
                      stream_ids=list(stream[lo:at + k]), prompt_ids=list(prompt[lo:at + k]), window_start=lo,
                      before_text=_decoded(recorder, stream[lo:at]),
                      stream_text=_decoded(recorder, stream[at:at + k]),
                      prompt_text=_decoded(recorder, prompt[at:at + k]))
    return report


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
    loss: dict[str, Any] | None = None
    try:
        turns = await recorder.close(session)
    except TokenAlignmentError as exc:
        if trajectory.outcome is not Outcome.ABORTED:
            loss = alignment_report(exc, trajectory, turns, recorder, "recorder_close")
            trajectory = abort(trajectory, f"token_alignment: {exc}", metrics)
    else:
        updated = policy_output_failure(trajectory, [t.policy_error for t in turns if t.policy_error])
        if updated.outcome is not trajectory.outcome:
            metrics.reclassify(trajectory.outcome, updated.outcome)
        trajectory = updated

    segments: TokenSegments | None = None
    if trajectory.outcome is not Outcome.ABORTED:
        try:
            segments = join_turns(turns, max_response_len=max_response_len, rebase=True)
        except TokenAlignmentError as exc:
            loss = alignment_report(exc, trajectory, turns, recorder, "join_turns")
            trajectory = abort(trajectory, f"token_alignment: {exc}", metrics)
    if segments is not None and max_prompt_len is not None and len(segments.prompt_ids) > max_prompt_len:
        trajectory, segments = abort(trajectory, "prompt_exceeds_prompt_length", metrics), None
    return FinishedSample(trajectory, segments, turns, loss)
