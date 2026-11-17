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

CompactionRL (``segmented=True``, the engine sampler's agentic round, with a
recorder that has ``close_segments``): a trial the records proxy compacted is
one stream per compaction segment.  Steps 3-4 run per segment (each segment's
prompt is its own ``system + u_resume(S) + last k steps``) and
:attr:`FinishedSample.parts` holds the joined segments; the outcome and the
reward (step 2) stay the trial's, once.  Only the last segment may be cut at
``max_response_len``: an earlier cut segment would lose the summary it ended
with, so the trial is ABORTED (``compaction_segment_truncated``, logged; the
launch preflight checks the widths).  ``max_prompt_len`` bounds the first
segment only (a later one starts from the summary and the last steps, within
the compaction budget).  A segment without a trained token is left out.  The
calls of a joined segment drop their rollout arrays (the joined segment keeps
the rows it trains on), so a compacted trial holds one R3 array per segment,
not two.  A trial with one segment is the stream of the unsegmented path,
unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .admission import HarnessMetrics
from .outcome import policy_output_failure
from .recorder import TokenRecorder
from .tokens import join_turns
from .types import AgentTrajectory, Outcome, TokenAlignmentError, TokenSegments, TokenTurn


@dataclass(frozen=True)
class CompactionSegment:
    """One compaction segment of a trial: its calls (a recorder's
    ``close_segments``) and, after :func:`finish_sample`, its joined stream.
    ``kept_steps``: steps kept from the segment before (None on the first);
    ``summary_ok``: the summary it ended with had a closed tag (None: no summary);
    ``context_budget``: the proxy's compaction ``C``; ``summary_tokens``: tokens
    its summary turn generated (0 without one); ``retry_replays``: the trial's
    resent requests the gateway answered from a lost reply (None: replay off)."""

    turns: tuple[TokenTurn, ...]
    kept_steps: int | None = None
    summary_ok: bool | None = None
    context_budget: int | None = None
    tokens: TokenSegments | None = None
    summary_tokens: int = 0
    retry_replays: int | None = None


@dataclass(frozen=True)
class FinishedSample:
    trajectory: AgentTrajectory
    segments: TokenSegments | None  # None: ABORTED, nothing to train; with parts: the last part's
    turns: Sequence[TokenTurn]
    loss: dict[str, Any] | None = None  # token_alignment abort report (module doc)
    # a compacted trial (two or more segments, each joined); empty: one stream in ``segments``
    parts: tuple[CompactionSegment, ...] = ()
    compacting: bool = False  # recorded through a compacting records proxy (``segmented``)


LOSS_CONTEXT_TOKENS = 16
SEGMENT_TRUNCATED = "compaction_segment_truncated"  # a segment before the last cut at max_response_len
logger = logging.getLogger(__name__)


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
                        max_prompt_len: int | None = None, segmented: bool = False) -> FinishedSample:
    """Steps 1-4 of the module doc for one closed trial. ``segmented``: read the
    calls per compaction segment when the recorder can (module doc)."""
    turns: Sequence[TokenTurn] = ()
    loss: dict[str, Any] | None = None
    recorded: Sequence[CompactionSegment] = ()
    close_segments = getattr(recorder, "close_segments", None) if segmented else None
    try:
        if close_segments is None:
            turns = await recorder.close(session)
        else:
            recorded = await close_segments(session)
            turns = [t for part in recorded for t in part.turns]
    except TokenAlignmentError as exc:
        if trajectory.outcome is not Outcome.ABORTED:
            loss = alignment_report(exc, trajectory, turns, recorder, "recorder_close")
            trajectory = abort(trajectory, f"token_alignment: {exc}", metrics)
    else:
        updated = policy_output_failure(trajectory, [t.policy_error for t in turns if t.policy_error])
        if updated.outcome is not trajectory.outcome:
            metrics.reclassify(trajectory.outcome, updated.outcome)
        trajectory = updated

    if len(recorded) > 1:
        return _finish_parts(trajectory, recorder, recorded, turns, metrics, max_response_len=max_response_len,
                             max_prompt_len=max_prompt_len)
    segments: TokenSegments | None = None
    if trajectory.outcome is not Outcome.ABORTED:
        try:
            segments = join_turns(turns, max_response_len=max_response_len, rebase=True)
        except TokenAlignmentError as exc:
            loss = alignment_report(exc, trajectory, turns, recorder, "join_turns")
            trajectory = abort(trajectory, f"token_alignment: {exc}", metrics)
    if segments is not None and max_prompt_len is not None and len(segments.prompt_ids) > max_prompt_len:
        trajectory, segments = abort(trajectory, "prompt_exceeds_prompt_length", metrics), None
    compacting = any(part.context_budget is not None for part in recorded)
    return FinishedSample(trajectory, segments, turns, loss, compacting=compacting)


def _light(turns: Sequence[TokenTurn]) -> tuple[TokenTurn, ...]:
    """``turns`` without their rollout arrays: the joined segment holds the rows
    it trains on, so the calls' own copies (R3 up to a whole segment each) go."""
    return tuple(t if t.routed_experts is None and t.completion_top_ids is None else
                 replace(t, routed_experts=None, routed_experts_cut=False, completion_top_ids=None,
                         completion_top_logprobs=None) for t in turns)


def _finish_parts(trajectory: AgentTrajectory, recorder: Any, recorded: Sequence[CompactionSegment],
                  turns: Sequence[TokenTurn], metrics: HarnessMetrics, *, max_response_len: int,
                  max_prompt_len: int | None) -> FinishedSample:
    """Steps 3-4 per compaction segment (module doc); the outcome is the trial's.

    ``max_prompt_len`` bounds the first segment only: a later one starts from
    ``system + u_resume(S) + last k steps``, bounded by the compaction budget.
    A segment without a trained token is left out (``num_segments`` counts
    the segments written)."""
    if trajectory.outcome is Outcome.ABORTED:
        return FinishedSample(trajectory, None, turns)
    parts: list[CompactionSegment] = []
    light = [_light(part.turns) for part in recorded]
    turns = [t for calls in light for t in calls]
    for number, part in enumerate(recorded):
        try:
            joined = join_turns(part.turns, max_response_len=max_response_len, rebase=True)
        except TokenAlignmentError as exc:
            loss = {**alignment_report(exc, trajectory, part.turns, recorder, "join_turns"), "segment": number}
            return FinishedSample(abort(trajectory, f"token_alignment: {exc}", metrics), None, turns, loss)
        if number == 0 and max_prompt_len is not None and len(joined.prompt_ids) > max_prompt_len:
            return FinishedSample(abort(trajectory, "prompt_exceeds_prompt_length", metrics), None, turns)
        if joined.truncated and number < len(recorded) - 1:
            # the cut tail is the segment's summary: never trained silently (preflight checks the widths);
            # counted as a lost trial under this reason in the round's ``agentic.lost``
            logger.warning("compaction segment %d of %s has more than max_response_len=%d response tokens; "
                           "the trial is dropped", number, getattr(trajectory.task, "trajectory_id", None),
                           max_response_len)
            return FinishedSample(abort(trajectory, f"{SEGMENT_TRUNCATED}: segment {number} has more "
                                        f"than {max_response_len} response tokens", metrics), None, turns)
        if any(joined.response_mask):
            parts.append(replace(part, turns=light[number], tokens=joined))
    if not parts:
        return FinishedSample(abort(trajectory, "compaction_no_trained_tokens", metrics), None, turns)
    return FinishedSample(trajectory, parts[-1].tokens, turns, None, tuple(parts), compacting=True)
