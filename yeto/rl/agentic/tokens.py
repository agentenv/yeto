"""Join recorded model calls into one prompt/response/mask stream (S25 N10.3).

A Harbor agent talks OpenAI text protocol; the token recorder in front of the
sampler sees each call as (rendered prompt ids, completion ids, logprobs).  A
token-in/token-out trainer (verl ``AgentLoopOutput``) needs one prompt and one
response with a loss mask.  The join is only valid when every call's prompt
starts with exactly the tokens of everything before it: then the extra prompt
tokens are what the environment appended (mask 0), and the completion tokens
are what the policy generated (mask 1).  Any other shape means the agent
rewrote history (context compaction, a re-rendered template), so the trained
tokens would not be the generated ones; that raises instead of guessing.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import TokenAlignmentError, TokenSegments, TokenTurn


def join_turns(turns: Sequence[TokenTurn], *, max_response_len: int | None = None) -> TokenSegments:
    """Join ``turns`` (in call order) into one :class:`TokenSegments`.

    ``max_response_len`` cuts the response tail (the trainer's fixed response
    width); a cut sets ``truncated``.  Raises :class:`TokenAlignmentError`
    when there are no turns or a turn's prompt does not extend the stream.
    """
    if not turns:
        raise TokenAlignmentError("no model calls were recorded")
    if max_response_len is not None and max_response_len <= 0:
        raise ValueError("max_response_len must be positive")

    prompt = turns[0].prompt_ids
    stream: list[int] = list(prompt)
    ids: list[int] = []
    mask: list[int] = []
    logprobs: list[float] = []
    for number, turn in enumerate(turns):
        if tuple(turn.prompt_ids[: len(stream)]) != tuple(stream):
            raise TokenAlignmentError(
                f"turn {number} prompt does not extend the recorded stream "
                f"(history rewritten at token {_first_difference(stream, turn.prompt_ids)})")
        appended = turn.prompt_ids[len(stream):]
        if number > 0 and not appended and not turn.completion_ids:
            raise TokenAlignmentError(f"turn {number} adds no tokens")
        ids.extend(appended)
        mask.extend([0] * len(appended))
        logprobs.extend([0.0] * len(appended))
        ids.extend(turn.completion_ids)
        mask.extend([1] * len(turn.completion_ids))
        logprobs.extend(turn.completion_logprobs)
        stream.extend(appended)
        stream.extend(turn.completion_ids)

    truncated = max_response_len is not None and len(ids) > max_response_len
    if truncated:
        del ids[max_response_len:], mask[max_response_len:], logprobs[max_response_len:]
    return TokenSegments(
        prompt_ids=tuple(prompt),
        response_ids=tuple(ids),
        response_mask=tuple(mask),
        response_logprobs=tuple(logprobs),
        truncated=truncated,
    )


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))
