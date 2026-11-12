"""Join recorded model calls into one prompt/response/mask stream.

A Harbor agent talks OpenAI text protocol; the token recorder in front of the
sampler sees each call as (rendered prompt ids, completion ids, logprobs).  A
token-in/token-out trainer (verl ``AgentLoopOutput``) needs one prompt and one
response with a loss mask.  The join is only valid when every call's prompt
starts with exactly the tokens of everything before it: then the extra prompt
tokens are what the environment appended (mask 0), and the completion tokens
are what the policy generated (mask 1).  Any other shape means the agent
rewrote history (context compaction, a re-rendered template), so the trained
tokens would not be the generated ones; that raises instead of guessing.

``rebase=True`` keeps such a trajectory instead: a call whose prompt does not
extend the stream but starts with the first call's prompt (a new recorder
session that re-rendered the history, e.g. after a gateway chain fork) becomes
the stream from that point on.  Its prompt ids are what the engine saw, so its
tail is mask-0 context and only completions sampled on that stream are trained
(mask 1): every trained token keeps the context and log-prob it was sampled
with, and R3 (the last call's rows) covers exactly that stream.  The earlier
completions lose their gradient, not the trial; ``rebased_turns`` counts it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import TokenAlignmentError, TokenSegments, TokenTurn


def join_turns(turns: Sequence[TokenTurn], *, max_response_len: int | None = None,
               rebase: bool = False) -> TokenSegments:
    """Join ``turns`` (in call order) into one :class:`TokenSegments`.

    ``rebase``: restart the stream at a call that re-rendered the history
    (module doc) instead of raising.

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
    with_top = turns[0].completion_top_ids is not None
    if any((t.completion_top_ids is not None) != with_top for t in turns):
        raise TokenAlignmentError("some turns carry top-K log-probs and others do not")
    # R3: only the last turn's rows are used (a recorder may drop the earlier ones)
    with_r3 = turns[-1].routed_experts is not None
    top_ids: list = []
    top_vals: list = []
    rebased = 0
    for number, turn in enumerate(turns):
        if tuple(turn.prompt_ids[: len(stream)]) != tuple(stream):
            at = _first_difference(stream, turn.prompt_ids)
            if not (rebase and number > 0 and tuple(turn.prompt_ids[:len(prompt)]) == tuple(prompt)):
                raise TokenAlignmentError(
                    f"turn {number} prompt does not extend the recorded stream "
                    f"(history rewritten at token {at})",
                    turn=number, position=at, stream=stream, prompt=turn.prompt_ids)
            # the engine's prompt is the stream from here: earlier completions become context
            rebased += 1
            stream = list(prompt)
            ids.clear(), mask.clear(), logprobs.clear(), top_ids.clear(), top_vals.clear()
        appended = turn.prompt_ids[len(stream):]
        if number > 0 and not appended and not turn.completion_ids:
            raise TokenAlignmentError(f"turn {number} adds no tokens")
        ids.extend(appended)
        mask.extend([0] * len(appended))
        logprobs.extend([0.0] * len(appended))
        ids.extend(turn.completion_ids)
        mask.extend([1] * len(turn.completion_ids))
        logprobs.extend(turn.completion_logprobs)
        if with_top:
            import numpy

            ids_k = numpy.asarray(turn.completion_top_ids, dtype=numpy.int32)
            k = ids_k.shape[1]
            if top_ids and top_ids[0].shape[1] != k:
                raise TokenAlignmentError(f"turn {number} has top-K width {k}, earlier turns {top_ids[0].shape[1]}")
            top_ids.append(numpy.full((len(appended), k), -1, dtype=numpy.int32))
            top_vals.append(numpy.full((len(appended), k), -numpy.inf, dtype=numpy.float32))
            top_ids.append(ids_k)
            top_vals.append(numpy.asarray(turn.completion_top_logprobs, dtype=numpy.float32))
        stream.extend(appended)
        stream.extend(turn.completion_ids)

    truncated = max_response_len is not None and len(ids) > max_response_len
    if truncated:
        del ids[max_response_len:], mask[max_response_len:], logprobs[max_response_len:]
    top = (None, None)
    if with_top:
        import numpy

        top = (numpy.concatenate(top_ids)[:len(ids)], numpy.concatenate(top_vals)[:len(ids)])
    routed = None
    if with_r3:  # the last turn's rows cover the whole stream (len(stream) - 1)
        rows = turns[-1].routed_experts
        end = len(prompt) + len(ids) - 1
        if len(rows) < end:  # a recorder's window shorter than the sample (fail closed)
            raise TokenAlignmentError(f"the last turn keeps {len(rows)} routed-expert rows; the sample "
                                      f"needs {end}")
        # a copy when shorter: a slice is a view that would keep every row alive
        routed = rows[:end].copy() if len(rows) > end and hasattr(rows, "copy") else rows[:end]
    return TokenSegments(
        prompt_ids=tuple(prompt),
        response_ids=tuple(ids),
        response_mask=tuple(mask),
        response_logprobs=tuple(logprobs),
        truncated=truncated,
        response_top_ids=top[0],
        response_top_logprobs=top[1],
        routed_experts=routed,
        rebased_turns=rebased,
    )


def _first_difference(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))
