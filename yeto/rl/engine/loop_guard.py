"""Cut repeating completions instead of penalising repetition (A-INF 5.5, D11).

A repetition penalty changes the behavior distribution, so rollouts do not use
one (``SamplingParams.check``). A completion that falls into a loop is cut at
the point the loop is detected instead: the kept tokens and their log-probs are
exactly what the engine sampled, so the behavior log-probs stay consistent; the
completion ends with ``finish_reason: loop_abort``.

Detection: some period ``p`` in ``1..max_period`` whose last ``p * repeats``
tokens are one ``p``-gram repeated ``repeats`` times. ``run[p]`` counts the
consecutive positions with ``token[i] == token[i - p]``; the loop is detected
at the first ``i`` with ``run[p] >= p * (repeats - 1)``. O(len * max_period).

Off by default: the thresholds must first be measured on round-0 samples
(A-INF 5.5). Cutting after the engine finished does not save generation time;
a streaming cut inside the adapter would, and is not built.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .sampler_engine import FINISH_LOOP_ABORT, Completion


@dataclass(frozen=True)
class LoopGuard:
    max_period: int
    repeats: int

    def __post_init__(self) -> None:
        if self.max_period < 1 or self.repeats < 2:
            raise ValueError("LoopGuard needs max_period >= 1 and repeats >= 2")

    def cut_at(self, token_ids: Sequence[int]) -> int | None:
        """Number of tokens to keep (through the detection point), or None."""
        run = [0] * (self.max_period + 1)
        for i in range(1, len(token_ids)):
            for p in range(1, min(self.max_period, i) + 1):
                run[p] = run[p] + 1 if token_ids[i] == token_ids[i - p] else 0
                if run[p] >= p * (self.repeats - 1):
                    return i + 1
        return None

    def apply(self, completion: Completion) -> Completion:
        keep = self.cut_at(completion.token_ids)
        if keep is None or keep >= len(completion.token_ids):
            return completion
        # The text of the kept tokens is not known without the tokenizer; the
        # sample's text is the engine text marked as cut (training reads tokens).
        return Completion(token_ids=completion.token_ids[:keep], logprobs=completion.logprobs[:keep],
                          text=completion.text, finish_reason=FINISH_LOOP_ABORT,
                          prompt_token_ids=completion.prompt_token_ids)
