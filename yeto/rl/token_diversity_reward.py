"""Dense token-diversity reward for rehearsals on models that cannot solve the task.

A test reward, not a product reward (S25 T-G1C). The word-diversity reward
(:mod:`yeto.rl.word_diversity_reward`) was gamed in T-G1B: after two steps the
cut model wrote one 512-token "word" without spaces, every sample scored 1.0,
the group variance was 0 and round 2 had no gradient. This reward counts token
ids, so one repeated token scores low however it is spelled:

    score = distinct response token ids / response tokens   (0.0 when empty)

The response ids are ``sample.tokens[-response_length:]`` (Miles ``Sample``
and the engine sampler's reward sample both carry ``tokens`` and
``response_length``). Range (0, 1]; the maximum needs every response token to
differ, which a sampled 512-token response at temperature 1 does not reach.
Launch with

    --reward-function yeto.rl.token_diversity_reward:score
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = ["diversity", "response_ids", "score"]


def diversity(ids: Sequence[int]) -> float:
    return len(set(ids)) / len(ids) if ids else 0.0


def response_ids(sample: Any) -> list[int]:
    tokens = list(getattr(sample, "tokens", None) or ())
    length = int(getattr(sample, "response_length", 0) or 0)
    if length > len(tokens):
        raise ValueError(f"response_length {length} > {len(tokens)} tokens")
    return tokens[len(tokens) - length:] if length else []


async def score(args, sample, **kwargs) -> float:
    return diversity(response_ids(sample))
