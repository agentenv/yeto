"""Dense word-diversity reward for rehearsals on models that cannot solve the task.

A test reward, not a product reward. The cut 4-layer GLM5.3-Flash rehearsal model
(S25 T-G1) never answers GSM8K and every response runs to ``max_tokens``: the
GSM8K reward and the length reward are constant within a group, so GRPO
advantages, the loss gradient and grad_norm are all 0 and the weights never move.
This reward depends on the text itself, so samples of one prompt differ even when
all of them are truncated:

    score = distinct whitespace words / words      (0.0 for an empty response)

Range [0, 1]; a response that repeats one word scores low. Launch with

    --reward-function yeto.rl.word_diversity_reward:score
"""

from __future__ import annotations

__all__ = ["diversity", "score"]


def diversity(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


async def score(args, sample, **kwargs) -> float:
    return diversity(sample.response or "")
