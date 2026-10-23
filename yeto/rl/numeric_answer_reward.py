"""Numeric final-answer reward for one-number labels (DAPO-Math-17k rows), in-package.

The prompt asks for "Answer: \\boxed{$Answer}" on the last line and the label is
one number (an optional GSM8K-style "... #### N" prefix is ignored). The grader,
``yeto.rl.rewards.builtin.numeric_answer_grade``, reads the last balanced
``\\boxed{...}`` after any ``</think>`` (else the last "Answer: ..." line),
cleans LaTeX number formatting, and needs one number equal to the label.
Reward 1.0 / 0.0; ``sample.metadata["success"]`` records the same flag as
``yeto.rl.gsm8k_reward``.

Launch as ``--reward-function yeto.rl.numeric_answer_reward:score``, or copy
this file next to the launcher as ``numeric_answer_reward.py`` and use
``--reward-function numeric_answer_reward:score`` (the s27-gr run.sh pattern).
``yeto.rl.gsm8k_reward`` is unchanged.
"""

from __future__ import annotations

from yeto.rl.math_reward import set_success
from yeto.rl.rewards.builtin import numeric_answer_grade as grade, numeric_answer_reward
from yeto.rl.rewards.types import Trajectory

__all__ = ["grade", "score"]


async def score(args, sample, **kwargs):
    result = numeric_answer_reward(Trajectory(response=sample.response or "", label=sample.label))
    set_success(sample, result.metadata["success"])
    return result.value
