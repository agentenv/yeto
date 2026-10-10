"""GSM8K exact-number reward, tracked in-package.

Numerically identical to the out-of-tree ``gsm8k_reward.py`` that GPU runs
load as ``--reward-function gsm8k_reward:score`` from the working directory
(copies under ``openspec/changes/*/evidence/**/gsm8k_reward.py``): the last
``\\boxed{...}`` number (else the last number in the final 200 characters)
must equal the number after ``####`` in the label. Reward 1.0 / 0.0.

It additionally records ``sample.metadata["success"]`` (bool: answer
correct), the full-success flag read by the critic fork's VAPO
positive-example LM loss. Launch as ``yeto.rl.gsm8k_reward:score``.
"""

from __future__ import annotations

# Neutral form: yeto.rl.rewards.builtin.gsm8k_reward (decoupling 3.2); this
# module keeps the Miles entry point and its signature.
from yeto.rl.math_reward import set_success, writes_success
from yeto.rl.rewards.builtin import _num, gsm8k_grade as grade, gsm8k_reward
from yeto.rl.rewards.types import Trajectory

__all__ = ["_num", "grade", "score"]


@writes_success
async def score(args, sample, **kwargs):
    result = gsm8k_reward(Trajectory(response=sample.response or "", label=sample.label))
    set_success(sample, result.metadata["success"])
    return result.value
