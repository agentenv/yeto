"""Two-component example reward for GDPO (rl-algo-seq-and-adv 5.4, G1).

``--custom-rm-path yeto.rl.algos.gdpo_reward.reward_func`` with
``advantage.gdpo = {"components": [{"name": "correctness", "weight": 1.0},
{"name": "format", "weight": 1.0}], "whiten": true}``.

* ``correctness``: 1.0 when the last ``\\boxed{...}`` answer after any
  ``</think>`` block matches the label (``yeto.rl.math_reward.score``), else 0.0;
* ``format``: 1.0 when the final answer (after ``</think>``) contains a
  ``\\boxed{`` answer at all, else 0.0.

The components go to ``sample.metadata["yeto_reward_components"]`` (design
D6); the scalar return value (correctness) is only used for metrics/logs.
Both entry points also set ``sample.metadata["success"]`` = correctness (the
format component never counts as success).
"""

from __future__ import annotations

# Neutral forms: yeto.rl.rewards.builtin.gdpo_reward / gdpo_correctness_reward
# (decoupling 3.2); this module keeps the Miles entry points and signatures.
from yeto.rl.rewards.builtin import GDPO_COMPONENTS as COMPONENTS
from yeto.rl.rewards.builtin import REWARD_COMPONENTS_KEY
from yeto.rl.rewards.builtin import gdpo_components, gdpo_correct as _correct
from yeto.rl.rewards.builtin import gdpo_correctness_result, gdpo_result

__all__ = ["COMPONENTS", "REWARD_COMPONENTS_KEY", "components", "correctness_reward",
           "reward_func"]


def components(response: str, label) -> dict[str, float]:
    # ``_correct`` is looked up at call time (tests replace it).
    return gdpo_components(response, label, correct=_correct)


def _apply(sample, result) -> float:
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata.update(result.metadata)
    return result.value


async def correctness_reward(args, sample, **kwargs) -> float:
    """Binary {0,1} correctness only (the ``correctness`` component), for MaxRL/MAPO/GSPO/rpp G1."""

    return _apply(sample, gdpo_correctness_result(components(sample.response or "", sample.label)))


async def reward_func(args, sample, **kwargs) -> float:
    return _apply(sample, gdpo_result(components(sample.response or "", sample.label)))
