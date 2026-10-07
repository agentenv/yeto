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

REWARD_COMPONENTS_KEY = "yeto_reward_components"
COMPONENTS = ("correctness", "format")


def _correct(response: str, label) -> bool:
    from yeto.rl.math_reward import score

    # GSM8K-style labels carry the reasoning before "#### <answer>".
    truth = None if label is None else str(label).split("####")[-1].strip().replace(",", "")
    return score(response, truth) == 1.0


def components(response: str, label) -> dict[str, float]:
    answer = response.split("</think>")[-1]
    has_box = "\\boxed{" in answer
    correct = has_box and _correct(response, label)
    return {"correctness": 1.0 if correct else 0.0, "format": 1.0 if has_box else 0.0}


async def correctness_reward(args, sample, **kwargs) -> float:
    """Binary {0,1} correctness only (the ``correctness`` component), for MaxRL/MAPO/GSPO/rpp G1."""

    from yeto.rl.math_reward import set_success

    value = components(sample.response or "", sample.label)["correctness"]
    set_success(sample, value == 1.0)
    return value


async def reward_func(args, sample, **kwargs) -> float:
    values = components(sample.response or "", sample.label)
    if not isinstance(sample.metadata, dict):
        sample.metadata = {}
    sample.metadata[REWARD_COMPONENTS_KEY] = values
    sample.metadata["success"] = values["correctness"] == 1.0
    return values["correctness"]
