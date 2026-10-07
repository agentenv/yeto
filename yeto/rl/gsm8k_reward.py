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

import re

from yeto.rl.math_reward import set_success


def _num(s):
    s = str(s).replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def grade(response: str, label) -> float:
    gold = _num(str(label).split("####")[-1]) if label is not None else None
    text = response or ""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    pred = _num(boxed[-1]) if boxed else _num(text[-200:])
    return 1.0 if (gold is not None and pred is not None and abs(pred - gold) < 1e-6) else 0.0


async def score(args, sample, **kwargs):
    value = grade(sample.response or "", sample.label)
    set_success(sample, value == 1.0)
    return value
