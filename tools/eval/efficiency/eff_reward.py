"""GSM8K exact-number reward for the bare-framework arms (no yeto import).

The grading rule is a verbatim copy of ``yeto.rl.rewards.builtin.gsm8k_grade``
(the reward of the yeto arms, ``--reward-function gsm8k_reward:score``); a CPU
test asserts both give the same value on the same cases. The file is copied
into the bare containers as ``/root/eff_reward.py``:

* verl: ``reward.custom_reward_function.path=/root/eff_reward.py`` and
  ``reward.custom_reward_function.name=compute_score``;
* Miles: ``--custom-rm-path eff_reward.score`` (``/root`` on ``PYTHONPATH``).
"""

from __future__ import annotations

import re


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


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """verl custom reward signature."""
    return grade(solution_str, ground_truth)


async def score(args, sample, **kwargs) -> float:
    """Miles custom reward signature (``--custom-rm-path``)."""
    return grade(sample.response or "", sample.label)
