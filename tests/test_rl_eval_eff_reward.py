"""The bare-framework GSM8K reward (yeto/rl/eval/efficiency/eff_reward.py) grades
exactly like the yeto reward ``gsm8k_grade``."""

from __future__ import annotations

import pytest

from yeto.rl.eval.efficiency import eff_reward


@pytest.mark.parametrize("response,label", [
    ("so \\boxed{42}", "#### 42"), ("the answer is 1,234", "x #### 1234"), ("\\boxed{3.5} then 7", "#### 3.5"),
    ("no number", "#### 1"), ("\\boxed{-2}", "#### 2"), ("", None),
])
def test_bare_reward_equals_yeto_reward(response, label):
    from yeto.rl.rewards.builtin import gsm8k_grade

    assert eff_reward.grade(response, label) == gsm8k_grade(response, label)
    assert eff_reward.compute_score("d", response, label) == gsm8k_grade(response, label)
