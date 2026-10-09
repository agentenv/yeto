"""verl ``compute_score`` wrappers over yeto's neutral reward functions (design D11).

verl loads ``reward.custom_reward_function.path`` / ``.name`` in its reward
workers; the function computes yeto's reward (``yeto.rl.rewards.builtin``),
so Miles and verl islands score with the same code.  Only mapped reward
names are accepted (``REWARDS``); anything else is refused before launch.
"""

from __future__ import annotations

# launcher --reward-function spelling -> function name in this module
REWARDS = {
    "gsm8k_reward:score": "compute_score_gsm8k",
    "yeto.rl.gsm8k_reward:score": "compute_score_gsm8k",
    "yeto.rl.gsm8k_reward.score": "compute_score_gsm8k",
}


class UnmappedReward(ValueError):
    pass


def function_for(reward_function: str) -> str:
    try:
        return REWARDS[reward_function]
    except KeyError:
        raise UnmappedReward(f"verl 后端未映射奖励函数 {reward_function!r}（已映射：{sorted(REWARDS)}）") from None


def compute_score_gsm8k(data_source, solution_str, ground_truth, extra_info=None, **_kwargs):
    from yeto.rl.rewards.builtin import gsm8k_reward
    from yeto.rl.rewards.types import Trajectory

    return float(gsm8k_reward(Trajectory(response=solution_str or "", label=ground_truth)).value)
