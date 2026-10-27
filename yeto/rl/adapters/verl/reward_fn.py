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
    # S27 G-DC: DAPO-Math rows (one-number labels), the reward of the Miles Nemotron runs
    "numeric_answer_reward:score": "compute_score_numeric_answer",
    "yeto.rl.numeric_answer_reward:score": "compute_score_numeric_answer",
    "yeto.rl.numeric_answer_reward.score": "compute_score_numeric_answer",
}

def validate_rlf_row(row):
    """Reject unverifiable RLF samples before verl accepts their float reward."""
    metadata = row.get("metadata", {}) if isinstance(row, dict) else getattr(row, "metadata", {})
    required = ("rlf_episode_id", "rlf_task_id", "grader_evidence", "success")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"RLF sample missing trusted metadata: {', '.join(missing)}")
    return metadata

def compute_score_rlf(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    """verl entry point for an already trusted RLF outcome.

    The numeric score is accepted only alongside the complete neutral metadata;
    this prevents a bare float reward path from bypassing admission.
    """
    row = extra_info if isinstance(extra_info, dict) else kwargs.get("metadata", {})
    metadata = validate_rlf_row(row)
    value = metadata.get("reward", kwargs.get("reward", 0.0))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("RLF reward must be numeric")
    return float(value)

REWARDS["yeto.rlf"] = "compute_score_rlf"

# S28 G-M2: engine health checks with a checkpoint that cannot solve any task
# (yeto.rl.bringup_reward). Never a training reward.
BRINGUP_REWARD = "yeto.rl.bringup_reward:score"


def compute_score_bringup(data_source, solution_str, ground_truth=None, extra_info=None, **_kwargs):
    from yeto.rl.bringup_reward import score

    return score(solution_str)


REWARDS[BRINGUP_REWARD] = "compute_score_bringup"

# S27 G-FS (VERL-PARITY #33): the agentic path's reward is the Harbor verdict the
# agent loop already put on the sample (``AgentLoopOutput.reward_score`` ->
# ``rm_scores``); verl's reward manager then returns that tensor and never calls a
# custom function (fork acad9875, workers/reward_manager/batch.py:82-84).  The
# launcher still requires a --reward-function, so an agentic run names this one:
# it asserts the short circuit instead of quietly scoring a trial with a math
# grader.  Never silently returns a number.
# ``yeto.rl.agentic_reward`` is a real module so the launcher can validate and
# hash the spec like any other reward function.
AGENTIC_REWARD = "yeto.rl.agentic_reward:score"


def compute_score_agentic(data_source, solution_str, ground_truth=None, extra_info=None, **_kwargs):
    from yeto.rl.agentic_reward import MESSAGE

    raise AssertionError(MESSAGE)


REWARDS[AGENTIC_REWARD] = "compute_score_agentic"
REWARDS["yeto.rl.agentic_reward.score"] = "compute_score_agentic"


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


def compute_score_numeric_answer(data_source, solution_str, ground_truth, extra_info=None, **_kwargs):
    """``yeto.rl.numeric_answer_reward`` (the same grader as the Miles islands)."""
    from yeto.rl.rewards.builtin import numeric_answer_reward
    from yeto.rl.rewards.types import Trajectory

    return float(numeric_answer_reward(Trajectory(response=solution_str or "", label=ground_truth)).value)
