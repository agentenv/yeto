"""Neutral forms of yeto's built-in rewards and the bounded non-zero-std filter (decoupling 3.2).

Each reward takes a :class:`Trajectory` and returns a :class:`RewardResult`;
the filter is an object that owns its state.  The Miles entry points keep their
original modules and signatures (``yeto.rl.gsm8k_reward.score``,
``yeto.rl.length_reward.reward_func``, ``yeto.rl.algos.gdpo_reward.*``,
``yeto.rl.filters.bounded_nonzero_reward_std``) and delegate here;
``yeto.rl.math_reward.reward_func`` is left byte-identical because its source
hash is part of the golden samples, and its neutral form below reuses its pure
``score`` (which still needs Miles' math graders at run time, D9c).
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Hashable, Sequence

from yeto.rl.rewards.types import FilterDecision, RewardResult, Trajectory

SUCCESS_KEY = "success"
REWARD_COMPONENTS_KEY = "yeto_reward_components"
LENGTH_CHAR_CAP = 1024


# -- math (MATH-500 style); grading needs Miles math_utils at run time -------

def math_reward(traj: Trajectory) -> RewardResult:
    from yeto.rl.math_reward import score

    value = score(traj.response or "", traj.label)
    return RewardResult(value=value, metadata={SUCCESS_KEY: value == 1.0})


# -- GSM8K exact number --------------------------------------------------------

def _num(s):
    s = str(s).replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def gsm8k_grade(response: str, label) -> float:
    gold = _num(str(label).split("####")[-1]) if label is not None else None
    text = response or ""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    pred = _num(boxed[-1]) if boxed else _num(text[-200:])
    return 1.0 if (gold is not None and pred is not None and abs(pred - gold) < 1e-6) else 0.0


def gsm8k_reward(traj: Trajectory) -> RewardResult:
    value = gsm8k_grade(traj.response or "", traj.label)
    return RewardResult(value=value, metadata={SUCCESS_KEY: value == 1.0})


# -- dense length / termination test reward -----------------------------------

def length_score(response: str, finished: float) -> float:
    n = min(len(response or ""), LENGTH_CHAR_CAP)
    return 0.5 * float(finished) + 0.5 * (1.0 - n / LENGTH_CHAR_CAP)


def length_reward(traj: Trajectory) -> RewardResult:
    finished = 1.0 if str(traj.status or "").upper().endswith("COMPLETED") else 0.0
    return RewardResult(value=length_score(traj.response or "", finished))


# -- GDPO two-component example -----------------------------------------------

GDPO_COMPONENTS = ("correctness", "format")


def gdpo_correct(response: str, label) -> bool:
    from yeto.rl.math_reward import score

    # GSM8K-style labels carry the reasoning before "#### <answer>".
    truth = None if label is None else str(label).split("####")[-1].strip().replace(",", "")
    return score(response, truth) == 1.0


def gdpo_components(response: str, label, *, correct=gdpo_correct) -> dict[str, float]:
    answer = response.split("</think>")[-1]
    has_box = "\\boxed{" in answer
    is_correct = has_box and correct(response, label)
    return {"correctness": 1.0 if is_correct else 0.0, "format": 1.0 if has_box else 0.0}


def gdpo_result(values: dict[str, float]) -> RewardResult:
    return RewardResult(value=values["correctness"],
                        metadata={REWARD_COMPONENTS_KEY: values,
                                  SUCCESS_KEY: values["correctness"] == 1.0})


def gdpo_correctness_result(values: dict[str, float]) -> RewardResult:
    value = values["correctness"]
    return RewardResult(value=value, metadata={SUCCESS_KEY: value == 1.0})


def gdpo_reward(traj: Trajectory) -> RewardResult:
    return gdpo_result(gdpo_components(traj.response or "", traj.label))


def gdpo_correctness_reward(traj: Trajectory) -> RewardResult:
    return gdpo_correctness_result(gdpo_components(traj.response or "", traj.label))


# -- bounded non-zero reward std group filter ----------------------------------

class BoundedNonzeroStdFilter:
    """Prefer non-zero reward variance, then accept a group after a bound.

    The first ``max_replacements`` zero-variance groups of a round are
    rejected and the next one accepted (``None`` = never accept).  Decisions
    are memoized per group key because a backend may show a group twice.  The
    state (``self.state``) is reset when the round id changes.
    """

    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    def reset(self, round_id: Any) -> None:
        self.state = {"rollout_id": round_id, "rejections": 0, "forced": 0, "decisions": {}}

    def cached(self, *, round_id: Any, key: Hashable) -> FilterDecision | None:
        """The memoized decision for ``key`` in round ``round_id`` (resets on a new round)."""
        if not self.state or self.state.get("rollout_id") != round_id:
            self.reset(round_id)
        previous = self.state["decisions"].get(key)
        return None if previous is None else FilterDecision(*previous)

    def __call__(self, group: Sequence[Trajectory], *, round_id: Any = None,
                 max_replacements: int | None = None,
                 key: Hashable | None = None) -> FilterDecision:
        if key is None:
            key = tuple(t.index if t.index is not None else id(t) for t in group)
        previous = self.cached(round_id=round_id, key=key)
        if previous is not None:
            return previous
        state = self.state

        rewards = [float(t.reward) for t in group]
        std = statistics.pstdev(rewards) if rewards else 0.0
        if not math.isfinite(std):
            std = 0.0
        if std > 1e-8:
            decision = (True, None)
        else:
            limit = max_replacements
            if limit is None or state["rejections"] < int(limit):
                state["rejections"] += 1
                value = round(rewards[0], 1) if rewards else 0.0
                decision = (False, f"zero_std_{value}")
            else:
                state["forced"] += 1
                decision = (True, f"bounded_fallback_after_{int(limit)}_replacements")
        state["decisions"][key] = decision
        return FilterDecision(*decision)
