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


# -- numeric final answer (DAPO-Math-17k style "Answer: \boxed{N}") ------------
#
# ``gsm8k_grade`` above stays byte-identical (golden samples, bare-arm copy in
# tools/eval/efficiency/eff_reward.py).  This grader is for rows whose label is
# one number and whose prompt asks for "Answer: \boxed{...}" on the last line.
# Differences from gsm8k_grade:
#   * grades only the text after the last ``</think>``;
#   * reads the last ``\boxed{...}`` with balanced braces (``\boxed{\frac{1}{2}}``);
#   * if there is no ``\boxed``, reads the last "Answer: ..." line; there is no
#     "last number in the last 200 characters" fallback, so a response cut off
#     mid-derivation scores 0 instead of matching a stray number;
#   * the answer must be one number after LaTeX clean-up (thousands separators
#     ``1\,007`` / ``1{,}440`` / ``1,440``, ``\text{}``, ``$``, units, ``x = ``,
#     ``^\circ``, ``\%``), or a simple ``\frac{a}{b}`` / ``a^{b}``; it never takes
#     the last number of an expression (``\boxed{2\sqrt{3}}`` is not 3).

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
_ANSWER_LINE = re.compile(r"answer\s*[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_TEXT_CMD = re.compile(r"\\(?:text|textbf|textrm|textit|mathrm|mathbf|mathit|bf|mbox|operatorname|answer)"
                       r"\s*\{([^{}]*)\}")
_FRAC = re.compile(rf"^({_NUMBER})?\\[dt]?frac\{{({_NUMBER})\}}\{{({_NUMBER})\}}$")
_POWER = re.compile(rf"^({_NUMBER})\^\{{?({_NUMBER})\}}?$")
_UNIT_TAIL = re.compile(rf"^({_NUMBER})[a-zA-Z]*\.?$")
_THOUSANDS = re.compile(r"(?<=\d)(?:\{,\}|,|\\,|\\!|\\ |~|\s)(?=\d{3}(?!\d))")
# spacing / delimiter / markup tokens removed before reading the number ("\\$" before "$")
_DROP_TOKENS = ("\\displaystyle", "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "~",
                "\\(", "\\)", "\\[", "\\]", "*", "\\$", "$", "\\%", "%")


def last_boxed(text: str) -> str | None:
    """Content of the last ``\\boxed{...}`` / ``\\fbox{...}`` with balanced braces;
    None if there is none or its braces never close (a cut-off response)."""
    starts = [m.end() for m in _BOXED.finditer(text)]
    for start in reversed(starts):
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
    return None


def answer_number(answer: str | None) -> float | None:
    """The number an extracted final answer states, or None (see section notes)."""
    if answer is None:
        return None
    s = str(answer)
    for _ in range(3):  # nested \text{\textbf{..}}
        s = _TEXT_CMD.sub(r"\1", s)
    # thousands separators only between digit groups: "1,2,3" stays a list (no match)
    s = _THOUSANDS.sub("", s)
    for token in _DROP_TOKENS:
        s = s.replace(token, "")
    s = re.sub(r"\^\s*\{?\\circ\}?|°", "", s)
    s = re.sub(r"\s+", "", s)
    s = s.split("=")[-1]  # "x=5", "n = 12"
    s = s.rstrip(".")
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if re.fullmatch(_NUMBER, s):
        return float(s)
    m = _UNIT_TAIL.match(s)
    if m:
        return float(m.group(1))
    m = _FRAC.match(s)
    if m and float(m.group(3)) != 0:
        whole = float(m.group(1)) if m.group(1) else 0.0
        frac = float(m.group(2)) / float(m.group(3))
        return whole - frac if whole < 0 else whole + frac
    m = _POWER.match(s)
    if m:
        try:
            return float(m.group(1)) ** float(m.group(2))
        except OverflowError:
            return None
    return None


def final_answer(response: str) -> str | None:
    """The final answer text of a response: last ``\\boxed{}`` after the last
    ``</think>``, else the last "Answer: ..." line, else None."""
    text = (response or "").split("</think>")[-1]
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    lines = _ANSWER_LINE.findall(text)
    return lines[-1] if lines else None


def numeric_answer_grade(response: str, label) -> float:
    if label is None:
        return 0.0
    gold = answer_number(str(label).split("####")[-1])
    pred = answer_number(final_answer(response))
    return 1.0 if (gold is not None and pred is not None and abs(pred - gold) < 1e-6) else 0.0


def numeric_answer_reward(traj: Trajectory) -> RewardResult:
    value = numeric_answer_grade(traj.response or "", traj.label)
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
