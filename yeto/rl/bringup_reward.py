"""Bring-up reward for engine health checks only (S28 G-M2). Not a task reward.

A model that cannot solve the task (a truncated or random checkpoint used to test the
engine path) earns 0 from every task grader, so GRPO's group advantages are all 0 and
the update does nothing. This score varies between samples of any text, so the health
check sees a real gradient and a real weight change: the fraction of the response's
characters that are ASCII letters or spaces, in [0, 1].
"""

from __future__ import annotations


def score(response: str, label=None, **_kwargs) -> float:
    if not response:
        return 0.0
    good = sum(1 for ch in response if ch == " " or ("a" <= ch.lower() <= "z"))
    return good / len(response)
