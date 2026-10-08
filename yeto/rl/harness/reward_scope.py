"""Harness reward scope rule (decoupling 3.5, audit A6; codex-harness 7.1).

Reward is attributed per trajectory; a per-segment scope would make sibling
segments pose as independent GRPO samples.  Backend adapters raise their own
configuration error with :func:`reward_scope_problem`'s text.
"""

from __future__ import annotations

from typing import Any

HARNESS_REWARD_SCOPES = ("trajectory",)


class HarnessConfigError(ValueError):
    pass


def reward_scope_problem(scope: Any) -> str | None:
    if scope is None or str(scope) in HARNESS_REWARD_SCOPES:
        return None
    return (f"harness reward_scope={scope!r} is not supported; segments share the "
            f"trajectory reward (allowed: {', '.join(HARNESS_REWARD_SCOPES)})")


def check_harness_reward_scope(scope: Any) -> None:
    problem = reward_scope_problem(scope)
    if problem is not None:
        raise HarnessConfigError(problem)
