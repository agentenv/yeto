"""Rollout-side metadata port for verl (decoupling 4.11 role).

On the verl backend the rollout runs inside verl's agent loop and the policy
token comes from the verified publication (``ports_impl``); there is no
separate rollout process reading a driver sink, so the port reports that
nothing is available rather than inventing values.
"""

from __future__ import annotations

from typing import Any


def sink_available() -> bool:
    return False


def current_policy_token() -> str | None:
    return None


def expected_policy_version(sample: Any = None) -> str | None:
    return None


def record_round_metadata(args: Any, round_id: Any, **counters: int) -> None:
    raise NotImplementedError("verl 后端的推理侧计数尚未接入（rl-verl-backend 2.4 之后）")
