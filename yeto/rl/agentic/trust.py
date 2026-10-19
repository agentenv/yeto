"""Trust boundary of the agentic outer layer (S25 N10.2; was L3/L9 of the Codex harness).

The policy controls everything the agent does inside the sandbox, so:

1. reward comes only from the harness verifier result, never from fields the
   agent run produced (``verified_reward``);
2. the agent process gets no trainer secrets or reward material in its
   environment (``check_agent_env``);
3. a task id cannot name a directory outside the task root (``task_dir``).

Harbor's own isolation (tests uploaded only at verify time, optional
``separate`` verifier container) is a second layer, not a replacement: which
mode a Harbor task uses is task data, so yeto checks its own rules here.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from pathlib import Path

from .types import AgentTask


class TrustViolation(RuntimeError):
    """An input crossed the trust boundary; fail closed (the run stops)."""


# Names a policy-controlled process must never see.  Matched case-insensitively
# against environment variable names handed to the agent.
_FORBIDDEN_ENV = re.compile(
    r"(REWARD|VERIFIER|WANDB|HF_TOKEN|HUGGING_FACE|MODAL_TOKEN|AWS_SECRET|"
    r"GITHUB_TOKEN|GH_TOKEN|DAYTONA_API_KEY|E2B_API_KEY|YETO_)", re.IGNORECASE)

REWARD_KEY = "reward"


def check_agent_env(env: Mapping[str, str]) -> dict[str, str]:
    """Return ``env`` unchanged, or raise if a name is on the forbidden list."""
    leaked = sorted(name for name in env if _FORBIDDEN_ENV.search(name))
    if leaked:
        raise TrustViolation(f"agent environment carries forbidden names: {leaked}")
    return dict(env)


def task_dir(root: Path, task: AgentTask) -> Path:
    """The task directory for ``task`` under ``root``; must exist and stay inside it."""
    base = root.resolve()
    path = (base / task.task_id).resolve()
    if base not in path.parents:
        raise TrustViolation(f"task id {task.task_id!r} escapes the task root")
    if not path.is_dir():
        raise FileNotFoundError(f"no task directory for {task.task_id!r} under {root}")
    return path


def verified_reward(rewards: Mapping[str, float], *, low: float = 0.0, high: float = 1.0) -> float:
    """The scalar reward from a verifier result; finite and inside ``[low, high]``.

    Harbor writes ``{"reward": x}`` for single-score tasks; a result with
    several keys must name ``reward`` explicitly (no guessing which one).
    """
    if REWARD_KEY in rewards:
        raw = rewards[REWARD_KEY]
    elif len(rewards) == 1:
        (raw,) = rewards.values()
    else:
        raise TrustViolation(f"verifier result has no {REWARD_KEY!r} among {sorted(rewards)}")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TrustViolation(f"verifier reward is not a number: {raw!r}")
    value = float(raw)
    if not math.isfinite(value) or not low <= value <= high:
        raise TrustViolation(f"verifier reward {value} is outside [{low}, {high}]")
    return value
