"""Framework-neutral trajectory, reward result and filter decision (decoupling 3.1).

Spec ``rl-neutral-rewards-harness`` / design D5: the fields are the union of
what today's Miles samples carry, so a backend adapter can convert a sample to
a :class:`Trajectory` and apply a :class:`RewardResult` / :class:`FilterDecision`
back without loss.  Neutral reward functions and group filters read and write
only these types -- never a framework namespace (``args``) or sample object.

Status values are neutral lower-case strings (normally one of ``STATUSES``);
adapters translate them and pass unknown framework values through as text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

STATUSES = ("pending", "completed", "truncated", "aborted", "failed")


@dataclass
class Trajectory:
    """One generated sample, framework-neutral."""

    group_index: int | None = None
    index: int | None = None
    prompt: Any = None
    response: str = ""
    label: Any = None
    tokens: Sequence[int] | None = None
    rollout_logprobs: Sequence[float] | None = None
    loss_mask: Sequence[int] | None = None
    policy_token: Any = None
    segments: Sequence[Any] | None = None
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # A reward already attached to the trajectory (group filters read it).
    reward: float | None = None


@dataclass(frozen=True)
class RewardResult:
    """What a neutral reward returns.

    ``metadata`` entries are merged into the trajectory's metadata by the
    adapter (``None`` means: leave metadata untouched).  ``aborted`` asks the
    adapter to mark the sample with its framework's abort status.
    """

    value: float
    aborted: bool = False
    reason: str | None = None
    metadata: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class FilterDecision:
    keep: bool
    reason: str | None = None


NeutralReward = Callable[[Trajectory], RewardResult]
NeutralGroupFilter = Callable[[Sequence[Trajectory]], FilterDecision]
