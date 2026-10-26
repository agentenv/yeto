"""Framework-neutral agentic task, verdict and trajectory (S25 N10.2).

The yeto core only knows these types.  A harness (Harbor today) turns an
:class:`AgentTask` into a :class:`TrialVerdict`; the outer layer
(``yeto.rl.agentic.runner``) classifies the verdict into an :class:`Outcome`;
a backend adapter (Miles agent function, verl ``HarborAgentLoop``) turns the
result into its own sample type.  Nothing here imports Harbor, Miles or verl.
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class Outcome(str, enum.Enum):
    """How a trajectory ended, from the trainer's point of view.

    * ``OK``: the verifier gave a verdict; train on it.
    * ``POLICY_LIMIT``: the policy hit its own budget (agent timeout, sequence
      or context length).  A trainable negative: reward is the verifier's if it
      ran, else 0.
    * ``ABORTED``: infrastructure failed (sandbox, verifier, sampler endpoint,
      unknown error).  No verdict exists, so the sample must never be trained
      on as a 0 reward; the backend drops it (Miles: ``Sample.Status.ABORTED``).
    """

    OK = "ok"
    POLICY_LIMIT = "policy_limit"
    ABORTED = "aborted"


@dataclass(frozen=True)
class AgentBudget:
    """Per-trajectory limits the harness enforces; ``None`` = harness default."""

    agent_timeout_s: float | None = None
    wall_clock_s: float = 7200.0
    max_seq_len: int | None = None

    def __post_init__(self) -> None:
        if self.agent_timeout_s is not None and not self.agent_timeout_s > 0:
            raise ValueError("agent_timeout_s must be positive")
        if not self.wall_clock_s > 0:
            raise ValueError("wall_clock_s must be positive")
        if self.agent_timeout_s is not None and self.agent_timeout_s >= self.wall_clock_s:
            raise ValueError("agent_timeout_s must be below wall_clock_s (the outer cap)")
        if self.max_seq_len is not None and self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")


@dataclass(frozen=True)
class AgentTask:
    """One agentic sample: which task, which agent, which sandbox provider.

    ``task_id`` names a task directory under the harness's task root (Harbor
    task format); ``agent`` is a harness agent name (``codex``, ``terminus-2``);
    ``agent_version`` pins the agent build (``None`` = harness default, which
    for Harbor's codex agent means ``@latest`` -- avoid that in training).
    """

    task_id: str
    agent: str
    provider: str
    agent_version: str | None = None
    model_name: str = "model"
    budget: AgentBudget = field(default_factory=AgentBudget)
    trajectory_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "agent", "provider"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise ValueError(f"AgentTask.{name} is not a safe identifier: {value!r}")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, defaults: Mapping[str, Any] | None = None) -> "AgentTask":
        """Build from a data row (``instance_id`` is accepted as Miles' name for ``task_id``)."""
        merged = {**(defaults or {}), **{k: v for k, v in row.items() if v is not None}}
        task_id = merged.get("task_id", merged.get("instance_id"))
        budget = AgentBudget(
            agent_timeout_s=_optional_float(merged.get("agent_timeout_s")),
            wall_clock_s=float(merged.get("wall_clock_s", AgentBudget.wall_clock_s)),
            max_seq_len=_optional_int(merged.get("max_seq_len")),
        )
        return cls(
            task_id=task_id,
            agent=merged.get("agent", merged.get("agent_name")),
            provider=merged.get("provider"),
            agent_version=merged.get("agent_version"),
            model_name=str(merged.get("model_name", "model")),
            budget=budget,
            trajectory_id=merged.get("trajectory_id"),
        )


@dataclass(frozen=True)
class TrialVerdict:
    """What a harness reports for one trial, before yeto classifies it.

    ``rewards`` is the verifier's output and the only source of reward
    (trust boundary); ``None`` means the verifier never produced a result.
    ``exception_type`` is the harness's exception class name, if any.
    """

    rewards: Mapping[str, float] | None
    exception_type: str | None = None
    exception_message: str = ""
    timings_s: Mapping[str, float] = field(default_factory=dict)
    agent_metrics: Mapping[str, Any] = field(default_factory=dict)
    trial_dir: str | None = None


@dataclass(frozen=True)
class TokenTurn:
    """One model call as the token recorder saw it (prompt rendered by the sampler).

    ``policy_version`` and ``provenance`` (the sampler source record,
    ``Provenance.to_record()``) are what the recorder knows about the call;
    None when it does not record them.
    """

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...]
    policy_version: int | None = None
    provenance: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if len(self.completion_ids) != len(self.completion_logprobs):
            raise TokenAlignmentError(
                f"turn has {len(self.completion_ids)} completion tokens but "
                f"{len(self.completion_logprobs)} logprobs")
        if not all(math.isfinite(lp) and lp <= 0.0 for lp in self.completion_logprobs):
            raise TokenAlignmentError("completion logprobs must be finite and <= 0")


class TokenAlignmentError(ValueError):
    """Recorded tokens do not form one consistent token stream."""


@dataclass(frozen=True)
class TokenSegments:
    """A whole trajectory as one prompt + one response with a loss mask.

    ``response_mask`` is 1 for tokens the policy generated and 0 for tokens the
    environment appended (tool output, next user message).  Logprobs of
    mask-0 tokens are 0.0 placeholders and never enter the loss.
    """

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    truncated: bool = False

    def __post_init__(self) -> None:
        n = len(self.response_ids)
        if len(self.response_mask) != n or len(self.response_logprobs) != n:
            raise TokenAlignmentError(
                f"response ids/mask/logprobs lengths differ: {n}/"
                f"{len(self.response_mask)}/{len(self.response_logprobs)}")
        if any(m not in (0, 1) for m in self.response_mask):
            raise TokenAlignmentError("response_mask must hold only 0 and 1")


@dataclass(frozen=True)
class AgentTrajectory:
    """The classified result of one agentic sample, backend-neutral."""

    task: AgentTask
    outcome: Outcome
    reward: float
    reason: str
    verdict: TrialVerdict | None
    policy_version: int | None = None
    tokens: TokenSegments | None = None

    def to_metadata(self) -> dict[str, Any]:
        """Flat, JSON-safe fields a backend sample can carry."""
        verdict = self.verdict
        return {
            OUTCOME_KEY: self.outcome.value,
            "reward": self.reward,
            "outcome_reason": self.reason,
            "policy_version": self.policy_version,
            "eval_report": dict(verdict.rewards or {}) if verdict else {},
            "agent_metrics": {**(dict(verdict.agent_metrics) if verdict else {}),
                              **(dict(verdict.timings_s) if verdict else {})},
            "trial_dir": verdict.trial_dir if verdict else None,
        }


OUTCOME_KEY = "yeto_agent_outcome"


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
