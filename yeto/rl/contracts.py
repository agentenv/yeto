"""Algorithm-neutral contracts shared by GRPO, DiLoCo, and future SAO runs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,255}\Z")
_ALGORITHMS = frozenset({"grpo", "sao"})
_TRAINER_EXCHANGE_MODES = frozenset({"dense", "pulseloco"})
_INFERENCE_PUBLICATION_MODES = frozenset({"full", "pulsesync"})


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256")


def _require_identifier(name: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} is not a closed identifier")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_algorithm(value: str) -> None:
    if value not in _ALGORITHMS:
        raise ValueError(f"unsupported local RL algorithm: {value!r}")


@dataclass(frozen=True)
class TrajectoryEnvelope:
    """One exactly-once trajectory and the policy/reward evidence that owns it."""

    trajectory_id: str
    task_id: str
    prompt_group_id: str
    sample_index: int
    behavior_policy_version: int
    behavior_policy_hash: str
    token_ids: tuple[int, ...]
    response_token_count: int
    behavior_logprobs_hash: str | None
    reward: float
    reward_contract_hash: str
    cleanup_evidence_hash: str
    evidence_kind: str | None = None
    active_token_count: int | None = None
    loss_mask_hash: str | None = None
    active_token_ids_hash: str | None = None
    active_behavior_logprobs_hash: str | None = None
    compaction_trajectory_id: str | None = None
    compaction_segment_index: int | None = None
    compaction_segment_type: str | None = None
    compaction_context_budget: int | None = None

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "task_id", "prompt_group_id"):
            _require_identifier(name, getattr(self, name))
        _require_nonnegative_int("sample_index", self.sample_index)
        _require_nonnegative_int(
            "behavior_policy_version", self.behavior_policy_version
        )
        for name in (
            "behavior_policy_hash",
            "reward_contract_hash",
            "cleanup_evidence_hash",
        ):
            _require_sha256(name, getattr(self, name))
        if self.behavior_logprobs_hash is not None:
            _require_sha256("behavior_logprobs_hash", self.behavior_logprobs_hash)
        if not self.token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.token_ids
        ):
            raise ValueError("trajectory token IDs must be non-empty and non-negative")
        _require_nonnegative_int("response_token_count", self.response_token_count)
        if not 1 <= self.response_token_count <= len(self.token_ids):
            raise ValueError(
                "trajectory response-token count must cover a non-empty token suffix"
            )
        if not math.isfinite(self.reward):
            raise ValueError("trajectory reward must be finite")
        if self.evidence_kind is not None:
            _require_identifier("evidence_kind", self.evidence_kind)
        active_fields = (
            self.active_token_count,
            self.loss_mask_hash,
            self.active_token_ids_hash,
        )
        if any(value is not None for value in active_fields):
            if any(value is None for value in active_fields):
                raise ValueError("trajectory active-token evidence is incomplete")
            _require_nonnegative_int(
                "active_token_count", self.active_token_count  # type: ignore[arg-type]
            )
            if self.active_token_count > self.response_token_count:  # type: ignore[operator]
                raise ValueError(
                    "trajectory active-token count exceeds its response length"
                )
            _require_sha256("loss_mask_hash", self.loss_mask_hash)  # type: ignore[arg-type]
            _require_sha256(
                "active_token_ids_hash", self.active_token_ids_hash  # type: ignore[arg-type]
            )
            if self.active_behavior_logprobs_hash is not None:
                _require_sha256(
                    "active_behavior_logprobs_hash",
                    self.active_behavior_logprobs_hash,
                )
        elif self.active_behavior_logprobs_hash is not None:
            raise ValueError(
                "trajectory active log probabilities lack active-token evidence"
            )

        compaction_fields = (
            self.compaction_trajectory_id,
            self.compaction_segment_index,
            self.compaction_segment_type,
            self.compaction_context_budget,
        )
        if any(value is not None for value in compaction_fields):
            if any(value is None for value in compaction_fields):
                raise ValueError("trajectory compaction identity is incomplete")
            _require_identifier(
                "compaction_trajectory_id",
                self.compaction_trajectory_id,  # type: ignore[arg-type]
            )
            _require_nonnegative_int(
                "compaction_segment_index",
                self.compaction_segment_index,  # type: ignore[arg-type]
            )
            if self.compaction_segment_type not in {"execution", "summary"}:
                raise ValueError("trajectory compaction segment type is invalid")
            if (
                type(self.compaction_context_budget) is not int
                or self.compaction_context_budget < 1
            ):
                raise ValueError("trajectory compaction context budget is invalid")


@dataclass(frozen=True)
class LocalStepReceipt:
    """Evidence emitted after one learner completes a safe local-step boundary."""

    algorithm: str
    learner_id: int
    learner_generation: int
    base_policy_version: int
    base_policy_hash: str
    input_batch_hash: str
    trajectory_ids: tuple[str, ...]
    trained_tokens: int
    optimizer_steps: int
    optimizer_step_succeeded: bool
    parameter_layout_hash: str

    def __post_init__(self) -> None:
        _require_algorithm(self.algorithm)
        for name in (
            "learner_id",
            "learner_generation",
            "base_policy_version",
            "trained_tokens",
            "optimizer_steps",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        for name in (
            "base_policy_hash",
            "input_batch_hash",
            "parameter_layout_hash",
        ):
            _require_sha256(name, getattr(self, name))
        if not self.trajectory_ids:
            raise ValueError("local-step receipt must bind at least one trajectory")
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError("local-step receipt contains duplicate trajectories")
        for trajectory_id in self.trajectory_ids:
            _require_identifier("trajectory_id", trajectory_id)
        if not isinstance(self.optimizer_step_succeeded, bool):
            raise TypeError("optimizer_step_succeeded must be a boolean")
        if self.optimizer_step_succeeded:
            if self.optimizer_steps < 1 or self.trained_tokens < 1:
                raise ValueError(
                    "a successful local step must train tokens and advance the optimizer"
                )
        elif self.optimizer_steps != 0:
            raise ValueError("a failed local step cannot claim optimizer advancement")


@dataclass(frozen=True)
class TrainerUpdateManifest:
    """Content identity for either a dense or PULSELoCo trainer update."""

    exchange_mode: str
    learner_id: int
    learner_generation: int
    base_policy_version: int
    target_policy_version: int
    parameter_layout_hash: str
    payload_hash: str
    payload_bytes: int
    fragment_count: int
    complete: bool
    error_feedback_version: int | None = None
    threshold_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if self.exchange_mode not in _TRAINER_EXCHANGE_MODES:
            raise ValueError(f"unsupported trainer exchange: {self.exchange_mode!r}")
        for name in (
            "learner_id",
            "learner_generation",
            "base_policy_version",
            "target_policy_version",
            "payload_bytes",
            "fragment_count",
        ):
            _require_nonnegative_int(name, getattr(self, name))
        for name in ("parameter_layout_hash", "payload_hash"):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.complete, bool):
            raise TypeError("trainer update completeness must be a boolean")
        if self.target_policy_version <= self.base_policy_version:
            raise ValueError("trainer update target must follow its base")
        if self.payload_bytes < 1 or self.fragment_count < 1 or not self.complete:
            raise ValueError("trainer update must bind one complete non-empty payload")
        if self.exchange_mode == "dense":
            if (
                self.error_feedback_version is not None
                or self.threshold_contract_hash is not None
            ):
                raise ValueError("dense trainer updates cannot carry PULSE state")
        else:
            if self.error_feedback_version is None:
                raise ValueError("PULSELoCo update must bind error-feedback state")
            _require_nonnegative_int(
                "error_feedback_version", self.error_feedback_version
            )
            if self.threshold_contract_hash is None:
                raise ValueError("PULSELoCo update must bind a threshold contract")
            _require_sha256("threshold_contract_hash", self.threshold_contract_hash)


@dataclass(frozen=True)
class InferencePublicationManifest:
    """Atomic full-checkpoint or PULSESync publication for rollout workers."""

    publication_mode: str
    base_policy_version: int | None
    target_policy_version: int
    target_policy_hash: str
    target_manifest_hash: str
    payload_hash: str
    payload_bytes: int
    complete: bool

    def __post_init__(self) -> None:
        if self.publication_mode not in _INFERENCE_PUBLICATION_MODES:
            raise ValueError(
                f"unsupported inference publication: {self.publication_mode!r}"
            )
        _require_nonnegative_int("target_policy_version", self.target_policy_version)
        _require_nonnegative_int("payload_bytes", self.payload_bytes)
        for name in (
            "target_policy_hash",
            "target_manifest_hash",
            "payload_hash",
        ):
            _require_sha256(name, getattr(self, name))
        if not isinstance(self.complete, bool):
            raise TypeError("inference publication completeness must be a boolean")
        if self.payload_bytes < 1 or not self.complete:
            raise ValueError("inference publication must be complete and non-empty")
        if self.publication_mode == "full":
            if self.base_policy_version is not None:
                raise ValueError("full publication cannot depend on a base policy")
        else:
            if self.base_policy_version is None:
                raise ValueError("PULSESync publication must bind an exact base")
            _require_nonnegative_int("base_policy_version", self.base_policy_version)
            if self.target_policy_version <= self.base_policy_version:
                raise ValueError("PULSESync target must follow its exact base")
