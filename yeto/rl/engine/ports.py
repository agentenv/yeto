"""Role-based engine ports (design D1). Import-light: no torch/ray/miles.

yeto's driver talks to an RL engine only through these roles. Return types
reuse ``yeto.rl.contracts``. Engine-internal units (cells, CellStatus) never
appear here. E1-E3 verbs (design D9, rl-infra-spec island-elastic-
reconfiguration) are reserved as comments and are NOT part of the R0 protocol;
an adapter advertises them via ``EngineCapabilities.port_verbs`` once added.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt

from .algorithm import AlgorithmSpec
from .capabilities import EngineCapabilities
from .trainable_state import TrainableState

OptimizerMode = Literal["preserve", "reset"]
PlacementKind = Literal["colocated", "fixed-partition"]

__all__ = [
    "AlgorithmSpec",
    "EngineCapabilities",
    "GroupMetadata",
    "InferencePublicationManifest",
    "LocalStepReceipt",
    "Placement",
    "PlacementDescription",
    "PolicyState",
    "PublicationResult",
    "Publisher",
    "RolloutBatchHandle",
    "RolloutPool",
    "TrainableState",
    "TrainerGroup",
]


@dataclass(frozen=True)
class GroupMetadata:
    """Per-group metadata extracted inside the rollout process (design D3)."""

    group_id: str
    sample_ids: tuple[str, ...]
    policy_token: str  # "yeto:<rollout_id>:<policy_hash>"
    reward_mean: float
    reward_std: float
    token_count: int


@dataclass(frozen=True)
class RolloutBatchHandle:
    """Opaque rollout result; yeto sees metadata only, never tokens/tensors.

    ``payload`` is the engine's data reference (e.g. a Ray ObjectRef to the
    upstream ``rollout_data_pack``); yeto code MUST NOT dereference it.
    """

    rollout_id: int
    policy_version: int
    policy_hash: str
    groups: tuple[GroupMetadata, ...]
    completed: int
    aborted: int
    payload: Any = field(default=None, repr=False, compare=False)
    # alignment A2/F5: groups the algorithm intentionally dropped this round
    # (terminal ``filtered``) and reusable leftovers (non-terminal
    # ``carried_over``). None = the engine does not report it (3.6/4.1).
    filtered: int | None = None
    carried_over: int | None = None
    # Non-zero advantages counted by the rollout-side reward dispatcher
    # (rl-algo-seq-and-adv R2); None = not reported. Read by per-algorithm
    # gradient expectations (``gradient_expectation(batch, metrics)``).
    nonzero_advantages: int | None = None
    # rl-infra-spec 1.7: summed non-generation (tool) time of the rollout's
    # samples; None = not reported.
    tool_wait_seconds: float | None = None

    def mismatched_groups(self, expected_token: str) -> tuple[GroupMetadata, ...]:
        return tuple(g for g in self.groups if g.policy_token != expected_token)


@dataclass(frozen=True)
class PublicationResult:
    """Full publication plus the member set that acknowledged it."""

    manifest: InferencePublicationManifest
    members: frozenset[str]


@dataclass(frozen=True)
class PlacementDescription:
    kind: PlacementKind
    trainer_gpus: tuple[str, ...]
    rollout_gpus: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)


@runtime_checkable
class RolloutPool(Protocol):
    def generate(self, rollout_id: int) -> RolloutBatchHandle: ...
    def abort(self) -> None: ...
    def members(self) -> frozenset[str]: ...
    # E1 (reserved): def add_engines(self, count: int, *, epoch: int) -> frozenset[str]: ...
    # E1 (reserved): def remove_engines(self, members: frozenset[str], *, epoch: int) -> frozenset[str]: ...


@runtime_checkable
class TrainerGroup(Protocol):
    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt: ...
    def onload(self) -> None: ...
    def offload(self) -> None: ...
    # E2 (reserved): def save_cut(self, *, epoch: int) -> str: ...  # snapshot id
    # E2 (reserved): def restore_cut(self, snapshot_id: str, *, epoch: int) -> None: ...
    # E3 (reserved): data-parallel resize / role transfer via Placement.reconfigure.


@runtime_checkable
class PolicyState(Protocol):
    def export(self) -> TrainableState: ...
    def apply(
        self,
        state: TrainableState,
        *,
        optimizer: OptimizerMode,
        local_step: int,
    ) -> None: ...


@runtime_checkable
class Publisher(Protocol):
    def publish(self, state: TrainableState) -> PublicationResult: ...


@runtime_checkable
class Placement(Protocol):
    def describe(self) -> PlacementDescription: ...
    # E3 (reserved): def reconfigure(self, target: PlacementDescription, *, epoch: int) -> PlacementDescription: ...


