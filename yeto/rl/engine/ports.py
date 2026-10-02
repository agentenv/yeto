"""Role-based engine ports (design D1). Import-light: no torch/ray/miles.

yeto's driver talks to an RL engine only through these roles. Return types
reuse ``yeto.rl.contracts``. Engine-internal units (cells, CellStatus) never
appear here. E1-E3 verbs (design D9, rl-infra-spec island-elastic-
reconfiguration) are NOT part of the R0 protocol. The E1 verbs (tasks 3.4a/3.4/
3.5) are defined as separate optional protocols (:class:`ElasticRolloutPool`,
:class:`MemberPublisher`, :class:`ReconfigurablePlacement`) so R0 adapters keep
satisfying the base protocols; an adapter advertises a verb via
``EngineCapabilities.port_verbs`` only once it implements it (3.4a).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from yeto.rl.contracts import InferencePublicationManifest, LocalStepReceipt

from .algorithm import AlgorithmSpec
from .capabilities import EngineCapabilities
from .trainable_state import TrainableState

class PublicationCause(str, Enum):
    """Why a (member) publication was refused (A4 E1-B): journaled in REBUILD_OLD /
    REBUILT_OLD and on the tape's ``rl_reconfiguration`` error."""

    PAYLOAD_MISMATCH = "payload_mismatch"  # engine read-back differs from the published payload
    TOKEN_MISMATCH = "token_mismatch"  # an engine does not report the policy token
    UPDATE_FAILED = "update_failed"  # update_weights / update_weight_version raised
    LORA_UNVERIFIABLE = "lora_unverifiable"  # LoRA mode but the read-back has no adapter keys
    OTHER = "other"


OptimizerMode = Literal["preserve", "reset"]
PlacementKind = Literal["colocated", "fixed-partition"]

__all__ = [
    "PublicationCause",
    "AlgorithmSpec",
    "EngineCapabilities",
    "GroupMetadata",
    "ElasticRolloutPool",
    "InferencePublicationManifest",
    "LocalStepReceipt",
    "MemberPublisher",
    "Placement",
    "PlacementDescription",
    "PolicyState",
    "PublicationResult",
    "Publisher",
    "ReconfigurablePlacement",
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
    # Samples of this kept group masked out of the loss by a spec-selected
    # sample filter (rl-algo-grpo-knobs D7; ledger terminal state
    # ``filtered``, alignment A2/F5). None: no sample filter configured.
    filtered_samples: int | None = None


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
    # Groups drawn from the data source (over-sampling included) and those
    # aborted in flight; None = unknown (see rollout_meta_hook.submitted_groups).
    submitted_groups: int | None = None
    aborted_in_flight_groups: int | None = None
    # rl-infra-spec 4.2 CutContext.data: the rollout data source position
    # after this rollout drew its prompts ({sample_offset, epoch_id,
    # sample_group_index, sample_index}) and its reuse-buffer length (0 on the
    # ports path, cut-audit §3). None = not reported.
    data_cursor: Mapping[str, int] | None = field(default=None, compare=False)
    buffer_length: int | None = None
    # IR-3: samples whose actual weight_version(s) differed from the driver's
    # expected_policy_version (ABORTED on the rollout side). None = not reported.
    policy_age_violation: int | None = None

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
    # IR-3: ``expected_policy_version`` is the driver's policy token
    # (``driver.policy_token(rollout_id, policy_hash)``); the pool hands it to
    # the rollout side so agentic generate code can compare it with the
    # per-call ``weight_version`` the engines report (age 0: must be equal).
    def generate(
        self, rollout_id: int, *, expected_policy_version: str | None = None
    ) -> RolloutBatchHandle: ...
    def abort(self) -> None: ...
    def members(self) -> frozenset[str]: ...
    # Optional (4.2): def data_cursor(self) -> Mapping[str, int] | None: ...
    #   the data cursor of the last generated batch (RolloutBatchHandle.data_cursor).
    # E1 verbs (3.4/3.4a): see ElasticRolloutPool below.


@runtime_checkable
class ElasticRolloutPool(RolloutPool, Protocol):
    """E1 rollout membership verbs (rl-infra-spec 3.4, design D1/D6).

    ``epoch`` is the membership epoch the caller (yeto controller journal, the
    single authority, 3.3a) expects the engine to be at; a mismatch is refused
    and a repeated call right after it committed is idempotent. Members are
    opaque yeto member ids (never cells). ``plan_add`` is pure: it names the
    members ``add_engines(count)`` will start, so the caller can journal them
    before acting. ``drain`` stops admission to ``members`` (they keep their
    in-flight requests) and waits until none is in flight; it returns False at
    the deadline without aborting anything; ``undrain`` cancels a drain.
    """

    def plan_add(self, count: int) -> frozenset[str]: ...
    def add_engines(
        self, count: int, *, epoch: int, members: frozenset[str] | None = None
    ) -> frozenset[str]: ...
    def remove_engines(self, members: frozenset[str], *, epoch: int) -> frozenset[str]: ...
    def drain(self, members: frozenset[str], deadline: float) -> bool: ...
    def undrain(self, members: frozenset[str]) -> None: ...
    def membership_status(self) -> Mapping[str, Any]: ...
    def restore_membership(
        self, *, epoch: int, incomplete: Any, last_op: Any, expected_current_epoch: int
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class TrainerGroup(Protocol):
    def train_step(self, batch: RolloutBatchHandle) -> LocalStepReceipt: ...
    def onload(self) -> None: ...
    def offload(self) -> None: ...
    # E2 (optional; advertised via EngineCapabilities.port_verbs, not part of
    # the R0 protocol so R0 fakes stay conforming). Types: yeto.rl.engine.cut,
    # miles_adapter.trainer.CutContext (rl-infra-spec 4.2, cut-audit.md):
    # def layout(self) -> dict[str, int]: ...          # from args
    # def actual_layout(self) -> dict[str, int]: ...   # read back from the ranks
    # def save_cut(self, *, epoch: int, context: CutContext) -> str: ...  # cut id
    # def restore_cut(self, cut_id: str, *, epoch: int, root: str,
    #                 expect: RestoreExpectation, shared_filesystem: bool = True) -> CutManifest: ...
    #   restore_cut is only for a freshly built trainer; any exception from it
    #   means RECOVERY_REQUIRED (never resume on that trainer).
    # Same-shape rebuild (4.3) is miles_adapter.trainer_rebuild.rebuild_same_shape
    # over a SwappableActor: the driver keeps its port objects (no rebind) and
    # re-publishes through IslandDriver.rebuild_trainer (4.4).
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
class MemberPublisher(Publisher, Protocol):
    """Member-scoped publication (3.5, design D6).

    Publishes ``state`` to exactly ``members`` at membership ``epoch``; every
    other member keeps its weights and version. New members stay out of
    routing until their payload read-back matches the publication, and the
    result lists only members that acknowledged it. ``token_rollout_id``
    selects the policy token (the version already served by the others).
    """

    def publish_members(
        self,
        state: TrainableState,
        members: frozenset[str],
        *,
        epoch: int,
        token_rollout_id: int | None = None,
    ) -> PublicationResult: ...


@runtime_checkable
class Placement(Protocol):
    def describe(self) -> PlacementDescription: ...


@runtime_checkable
class ReconfigurablePlacement(Placement, Protocol):
    """``Placement.reconfigure(plan, epoch)`` (design D1; moved from E3 to E1 by 3.4a).

    ``plan`` is the target :class:`PlacementDescription`; ``epoch`` is the
    config epoch the new placement is committed as (current + 1). E1 accepts
    only plans that keep the trainer GPUs; role transfer is E3 (4.7).
    """

    def reconfigure(self, plan: PlacementDescription, *, epoch: int) -> PlacementDescription: ...


