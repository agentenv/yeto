"""``IslandDriver``: yeto-owned colocated-serial RL island loop (design D1/D2).

Each round runs, strictly in this order (spec "串行共置执行循环"):

1. generate complete groups with the currently published policy;
2. one trainer optimizer step (then the per-round grad_norm invariant);
3. the outer-sync bridge at the safe boundary (the only place trainer weights
   change besides the optimizer step);
4. one full publication, acknowledged by every rollout member;
5. the next round.

Offload/onload follows the serial colocated branch of upstream ``train.py``:
the trainer is resident while it trains, while the bridge applies a cut, and
while the publisher reads its weights; it is offloaded before generation.
Rollout-side offload is internal to the adapter (the ``RolloutPool`` port has
no offload verb).

The driver talks to the engine only through ``yeto.rl.engine.ports``. Outer
synchronization is a :class:`SyncSession` (see ``bridges.py``). Island progress
checkpoints never contain LoRA or optimizer state; restart applies the
authoritative cut with an optimizer reset.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from yeto.rl.core import LocalRoundStats, StrictRlInvariantError

from .algorithm import AlgorithmSpec
from .capabilities import EngineCapabilities
from .ports import (
    Placement,
    PolicyState,
    Publisher,
    RolloutBatchHandle,
    RolloutPool,
    TrainerGroup,
)
from .trainable_state import TrainableState, require_supported_layout

EXECUTION_MODE = "colocated-serial"


class DriverError(RuntimeError):
    """The island loop cannot continue safely."""


class PublicationError(DriverError):
    """A publication was incomplete or did not match the requested policy."""


class PolicyIdentityError(DriverError):
    """Generated groups were not sampled from the published policy."""


class RoundFailedError(DriverError):
    """The trainer reported a failed optimizer step; nothing was committed."""


def policy_token(rollout_id: int, policy_hash: str) -> str:
    """Rollout policy token, ``yeto:<rollout_id>:<policy_tensor_hash>`` (D3)."""

    from yeto.rl.core import policy_snapshot_token

    return policy_snapshot_token(rollout_id, policy_hash)


@dataclass(frozen=True)
class TrainStepMetrics:
    """Scalar trainer telemetry for the round just trained.

    ``LocalStepReceipt`` carries no grad_norm, so the driver reads it from the
    trainer's optional ``step_metrics()`` verb (see report: port gap).
    """

    grad_norm: float
    loss: float | None = None
    pg_loss: float | None = None
    lr: float | None = None
    mean_kl: float | None = None
    ess_ratio: float | None = None
    clip_fraction: float | None = None
    train_step: int | None = None


@dataclass(frozen=True)
class SyncStart:
    """Authoritative starting point returned by a sync session."""

    state: TrainableState  # already applied to the trainer
    rollout_id: int
    finished: bool = False


@dataclass(frozen=True)
class SyncBoundary:
    state: TrainableState  # already applied; published next as rollout_id + 1
    stop: bool = False


class SyncSession(Protocol):
    def start(self, driver: "IslandDriver") -> SyncStart: ...

    def boundary(
        self, driver: "IslandDriver", *, rollout_id: int, stats: LocalRoundStats
    ) -> SyncBoundary: ...

    def published(self, driver: "IslandDriver", *, rollout_id: int, policy_hash: str) -> None: ...

    def finish(self, driver: "IslandDriver") -> None: ...

    def close(self) -> None: ...


class ProgressStore(Protocol):
    def after_generate(
        self, *, rollout_id: int, policy_token: str, metrics: Mapping[str, float]
    ) -> None: ...


class EventTape:
    """JSONL RL event tape in the legacy format (``island_id``, ``time_unix``)."""

    def __init__(self, path: str | Path, island_id: int, *, args: Any = None) -> None:
        self.path = Path(path).expanduser()
        self.island_id = int(island_id)
        self.args = args

    def append(self, event: Mapping[str, Any]) -> None:
        if self.args is not None:
            from yeto.rl.miles import _append_rl_event

            _append_rl_event(self.args, dict(event))
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"island_id": self.island_id, "time_unix": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def _round_metrics(batch: RolloutBatchHandle) -> dict[str, float]:
    return {
        "active_groups": float(len(batch.groups)),
        "cancelled_groups": float(batch.aborted),
        "tool_wait_seconds": 0.0,
        "group_p50_seconds": 0.0,
        "group_p95_seconds": 0.0,
        "group_p99_seconds": 0.0,
    }


def _pooled_reward(batch: RolloutBatchHandle) -> tuple[float, float]:
    total = sum(len(g.sample_ids) for g in batch.groups)
    if total == 0:
        return 0.0, 0.0
    mean = sum(g.reward_mean * len(g.sample_ids) for g in batch.groups) / total
    second = (
        sum((g.reward_std**2 + g.reward_mean**2) * len(g.sample_ids) for g in batch.groups)
        / total
    )
    return mean, math.sqrt(max(0.0, second - mean * mean))


class IslandDriver:
    def __init__(
        self,
        *,
        learner_id: int,
        rollout: RolloutPool,
        trainer: TrainerGroup,
        policy_state: PolicyState,
        publisher: Publisher,
        placement: Placement,
        capabilities: EngineCapabilities,
        algorithm: AlgorithmSpec,
        sync: SyncSession,
        events: EventTape,
        layout: str = "lora",
        progress: ProgressStore | None = None,
        evaluate: Callable[[int], Mapping[str, float]] | None = None,
        eval_interval: int | None = None,
        max_rollouts: int | None = None,
    ) -> None:
        self.learner_id = int(learner_id)
        self.rollout = rollout
        self.trainer = trainer
        self.policy_state = policy_state
        self.publisher = publisher
        self.placement = placement
        self.capabilities = capabilities
        self.algorithm = algorithm
        self.sync = sync
        self.events = events
        self.layout = layout
        self.progress = progress
        self.evaluate = evaluate
        self.eval_interval = eval_interval
        self.max_rollouts = max_rollouts
        self.colocated = False
        self.expected_token: str | None = None
        self.published_version: int | None = None
        self.rounds_completed = 0

    # -- events ----------------------------------------------------------
    def emit(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})

    def phase(self, name: str, **fields: Any) -> None:
        self.emit("rl_driver_phase", phase=name, **fields)

    # -- startup -----------------------------------------------------------
    def handshake(self) -> None:
        """Capability check; runs before any engine verb is called."""

        require_supported_layout(self.layout, self.capabilities.parameter_layouts)
        description = self.placement.describe()
        self.capabilities.check(
            layout=self.layout,
            placement=description.kind,
            execution_mode=EXECUTION_MODE,
            algorithm=self.algorithm,
        )
        if not callable(getattr(self.trainer, "step_metrics", None)):
            raise DriverError(
                "trainer group does not report grad_norm; the per-round "
                "gradient invariant cannot be checked"
            )
        self.colocated = description.kind == "colocated"
        self.emit(
            "rl_driver_start",
            execution_mode=EXECUTION_MODE,
            placement=description.kind,
            **{"rl/algorithm_spec_sha256": self.algorithm.sha256()},
            runtime_fingerprint=self.capabilities.runtime_fingerprint,
        )

    # -- helpers used by sync sessions -----------------------------------
    def apply_policy(
        self, state: TrainableState, *, optimizer: str, local_step: int
    ) -> TrainableState:
        """Apply one cut to the (resident) trainer and verify it round-trips."""

        self.phase("apply", policy_version=state.policy_version, optimizer=optimizer)
        started = time.monotonic()
        self.policy_state.apply(state, optimizer=optimizer, local_step=local_step)
        applied = self.policy_state.export()
        expected = state.policy_tensor_hash()
        if applied.policy_tensor_hash() != expected:
            raise StrictRlInvariantError(
                "policy_hash_mismatch_after_apply",
                "policy hash mismatch after trainer apply",
            )
        self.emit(
            "rl_policy_apply",
            policy_version=state.policy_version,
            optimizer=optimizer,
            local_step=local_step,
            partial_fragment_apply=optimizer == "preserve",
            **{
                "sync/global_policy_hash": expected,
                "sync/apply_seconds": time.monotonic() - started,
            },
        )
        return state

    def export_local(self) -> TrainableState:
        return self.policy_state.export()

    # -- per-round steps -------------------------------------------------
    def publish(self, state: TrainableState, *, rollout_id: int) -> None:
        if state.policy_version != rollout_id:
            raise PublicationError(
                f"publication of policy version {state.policy_version} "
                f"for rollout {rollout_id}"
            )
        expected_hash = state.policy_tensor_hash()
        self.phase("publish", policy_version=rollout_id)
        result = self.publisher.publish(state)
        manifest = result.manifest
        if (
            manifest.publication_mode != "full"
            or manifest.target_policy_version != rollout_id
            or manifest.target_policy_hash != expected_hash
        ):
            raise PublicationError(
                "publisher acknowledged a different policy: "
                f"v{manifest.target_policy_version} {manifest.target_policy_hash}"
            )
        members = frozenset(self.rollout.members())
        if not members or result.members != members:
            missing = sorted(members - result.members)
            raise PublicationError(
                f"partial publication of policy v{rollout_id}; unacknowledged "
                f"rollout members: {missing}"
            )
        self.expected_token = policy_token(rollout_id, expected_hash)
        self.published_version = rollout_id
        self.sync.published(self, rollout_id=rollout_id, policy_hash=expected_hash)
        self.emit(
            "rl_publication",
            policy_version=rollout_id,
            **{
                "rl/policy_token": self.expected_token,
                "sync/publication_payload_bytes": manifest.payload_bytes,
                "sync/publication_payload_hash": manifest.payload_hash,
                "sync/publication_members": sorted(result.members),
            },
        )

    def _generate(self, rollout_id: int) -> RolloutBatchHandle:
        if self.published_version != rollout_id or self.expected_token is None:
            raise PublicationError(
                f"rollout {rollout_id} has no complete publication manifest"
            )
        if self.colocated:
            self.phase("offload", rollout_id=rollout_id)
            self.trainer.offload()
        self.phase("generate", rollout_id=rollout_id, policy_version=rollout_id)
        batch = self.rollout.generate(rollout_id)
        if batch.rollout_id != rollout_id or batch.policy_version != rollout_id:
            raise PolicyIdentityError(
                f"rollout pool returned rollout {batch.rollout_id} "
                f"(policy v{batch.policy_version}) for rollout {rollout_id}"
            )
        mismatched = batch.mismatched_groups(self.expected_token)
        if mismatched or policy_token(rollout_id, batch.policy_hash) != self.expected_token:
            detail = ", ".join(f"{g.group_id}={g.policy_token}" for g in mismatched)
            raise PolicyIdentityError(
                f"groups not sampled from {self.expected_token}: "
                f"{detail or batch.policy_hash}"
            )
        if not batch.groups:
            raise PolicyIdentityError(f"rollout {rollout_id} produced no complete group")
        bad = [
            g.group_id
            for g in batch.groups
            if not (math.isfinite(g.reward_mean) and math.isfinite(g.reward_std))
        ]
        if bad:
            # NaN would read as "zero variance" in the gradient invariant and
            # poison the pooled reward statistics; refuse before training.
            raise StrictRlInvariantError(
                "nonfinite_reward",
                f"rollout {rollout_id}: non-finite reward in groups {bad}",
            )
        return batch

    def _check_gradient(self, rollout_id: int, batch, receipt, metrics) -> None:
        if not receipt.optimizer_step_succeeded:
            raise RoundFailedError(f"rollout {rollout_id}: optimizer step failed")
        # GRPO advantages are all zero iff every group has zero reward variance.
        advantages_nonzero = any(g.reward_std > 0 for g in batch.groups)
        grad_norm = float(metrics.grad_norm)
        if not math.isfinite(grad_norm):
            raise StrictRlInvariantError(
                "nonfinite_grad_norm", f"rollout {rollout_id}: grad_norm={grad_norm}"
            )
        if advantages_nonzero and grad_norm == 0.0:
            raise StrictRlInvariantError(
                "zero_grad_norm_with_nonzero_advantages",
                f"rollout {rollout_id}: non-zero advantages produced grad_norm 0; "
                "adapter gradients are not flowing",
            )

    def _stats(self, rollout_id, batch, metrics, rollout_seconds, train_seconds):
        reward_mean, reward_std = _pooled_reward(batch)
        groups = len(batch.groups)
        return LocalRoundStats(
            island_id=self.learner_id,
            local_round_id=rollout_id + 1,
            base_policy_version=rollout_id,
            active_groups=groups,
            completed_groups=groups,
            cancelled_groups=int(batch.aborted),
            completed_trajectories=sum(len(g.sample_ids) for g in batch.groups),
            action_tokens=sum(int(g.token_count) for g in batch.groups),
            tool_wait_seconds=0.0,
            group_p50_seconds=0.0,
            group_p95_seconds=0.0,
            group_p99_seconds=0.0,
            reward_mean=reward_mean,
            reward_std=reward_std,
            zero_variance_group_ratio=sum(g.reward_std == 0 for g in batch.groups) / groups,
            mean_kl=metrics.mean_kl,
            ess_ratio=metrics.ess_ratio,
            clip_fraction=metrics.clip_fraction,
            delta_l2_norm=0.0,
            rollout_seconds=rollout_seconds,
            train_seconds=train_seconds,
            train_step=metrics.train_step,
            loss=metrics.loss,
            pg_loss=metrics.pg_loss,
            grad_norm=metrics.grad_norm,
            lr=metrics.lr,
        )

    def _maybe_eval(self, rollout_id: int, *, force: bool = False) -> None:
        if self.evaluate is None or not self.eval_interval:
            return
        if not force and rollout_id % self.eval_interval:
            return
        self.phase("eval", rollout_id=rollout_id)
        metrics = self.evaluate(rollout_id)
        self.emit(
            "rl_eval",
            policy_version=rollout_id,
            **{"rl/policy_token": self.expected_token},
            **{f"eval/{k}": float(v) for k, v in dict(metrics).items()},
        )

    def run_round(self, rollout_id: int) -> SyncBoundary:
        started = time.monotonic()
        batch = self._generate(rollout_id)
        rollout_seconds = time.monotonic() - started
        if self.progress is not None:
            self.progress.after_generate(
                rollout_id=rollout_id,
                policy_token=self.expected_token,
                metrics={**_round_metrics(batch), "rollout_seconds": rollout_seconds},
            )
        if self.colocated:
            self.phase("onload", rollout_id=rollout_id)
            self.trainer.onload()
        self.phase("train", rollout_id=rollout_id)
        started = time.monotonic()
        receipt = self.trainer.train_step(batch)
        train_seconds = time.monotonic() - started
        raw = self.trainer.step_metrics()
        metrics = raw if isinstance(raw, TrainStepMetrics) else TrainStepMetrics(**dict(raw))
        self._check_gradient(rollout_id, batch, receipt, metrics)
        stats = self._stats(rollout_id, batch, metrics, rollout_seconds, train_seconds)
        self.phase("sync", rollout_id=rollout_id)
        boundary = self.sync.boundary(self, rollout_id=rollout_id, stats=stats)
        self.publish(boundary.state, rollout_id=rollout_id + 1)
        self.rounds_completed += 1
        return boundary

    def run(self) -> TrainableState:
        self.handshake()
        try:
            try:
                start = self.sync.start(self)
                state = start.state
                self.publish(state, rollout_id=start.rollout_id)
                self._maybe_eval(start.rollout_id, force=start.rollout_id == 0)
                rollout_id = start.rollout_id
                finished = start.finished
                while not finished:
                    if self.max_rollouts is not None and rollout_id >= self.max_rollouts:
                        raise DriverError("run reached max_rollouts without a stop")
                    boundary = self.run_round(rollout_id)
                    state = boundary.state
                    rollout_id += 1
                    self._maybe_eval(rollout_id, force=boundary.stop)
                    finished = boundary.stop
                self.phase("finish", rollout_id=rollout_id)
                self.sync.finish(self)
                return state
            except StrictRlInvariantError as error:
                self.emit(
                    "rl_strict_failure",
                    metric=error.metric,
                    value=1,
                    error=f"{type(error).__name__}: {error}",
                )
                raise
        finally:
            self.sync.close()
