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

Execution profiles (rl-infra-spec 1.4/2.2). With ``profile=None`` the driver
keeps the R0 behaviour byte-for-byte (execution mode label
``colocated-serial``, no extra events). With an :class:`ExecutionProfile`:

* the profile's AlgorithmSpec binding is checked in :meth:`handshake`, before
  any engine verb (alignment A1), and the placement kind must match the mode
  (``colocated`` <-> ``colocated-serial``, ``fixed-partition`` <->
  ``partitioned-*``);
* ``partitioned-serial`` runs the same algorithm order on disjoint trainer and
  rollout GPU groups: no offload/onload, one batch in flight, and readiness
  (``generate_blockers``/``train_blockers``) is enforced from a
  :class:`ReadinessSnapshot` before every generation and train step, on top of
  the R0 per-group policy-token check;
* ``partitioned-overlap`` runs only the age-0 overlap implemented in
  ``overlap.py`` (task 2.3): evaluation of the published policy on the rollout
  GPUs overlaps train/outer_sync of the same round and is joined before the
  next publication. Every other overlap (in particular generation ahead of
  publication) is refused.

Reconfiguration (rl-infra-spec 3.1-3.7) is opt-in (``controller=``): the
loop offers the :class:`~yeto.rl.engine.controller.IslandController` one safe
point per round boundary -- after the round's publication was acknowledged
and before the next generation (every optimizer step returned, no gradient
accumulation open, no batch in flight, outer boundary returned non-stop). The
controller runs at most one transaction there; while it runs, admission of new
generations is fenced. ``ledger=`` records every batch
``prepared -> optimizer_applied -> outer_recorded`` durably (3.6). Without
either, the loop and its event tape are unchanged.

Observation (task 1.7) is opt-in (``observe=True``): it adds
``rl_timeline_span`` / ``rl_readiness`` / ``rl_round_labels`` events tagged with
the profile contract hash, config epoch and weight transport. Off, the event
tape is identical to the R0 driver's.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from yeto.rl.core import (
    LocalRoundStats,
    StrictRlInvariantError,
    require_nonzero_learning_rate,
)

from .algorithm import AlgorithmSpec
from .capabilities import EngineCapabilities
from .execution_profile import (
    ExecutionProfile,
    ProfileError,
    ReadinessSnapshot,
    check_algorithm_contract,
    generate_blockers,
    require,
    train_blockers,
)
from .overlap import EvalOverlap, EvalStarter, overlap_refusal
from .ports import (
    Placement,
    PolicyState,
    PublicationCause,
    Publisher,
    RolloutBatchHandle,
    RolloutPool,
    TrainerGroup,
)
from .trainable_state import TrainableState, require_supported_layout

EXECUTION_MODE = "colocated-serial"
# Modes the driver can run (partitioned-overlap: eval||train/outer_sync only, 2.3).
DRIVER_MODES = frozenset({"colocated-serial", "partitioned-serial", "partitioned-overlap"})
_MODE_PLACEMENT = {
    "colocated-serial": "colocated",
    "partitioned-serial": "fixed-partition",
    "partitioned-overlap": "fixed-partition",
}
# Upstream weight transport by placement (miles protocol.py:73-89): colocate
# uses CUDA IPC; a LoRA fixed partition must use NCCL broadcast.
_DEFAULT_TRANSPORT = {"colocated": "cuda-ipc", "fixed-partition": "nccl-broadcast"}
# Timeline role per driver phase (execution_profile.TASK_ROLE vocabulary).
_PHASE_ROLE = {
    "generate": "rollout",
    "train": "trainer",
    "sync": "trainer",
    "apply": "trainer",
    "publish": "trainer+rollout",
    "eval": "rollout",
    "offload": "trainer",
    "onload": "trainer",
}
_PHASE_TASK = {"sync": "outer_sync"}


FAULT_INJECTION_ENV = "YETO_RL_FAULT_INJECTION"
FAULT_INJECTION_FILE = "yeto-rl-fault-injection.json"  # at the repo root


def load_fault_injection(environ: Mapping[str, str] | None = None) -> dict[str, float]:
    """Test-only fault injection (rl-infra-spec 2.3 X9); empty = off (default).

    Source: the JSON file named by ``YETO_RL_FAULT_INJECTION``, else
    ``<repo>/yeto-rl-fault-injection.json`` (present only in an experiment's
    frozen code snapshot). Keys: ``publish_delay_s``.
    """
    import os

    environ = os.environ if environ is None else environ
    path = environ.get(FAULT_INJECTION_ENV)
    candidate = Path(path) if path else Path(__file__).resolve().parents[3] / FAULT_INJECTION_FILE
    if not candidate.is_file():
        return {}
    raw = json.loads(candidate.read_text(encoding="utf-8"))
    unknown = sorted(set(raw) - {"publish_delay_s"})
    if unknown:
        raise ValueError(f"unknown fault injection keys {unknown}")
    delay = float(raw.get("publish_delay_s", 0.0))
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("publish_delay_s must be a non-negative number")
    return {"publish_delay_s": delay} if delay else {}


class DriverError(RuntimeError):
    """The island loop cannot continue safely."""


class RebuildNotStarted(DriverError):
    """4.4: the trainer rebuild was refused before ``rebuild()`` ran (trainer untouched)."""


class PublicationError(DriverError):
    """A publication was incomplete or did not match the requested policy."""

    def __init__(self, message: str = "", *, cause: PublicationCause = PublicationCause.OTHER,
                 engine_ids: Any = ()) -> None:
        super().__init__(message)
        self.cause = PublicationCause(cause)
        self.engine_ids = tuple(sorted(str(e) for e in engine_ids))


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
    # LR each optimizer step of the round applied (read inside the step, before
    # the scheduler advances; ``lr`` is the engine's logged post-step value).
    applied_lrs: tuple[float, ...] | None = None
    # Fraction of loss tokens a masking mechanism removed this round
    # (rl-algorithm-capabilities D6); None when the engine does not report it,
    # which keeps the stricter R0 gradient rule.
    masked_fraction: float | None = None


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

    # Optional: whether the round ``rollout_id`` is the island's last one
    # (strict: local_round_id >= global_rounds; decoupled: the final cut is
    # known).  Sessions without it are treated as non-final.
    # def is_final_round(self, driver, *, rollout_id: int) -> bool: ...

    # Optional (rl-infra-spec 3.8): the outer phase at the round-boundary safe
    # point, one of ``pause_audit.OUTER_PHASES``. Sessions without it are at
    # the pausable phase (the safe point is only offered after a non-stop
    # boundary and a complete publication).
    # def outer_phase(self, driver, *, rollout_id: int) -> str: ...


class ProgressStore(Protocol):
    def after_generate(
        self, *, rollout_id: int, policy_token: str, metrics: Mapping[str, float]
    ) -> None: ...


ECHO_EVENTS_FILE = "yeto-rl-echo-events"  # at the repo root (experiment snapshots only)
# One echo mechanism: yeto.rl.event_echo (YETO_RL_ECHO_EVENTS=1, echoed by the
# single tape writer ``append_record``). The snapshot marker file used by the
# rl-infra-spec E0 experiments just turns that switch on.
if (Path(__file__).resolve().parents[3] / ECHO_EVENTS_FILE).is_file():
    from yeto.rl.event_echo import enable_echo as _enable_echo

    _enable_echo()


class EventTape:
    """JSONL RL event tape in the legacy format (``island_id``, ``time_unix``)."""

    def __init__(self, path: str | Path, island_id: int, *, args: Any = None) -> None:
        self.path = Path(path).expanduser()
        self.island_id = int(island_id)
        self.args = args

    def append(self, event: Mapping[str, Any]) -> None:
        from yeto.rl.event_echo import append_record

        if self.args is not None:
            from yeto.rl.miles import _append_rl_event

            _append_rl_event(self.args, dict(event))
            return
        record = {"island_id": self.island_id, "time_unix": time.time(), **event}
        append_record(self.path, record)


def _round_metrics(batch: RolloutBatchHandle) -> dict[str, float]:
    return {
        "active_groups": float(len(batch.groups)),
        "cancelled_groups": float(batch.aborted),
        "tool_wait_seconds": float(getattr(batch, "tool_wait_seconds", None) or 0.0),
        "group_p50_seconds": 0.0,
        "group_p95_seconds": 0.0,
        "group_p99_seconds": 0.0,
    }


def _filtered_samples(batch: RolloutBatchHandle) -> int | None:
    counts = [getattr(g, "filtered_samples", None) for g in batch.groups]
    if all(c is None for c in counts):
        return None
    return sum(int(c or 0) for c in counts)


def _dynamic_filter_counts(batch: RolloutBatchHandle) -> dict[str, int]:
    """Legacy ``rl/dynamic_filter/*`` round stats from the rollout metadata.

    The all-samples hook sees every generated group; ``filtered`` counts the
    generated groups that were not trained this round (dynamic-filter drops
    and over-sampling leftovers; whether leftovers are reused -- A2/F5
    ``carried_over`` -- is audited in 4.1). ``replacement_attempts`` is a PROXY
    (Miles does not report its resample count); ``rl_round_trained`` records
    the source of each value. Unknown (None) keeps the defaults.
    """
    filtered = getattr(batch, "filtered", None)
    if filtered is None:
        return {}
    trained = len(batch.groups)
    return {
        "dynamic_filter_generated_groups": trained + int(filtered),
        "dynamic_filter_dropped_groups": int(filtered),
        "dynamic_filter_replacement_attempts": int(filtered),
    }


def _sample_ids_sha256(batch: RolloutBatchHandle) -> str:
    import hashlib

    ids = sorted(f"{g.group_id}/{s}" for g in batch.groups for s in g.sample_ids)
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


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
        evaluate_start: EvalStarter | None = None,
        max_rollouts: int | None = None,
        profile: ExecutionProfile | None = None,
        observe: bool = False,
        config_epoch: int = 0,
        clock: Callable[[], float] = time.monotonic,
        controller: Any = None,
        ledger: Any = None,
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
        # Optimizer steps the trainer's scheduler has counted (4.2/4.4 cut
        # progress): set by every apply, advanced by every trained round.
        self.local_step = 0
        self.profile = profile
        self.observe = bool(observe)
        self.config_epoch = int(config_epoch)
        self.clock = clock
        self.execution_mode = profile.execution_mode if profile is not None else EXECUTION_MODE
        self.weight_transport: str | None = None
        self.trained_version: int | None = None
        self._open_span: tuple[str, float, int | None] | None = None
        self.fault_injection = load_fault_injection()
        self.controller = controller
        self.ledger = ledger
        self.published_state: TrainableState | None = None
        self.at_safe_point = False
        self.eval_overlap: EvalOverlap | None = None
        if profile is not None and profile.execution_mode == "partitioned-overlap" and evaluate_start:
            self.eval_overlap = EvalOverlap(evaluate_start, emit=self.emit, clock=self.clock)

    # -- events ----------------------------------------------------------
    def emit(self, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields})

    def phase(self, name: str, **fields: Any) -> None:
        if self.observe:
            self._close_span()
            self._open_span = (name, self.clock(), fields.get("rollout_id"))
        self.emit("rl_driver_phase", phase=name, **fields)

    @property
    def _gated(self) -> bool:
        """Readiness gating runs only in partitioned modes.

        colocated-serial keeps the R0 behaviour exactly (review F3: the train
        gate would require a full rollout_batch_size of groups, which R0 does
        not). In partitioned-serial the gate largely duplicates the R0 checks
        (publication manifest, per-group token); it is kept as the hook that
        partitioned-overlap will need, not as independent evidence (F4).
        """
        return self.profile is not None and self.execution_mode != "colocated-serial"

    # -- observation (task 1.7; opt-in) ------------------------------------
    @property
    def profile_hash(self) -> str | None:
        return self.profile.contract_hash if self.profile is not None else None

    def _labels(self) -> dict[str, Any]:
        return {
            "profile_hash": self.profile_hash,
            "config_epoch": self.config_epoch,
            "execution_mode": self.execution_mode,
            "weight_transport": self.weight_transport,
        }

    def _close_span(self) -> None:
        if not self.observe or self._open_span is None:
            return
        name, start, rollout_id = self._open_span
        self._open_span = None
        if name not in _PHASE_ROLE:
            return
        self.emit(
            "rl_timeline_span",
            task=_PHASE_TASK.get(name, name),
            role=_PHASE_ROLE[name],
            kind="transfer" if name in ("publish", "offload", "onload") else "compute",
            start=start,
            end=self.clock(),
            rollout_id=rollout_id,
            profile_hash=self.profile_hash,
            epoch=self.config_epoch,
        )

    def _snapshot(
        self, rollout_id: int, *, safe_point: bool = True, **fields: Any
    ) -> ReadinessSnapshot:
        trained = self.trained_version if self.trained_version is not None else rollout_id
        published = self.published_version if self.published_version is not None else -1
        snap = ReadinessSnapshot(
            rollout_id=rollout_id,
            optimizer_step=self.rounds_completed,
            trained_policy_version=trained,
            published_policy_version=published,
            publication_complete=self.expected_token is not None,
            driver_safe_point=safe_point,
            config_epoch=self.config_epoch,
            eval_in_flight=self.eval_overlap.in_flight if self.eval_overlap is not None else 0,
            **fields,
        )
        if self.observe:
            self.emit(
                "rl_readiness",
                rollout_id=rollout_id,
                trained_policy_version=snap.trained_policy_version,
                published_policy_version=snap.published_policy_version,
                ready_groups=len(snap.ready_group_ids),
                inflight_batches=snap.inflight_batches,
                **({"eval_in_flight": snap.eval_in_flight} if self.eval_overlap else {}),
                profile_hash=self.profile_hash,
                epoch=self.config_epoch,
            )
        return snap

    # -- startup -----------------------------------------------------------
    def handshake(self) -> None:
        """Capability check; runs before any engine verb is called."""

        require_supported_layout(self.layout, self.capabilities.parameter_layouts)
        if self.profile is not None:
            refusal = overlap_refusal(self.profile)
            if self.execution_mode not in DRIVER_MODES or refusal:
                raise DriverError(
                    f"execution mode {self.execution_mode!r} is not runnable: "
                    f"{refusal or 'unknown mode'} (rl-infra-spec 2.3)"
                )
            if self.execution_mode == "partitioned-overlap" and (
                self.eval_overlap is None or self.evaluate is None or not self.eval_interval
            ):
                raise DriverError(
                    "partitioned-overlap needs evaluate, eval_interval and evaluate_start "
                    "(the eval||train overlap is its only overlapped task; rl-infra-spec 2.3)"
                )
            try:
                check_algorithm_contract(self.profile, self.algorithm)
            except ProfileError as error:
                raise DriverError(f"execution profile rejected: {error}") from error
        description = self.placement.describe()
        if self.profile is not None and description.kind != _MODE_PLACEMENT[self.execution_mode]:
            raise DriverError(
                f"{self.execution_mode} needs placement "
                f"{_MODE_PLACEMENT[self.execution_mode]!r}, got {description.kind!r}"
            )
        self.capabilities.check(
            layout=self.layout,
            placement=description.kind,
            execution_mode=self.execution_mode,
            algorithm=self.algorithm,
            **(
                {"max_policy_age": self.profile.max_policy_age}
                if self.profile is not None
                else {}
            ),
        )
        if not callable(getattr(self.trainer, "step_metrics", None)):
            raise DriverError(
                "trainer group does not report grad_norm; the per-round "
                "gradient invariant cannot be checked"
            )
        self.colocated = description.kind == "colocated"
        self.weight_transport = str(
            description.extra.get("weight_transport")
            or _DEFAULT_TRANSPORT.get(description.kind, "unknown")
        )
        extra_start: dict[str, Any] = {}
        if self.profile is not None and self.observe:
            # observe=False keeps rl_driver_start byte-identical to R0 (1.7).
            extra_start = {"profile_hash": self.profile_hash, "config_epoch": self.config_epoch}
        self.emit(
            "rl_driver_start",
            **extra_start,
            execution_mode=self.execution_mode,
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
        self.local_step = int(local_step)
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
        delay = self.fault_injection.get("publish_delay_s")
        if delay:
            self.emit("rl_fault_injected", kind="publish_delay", seconds=delay,
                      policy_version=rollout_id)
            time.sleep(delay)
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
        self.published_state = state
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

    def _refuse_if_recovery_required(self) -> None:
        """3.7 (recovery-design.md F13): an island that opened in RECOVERY_REQUIRED
        (restart recovery impossible or failed) never publishes, generates or
        trains; the run fails before the first publication."""
        error = getattr(self.controller, "recovery_required", None) if self.controller else None
        if error:
            self.emit("rl_reconfiguration", rollout_id=None, result="RECOVERY_REQUIRED",
                      error=str(error), config_epoch=self.config_epoch)
            raise DriverError(f"island is RECOVERY_REQUIRED: {error}")

    def _probe_nodes(self, rollout_id: int) -> None:
        """rl-multinode-island D9: before each round ask the controller whether every
        island node is still alive; a loss is RECOVERY_REQUIRED and ends the run
        (no partial-node training or generation)."""
        check = getattr(self.controller, "check_nodes", None) if self.controller else None
        if not callable(check):
            return
        error = check()
        if error:
            self.emit("rl_reconfiguration", rollout_id=rollout_id, result="RECOVERY_REQUIRED",
                      error=str(error), config_epoch=self.config_epoch)
            raise DriverError(f"island is RECOVERY_REQUIRED: {error}")

    def _confirm_recovery(self, rollout_id: int) -> None:
        """3.7 restart recovery: after the first full publication covered the
        rebuilt members, the controller verifies them (members, policy token,
        router admission, trainer layout, ledger) and reopens admission; a failed
        verification is RECOVERY_REQUIRED and ends the run (nothing consumed)."""
        confirm = getattr(self.controller, "confirm_recovery", None) if self.controller else None
        if not callable(confirm):
            return
        from .controller import RecoveryRequired

        pending = getattr(self.controller, "recovery_pending", None)
        try:
            confirm(self)
        except RecoveryRequired as error:
            self.emit("rl_reconfiguration", rollout_id=rollout_id, result="RECOVERY_REQUIRED",
                      error=str(error), config_epoch=self.config_epoch)
            raise DriverError(f"island is RECOVERY_REQUIRED: {error}") from error
        if pending is not None:
            self.emit("rl_reconfiguration", rollout_id=rollout_id, result="RECOVERED",
                      recovery_id=pending.get("recovery_id"), config_epoch=self.config_epoch,
                      members=sorted(self.rollout.members()))

    def _generate(self, rollout_id: int) -> RolloutBatchHandle:
        if self.published_version != rollout_id or self.expected_token is None:
            raise PublicationError(
                f"rollout {rollout_id} has no complete publication manifest"
            )
        if self.controller is not None and not self.controller.admission_open:
            # 3.3 admission fence: no new batch while a reconfiguration holds it.
            raise DriverError(f"generation of rollout {rollout_id} refused: admission fenced")
        if self.colocated:
            self.phase("offload", rollout_id=rollout_id)
            self.trainer.offload()
        if self.eval_overlap is not None:
            self.eval_overlap.before_generate(rollout_id)
        if self._gated:
            require(
                generate_blockers(self.profile, self._snapshot(rollout_id)),
                f"generation of rollout {rollout_id}",
            )
        self.phase("generate", rollout_id=rollout_id, policy_version=rollout_id)
        with self._load_sampler(rollout_id):
            # IR-3: the rollout side learns the target policy token up front.
            batch = self.rollout.generate(
                rollout_id, expected_policy_version=self.expected_token
            )
        if batch.rollout_id != rollout_id or batch.policy_version != rollout_id:
            raise PolicyIdentityError(
                f"rollout pool returned rollout {batch.rollout_id} "
                f"(policy v{batch.policy_version}) for rollout {rollout_id}"
            )
        violations = getattr(batch, "policy_age_violation", None)
        if violations:
            # IR-3 (age 0): a sample generated by a policy other than the
            # expected token was ABORTED on the rollout side; the driver never
            # trains on that batch silently.
            raise PolicyIdentityError(
                f"rollout {rollout_id}: {violations} sample(s) not generated by "
                f"{self.expected_token} (policy_age_violation)"
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
        if self.eval_overlap is not None:
            self.eval_overlap.after_generate(
                rollout_id, token=self.expected_token, published_version=self.published_version
            )
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
        grad_norm = float(metrics.grad_norm)
        if not math.isfinite(grad_norm):
            raise StrictRlInvariantError(
                "nonfinite_grad_norm", f"rollout {rollout_id}: grad_norm={grad_norm}"
            )
        # Per-algorithm rule (rl-algorithm-capabilities D6); default GRPO:
        # some group has non-zero reward variance, exactly as in R0. A
        # description without the per-algorithm API keeps the R0 rule.
        r0_expects = any(g.reward_std > 0 for g in batch.groups)
        judge = getattr(self.algorithm, "gradient_expectation", None)
        if callable(judge):
            expects_gradient, relaxed_by = judge(batch, metrics)
        else:
            expects_gradient, relaxed_by = r0_expects, None
        if expects_gradient and grad_norm == 0.0:
            if relaxed_by and relaxed_by.startswith("tightened:"):
                # A mechanism rule requires a gradient the R0 rule cannot see.
                raise StrictRlInvariantError(
                    "zero_grad_norm_with_required_gradient",
                    f"rollout {rollout_id}: {relaxed_by[len('tightened:'):]} requires a "
                    "gradient but grad_norm is 0; adapter gradients are not flowing",
                )
            raise StrictRlInvariantError(
                "zero_grad_norm_with_nonzero_advantages",
                f"rollout {rollout_id}: non-zero advantages produced grad_norm 0; "
                "adapter gradients are not flowing",
            )
        if grad_norm == 0.0 and r0_expects:
            # A declared mechanism legitimately lifted the expectation; the
            # event names how (full mask vs a mechanism's gradient rule).
            masked = bool(relaxed_by) and relaxed_by.startswith("masked:")
            self.emit(
                "rl_zero_gradient_masked" if masked else "rl_zero_gradient_rule_relaxed",
                rollout_id=rollout_id,
                masked_fraction=metrics.masked_fraction,
                relaxed_by=relaxed_by,
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
            tool_wait_seconds=float(getattr(batch, "tool_wait_seconds", None) or 0.0),
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
            applied_lr=None if not metrics.applied_lrs else min(metrics.applied_lrs),
            applied_lrs=metrics.applied_lrs or None,
            **_dynamic_filter_counts(batch),
        )

    def _mismatch_fields(self) -> dict[str, Any]:
        """A5: mismatch metrics with profile/epoch/transport labels; nothing when absent."""
        probe = getattr(self.trainer, "algorithm_metrics", None)
        values = dict(probe() or {}) if callable(probe) else {}
        if not values:
            return {}
        return {"mismatch": values, **{f"label/{k}": v for k, v in self._labels().items()}}

    load_sample_interval_s = 5.0

    @contextmanager
    def _load_sampler(self, rollout_id: int):
        """1.7 (observe only): sample engine in-flight counts while generating."""
        probe = getattr(self.rollout, "load_sample", None)
        if not self.observe or not callable(probe):
            yield
            return
        import threading

        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(self.load_sample_interval_s):
                sample = probe()
                if sample is not None:
                    self.emit("rl_load_sample", rollout_id=rollout_id, **sample,
                              profile_hash=self.profile_hash, epoch=self.config_epoch)

        thread = threading.Thread(target=loop, name="yeto-load-sampler", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=self.load_sample_interval_s + 5)

    def _emit_round_labels(self, rollout_id, batch, metrics) -> None:
        """A5: per-round algorithm metrics carry the same profile/epoch/transport labels."""

        values = {
            "rl/clip_fraction": metrics.clip_fraction,
            "rl/mean_kl": metrics.mean_kl,
            "rl/ess_ratio": metrics.ess_ratio,
            "rl/groups": len(batch.groups),
            "rl/aborted_groups": int(batch.aborted),
            # A2/F5: terminal filtered vs non-terminal carried_over; None when
            # the engine does not report it (never inferred from aborted).
            "rl/filtered_groups": batch.filtered,
            "rl/carried_over_groups": batch.carried_over,
        }
        extra = getattr(self.trainer, "algorithm_metrics", None)
        if callable(extra):
            values.update({str(k): v for k, v in dict(extra() or {}).items()})
        self.emit("rl_round_labels", rollout_id=rollout_id, **self._labels(), **values)

    def _is_final_round(self, rollout_id: int) -> bool:
        probe = getattr(self.sync, "is_final_round", None)
        return bool(probe(self, rollout_id=rollout_id)) if callable(probe) else False

    def _maybe_eval(self, rollout_id: int, *, force: bool = False, defer: bool = False) -> None:
        if self.evaluate is None or not self.eval_interval:
            return
        if not force and rollout_id % self.eval_interval:
            return
        if defer and self.eval_overlap is not None:
            # 2.3: started after the next generation, joined before the next publish.
            self.eval_overlap.schedule(
                rollout_id, token=self.expected_token, published_version=self.published_version
            )
            return
        self.phase("eval", rollout_id=rollout_id)
        metrics = self.evaluate(rollout_id)
        self.emit(
            "rl_eval",
            policy_version=rollout_id,
            **{"rl/policy_token": self.expected_token},
            **{f"eval/{k}": float(v) for k, v in dict(metrics).items()},
        )

    def _join_eval(self) -> None:
        if self.eval_overlap is None:
            return
        done = self.eval_overlap.before_publish(
            token=self.expected_token, published_version=self.published_version
        )
        if done is None:
            return
        if self.observe and done["begin"] is not None and done["end"] is not None:
            # The eval's real interval (recorded inside the eval coroutine), not
            # the generate->join window, so overlap with train is falsifiable.
            self.emit(
                "rl_timeline_span", task="eval", role="rollout", kind="compute",
                start=done["begin"], end=done["end"], rollout_id=done["rollout_id"],
                profile_hash=self.profile_hash, epoch=self.config_epoch,
            )
        self.emit(
            "rl_eval",
            policy_version=done["rollout_id"],
            overlapped=True,
            **{"rl/policy_token": done["token"]},
            **{f"eval/{k}": float(v) for k, v in done["metrics"].items()},
        )

    def run_round(self, rollout_id: int) -> SyncBoundary:
        self.at_safe_point = False
        self._probe_nodes(rollout_id)
        started = time.monotonic()
        batch = self._generate(rollout_id)
        rollout_seconds = time.monotonic() - started
        if self.ledger is not None:
            self.ledger.prepare(batch, policy_token=self.expected_token)
        try:
            return self._train_round(rollout_id, batch, rollout_seconds)
        except BaseException as error:
            if self.ledger is not None and self.ledger.state(rollout_id) == "prepared":
                self.ledger.discard(rollout_id, error=f"{type(error).__name__}: {error}")
            raise

    def _train_round(self, rollout_id: int, batch: RolloutBatchHandle,
                     rollout_seconds: float) -> SyncBoundary:
        if self.progress is not None:
            self.progress.after_generate(
                rollout_id=rollout_id,
                policy_token=self.expected_token,
                metrics={**_round_metrics(batch), "rollout_seconds": rollout_seconds},
            )
        if self.colocated:
            self.phase("onload", rollout_id=rollout_id)
            self.trainer.onload()
        if self._gated:
            # Every group carries the published token (checked in _generate);
            # the batch is one complete, single-policy batch.
            snap = self._snapshot(
                rollout_id,
                ready_group_ids=tuple(g.group_id for g in batch.groups),
                group_policy_versions={g.group_id: batch.policy_version for g in batch.groups},
                inflight_batches=1,
            )
            # The batch size is the engine's (partial rollouts / filtering may
            # yield fewer groups, as in R0); only policy identity is gated here.
            blockers = [
                b for b in train_blockers(self.profile, snap) if "complete groups ready" not in b
            ]
            require(blockers, f"train step of rollout {rollout_id}")
        self.phase("train", rollout_id=rollout_id)
        started = time.monotonic()
        receipt = self.trainer.train_step(batch)
        train_seconds = time.monotonic() - started
        raw = self.trainer.step_metrics()
        metrics = raw if isinstance(raw, TrainStepMetrics) else TrainStepMetrics(**dict(raw))
        self._check_gradient(rollout_id, batch, receipt, metrics)
        self.trained_version = rollout_id + 1
        if self.ledger is not None:
            self.ledger.optimizer_applied(
                rollout_id, input_batch_hash=getattr(receipt, "input_batch_hash", None)
            )
        # Per-round accounting of this island (rl-algo-grpo-knobs 7.2,
        # rl-algo-seq-and-adv): what was trained, masked and counted.
        self.emit(
            "rl_round_trained",
            rollout_id=rollout_id,
            trained_groups=len(batch.groups),
            trained_samples=sum(len(g.sample_ids) for g in batch.groups),
            trained_sample_ids_sha256=_sample_ids_sha256(batch),
            masked_fraction=metrics.masked_fraction,
            clip_fraction=metrics.clip_fraction,
            applied_lrs=list(metrics.applied_lrs) if metrics.applied_lrs else None,
            # rl_local_round dynamic_filter_* provenance: generated/dropped are
            # counted from the all-samples hook; replacement_attempts is a proxy
            # (= groups not trained), not Miles' actual resample count.
            **(
                {"dynamic_filter_source": {
                    "generated_groups": "all_samples_hook_completed_groups",
                    "dropped_groups": "all_samples_hook_not_trained",
                    "replacement_attempts": "proxy_filtered",
                }}
                if getattr(batch, "filtered", None) is not None
                else {}
            ),
            nonzero_advantages=getattr(batch, "nonzero_advantages", None),
            # A2/F5 terminal ``filtered`` at sample level: samples of trained
            # groups masked by a spec-selected sample filter (1b D7); None when
            # no sample filter is configured.
            filtered_samples=_filtered_samples(batch),
            tool_wait_seconds=getattr(batch, "tool_wait_seconds", None),
            submitted_groups=getattr(batch, "submitted_groups", None),
            aborted_in_flight_groups=getattr(batch, "aborted_in_flight_groups", None),
            **self._mismatch_fields(),
            # rl-infra-spec 4.4/A6b: the rollout data cursor after this batch (only
            # when the rollout reports it, i.e. --rl-elastic metadata; else absent)
            **({"data_cursor": dict(batch.data_cursor)}
               if getattr(batch, "data_cursor", None) else {}),
        )
        self.local_step += (
            int(self.profile.optimizer_steps_per_round) if self.profile is not None else 1
        )
        stats = self._stats(rollout_id, batch, metrics, rollout_seconds, train_seconds)
        if self.observe:
            self._emit_round_labels(rollout_id, batch, metrics)
        # Zero-LR invariant: a non-final round must not commit a zero update.
        require_nonzero_learning_rate(stats, final_round=self._is_final_round(rollout_id))
        self.phase("sync", rollout_id=rollout_id)
        boundary = self.sync.boundary(self, rollout_id=rollout_id, stats=stats)
        if self.ledger is not None:
            self.ledger.outer_recorded(rollout_id, next_policy_version=rollout_id + 1)
        self._join_eval()
        self.publish(boundary.state, rollout_id=rollout_id + 1)
        self._close_span()
        self.rounds_completed += 1
        return boundary

    # -- reconfiguration safe point (3.1) -------------------------------------
    def safe_point_snapshot(self, rollout_id: int) -> ReadinessSnapshot:
        """The island at a round boundary (design D4, serial safe point).

        Reached only between ``publish`` of rollout ``rollout_id`` and its
        generation: the previous optimizer step returned (no gradient
        accumulation open), its batch was consumed, the outer boundary returned
        a non-stop result and every member acknowledged the policy. Engine
        in-flight counts come from the rollout's optional ``trajectory_load``
        probe; without it the serial boundary itself proves that no request is
        active (``generate`` returned).
        """
        probe = getattr(self.trainer, "grad_accumulation_open", None)
        grad_open = bool(probe()) if callable(probe) else False
        load_probe = getattr(self.rollout, "trajectory_load", None)
        load = load_probe() if callable(load_probe) else None
        load = dict(load or {})
        unconsumed = self.ledger.unconsumed() if self.ledger is not None else []
        return ReadinessSnapshot(
            rollout_id=rollout_id,
            optimizer_step=self.rounds_completed,
            trained_policy_version=(
                self.trained_version if self.trained_version is not None else rollout_id
            ),
            published_policy_version=(
                self.published_version if self.published_version is not None else -1
            ),
            publication_complete=self.expected_token is not None,
            active_requests=int(load.get("active_requests") or 0),
            tool_wait=int(load.get("tool_wait") or 0),
            # 2.3: a deferred/overlapped eval still holds the rollout role; the
            # controller's WAIT_SAFE refuses to drain/remove until it is joined.
            eval_in_flight=self.eval_overlap.in_flight if self.eval_overlap is not None else 0,
            inflight_batches=len(unconsumed),
            grad_accumulation_open=grad_open,
            driver_safe_point=self.at_safe_point,
            config_epoch=self.config_epoch,
        )

    def outer_phase(self, rollout_id: int) -> str:
        """The outer-sync phase at this safe point (3.8; pause_audit.OUTER_PHASES)."""
        from .pause_audit import PAUSABLE_PHASE

        probe = getattr(self.sync, "outer_phase", None)
        return str(probe(self, rollout_id=rollout_id)) if callable(probe) else PAUSABLE_PHASE

    def refuse_reconfiguration_for_finalization(self, rollout_id: int) -> None:
        """3.8/X6: once the run is finalizing no switch may start; a pending
        request is cancelled (journaled) and later requests are rejected."""
        if self.controller is None:
            return
        refuse = getattr(self.controller, "enter_finalization", None)
        if not callable(refuse):
            return
        poll = getattr(self.controller, "poll_commands", None)
        if callable(poll):
            poll()
        cancelled = refuse(rollout_id=rollout_id)
        for request_id in cancelled:
            self.emit("rl_reconfiguration", rollout_id=rollout_id, result="CANCELLED",
                      request_id=request_id, error="finalization refuses reconfiguration",
                      config_epoch=self.config_epoch)

    def safe_point(self, rollout_id: int) -> str | None:
        """Offer the controller the round-boundary safe point; returns its result phase."""
        self.at_safe_point = True
        if self.controller is None:
            return None
        poll = getattr(self.controller, "poll_commands", None)
        if callable(poll):
            poll()
        if not self.controller.has_pending() and not self.controller.recovery_required:
            return None
        epoch_before = self.config_epoch
        self.phase("reconfigure", rollout_id=rollout_id, config_epoch=epoch_before)
        from .controller import RecoveryRequired

        # 2.3 + 3.x: an eval scheduled but not yet started (it starts after the
        # next generation, on whatever rollout members the reconfiguration leaves).
        eval_due = (
            {"eval_due": None if self.eval_overlap.due is None else self.eval_overlap.due[0]}
            if self.eval_overlap is not None else {}
        )
        outer_phase = self.outer_phase(rollout_id)
        try:
            result = self.controller.run_at_safe_point(
                self, self.safe_point_snapshot(rollout_id), outer_phase=outer_phase
            )
        except RecoveryRequired as error:
            self.emit("rl_reconfiguration", rollout_id=rollout_id, result="RECOVERY_REQUIRED",
                      error=str(error), config_epoch=self.config_epoch, **eval_due)
            raise DriverError(f"island is RECOVERY_REQUIRED: {error}") from error
        if result is not None:
            # the reason of a refusal (REBUILT_OLD: structured publication cause + the
            # disagreeing engine ids) rides on the tape record
            outcome = getattr(self.controller, "last_outcome", None) or {}
            why = ({k: outcome[k] for k in ("error", "cause", "inconsistent_engines")
                    if k in outcome} if outcome.get("phase") == result else {})
            self.emit("rl_reconfiguration", rollout_id=rollout_id, result=result,
                      config_epoch_from=epoch_before, config_epoch=self.config_epoch,
                      members=sorted(self.rollout.members()), **why, **eval_due)
        return result

    # -- same-shape trainer rebuild (4.4) --------------------------------------
    def rebuild_trainer(self, rebuild: Callable[[], Any], *, cut_policy_hash: str) -> Any:
        """Replace the trainer behind the ports and re-publish the same policy (4.4).

        Called at a safe point. ``rebuild`` swaps the handle behind the
        adapter's ``SwappableActor`` and restores the cut (e.g. a closure over
        ``miles_adapter.trainer_rebuild.rebuild_same_shape``); the port objects
        stay, so there is no rebind, no ``initialize`` and no
        ``after_local_train`` (the sync session is not called). Before and
        after, the trainer's policy hash must equal the cut's
        ``progress.policy_hash`` and the currently published policy; the
        re-publication goes through the same publisher/member check as
        :meth:`publish` but does not notify the sync session (no outer effect).
        """
        state = self.published_state
        if state is None or self.published_version is None:
            raise RebuildNotStarted("trainer rebuild before any publication")
        if not self.at_safe_point:
            raise RebuildNotStarted("trainer rebuild outside a safe point")
        published_hash = state.policy_tensor_hash()
        if cut_policy_hash != published_hash:
            raise RebuildNotStarted(
                f"cut policy {cut_policy_hash} is not the published policy {published_hash}"
            )
        self.phase("rebuild", rollout_id=self.published_version)
        result = rebuild()
        restored = self.policy_state.export().policy_tensor_hash()
        if restored != cut_policy_hash:
            raise StrictRlInvariantError(
                "policy_hash_mismatch_after_rebuild",
                f"restored trainer holds {restored}, the cut holds {cut_policy_hash}",
            )
        rollout_id = self.published_version
        if self._rollout_colocated():
            # Colocated: the engines share the GPUs and still hold exactly this
            # policy (checked above: restored == cut == published), resident since
            # the last publish. Publishing again would ask SGLang to resume weights
            # that were never offloaded (KeyError 'weights', scheduler exit), so the
            # engines are left as they are; only the trainer was rebuilt.
            members = frozenset(self.rollout.members())
            self.emit(
                "rl_trainer_rebuilt",
                policy_version=rollout_id,
                republished=False,
                republish_skipped="colocated: engines already hold the published policy",
                **{"rl/policy_token": self.expected_token,
                   "sync/publication_members": sorted(members)},
            )
            return result
        result_pub = self.publisher.publish(state)
        manifest = result_pub.manifest
        if (manifest.target_policy_version != rollout_id
                or manifest.target_policy_hash != cut_policy_hash):
            raise PublicationError("re-publication after rebuild acknowledged another policy")
        members = frozenset(self.rollout.members())
        if not members or result_pub.members != members:
            raise PublicationError(
                f"partial re-publication after rebuild; missing {sorted(members - result_pub.members)}"
            )
        self.emit(
            "rl_trainer_rebuilt",
            policy_version=rollout_id,
            **{"rl/policy_token": self.expected_token,
               "sync/publication_members": sorted(result_pub.members)},
        )
        return result

    def _rollout_colocated(self) -> bool:
        describe = getattr(self.placement, "describe", None)
        kind = getattr(describe(), "kind", None) if callable(describe) else None
        if kind is not None:
            return kind == "colocated"
        return self.execution_mode == "colocated-serial"

    def run(self) -> TrainableState:
        self.handshake()
        self._refuse_if_recovery_required()
        try:
            try:
                start = self.sync.start(self)
                if self.ledger is not None:
                    self.ledger.rebase(start.rollout_id)
                state = start.state
                self.publish(state, rollout_id=start.rollout_id)
                self._confirm_recovery(start.rollout_id)
                self._maybe_eval(start.rollout_id, force=start.rollout_id == 0,
                                 defer=not start.finished)
                rollout_id = start.rollout_id
                finished = start.finished
                while not finished:
                    if self.max_rollouts is not None and rollout_id >= self.max_rollouts:
                        raise DriverError("run reached max_rollouts without a stop")
                    self.safe_point(rollout_id)
                    boundary = self.run_round(rollout_id)
                    state = boundary.state
                    rollout_id += 1
                    self._maybe_eval(rollout_id, force=boundary.stop, defer=not boundary.stop)
                    finished = boundary.stop
                self.refuse_reconfiguration_for_finalization(rollout_id)
                self.phase("finish", rollout_id=rollout_id)
                self._close_span()
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
            if self.eval_overlap is not None:
                self.eval_overlap.abort()  # no orphan eval task on error/cancel paths
            self.sync.close()
