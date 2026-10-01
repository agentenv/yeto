"""``ExecutionProfile`` and ``ReadinessSnapshot`` (rl-infra-spec task 1.4, design D0/D3).

Pure and import-light: no torch/ray/miles. The profile freezes, for one run,
which island-internal tasks depend on which, which pairs may overlap, how much
work may be in flight and when a quiescent cut is reachable. The algorithm and
the bridge still decide readiness; this module only answers "may this task
start now under this profile?" from a :class:`ReadinessSnapshot`.

Vocabulary is shared with PR #66 (``yeto.rl.elastic_benchmark.manifest``):
execution modes are ``colocated-serial`` / ``partitioned-serial`` /
``partitioned-overlap`` and a profile can be built from a study manifest's
``profile`` block. Three rules are enforced here (acceptance X9):

* a strict profile (``max_policy_age == 0``) never lets generation start on an
  older policy than the trainer's latest applied update;
* the outer protocol (``strict-avg`` / ``decoupled``) is orthogonal to the
  island-internal policy age: a decoupled outer protocol does NOT make the
  island asynchronous, and it cannot be used to justify ``max_policy_age > 0``;
* a non-zero policy age needs a named algorithm contract that this change does
  not provide (one-step-off-policy is a separate design), so such profiles are
  rejected rather than silently accepted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from ..elastic_benchmark.manifest import (
    EXECUTION_MODES,
    OUTER_PROTOCOLS,
    canonical_execution_mode,
)

PROFILE_SCHEMA = "yeto-rl-execution-profile-v1"

# Island-internal tasks of one round. ``reward`` includes group completion and
# filtering; ``outer_sync`` is the bridge boundary (strict/decoupled hook).
TASKS = ("generate", "reward", "train", "outer_sync", "publish", "eval", "checkpoint")

# Resource role per task in partitioned modes ("cpu" tasks hold no GPU role).
TASK_ROLE = {
    "generate": "rollout",
    "reward": "cpu",
    "train": "trainer",
    "outer_sync": "trainer",
    "publish": "trainer+rollout",
    "eval": "rollout",
    "checkpoint": "trainer",
}

# Same-round data dependencies (task -> tasks it waits for). They are the
# algorithm's, not the scheduler's, so they are identical in every mode.
SAME_ROUND_DEPS: Mapping[str, tuple[str, ...]] = {
    "generate": (),
    "reward": ("generate",),
    "train": ("reward",),
    "outer_sync": ("train",),
    "publish": ("outer_sync",),
    "eval": ("publish",),
    "checkpoint": ("outer_sync",),
}

# Algorithm contracts this change knows. Only "on-policy" is certified; any
# other contract (e.g. "one-step-off-policy") is a separate algorithm design.
ALGORITHM_CONTRACTS = ("on-policy",)
FUTURE_CONTRACTS = ("one-step-off-policy",)


class ProfileError(ValueError):
    """The execution profile is malformed or asks for an uncertified contract."""


UNKNOWN = object()  # a precondition whose value the caller has no source for


def check_overlap_eval(*, placement_kind: Any, eval_uses_snapshots: Any = UNKNOWN,
                       eval_interval: Any = UNKNOWN) -> None:
    """Preconditions of eval overlap (task 2.3); shared by the island
    (miles_adapter.entry) and the launcher's local pre-provisioning check.

    A value passed as :data:`UNKNOWN` is not checked here: a caller without a
    real source for it (the launcher has no eval configuration) leaves that
    precondition to the island, which always passes every value.
    """

    if placement_kind is not UNKNOWN and placement_kind != "fixed-partition":
        raise ProfileError("eval overlap (2.3) needs a fixed-partition placement")
    if eval_uses_snapshots is not UNKNOWN and eval_uses_snapshots:
        # Miles would fire the eval and return; its end could then cross the
        # next publication, which the 2.3 join guard cannot see.
        raise ProfileError("eval overlap (2.3) is refused with --eval-uses-snapshots")
    if eval_interval is not UNKNOWN and not eval_interval:
        raise ProfileError("eval overlap (2.3) needs --eval-interval")


def check_elastic_placement(placement_kind: str) -> None:
    """``--rl-elastic`` (3.x) needs a partitioned island: the controller refuses
    every rollout reconfiguration on a colocated-serial profile."""

    if placement_kind != "fixed-partition":
        raise ProfileError("--rl-elastic needs a fixed-partition placement (colocated "
                           "has no rollout reconfiguration)")


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _pair(a: str, b: str) -> tuple[str, str]:
    if a not in TASKS or b not in TASKS:
        raise ProfileError(f"unknown task in overlap pair {(a, b)}; tasks are {TASKS}")
    if a == b:
        raise ProfileError(f"overlap pair needs two different tasks: {(a, b)}")
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _instance_deps(task: str, rnd: int, age: int) -> list[tuple[str, int]]:
    """Direct dependencies of task instance ``(task, rnd)`` across rounds."""
    deps = [(d, rnd) for d in SAME_ROUND_DEPS[task]]
    if task == "generate":
        deps.append(("publish", rnd - 1 - age))  # policy it samples from
    if task == "train":
        deps.append(("outer_sync", rnd - 1))  # weights the bridge applied
    deps.append((task, rnd - 1))  # a task type runs its rounds in order
    return [(t, r) for t, r in deps if r >= 0]


def _ancestors(task: str, rnd: int, age: int) -> set[tuple[str, int]]:
    stack, seen = [(task, rnd)], set()
    while stack:
        for dep in _instance_deps(*stack.pop(), age):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


def _independent_alignment(a: str, b: str, age: int) -> tuple[int, int] | None:
    """A round alignment in which neither instance transitively waits on the other."""
    base = 2 + age
    for ra, rb in ((base, base), (base, base + 1), (base + 1, base)):
        if (b, rb) not in _ancestors(a, ra, age) and (a, ra) not in _ancestors(b, rb, age):
            return ra, rb
    return None


@dataclass(frozen=True)
class ExecutionProfile:
    """Fixed for a run; the controller never rewrites it (design D0)."""

    name: str
    execution_mode: str
    outer_protocol: str
    algorithm_contract: str = "on-policy"
    max_policy_age: int = 0
    groups_per_batch: int = 1
    samples_per_group: int = 1
    optimizer_steps_per_round: int = 1
    max_inflight_batches: int = 1
    ready_buffer_groups: int = 0  # extra complete groups buffered beyond one batch
    allowed_overlap: frozenset[tuple[str, str]] = frozenset()
    publish_rule: str = "after-every-update"
    # alignment.md A1: the algorithm contract identity is the canonical hash of
    # the run's ``AlgorithmSpec`` (``AlgorithmSpec.sha256()``, 64 hex). None =
    # unbound profile (planning only); the driver refuses to run one.
    algorithm_spec_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("profile name is required")
        digest = self.algorithm_spec_sha256
        if digest is not None and not _is_sha256_hex(digest):
            raise ProfileError("algorithm_spec_sha256 must be 64 lowercase hex characters")
        object.__setattr__(self, "execution_mode", canonical_execution_mode(self.execution_mode))
        if self.execution_mode not in EXECUTION_MODES:
            raise ProfileError(f"execution_mode must be one of {EXECUTION_MODES}")
        if self.outer_protocol not in OUTER_PROTOCOLS:
            raise ProfileError(f"outer_protocol must be one of {OUTER_PROTOCOLS}")
        if self.algorithm_contract not in ALGORITHM_CONTRACTS:
            raise ProfileError(
                f"algorithm contract {self.algorithm_contract!r} is not certified in this "
                "change; a new staleness contract needs its own algorithm design"
            )
        for key in ("groups_per_batch", "samples_per_group", "optimizer_steps_per_round",
                    "max_inflight_batches"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ProfileError(f"{key} must be a positive integer")
        if not isinstance(self.ready_buffer_groups, int) or self.ready_buffer_groups < 0:
            raise ProfileError("ready_buffer_groups must be a non-negative integer")
        if (self.groups_per_batch * self.samples_per_group) % self.optimizer_steps_per_round:
            raise ProfileError(
                "groups_per_batch * samples_per_group must be divisible by "
                "optimizer_steps_per_round (no floor division of samples)"
            )
        if self.max_policy_age != 0:
            # On-policy is the only certified contract and it pins the age to 0.
            # A decoupled OUTER protocol is not an island-internal async contract.
            raise ProfileError(
                "max_policy_age must be 0 under the on-policy contract "
                f"(outer_protocol={self.outer_protocol!r} does not relax island staleness)"
            )
        if self.max_policy_age == 0 and self.max_inflight_batches != 1:
            # Age 0: generation of batch r+1 waits for the publication of the
            # update trained on batch r, so a second batch can never be in flight.
            raise ProfileError("max_policy_age 0 allows exactly one batch in flight")
        pairs = frozenset(_pair(*p) for p in self.allowed_overlap)
        object.__setattr__(self, "allowed_overlap", pairs)
        if self.execution_mode != "partitioned-overlap":
            if pairs:
                raise ProfileError(f"{self.execution_mode} declares no overlapping tasks")
            if self.max_inflight_batches != 1:
                raise ProfileError(f"{self.execution_mode} allows exactly one batch in flight")
        for a, b in pairs:
            illegal = overlap_violation(self, a, b)
            if illegal:
                raise ProfileError(f"overlap {a}||{b} is illegal: {illegal}")

    # -- identity ------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body.pop("extra")
        body["allowed_overlap"] = sorted(list(p) for p in self.allowed_overlap)
        body["schema"] = PROFILE_SCHEMA
        return body

    @property
    def contract_hash(self) -> str:
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest()

    @property
    def batch_samples(self) -> int:
        return self.groups_per_batch * self.samples_per_group

    @property
    def partitioned(self) -> bool:
        return self.execution_mode != "colocated-serial"

    @classmethod
    def from_manifest_profile(
        cls, profile: Mapping[str, Any], work: Mapping[str, Any]
    ) -> "ExecutionProfile":
        """Build from a PR #66 study manifest ``profile`` + ``work`` block."""
        overlap = profile.get("allowed_overlap", [])
        max_groups = profile.get("max_in_flight_groups", work["groups_per_update"])
        return cls(
            name=profile["name"],
            execution_mode=profile["execution_mode"],
            outer_protocol=profile["outer_protocol"],
            algorithm_contract=profile.get("algorithm_contract", "on-policy"),
            max_policy_age=profile.get("max_policy_age", 0),
            groups_per_batch=work["groups_per_update"],
            samples_per_group=work["samples_per_group"],
            optimizer_steps_per_round=profile["optimizer_steps_per_round"],
            max_inflight_batches=profile.get("max_inflight_batches", 1),
            ready_buffer_groups=max(0, max_groups - work["groups_per_update"]),
            allowed_overlap=frozenset(tuple(p) for p in overlap),
            publish_rule=profile.get("publish_rule", "after-every-update"),
            algorithm_spec_sha256=profile.get("algorithm_spec_sha256"),
        )

    def bind_algorithm(self, algorithm: Any) -> "ExecutionProfile":
        """This profile bound to ``algorithm`` (checked by :func:`check_algorithm_contract`)."""
        from dataclasses import replace

        bound = replace(self, algorithm_spec_sha256=algorithm.sha256())
        check_algorithm_contract(bound, algorithm)
        return bound


# Largest policy age any execution mode certified by THIS change can produce
# (design D0 / alignment A1, F6): every mode, partitioned-overlap included, is
# 0. A larger value can only come from a separately certified algorithm contract.
CERTIFIED_MODE_MAX_POLICY_AGE: Mapping[str, int] = {m: 0 for m in EXECUTION_MODES}


def execution_max_policy_staleness(modes: Iterable[str]) -> int:
    """Value for ``EngineCapabilities.execution.max_policy_staleness`` (A1).

    The maximum policy age the declared execution modes can produce. Unknown
    modes are refused rather than assumed stale-free.
    """
    ages = []
    for mode in modes:
        mode = canonical_execution_mode(mode)
        if mode not in CERTIFIED_MODE_MAX_POLICY_AGE:
            raise ProfileError(f"execution mode {mode!r} has no certified policy age")
        ages.append(CERTIFIED_MODE_MAX_POLICY_AGE[mode])
    return max(ages, default=0)


def algorithm_max_policy_staleness(algorithm: Any) -> int:
    """``AlgorithmSpec.execution.max_policy_staleness`` (rl-algorithm-capabilities D1/D4).

    The v1 spec (5 flat fields, R0) has no ``execution`` group; every v1
    algorithm is on-policy GRPO, so it reads as 0. INTERFACE NOTE: written
    against the P0 design (``spec.execution.max_policy_staleness``) before the
    ALGO-CAP v2 dataclass was frozen; re-check when it lands.
    """
    execution = getattr(algorithm, "execution", None)
    if execution is None:
        return 0
    value = getattr(execution, "max_policy_staleness", None)
    if value is None and isinstance(execution, Mapping):
        value = execution.get("max_policy_staleness")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfileError(
            f"AlgorithmSpec.execution.max_policy_staleness must be a non-negative int, got {value!r}"
        )
    return value


def check_algorithm_contract(profile: ExecutionProfile, algorithm: Any) -> None:
    """A1: run BEFORE any GPU process exists. Raises :class:`ProfileError`.

    * the profile must be bound to exactly this algorithm description
      (``algorithm_spec_sha256 == algorithm.sha256()``);
    * ``max_policy_age <= AlgorithmSpec.execution.max_policy_staleness``;
    * the mode's certified maximum age must not exceed what the algorithm
      tolerates either (the engine side of the same comparison).
    """
    digest = algorithm.sha256()
    if profile.algorithm_spec_sha256 is None:
        raise ProfileError(
            f"profile {profile.name!r} is not bound to an AlgorithmSpec "
            "(algorithm_spec_sha256 missing)"
        )
    if profile.algorithm_spec_sha256 != digest:
        raise ProfileError(
            f"profile {profile.name!r} is bound to algorithm {profile.algorithm_spec_sha256}, "
            f"run uses {digest}"
        )
    tolerated = algorithm_max_policy_staleness(algorithm)
    if profile.max_policy_age > tolerated:
        raise ProfileError(
            f"profile max_policy_age={profile.max_policy_age} exceeds the algorithm's "
            f"max_policy_staleness={tolerated}"
        )
    produced = execution_max_policy_staleness([profile.execution_mode])
    if produced > tolerated:
        raise ProfileError(
            f"{profile.execution_mode} may produce policy age {produced}; the algorithm "
            f"tolerates {tolerated}"
        )


def overlap_violation(profile: ExecutionProfile, a: str, b: str) -> str | None:
    """Why tasks ``a`` and ``b`` may not run concurrently, or None when legal.

    Legal only when some round alignment (same round, or one round apart)
    leaves the two instances independent under the profile's policy age, and
    they do not need the same GPU role. With age 0, next-round generation
    always waits for this round's publish, so it cannot overlap train/sync.
    """
    a, b = _pair(a, b)
    if profile.execution_mode != "partitioned-overlap":
        return f"{profile.execution_mode} serializes all tasks"
    if _independent_alignment(a, b, profile.max_policy_age) is None:
        hint = ""
        if "generate" in (a, b) and profile.max_policy_age == 0:
            hint = " (age 0: next-round generation needs this round's published policy)"
        return "data dependency in every round alignment" + hint
    ra, rb = TASK_ROLE[a], TASK_ROLE[b]
    if ra != "cpu" and rb != "cpu" and (set(ra.split("+")) & set(rb.split("+"))):
        return f"both need the {ra}/{rb} GPU role"
    return None


def dependency_table(profile: ExecutionProfile) -> list[dict[str, Any]]:
    """Per-task row: same-round deps, cross-round deps, role, legal overlaps."""
    rows = []
    for task in TASKS:
        cross = []
        if task == "generate":
            cross.append(
                "publish[r-1]" if profile.max_policy_age == 0
                else f"publish[r-1-{profile.max_policy_age}]"
            )
        if task == "train":
            cross.append("outer_sync[r-1]")
        role = TASK_ROLE[task] if profile.partitioned else "shared-pool"
        overlaps = sorted(
            other for other in TASKS
            if other != task and tuple(sorted((task, other))) in profile.allowed_overlap
        )
        rows.append(
            {
                "task": task,
                "depends_on": list(SAME_ROUND_DEPS[task]),
                "depends_on_previous_round": cross,
                "role": role,
                "serialized_by_resource": profile.execution_mode == "colocated-serial",
                "may_overlap_with": overlaps,
            }
        )
    return rows


@dataclass(frozen=True)
class ReadinessSnapshot:
    """What the controller reads (design D3). Filled by the driver/bridge."""

    rollout_id: int
    optimizer_step: int
    trained_policy_version: int  # version the trainer holds after its last update
    published_policy_version: int  # version every rollout member acknowledged
    publication_complete: bool
    ready_group_ids: tuple[str, ...] = ()
    group_policy_versions: Mapping[str, int] = field(default_factory=dict)
    reward_pending: int = 0
    unfinished_trajectories: int = 0
    active_requests: int = 0
    tool_wait: int = 0
    inflight_batches: int = 0
    grad_accumulation_open: bool = False
    publish_in_flight: bool = False
    outstanding_submissions: int = 0
    sync_phase: str = "idle"  # idle | waiting-permit | submitting | applying | finalizing
    driver_safe_point: bool = False
    config_epoch: int = 0
    eval_in_flight: int = 0  # overlapped eval (overlap.py) still holding the rollout role


class ReadinessError(RuntimeError):
    """A task was asked to start while the profile forbids it."""


def generate_blockers(profile: ExecutionProfile, snap: ReadinessSnapshot) -> list[str]:
    """Reasons generation of the next batch may not start (empty = may start)."""
    out = []
    if not snap.publication_complete:
        out.append("publication not acknowledged by every rollout member")
    lag = snap.trained_policy_version - snap.published_policy_version
    if lag > profile.max_policy_age:
        out.append(
            f"published policy v{snap.published_policy_version} is {lag} behind trained "
            f"v{snap.trained_policy_version} (max_policy_age={profile.max_policy_age})"
        )
    if snap.inflight_batches >= profile.max_inflight_batches:
        out.append(
            f"backpressure: {snap.inflight_batches} batches in flight "
            f"(max {profile.max_inflight_batches})"
        )
    capacity = profile.groups_per_batch + profile.ready_buffer_groups
    if len(snap.ready_group_ids) >= capacity:
        out.append(f"backpressure: ready buffer full ({len(snap.ready_group_ids)}/{capacity})")
    if snap.sync_phase == "finalizing":
        out.append("outer finalization in progress")
    return out


def train_blockers(profile: ExecutionProfile, snap: ReadinessSnapshot) -> list[str]:
    """Reasons a train step may not start: complete, same-policy groups only."""
    out = []
    if len(snap.ready_group_ids) < profile.groups_per_batch:
        out.append(f"{len(snap.ready_group_ids)}/{profile.groups_per_batch} complete groups ready")
    if snap.reward_pending:
        out.append(f"{snap.reward_pending} rewards pending")
    batch = snap.ready_group_ids[: profile.groups_per_batch]
    versions = {snap.group_policy_versions.get(g) for g in batch}
    if None in versions:
        out.append("ready group without a recorded policy version")
    versions.discard(None)
    if len(versions) > 1:
        out.append(f"batch mixes policy versions {sorted(versions)}")
    for v in versions:
        if snap.trained_policy_version - v > profile.max_policy_age:
            out.append(f"groups sampled from v{v}, trainer at v{snap.trained_policy_version}")
    if snap.publish_in_flight and profile.execution_mode != "partitioned-overlap":
        out.append("publication in flight")
    return out


def quiescent_cut_blockers(profile: ExecutionProfile, snap: ReadinessSnapshot) -> list[str]:
    """Reasons the island is not at a full-island quiescent cut (design D4)."""
    out = []
    checks = (
        (not snap.driver_safe_point, "driver is not at a safe point"),
        (snap.grad_accumulation_open, "gradient accumulation is open"),
        (snap.publish_in_flight, "publication in flight"),
        (not snap.publication_complete, "last publication not acknowledged"),
        (snap.trained_policy_version != snap.published_policy_version,
         "trained policy not yet published"),
        (snap.reward_pending > 0, f"{snap.reward_pending} rewards pending"),
        (snap.active_requests > 0, f"{snap.active_requests} active engine requests"),
        (snap.tool_wait > 0, f"{snap.tool_wait} trajectories waiting on tools"),
        (snap.unfinished_trajectories > 0,
         f"{snap.unfinished_trajectories} unfinished trajectories"),
        (snap.inflight_batches > 0, f"{snap.inflight_batches} batches in flight"),
        (snap.outstanding_submissions > 0,
         f"{snap.outstanding_submissions} outer submissions outstanding"),
        (snap.sync_phase != "idle", f"outer sync phase {snap.sync_phase!r}"),
        (snap.eval_in_flight > 0, f"{snap.eval_in_flight} overlapped evals in flight"),
    )
    out.extend(reason for failed, reason in checks if failed)
    return out


def require(blockers: Iterable[str], what: str) -> None:
    reasons = list(blockers)
    if reasons:
        raise ReadinessError(f"{what} refused: " + "; ".join(reasons))
