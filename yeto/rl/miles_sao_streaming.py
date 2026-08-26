"""Full-parameter SAO over role-isolated streaming DiLoCo sessions.

The actor and critic deliberately use independent syncer processes.  Miles
represents each role as a complete, contiguous full-parameter cut, while one
syncer process owns exactly one layout and session.  Paired fragment updates
may remain in flight across multiple local SAO rounds; received role updates
are staged and are applied only as matching actor/critic logical pairs at a
Miles train/publication-safe boundary.  Only actor content is ever published
to rollout workers.

The pairing guarantee is fail-stop and publication-atomic, not crash-atomic
across the two independent syncers.  A permanent one-role failure or durable
checkpoint skew closes both streams and requires operator recovery; this
module does not claim a two-phase commit or durable two-role rollback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from ..protocol import (
    DTYPE_F32,
    FinalManifest,
    PartialMessageGenerationLost,
    PullRequest,
    SyncerClient,
)
from .dense_sweep_wire import MilesRayInboundChunkSink
from .local_learner import ComponentIdentity
from .miles_chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
    ReferencedPolicyCut,
    StoredAuthoritativeFragment,
)
from .trajectory_evidence import (
    TrajectoryBatchEvidence,
    read_trajectory_batch_evidence,
    trajectory_batch_evidence_path,
)

_LOWER_HEX = frozenset("0123456789abcdef")


def _sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class MilesSaoRoleStreamConfig:
    """Closed configuration for one actor or critic streaming session."""

    role: str
    component: ComponentIdentity
    syncer_addr: tuple[str, int]
    learner_id: int
    learner_generation: int
    learner_generations: tuple[int, ...]
    total_fragment_steps: int
    expected_fragments: int
    expected_layout_hash: str
    local_horizon: int
    optimizer_steps_per_round: int
    training_contract_hash: str
    syncer_profile_hash: str
    pipeline_depth: int
    wan_streams: int = 4
    wait_timeout: float = 900.0
    poll_seconds: float = 0.01
    max_fragment_bytes: int = 2 << 30
    max_chunk_bytes: int = 256 << 20

    def __post_init__(self) -> None:
        if self.role not in {"actor", "critic"} or self.component.role != self.role:
            raise ValueError("SAO role-stream component identity changed")
        host, port = self.syncer_addr
        if (
            not isinstance(host, str)
            or not host
            or type(port) is not int
            or not 0 < port < 65536
        ):
            raise ValueError("SAO role-stream syncer address is invalid")
        if (
            not self.learner_generations
            or type(self.learner_id) is not int
            or self.learner_id not in range(len(self.learner_generations))
            or type(self.learner_generation) is not int
            or self.learner_generation < 0
            or self.learner_generations[self.learner_id] != self.learner_generation
            or any(
                type(value) is not int or value < 0
                for value in self.learner_generations
            )
        ):
            raise ValueError("SAO role-stream learner is outside the frozen roster")
        if (
            type(self.expected_fragments) is not int
            or self.expected_fragments < 1
            or type(self.total_fragment_steps) is not int
            or self.total_fragment_steps < self.expected_fragments
            or self.total_fragment_steps % self.expected_fragments
        ):
            raise ValueError("SAO role-stream budget must contain complete sweeps")
        for name in (
            "local_horizon",
            "optimizer_steps_per_round",
            "pipeline_depth",
            "wan_streams",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"SAO role-stream {name} must be positive")
        if self.wait_timeout <= 0 or not 0 < self.poll_seconds <= 1:
            raise ValueError("SAO role-stream wait bounds are invalid")
        if (
            type(self.max_fragment_bytes) is not int
            or not 4 <= self.max_fragment_bytes <= 2 << 30
            or self.max_fragment_bytes % 4
            or type(self.max_chunk_bytes) is not int
            or not 4 <= self.max_chunk_bytes < 1 << 30
            or self.max_chunk_bytes % 4
        ):
            raise ValueError("SAO role-stream transport bounds are invalid")
        _sha256("expected_layout_hash", self.expected_layout_hash)
        _sha256("training_contract_hash", self.training_contract_hash)
        _sha256("syncer_profile_hash", self.syncer_profile_hash)

    @property
    def roster(self) -> dict[int, int]:
        return dict(enumerate(self.learner_generations))


@dataclass(frozen=True)
class MilesSaoStreamingConfig:
    """Actor and critic streams that form one externally visible SAO learner."""

    actor: MilesSaoRoleStreamConfig
    critic: MilesSaoRoleStreamConfig

    def __post_init__(self) -> None:
        if self.actor.role != "actor" or self.critic.role != "critic":
            raise ValueError("SAO streaming configuration must contain both roles")
        if self.actor.syncer_addr == self.critic.syncer_addr:
            raise ValueError("SAO actor and critic require distinct syncer ports")
        lockstep = (
            "learner_id",
            "learner_generation",
            "learner_generations",
            "total_fragment_steps",
            "expected_fragments",
            "local_horizon",
            "pipeline_depth",
            "training_contract_hash",
            "syncer_profile_hash",
        )
        if any(
            getattr(self.actor, name) != getattr(self.critic, name) for name in lockstep
        ):
            raise ValueError(
                "SAO actor and critic streams must use one lockstep schedule"
            )


@dataclass(frozen=True)
class RoleFragmentSubmission:
    role: str
    fragment_id: int
    global_step: int
    round_attempt: int
    base_version: int
    c_steps: int
    c_tokens: int
    payload_bytes: int
    pull_to_push_seconds: float


@dataclass(frozen=True)
class _RoleSubmissionCandidate:
    """One immutable answer to a role syncer's retained PULL permit."""

    role: str
    permit: PullRequest
    base_version: int
    local_step: int
    c_steps: int
    c_tokens: int
    payload_bytes: int
    submission_cut: ReferencedPolicyCut
    anchor: StoredAuthoritativeFragment

    @property
    def logical_key(self) -> tuple[int, int, int]:
        return (self.permit.global_step, self.permit.fragment_id, self.base_version)


@dataclass
class _PairedSubmission:
    """Retained actor/critic operands for one independently in-flight pair."""

    actor: _RoleSubmissionCandidate
    critic: _RoleSubmissionCandidate
    actor_submission: RoleFragmentSubmission | None = None
    critic_submission: RoleFragmentSubmission | None = None
    retained_at: float = field(default_factory=time.monotonic)

    @property
    def logical_key(self) -> tuple[int, int, int]:
        return self.actor.logical_key

    @property
    def complete(self) -> bool:
        return self.actor_submission is not None and self.critic_submission is not None


@dataclass(frozen=True)
class RoleRoundResult:
    role: str
    receipt: object
    applied_fragments: tuple[int, ...]
    submissions: tuple[RoleFragmentSubmission, ...]
    terminal_submitted: bool


@dataclass(frozen=True)
class _PendingPublication:
    rollout_id: int
    actor_content_hash: str
    actor_fragment_versions: tuple[int, ...]
    critic_fragment_versions: tuple[int, ...]
    actor_result: RoleRoundResult
    critic_result: RoleRoundResult
    terminal: bool


def sao_role_stream_session_contract_hash(
    layout_hash: str,
    config: MilesSaoRoleStreamConfig,
) -> bytes:
    """Bind a reconnecting role stream to one immutable SAO experiment."""

    _sha256("layout_hash", layout_hash)
    payload = {
        "schema": "yeto-sao-full-parameter-role-stream-v1",
        "algorithm": "sao",
        "role": config.role,
        "component": asdict(config.component),
        "parameter_layout_hash": layout_hash,
        "training_contract_hash": config.training_contract_hash,
        "syncer_profile_hash": config.syncer_profile_hash,
        "profile": {
            "learner_generations": [
                {"learner_id": learner_id, "generation": generation}
                for learner_id, generation in config.roster.items()
            ],
            "total_fragment_steps": config.total_fragment_steps,
            "fragments": config.expected_fragments,
            "pipeline_depth": config.pipeline_depth,
            "local_horizon": config.local_horizon,
            "optimizer_steps_per_round": config.optimizer_steps_per_round,
            "wire_dtype": "fp32",
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"yeto-sao-role-stream-session-v1\0" + encoded).digest()


class MilesFullParameterRoleStream:
    """Reference-backed full-parameter transport for one SAO trainable role."""

    def __init__(self, config: MilesSaoRoleStreamConfig, *, ray_module=None) -> None:
        self.config = config
        self.ray_module = ray_module
        self.group = None
        self.adapter: MilesChunkedFullParameterAdapter | None = None
        self.client = None
        self.sink = None
        self.current: ReferencedPolicyCut | None = None
        # Keep the latest post-train/pre-broadcast cut until the next local
        # round.  Independent role syncers can deliver the terminal PULL at
        # slightly different times; this immutable cut lets the terminal
        # barrier answer a delayed or replayed permit without another train
        # step, even if a partial broadcast has relabelled ``current`` as
        # generation zero in the meantime.
        self.submission_cut: ReferencedPolicyCut | None = None
        self.submission_fragment_versions: tuple[int, ...] | None = None
        self.submission_rounds = 0
        self.submission_tokens = 0
        self.transport_version = 0
        self.local_generation = 0
        self.local_rounds = 0
        self.action_tokens = 0
        self._anchors: list[StoredAuthoritativeFragment | None] = []
        self._fragment_versions: list[int | None] = []
        self._rounds_at_anchor: list[int] = []
        self._tokens_at_anchor: list[int] = []
        self._pending: dict[int, PullRequest] = {}
        self._answered: dict[tuple[int, int], int] = {}
        self._connection_generation: int | None = None
        self._staged: dict[tuple[int, int], StoredAuthoritativeFragment] = {}
        self._pinned_cuts: dict[int, tuple[ReferencedPolicyCut, int]] = {}
        self._pinned_anchors: dict[int, tuple[StoredAuthoritativeFragment, int]] = {}
        self.terminal_manifest: FinalManifest | None = None
        self.terminal_submitted = False
        self.final_applied = False
        self.final_acknowledged = False

    @property
    def fragment_versions(self) -> tuple[int, ...]:
        if not self._fragment_versions or any(
            version is None for version in self._fragment_versions
        ):
            raise RuntimeError(f"SAO {self.config.role} fragment cut is incomplete")
        return tuple(int(version) for version in self._fragment_versions)

    @property
    def content_hash(self) -> str:
        if self.current is None:
            raise RuntimeError(f"SAO {self.config.role} stream is not initialized")
        return self.current.content_hash

    @property
    def finalizing(self) -> bool:
        event = getattr(self.client, "finalizing", None)
        return bool(
            event is not None
            and callable(getattr(event, "is_set", None))
            and event.is_set()
        )

    async def initialize(self, group) -> None:
        if self.group is not None:
            raise RuntimeError(f"SAO {self.config.role} stream initialized twice")
        adapter, initial = await MilesChunkedFullParameterAdapter.initialize(
            group,
            policy_version=0,
            algorithm="sao",
            stream_role=self.config.role,
            components=(self.config.component,),
            minimum_fragments=self.config.expected_fragments,
            expected_fragments=self.config.expected_fragments,
            expected_layout_hash=self.config.expected_layout_hash,
            max_fragment_bytes=self.config.max_fragment_bytes,
            max_chunk_bytes=self.config.max_chunk_bytes,
        )
        client = SyncerClient(
            self.config.syncer_addr,
            self.config.learner_id,
            adapter.layout.fragments,
            dtype=DTYPE_F32,
            num_streams=self.config.wan_streams,
            max_reconnects=None,
            session_contract_hash=sao_role_stream_session_contract_hash(
                adapter.layout.layout_hash,
                self.config,
            ),
            syncer_profile_hash=bytes.fromhex(self.config.syncer_profile_hash),
        )
        sink = adapter.fragment_sink(initial, ray_module=self.ray_module)
        client.install_inbound_chunk_sink(MilesRayInboundChunkSink(sink))
        try:
            client.start()
            if self.config.learner_id == 0:
                for fragment_id in range(adapter.layout.fragments.num_fragments):
                    queued = client.send_init_parts(
                        fragment_id,
                        adapter.fragment_parts(
                            initial,
                            fragment_id,
                            ray_module=self.ray_module,
                        )(),
                    )
                    if queued is not True:
                        raise RuntimeError(
                            f"SAO {self.config.role} initial fragment was not queued"
                        )
            authoritative = self._wait_for_initial_fragments(client, adapter)
            adapter.verify_initial_fragments(initial, authoritative)
        except BaseException:
            client.close()
            sink.release_all()
            adapter.release(initial)
            raise
        count = adapter.layout.fragments.num_fragments
        self.group = group
        self.adapter = adapter
        self.client = client
        self.sink = sink
        self.current = initial
        self._anchors = list(authoritative)
        self._fragment_versions = [0] * count
        self._rounds_at_anchor = [0] * count
        self._tokens_at_anchor = [0] * count

    def _wait_for_initial_fragments(
        self,
        client,
        adapter: MilesChunkedFullParameterAdapter,
    ) -> tuple[StoredAuthoritativeFragment, ...]:
        deadline = time.monotonic() + self.config.wait_timeout
        observed: dict[int, StoredAuthoritativeFragment] = {}
        count = adapter.layout.fragments.num_fragments
        while len(observed) < count:
            client.check_health()
            for update in client.drain_updates():
                stored = self._stored_update(update)
                if update.version != 0 or stored.fragment_id in observed:
                    raise RuntimeError(
                        f"SAO {self.config.role} initial broadcast changed"
                    )
                observed[stored.fragment_id] = stored
            if len(observed) < count:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"SAO {self.config.role} timed out waiting for initial cut"
                    )
                time.sleep(self.config.poll_seconds)
        return tuple(observed[index] for index in range(count))

    @staticmethod
    def _stored_update(update) -> StoredAuthoritativeFragment:
        if not getattr(update, "stored", False) or not isinstance(
            update.data, StoredAuthoritativeFragment
        ):
            raise RuntimeError(
                "SAO streaming received an untyped full-parameter fragment"
            )
        if (
            update.data.fragment_id != update.fragment_id
            or update.data.version != update.version
        ):
            raise RuntimeError("SAO streaming fragment envelope identity changed")
        return update.data

    async def complete_local_round(
        self,
        *,
        rollout_id: int,
        trajectories: TrajectoryBatchEvidence,
    ) -> RoleRoundResult:
        if (
            self.adapter is None
            or self.current is None
            or self.group is None
            or self.client is None
            or self.final_applied
        ):
            raise RuntimeError(f"SAO {self.config.role} stream is not trainable")
        expected_generation = self.local_generation + 1
        receipt = await self.adapter.record_local_round(
            self.group,
            anchor=self.current,
            rollout_id=rollout_id,
            learner_id=self.config.learner_id,
            learner_generation=self.config.learner_generation,
            trajectories=trajectories,
            role=self.config.role,
            expected_local_step_generation=expected_generation,
            expected_optimizer_steps=self.config.optimizer_steps_per_round,
        )
        local = await self.adapter.capture(
            self.group,
            policy_version=self.transport_version,
            local_step_generation=expected_generation,
        )
        previous = self.current
        previous_submission = self.submission_cut
        self.current = local
        self.submission_cut = local
        self.submission_fragment_versions = self.fragment_versions
        self.local_generation = expected_generation
        self.local_rounds += 1
        self.action_tokens += trajectories.trained_tokens
        self.submission_rounds = self.local_rounds
        self.submission_tokens = self.action_tokens
        self._release_cuts(previous, previous_submission, keep=(local,))

        # The policy-level coordinator owns permit matching and release.  A
        # role must never answer a PULL here: the peer role may not yet have
        # the corresponding permit or an H-qualified immutable cut.
        return RoleRoundResult(
            self.config.role,
            receipt,
            (),
            (),
            False,
        )

    async def stage_available_broadcasts(self) -> tuple[tuple[int, int], ...]:
        """Drain complete broadcasts without mutating the trainable model."""

        if self.client is None:
            raise RuntimeError(f"SAO {self.config.role} client is unavailable")
        self.client.check_health()
        staged = []
        for update in sorted(
            self.client.drain_updates(),
            key=lambda value: (value.fragment_id, value.version),
        ):
            stored = self._stored_update(update)
            self._validate_fragment_version(stored.fragment_id, stored.version)
            previous_version = self.fragment_versions[stored.fragment_id]
            if stored.version <= previous_version:
                continue
            key = (stored.fragment_id, stored.version)
            previous = self._staged.get(key)
            if (
                previous is not None
                and previous.wire_payload_hash != stored.wire_payload_hash
            ):
                raise RuntimeError("SAO staged fragment replay changed bytes")
            if previous is None:
                self._staged[key] = stored
                staged.append(key)
        return tuple(staged)

    @property
    def staged_fragment_versions(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._staged))

    def has_staged_broadcast(self, fragment_id: int, version: int) -> bool:
        return (fragment_id, version) in self._staged

    async def apply_staged_broadcast(
        self,
        fragment_id: int,
        version: int,
    ) -> tuple[int, ...]:
        """Apply one coordinator-approved staged fragment at a safe boundary."""

        key = (fragment_id, version)
        stored = self._staged.get(key)
        if stored is None:
            raise RuntimeError(
                f"SAO {self.config.role} matching staged fragment disappeared"
            )
        expected_base = max(0, version - self.config.expected_fragments)
        if self.fragment_versions[fragment_id] != expected_base:
            raise RuntimeError(
                f"SAO {self.config.role} staged fragment skipped its base version"
            )
        await self._apply_authoritative({fragment_id: stored})
        self._staged.pop(key, None)
        return (fragment_id,)

    async def _apply_authoritative(
        self,
        selected: Mapping[int, StoredAuthoritativeFragment],
    ) -> None:
        if self.adapter is None or self.current is None or self.group is None:
            raise RuntimeError(f"SAO {self.config.role} stream is unavailable")
        target_version = self.transport_version + 1
        target = self.adapter.assemble_mixed_target(
            self.current,
            target_policy_version=target_version,
            authoritative_fragments=tuple(selected.values()),
        )
        before = tuple(await self.group.full_parameter_optimizer_states())
        await self.adapter.apply(
            self.group,
            target,
            commit_token=self._apply_token(target_version, selected),
        )
        after = tuple(await self.group.full_parameter_optimizer_states())
        _validate_optimizer_apply_transition(
            before,
            after,
            role=self.config.role,
            base_policy_version=self.transport_version,
            base_local_generation=self.local_generation,
            target_policy_version=target_version,
        )
        previous_cut = self.current
        self.current = target
        self.transport_version = target_version
        self.local_generation = 0
        self._release_cuts(
            previous_cut,
            keep=(self.current, self.submission_cut),
        )
        for fragment_id, stored in selected.items():
            previous_anchor = self._anchors[fragment_id]
            if previous_anchor is not None and previous_anchor is not stored:
                self._release_anchor(previous_anchor)
            self._anchors[fragment_id] = stored
            self._fragment_versions[fragment_id] = stored.version
            self._rounds_at_anchor[fragment_id] = self.local_rounds
            self._tokens_at_anchor[fragment_id] = self.action_tokens

    def _apply_token(
        self,
        target_version: int,
        fragments: Mapping[int, StoredAuthoritativeFragment],
    ) -> str:
        payload = json.dumps(
            {
                "role": self.config.role,
                "learner_id": self.config.learner_id,
                "transport_version": target_version,
                "fragments": [
                    {
                        "fragment_id": fragment_id,
                        "version": stored.version,
                        "payload_hash": stored.wire_payload_hash,
                    }
                    for fragment_id, stored in sorted(fragments.items())
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return f"sao-{self.config.role}:{hashlib.sha256(payload).hexdigest()}"

    def ready_submission_candidates(self) -> tuple[_RoleSubmissionCandidate, ...]:
        """Return H-qualified retained permits without releasing any bytes."""

        if (
            self.client is None
            or self.adapter is None
            or self.submission_cut is None
            or self.submission_fragment_versions is None
            or self.final_applied
        ):
            return ()
        self.client.check_health()
        self._queue_pulls()
        candidates = []
        for fragment_id in sorted(self._pending):
            permit = self._pending[fragment_id]
            version = self.fragment_versions[fragment_id]
            expected_base = max(
                0,
                permit.global_step - self.config.expected_fragments,
            )
            if (
                version != expected_base
                or self.submission_fragment_versions[fragment_id] != expected_base
            ):
                continue
            c_steps = self.submission_rounds - self._rounds_at_anchor[fragment_id]
            if c_steps < self.config.local_horizon:
                continue
            c_tokens = self.submission_tokens - self._tokens_at_anchor[fragment_id]
            if c_tokens < 1:
                raise RuntimeError("SAO role-stream token counter moved backwards")
            anchor = self._anchors[fragment_id]
            if anchor is None:
                raise RuntimeError(
                    "SAO role-stream fragment has no authoritative anchor"
                )
            candidates.append(
                _RoleSubmissionCandidate(
                    self.config.role,
                    permit,
                    version,
                    self.submission_rounds,
                    c_steps,
                    c_tokens,
                    self.adapter.layout.fragments.fragments[fragment_id].numel * 4,
                    self.submission_cut,
                    anchor,
                )
            )
        return tuple(candidates)

    def retain_candidate(self, candidate: _RoleSubmissionCandidate) -> None:
        """Pin an exact local cut and authoritative anchor across boundaries."""

        if (
            candidate.role != self.config.role
            or self.adapter is None
            or candidate.submission_cut is not self.submission_cut
            or self._anchors[candidate.permit.fragment_id] is not candidate.anchor
        ):
            raise RuntimeError("SAO paired submission candidate cannot be retained")
        self._pin(self._pinned_cuts, candidate.submission_cut)
        try:
            self._pin(self._pinned_anchors, candidate.anchor)
        except BaseException:
            self._unpin_cut(candidate.submission_cut)
            raise

    def release_candidate(self, candidate: _RoleSubmissionCandidate) -> None:
        """Release a committed pair's exact operands when no pair still owns them."""

        self._unpin_cut(candidate.submission_cut)
        self._unpin_anchor(candidate.anchor)

    def retained_replay_candidate(
        self,
        candidate: _RoleSubmissionCandidate,
    ) -> _RoleSubmissionCandidate | None:
        """Rebind an exact retained operand to a reconnect's newer PULL attempt."""

        if self.client is None or self.final_applied:
            return None
        self.client.check_health()
        self._queue_pulls()
        permit = self._pending.get(candidate.permit.fragment_id)
        if permit is None:
            return None
        if (
            permit.global_step != candidate.permit.global_step
            or permit.fragment_id != candidate.permit.fragment_id
            or permit.round_attempt < candidate.permit.round_attempt
            or id(candidate.submission_cut) not in self._pinned_cuts
            or id(candidate.anchor) not in self._pinned_anchors
        ):
            raise RuntimeError("SAO reconnect replay changed a retained candidate")
        return replace(candidate, permit=permit)

    def push_candidate(
        self,
        candidate: _RoleSubmissionCandidate,
    ) -> RoleFragmentSubmission | None:
        """Release one coordinator-approved candidate on its exact permit.

        A zero-chunk outage returns ``None``.  A partial-generation loss is
        allowed to escape to the paired coordinator, which retains both cuts
        and waits for this exact logical PULL to be replayed after reconnect.
        """

        if (
            candidate.role != self.config.role
            or self.client is None
            or self.adapter is None
            or self.final_applied
            or id(candidate.submission_cut) not in self._pinned_cuts
            or id(candidate.anchor) not in self._pinned_anchors
        ):
            raise RuntimeError("SAO paired submission candidate changed")
        permit = self._pending.get(candidate.permit.fragment_id)
        if permit != candidate.permit:
            return None
        queued = self.client.push_fragment_parts(
            candidate.permit.fragment_id,
            candidate.permit.global_step,
            candidate.permit.round_attempt,
            candidate.base_version,
            candidate.local_step,
            candidate.c_steps,
            candidate.c_tokens,
            self.adapter.delta_parts_from_authoritative(
                candidate.anchor,
                candidate.submission_cut,
                candidate.permit.fragment_id,
                ray_module=self.ray_module,
            )(),
        )
        if queued is not True:
            return None
        self._answered[
            (candidate.permit.global_step, candidate.permit.round_attempt)
        ] = int(getattr(self.client, "connection_generation", 0))
        self._pending.pop(candidate.permit.fragment_id)
        terminal = candidate.permit.global_step == self.config.total_fragment_steps
        self.terminal_submitted = self.terminal_submitted or terminal
        return RoleFragmentSubmission(
            self.config.role,
            candidate.permit.fragment_id,
            candidate.permit.global_step,
            candidate.permit.round_attempt,
            candidate.base_version,
            candidate.c_steps,
            candidate.c_tokens,
            candidate.payload_bytes,
            max(0.0, time.monotonic() - candidate.permit.received_at),
        )

    def _queue_pulls(self) -> None:
        connection_generation = int(getattr(self.client, "connection_generation", 0))
        if self._connection_generation != connection_generation:
            self._pending.clear()
            self._answered.clear()
            self._connection_generation = connection_generation
        for permit in self.client.drain_pulls():
            self._validate_permit(permit)
            key = (permit.global_step, permit.round_attempt)
            if self._answered.get(key) == connection_generation:
                continue
            previous = self._pending.get(permit.fragment_id)
            if previous is not None and previous != permit:
                raise RuntimeError("SAO role-stream received conflicting PULLs")
            self._pending.setdefault(permit.fragment_id, permit)

    def _validate_permit(self, permit: PullRequest) -> None:
        if (
            not 0 <= permit.fragment_id < self.config.expected_fragments
            or not 1 <= permit.global_step <= self.config.total_fragment_steps
            or (permit.global_step - 1) % self.config.expected_fragments
            != permit.fragment_id
            or permit.round_attempt < 1
        ):
            raise RuntimeError("SAO role-stream received an invalid PULL")

    def _validate_fragment_version(self, fragment_id: int, version: int) -> None:
        if (
            not 0 <= fragment_id < self.config.expected_fragments
            or not 0 <= version <= self.config.total_fragment_steps
            or (
                version > 0
                and (version - 1) % self.config.expected_fragments != fragment_id
            )
        ):
            raise RuntimeError("SAO role-stream received an invalid fragment version")

    async def wait_and_apply_final(self) -> None:
        if self.final_applied:
            return
        if self.client is None:
            raise RuntimeError(f"SAO {self.config.role} client is unavailable")
        manifest, fragments = await asyncio.to_thread(
            self.client.wait_for_final_fragments,
            self.config.wait_timeout,
        )
        expected = tuple(
            range(
                self.config.total_fragment_steps - self.config.expected_fragments + 1,
                self.config.total_fragment_steps + 1,
            )
        )
        if (
            manifest.global_step != self.config.total_fragment_steps
            or manifest.versions != expected
            or [fragment.fragment_id for fragment in fragments]
            != list(range(self.config.expected_fragments))
        ):
            raise RuntimeError(f"SAO {self.config.role} final cut is incomplete")
        selected = {}
        for fragment in fragments:
            stored = self._stored_update(fragment)
            if stored.version != manifest.versions[stored.fragment_id]:
                raise RuntimeError("SAO final fragment differs from its manifest")
            selected[stored.fragment_id] = stored
        await self._apply_authoritative(selected)
        self._staged.clear()
        self.terminal_manifest = manifest
        self.final_applied = True
        self._release_cuts(
            self.submission_cut,
            keep=(self.current,),
        )
        self.submission_cut = None
        self.submission_fragment_versions = None

    def acknowledge_final(self, manifest: FinalManifest | None = None) -> None:
        manifest = self.terminal_manifest if manifest is None else manifest
        if (
            not self.final_applied
            or manifest is None
            or manifest != self.terminal_manifest
            or self.client is None
        ):
            raise RuntimeError(f"SAO {self.config.role} final ACK is out of order")
        if self.final_acknowledged:
            return
        self.client.acknowledge_finalization(manifest)
        self.final_acknowledged = True

    def discard_detached_submission_cut(self) -> None:
        """Drop the temporary pre-apply cut after a nonterminal boundary."""

        if self.submission_cut is self.current:
            return
        self._release_cuts(
            self.submission_cut,
            keep=(self.current,),
        )
        self.submission_cut = None
        self.submission_fragment_versions = None

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        if self.sink is not None:
            self.sink.release_all()
        retained_cuts = tuple(value for value, _count in self._pinned_cuts.values())
        self._pinned_cuts.clear()
        self._pinned_anchors.clear()
        self._release_cuts(self.current, self.submission_cut, *retained_cuts)
        self._staged.clear()
        self.current = None
        self.submission_cut = None
        self.submission_fragment_versions = None

    @staticmethod
    def _pin(store: dict[int, tuple[Any, int]], value: Any) -> None:
        key = id(value)
        previous = store.get(key)
        if previous is None:
            store[key] = (value, 1)
            return
        if previous[0] is not value:
            raise RuntimeError("SAO retained reference identity collided")
        store[key] = (value, previous[1] + 1)

    def _unpin_cut(self, cut: ReferencedPolicyCut) -> None:
        remaining = self._unpin(self._pinned_cuts, cut)
        if (
            remaining == 0
            and cut is not self.current
            and cut is not self.submission_cut
            and self.adapter is not None
        ):
            self.adapter.release(cut)

    def _unpin_anchor(self, anchor: StoredAuthoritativeFragment) -> None:
        remaining = self._unpin(self._pinned_anchors, anchor)
        if remaining == 0 and all(value is not anchor for value in self._anchors):
            anchor.release()

    @staticmethod
    def _unpin(store: dict[int, tuple[Any, int]], value: Any) -> int:
        key = id(value)
        previous = store.get(key)
        if previous is None or previous[0] is not value or previous[1] < 1:
            raise RuntimeError("SAO retained reference was released out of order")
        remaining = previous[1] - 1
        if remaining:
            store[key] = (value, remaining)
        else:
            store.pop(key)
        return remaining

    def _release_anchor(self, anchor: StoredAuthoritativeFragment) -> None:
        if id(anchor) not in self._pinned_anchors:
            anchor.release()

    def _release_cuts(
        self,
        *cuts: ReferencedPolicyCut | None,
        keep: Sequence[ReferencedPolicyCut | None] = (),
    ) -> None:
        if self.adapter is None:
            return
        keep_ids = {id(value) for value in keep if value is not None}
        released: set[int] = set()
        for cut in cuts:
            if cut is None or id(cut) in keep_ids or id(cut) in released:
                continue
            if id(cut) in self._pinned_cuts:
                continue
            self.adapter.release(cut)
            released.add(id(cut))


class MilesSaoStreamingPolicySync:
    """Nonblocking paired-stream callback with fail-stop cross-role safety.

    Pair submission and WAN completion overlap later local SAO rounds up to
    ``pipeline_depth``.  Broadcasts are staged per role, then matching logical
    actor/critic fragments are applied at a Miles callback boundary.  The two
    applies cannot be made crash-atomic across independent syncers/processes;
    any apply skew or permanent role failure therefore fail-stops both roles.
    """

    supports_critic = True
    requires_exact_publication_info = True

    def __init__(self, args: Any, config: MilesSaoStreamingConfig) -> None:
        self.args = args
        self.config = config
        self.actor = MilesFullParameterRoleStream(config.actor)
        self.critic = MilesFullParameterRoleStream(config.critic)
        self.actor_model = None
        self.critic_model = None
        self.rollout_manager = None
        self.initial_publication_pending = False
        self.pending_publication: _PendingPublication | None = None
        self.published_rollout_id = 0
        self.published_actor_hash: str | None = None
        self.finished = False
        self._inflight_pairs: dict[tuple[int, int, int], _PairedSubmission] = {}
        self._paired_journal: list[
            tuple[RoleFragmentSubmission, RoleFragmentSubmission]
        ] = []
        self._pull_skew_since: dict[tuple[int, int, int], float] = {}
        self._terminal_seen = False
        self._paired_failure: BaseException | None = None

    async def initialize(self, *, actor_model, critic_model, rollout_manager) -> None:
        if os.environ.get("MILES_EXPERIMENTAL_FT_TRAINER", "0") != "0":
            raise RuntimeError("SAO full-parameter streaming requires Miles v1 actors")
        if self.actor_model is not None:
            raise RuntimeError("SAO streaming synchronizer initialized twice")
        if int(getattr(self.args, "start_rollout_id", 0)) != 0:
            raise RuntimeError("SAO streaming currently requires a version-zero start")
        if getattr(self.args, "yeto_rl_trajectory_evidence_dir", None) is None:
            raise RuntimeError("SAO streaming requires immutable trajectory evidence")
        await self.actor.initialize(actor_model)
        try:
            await self.critic.initialize(critic_model)
        except BaseException:
            self.actor.close()
            raise
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.rollout_manager = rollout_manager
        self.initial_publication_pending = True
        self.published_rollout_id = 0
        self.args.start_rollout_id = 0

    async def after_local_train(
        self,
        *,
        rollout_id: int,
        actor_model,
        critic_model,
        rollout_data,
    ) -> bool:
        del rollout_data
        if (
            actor_model is not self.actor_model
            or critic_model is not self.critic_model
            or self.initial_publication_pending
            or self.pending_publication is not None
            or self.finished
            or rollout_id != self.published_rollout_id
            or self.published_actor_hash != self.actor.content_hash
        ):
            raise RuntimeError("Miles called SAO streaming outside its safe boundary")
        from .miles import get_current_published_policy_identity

        if get_current_published_policy_identity(
            self.args,
            expected_policy_version=rollout_id,
        ) != (rollout_id, self.published_actor_hash):
            raise RuntimeError("SAO rollout does not bind the published actor")
        trajectories = self._trajectory_evidence(rollout_id)
        actor_result, critic_result = await self._complete_local_rounds(
            rollout_id=rollout_id,
            trajectories=trajectories,
        )
        actor_submissions, critic_submissions = await self._submit_paired_ready()
        actor_applied, critic_applied = await self._commit_staged_paired_broadcasts()
        actor_result = self._extend_result(
            actor_result,
            submissions=actor_submissions,
            applied=actor_applied,
        )
        critic_result = self._extend_result(
            critic_result,
            submissions=critic_submissions,
            applied=critic_applied,
        )
        terminal = (
            self.actor.finalizing or self.critic.finalizing or self._terminal_seen
        )
        if terminal:
            (
                actor_result,
                critic_result,
            ) = await self._finish_paired_without_more_training(
                actor_result,
                critic_result,
            )
        else:
            self.actor.discard_detached_submission_cut()
            self.critic.discard_detached_submission_cut()
        actor_versions = self.actor.fragment_versions
        critic_versions = self.critic.fragment_versions
        if actor_versions != critic_versions:
            error = RuntimeError(
                "SAO actor and critic committed fragment cuts diverged"
            )
            self._fail_stop(error)
            raise error
        pending = _PendingPublication(
            rollout_id=rollout_id + 1,
            actor_content_hash=self.actor.content_hash,
            actor_fragment_versions=actor_versions,
            critic_fragment_versions=critic_versions,
            actor_result=actor_result,
            critic_result=critic_result,
            terminal=terminal,
        )
        self.pending_publication = pending
        self._record_round(pending)
        return terminal

    async def _complete_local_rounds(
        self,
        *,
        rollout_id: int,
        trajectories: TrajectoryBatchEvidence,
    ) -> tuple[RoleRoundResult, RoleRoundResult]:
        """Cancel the peer callback and fail-stop both streams on role failure."""

        tasks = (
            asyncio.create_task(
                self.actor.complete_local_round(
                    rollout_id=rollout_id,
                    trajectories=trajectories,
                )
            ),
            asyncio.create_task(
                self.critic.complete_local_round(
                    rollout_id=rollout_id,
                    trajectories=trajectories,
                )
            ),
        )
        try:
            actor_result, critic_result = await asyncio.gather(*tasks)
            return actor_result, critic_result
        except BaseException as error:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._fail_stop(error)
            raise

    def _fail_stop(self, error: BaseException) -> None:
        if self._paired_failure is None:
            self._paired_failure = error
        for stream in (self.actor, self.critic):
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as close_error:  # noqa: BLE001 - fail-stop cleanup
                    error.add_note(
                        f"{getattr(stream, 'role', 'role')} close failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )

    @staticmethod
    def _extend_result(
        result: RoleRoundResult,
        *,
        submissions: Sequence[RoleFragmentSubmission] = (),
        applied: Sequence[int] = (),
    ) -> RoleRoundResult:
        merged_submissions = result.submissions + tuple(submissions)
        return RoleRoundResult(
            result.role,
            result.receipt,
            tuple(dict.fromkeys(result.applied_fragments + tuple(applied))),
            merged_submissions,
            result.terminal_submitted,
        )

    def _new_pair(
        self,
        actor: _RoleSubmissionCandidate,
        critic: _RoleSubmissionCandidate,
    ) -> _PairedSubmission:
        if (
            actor.logical_key != critic.logical_key
            or actor.local_step != critic.local_step
            or actor.c_steps != critic.c_steps
            or actor.c_tokens != critic.c_tokens
            or actor.c_steps < self.config.actor.local_horizon
            or critic.c_steps < self.config.critic.local_horizon
        ):
            raise RuntimeError(
                "SAO actor and critic retained cuts are not one H-qualified round"
            )
        return _PairedSubmission(actor, critic)

    @staticmethod
    def _same_candidate(
        retained: _RoleSubmissionCandidate,
        replayed: _RoleSubmissionCandidate,
    ) -> bool:
        return (
            retained.logical_key == replayed.logical_key
            and replayed.permit.round_attempt >= retained.permit.round_attempt
            and retained.local_step == replayed.local_step
            and retained.c_steps == replayed.c_steps
            and retained.c_tokens == replayed.c_tokens
            and retained.payload_bytes == replayed.payload_bytes
            and retained.submission_cut is replayed.submission_cut
            and retained.anchor is replayed.anchor
        )

    @property
    def _effective_pipeline_depth(self) -> int:
        return min(
            self.config.actor.pipeline_depth,
            self.config.actor.expected_fragments,
        )

    @staticmethod
    def _candidate_map(
        candidates: Sequence[_RoleSubmissionCandidate],
        *,
        role: str,
    ) -> dict[tuple[int, int, int], _RoleSubmissionCandidate]:
        result = {}
        for candidate in candidates:
            if candidate.logical_key in result:
                raise RuntimeError(f"SAO {role} returned duplicate logical PULLs")
            result[candidate.logical_key] = candidate
        return result

    def _retain_pair(self, pair: _PairedSubmission) -> None:
        retained_actor = False
        try:
            retain = getattr(self.actor, "retain_candidate", None)
            if callable(retain):
                retain(pair.actor)
            retained_actor = True
            retain = getattr(self.critic, "retain_candidate", None)
            if callable(retain):
                retain(pair.critic)
        except BaseException:
            if retained_actor:
                release = getattr(self.actor, "release_candidate", None)
                if callable(release):
                    release(pair.actor)
            raise

    def _release_pair(self, pair: _PairedSubmission) -> None:
        for stream, candidate in (
            (self.actor, pair.actor),
            (self.critic, pair.critic),
        ):
            release = getattr(stream, "release_candidate", None)
            if callable(release):
                release(candidate)

    @staticmethod
    def _retained_replay(
        retained: _RoleSubmissionCandidate,
        observed: _RoleSubmissionCandidate | None,
    ) -> _RoleSubmissionCandidate | None:
        if observed is None:
            return None
        if (
            observed.logical_key != retained.logical_key
            or observed.permit.round_attempt < retained.permit.round_attempt
        ):
            raise RuntimeError("SAO reconnect replay changed logical identity")
        # The current model may have advanced through more local rounds while
        # this PUSH was in flight.  Only the new permit is consumed: bytes,
        # counters, cut, and anchor remain the exact retained operands.
        return replace(retained, permit=observed.permit)

    def _push_pair_once(
        self,
        pair: _PairedSubmission,
        *,
        actor_observed: _RoleSubmissionCandidate | None,
        critic_observed: _RoleSubmissionCandidate | None,
    ) -> tuple[RoleFragmentSubmission | None, RoleFragmentSubmission | None]:
        produced: list[RoleFragmentSubmission | None] = []
        for role_name, stream, observed in (
            ("actor", self.actor, actor_observed),
            ("critic", self.critic, critic_observed),
        ):
            retained = getattr(pair, role_name)
            replayed = self._retained_replay(retained, observed)
            if replayed is None:
                produced.append(None)
                continue
            setattr(pair, role_name, replayed)
            # A PULL visible after a previous successful enqueue proves that
            # generation did not commit it.  Replay the exact retained bytes
            # under the newly observed attempt.
            setattr(pair, f"{role_name}_submission", None)
            try:
                submission = stream.push_candidate(replayed)
            except PartialMessageGenerationLost:
                submission = None
            if submission is not None:
                setattr(pair, f"{role_name}_submission", submission)
            produced.append(submission)
        return produced[0], produced[1]

    async def _submit_paired_ready(
        self,
    ) -> tuple[tuple[RoleFragmentSubmission, ...], tuple[RoleFragmentSubmission, ...]]:
        """Nonblockingly fill the paired pipeline from currently visible PULLs."""

        if self._paired_failure is not None:
            raise RuntimeError(
                "SAO paired submission coordinator is fail-stopped"
            ) from (self._paired_failure)
        actor_submissions: list[RoleFragmentSubmission] = []
        critic_submissions: list[RoleFragmentSubmission] = []
        try:
            actor_by_key = self._candidate_map(
                self.actor.ready_submission_candidates(),
                role="actor",
            )
            critic_by_key = self._candidate_map(
                self.critic.ready_submission_candidates(),
                role="critic",
            )
            now = time.monotonic()
            total = self.config.actor.total_fragment_steps
            if any(key[0] == total for key in actor_by_key | critic_by_key):
                self._terminal_seen = True

            # Existing pairs consume only a replayed permit.  Their original
            # reference-backed operands stay pinned even after later local
            # training boundaries replace the role's current cut.
            for key, pair in tuple(self._inflight_pairs.items()):
                actor_submission, critic_submission = self._push_pair_once(
                    pair,
                    actor_observed=actor_by_key.pop(key, None),
                    critic_observed=critic_by_key.pop(key, None),
                )
                if actor_submission is not None:
                    actor_submissions.append(actor_submission)
                if critic_submission is not None:
                    critic_submissions.append(critic_submission)

            capacity = self._effective_pipeline_depth - len(self._inflight_pairs)
            shared = sorted(actor_by_key.keys() & critic_by_key.keys())
            for key in shared[: max(0, capacity)]:
                pair = self._new_pair(actor_by_key.pop(key), critic_by_key.pop(key))
                self._retain_pair(pair)
                self._inflight_pairs[key] = pair
                actor_submission, critic_submission = self._push_pair_once(
                    pair,
                    actor_observed=pair.actor,
                    critic_observed=pair.critic,
                )
                if actor_submission is not None:
                    actor_submissions.append(actor_submission)
                if critic_submission is not None:
                    critic_submissions.append(critic_submission)

            unmatched = set(actor_by_key) ^ set(critic_by_key)
            self._pull_skew_since = {
                key: self._pull_skew_since.get(key, now) for key in unmatched
            }
            timeout = min(
                self.config.actor.wait_timeout,
                self.config.critic.wait_timeout,
            )
            if any(
                now - started >= timeout for started in self._pull_skew_since.values()
            ):
                raise TimeoutError(
                    "SAO timed out waiting for a peer role's matching PULL"
                )
            for pair in self._inflight_pairs.values():
                if now - pair.retained_at < timeout:
                    continue
                if pair.complete:
                    raise TimeoutError(
                        "SAO timed out waiting for a paired WAN broadcast"
                    )
                raise TimeoutError(
                    "SAO timed out with a one-sided paired submission in flight"
                )
            return tuple(actor_submissions), tuple(critic_submissions)
        except BaseException as error:
            self._fail_stop(error)
            raise

    async def _commit_staged_paired_broadcasts(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Apply currently matched role broadcasts without waiting for WAN I/O."""

        actor_applied: list[int] = []
        critic_applied: list[int] = []
        try:
            await asyncio.gather(
                self.actor.stage_available_broadcasts(),
                self.critic.stage_available_broadcasts(),
            )
            known = {
                (pair.actor.permit.fragment_id, pair.actor.permit.global_step)
                for pair in self._inflight_pairs.values()
            }
            actor_staged = set(self.actor.staged_fragment_versions)
            critic_staged = set(self.critic.staged_fragment_versions)
            unknown = (actor_staged | critic_staged) - known
            if unknown:
                raise RuntimeError(
                    "SAO role syncer broadcast has no retained paired submission"
                )

            for key, pair in sorted(self._inflight_pairs.items()):
                step, fragment_id, _base = key
                if step == self.config.actor.total_fragment_steps:
                    continue
                actor_ready = self.actor.has_staged_broadcast(fragment_id, step)
                critic_ready = self.critic.has_staged_broadcast(fragment_id, step)
                if not (actor_ready and critic_ready):
                    continue
                if not pair.complete:
                    raise RuntimeError(
                        "SAO broadcast arrived before both paired PUSHes were retained"
                    )
                actor_update, critic_update = await asyncio.gather(
                    self.actor.apply_staged_broadcast(fragment_id, step),
                    self.critic.apply_staged_broadcast(fragment_id, step),
                )
                actor_applied.extend(actor_update)
                critic_applied.extend(critic_update)
                if (
                    self.actor.fragment_versions[fragment_id] != step
                    or self.critic.fragment_versions[fragment_id] != step
                ):
                    raise RuntimeError("SAO paired apply committed mismatched versions")
                assert pair.actor_submission is not None
                assert pair.critic_submission is not None
                self._paired_journal.append(
                    (pair.actor_submission, pair.critic_submission)
                )
                self._release_pair(pair)
                self._inflight_pairs.pop(key)
            return (
                tuple(dict.fromkeys(actor_applied)),
                tuple(dict.fromkeys(critic_applied)),
            )
        except BaseException as error:
            self._fail_stop(error)
            raise

    async def _finish_paired_without_more_training(
        self,
        actor_result: RoleRoundResult,
        critic_result: RoleRoundResult,
    ) -> tuple[RoleRoundResult, RoleRoundResult]:
        """Reach both final manifests without manufacturing another train step."""

        deadline = time.monotonic() + min(
            self.config.actor.wait_timeout,
            self.config.critic.wait_timeout,
        )
        while not (self.actor.finalizing and self.critic.finalizing):
            actor_submissions, critic_submissions = await self._submit_paired_ready()
            (
                actor_applied,
                critic_applied,
            ) = await self._commit_staged_paired_broadcasts()
            actor_result = self._extend_result(
                actor_result,
                submissions=actor_submissions,
                applied=actor_applied,
            )
            critic_result = self._extend_result(
                critic_result,
                submissions=critic_submissions,
                applied=critic_applied,
            )
            if self.actor.finalizing and self.critic.finalizing:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "SAO paired terminal barrier could not reach both final cuts"
                )
            await asyncio.sleep(
                min(
                    self.config.actor.poll_seconds,
                    self.config.critic.poll_seconds,
                )
            )
        try:
            await asyncio.gather(
                self.actor.wait_and_apply_final(),
                self.critic.wait_and_apply_final(),
            )
            if self.actor.fragment_versions != self.critic.fragment_versions:
                raise RuntimeError("SAO actor and critic final manifests diverged")
            for key, pair in sorted(self._inflight_pairs.items()):
                if not pair.complete:
                    raise RuntimeError(
                        "SAO finalization retained an incomplete paired submission"
                    )
                assert pair.actor_submission is not None
                assert pair.critic_submission is not None
                self._paired_journal.append(
                    (pair.actor_submission, pair.critic_submission)
                )
                self._release_pair(pair)
                self._inflight_pairs.pop(key)
        except BaseException as error:
            self._fail_stop(error)
            raise
        return (
            RoleRoundResult(
                actor_result.role,
                actor_result.receipt,
                actor_result.applied_fragments,
                actor_result.submissions,
                bool(
                    getattr(
                        self.actor,
                        "terminal_submitted",
                        actor_result.terminal_submitted,
                    )
                ),
            ),
            RoleRoundResult(
                critic_result.role,
                critic_result.receipt,
                critic_result.applied_fragments,
                critic_result.submissions,
                bool(
                    getattr(
                        self.critic,
                        "terminal_submitted",
                        critic_result.terminal_submitted,
                    )
                ),
            ),
        )

    async def after_inference_publication(
        self,
        *,
        rollout_id: int | None,
        actor_model,
        publication_info=None,
    ) -> None:
        if actor_model is not self.actor_model:
            raise RuntimeError("SAO publication used the wrong actor group")
        if rollout_id is None:
            if (
                not self.initial_publication_pending
                or self.pending_publication is not None
            ):
                raise RuntimeError("unexpected initial SAO inference publication")
            await self._publish_identity(
                0,
                self.actor.content_hash,
                publication_info=publication_info,
            )
            self.published_actor_hash = self.actor.content_hash
            self.initial_publication_pending = False
            self._record_publication(terminal=False)
            return
        pending = self.pending_publication
        if pending is None or rollout_id + 1 != pending.rollout_id:
            raise RuntimeError("SAO inference publication has no matching trainer cut")
        await self._publish_identity(
            pending.rollout_id,
            pending.actor_content_hash,
            publication_info=publication_info,
        )
        self.published_rollout_id = pending.rollout_id
        self.published_actor_hash = pending.actor_content_hash
        if pending.terminal:
            failures = await asyncio.gather(
                self._acknowledge_role(self.actor),
                self._acknowledge_role(self.critic),
                return_exceptions=True,
            )
            errors = [value for value in failures if isinstance(value, BaseException)]
            self.finished = bool(
                not errors
                and self.actor.final_acknowledged
                and self.critic.final_acknowledged
            )
            if errors:
                failure = RuntimeError(
                    f"SAO terminal ACK publication failed for {len(errors)} role(s)"
                )
                for error in errors:
                    failure.add_note(f"{type(error).__name__}: {error}")
                raise failure from errors[0]
        self.pending_publication = None
        self._record_publication(terminal=pending.terminal)

    @staticmethod
    async def _acknowledge_role(stream: MilesFullParameterRoleStream) -> None:
        stream.acknowledge_final()

    async def _publish_identity(
        self,
        rollout_id: int,
        policy_hash: str,
        *,
        publication_info,
    ) -> None:
        token = f"yeto:{rollout_id}:{policy_hash}"
        if publication_info is None:
            raise RuntimeError(
                "SAO streaming requires exact publication worker evidence"
            )
        engines: Sequence[Any] = tuple(getattr(publication_info, "rollout_engines", ()))
        if not engines:
            raise RuntimeError(
                "SAO streaming publication evidence has no inference worker"
            )
        await asyncio.gather(
            *(engine.update_weight_version.remote(token) for engine in engines)
        )
        identity = await self.rollout_manager.set_external_policy_identity.remote(
            rollout_id,
            policy_hash,
        )
        if identity != (rollout_id, policy_hash):
            raise RuntimeError(
                "rollout process installed the wrong SAO policy identity"
            )
        from .miles import set_current_published_policy_identity

        set_current_published_policy_identity(
            self.args,
            policy_version=rollout_id,
            policy_hash=policy_hash,
        )
        self.args.yeto_rl_policy_token = token

    def _trajectory_evidence(self, rollout_id: int) -> TrajectoryBatchEvidence:
        directory = getattr(self.args, "yeto_rl_trajectory_evidence_dir", None)
        if type(directory) is not str:
            raise RuntimeError("SAO trajectory evidence directory is unavailable")
        path = trajectory_batch_evidence_path(directory, rollout_id)
        if not path.exists():
            raise RuntimeError("Miles did not persist this SAO rollout evidence")
        evidence = read_trajectory_batch_evidence(path)
        if evidence.rollout_id != rollout_id:
            raise RuntimeError("SAO trajectory evidence rollout identity changed")
        return evidence

    def _record_round(self, pending: _PendingPublication) -> None:
        self._append_event(
            {
                "event": "rl_sao_streaming_round",
                "rollout_id": pending.rollout_id - 1,
                "target_rollout_id": pending.rollout_id,
                "actor_policy_hash": pending.actor_content_hash,
                "actor_fragment_versions": list(pending.actor_fragment_versions),
                "critic_fragment_versions": list(pending.critic_fragment_versions),
                "actor_applied_fragments": list(pending.actor_result.applied_fragments),
                "critic_applied_fragments": list(
                    pending.critic_result.applied_fragments
                ),
                "actor_submitted_fragments": [
                    value.fragment_id for value in pending.actor_result.submissions
                ],
                "critic_submitted_fragments": [
                    value.fragment_id for value in pending.critic_result.submissions
                ],
                "terminal": pending.terminal,
            }
        )

    def _record_publication(self, *, terminal: bool) -> None:
        self._append_event(
            {
                "event": "rl_sao_streaming_publication",
                "rollout_id": self.published_rollout_id,
                "actor_policy_hash": self.published_actor_hash,
                "actor_fragment_versions": list(self.actor.fragment_versions),
                "critic_fragment_versions": list(self.critic.fragment_versions),
                "terminal": terminal,
            }
        )

    def _append_event(self, event: dict[str, Any]) -> None:
        from .miles import _append_rl_event

        _append_rl_event(self.args, event)

    async def finalize(self) -> None:
        if (
            not self.finished
            or self.initial_publication_pending
            or self.pending_publication is not None
            or not self.actor.final_acknowledged
            or not self.critic.final_acknowledged
        ):
            raise RuntimeError("Miles stopped before SAO streaming finalization")
        self.actor.close()
        self.critic.close()


def _validate_optimizer_apply_transition(
    before: tuple[Any, ...],
    after: tuple[Any, ...],
    *,
    role: str,
    base_policy_version: int,
    base_local_generation: int,
    target_policy_version: int,
) -> None:
    if not before or len(before) != len(after):
        raise RuntimeError("SAO global apply returned incomplete Adam evidence")
    volatile = {"installed_policy_version", "local_step_generation"}
    for old, new in zip(before, after, strict=True):
        old_values = vars(old)
        new_values = vars(new)
        if (
            set(old_values) != set(new_values)
            or old.role != role
            or new.role != role
            or old.installed_policy_version != base_policy_version
            or old.local_step_generation != base_local_generation
            or new.installed_policy_version != target_policy_version
            or new.local_step_generation != 0
            or any(
                old_values[name] != new_values[name]
                for name in old_values.keys() - volatile
            )
        ):
            raise RuntimeError("SAO global apply changed local Adam state")


def create_miles_sao_streaming_sync(args) -> MilesSaoStreamingPolicySync:
    config = getattr(args, "yeto_rl_sao_streaming_config", None)
    if not isinstance(config, MilesSaoStreamingConfig):
        raise TypeError("Miles SAO streaming configuration is missing")
    return MilesSaoStreamingPolicySync(args, config)
