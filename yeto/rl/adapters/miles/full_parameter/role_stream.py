"""One trainable role's full-parameter streaming DiLoCo session.

A role stream owns one syncer connection and one reference-backed, chunked
full-parameter cut of one Miles trainer group.  It stages authoritative
fragment broadcasts and answers syncer PULL permits with immutable submission
candidates.  It never decides *when* to submit or apply: the engine-neutral
``yeto.rl.engine.full_parameter_sync`` pairing rule does that for every role
of an algorithm (one role for GRPO, actor and critic for SAO/PPO).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from yeto.protocol import (
    DTYPE_F32,
    FinalManifest,
    PullRequest,
    SyncerClient,
)
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
    ReferencedPolicyCut,
    StoredAuthoritativeFragment,
)
from yeto.rl.adapters.miles.full_parameter.inbound_sink import MilesRayInboundChunkSink
from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, WeightsChanged
from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS, UnsupportedLayoutError
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.trajectory_evidence import TrajectoryBatchEvidence

_LOWER_HEX = frozenset("0123456789abcdef")


def _never_materialize(_policy_version: int):
    raise UnsupportedLayoutError(LAYOUT_FRAGMENTS)


def require_sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class RoleStreamConfig:
    """Closed configuration for one trainable role's streaming session."""

    algorithm: str
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
        if not self.algorithm.isidentifier():
            raise ValueError("role-stream algorithm name is invalid")
        if self.role not in {"actor", "critic"} or self.component.role != self.role:
            raise ValueError("role-stream component identity changed")
        host, port = self.syncer_addr
        if (
            not isinstance(host, str)
            or not host
            or type(port) is not int
            or not 0 < port < 65536
        ):
            raise ValueError("role-stream syncer address is invalid")
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
            raise ValueError("role-stream learner is outside the frozen roster")
        if (
            type(self.expected_fragments) is not int
            or self.expected_fragments < 1
            or type(self.total_fragment_steps) is not int
            or self.total_fragment_steps < self.expected_fragments
            or self.total_fragment_steps % self.expected_fragments
        ):
            raise ValueError("role-stream budget must contain complete sweeps")
        for name in (
            "local_horizon",
            "optimizer_steps_per_round",
            "pipeline_depth",
            "wan_streams",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"role-stream {name} must be positive")
        if self.wait_timeout <= 0 or not 0 < self.poll_seconds <= 1:
            raise ValueError("role-stream wait bounds are invalid")
        if (
            type(self.max_fragment_bytes) is not int
            or not 4 <= self.max_fragment_bytes <= 2 << 30
            or self.max_fragment_bytes % 4
            or type(self.max_chunk_bytes) is not int
            or not 4 <= self.max_chunk_bytes < 1 << 30
            or self.max_chunk_bytes % 4
        ):
            raise ValueError("role-stream transport bounds are invalid")
        require_sha256("expected_layout_hash", self.expected_layout_hash)
        require_sha256("training_contract_hash", self.training_contract_hash)
        require_sha256("syncer_profile_hash", self.syncer_profile_hash)

    @property
    def roster(self) -> dict[int, int]:
        return dict(enumerate(self.learner_generations))




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
class RoleSubmissionCandidate:
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


@dataclass(frozen=True)
class RoleRoundResult:
    role: str
    receipt: object
    applied_fragments: tuple[int, ...]
    submissions: tuple[RoleFragmentSubmission, ...]
    terminal_submitted: bool


def role_stream_session_contract_hash(
    layout_hash: str,
    config: RoleStreamConfig,
) -> bytes:
    """Bind a reconnecting role stream to one immutable experiment."""

    from yeto.rl.adapters.miles.identity import backend_identity

    require_sha256("layout_hash", layout_hash)
    payload = {
        "schema": "yeto-full-parameter-role-stream-v2",
        "algorithm": config.algorithm,
        "role": config.role,
        "component": asdict(config.component),
        "parameter_layout_hash": layout_hash,
        "training_contract_hash": config.training_contract_hash,
        "backend_identity_sha256": backend_identity().sha256(),
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
    return hashlib.sha256(b"yeto-role-stream-session-v2\0" + encoded).digest()


class MilesFullParameterRoleStream:
    """Reference-backed full-parameter transport for one trainable role."""

    def __init__(self, config: RoleStreamConfig, *, ray_module=None) -> None:
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
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} fragment cut is incomplete")
        return tuple(int(version) for version in self._fragment_versions)

    @property
    def content_hash(self) -> str:
        if self.current is None:
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream is not initialized")
        return self.current.content_hash

    def resident_state(self, policy_version: int) -> TrainerResidentState:
        """The current cut as a trainer-resident policy handle (no tensors).

        The policy tensor hash is the chunked cut identity, the payload hash its
        content identity; fragments never leave the trainer (P0-D1).
        """

        if self.current is None or self.adapter is None:
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream is not initialized")
        cut = self.current
        payload_bytes = sum(fragment.numel * 4 for fragment in self.adapter.layout.fragments.fragments)
        return TrainerResidentState(
            layout=LAYOUT_FRAGMENTS,
            base_model_revision=self.config.component.model_revision,
            config_hash=self.config.component.config_hash,
            layout_hash=cut.layout_hash,
            policy_version=int(policy_version),
            digest=PolicyDigest(cut.policy_hash, cut.content_hash, payload_bytes, ()),
            materialize_fn=_never_materialize,
            holds_fn=self._check_holds,
        )

    def _check_holds(self, state: TrainerResidentState) -> None:
        if self.current is None or self.current.policy_hash != state.policy_tensor_hash():
            raise WeightsChanged(f"{self.config.role} trainer no longer holds the published cut")

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
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream initialized twice")
        adapter, initial = await MilesChunkedFullParameterAdapter.initialize(
            group,
            policy_version=0,
            algorithm=self.config.algorithm,
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
            session_contract_hash=role_stream_session_contract_hash(
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
                            f"{self.config.algorithm} {self.config.role} initial fragment was not queued"
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
                        f"{self.config.algorithm} {self.config.role} initial broadcast changed"
                    )
                observed[stored.fragment_id] = stored
            if len(observed) < count:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{self.config.algorithm} {self.config.role} timed out waiting for initial cut"
                    )
                time.sleep(self.config.poll_seconds)
        return tuple(observed[index] for index in range(count))

    @staticmethod
    def _stored_update(update) -> StoredAuthoritativeFragment:
        if not getattr(update, "stored", False) or not isinstance(
            update.data, StoredAuthoritativeFragment
        ):
            raise RuntimeError(
                "role-stream streaming received an untyped full-parameter fragment"
            )
        if (
            update.data.fragment_id != update.fragment_id
            or update.data.version != update.version
        ):
            raise RuntimeError("role-stream streaming fragment envelope identity changed")
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
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream is not trainable")
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
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} client is unavailable")
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
                raise RuntimeError("role-stream staged fragment replay changed bytes")
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
                f"{self.config.algorithm} {self.config.role} matching staged fragment disappeared"
            )
        expected_base = max(0, version - self.config.expected_fragments)
        if self.fragment_versions[fragment_id] != expected_base:
            raise RuntimeError(
                f"{self.config.algorithm} {self.config.role} staged fragment skipped its base version"
            )
        await self._apply_authoritative({fragment_id: stored})
        self._staged.pop(key, None)
        return (fragment_id,)

    async def _apply_authoritative(
        self,
        selected: Mapping[int, StoredAuthoritativeFragment],
    ) -> None:
        if self.adapter is None or self.current is None or self.group is None:
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream is unavailable")
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
        validate_optimizer_apply_transition(
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
        return f"{self.config.algorithm}-{self.config.role}:{hashlib.sha256(payload).hexdigest()}"

    def ready_submission_candidates(self) -> tuple[RoleSubmissionCandidate, ...]:
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
                raise RuntimeError("role-stream token counter moved backwards")
            anchor = self._anchors[fragment_id]
            if anchor is None:
                raise RuntimeError(
                    "role-stream fragment has no authoritative anchor"
                )
            candidates.append(
                RoleSubmissionCandidate(
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

    def retain_candidate(self, candidate: RoleSubmissionCandidate) -> None:
        """Pin an exact local cut and authoritative anchor across boundaries."""

        if (
            candidate.role != self.config.role
            or self.adapter is None
            or candidate.submission_cut is not self.submission_cut
            or self._anchors[candidate.permit.fragment_id] is not candidate.anchor
        ):
            raise RuntimeError("role-stream paired submission candidate cannot be retained")
        self._pin(self._pinned_cuts, candidate.submission_cut)
        try:
            self._pin(self._pinned_anchors, candidate.anchor)
        except BaseException:
            self._unpin_cut(candidate.submission_cut)
            raise

    def release_candidate(self, candidate: RoleSubmissionCandidate) -> None:
        """Release a committed pair's exact operands when no pair still owns them."""

        self._unpin_cut(candidate.submission_cut)
        self._unpin_anchor(candidate.anchor)

    def retained_replay_candidate(
        self,
        candidate: RoleSubmissionCandidate,
    ) -> RoleSubmissionCandidate | None:
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
            raise RuntimeError("role-stream reconnect replay changed a retained candidate")
        return replace(candidate, permit=permit)

    def push_candidate(
        self,
        candidate: RoleSubmissionCandidate,
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
            raise RuntimeError("role-stream paired submission candidate changed")
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
                raise RuntimeError("role-stream received conflicting PULLs")
            self._pending.setdefault(permit.fragment_id, permit)

    def _validate_permit(self, permit: PullRequest) -> None:
        if (
            not 0 <= permit.fragment_id < self.config.expected_fragments
            or not 1 <= permit.global_step <= self.config.total_fragment_steps
            or (permit.global_step - 1) % self.config.expected_fragments
            != permit.fragment_id
            or permit.round_attempt < 1
        ):
            raise RuntimeError("role-stream received an invalid PULL")

    def _validate_fragment_version(self, fragment_id: int, version: int) -> None:
        if (
            not 0 <= fragment_id < self.config.expected_fragments
            or not 0 <= version <= self.config.total_fragment_steps
            or (
                version > 0
                and (version - 1) % self.config.expected_fragments != fragment_id
            )
        ):
            raise RuntimeError("role-stream received an invalid fragment version")

    async def wait_and_apply_final(self) -> None:
        if self.final_applied:
            return
        if self.client is None:
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} client is unavailable")
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
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} final cut is incomplete")
        selected = {}
        for fragment in fragments:
            stored = self._stored_update(fragment)
            if stored.version != manifest.versions[stored.fragment_id]:
                raise RuntimeError("role-stream final fragment differs from its manifest")
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
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} final ACK is out of order")
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
            raise RuntimeError("role-stream retained reference identity collided")
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
            raise RuntimeError("role-stream retained reference was released out of order")
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


def validate_optimizer_apply_transition(
    before: tuple[Any, ...],
    after: tuple[Any, ...],
    *,
    role: str,
    base_policy_version: int,
    base_local_generation: int,
    target_policy_version: int,
) -> None:
    """A global apply may only move the policy version and reset the local
    step generation; every other Adam field must be bit-identical."""
    if not before or len(before) != len(after):
        raise RuntimeError("global apply returned incomplete Adam evidence")
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
            raise RuntimeError("global apply changed local Adam state")
