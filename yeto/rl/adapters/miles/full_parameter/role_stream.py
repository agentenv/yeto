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
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from yeto.protocol import (
    DEFAULT_I4_BLOCK,
    PartialMessageGenerationLost,
    DEFAULT_I8_BLOCK,
    DTYPE_FLAG_BCAST_DELTA,
    DTYPE_FLAG_TENSOR_BLOCKS,
    FULL_PARAMETER_PUSH_DTYPES,
    I4_BLOCK_MAX,
    I4_BLOCK_MIN,
    I8_BLOCK_MAX,
    I8_BLOCK_MIN,
    FinalManifest,
    PullRequest,
    SyncerClient,
    timed_parts,
)
from yeto.push_error_feedback import RESIDUAL_DTYPES, PushErrorFeedback
from yeto.syncer_shards import ShardedSyncerClient
from yeto.rl.adapters.miles.full_parameter.inbound_sink import MilesRayInboundChunkSink
from yeto.rl.adapters.miles.full_parameter.peer_recovery import PeerRecoveryOptions
from yeto.rl.adapters.miles.full_parameter.rank_stats import report_rank_stats
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
    ReferencedPolicyCut,
    StoredAuthoritativeFragment,
    defer_capture_supported,
    lazy_capture_supported,
    StoredFragmentVerifier,
    optimizer_proof_without_model_check_supported,
    timed_group,
    verified_apply_supported,
)
from yeto.rl.host_info import print_host_report, report_rank_hosts
from yeto.rl.engine.policy_digest import (
    PolicyDigest,
    TrainerResidentState,
    WeightsChanged,
)
from yeto.rl.engine.trainable_state import LAYOUT_FRAGMENTS, UnsupportedLayoutError
from yeto.rl.adapters.miles.routing_plugin import (
    DRAIN_ROUTED_TOKEN_COUNTS,
    INSTALL_ROUTED_TOKEN_COUNTER,
)
from yeto.rl.local_learner import MATRIX_MERGES, ComponentIdentity, layout_stream_role
from yeto.rl.moe_layout import RoutedTokens, check_expert_partition, fragment_expert_tags
from yeto.rl.pool_client import LeaseHeartbeat
from yeto.rl.trajectory_evidence import TrajectoryBatchEvidence

# Seconds between progress lines while an island waits for its initial cut.
INITIAL_CUT_REPORT_SECONDS = 30.0
# The initial-cut wait grows with the model: long enough to move the cut at
# an assumed worst-case 32 MiB/s head link (X6: 116.5 GB of f32 timed out the
# flat 900 s default, which stays the floor).
INITIAL_CUT_MIN_BYTES_PER_SECOND = 32 << 20


def initial_cut_timeout(configured: float, model_bytes: int, islands: int, *, local_init: bool) -> float:
    """Seconds an island waits for its initial cut.

    Learner 0 uploads ``model_bytes`` of INIT. Without local init the syncer
    then broadcasts them to every island over the head's one link, so the cut
    moves ``(1 + islands) * model_bytes``; with local init only the upload.
    The configured timeout is the floor."""
    moved = model_bytes * (1 if local_init else 1 + islands)
    return max(configured, moved / INITIAL_CUT_MIN_BYTES_PER_SECOND)

_LOWER_HEX = frozenset("0123456789abcdef")


def _never_materialize(_policy_version: int):
    raise UnsupportedLayoutError(LAYOUT_FRAGMENTS)


TERMINAL_RETIRE_ENV = "YETO_FULL_PARAMETER_TERMINAL_RETIRE"

# RoleStreamConfig.final_cut
FINAL_CUT_MODES = ("download", "ack")

# The first push of a fragment from an anchor every fragment shares (stream
# start, peer recovery) waits ``fragment_id * H // P`` more local rounds, so
# the fragments that become pushable together at round H spread over H
# rounds (``first_push_phase``). "0" turns it off.
STAGGER_FIRST_PUSH_ENV = "YETO_FULL_PARAMETER_STAGGER_FIRST_PUSH"


def first_push_phase(fragment_id: int, fragments: int, horizon: int) -> int:
    """Extra local rounds before fragment ``fragment_id``'s first push.

    Every fragment is anchored at the same round when a stream starts, so with
    H > 1 all fragments with a PULL become pushable at round H together (the
    syncer opens its whole pipeline at once). Phase ``p * H // P`` in [0, H)
    staggers them by fragment index, the order of the syncer's rounds
    (fragment ``(t - 1) mod P`` at step t): about P / H fragments become
    pushable per round. Later pushes keep the H gate unchanged, so every
    fragment still syncs every H local rounds."""

    if horizon <= 1 or fragments <= 1:
        return 0
    return fragment_id * horizon // fragments


# Lazy capture: every N-th local round still captures with content hashes
# and the whole-model model/master check (0: never).
FULL_CHECK_EVERY_ENV = "YETO_FULL_PARAMETER_FULL_CHECK_EVERY"
DEFAULT_FULL_CHECK_EVERY = 16


def full_check_every() -> int:
    raw = os.environ.get(FULL_CHECK_EVERY_ENV, str(DEFAULT_FULL_CHECK_EVERY))
    try:
        every = int(raw)
    except ValueError as error:
        raise ValueError(f"{FULL_CHECK_EVERY_ENV} must be an integer, got {raw!r}") from error
    if every < 0:
        raise ValueError(f"{FULL_CHECK_EVERY_ENV} must be >= 0, got {every}")
    return every


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
    # P0-4.8: the island's AlgorithmSpec hash; placement capabilities stay out.
    algorithm_spec_sha256: str
    wan_streams: int = 4
    # S26 E3b transport options (yeto/wan_streams.py); not part of the session
    # contract: they change how bytes travel, never which bytes.
    wan_stream_schedule: str = "rr"
    wan_stall_timeout: float = 0.0  # seconds; 0 = off
    wait_timeout: float = 900.0
    poll_seconds: float = 0.01
    max_fragment_bytes: int = 2 << 30
    max_chunk_bytes: int = 256 << 20
    # S26 E3 wire options (default exact f32): push outer deltas as bf16
    # (session dtype 4), block-scaled int8 (dtype 5) or the q4 block codec
    # (dtype 6, int4); receive BCAST as bf16
    # deltas against a shared f32 anchor. FINAL always stays f32.
    push_dtype: str = "f32"
    push_int8_block: int = DEFAULT_I8_BLOCK
    # int4 pushes: values per absmax scale; stochastic rounding (seeded by
    # the learner id) instead of round-to-nearest.
    push_int4_block: int = DEFAULT_I4_BLOCK
    push_int4_stochastic: bool = False
    # int4 pushes: island-side error feedback, off or the residual dtype
    # (bf16 or f32); see yeto.push_error_feedback for its lifetime.
    push_int4_error_feedback: str = "off"
    # int8/int4/int4-log pushes: quantization blocks restart at every tensor
    # boundary of a fragment (session dtype flag), so a tensor replicated on
    # several ranks quantizes the same whatever its neighbours are. On by
    # default for those dtypes (ignored for f32/bf16); False keeps blocks
    # running across the fragment, the layout of sessions started before the
    # flag existed. A resumed run must repeat the choice it started with: the
    # syncer checkpoint binds the session contract, and a mismatch is refused.
    push_tensor_blocks: bool = True
    bcast_delta: bool = False
    # S26 E3: train round k+1 while sweep k syncs, then rebase the trainer
    # (see FullParameterSync._overlap_boundary); the local weights lag the
    # global policy by one outer step (yeto.rl.engine.sync_overlap).
    overlap: bool = False
    # S27 P2 asynchronous rounds: the syncer may close a fragment round
    # without this island (quorum K below the island count); a newer PULL then
    # supersedes an unanswered older one and the broadcast overwrites.
    partial_quorum: bool = False
    # Merge mode of every non-vocabulary fragment ("avg" or "rda"); the
    # vocabulary tables always average. Bound through the layout hash.
    matrix_merge: str = "avg"
    # Fragment-sharded syncer head: every shard's address in shard order
    # (the first is ``syncer_addr``); empty for one syncer process. Not part
    # of the session contract: shards run the unsharded step schedule, so
    # the shard count changes where bytes go, never which bytes.
    syncer_shards: tuple[tuple[str, int], ...] = ()
    # S27 E-3: the syncer incarnation this island was launched for (a newer
    # syncer refuses the HELLO), and peer recovery (donor snapshots, and
    # recovery of a fresh trainer). Transport options, not session contract.
    syncer_epoch: int | None = None
    peer_recovery: PeerRecoveryOptions | None = None
    # MoE: every push carries the tokens this island routed to each expert
    # during the push's outer interval; the syncer (--expert-token-weights)
    # weights that expert's tensors by them. Needs a layout with experts.
    expert_token_weights: bool = False
    # Island pool (--rl-sync-island-lease-s; asynchronous rounds only): the
    # syncer's lease on this island. The stream heartbeats every lease/3; a
    # lease expiry or a lost link redials (re-JOIN) and the syncer's catch-up
    # may skip fragment versions. None = fixed roster. Not part of the
    # session contract (membership is a syncer option).
    island_lease_s: float | None = None
    # Local init (S27): this island loaded the run's base checkpoint and keeps
    # it as version 0 once the syncer confirms every island's fragment digests
    # agree (MSG_INIT_CONFIRM); no initial broadcast is needed. False waits for
    # the broadcast of learner 0's INIT. Not part of the session contract: it
    # changes which bytes travel at startup, never the version-0 values.
    local_init: bool = False
    # What the island needs from the final cut. "download": every final f32
    # fragment, applied to the trainer, then the ACK (the island holds the
    # final parameters, e.g. to evaluate them). "ack": the final version
    # manifest only; the island ACKs once its pushes have drained and the
    # final parameters stay with the syncer (syncer --final-cut-payload
    # manifest). Must match the syncer, whose profile hash binds it.
    final_cut: str = "download"
    # S27 G-CG verify-ahead: check each stored broadcast's chunks on a
    # background thread when it arrives, so the apply at the next boundary
    # names it and the trainer's prepare reads none of its chunks (Miles
    # ``verified_apply``; the same check, off the training path). False keeps
    # the check in the trainer's prepare. Not part of the session contract:
    # the applied values are the same either way.
    verify_ahead: bool = True

    @property
    def tensor_blocks_active(self) -> bool:
        """Whether this session's block-quantized pushes restart blocks per tensor."""
        return self.push_tensor_blocks and self.push_dtype in ("int8", "int4", "int4-log")

    @property
    def session_dtype(self) -> int:
        base = FULL_PARAMETER_PUSH_DTYPES[self.push_dtype]
        return (
            base
            | (DTYPE_FLAG_BCAST_DELTA if self.bcast_delta else 0)
            | (DTYPE_FLAG_TENSOR_BLOCKS if self.tensor_blocks_active else 0)
        )

    @property
    def wire_dtype_label(self) -> str:
        """Session-contract label; plain f32 keeps the historical ``fp32``."""
        label = {"f32": "fp32", "bf16": "fp32-push-bf16",
                 "int8": f"fp32-push-int8-b{self.push_int8_block}",
                 "int4": f"fp32-push-int4-b{self.push_int4_block}"
                         + ("-sr" if self.push_int4_stochastic else "")
                         + ("" if self.push_int4_error_feedback == "off"
                            else f"-ef-{self.push_int4_error_feedback}"),
                 "int4-log": "fp32-push-q4"}[self.push_dtype]
        label += "-tb" if self.tensor_blocks_active else ""
        return label + ("+bcast-bf16-delta" if self.bcast_delta else "")

    def __post_init__(self) -> None:
        if not self.algorithm.isidentifier():
            raise ValueError("role-stream algorithm name is invalid")
        if self.final_cut not in FINAL_CUT_MODES:
            raise ValueError(f"role-stream final cut must be one of {FINAL_CUT_MODES}")
        if self.role not in {"actor", "critic"} or self.component.role != self.role:
            raise ValueError("role-stream component identity changed")
        for host, port in (self.syncer_addr, *self.syncer_shards):
            if (
                not isinstance(host, str)
                or not host
                or type(port) is not int
                or not 0 < port < 65536
            ):
                raise ValueError("role-stream syncer address is invalid")
        if self.syncer_shards:
            if len(self.syncer_shards) < 2 or self.syncer_shards[0] != self.syncer_addr:
                raise ValueError("role-stream syncer shards must list two or more "
                                 "addresses, the first being the syncer address")
            if len(set(self.syncer_shards)) != len(self.syncer_shards):
                raise ValueError("role-stream syncer shards must be distinct")
            if not self.partial_quorum:
                raise ValueError("a sharded syncer head needs asynchronous rounds "
                                 "(the strict sweep spans every fragment)")
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
        if self.wan_stream_schedule not in ("rr", "adaptive"):
            raise ValueError("role-stream wan stream schedule must be rr or adaptive")
        if not self.wan_stall_timeout >= 0:
            raise ValueError("role-stream wan stall timeout must be >= 0")
        if self.push_dtype not in FULL_PARAMETER_PUSH_DTYPES:
            raise ValueError("role-stream push dtype must be one of "
                             + ", ".join(FULL_PARAMETER_PUSH_DTYPES))
        if type(self.push_int8_block) is not int or not I8_BLOCK_MIN <= self.push_int8_block <= I8_BLOCK_MAX:
            raise ValueError("role-stream int8 block size is invalid")
        if self.matrix_merge not in MATRIX_MERGES:
            raise ValueError("role-stream matrix merge must be avg or rda")
        if self.overlap and self.partial_quorum:
            raise ValueError("sync/train overlap needs the strict sweep roster")
        if self.island_lease_s is not None and (
                not self.partial_quorum or not self.island_lease_s > 0):
            raise ValueError("role-stream island lease needs asynchronous rounds and a positive lease")
        if type(self.push_int4_block) is not int or (
            not I4_BLOCK_MIN <= self.push_int4_block <= I4_BLOCK_MAX or self.push_int4_block % 2
        ):
            raise ValueError("role-stream int4 block size is invalid")
        if self.push_int4_error_feedback not in ("off", *RESIDUAL_DTYPES):
            raise ValueError("role-stream int4 error feedback must be off, bf16 or f32")
        error_feedback = self.push_int4_error_feedback != "off"
        if (
            self.push_int4_block != DEFAULT_I4_BLOCK or self.push_int4_stochastic or error_feedback
        ) and self.push_dtype != "int4":
            raise ValueError("role-stream int4 options need push dtype int4")
        if error_feedback and self.push_int4_stochastic:
            raise ValueError("role-stream int4 error feedback needs nearest rounding")
        if self.syncer_epoch is not None and (type(self.syncer_epoch) is not int or self.syncer_epoch < 0):
            raise ValueError("role-stream syncer epoch must be a non-negative integer")
        if self.peer_recovery is not None and (not self.partial_quorum or self.bcast_delta):
            # A recovering island is off the roster until it is restored, and
            # the replay resends whole fragments (broadcasts overwrite).
            raise ValueError("peer recovery needs asynchronous rounds without broadcast deltas")
        if self.peer_recovery is not None and self.syncer_shards:
            raise ValueError("peer recovery needs one syncer process, not a sharded head")
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
        require_sha256("algorithm_spec_sha256", self.algorithm_spec_sha256)

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
class PreparedPush:
    """One candidate's wire bytes computed ahead (``prepare_push``)."""

    submission_cut: Any
    anchor: Any
    encoded: Any  # yeto.protocol.EncodedPush
    timings: dict[str, float]


@dataclass(frozen=True)
class SentPush:
    """What :meth:`MilesFullParameterRoleStream.send_prepared` did."""

    queued: bool
    link_generation: int
    timings: dict[str, float]
    finished_at: float  # time.monotonic() when the last chunk was queued


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
    # Expert token weights: per-tensor routed-token tags of this push.
    expert_tags: tuple[tuple[int, int, int], ...] | None = None
    # Lazy capture: the content descriptor of the submission row (whose own
    # descriptor is a label); the push reads the local chunks through it.
    local_descriptor: Any = None

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
        "schema": "yeto-full-parameter-role-stream-v3",
        "algorithm": config.algorithm,
        "role": config.role,
        "component": asdict(config.component),
        "parameter_layout_hash": layout_hash,
        "training_contract_hash": config.training_contract_hash,
        "backend_identity_sha256": backend_identity().sha256(),
        "algorithm_spec_sha256": config.algorithm_spec_sha256,
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
            "wire_dtype": config.wire_dtype_label,
            **({"sync_train_overlap": True} if config.overlap else {}),
            **({"partial_quorum": True} if config.partial_quorum else {}),
            **({"expert_token_weights": True} if config.expert_token_weights else {}),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"yeto-role-stream-session-v3\0" + encoded).digest()


class PoolFinishedBeforeJoin(RuntimeError):
    """A (re)started island reached the pool after its final cut."""


def _local_descriptor_kwargs(candidate: RoleSubmissionCandidate) -> dict[str, Any]:
    return {} if candidate.local_descriptor is None else {"local_descriptor": candidate.local_descriptor}


def _is_set(event) -> bool:
    return bool(event is not None and callable(getattr(event, "is_set", None)) and event.is_set())


class MilesFullParameterRoleStream:
    """Reference-backed full-parameter transport for one trainable role."""

    def __init__(self, config: RoleStreamConfig, *, ray_module=None) -> None:
        self.config = config
        self.ray_module = ray_module
        self.group = None
        self.adapter: MilesChunkedFullParameterAdapter | None = None
        self.client = None
        # S27 G-AG: summed push time split (_note_push_timing), read by the
        # session's round event.
        self.push_timing_totals: dict[str, float] = {}
        # Verify-ahead (S27 G-CG): replaced fragments an apply named as checked
        # off the trainer ("applied") or whose check was still running, so the
        # trainer checked them itself ("pending").
        self.verify_ahead_counters: dict[str, int] = {"applied": 0, "pending": 0}
        self._verifier: StoredFragmentVerifier | None = None
        self.heartbeat: LeaseHeartbeat | None = None
        self.sink = None
        self.current: ReferencedPolicyCut | None = None
        # Keep the latest post-train/pre-broadcast cut until the next local
        # round.  Independent role syncers can deliver the terminal PULL at
        # slightly different times; this immutable cut lets the terminal
        # barrier answer a delayed or replayed permit without another train
        # step, even if a partial broadcast has relabelled ``current`` as
        # generation zero in the meantime.
        self.submission_cut: ReferencedPolicyCut | None = None
        # Lazy capture: content descriptors of the submission cut's
        # materialized rows (its own rows carry labels).
        self._submission_descriptors: dict[int, Any] = {}
        self.submission_fragment_versions: tuple[int, ...] | None = None
        self.submission_rounds = 0
        self.submission_tokens = 0
        self.transport_version = 0
        # Terminal phase: fragments whose last round closed without a trainer
        # apply (``retire_staged_broadcast``); the FINAL apply covers them.
        self.retired_fragments: set[int] = set()
        # Decided once the trainer group is known (``_defers_capture``,
        # ``_proof_kwargs``).
        self._defer_capture: bool | None = None
        self._lazy_capture: bool | None = None
        self._full_check_every = full_check_every()
        # Whether the last local round's capture was lazy (round event).
        self.last_capture_lazy = False
        self._proof_without_model_check: bool | None = None
        self.local_generation = 0
        self.local_rounds = 0
        self.action_tokens = 0
        self._anchors: list[StoredAuthoritativeFragment | None] = []
        self._fragment_versions: list[int | None] = []
        self._rounds_at_anchor: list[int] = []
        # Extra local rounds before a fragment's first push from the shared
        # start anchor (first_push_phase); 0 once its anchor moved.
        self._first_push_phase: list[int] = []
        self._tokens_at_anchor: list[int] = []
        # Expert token weights: cumulative routed-token counts, kept like the
        # token counters above (empty when the option is off).
        self.action_routed = RoutedTokens()
        self.submission_routed = RoutedTokens()
        self._routed_at_anchor: list[RoutedTokens] = []
        self._pending: dict[int, PullRequest] = {}
        self._answered: dict[tuple[int, int], int] = {}
        # Per fragment: the generation of the link its PULLs arrive on.
        self._link_generations: dict[int, int] | None = None
        self._staged: dict[tuple[int, int], StoredAuthoritativeFragment] = {}
        self._pinned_cuts: dict[int, tuple[ReferencedPolicyCut, int]] = {}
        self._pinned_anchors: dict[int, tuple[StoredAuthoritativeFragment, int]] = {}
        # Overlap: the cut the current local round started from, and the
        # round/token counters at that point (the next push's c_steps base).
        self._round_base: ReferencedPolicyCut | None = None
        self._rounds_at_base = 0
        self._tokens_at_base = 0
        # S27 E-3 peer recovery: the donor service, and how this stream started.
        self._donor = None
        self.recovery_outcome = None
        self._routed_at_base = RoutedTokens()
        self.terminal_manifest: FinalManifest | None = None
        self.terminal_submitted = False
        # The final cut is reached (no more training or pushes); with
        # final_cut "download" the trainer then also holds it (final_applied).
        self.final_reached = False
        self.final_applied = False
        # int4 error feedback: per-fragment residuals, empty for every new
        # stream (fresh start or cut resume); never checkpointed.
        self.error_feedback = (
            None if config.push_int4_error_feedback == "off"
            else PushErrorFeedback(config.push_int4_block, config.push_int4_error_feedback)
        )
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
    def may_rejoin(self) -> bool:
        """An island pool with asynchronous rounds: a (re)started island may
        join a running pool, whose initial cut is the current one."""

        return self.config.island_lease_s is not None and self.config.partial_quorum

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
        # S27 G-AG diagnostics: host facts of the driver and of every trainer
        # rank, then one timed line per Miles group call.
        print_host_report(f"{self.config.role} driver learner {self.config.learner_id}")
        await report_rank_hosts(group, f"{self.config.role} learner {self.config.learner_id}")
        group = timed_group(group, self.config.role)
        adapter, initial = await MilesChunkedFullParameterAdapter.initialize(
            group,
            policy_version=0,
            algorithm=self.config.algorithm,
            stream_role=layout_stream_role(self.config.algorithm, self.config.role),
            components=(self.config.component,),
            minimum_fragments=self.config.expected_fragments,
            expected_fragments=self.config.expected_fragments,
            expected_layout_hash=self.config.expected_layout_hash,
            max_fragment_bytes=self.config.max_fragment_bytes,
            max_chunk_bytes=self.config.max_chunk_bytes,
            matrix_merge=self.config.matrix_merge,
        )
        if self.config.expert_token_weights:
            await self._install_routed_token_counter(group, adapter)
        client = self._new_client(
            adapter.layout.fragments,
            dtype=self.config.session_dtype,
            push_i8_block=self.config.push_int8_block,
            push_i4_block=self.config.push_int4_block,
            push_i4_stochastic_seed=self.config.learner_id if self.config.push_int4_stochastic else None,
            num_streams=self.config.wan_streams,
            stream_schedule=self.config.wan_stream_schedule,
            stream_stall_timeout=self.config.wan_stall_timeout or None,
            max_reconnects=None,
            session_contract_hash=role_stream_session_contract_hash(
                adapter.layout.layout_hash,
                self.config,
            ),
            syncer_profile_hash=bytes.fromhex(self.config.syncer_profile_hash),
            syncer_epoch=self.config.syncer_epoch,
            recover=self._recovering,
            expert_tokens=self.config.expert_token_weights,
        )
        heartbeat = (
            LeaseHeartbeat(client, self.config.island_lease_s,
                           progress=lambda: self.local_rounds,
                           name=f"{self.config.role}-heartbeat")
            if self.config.island_lease_s is not None else None
        )
        sink = adapter.fragment_sink(initial, ray_module=self.ray_module)
        if self.config.verify_ahead and verified_apply_supported(group):
            self._verifier = StoredFragmentVerifier(ray_module=self.ray_module)
            sink.verifier = self._verifier
        client.install_inbound_chunk_sink(MilesRayInboundChunkSink(sink))
        if self._recovering:
            await self._initialize_recovering(group, adapter, initial, client, sink, heartbeat)
            return
        local: tuple[StoredAuthoritativeFragment, ...] | None = None
        try:
            if self.config.local_init:
                local, digests = await asyncio.to_thread(
                    self._local_version_zero, adapter, initial, sink)
                client.set_init_digests(
                    hashlib.sha256(adapter.layout.layout_hash.encode()).digest(), digests)
            client.start()
            if heartbeat is not None:
                heartbeat.start()  # the initial cut may take longer than the lease
            init_parts = None
            if self.config.learner_id == 0:
                def init_parts(fragment_id: int):
                    return adapter.fragment_parts(
                        initial, fragment_id, ray_module=self.ray_module
                    )()

                # An outage here is not fatal: the wait below replays INIT on
                # the next connection group until the initial cut arrives.
                client.send_initial_fragments(init_parts)
            # Wait off the event loop: an island initializes its roles
            # concurrently, and each role's initial cut needs every island's
            # INIT for that role. A blocking wait here would stop this
            # island's other role from connecting, and two islands blocked
            # on different roles deadlock (s23-g3-sao-main-20261014a).
            authoritative = await asyncio.to_thread(
                self._wait_for_initial_fragments, client, adapter, init_parts, local
            )
            adapter.verify_initial_fragments(initial, authoritative, allow_newer=self.may_rejoin)
            if local is not None:  # a rejoin's newer fragments replace version 0
                kept = {id(stored) for stored in authoritative}
                for stored in local:
                    if id(stored) not in kept:
                        stored.release()
        except BaseException:
            if heartbeat is not None:
                heartbeat.stop()
            client.close()
            sink.release_all()
            adapter.release(initial)
            raise
        count = adapter.layout.fragments.num_fragments
        self.group = group
        self.adapter = adapter
        self.client = client
        self.heartbeat = heartbeat
        self.sink = sink
        self.current = initial
        self._anchors = list(authoritative)
        self._fragment_versions = [0] * count
        self._rounds_at_anchor = [0] * count
        self._tokens_at_anchor = [0] * count
        self._routed_at_anchor = [RoutedTokens()] * count
        newer = {stored.fragment_id: stored for stored in authoritative if stored.version > 0}
        if newer:
            # A restarted island joined a running pool: the syncer resent
            # every fragment at its current version. Those values replace the
            # checkpoint weights and become this island's anchors, so its
            # first push is measured from the pool's current cut.
            # Before the first local step Megatron has no optimizer-state
            # proof (no rollout trained yet), so none is compared.
            await self._apply_authoritative(newer, optimizer_proof=False)
            print(
                f"[yeto-initial-cut] learner {self.config.learner_id} caught up to a running "
                f"pool: {len(newer)}/{count} fragment(s) newer than version 0, "
                f"versions {list(self.fragment_versions)}",
                file=sys.stderr,
                flush=True,
            )
        self._first_push_phase = self._start_phases(count)

    def _first_push_extra(self, fragment_id: int) -> int:
        """Local rounds a fragment waits beyond H (no phases: none)."""
        return self._first_push_phase[fragment_id] if self._first_push_phase else 0

    def _clear_first_push_phase(self, fragment_id: int) -> None:
        # Empty while the stream initializes (the start phases come after).
        if self._first_push_phase:
            self._first_push_phase[fragment_id] = 0

    def _start_phases(self, count: int) -> list[int]:
        """First-push phases of a stream whose fragments all share one anchor
        round (asynchronous rounds with H > 1; see first_push_phase)."""

        if not self.config.partial_quorum or os.environ.get(STAGGER_FIRST_PUSH_ENV, "1") == "0":
            return [0] * count
        return [first_push_phase(fragment_id, count, self.config.local_horizon) for fragment_id in range(count)]

    async def _install_routed_token_counter(self, group, adapter) -> None:
        if check_expert_partition(adapter.layout.specs) is None:
            raise ValueError(
                f"{self.config.role}: expert token weights need a mixture-of-experts layout"
            )
        hooked = await group.run_plugin(INSTALL_ROUTED_TOKEN_COUNTER)
        if not any(hooked):
            raise RuntimeError(f"{self.config.role}: no rank holds a top-k MoE router to count")

    async def _drain_routed_tokens(self) -> RoutedTokens:
        """Tokens every rank routed since the previous drain, summed."""

        total = RoutedTokens()
        for per_rank in await self.group.run_plugin(DRAIN_ROUTED_TOKEN_COUNTS):
            total = total + RoutedTokens.from_mapping(
                {int(layer): row for layer, row in per_rank.items()}
            )
        return total

    def _new_client(self, layout, **kwargs):
        """One ``SyncerClient``, or one per shard of a sharded head."""
        if self.config.syncer_shards:
            if len(self.config.syncer_shards) > layout.num_fragments:
                raise ValueError(
                    f"{len(self.config.syncer_shards)} syncer shards need at least as many "
                    f"fragments, layout has {layout.num_fragments}")
            return ShardedSyncerClient(
                list(self.config.syncer_shards), self.config.learner_id, layout, **kwargs)
        return SyncerClient(self.config.syncer_addr, self.config.learner_id, layout, **kwargs)

    @property
    def _recovering(self) -> bool:
        options = self.config.peer_recovery
        return options is not None and options.recover

    def _snapshot_store(self):
        from yeto.rl.peer_recovery import snapshot_store_for

        options = self.config.peer_recovery
        return snapshot_store_for(options.store, staging=options.staging)

    def _recovery_island(self, group, adapter, *, recovering: bool):
        from yeto.rl.adapters.miles.full_parameter.peer_recovery import (
            MilesPeerRecoveryIsland,
            MilesRankShards,
        )

        shards = MilesRankShards(group, adapter.layout.layout_hash)
        return MilesPeerRecoveryIsland(self, shards, asyncio.get_running_loop(), recovering=recovering)

    async def _initialize_recovering(self, group, adapter, initial, client, sink, heartbeat) -> None:
        """S27 E-3: this trainer is fresh after the island lost its state.

        No INIT is sent (the syncer holds the run's fragments). A donor's
        snapshot restores the inner optimizer state on every rank, then the
        syncer resends every fragment and they become this island's anchors
        in one apply (``peer_recovery`` module doc)."""
        from yeto.rl.peer_recovery import recover

        options = self.config.peer_recovery
        island = self._recovery_island(group, adapter, recovering=True)
        count = adapter.layout.fragments.num_fragments
        try:
            client.start()
            if heartbeat is not None:
                heartbeat.start()  # a peer restore may take longer than the lease
            outcome = await asyncio.to_thread(
                recover, client, island, self._snapshot_store(), timeout=options.timeout_s)
            client.mark_initial_cut_received()
            if sorted(island.resent) != list(range(count)):
                raise RuntimeError(f"{self.config.role} recovery did not receive every fragment")
            self.group = group
            self.adapter = adapter
            self.client = client
            self.heartbeat = heartbeat
            self.sink = sink
            self.current = initial
            self._anchors = [None] * count
            self._fragment_versions = [0] * count
            self._rounds_at_anchor = [0] * count
            self._tokens_at_anchor = [0] * count
            self._routed_at_anchor = [RoutedTokens()] * count
            resent, island.resent = dict(island.resent), {}
            await self._apply_authoritative(resent, optimizer_proof=False)
            self._first_push_phase = self._start_phases(count)
        except BaseException:
            if heartbeat is not None:
                heartbeat.stop()
            island.release()
            client.close()
            sink.release_all()
            adapter.release(initial)
            self.group = self.adapter = self.client = self.sink = self.current = None
            raise
        self.recovery_outcome = outcome
        print(f"[yeto-peer-recovery] {self.config.role} learner {self.config.learner_id}: "
              f"{outcome.path} recovery (donor {outcome.donor}) in {outcome.steps} syncer step(s), "
              f"{outcome.seconds:.1f}s; fragment versions {list(self.fragment_versions)}",
              file=sys.stderr, flush=True)

    async def serve_peer_snapshots(self) -> int:
        """Donor side, at a safe point (no optimizer step running): answer the
        syncer's snapshot requests this island is caught up for, and sweep
        expired snapshots. Returns how many were served."""
        if self.config.peer_recovery is None or self.client is None:
            return 0
        if self._donor is None:
            from yeto.rl.peer_recovery import DonorService

            island = self._recovery_island(self.group, self.adapter, recovering=False)
            # A file island (rank shards with the FP32 main copy and moments, hundreds
            # of GB for a 30B model): the service hashes and commits in the background
            # and logs each refused or finished snapshot.
            self._donor = DonorService(self.client, island, self._snapshot_store(),
                                       ttl_s=self.config.peer_recovery.ttl_s,
                                       log=lambda text: print(text, file=sys.stderr, flush=True),
                                       name=self.config.role)
        return await asyncio.to_thread(self._donor.poll)

    def _local_version_zero(self, adapter, initial, sink):
        """Local init: this island's own version-0 cut is its version-0
        anchors (shared, not copied), hashed exactly as the syncer hashes
        INIT_PARAMS."""
        stored, digests = [], []
        for fragment_id in range(adapter.layout.fragments.num_fragments):
            digest = hashlib.sha256()
            for part in adapter.fragment_parts(initial, fragment_id, ray_module=self.ray_module)():
                digest.update(part)
            digests.append(digest.digest())
            stored.append(sink.adopt(initial, fragment_id, 0, digest.hexdigest()))
        return tuple(stored), digests

    def _wait_for_initial_fragments(
        self,
        client,
        adapter: MilesChunkedFullParameterAdapter,
        init_parts=None,
        local: Sequence[StoredAuthoritativeFragment] | None = None,
    ) -> tuple[StoredAuthoritativeFragment, ...]:
        """The version-0 cut: the syncer's broadcast, or with ``local`` this
        island's own fragments once the syncer confirms the digests agree."""
        started = time.monotonic()
        count = adapter.layout.fragments.num_fragments
        fragments = getattr(adapter.layout.fragments, "fragments", ())
        model_bytes = sum(4 * fragment.numel for fragment in fragments)
        deadline = started + initial_cut_timeout(
            self.config.wait_timeout, model_bytes, len(self.config.learner_generations),
            local_init=local is not None)
        next_report = started + INITIAL_CUT_REPORT_SECONDS
        observed: dict[int, StoredAuthoritativeFragment] = {}
        self._report_initial_cut(client, observed, count, started, "waiting")
        while len(observed) < count:
            client.check_health()
            if local is not None and client.initial_cut_confirmed():
                for stored in observed.values():
                    if all(stored is not mine for mine in local):
                        stored.release()
                self._report_initial_cut(client, observed, count, started, "confirmed local")
                client.mark_initial_cut_received()
                return tuple(local)
            if self.may_rejoin and _is_set(getattr(client, "finalizing", None)):
                # The pool reached its final cut before this island caught up;
                # waiting for broadcasts would only run into the timeout.
                self._report_initial_cut(client, observed, count, started, "pool finished")
                raise PoolFinishedBeforeJoin(
                    f"{self.config.algorithm} {self.config.role}: the island pool reached "
                    "its final cut before this island caught up"
                )
            if init_parts is not None and client.init_replay_due():
                # S26 E3b (s26-e3b-p06-C32a): the group carrying INIT died
                # before the syncer applied it; send it again on this group.
                self._report_initial_cut(client, observed, count, started, "replaying INIT")
                client.send_initial_fragments(init_parts)
            for update in client.drain_updates():
                stored = self._stored_update(update)
                previous = observed.get(stored.fragment_id)
                if self.may_rejoin:
                    # A running pool sends its catch-up (every fragment at its
                    # current version) and may broadcast newer rounds before
                    # the catch-up completes: keep the newest version.
                    if previous is not None and stored.version <= previous.version:
                        if (
                            stored.version == previous.version
                            and stored.wire_payload_hash != previous.wire_payload_hash
                        ):
                            raise RuntimeError(
                                f"{self.config.algorithm} {self.config.role} catch-up fragment "
                                f"{stored.fragment_id} version {stored.version} changed bytes"
                            )
                        stored.release()
                        continue
                    if previous is not None:
                        previous.release()
                elif update.version != 0 or previous is not None:
                    raise RuntimeError(
                        f"{self.config.algorithm} {self.config.role} initial broadcast changed"
                    )
                observed[stored.fragment_id] = stored
            if len(observed) < count:
                now = time.monotonic()
                if now >= next_report:
                    self._report_initial_cut(client, observed, count, started, "waiting")
                    next_report = now + INITIAL_CUT_REPORT_SECONDS
                if now >= deadline:
                    self._report_initial_cut(client, observed, count, started, "timed out")
                    raise TimeoutError(
                        f"{self.config.algorithm} {self.config.role} timed out waiting for initial cut"
                    )
                time.sleep(self.config.poll_seconds)
        self._report_initial_cut(client, observed, count, started, "received")
        client.mark_initial_cut_received()
        return tuple(observed[index] for index in range(count))

    def _report_initial_cut(
        self,
        client,
        observed: Mapping[int, StoredAuthoritativeFragment],
        count: int,
        started: float,
        state: str,
    ) -> None:
        """One stderr line on the initial cut: how much of INIT_PARAMS left
        this island and how much of the broadcast came back."""

        progress_of = getattr(client, "send_progress", None)
        progress = progress_of() if callable(progress_of) else None
        sent = (
            ""
            if progress is None
            else (
                f"; sent {progress.bytes_sent / 2**20:.1f} MiB, "
                f"{progress.queued_chunks} chunk(s) queued, "
                f"{progress.reconnects} redial(s), "
                f"connected={progress.connected}"
            )
        )
        print(
            f"[yeto-initial-cut] learner {self.config.learner_id} {state}: "
            f"{len(observed)}/{count} fragment(s) after "
            f"{time.monotonic() - started:.0f}s{sent}",
            file=sys.stderr,
            flush=True,
        )

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
        hold_submission: bool = False,
    ) -> RoleRoundResult:
        """Record one trained round and capture its cut.

        ``hold_submission`` (overlap): the previous sweep is still in flight,
        so the new cut is not yet submittable; it becomes the submission only
        after :meth:`apply_staged_broadcast` rebased it and
        :meth:`finish_rebase`. The round base stays pinned until then."""
        if (
            self.adapter is None
            or self.current is None
            or self.group is None
            or self.client is None
            or self.final_reached
        ):
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream is not trainable")
        expected_generation = self.local_generation + 1
        lazy = self._captures_lazily(self.local_rounds + 1)
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
            **({"check_model": False} if lazy else {}),
        )
        local = await self.adapter.capture(
            self.group,
            policy_version=self.transport_version,
            local_step_generation=expected_generation,
            defer_payload=self._defers_capture(),
            **({"lazy": True} if lazy else {}),
        )
        self.last_capture_lazy = lazy
        await report_rank_stats(self.group, "export", self.config.role)
        previous = self.current
        previous_submission = self.submission_cut
        self.current = local
        self.local_generation = expected_generation
        self.local_rounds += 1
        self.action_tokens += trajectories.trained_tokens
        if self.config.expert_token_weights:
            self.action_routed = self.action_routed + await self._drain_routed_tokens()
        if hold_submission:
            if not self.config.overlap or self._round_base is None:
                raise RuntimeError(f"{self.config.role} holds a submission without an overlap round base")
            self._release_cuts(previous, keep=(local, self._round_base, previous_submission))
        else:
            self.submission_cut = local
            self._submission_descriptors = {}
            self.submission_fragment_versions = self.fragment_versions
            self.submission_rounds = self.local_rounds
            self.submission_tokens = self.action_tokens
            self.submission_routed = self.action_routed
            self._release_cuts(previous, previous_submission, keep=(local, self._round_base))

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

    def _defers_capture(self) -> bool:
        """S27 G-AA deferred capture: a local round's capture hashes every
        fragment but stores none in Ray; :meth:`materialize_submission` stores
        the fragments a push needs.  Only for asynchronous rounds without
        overlap (every reader of the cut bytes is then a push of a
        materialized fragment or a skip-resident apply that never reads a
        deferred row) and only when the trainer's Miles supports it."""

        if self._defer_capture is None:
            self._defer_capture = bool(
                self.config.partial_quorum
                and not self.config.overlap
                and self.group is not None
                and defer_capture_supported(self.group)
            )
        return self._defer_capture

    def _captures_lazily(self, local_round: int) -> bool:
        """Lazy capture of local round ``local_round`` (1-based): only with a
        deferred capture whose Miles supports it, and not on every
        ``YETO_FULL_PARAMETER_FULL_CHECK_EVERY``-th round, which still hashes
        every fragment and compares the whole model with its masters."""

        if self._lazy_capture is None:
            self._lazy_capture = bool(
                self._defers_capture() and self.group is not None and lazy_capture_supported(self.group))
        if not self._lazy_capture:
            return False
        every = self._full_check_every
        return not (every and local_round % every == 0)

    async def materialize_submission(self, fragment_ids: Sequence[int]) -> tuple[int, ...]:
        """Store the deferred fragments ``fragment_ids`` of the submission cut.

        Called by the pairing before it pushes them, while the trainer still
        holds their bytes (no local step since the capture, and an apply
        never writes a fragment this island still has to push from this
        cut: such a fragment's version would move past the submission's).
        With lazy capture Miles also returns each fragment's content
        descriptor; its push reads the stored chunks through it."""

        if self.adapter is None or self.group is None or self.submission_cut is None:
            return ()
        describe = bool(self._lazy_capture)
        stored = await self.adapter.materialize(
            self.group, self.submission_cut, fragment_ids, **({"describe": True} if describe else {}))
        if describe:
            self._submission_descriptors.update(stored)
        if stored:
            await report_rank_stats(self.group, "materialize", self.config.role)
        return tuple(stored)

    def _proof_kwargs(self) -> dict[str, bool]:
        """The optimizer proofs around an apply skip their whole-model
        model/master comparison when Miles supports it: the commit itself
        copies every master into the model and ends with that comparison."""

        if self._proof_without_model_check is None:
            self._proof_without_model_check = bool(
                self.group is not None and optimizer_proof_without_model_check_supported(self.group))
        return {"check_model": False} if self._proof_without_model_check else {}

    async def stage_available_broadcasts(self) -> tuple[tuple[int, int], ...]:
        """Drain complete broadcasts without mutating the trainable model."""

        return self.stage_available_broadcasts_now()

    def stage_available_broadcasts_now(self) -> tuple[tuple[int, int], ...]:
        """:meth:`stage_available_broadcasts` without the event loop (the
        pairing's timeout check stages arrived broadcasts before it judges)."""

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
    def monotonic_broadcasts(self) -> bool:
        """The syncer client drops a broadcast older than one it already
        delivered for that fragment, so a staged version jump is final."""

        return True

    @property
    def staged_fragment_versions(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._staged))

    def has_staged_broadcast(self, fragment_id: int, version: int) -> bool:
        return (fragment_id, version) in self._staged

    async def apply_staged_broadcast(
        self,
        fragment_id: int,
        version: int,
        *,
        rebase: bool = False,
        catch_up: bool = False,
    ) -> tuple[int, ...]:
        """Apply one coordinator-approved staged fragment at a safe boundary.

        ``rebase`` (overlap): the trainer keeps its progress since the round
        base, ``theta_f <- G_f + (current_f - base_f)``; ``G_f`` becomes the
        anchor, and the counters at the round base become the anchor's, so
        the next push carries exactly that progress.

        ``catch_up`` (partial quorum): the fragment jumps over rounds this
        island never received (a re-JOIN catch-up); the global value
        overwrites it like any closed round."""

        stored = self._staged_for_apply(fragment_id, version, rebase=rebase, catch_up=catch_up)
        if not rebase:
            await self._apply_authoritative({fragment_id: stored})
        else:
            if not self.config.overlap or self._round_base is None or self.current is None:
                raise RuntimeError(f"{self.config.role} rebase without an overlap round base")
            rebased = self.adapter.rebase_fragment(
                stored, self.current, self._round_base, ray_module=self.ray_module)
            try:
                await self._apply_authoritative(
                    {fragment_id: rebased},
                    anchors={fragment_id: stored},
                    counters=(self._rounds_at_base, self._tokens_at_base, self._routed_at_base),
                )
            finally:
                rebased.release()  # the applied target holds its own references
        self._close_staged_round(fragment_id, version)
        return (fragment_id,)

    async def apply_staged_broadcasts(
        self, items: Sequence[tuple[int, int, bool]]
    ) -> tuple[int, ...]:
        """Apply several staged broadcasts ``(fragment_id, version, catch_up)``
        of distinct fragments in one trainer apply (partial quorum).

        Same checks and bookkeeping as one :meth:`apply_staged_broadcast` per
        item; the fragments occupy disjoint master ranges, so one apply of all
        of them writes the same bytes as one apply each, and the whole-model
        costs of an apply (optimizer proofs, master-to-model copy, checks) are
        paid once."""

        if self.config.overlap:
            raise RuntimeError(f"{self.config.role} batches applies only without overlap")
        fragment_ids = [fragment_id for fragment_id, _version, _catch_up in items]
        if not items or len(set(fragment_ids)) != len(fragment_ids):
            raise RuntimeError(f"{self.config.role} batched apply needs distinct fragments")
        selected = {
            fragment_id: self._staged_for_apply(fragment_id, version, rebase=False, catch_up=catch_up)
            for fragment_id, version, catch_up in items
        }
        await self._apply_authoritative(selected)
        for fragment_id, version, _catch_up in items:
            self._close_staged_round(fragment_id, version)
        return tuple(fragment_ids)

    def _staged_for_apply(
        self, fragment_id: int, version: int, *, rebase: bool, catch_up: bool
    ) -> StoredAuthoritativeFragment:
        """The staged broadcast ``(fragment_id, version)`` once its base checks pass."""

        stored = self._staged.get((fragment_id, version))
        if stored is None:
            raise RuntimeError(
                f"{self.config.algorithm} {self.config.role} matching staged fragment disappeared"
            )
        expected_base = max(0, version - self.config.expected_fragments)
        held = self.fragment_versions[fragment_id]
        if catch_up:
            if not self.config.partial_quorum or rebase or held >= expected_base:
                raise RuntimeError(
                    f"{self.config.algorithm} {self.config.role} catch-up needs a partial quorum "
                    "and a version past the next round"
                )
        elif held != expected_base:
            raise RuntimeError(
                f"{self.config.algorithm} {self.config.role} staged fragment skipped its base version"
            )
        return stored

    def _close_staged_round(self, fragment_id: int, version: int) -> None:
        self._staged.pop((fragment_id, version), None)
        for older in [k for k in self._staged if k[0] == fragment_id and k[1] < version]:
            self._staged.pop(older).release()
        if self.config.partial_quorum:
            # The round closed: an unanswered PULL of it can never commit.
            permit = self._pending.get(fragment_id)
            if permit is not None and permit.global_step <= version:
                self._pending.pop(fragment_id)

    @property
    def retires_terminal_broadcasts(self) -> bool:
        """Whether :meth:`retire_staged_broadcast` may skip a terminal apply.

        Only for a partial quorum without overlap and without peer recovery
        (a donor would serve trainer state behind its fragment versions);
        ``YETO_FULL_PARAMETER_TERMINAL_RETIRE=0`` keeps every apply."""

        return (
            self.config.partial_quorum
            and not self.config.overlap
            and self.config.peer_recovery is None
            and os.environ.get(TERMINAL_RETIRE_ENV, "1") != "0"
        )

    async def retire_staged_broadcast(
        self,
        fragment_id: int,
        version: int,
        *,
        catch_up: bool = False,
    ) -> tuple[int, ...]:
        """Terminal phase (partial quorum): close a fragment's last round
        without writing it into the trainer.

        The pairing calls this only after the island stopped training and
        only for a round with no later round of that fragment: nothing reads
        the trainer's value of the fragment again before the FINAL cut, and
        the FINAL apply overwrites every fragment.  The broadcast becomes the
        fragment's anchor and version exactly as an apply would record them,
        so pushes, closed-round checks and the FINAL manifest see the same
        state; only the trainer apply (whole-model costs per fragment) is
        left out.  The final parameters are the FINAL bytes either way."""

        if not self.retires_terminal_broadcasts:
            raise RuntimeError(f"{self.config.role} cannot retire a broadcast without applying it")
        if version < self.config.total_fragment_steps - self.config.expected_fragments + 1:
            raise RuntimeError(f"{self.config.role} retires only a fragment's last round")
        stored = self._staged_for_apply(fragment_id, version, rebase=False, catch_up=catch_up)
        previous_anchor = self._anchors[fragment_id]
        if previous_anchor is not None and previous_anchor is not stored:
            self._release_anchor(previous_anchor)
        self._anchors[fragment_id] = stored
        self._fragment_versions[fragment_id] = stored.version
        self._rounds_at_anchor[fragment_id] = self.local_rounds
        self._clear_first_push_phase(fragment_id)
        self._tokens_at_anchor[fragment_id] = self.action_tokens
        self._routed_at_anchor[fragment_id] = self.action_routed
        self.retired_fragments.add(fragment_id)
        self._close_staged_round(fragment_id, version)
        return (fragment_id,)

    def finish_rebase(self) -> None:
        """Overlap: the rebased cut is the next submission (``G + progress``).

        The applied cut has generation 0; the submission is the same weights
        labeled with the rounds trained since the round base."""

        if not self.config.overlap or self.current is None or self.adapter is None:
            raise RuntimeError(f"{self.config.role} finish_rebase outside overlap")
        previous_submission = self.submission_cut
        self.submission_cut = self.adapter.relabel_local_progress(
            self.current, local_rounds=self.local_rounds - self._rounds_at_base)
        self._submission_descriptors = {}
        self.submission_fragment_versions = self.fragment_versions
        self.submission_rounds = self.local_rounds
        self.submission_tokens = self.action_tokens
        self.submission_routed = self.action_routed
        self._release_cuts(previous_submission, keep=(self.current, self.submission_cut, self._round_base))

    def mark_round_base(self) -> None:
        """Overlap: pin the cut the next local round starts from."""

        if not self.config.overlap or self.current is None:
            raise RuntimeError(f"{self.config.role} mark_round_base outside overlap")
        previous = self._round_base
        self._round_base = self.current
        self._rounds_at_base = self.local_rounds
        self._tokens_at_base = self.action_tokens
        self._routed_at_base = self.action_routed
        self._release_cuts(previous, keep=(self.current, self.submission_cut))

    async def _apply_authoritative(
        self,
        selected: Mapping[int, StoredAuthoritativeFragment],
        *,
        anchors: Mapping[int, StoredAuthoritativeFragment] | None = None,
        counters: tuple[int, int, RoutedTokens] | None = None,
        optimizer_proof: bool = True,
    ) -> None:
        """Apply ``selected`` fragments; ``anchors``/``counters`` override the
        recorded anchor and its round/token counters (overlap rebase).

        ``optimizer_proof=False``: the trainer has not trained yet (an initial
        catch-up, or the first apply after a peer snapshot restore), so Megatron
        has no optimizer-state proof to compare: it records the last rollout ID
        only at a local step. A restore is checked against the donor's snapshot
        digest instead."""
        if self.adapter is None or self.current is None or self.group is None:
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} stream is unavailable")
        target_version = self.transport_version + 1
        target = self.adapter.assemble_mixed_target(
            self.current,
            target_policy_version=target_version,
            authoritative_fragments=tuple(selected.values()),
        )
        verified = self._verified_ahead(selected)
        before = (tuple(await self.group.full_parameter_optimizer_states(**self._proof_kwargs()))
                  if optimizer_proof else None)
        await self.adapter.apply(
            self.group,
            target,
            commit_token=self._apply_token(target_version, selected),
            **({"verified_fragments": verified} if verified else {}),
        )
        await report_rank_stats(self.group, "apply", self.config.role)
        if before is not None:
            after = tuple(await self.group.full_parameter_optimizer_states(**self._proof_kwargs()))
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
            keep=(self.current, self.submission_cut, self._round_base),
        )
        rounds, tokens, routed = (
            counters if counters is not None
            else (self.local_rounds, self.action_tokens, self.action_routed)
        )
        for fragment_id, applied in selected.items():
            stored = applied if anchors is None else anchors[fragment_id]
            previous_anchor = self._anchors[fragment_id]
            if previous_anchor is not None and previous_anchor is not stored:
                self._release_anchor(previous_anchor)
            self._anchors[fragment_id] = stored
            self._fragment_versions[fragment_id] = stored.version
            self._rounds_at_anchor[fragment_id] = rounds
            self._clear_first_push_phase(fragment_id)
            self._tokens_at_anchor[fragment_id] = tokens
            self._routed_at_anchor[fragment_id] = routed

    def _verified_ahead(self, selected: Mapping[int, StoredAuthoritativeFragment]) -> dict[int, str]:
        """``{fragment: payload hash}`` of the fragments in ``selected`` whose
        stored chunks the verifier already checked; a fragment whose check is
        still running is checked by the trainer as before.  A failed check
        fails this apply before the trainer changes, as its prepare would."""

        verified: dict[int, str] = {}
        for fragment_id, stored in selected.items():
            try:
                payload_hash = stored.verified_payload_hash()
            except Exception as error:
                raise RuntimeError(
                    f"{self.config.role} broadcast of fragment {fragment_id} v{stored.version} "
                    "failed its off-trainer chunk check"
                ) from error
            if payload_hash is not None:
                verified[fragment_id] = payload_hash
            elif stored.verification is not None:
                self.verify_ahead_counters["pending"] += 1
        self.verify_ahead_counters["applied"] += len(verified)
        return verified

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

    def ready_submission_candidates(
        self, *, include_deferred: bool = False
    ) -> tuple[RoleSubmissionCandidate, ...]:
        """Return H-qualified retained permits without releasing any bytes.

        A fragment of a deferred capture is a candidate only once it is
        materialized (``include_deferred``: also before, to plan which)."""

        if (
            self.client is None
            or self.adapter is None
            or self.submission_cut is None
            or self.submission_fragment_versions is None
            or self.final_reached
        ):
            return ()
        self.client.check_health()
        self._queue_pulls()
        deferred = (self.adapter.deferred_fragment_ids(self.submission_cut)
                    if self._defer_capture and not include_deferred else frozenset())
        candidates = []
        for fragment_id in sorted(self._pending):
            if fragment_id in deferred:
                continue
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
            if c_steps < self.config.local_horizon + self._first_push_extra(fragment_id):
                continue
            c_tokens = self.submission_tokens - self._tokens_at_anchor[fragment_id]
            if c_tokens < 1:
                raise RuntimeError("role-stream token counter moved backwards")
            anchor = self._anchors[fragment_id]
            if anchor is None:
                raise RuntimeError(
                    "role-stream fragment has no authoritative anchor"
                )
            local_descriptor = self._submission_descriptors.get(fragment_id)
            if self._lazy_capture and local_descriptor is None and not include_deferred:
                # Not materialized through this stream: no content descriptor
                # to check its chunks against.
                continue
            fragment = self.adapter.layout.fragments.fragments[fragment_id]
            expert_tags = None
            if self.config.expert_token_weights:
                expert_tags = fragment_expert_tags(
                    [name for name, _ in fragment.tensors],
                    self.submission_routed - self._routed_at_anchor[fragment_id],
                )
            candidates.append(
                RoleSubmissionCandidate(
                    self.config.role,
                    permit,
                    version,
                    self.submission_rounds,
                    c_steps,
                    c_tokens,
                    fragment.numel * 4,
                    self.submission_cut,
                    anchor,
                    expert_tags,
                    local_descriptor,
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

        if self.client is None or self.final_reached:
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

    @property
    def encodes_ahead(self) -> bool:
        """Whether :meth:`prepare_push` may compute a push's wire bytes ahead
        of it and off the event-loop thread (S27 G-AG): no int4 error
        feedback (its residual state is committed per push, in push order)
        and a deterministic codec (``SyncerClient.can_encode_ahead``)."""
        return (self.error_feedback is None and self.client is not None and self.adapter is not None
                and bool(getattr(self.client, "can_encode_ahead", False))
                and callable(getattr(self.client, "encode_push_parts", None)))

    def prepare_push(self, candidate: RoleSubmissionCandidate) -> PreparedPush:
        """Compute one candidate's delta and wire encoding now (thread-safe:
        reads the retained cut and anchor, writes nothing shared).  The bytes
        equal what :meth:`push_candidate` streams for the same candidate."""

        if not self.encodes_ahead:
            raise RuntimeError("role stream cannot encode a push ahead")
        fragment_id = candidate.permit.fragment_id
        timings: dict[str, float] = {}
        started = time.monotonic()
        parts = timed_parts(self.adapter.delta_parts_from_authoritative(
            candidate.anchor, candidate.submission_cut, fragment_id, ray_module=self.ray_module,
            **_local_descriptor_kwargs(candidate),
        )(), timings, "delta_s")
        encoded = self.client.encode_push_parts(fragment_id, parts, timings=timings)
        timings["prepare_s"] = time.monotonic() - started
        return PreparedPush(candidate.submission_cut, candidate.anchor, encoded, timings)

    def push_candidate(
        self,
        candidate: RoleSubmissionCandidate,
        prepared: PreparedPush | None = None,
    ) -> RoleFragmentSubmission | None:
        """Release one coordinator-approved candidate on its exact permit.

        A zero-chunk outage returns ``None``.  A partial-generation loss is
        allowed to escape to the paired coordinator, which retains both cuts
        and waits for this exact logical PULL to be replayed after reconnect.
        ``prepared`` (:meth:`prepare_push` of this candidate's cut and anchor)
        sends bytes computed ahead instead of streaming them now.
        """

        if not self.pushable(candidate):
            return None
        fragment_id = candidate.permit.fragment_id
        if prepared is not None:
            return self._push_prepared(candidate, prepared)
        # S27 G-AG: split this push's time (the parts are yielded unchanged).
        timings: dict[str, float] = {}
        started = time.monotonic()
        parts = timed_parts(self.adapter.delta_parts_from_authoritative(
            candidate.anchor,
            candidate.submission_cut,
            fragment_id,
            ray_module=self.ray_module,
            **_local_descriptor_kwargs(candidate),
        )(), timings, "delta_s")
        if self.error_feedback is not None:
            parts = timed_parts(self.error_feedback.parts(
                fragment_id,
                candidate.permit.global_step,
                self.config.expected_fragments,
                self.adapter.layout.fragments.fragments[fragment_id].numel,
                parts,
                self.client.push_segments(fragment_id),
            ), timings, "feedback_incl_s")
        extra: dict[str, Any] = {} if candidate.expert_tags is None else {"expert_tags": candidate.expert_tags}
        if getattr(self.client, "push_timings_supported", False):
            extra["timings"] = timings
        try:
            queued = self.client.push_fragment_parts(
                fragment_id,
                candidate.permit.global_step,
                candidate.permit.round_attempt,
                candidate.base_version,
                candidate.local_step,
                candidate.c_steps,
                candidate.c_tokens,
                parts,
                **extra,
            )
        except BaseException:
            if self.error_feedback is not None:
                self.error_feedback.discard(fragment_id)
            raise
        finally:
            self._note_push_timing(candidate, timings, time.monotonic() - started)
        if self.error_feedback is not None:
            if queued is True:
                self.error_feedback.commit(fragment_id, candidate.permit.global_step)
            else:
                self.error_feedback.discard(fragment_id)
        if queued is not True:
            return None
        return self._pushed(candidate)

    def _push_prepared(self, candidate: RoleSubmissionCandidate,
                       prepared: PreparedPush) -> RoleFragmentSubmission | None:
        started = time.monotonic()
        try:
            sent = self.send_prepared(candidate, prepared)
        except BaseException:
            self.note_push_timing(candidate, {**prepared.timings, "push_s": time.monotonic() - started})
            raise
        self.note_push_timing(candidate, sent.timings)
        if not sent.queued:
            return None
        return self._pushed(candidate, sent.link_generation)

    # -- background pushes (S27 G-AP) ----------------------------------------
    # ``RolePairing`` runs prepare_push + send_prepared on a worker thread
    # while the island trains; pushable, note_pushed and note_push_timing run
    # on the coordinator's thread, which owns every stream table.

    def pushable(self, candidate: RoleSubmissionCandidate) -> bool:
        """Whether ``candidate`` may be pushed now: its operands are retained
        and its permit is still the pending one (a superseded or answered
        permit returns False).  Raises when the candidate no longer belongs
        to this stream."""

        if (
            candidate.role != self.config.role
            or self.client is None
            or self.adapter is None
            or self.final_reached
            or id(candidate.submission_cut) not in self._pinned_cuts
            or id(candidate.anchor) not in self._pinned_anchors
        ):
            raise RuntimeError("role-stream paired submission candidate changed")
        return self._pending.get(candidate.permit.fragment_id) == candidate.permit

    def send_prepared(self, candidate: RoleSubmissionCandidate, prepared: PreparedPush) -> SentPush:
        """Enqueue bytes :meth:`prepare_push` computed for ``candidate``.

        Thread-safe: touches only the syncer client (whose enqueue is
        thread-safe) and no stream table.  Returns whether every chunk was
        queued and the generation of the link it was queued on; a
        partial-generation loss is reported as not queued (the pairing waits
        for the replayed PULL, as for a synchronous push)."""

        if (prepared.submission_cut is not candidate.submission_cut or prepared.anchor is not candidate.anchor
                or prepared.encoded.fragment_id != candidate.permit.fragment_id):
            raise RuntimeError("prepared push belongs to another candidate")
        extra: dict[str, Any] = {} if candidate.expert_tags is None else {"expert_tags": candidate.expert_tags}
        timings = dict(prepared.timings)
        link_generation = self._link_generation(candidate.permit.fragment_id)
        started = time.monotonic()
        try:
            queued = self.client.push_fragment_parts(
                candidate.permit.fragment_id,
                candidate.permit.global_step,
                candidate.permit.round_attempt,
                candidate.base_version,
                candidate.local_step,
                candidate.c_steps,
                candidate.c_tokens,
                (),
                encoded=prepared.encoded,
                **extra,
            )
        except PartialMessageGenerationLost:
            queued = False
        timings["push_s"] = time.monotonic() - started
        return SentPush(queued is True, link_generation, timings, time.monotonic())

    def note_pushed(self, candidate: RoleSubmissionCandidate, link_generation: int) -> RoleFragmentSubmission:
        """Record a background push that :meth:`send_prepared` queued."""

        return self._pushed(candidate, link_generation)

    def note_push_timing(self, candidate: RoleSubmissionCandidate, timings: Mapping[str, float]) -> None:
        """Account one prepared push's time split (:meth:`_note_push_timing`)."""

        self._note_push_timing(candidate, dict(timings), timings.get("push_s", 0.0), prepared=True)

    def _pushed(self, candidate: RoleSubmissionCandidate, link_generation: int | None = None) -> RoleFragmentSubmission:
        self._answered[
            (candidate.permit.global_step, candidate.permit.round_attempt)
        ] = self._link_generation(candidate.permit.fragment_id) if link_generation is None else link_generation
        # A background push is recorded after the fact: a newer PULL may have
        # superseded the permit, or the closed round dropped it, meanwhile.
        if self._pending.get(candidate.permit.fragment_id) == candidate.permit:
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

    def _note_push_timing(self, candidate: RoleSubmissionCandidate, timings: dict[str, float],
                          total: float, *, prepared: bool = False) -> None:
        """Print one push's time split and add it to :attr:`push_timing_totals`.

        ``delta_s``: the delta from the cut and the anchor (Ray get, checks,
        subtraction); ``feedback_s``: int4 error feedback; ``encode_s``: the
        wire encoding; ``enqueue_s``: framing and enqueueing (it waits only
        when the send queue is full; the upload itself is asynchronous);
        ``total_s``: time on the pushing thread.  A prepared push
        (``prepared=1``) computed delta and encoding ahead on a worker
        thread (``prepare_s``), so its ``total_s`` is the enqueue only."""

        delta = timings.get("delta_s", 0.0)
        feedback = max(0.0, timings.get("feedback_incl_s", delta) - delta)
        produced = timings.get("produce_s")
        encode = None if produced is None else max(0.0, produced - delta - feedback)
        if prepared:
            enqueue = timings.get("push_s", total)
        else:
            enqueue = None if produced is None else max(0.0, timings.get("push_s", total) - produced)
        split = {"delta_s": delta, "feedback_s": feedback, "encode_s": encode, "enqueue_s": enqueue,
                 "prepare_s": timings.get("prepare_s") if prepared else None, "total_s": total}
        totals = self.push_timing_totals
        totals["count"] = totals.get("count", 0) + 1
        for name, value in split.items():
            if value is not None:
                totals[name] = totals.get(name, 0.0) + value
        text = " ".join(f"{name}={value:.3f}" for name, value in split.items() if value is not None)
        print(f"[yeto-full-parameter] push role={self.config.role} fragment={candidate.permit.fragment_id} "
              f"step={candidate.permit.global_step} c_steps={candidate.c_steps} prepared={int(prepared)} {text}",
              file=sys.stderr, flush=True)

    def _queue_pulls(self) -> None:
        # A redial drops the permits and answers of that link only: the
        # syncer resends PULLs of open rounds on the new group. With a
        # sharded head each shard is its own link.
        fragments = self.config.expected_fragments
        links = {fid: self._link_generation(fid) for fid in range(fragments)}
        if self._link_generations != links:
            previous = self._link_generations or {}
            stale = {fid for fid, generation in links.items() if previous.get(fid) != generation}
            for fid in stale:
                self._pending.pop(fid, None)
            self._answered = {
                key: generation for key, generation in self._answered.items()
                if (key[0] - 1) % fragments not in stale
            }
            self._link_generations = links
        for permit in self.client.drain_pulls():
            self._validate_permit(permit)
            key = (permit.global_step, permit.round_attempt)
            if self._answered.get(key) == links[permit.fragment_id]:
                continue
            previous = self._pending.get(permit.fragment_id)
            if (
                self.config.partial_quorum
                and previous is not None
                and previous.global_step != permit.global_step
            ):
                # The syncer permits a fragment's next round only after the
                # previous one committed, so the newer permit supersedes; an
                # older one is a closed round (its broadcast may still be on
                # another WAN stream).
                if permit.global_step > previous.global_step:
                    self._pending[permit.fragment_id] = permit
                continue
            if (
                previous is not None
                and previous.global_step == permit.global_step
                and previous.round_attempt != permit.round_attempt
            ):
                # The syncer retries a round (same step and base) under a new
                # attempt after a quorum timeout and refuses pushes of the old
                # attempt. The newest attempt supersedes; a retained pair
                # rebinds its exact operands to it (``retained_replay_candidate``).
                if permit.round_attempt > previous.round_attempt:
                    self._pending[permit.fragment_id] = permit
                continue
            if previous is not None and previous != permit:
                raise RuntimeError("role-stream received conflicting PULLs")
            self._pending.setdefault(permit.fragment_id, permit)

    def _link_generation(self, fragment_id: int) -> int:
        """Connection generation of the link that carries ``fragment_id``."""
        generation_of = getattr(self.client, "connection_generation_of", None)
        if callable(generation_of):
            return int(generation_of(fragment_id))
        return int(getattr(self.client, "connection_generation", 0))

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
        """Reach the final cut: wait for the syncer's final manifest and, with
        final_cut "download", every final f32 fragment, which the trainer
        then applies. With final_cut "ack" the trainer keeps its local
        weights: the final parameters stay with the syncer."""

        if self.final_reached:
            return
        if self.client is None:
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} client is unavailable")
        expected = tuple(
            range(
                self.config.total_fragment_steps - self.config.expected_fragments + 1,
                self.config.total_fragment_steps + 1,
            )
        )
        if self.config.final_cut == "ack":
            manifest = await asyncio.to_thread(self.client.wait_for_final_manifest, self.config.wait_timeout)
            fragments = None
        else:
            manifest, fragments = await asyncio.to_thread(
                self.client.wait_for_final_fragments,
                self.config.wait_timeout,
            )
        if (
            manifest.global_step != self.config.total_fragment_steps
            or manifest.versions != expected
            or (fragments is not None and [fragment.fragment_id for fragment in fragments]
                != list(range(self.config.expected_fragments)))
        ):
            raise RuntimeError(f"{self.config.algorithm} {self.config.role} final cut is incomplete")
        if fragments is not None:
            selected = {}
            for fragment in fragments:
                stored = self._stored_update(fragment)
                if stored.version != manifest.versions[stored.fragment_id]:
                    raise RuntimeError("role-stream final fragment differs from its manifest")
                selected[stored.fragment_id] = stored
            await self._apply_authoritative(selected)
            self.final_applied = True
        self._staged.clear()
        self.terminal_manifest = manifest
        self.final_reached = True
        self._release_cuts(
            self.submission_cut,
            keep=(self.current,),
        )
        self.submission_cut = None
        self._submission_descriptors = {}
        self.submission_fragment_versions = None

    def acknowledge_final(self, manifest: FinalManifest | None = None) -> None:
        manifest = self.terminal_manifest if manifest is None else manifest
        if (
            not self.final_reached
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
        self._submission_descriptors = {}
        self.submission_fragment_versions = None

    def close(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.stop()
        if self.client is not None:
            self.client.close()
        if self._verifier is not None:
            self._verifier.close()
        if self.sink is not None:
            self.sink.release_all()
        retained_cuts = tuple(value for value, _count in self._pinned_cuts.values())
        self._pinned_cuts.clear()
        self._pinned_anchors.clear()
        self._release_cuts(self.current, self.submission_cut, self._round_base, *retained_cuts)
        self._staged.clear()
        self._round_base = None
        self.current = None
        self.submission_cut = None
        self._submission_descriptors = {}
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
            and cut is not self._round_base
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
    base_policy_version: int | None,
    base_local_generation: int | None,
    target_policy_version: int,
) -> None:
    """A global apply may only move the policy version and reset the local
    step generation; every other Adam field must be bit-identical. A base of
    None is not checked (the first apply after a peer snapshot restore)."""
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
            or base_policy_version is not None and old.installed_policy_version != base_policy_version
            or base_local_generation is not None and old.local_step_generation != base_local_generation
            or new.installed_policy_version != target_policy_version
            or new.local_step_generation != 0
            or any(
                old_values[name] != new_values[name]
                for name in old_values.keys() - volatile
            )
        ):
            raise RuntimeError("global apply changed local Adam state")
