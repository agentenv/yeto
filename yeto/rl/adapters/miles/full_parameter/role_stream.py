"""Miles binding of one trainable role's full-parameter streaming session.

The engine-neutral stream (anchors, PULL permits, retention, initial and final
cuts, peer recovery) is :class:`yeto.rl.engine.full_parameter_stream.FullParameterRoleStream`;
this module binds it to Miles: the reference-backed chunked cut of a Miles
trainer group (``models/chunked_full_parameter.py``), the Ray inbound chunk
sink, the Miles backend identity of the session contract, Megatron's
optimizer-state proofs, the MoE routed-token counter and the peer-recovery
rank shards.  The public names of the stream module stay importable from here.
"""

from __future__ import annotations

import asyncio
from typing import Any

from yeto.rl.adapters.miles.full_parameter.inbound_sink import MilesRayInboundChunkSink
from yeto.rl.adapters.miles.full_parameter.peer_recovery import PeerRecoveryOptions  # noqa: F401 - re-export
from yeto.rl.adapters.miles.full_parameter.rank_stats import report_rank_stats
from yeto.rl.adapters.miles.models.chunked_full_parameter import (
    MilesChunkedFullParameterAdapter,
    ReferencedPolicyCut,
    StoredAuthoritativeFragment,
    StoredFragmentVerifier,
    defer_capture_supported,
    lazy_capture_supported,
    optimizer_proof_without_model_check_supported,
    timed_group,
    verified_apply_supported,
)
from yeto.rl.adapters.miles.routing_plugin import (
    DRAIN_ROUTED_TOKEN_COUNTS,
    INSTALL_ROUTED_TOKEN_COUNTER,
)
from yeto.rl.engine.full_parameter_stream import (  # noqa: F401 - the stream's public names
    DEFAULT_FULL_CHECK_EVERY,
    FINAL_CUT_MODES,
    FULL_CHECK_EVERY_ENV,
    INITIAL_CUT_MIN_BYTES_PER_SECOND,
    STAGGER_FIRST_PUSH_ENV,
    TERMINAL_RETIRE_ENV,
    FullParameterRoleStream,
    PoolFinishedBeforeJoin,
    PreparedPush,
    RoleFragmentSubmission,
    RoleRoundResult,
    RoleStreamConfig,
    RoleSubmissionCandidate,
    SentPush,
    first_push_phase,
    full_check_every,
    initial_cut_timeout,
    require_sha256,
    session_contract_hash,
)
from yeto.rl.engine.shape_free_layout import LAYOUT_SHAPE_FREE
from yeto.rl.host_info import print_host_report, report_rank_hosts
from yeto.rl.local_learner import layout_stream_role
from yeto.rl.moe_layout import RoutedTokens, check_expert_partition


def role_stream_session_contract_hash(
    layout_hash: str,
    config: RoleStreamConfig,
) -> bytes:
    """Bind a reconnecting Miles role stream to one immutable experiment."""

    from yeto.rl.adapters.miles.identity import backend_identity

    return session_contract_hash(layout_hash, config, backend_identity_sha256=backend_identity().sha256())


class MilesFullParameterRoleStream(FullParameterRoleStream):
    """Reference-backed full-parameter transport for one Miles trainable role."""

    stored_fragment_type = StoredAuthoritativeFragment

    async def _initialize_adapter(self, group) -> tuple[MilesChunkedFullParameterAdapter, ReferencedPolicyCut]:
        return await MilesChunkedFullParameterAdapter.initialize(
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
            **(
                {}
                if self.config.shape_free is None
                else {"layout_mode": LAYOUT_SHAPE_FREE, "granularity": self.config.shape_free}
            ),
        )

    def _inbound_chunk_sink(self, sink) -> MilesRayInboundChunkSink:
        return MilesRayInboundChunkSink(sink)

    def _session_contract_hash(self, layout_hash: str) -> bytes:
        return role_stream_session_contract_hash(layout_hash, self.config)

    async def _prepare_group(self, group):
        # S27 G-AG diagnostics: host facts of the driver and of every trainer
        # rank, then one timed line per Miles group call.
        print_host_report(f"{self.config.role} driver learner {self.config.learner_id}")
        await report_rank_hosts(group, f"{self.config.role} learner {self.config.learner_id}")
        return timed_group(group, self.config.role)

    async def _report_rank_stats(self, phase: str) -> None:
        await report_rank_stats(self.group, phase, self.config.role)

    def _supports_deferred_capture(self) -> bool:
        return defer_capture_supported(self.group)

    def _supports_lazy_capture(self) -> bool:
        return lazy_capture_supported(self.group)

    def _supports_proof_without_model_check(self) -> bool:
        return optimizer_proof_without_model_check_supported(self.group)

    def _stored_fragment_verifier(self, group) -> StoredFragmentVerifier | None:
        # S27 G-CG verify-ahead: Miles' own prepare check
        # (verify_stored_fragment_chunks) on a background thread; the apply
        # names the checked fragments (Miles transport feature verified_apply).
        if not verified_apply_supported(group):
            return None
        return StoredFragmentVerifier(ray_module=self.ray_module)

    async def _optimizer_states(self) -> tuple[Any, ...]:
        return tuple(await self.group.full_parameter_optimizer_states(**self._proof_kwargs()))

    def _check_optimizer_transition(self, before, after, *, target_policy_version: int) -> None:
        validate_optimizer_apply_transition(
            before,
            after,
            role=self.config.role,
            base_policy_version=self.transport_version,
            base_local_generation=self.local_generation,
            target_policy_version=target_policy_version,
        )

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

    def _recovery_island(self, group, adapter, *, recovering: bool):
        from yeto.rl.adapters.miles.full_parameter.peer_recovery import (
            MilesPeerRecoveryIsland,
            MilesRankShards,
        )

        shards = MilesRankShards(group, adapter.layout.layout_hash)
        return MilesPeerRecoveryIsland(self, shards, asyncio.get_running_loop(), recovering=recovering)


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
