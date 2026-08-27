"""Reference-backed Miles full-parameter boundary for dense DiLoCo."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .contracts import LocalStepReceipt
from .local_learner import ComponentIdentity, ParameterLayout, ParameterSpec
from .trajectory_evidence import TrajectoryBatchEvidence


class MilesChunkedFullParameterGroup(Protocol):
    async def full_parameter_shard_manifests(self) -> Sequence[Any]: ...

    async def install_full_parameter_fragment_plans(
        self,
        plans: tuple[Any, ...],
    ) -> int: ...

    async def export_full_parameter_chunked_cut(
        self,
        policy_version: int,
        local_step_generation: int = 0,
        *,
        max_chunk_bytes: int,
    ) -> Any: ...

    async def apply_full_parameter_chunked_cut(
        self,
        cut: Any,
        *,
        commit_token: str | None = None,
        max_commit_attempts: int = 3,
    ) -> int: ...

    async def record_full_parameter_local_step(
        self,
        *,
        base_policy_version: int,
        rollout_id: int,
    ) -> Sequence[Any]: ...

    async def full_parameter_optimizer_states(self) -> Sequence[Any]: ...


@dataclass(frozen=True)
class ReferencedPolicyCut:
    policy_version: int
    local_step_generation: int
    layout_hash: str
    policy_hash: str
    content_hash: str
    transport_cut: Any


@dataclass(frozen=True)
class StoredAuthoritativeFragment:
    fragment_id: int
    version: int
    parameter_layout_hash: str
    topology: Any
    plan_hash: str
    descriptor: Any
    refs: list[object | None]
    wire_payload_hash: str

    def release(self) -> int:
        released = 0
        for index, reference in enumerate(self.refs):
            if reference is not None:
                self.refs[index] = None
                released += 1
        return released


class AuthoritativeFragmentSink:
    """Idempotently convert one bounded wire fragment to immutable Ray refs."""

    def __init__(self, reference: ReferencedPolicyCut, *, ray_module=None) -> None:
        # Inbound parsing needs only immutable topology/fragment/chunk
        # descriptors.  A metadata-only cut prevents this long-lived sink
        # from pinning the model-sized version-zero Ray payload for the full
        # run.
        self.reference = _metadata_only_reference(reference)
        self.ray_module = ray_module
        self._stored: dict[tuple[int, int], StoredAuthoritativeFragment] = {}

    def __call__(
        self,
        fragment_id: int,
        version: int,
        payload: bytes | bytearray | memoryview,
    ) -> StoredAuthoritativeFragment:
        from miles.ray.full_parameter_transport import (
            store_full_parameter_fragment_payload,
        )

        raw = memoryview(payload).cast("B")
        payload_hash = hashlib.sha256(raw).hexdigest()
        key = (fragment_id, version)
        previous = self._stored.get(key)
        if previous is not None:
            raw.release()
            if previous.wire_payload_hash != payload_hash:
                raise RuntimeError("authoritative fragment replay changed bytes")
            return previous
        raw.release()
        descriptor, refs = store_full_parameter_fragment_payload(
            self.reference.transport_cut,
            fragment_id,
            payload,
            ray_module=self.ray_module,
        )
        shard = _fragment_owner_shard(
            self.reference.transport_cut,
            fragment_id,
        )
        stored = StoredAuthoritativeFragment(
            fragment_id=fragment_id,
            version=version,
            parameter_layout_hash=shard.parameter_layout_hash,
            topology=shard.topology,
            plan_hash=shard.plan_hash,
            descriptor=descriptor,
            refs=refs,
            wire_payload_hash=payload_hash,
        )
        self._stored[key] = stored
        return stored

    def release_all(self) -> int:
        return sum(fragment.release() for fragment in self._stored.values())


@dataclass(frozen=True)
class MilesChunkedFullParameterAdapter:
    layout: ParameterLayout
    manifests: tuple[Any, ...]
    plans: tuple[Any, ...]
    expected_parameter_tensor_count: int
    expected_parameter_scalar_count: int
    max_chunk_bytes: int
    max_fragment_bytes: int

    @classmethod
    async def initialize(
        cls,
        group: MilesChunkedFullParameterGroup,
        *,
        policy_version: int,
        algorithm: str,
        components: Sequence[ComponentIdentity],
        minimum_fragments: int,
        expected_fragments: int | None = None,
        expected_layout_hash: str | None = None,
        max_fragment_bytes: int = 2 << 30,
        max_chunk_bytes: int = 256 << 20,
        stream_role: str | None = None,
    ) -> tuple[MilesChunkedFullParameterAdapter, ReferencedPolicyCut]:
        manifests = tuple(
            sorted(
                await group.full_parameter_shard_manifests(),
                key=lambda value: value.topology,
            )
        )
        adapter = cls.create(
            manifests,
            algorithm=algorithm,
            components=components,
            minimum_fragments=minimum_fragments,
            max_fragment_bytes=max_fragment_bytes,
            max_chunk_bytes=max_chunk_bytes,
            stream_role=stream_role,
        )
        actual_fragments = adapter.layout.fragments.num_fragments
        if (
            expected_fragments is not None and actual_fragments != expected_fragments
        ) or (
            expected_layout_hash is not None
            and adapter.layout.layout_hash != expected_layout_hash
        ):
            fragment_bytes = tuple(
                fragment.numel * 4 for fragment in adapter.layout.fragments.fragments
            )
            raise RuntimeError(
                "Miles full-parameter layout differs from the frozen probe: "
                f"expected_fragments={expected_fragments}, "
                f"actual_fragments={actual_fragments}, "
                f"expected_layout_hash={expected_layout_hash}, "
                f"actual_layout_hash={adapter.layout.layout_hash}, "
                f"fragment_bytes={fragment_bytes}"
            )
        installed = await group.install_full_parameter_fragment_plans(adapter.plans)
        if installed != adapter.layout.fragments.num_fragments:
            raise RuntimeError("Miles installed an incomplete owner fragment plan")
        cut = await adapter.capture(
            group,
            policy_version=policy_version,
            local_step_generation=0,
        )
        return adapter, cut

    @classmethod
    def create(
        cls,
        manifests: Sequence[Any],
        *,
        algorithm: str,
        components: Sequence[ComponentIdentity],
        minimum_fragments: int,
        max_fragment_bytes: int,
        max_chunk_bytes: int,
        stream_role: str | None = None,
    ) -> MilesChunkedFullParameterAdapter:
        from miles.backends.megatron_utils.full_parameter_state import (
            FullParameterFragmentPlan,
            FullParameterOwnerFragmentPlan,
        )

        ordered = tuple(sorted(manifests, key=lambda value: value.topology))
        if not ordered or len({value.topology for value in ordered}) != len(ordered):
            raise ValueError("Miles full-parameter manifests are incomplete")
        specs = tuple(
            ParameterSpec(
                spec.role,
                spec.name,
                tuple(spec.shape),
                spec.dtype,
                spec.numel,
                spec.shard_id,
            )
            for manifest in ordered
            for spec in manifest.specs
        )
        if len({spec.wire_name for spec in specs}) != len(specs):
            raise ValueError("Miles full-parameter manifests overlap")
        if (
            type(max_fragment_bytes) is not int
            or max_fragment_bytes < 4
            or max_fragment_bytes > 2 << 30
            or max_fragment_bytes % 4
        ):
            raise ValueError("Miles semantic fragment byte bound is invalid")
        start = max(minimum_fragments, len(ordered))
        layout = None
        for count in range(start, len(specs) + 1):
            candidate = ParameterLayout.create(
                algorithm=algorithm,
                components=components,
                specs=specs,
                num_fragments=count,
                fragment_strategy="owner_affine",
                stream_role=stream_role,
            )
            if all(
                fragment.numel * 4 <= max_fragment_bytes
                for fragment in candidate.fragments.fragments
            ):
                layout = candidate
                break
        if layout is None:
            raise ValueError(
                "a single topology-owned parameter exceeds the fragment byte bound"
            )

        manifests_by_owner = {}
        for manifest in ordered:
            owners = {(spec.role, spec.shard_id) for spec in manifest.specs}
            if len(owners) != 1:
                raise ValueError("Miles manifest mixes role/topology owners")
            owner = next(iter(owners))
            if owner in manifests_by_owner:
                raise ValueError("Miles manifests repeat a role/topology owner")
            manifests_by_owner[owner] = manifest
        plans = []
        for owner, manifest in sorted(
            manifests_by_owner.items(),
            key=lambda item: item[1].topology,
        ):
            fragments = []
            for fragment_id, fragment in enumerate(layout.fragments.fragments):
                if layout.fragment_owner(fragment_id) != owner:
                    continue
                fragments.append(
                    FullParameterFragmentPlan(
                        fragment_id=fragment_id,
                        wire_names=tuple(name for name, _ in fragment.tensors),
                        numel=fragment.numel,
                    )
                )
            plans.append(
                FullParameterOwnerFragmentPlan(
                    topology=manifest.topology,
                    parameter_layout_hash=layout.layout_hash,
                    fragments=tuple(fragments),
                )
            )
        if sum(len(plan.fragments) for plan in plans) != layout.fragments.num_fragments:
            raise RuntimeError("owner fragment plans do not cover the layout")
        return cls(
            layout=layout,
            manifests=ordered,
            plans=tuple(plans),
            expected_parameter_tensor_count=len(specs),
            expected_parameter_scalar_count=sum(spec.numel for spec in specs),
            max_chunk_bytes=max_chunk_bytes,
            max_fragment_bytes=max_fragment_bytes,
        )

    async def capture(
        self,
        group: MilesChunkedFullParameterGroup,
        *,
        policy_version: int,
        local_step_generation: int,
    ) -> ReferencedPolicyCut:
        cut = await group.export_full_parameter_chunked_cut(
            policy_version,
            local_step_generation,
            max_chunk_bytes=self.max_chunk_bytes,
        )
        return self.validate_cut(cut, policy_version, local_step_generation)

    def validate_cut(
        self,
        cut: Any,
        policy_version: int,
        local_step_generation: int,
    ) -> ReferencedPolicyCut:
        from miles.ray.full_parameter_transport import (
            full_parameter_chunked_cut_content_identity,
            full_parameter_chunked_cut_identity,
        )

        if (
            cut.policy_version != policy_version
            or cut.local_step_generation != local_step_generation
            or cut.parameter_layout_hash != self.layout.layout_hash
            or len(cut.shards) != len(self.plans)
        ):
            raise ValueError("Miles reference-backed cut identity changed")
        expected = {plan.topology: plan.plan_hash for plan in self.plans}
        if {shard.topology: shard.plan_hash for shard in cut.shards} != expected:
            raise ValueError("Miles reference-backed owner plan changed")
        return ReferencedPolicyCut(
            policy_version=policy_version,
            local_step_generation=local_step_generation,
            layout_hash=self.layout.layout_hash,
            policy_hash=full_parameter_chunked_cut_identity(cut),
            content_hash=full_parameter_chunked_cut_content_identity(cut),
            transport_cut=cut,
        )

    def delta_parts(
        self,
        anchor: ReferencedPolicyCut,
        local: ReferencedPolicyCut,
        fragment_id: int,
        *,
        ray_module=None,
    ) -> Callable[[], Any]:
        from miles.ray.full_parameter_transport import (
            iter_full_parameter_fragment_delta_parts,
        )

        if (
            anchor.layout_hash != self.layout.layout_hash
            or local.layout_hash != self.layout.layout_hash
            or anchor.policy_version != local.policy_version
            or anchor.local_step_generation != 0
            or local.local_step_generation != 1
        ):
            raise ValueError("Miles local cut does not bind the reference anchor")

        def parts():
            return iter_full_parameter_fragment_delta_parts(
                anchor.transport_cut,
                local.transport_cut,
                fragment_id,
                ray_module=ray_module,
            )

        return parts

    def delta_parts_from_authoritative(
        self,
        stored: StoredAuthoritativeFragment,
        local: ReferencedPolicyCut,
        local_fragment_id: int,
        *,
        ray_module=None,
    ) -> Callable[[], Any]:
        """Stream ``local - authoritative`` for one arbitrary local horizon.

        The authoritative fragment may have arrived independently of the
        learner's current complete cut.  Miles validates its exact owner,
        plan, chunk geometry, values, and refs while keeping both operands
        reference-backed.
        """

        from miles.ray.full_parameter_transport import (
            iter_full_parameter_authoritative_fragment_delta_parts,
        )

        if (
            local.layout_hash != self.layout.layout_hash
            or local.local_step_generation < 1
        ):
            raise ValueError("Miles local cut has no synchronized local horizon")
        if (
            isinstance(local_fragment_id, bool)
            or not isinstance(local_fragment_id, int)
            or not 0 <= local_fragment_id < self.layout.fragments.num_fragments
        ):
            raise ValueError("Miles local fragment ID is outside the layout")
        owner = _fragment_owner_shard(local.transport_cut, local_fragment_id)
        if (
            isinstance(stored.version, bool)
            or not isinstance(stored.version, int)
            or stored.version < 0
            or stored.fragment_id != local_fragment_id
            or stored.parameter_layout_hash != self.layout.layout_hash
            or stored.topology != owner.topology
            or stored.plan_hash != owner.plan_hash
        ):
            raise ValueError("authoritative fragment provenance changed")

        def parts():
            return iter_full_parameter_authoritative_fragment_delta_parts(
                local.transport_cut,
                local_fragment_id,
                authoritative_parameter_layout_hash=(stored.parameter_layout_hash),
                authoritative_topology=stored.topology,
                authoritative_plan_hash=stored.plan_hash,
                authoritative_descriptor=stored.descriptor,
                authoritative_refs=stored.refs,
                ray_module=ray_module,
            )

        return parts

    def fragment_parts(
        self,
        cut: ReferencedPolicyCut,
        fragment_id: int,
        *,
        ray_module=None,
    ) -> Callable[[], Any]:
        from miles.ray.full_parameter_transport import (
            iter_full_parameter_fragment_parts,
        )

        if cut.layout_hash != self.layout.layout_hash:
            raise ValueError("Miles fragment source layout changed")

        def parts():
            return iter_full_parameter_fragment_parts(
                cut.transport_cut,
                fragment_id,
                ray_module=ray_module,
            )

        return parts

    async def record_grpo_local_step(
        self,
        group: MilesChunkedFullParameterGroup,
        *,
        anchor: ReferencedPolicyCut,
        rollout_id: int,
        learner_id: int,
        learner_generation: int,
        trajectories: TrajectoryBatchEvidence,
    ) -> LocalStepReceipt:
        """Bind one H=1 Megatron step to the accepted GRPO trajectories."""

        if (
            anchor.layout_hash != self.layout.layout_hash
            or anchor.policy_version != rollout_id
            or trajectories.rollout_id != rollout_id
            or trajectories.behavior_policy_hash != anchor.policy_hash
            or not trajectories.envelopes
        ):
            raise ValueError("Miles local-step evidence does not bind the anchor")
        return await self.record_local_round(
            group,
            anchor=anchor,
            rollout_id=rollout_id,
            learner_id=learner_id,
            learner_generation=learner_generation,
            trajectories=trajectories,
            role="actor",
            expected_local_step_generation=1,
            expected_optimizer_steps=1,
        )

    async def record_local_round(
        self,
        group: MilesChunkedFullParameterGroup,
        *,
        anchor: ReferencedPolicyCut,
        rollout_id: int,
        learner_id: int,
        learner_generation: int,
        trajectories: TrajectoryBatchEvidence,
        role: str,
        expected_local_step_generation: int,
        expected_optimizer_steps: int,
    ) -> LocalStepReceipt:
        """Bind one actor or critic local round to exact rank receipts."""

        if role not in {"actor", "critic"}:
            raise ValueError("Miles local-round role is unsupported")
        if {spec.role for spec in self.layout.specs} != {role} or (
            self.layout.algorithm == "sao" and self.layout.stream_role != role
        ):
            raise ValueError("Miles local-round role does not own this layout")
        for name, value in (
            ("learner_id", learner_id),
            ("learner_generation", learner_generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (
            ("expected_local_step_generation", expected_local_step_generation),
            ("expected_optimizer_steps", expected_optimizer_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            anchor.layout_hash != self.layout.layout_hash
            or anchor.local_step_generation != expected_local_step_generation - 1
            or trajectories.rollout_id != rollout_id
            or not trajectories.envelopes
            or (
                role == "actor"
                and trajectories.behavior_policy_hash
                != (
                    anchor.content_hash
                    if self.layout.algorithm == "sao"
                    else anchor.policy_hash
                )
            )
        ):
            raise ValueError("Miles local-round evidence does not bind the anchor")
        receipts = tuple(
            await group.record_full_parameter_local_step(
                base_policy_version=anchor.policy_version,
                rollout_id=rollout_id,
            )
        )
        expected_topologies = {manifest.topology for manifest in self.manifests}
        by_topology = {}
        for receipt in receipts:
            if receipt.topology in by_topology:
                raise RuntimeError("Miles returned duplicate local-step receipts")
            if (
                receipt.topology not in expected_topologies
                or receipt.role != role
                or receipt.base_policy_version != anchor.policy_version
                or receipt.local_step_generation != expected_local_step_generation
                or receipt.rollout_id != rollout_id
                or receipt.optimizer_steps != expected_optimizer_steps
                or receipt.scheduler_end_steps <= receipt.scheduler_start_steps
            ):
                raise RuntimeError("Miles local-round receipt changed")
            by_topology[receipt.topology] = receipt
        if set(by_topology) != expected_topologies:
            raise RuntimeError("Miles local-step receipts do not cover every owner")
        if (
            len(
                {
                    (receipt.scheduler_start_steps, receipt.scheduler_end_steps)
                    for receipt in receipts
                }
            )
            != 1
        ):
            raise RuntimeError("Miles ranks disagree on scheduler progress")
        return LocalStepReceipt(
            algorithm=self.layout.algorithm,
            learner_id=learner_id,
            learner_generation=learner_generation,
            base_policy_version=anchor.policy_version,
            base_policy_hash=(
                anchor.content_hash
                if self.layout.algorithm == "sao"
                else anchor.policy_hash
            ),
            input_batch_hash=trajectories.input_batch_hash,
            trajectory_ids=trajectories.trajectory_ids,
            trained_tokens=trajectories.trained_tokens,
            optimizer_steps=expected_optimizer_steps,
            optimizer_step_succeeded=True,
            parameter_layout_hash=self.layout.layout_hash,
        )

    def fragment_sink(
        self,
        reference: ReferencedPolicyCut,
        *,
        ray_module=None,
    ) -> AuthoritativeFragmentSink:
        if reference.layout_hash != self.layout.layout_hash:
            raise ValueError("Miles fragment sink reference layout changed")
        return AuthoritativeFragmentSink(reference, ray_module=ray_module)

    def verify_initial_fragments(
        self,
        reference: ReferencedPolicyCut,
        fragments: Sequence[StoredAuthoritativeFragment],
    ) -> None:
        """Require the syncer's authoritative v0 bytes to equal the checkpoint."""

        expected = {
            descriptor.fragment_id: (shard, descriptor)
            for shard in reference.transport_cut.shards
            for descriptor in shard.fragments
        }
        observed = {}
        for stored in fragments:
            row = expected.get(stored.fragment_id)
            if (
                row is None
                or stored.fragment_id in observed
                or stored.version != 0
                or stored.parameter_layout_hash != self.layout.layout_hash
                or stored.topology != row[0].topology
                or stored.plan_hash != row[0].plan_hash
                or stored.descriptor != row[1]
            ):
                raise RuntimeError("dense syncer version-zero fragment changed")
            observed[stored.fragment_id] = stored
        if set(observed) != set(expected):
            raise RuntimeError("dense syncer version-zero cut is incomplete")

    def assemble_target(
        self,
        reference: ReferencedPolicyCut,
        *,
        target_policy_version: int,
        fragments: Sequence[StoredAuthoritativeFragment],
    ) -> ReferencedPolicyCut:
        from miles.ray.full_parameter_transport import (
            assemble_full_parameter_chunked_target,
        )

        by_id = {}
        reference_shards = {
            shard.topology: shard for shard in reference.transport_cut.shards
        }
        for stored in fragments:
            shard = reference_shards.get(stored.topology)
            if (
                stored.fragment_id in by_id
                or shard is None
                or stored.parameter_layout_hash != self.layout.layout_hash
                or stored.plan_hash != shard.plan_hash
                or stored.version
                != self._fragment_version(target_policy_version, stored.fragment_id)
            ):
                raise ValueError("authoritative fragment provenance changed")
            by_id[stored.fragment_id] = (stored.descriptor, stored.refs)
        target = assemble_full_parameter_chunked_target(
            reference.transport_cut,
            target_policy_version=target_policy_version,
            authoritative_fragments=by_id,
        )
        return self.validate_cut(target, target_policy_version, 0)

    def assemble_mixed_target(
        self,
        current: ReferencedPolicyCut,
        *,
        target_policy_version: int,
        authoritative_fragments: Sequence[StoredAuthoritativeFragment],
    ) -> ReferencedPolicyCut:
        """Replace an arbitrary fragment subset in a complete current cut."""

        from miles.ray.full_parameter_transport import (
            assemble_full_parameter_chunked_target,
        )

        if current.layout_hash != self.layout.layout_hash:
            raise ValueError("Miles current cut layout changed")
        rows = {
            descriptor.fragment_id: (descriptor, references)
            for shard in current.transport_cut.shards
            for descriptor, references in zip(
                shard.fragments,
                shard.chunk_refs,
                strict=True,
            )
        }
        if set(rows) != set(range(self.layout.fragments.num_fragments)):
            raise ValueError("Miles current cut does not cover the layout")
        replaced: set[int] = set()
        for stored in authoritative_fragments:
            if stored.fragment_id in replaced or stored.fragment_id not in rows:
                raise ValueError("authoritative fragment subset is malformed")
            owner = _fragment_owner_shard(
                current.transport_cut,
                stored.fragment_id,
            )
            expected_descriptor, _expected_refs = rows[stored.fragment_id]
            if (
                isinstance(stored.version, bool)
                or not isinstance(stored.version, int)
                or stored.version < 0
                or stored.parameter_layout_hash != self.layout.layout_hash
                or stored.topology != owner.topology
                or stored.plan_hash != owner.plan_hash
                or not _same_fragment_geometry(
                    expected_descriptor,
                    stored.descriptor,
                )
            ):
                raise ValueError("authoritative fragment provenance changed")
            rows[stored.fragment_id] = (stored.descriptor, stored.refs)
            replaced.add(stored.fragment_id)
        target = assemble_full_parameter_chunked_target(
            current.transport_cut,
            target_policy_version=target_policy_version,
            authoritative_fragments=rows,
        )
        return self.validate_cut(target, target_policy_version, 0)

    async def apply(
        self,
        group: MilesChunkedFullParameterGroup,
        target: ReferencedPolicyCut,
        *,
        commit_token: str | None = None,
    ) -> int:
        if target.layout_hash != self.layout.layout_hash:
            raise ValueError("Miles target layout changed")
        applied = await group.apply_full_parameter_chunked_cut(
            target.transport_cut,
            commit_token=commit_token,
        )
        if applied != self.expected_parameter_tensor_count:
            raise RuntimeError("Miles applied an incomplete reference-backed cut")
        return applied

    def release(self, cut: ReferencedPolicyCut) -> int:
        from miles.ray.full_parameter_transport import (
            release_full_parameter_chunked_cut,
        )

        return release_full_parameter_chunked_cut(cut.transport_cut)

    def _fragment_version(self, policy_version: int, fragment_id: int) -> int:
        if policy_version < 1:
            return 0
        return (
            (policy_version - 1) * self.layout.fragments.num_fragments + fragment_id + 1
        )


def _fragment_owner_shard(cut: Any, fragment_id: int) -> Any:
    matches = [
        shard
        for shard in cut.shards
        if any(descriptor.fragment_id == fragment_id for descriptor in shard.fragments)
    ]
    if len(matches) != 1:
        raise ValueError("authoritative fragment has no unique owner")
    return matches[0]


def _same_fragment_geometry(expected: Any, observed: Any) -> bool:
    try:
        return (
            expected.fragment_id == observed.fragment_id
            and expected.numel == observed.numel
            and tuple(
                (chunk.chunk_index, chunk.flat_offset, chunk.numel)
                for chunk in expected.chunks
            )
            == tuple(
                (chunk.chunk_index, chunk.flat_offset, chunk.numel)
                for chunk in observed.chunks
            )
        )
    except (AttributeError, TypeError):
        return False


_METADATA_REFERENCE = object()


def _metadata_only_reference(reference: ReferencedPolicyCut) -> ReferencedPolicyCut:
    """Clone cut descriptors while replacing every model payload ObjectRef."""

    from miles.ray.full_parameter_transport import FullParameterChunkedCut

    cut = reference.transport_cut
    shards = tuple(
        replace(
            shard,
            chunk_refs=[
                [_METADATA_REFERENCE] * len(descriptor.chunks)
                for descriptor in shard.fragments
            ],
        )
        for shard in cut.shards
    )
    metadata = FullParameterChunkedCut(
        policy_version=cut.policy_version,
        local_step_generation=cut.local_step_generation,
        parameter_layout_hash=cut.parameter_layout_hash,
        shards=shards,
    )
    return ReferencedPolicyCut(
        policy_version=reference.policy_version,
        local_step_generation=reference.local_step_generation,
        layout_hash=reference.layout_hash,
        policy_hash=reference.policy_hash,
        content_hash=reference.content_hash,
        transport_cut=metadata,
    )
