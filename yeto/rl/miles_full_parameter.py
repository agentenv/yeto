"""Adapter between Miles Megatron shards and Yeto parameter cuts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from .contracts import LocalStepReceipt
from .local_learner import (
    ComponentIdentity,
    ParameterCut,
    ParameterLayout,
    ParameterSpec,
    make_parameter_cut,
    parameter_values,
)
from .trajectory_evidence import TrajectoryBatchEvidence

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MilesFullParameterGroup(Protocol):
    """The narrow async surface implemented by Miles ``RayTrainGroup``."""

    async def export_full_parameter_shards(
        self,
        policy_version: int,
        local_step_generation: int = 0,
    ) -> Sequence[Any]: ...

    async def apply_full_parameter_shards(
        self,
        states: tuple[Any, ...],
    ) -> int: ...

    async def record_full_parameter_local_step(
        self,
        *,
        base_policy_version: int,
        rollout_id: int,
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class MilesFullParameterAdapter:
    """Translate a complete Miles actor group to one role-qualified layout.

    The adapter intentionally knows nothing about GRPO losses, SecRLEnv, or the
    sync protocol. It only preserves the exact Miles topology/spec templates
    while converting FP32 optimizer-master values to and from Yeto cuts.
    """

    layout: ParameterLayout
    shard_templates: tuple[Any, ...]
    expected_parameter_tensor_count: int
    expected_parameter_scalar_count: int

    @classmethod
    def create(
        cls,
        shards: Sequence[Any],
        *,
        algorithm: str,
        components: Sequence[ComponentIdentity],
        num_fragments: int,
    ) -> MilesFullParameterAdapter:
        ordered = _validate_and_order_shards(shards)
        specs = tuple(
            ParameterSpec(
                spec.role,
                spec.name,
                tuple(spec.shape),
                spec.dtype,
                spec.numel,
                spec.shard_id,
            )
            for state in ordered
            for spec in state.specs
        )
        layout = ParameterLayout.create(
            algorithm=algorithm,
            components=components,
            specs=specs,
            num_fragments=num_fragments,
        )
        templates = tuple(replace(state, tensors={}) for state in ordered)
        return cls(
            layout,
            templates,
            len(specs),
            sum(spec.numel for spec in specs),
        )

    @classmethod
    async def capture_initial(
        cls,
        group: MilesFullParameterGroup,
        *,
        policy_version: int,
        algorithm: str,
        components: Sequence[ComponentIdentity],
        num_fragments: int,
    ) -> tuple[MilesFullParameterAdapter, ParameterCut]:
        """Capture the initial group cut after the caller reaches a safe step boundary."""

        shards = await group.export_full_parameter_shards(
            policy_version,
            local_step_generation=0,
        )
        adapter = cls.create(
            shards,
            algorithm=algorithm,
            components=components,
            num_fragments=num_fragments,
        )
        return adapter, adapter.cut_from_shards(
            shards,
            expected_policy_version=policy_version,
            expected_local_step_generation=0,
        )

    async def capture(
        self,
        group: MilesFullParameterGroup,
        *,
        policy_version: int,
        local_step_generation: int = 0,
    ) -> ParameterCut:
        """Capture one subsequent complete cut from the same Miles topology."""

        shards = await group.export_full_parameter_shards(
            policy_version,
            local_step_generation=local_step_generation,
        )
        return self.cut_from_shards(
            shards,
            expected_policy_version=policy_version,
            expected_local_step_generation=local_step_generation,
        )

    def cut_from_shards(
        self,
        shards: Sequence[Any],
        *,
        expected_policy_version: int,
        expected_local_step_generation: int = 0,
    ) -> ParameterCut:
        ordered = _validate_and_order_shards(shards)
        if len(ordered) != len(self.shard_templates):
            raise ValueError("Miles shard count changed")
        values = {}
        for state, template in zip(ordered, self.shard_templates, strict=True):
            if (
                state.policy_version != expected_policy_version
                or getattr(state, "local_step_generation", 0)
                != expected_local_step_generation
                or state.topology != template.topology
                or state.layout_hash != template.layout_hash
                or state.specs != template.specs
            ):
                raise ValueError("Miles full-parameter shard identity changed")
            values.update(
                (name, value.clone()) for name, value in state.tensors.items()
            )
        return make_parameter_cut(
            self.layout,
            policy_version=expected_policy_version,
            values=values,
        )

    def shards_from_cut(self, cut: ParameterCut) -> tuple[Any, ...]:
        values = parameter_values(self.layout, cut)
        states = []
        for template in self.shard_templates:
            names = tuple(spec.wire_name for spec in template.specs)
            tensors = {name: values[name].clone() for name in names}
            states.append(
                replace(
                    template,
                    policy_version=cut.policy_version,
                    tensors=tensors,
                )
            )
        return tuple(states)

    async def apply(
        self,
        group: MilesFullParameterGroup,
        cut: ParameterCut,
    ) -> int:
        """Apply a complete cut; Miles prevalidates every rank before copying."""

        applied = await group.apply_full_parameter_shards(self.shards_from_cut(cut))
        if (
            isinstance(applied, bool)
            or not isinstance(applied, int)
            or applied != self.expected_parameter_tensor_count
        ):
            raise RuntimeError("Miles applied an incomplete full-parameter cut")
        return applied

    async def record_grpo_local_step(
        self,
        group: MilesFullParameterGroup,
        *,
        anchor: ParameterCut,
        rollout_id: int,
        learner_id: int,
        learner_generation: int,
        trajectories: TrajectoryBatchEvidence,
    ) -> LocalStepReceipt:
        """Bind Miles' scheduler receipt to the exact accepted trajectory batch."""

        if (
            anchor.layout_hash != self.layout.layout_hash
            or anchor.policy_version != rollout_id
            or trajectories.rollout_id != rollout_id
            or trajectories.behavior_policy_hash != anchor.policy_hash
            or not trajectories.envelopes
        ):
            raise ValueError("Miles local-step evidence does not bind the anchor")
        for name, value in (
            ("learner_id", learner_id),
            ("learner_generation", learner_generation),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        receipts = tuple(
            await group.record_full_parameter_local_step(
                base_policy_version=anchor.policy_version,
                rollout_id=rollout_id,
            )
        )
        if not receipts or len(receipts) != len(self.shard_templates):
            raise RuntimeError("Miles returned incomplete local-step receipts")
        templates = {
            template.topology: tuple(template.specs)
            for template in self.shard_templates
        }
        topology_receipts = {}
        for receipt in receipts:
            topology = receipt.topology
            if topology in topology_receipts or topology not in templates:
                raise RuntimeError("Miles local-step receipt topology changed")
            if (
                receipt.role != "actor"
                or receipt.base_policy_version != anchor.policy_version
                or receipt.local_step_generation != 1
                or receipt.rollout_id != rollout_id
                or receipt.optimizer_steps != 1
                or receipt.scheduler_end_steps <= receipt.scheduler_start_steps
            ):
                raise RuntimeError("Miles local-step receipt is outside H=1")
            topology_receipts[topology] = receipt
        if set(topology_receipts) != set(templates):
            raise RuntimeError("Miles local-step receipts do not cover every shard")
        scheduler_spans = {
            (
                receipt.scheduler_start_steps,
                receipt.scheduler_end_steps,
            )
            for receipt in receipts
        }
        if len(scheduler_spans) != 1:
            raise RuntimeError("Miles ranks disagree on scheduler progress")
        return LocalStepReceipt(
            algorithm="grpo",
            learner_id=learner_id,
            learner_generation=learner_generation,
            base_policy_version=anchor.policy_version,
            base_policy_hash=anchor.policy_hash,
            input_batch_hash=trajectories.input_batch_hash,
            trajectory_ids=trajectories.trajectory_ids,
            trained_tokens=trajectories.trained_tokens,
            optimizer_steps=1,
            optimizer_step_succeeded=True,
            parameter_layout_hash=self.layout.layout_hash,
        )


def _validate_and_order_shards(shards: Sequence[Any]) -> tuple[Any, ...]:
    if not shards:
        raise ValueError("Miles full-parameter cut is empty")
    ordered = tuple(sorted(shards, key=_shard_identity))
    identities = [_shard_identity(state) for state in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("Miles full-parameter cut has duplicate shards")
    all_names = set()
    versions = set()
    generations = set()
    for state in ordered:
        if not _SHA256.fullmatch(state.layout_hash):
            raise ValueError("Miles shard layout hash is malformed")
        versions.add(state.policy_version)
        generations.add(getattr(state, "local_step_generation", 0))
        specs = tuple(state.specs)
        if not specs or specs != tuple(sorted(specs)):
            raise ValueError("Miles full-parameter specs are not canonical")
        names = {spec.wire_name for spec in specs}
        if set(state.tensors) != names or all_names.intersection(names):
            raise ValueError("Miles full-parameter cut is incomplete or overlapping")
        all_names.update(names)
        for spec in specs:
            value = state.tensors[spec.wire_name]
            _validate_tensor(spec, value)
    if len(versions) != 1:
        raise ValueError("Miles full-parameter shards mix policy versions")
    if len(generations) != 1:
        raise ValueError("Miles full-parameter shards mix local-step generations")
    return ordered


def _shard_identity(state: Any) -> str:
    specs = tuple(state.specs)
    if not specs:
        raise ValueError("Miles full-parameter shard is empty")
    identities = {spec.shard_id for spec in specs}
    if len(identities) != 1:
        raise ValueError("Miles state mixes topology shards")
    return next(iter(identities))


def _validate_tensor(spec: Any, value: Any) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or tuple(value.shape) != tuple(spec.shape)
        or value.numel() != spec.numel
        or not torch.isfinite(value).all().item()
    ):
        raise ValueError(f"malformed Miles tensor {spec.wire_name!r}")
