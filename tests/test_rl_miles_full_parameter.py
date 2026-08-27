from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest
import torch

from yeto.rl.contracts import TrajectoryEnvelope
from yeto.rl.local_learner import ComponentIdentity
from yeto.rl.miles_full_parameter import MilesFullParameterAdapter
from yeto.rl.trajectory_evidence import TrajectoryBatchEvidence

REVISION = "a" * 40
CONFIG_HASH = "b" * 64


@dataclass(frozen=True, order=True)
class Spec:
    role: str
    shard_id: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int

    @property
    def wire_name(self):
        return f"{self.role}::{self.shard_id}::{self.name}"


@dataclass(frozen=True)
class State:
    policy_version: int
    topology: int
    layout_hash: str
    specs: tuple[Spec, ...]
    tensors: dict[str, torch.Tensor]
    local_step_generation: int = 0


class Group:
    def __init__(self, shards):
        self.shards = tuple(shards)
        self.applied = None

    async def export_full_parameter_shards(
        self,
        policy_version,
        local_step_generation=0,
    ):
        assert policy_version == self.shards[0].policy_version
        assert local_step_generation == self.shards[0].local_step_generation
        return self.shards

    async def apply_full_parameter_shards(self, states):
        self.applied = states
        return sum(len(state.specs) for state in states)

    async def record_full_parameter_local_step(
        self,
        *,
        base_policy_version,
        rollout_id,
    ):
        return tuple(
            SimpleReceipt(
                state.topology,
                "actor",
                base_policy_version,
                1,
                rollout_id,
                1,
                12,
                13,
            )
            for state in self.shards
        )


@dataclass(frozen=True)
class SimpleReceipt:
    topology: int
    role: str
    base_policy_version: int
    local_step_generation: int
    rollout_id: int
    optimizer_steps: int
    scheduler_start_steps: int
    scheduler_end_steps: int


def shards(version=3, offset=0.0):
    result = []
    for rank in range(2):
        shard_id = f"tp{rank}-of-2.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1"
        spec = Spec(
            "actor",
            shard_id,
            "model.proj.weight",
            (2, 2),
            "float32",
            4,
        )
        result.append(
            State(
                version,
                rank,
                str(rank + 1) * 64,
                (spec,),
                {
                    spec.wire_name: torch.arange(4, dtype=torch.float32).reshape(2, 2)
                    + rank
                    + offset
                },
            )
        )
    return tuple(result)


def test_adapter_round_trips_complete_topology_cut():
    group = Group(shards())
    adapter, anchor = asyncio.run(
        MilesFullParameterAdapter.capture_initial(
            group,
            policy_version=3,
            algorithm="grpo",
            components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
            num_fragments=2,
        )
    )
    updated = adapter.cut_from_shards(
        shards(version=4, offset=0.25),
        expected_policy_version=4,
    )

    assert anchor.policy_version == 3
    assert updated.policy_version == 4
    assert len(adapter.layout.specs) == 2
    assert asyncio.run(adapter.apply(group, updated)) == 2
    assert group.applied is not None
    assert [state.topology for state in group.applied] == [0, 1]
    assert all(state.policy_version == 4 for state in group.applied)
    assert all(
        torch.equal(
            next(iter(state.tensors.values())),
            next(iter(expected.tensors.values())),
        )
        for state, expected in zip(group.applied, shards(4, 0.25), strict=True)
    )


def test_adapter_rejects_missing_mixed_or_changed_shards_before_apply():
    initial = shards()
    adapter = MilesFullParameterAdapter.create(
        initial,
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        num_fragments=2,
    )

    with pytest.raises(ValueError, match="count changed"):
        adapter.cut_from_shards(initial[:1], expected_policy_version=3)
    with pytest.raises(ValueError, match="mix policy versions"):
        adapter.cut_from_shards(
            (initial[0], replace(initial[1], policy_version=4)),
            expected_policy_version=3,
        )
    with pytest.raises(ValueError, match="mix local-step generations"):
        adapter.cut_from_shards(
            (initial[0], replace(initial[1], local_step_generation=1)),
            expected_policy_version=3,
        )
    with pytest.raises(ValueError, match="identity changed"):
        adapter.cut_from_shards(
            (initial[0], replace(initial[1], layout_hash="f" * 64)),
            expected_policy_version=3,
        )


def test_adapter_rejects_nonfinite_tensor_and_incomplete_application():
    initial = shards()
    first = initial[0]
    name = next(iter(first.tensors))
    malformed = replace(
        first,
        tensors={name: torch.full((2, 2), float("nan"))},
    )
    with pytest.raises(ValueError, match="malformed Miles tensor"):
        MilesFullParameterAdapter.create(
            (malformed, initial[1]),
            algorithm="grpo",
            components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
            num_fragments=2,
        )

    class IncompleteGroup(Group):
        async def apply_full_parameter_shards(self, states):
            self.applied = states
            return 1

    adapter = MilesFullParameterAdapter.create(
        initial,
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        num_fragments=2,
    )
    cut = adapter.cut_from_shards(initial, expected_policy_version=3)
    with pytest.raises(RuntimeError, match="incomplete"):
        asyncio.run(adapter.apply(IncompleteGroup(initial), cut))


def test_adapter_captures_local_generation_without_relabeling_global_base():
    local_shards = tuple(
        replace(
            state,
            local_step_generation=1,
            tensors={name: value + 0.25 for name, value in state.tensors.items()},
        )
        for state in shards()
    )
    adapter = MilesFullParameterAdapter.create(
        shards(),
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        num_fragments=2,
    )

    local = asyncio.run(
        adapter.capture(
            Group(local_shards),
            policy_version=3,
            local_step_generation=1,
        )
    )

    assert local.policy_version == 3
    with pytest.raises(ValueError, match="identity changed"):
        adapter.cut_from_shards(
            local_shards,
            expected_policy_version=3,
            expected_local_step_generation=0,
        )


def test_adapter_binds_every_rank_scheduler_receipt_to_trajectory_batch():
    group = Group(shards())
    adapter = MilesFullParameterAdapter.create(
        shards(),
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        num_fragments=2,
    )
    anchor = adapter.cut_from_shards(shards(), expected_policy_version=3)
    trajectory = TrajectoryEnvelope(
        trajectory_id="trajectory-1",
        task_id="CVE-2026-0001",
        prompt_group_id="r3:g0",
        sample_index=0,
        behavior_policy_version=3,
        behavior_policy_hash=anchor.policy_hash,
        token_ids=(1, 2, 3),
        response_token_count=2,
        behavior_logprobs_hash="c" * 64,
        reward=1.0,
        reward_contract_hash="d" * 64,
        cleanup_evidence_hash="e" * 64,
    )
    evidence = TrajectoryBatchEvidence(
        3,
        anchor.policy_hash,
        "f" * 64,
        2,
        (trajectory,),
    )

    receipt = asyncio.run(
        adapter.record_grpo_local_step(
            group,
            anchor=anchor,
            rollout_id=3,
            learner_id=9,
            learner_generation=4,
            trajectories=evidence,
        )
    )

    assert receipt.base_policy_hash == anchor.policy_hash
    assert receipt.input_batch_hash == "f" * 64
    assert receipt.trajectory_ids == ("trajectory-1",)
    assert receipt.trained_tokens == 2
    assert receipt.optimizer_steps == 1


def test_adapter_rejects_rank_receipt_disagreement_before_emitting_local_credit():
    class DivergentGroup(Group):
        async def record_full_parameter_local_step(self, **kwargs):
            values = list(await super().record_full_parameter_local_step(**kwargs))
            values[1] = replace(values[1], scheduler_end_steps=14)
            return tuple(values)

    group = DivergentGroup(shards())
    adapter = MilesFullParameterAdapter.create(
        shards(),
        algorithm="grpo",
        components=(ComponentIdentity("actor", REVISION, CONFIG_HASH),),
        num_fragments=2,
    )
    anchor = adapter.cut_from_shards(shards(), expected_policy_version=3)
    trajectory = TrajectoryEnvelope(
        "trajectory-1",
        "CVE-2026-0001",
        "r3:g0",
        0,
        3,
        anchor.policy_hash,
        (1,),
        1,
        None,
        1.0,
        "d" * 64,
        "e" * 64,
    )
    evidence = TrajectoryBatchEvidence(
        3, anchor.policy_hash, "f" * 64, 1, (trajectory,)
    )
    with pytest.raises(RuntimeError, match="disagree on scheduler"):
        asyncio.run(
            adapter.record_grpo_local_step(
                group,
                anchor=anchor,
                rollout_id=3,
                learner_id=9,
                learner_generation=4,
                trajectories=evidence,
            )
        )
