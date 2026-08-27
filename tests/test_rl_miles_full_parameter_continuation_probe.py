from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.miles_full_parameter_continuation_probe import (
    MilesFullParameterContinuationProbeSync,
)

REVISION = "a" * 40
CONFIG_HASH = "b" * 64


@dataclass(frozen=True, order=True)
class Topology:
    tp_rank: int

    @property
    def shard_id(self):
        return f"tp{self.tp_rank}-of-2.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1"


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
    topology: Topology
    layout_hash: str
    specs: tuple[Spec, ...]
    tensors: dict[str, torch.Tensor]
    local_step_generation: int = 0


@dataclass(frozen=True)
class Receipt:
    topology: Topology
    role: str
    base_policy_version: int
    local_step_generation: int
    rollout_id: int
    optimizer_steps: int
    scheduler_start_steps: int
    scheduler_end_steps: int


@dataclass(frozen=True)
class OptimizerProof:
    topology: Topology
    role: str
    installed_policy_version: int
    local_step_generation: int
    last_rollout_id: int
    scheduler_num_steps: int
    populated_parameter_count: int
    optimizer_state_tensor_count: int
    optimizer_state_scalar_count: int
    selected_wire_name: str
    selected_state_sha256: str
    model_master_parameter_count: int


def _shards(version=0, generation=0, offset=0.0):
    result = []
    for rank in range(2):
        topology = Topology(rank)
        spec = Spec(
            "actor",
            topology.shard_id,
            "model.weight",
            (2, 2),
            "float32",
            4,
        )
        result.append(
            State(
                version,
                topology,
                str(rank + 1) * 64,
                (spec,),
                {
                    spec.wire_name: torch.arange(4, dtype=torch.float32)
                    .reshape(2, 2)
                    .add(rank + offset)
                },
                generation,
            )
        )
    return tuple(result)


class ContinuationGroup:
    def __init__(self, *, mutate=True, mutate_moments_on_apply=False):
        self.shards = _shards()
        self.policy_version = 0
        self.generation = 0
        self.rollout_id = 0
        self.scheduler = 0
        self.moment_generation = 0
        self.mutate = mutate
        self.mutate_moments_on_apply = mutate_moments_on_apply

    async def export_full_parameter_shards(
        self,
        policy_version,
        local_step_generation=0,
    ):
        assert policy_version == self.policy_version
        assert local_step_generation == self.generation
        return self.shards

    async def record_full_parameter_local_step(
        self,
        *,
        base_policy_version,
        rollout_id,
    ):
        assert base_policy_version == self.policy_version
        assert self.generation == 0
        self.generation = 1
        self.rollout_id = rollout_id
        self.scheduler += 1
        self.moment_generation += 1
        if self.mutate:
            self.shards = tuple(
                replace(
                    state,
                    local_step_generation=1,
                    tensors={
                        name: value + 0.25 for name, value in state.tensors.items()
                    },
                )
                for state in self.shards
            )
        else:
            self.shards = tuple(
                replace(state, local_step_generation=1) for state in self.shards
            )
        return tuple(
            Receipt(
                state.topology,
                "actor",
                base_policy_version,
                1,
                rollout_id,
                1,
                self.scheduler - 1,
                self.scheduler,
            )
            for state in self.shards
        )

    async def apply_full_parameter_shards(self, states):
        self.shards = tuple(
            replace(
                state,
                tensors={name: value.clone() for name, value in state.tensors.items()},
            )
            for state in states
        )
        self.policy_version = states[0].policy_version
        self.generation = 0
        if self.mutate_moments_on_apply:
            self.moment_generation += 1
        return sum(len(state.specs) for state in states)

    async def full_parameter_optimizer_states(self):
        return tuple(
            OptimizerProof(
                state.topology,
                "actor",
                self.policy_version,
                self.generation,
                self.rollout_id,
                self.scheduler,
                1,
                3,
                9,
                state.specs[0].wire_name,
                str(self.moment_generation + state.topology.tp_rank + 1) * 64,
                1,
            )
            for state in self.shards
        )


def _configure(monkeypatch, evidence):
    monkeypatch.setenv("YETO_FULL_PARAMETER_CONTINUATION_EVIDENCE", str(evidence))
    monkeypatch.setenv("YETO_FULL_PARAMETER_MODEL_REVISION", REVISION)
    monkeypatch.setenv("YETO_FULL_PARAMETER_CONFIG_HASH", CONFIG_HASH)
    monkeypatch.setenv("YETO_FULL_PARAMETER_FRAGMENT_COUNT", "2")
    monkeypatch.setenv("YETO_FULL_PARAMETER_CONVERSION_MANIFEST_SHA256", "c" * 64)
    monkeypatch.setenv("YETO_MILES_IMAGE_DIGEST", f"sha256:{'d' * 64}")
    yeto_source = evidence.parent / "yeto-source"
    miles_source = evidence.parent / "miles-source"
    yeto_source.mkdir(exist_ok=True)
    miles_source.mkdir(exist_ok=True)
    (yeto_source / "module.py").write_text("YETO = True\n")
    (miles_source / "module.py").write_text("MILES = True\n")
    monkeypatch.setenv("YETO_FULL_PARAMETER_YETO_SOURCE_ROOT", str(yeto_source))
    monkeypatch.setenv("YETO_FULL_PARAMETER_MILES_SOURCE_ROOT", str(miles_source))


def _args():
    return SimpleNamespace(
        start_rollout_id=0,
        num_rollout=2,
        num_steps_per_rollout=1,
        global_batch_size=1,
        debug_train_only=True,
        loss_type="sft_loss",
    )


def test_two_step_probe_preserves_moments_and_advances_two_global_cuts(
    monkeypatch,
    tmp_path,
):
    evidence = tmp_path / "continuation.json"
    _configure(monkeypatch, evidence)
    monkeypatch.setattr(
        "yeto.rl.miles_full_parameter_continuation_probe._hardware_identity",
        lambda: {"gpu_count": 2},
    )
    group = ContinuationGroup()
    probe = MilesFullParameterContinuationProbeSync(_args())

    asyncio.run(probe.initialize(actor_model=group, rollout_manager=object()))
    assert (
        asyncio.run(
            probe.after_local_train(
                rollout_id=0,
                actor_model=group,
                rollout_data=object(),
            )
        )
        is False
    )
    assert (
        asyncio.run(
            probe.after_local_train(
                rollout_id=1,
                actor_model=group,
                rollout_data=object(),
            )
        )
        is True
    )
    asyncio.run(probe.finalize())

    payload = json.loads(evidence.read_text())
    assert payload["final_policy_version"] == 2
    assert payload["first_changed_fragment_count"] == 2
    assert payload["second_changed_fragment_count"] == 2
    assert payload["optimizer_state_preserved_across_apply"] is True
    assert (
        payload["optimizer_state_proof_scope"]
        == "selected_parameter_per_topology_plus_cardinalities"
    )
    assert (
        payload["first_optimizer_state_before_apply"]
        == payload["first_optimizer_state_after_apply"]
    )
    assert (
        payload["second_optimizer_state_before_apply"]
        == payload["second_optimizer_state_after_apply"]
    )
    assert payload["next_step_after_global_apply_verified"] is True
    assert payload["initial_policy_hash"] != payload["first_local_policy_hash"]
    assert payload["first_global_policy_hash"] != payload["second_local_policy_hash"]
    assert evidence.stat().st_mode & 0o777 == 0o600


def test_two_step_probe_rejects_unchanged_local_payload(monkeypatch, tmp_path):
    evidence = tmp_path / "continuation.json"
    _configure(monkeypatch, evidence)
    monkeypatch.setattr(
        "yeto.rl.miles_full_parameter_continuation_probe._hardware_identity",
        lambda: {"gpu_count": 2},
    )
    group = ContinuationGroup(mutate=False)
    probe = MilesFullParameterContinuationProbeSync(_args())
    asyncio.run(probe.initialize(actor_model=group, rollout_manager=object()))

    with pytest.raises(RuntimeError, match="changed no parameters"):
        asyncio.run(
            probe.after_local_train(
                rollout_id=0,
                actor_model=group,
                rollout_data=object(),
            )
        )


def test_two_step_probe_rejects_optimizer_state_mutation_during_apply(
    monkeypatch,
    tmp_path,
):
    evidence = tmp_path / "continuation.json"
    _configure(monkeypatch, evidence)
    monkeypatch.setattr(
        "yeto.rl.miles_full_parameter_continuation_probe._hardware_identity",
        lambda: {"gpu_count": 2},
    )
    group = ContinuationGroup(mutate_moments_on_apply=True)
    probe = MilesFullParameterContinuationProbeSync(_args())
    asyncio.run(probe.initialize(actor_model=group, rollout_manager=object()))

    with pytest.raises(RuntimeError, match="changed Adam optimizer state"):
        asyncio.run(
            probe.after_local_train(
                rollout_id=0,
                actor_model=group,
                rollout_data=object(),
            )
        )
