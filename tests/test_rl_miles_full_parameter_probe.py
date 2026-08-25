from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.miles_full_parameter_probe import MilesFullParameterProbeSync

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


def _shards(version=0):
    result = []
    for rank in range(2):
        shard_id = f"tp{rank}-of-2.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1"
        spec = Spec("actor", shard_id, "model.weight", (2, 2), "float32", 4)
        result.append(
            State(
                version,
                rank,
                str(rank + 1) * 64,
                (spec,),
                {
                    spec.wire_name: torch.arange(4, dtype=torch.float32).reshape(2, 2)
                    + rank
                },
            )
        )
    return tuple(result)


class RoundTripGroup:
    def __init__(self):
        self.shards = _shards()

    async def export_full_parameter_shards(
        self,
        policy_version,
        local_step_generation=0,
    ):
        assert policy_version == self.shards[0].policy_version
        assert local_step_generation == self.shards[0].local_step_generation
        return self.shards

    async def apply_full_parameter_shards(self, states):
        self.shards = tuple(
            replace(
                state,
                tensors={name: value.clone() for name, value in state.tensors.items()},
            )
            for state in states
        )
        return sum(len(state.specs) for state in states)


def _configure(monkeypatch, evidence):
    monkeypatch.setenv("YETO_FULL_PARAMETER_PROBE_EVIDENCE", str(evidence))
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


def test_probe_round_trips_one_changed_scalar_and_writes_private_evidence(
    monkeypatch,
    tmp_path,
):
    evidence = tmp_path / "probe.json"
    _configure(monkeypatch, evidence)
    monkeypatch.setattr(
        "yeto.rl.miles_full_parameter_probe._hardware_identity",
        lambda: {"gpu_count": 2},
    )
    args = SimpleNamespace(start_rollout_id=0, num_rollout=1)
    probe = MilesFullParameterProbeSync(args)

    asyncio.run(
        probe.initialize(actor_model=RoundTripGroup(), rollout_manager=object())
    )
    asyncio.run(probe.finalize())

    payload = json.loads(evidence.read_text())
    assert args.start_rollout_id == 1
    assert payload["schema"] == "yeto-miles-full-parameter-probe-v1"
    assert payload["conversion_manifest_sha256"] == "c" * 64
    assert payload["hardware"] == {"gpu_count": 2}
    assert payload["yeto_source"]["file_count"] == 1
    assert payload["initial_policy_version"] == 0
    assert payload["target_policy_version"] == 1
    assert payload["initial_policy_hash"] != payload["target_policy_hash"]
    assert payload["target_policy_hash"] == payload["observed_policy_hash"]
    assert (
        payload["parameter_tensor_count"]
        == payload["applied_parameter_tensor_count"]
        == 2
    )
    assert payload["parameter_scalar_count"] == 8
    assert payload["changed_fragment_count"] == 1
    assert payload["changed_scalar_count"] == 1
    assert payload["master_round_trip_exact"] is True
    assert payload["model_master_round_trip_verified"] is True
    assert evidence.stat().st_mode & 0o777 == 0o600


def test_probe_rejects_scheduler_drift_and_unexpected_training(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path / "probe.json")
    monkeypatch.setattr(
        "yeto.rl.miles_full_parameter_probe._hardware_identity",
        lambda: {"gpu_count": 2},
    )
    probe = MilesFullParameterProbeSync(
        SimpleNamespace(start_rollout_id=0, num_rollout=2)
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        asyncio.run(
            probe.initialize(actor_model=RoundTripGroup(), rollout_manager=object())
        )
    with pytest.raises(RuntimeError, match="unexpectedly entered"):
        asyncio.run(probe.after_local_train())
