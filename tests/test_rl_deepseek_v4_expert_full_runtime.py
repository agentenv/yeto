from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles.models.deepseek_v4 import expert_full_runtime as runtime
from yeto.rl.adapters.miles.models.deepseek_v4.expert_clone import (
    EXPERT_PARALLEL_SIZE,
    NUM_LAYERS,
    ORIGINAL_EXPERTS_PER_RANK,
    TRAINING_EXPERTS_PER_RANK,
)


def _expert_name(
    expert: int,
    projection: str = "gate_proj",
    *,
    layer: int = 0,
) -> str:
    return (
        f"base_model.model.model.layers.{layer}.mlp.experts."
        f"{expert}.{projection}.weight"
    )


def _expert_tasks(expert: int, *, layer: int = 0) -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(
            mapping=SimpleNamespace(
                hf_param={
                    "gate": _expert_name(expert, layer=layer),
                    "up": _expert_name(expert, "up_proj", layer=layer),
                }
            )
        ),
        SimpleNamespace(
            mapping=SimpleNamespace(
                hf_param=_expert_name(expert, "down_proj", layer=layer)
            )
        ),
    )


def test_collective_expert_tasks_have_identical_ep8_topology():
    expected_count = NUM_LAYERS * 4 * 2
    for rank in range(EXPERT_PARALLEL_SIZE):
        tasks = [
            task
            for layer in range(NUM_LAYERS)
            for expert in range(
                rank * TRAINING_EXPERTS_PER_RANK,
                (rank + 1) * TRAINING_EXPERTS_PER_RANK,
            )
            for task in _expert_tasks(expert, layer=layer)
        ]

        selected = runtime.filter_collective_expert_tasks(tasks, expert_count=16)

        physical_ids = [
            int(runtime._mapping_hf_names(task)[0].split(".experts.")[1].split(".")[0])
            for task in selected
        ]
        assert len(selected) == expected_count
        assert {expert % TRAINING_EXPERTS_PER_RANK for expert in physical_ids} == {
            ORIGINAL_EXPERTS_PER_RANK + offset for offset in range(4)
        }
        assert {expert // TRAINING_EXPERTS_PER_RANK for expert in physical_ids} == {
            rank
        }


def _optimizer_fixture(*, sharded=False, optimizer_owned=True, complete=True):
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    main = torch.nn.Parameter(torch.zeros(4 if complete else 2, dtype=torch.float32))
    parameter.main_param = main
    parameter.main_param_sharded = sharded
    chunk = torch.nn.Module()
    chunk.register_parameter("weight", parameter)
    owned = main if optimizer_owned else torch.nn.Parameter(torch.zeros(4))
    optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(param_groups=[{"params": [owned]}])
    )
    return chunk, optimizer


def test_complete_optimizer_master_gate_accepts_replicated_owned_fp32():
    chunk, optimizer = _optimizer_fixture()

    runtime._validate_complete_optimizer_masters([chunk], optimizer)


@pytest.mark.parametrize(
    ("fixture_kwargs", "match"),
    (
        ({"sharded": True}, "complete, optimizer-owned"),
        ({"optimizer_owned": False}, "complete, optimizer-owned"),
        ({"complete": False}, "complete, optimizer-owned"),
    ),
)
def test_complete_optimizer_master_gate_rejects_dp_shards_and_drift(
    fixture_kwargs,
    match,
):
    chunk, optimizer = _optimizer_fixture(**fixture_kwargs)

    with pytest.raises(RuntimeError, match=match):
        runtime._validate_complete_optimizer_masters([chunk], optimizer)


