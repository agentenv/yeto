"""Global expert names when only the MoE layer carries ``local_expert_indices``.

Megatron-Core 0.18 (the ms-swift production stack) sets
``local_expert_indices`` on ``MoELayer`` but not on its ``experts`` child
(``TEGroupedMLP`` / ``SequentialMLP``); the sync must still give each
expert-parallel rank's experts their global indices, or the experts of every
EP rank collapse onto the names of EP rank 0's.
"""

from __future__ import annotations

import pytest
import torch

from yeto.swift_megatron import sync as sm


class _Module(torch.nn.Module):
    pass


def _moe_chunk(ep_rank: int, local_experts: int, *, indices_on_experts: bool, grouped: bool):
    root = _Module()
    root.decoder = _Module()
    root.decoder.layers = torch.nn.ModuleList()
    layer = _Module()
    layer.layer_number = 1
    layer.mlp = _Module()
    indices = [ep_rank * local_experts + e for e in range(local_experts)]
    layer.mlp.local_expert_indices = indices
    experts = _Module()
    if indices_on_experts:
        experts.local_expert_indices = indices
    if grouped:
        experts.linear_fc1 = _Module()
        for e in range(local_experts):
            p = torch.nn.Parameter(torch.zeros(2, 2))
            p.allreduce = False
            setattr(experts.linear_fc1, f"weight{e}", p)
    else:
        experts.local_experts = torch.nn.ModuleList()
        for _ in range(local_experts):
            mlp = _Module()
            mlp.linear_fc1 = _Module()
            mlp.linear_fc1.weight = torch.nn.Parameter(torch.zeros(2, 2))
            mlp.linear_fc1.weight.allreduce = False
            experts.local_experts.append(mlp)
    layer.mlp.experts = experts
    root.decoder.layers.append(layer)
    return root


def _ctx():
    return sm.ParallelContext(rank=0, world=1, tp_group=None, tp_rank=0, tp_size=1,
                              etp_group=None, etp_rank=0, etp_size=1)


@pytest.mark.parametrize("grouped", [True, False])
def test_expert_indices_fall_back_to_the_moe_layer(grouped):
    names = set()
    for ep_rank in range(4):
        chunk = _moe_chunk(ep_rank, 2, indices_on_experts=False, grouped=grouped)
        names |= {s.name for s in sm.local_shards([chunk], _ctx())}
    if grouped:
        expected = {f"decoder.layers.0.mlp.experts.linear_fc1.weight{e}" for e in range(8)}
    else:
        expected = {f"decoder.layers.0.mlp.experts.local_experts.{e}.linear_fc1.weight" for e in range(8)}
    assert names == expected


def test_container_indices_still_win():
    chunk = _moe_chunk(1, 2, indices_on_experts=True, grouped=True)
    assert sm._expert_modules(chunk) == [("decoder.layers.0.mlp.experts.", [2, 3])]


def test_expert_without_any_indices_is_refused():
    chunk = _moe_chunk(1, 2, indices_on_experts=False, grouped=True)
    del chunk.decoder.layers[0].mlp.local_expert_indices
    with pytest.raises(sm.SwiftMegatronSyncError, match="local_expert_indices"):
        sm.local_shards([chunk], _ctx())
