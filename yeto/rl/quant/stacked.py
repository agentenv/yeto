"""Trainer-native stacked expert layout -> per-expert source names (S25 N8.5b).

The quantization layout (:mod:`.requant`) names one BF16 source matrix per
routed expert, as the HF per-expert checkpoint does::

    <layer>.mlp.experts.<e>.gate_proj.weight   [I, H]
    <layer>.mlp.experts.<e>.up_proj.weight     [I, H]
    <layer>.mlp.experts.<e>.down_proj.weight   [H, I]

Trainers hold the experts stacked. The HF native layout (FSDP2 / verl, and the
copy in Modal ``glm-rl-tglm:/native/apex-flash-1-cecb5eeb``) stores, per layer::

    <layer>.mlp.experts.gate_up_proj   [E, 2I, H]   rows 0..I-1 gate, I..2I-1 up
    <layer>.mlp.experts.down_proj      [E, H, I]

Megatron (Miles) stores one ``linear_fc1`` ``[2I, H]`` (gate rows, then up
rows) and one ``linear_fc2`` ``[H, I]`` per local expert; its adapter
(:mod:`yeto.rl.adapters.miles.quant_shards`) maps them with
:func:`split_gate_up`.

This module only renames and slices: the values are not changed, so a stacked
source quantizes to the same bytes as the per-expert one.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch

STACKED_EXPERTS = re.compile(r"^(?P<prefix>.+\.mlp\.experts)\.(?P<kind>gate_up_proj|down_proj)$")
GATE, UP, DOWN = "gate_proj", "up_proj", "down_proj"


def expert_weight_name(experts_prefix: str, expert: int, proj: str) -> str:
    """``<prefix>.<expert>.<proj>.weight`` (``prefix`` ends in ``.mlp.experts``)."""
    return f"{experts_prefix}.{int(expert)}.{proj}.weight"


def split_gate_up(experts_prefix: str, expert: int, gate_up: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    """The gate and up matrices of one expert's merged ``[2I, H]`` weight (gate rows first)."""
    if gate_up.dim() != 2 or gate_up.shape[0] % 2:
        raise ValueError(f"{experts_prefix}.{expert}: merged gate/up weight must be [2I, H], got {tuple(gate_up.shape)}")
    inner = gate_up.shape[0] // 2
    yield expert_weight_name(experts_prefix, expert, GATE), gate_up[:inner]
    yield expert_weight_name(experts_prefix, expert, UP), gate_up[inner:]


def unstack_experts(name: str, tensor: torch.Tensor, first_expert: int = 0) -> Iterator[tuple[str, torch.Tensor]]:
    """Per-expert matrices of a stacked ``gate_up_proj`` / ``down_proj`` (or of its
    dim-0 slice that starts at expert ``first_expert``)."""
    match = STACKED_EXPERTS.match(name)
    if match is None:
        raise ValueError(f"{name}: not a stacked expert tensor")
    if tensor.dim() != 3:
        raise ValueError(f"{name}: stacked experts must be 3-D, got {tuple(tensor.shape)}")
    prefix = match["prefix"]
    for j in range(tensor.shape[0]):
        if match["kind"] == "down_proj":
            yield expert_weight_name(prefix, first_expert + j, DOWN), tensor[j]
        else:
            yield from split_gate_up(prefix, first_expert + j, tensor[j])


@dataclass(frozen=True)
class ExpertView:
    """Rows ``[row_start, row_stop)`` of expert ``expert`` in the stored stacked tensor ``stored``."""

    stored: str
    expert: int
    row_start: int
    row_stop: int

    def read(self, stacked_slice) -> torch.Tensor:
        """Read the view from a safetensors slice of ``stored`` (only these rows are read)."""
        return stacked_slice[self.expert, self.row_start:self.row_stop]


@dataclass(frozen=True)
class StackedSourceIndex:
    """Per-expert names of the stacked tensors in a checkpoint index."""

    views: Mapping[str, ExpertView]
    stacked_names: frozenset[str]

    @classmethod
    def from_names(cls, names: Sequence[str] | Mapping[str, object],
                   shape_of: Callable[[str], Sequence[int]]) -> "StackedSourceIndex":
        """``shape_of(name)`` is called for stacked tensors only (a header read)."""
        views: dict[str, ExpertView] = {}
        stacked = [n for n in names if STACKED_EXPERTS.match(n)]
        for name in stacked:
            match = STACKED_EXPERTS.match(name)
            shape = tuple(shape_of(name))
            if len(shape) != 3:
                raise ValueError(f"{name}: stacked experts must be 3-D, got {shape}")
            experts, rows = shape[0], shape[1]
            if match["kind"] == "down_proj":
                parts = ((DOWN, 0, rows),)
            else:
                if rows % 2:
                    raise ValueError(f"{name}: merged gate/up has an odd row count {rows}")
                parts = ((GATE, 0, rows // 2), (UP, rows // 2, rows))
            for e in range(experts):
                for proj, a, b in parts:
                    views[expert_weight_name(match["prefix"], e, proj)] = ExpertView(name, e, a, b)
        if clash := sorted(set(views) & set(names)):
            raise ValueError(f"source has both stacked and per-expert experts, e.g. {clash[:3]}")
        return cls(views, frozenset(stacked))
