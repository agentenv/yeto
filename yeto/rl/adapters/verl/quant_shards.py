"""FSDP2 (verl / HF) parameters -> re-quantization source slices (S25 N8.2 / N8.5b adapter).

FSDP2 shards every parameter on dim 0 over a 1-D mesh (``Shard(0)``): rank
``r`` holds rows ``[r * c, min((r + 1) * c, N))`` with ``c = ceil(N / world)``
(``torch.chunk``). The HF model keeps the routed experts stacked
(``mlp.experts.gate_up_proj`` ``[E, 2I, H]``, ``mlp.experts.down_proj``
``[E, H, I]``), so a dim-0 shard of them is a range of whole experts. This
adapter turns the local shards into :class:`~yeto.rl.quant.shards.SourceSlice`
items: stacked experts into per-expert matrices (:mod:`yeto.rl.quant.stacked`),
other parameters into row slices. No parameter is gathered.

``rename`` maps a module parameter name to the checkpoint name of the layout
(default: identity; e.g. strip a wrapper prefix). A replicated parameter (a
plain tensor or a ``Replicate`` DTensor) is given by mesh coordinate 0 only.

Limits (explicit errors): DTensors on a mesh of more than one dimension, or
with a placement other than ``Shard(0)`` / ``Replicate``. Status: CPU tests with
real DTensors on 2 gloo ranks; not run inside verl.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import torch

from yeto.rl.quant.shards import SourceSlice
from yeto.rl.quant.stacked import STACKED_EXPERTS, unstack_experts


def shard0_rows(rows: int, coordinate: int, world: int) -> tuple[int, int]:
    """The rows ``[start, stop)`` a ``Shard(0)`` coordinate holds (``torch.chunk`` split)."""
    chunk = -(-int(rows) // int(world)) if rows else 0
    start = min(int(coordinate) * chunk, rows)
    return start, min(start + chunk, rows)


def fsdp2_slices(named_params: Iterable[tuple[str, torch.Tensor]], *, rank: int = 0,
                 rename: Callable[[str], str] = lambda name: name) -> Iterator[SourceSlice]:
    """Source slices of the parameters this rank holds. ``rank`` decides who gives
    plain (non-DTensor) tensors: rank 0 only."""
    from torch.distributed.tensor import DTensor, Replicate, Shard

    for module_name, param in named_params:
        name = rename(module_name)
        if isinstance(param, DTensor):
            mesh, placements = param.device_mesh, tuple(param.placements)
            if mesh.ndim != 1:
                raise ValueError(f"{module_name}: {mesh.ndim}-D device mesh is not supported (FSDP2 uses 1-D)")
            coordinate = mesh.get_local_rank()
            if placements == (Replicate(),):
                if coordinate == 0:
                    yield from _slices(name, param.to_local().detach(), 0)
                continue
            if placements != (Shard(0),):
                raise ValueError(f"{module_name}: placement {placements} is not supported (Shard(0) or Replicate)")
            local = param.to_local().detach()
            start, stop = shard0_rows(param.shape[0], coordinate, mesh.size())
            if local.shape[0] != stop - start:
                raise ValueError(f"{module_name}: local shard has {local.shape[0]} rows, expected {stop - start}")
            if stop > start:
                yield from _slices(name, local, start)
        elif int(rank) == 0:
            yield from _slices(name, param.detach(), 0)


def _slices(name: str, tensor: torch.Tensor, row_start: int) -> Iterator[SourceSlice]:
    if STACKED_EXPERTS.match(name):
        for expert_name, matrix in unstack_experts(name, tensor, first_expert=row_start):
            yield SourceSlice(expert_name, matrix)
    else:
        yield SourceSlice(name, tensor, row_start)
