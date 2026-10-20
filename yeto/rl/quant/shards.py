"""Distributed re-quantization: each trainer rank quantizes the weights it holds (S25 N8.2, option R3).

No rank gathers the whole BF16 model. Each rank gives the source tensors it
holds as :class:`SourceSlice` items (per-expert source names of the target
layout, see :mod:`.stacked`; a slice is a range of rows) and writes one part
file with the quantized result of those rows (:func:`write_part`). Then the
parts are merged into the output shards of the layout (:func:`merge_parts`),
each rank merging a disjoint subset of the shards, and the index and metadata
are written last (:func:`~.requant.finalize_checkpoint`).

Why row slices work: both schemes quantize each output row on its own (W4A8:
one scale per row; MXFP4: one scale per 32 inputs of a row), AWQ balance
factors scale columns, and casts work per element. So the rows of a slice
quantize to the same rows as in the whole tensor, and the merged shard is byte
for byte the file :func:`~.requant.requantize_checkpoint` writes from the whole
source (both write through :func:`~.requant.write_output_shard`). Limits: an AWQ
smooth layer (1-D) must be given whole; 0-D tensors must be given whole.

Ownership: every row of every source tensor must come from exactly one rank;
the merge fails on a gap or an overlap. Replicated parameters (for example the
non-expert weights on every expert-parallel rank of Megatron) must be given by
one rank only; the trainer adapters decide which.

Part file: ``<part_dir>/part-<rank>.safetensors``; tensor key
``<target name>@<first row>``.

Status: CPU tests (2 gloo ranks, both schemes, byte-equal to the single-process
output; tests/test_rl_quant_shards.py). Inside a real Miles trainer (2xH100,
glm5_next 4-layer mini, TP2/EP2, through the Miles adapter): byte-equal for both
schemes (T-G3, 2026-10-20). Not run inside verl.
"""

from __future__ import annotations

import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .awq import AwqSmoothing
from .requant import (
    CheckpointLayout,
    LayoutError,
    check_smoothing,
    finalize_checkpoint,
    read_header,
    transform_tensor,
    write_output_shard,
)

PART_DIR = ".requant-parts"
_KEY_SEP = "@"


@dataclass(frozen=True)
class SourceSlice:
    """Rows ``[row_start, row_start + len(tensor))`` of the source tensor ``name``."""

    name: str
    tensor: torch.Tensor
    row_start: int = 0


def part_path(part_dir: Path, rank: int) -> Path:
    return Path(part_dir) / f"part-{int(rank):05d}.safetensors"


def write_part(part_dir: Path, rank: int, slices: Iterable[SourceSlice], template_dir: Path,
               smoothing: AwqSmoothing) -> dict[str, int]:
    """Quantize this rank's slices and write its part file; returns counts."""
    layout = CheckpointLayout.load(template_dir)
    check_smoothing(layout, smoothing)
    out: dict[str, torch.Tensor] = {}
    sources = 0
    for item in slices:
        start = int(item.row_start)
        if start < 0 or (item.tensor.dim() == 0 and start):
            raise LayoutError(f"{item.name}: bad first row {start}")
        for target, tensor in transform_tensor(item.name, item.tensor, layout, smoothing).items():
            spec = layout.tensors[target]
            if tensor.dim() != len(spec.shape) or tuple(tensor.shape[1:]) != spec.shape[1:] \
                    or (tensor.dim() and start + tensor.shape[0] > spec.shape[0]):
                raise LayoutError(f"{target}: slice {tuple(tensor.shape)} at row {start} "
                                  f"does not fit the layout shape {spec.shape}")
            key = f"{target}{_KEY_SEP}{start}"
            if key in out:
                raise LayoutError(f"{target}: rows from {start} given twice on rank {rank}")
            # A cast to the same dtype returns the caller's tensor, possibly a view
            # that shares storage with another slice; safetensors refuses shared storage.
            out[key] = tensor.clone(memory_format=torch.contiguous_format)
        sources += 1
    path = part_path(part_dir, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(out, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, path)
    return {"sources": sources, "tensors": len(out)}


def _part_index(part_dir: Path) -> dict[str, list[tuple[int, Path, str]]]:
    """target -> [(first row, part file, key)] over every part file, sorted by row."""
    index: dict[str, list[tuple[int, Path, str]]] = defaultdict(list)
    for path in sorted(Path(part_dir).glob("part-*.safetensors")):
        for key in read_header(path):
            target, _, start = key.rpartition(_KEY_SEP)
            index[target].append((int(start), path, key))
    return {t: sorted(v, key=lambda x: x[0]) for t, v in index.items()}


def merge_parts(part_dir: Path, template_dir: Path, out_dir: Path,
                shards: Iterable[str] | None = None) -> dict[str, int]:
    """Write the given output shards (default: all) from the part files.

    Fails when a tensor of these shards has a gap, an overlap, or a part names a
    tensor the layout does not have."""
    layout = CheckpointLayout.load(template_dir)
    all_shards = layout.shards()
    selected = list(all_shards) if shards is None else list(shards)
    if unknown := set(selected) - set(all_shards):
        raise LayoutError(f"unknown output shards {sorted(unknown)}")
    index = _part_index(part_dir)
    if stray := sorted(set(index) - set(layout.tensors)):
        raise LayoutError(f"parts hold {len(stray)} tensors the layout does not have, e.g. {stray[:3]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[Path, Any] = {}
    stats = {"shards": 0, "tensors": 0, "slices": 0}
    try:
        for shard in selected:
            tensors: dict[str, torch.Tensor] = {}
            for name in all_shards[shard]:
                pieces = index.get(name)
                if not pieces:
                    raise LayoutError(f"{name}: no rank gave this tensor")
                read = []
                for start, path, key in pieces:
                    if path not in files:
                        files[path] = safe_open(str(path), framework="pt", device="cpu")
                    read.append((start, files[path].get_tensor(key)))
                tensors[name] = _join_rows(name, read, layout.tensors[name].shape)
                stats["slices"] += len(read)
            write_output_shard(out_dir, shard, tensors, layout)
            stats["shards"] += 1
            stats["tensors"] += len(tensors)
    finally:
        files.clear()
    return stats


def _join_rows(name: str, pieces: list[tuple[int, torch.Tensor]], shape: tuple[int, ...]) -> torch.Tensor:
    if not shape:
        if len(pieces) != 1:
            raise LayoutError(f"{name}: scalar given {len(pieces)} times")
        return pieces[0][1]
    row = 0
    for start, piece in pieces:
        if start != row:
            kind = "overlap" if start < row else "gap"
            raise LayoutError(f"{name}: rows {min(start, row)}..{max(start, row)} are an {kind} between ranks")
        row += piece.shape[0]
    if row != shape[0]:
        raise LayoutError(f"{name}: rows {row}..{shape[0]} are a gap (no rank gave them)")
    return pieces[0][1] if len(pieces) == 1 else torch.cat([p for _, p in pieces], dim=0)


def requantize_distributed(slices: Iterable[SourceSlice], template_dir: Path, out_dir: Path,
                           smoothing: AwqSmoothing, *, rank: int, world_size: int,
                           barrier: Callable[[], None]) -> dict[str, Any]:
    """Every rank calls this with its own slices; the complete checkpoint is in
    ``out_dir`` when it returns on any rank.

    ``out_dir`` must be on storage all ranks share (the release directory).
    ``barrier()`` synchronizes the ranks (:func:`torch_barrier` under
    ``torch.distributed``). Steps: write the part; barrier; merge shards
    ``rank::world_size``; barrier; rank 0 writes the index and metadata and
    removes the parts; barrier."""
    rank, world_size = int(rank), int(world_size)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} not in a world of {world_size}")
    template_dir, out_dir = Path(template_dir), Path(out_dir)
    part_dir = out_dir / PART_DIR
    stats: dict[str, Any] = {"part": write_part(part_dir, rank, slices, template_dir, smoothing)}
    barrier()
    shards = list(CheckpointLayout.load(template_dir).shards())[rank::world_size]
    stats["merge"] = merge_parts(part_dir, template_dir, out_dir, shards)
    barrier()
    if rank == 0:
        finalize_checkpoint(template_dir, out_dir)
        shutil.rmtree(part_dir)
    barrier()
    return stats


def torch_barrier() -> None:
    """``torch.distributed.barrier`` of the default group (the trainer's)."""
    import torch.distributed as dist

    dist.barrier()
