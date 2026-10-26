"""Full-parameter outer sync of the verl Megatron engine island (S27 G-FG).

The verl Megatron island (:mod:`.megatron_ports`, S27 G-DC) trains with
verl's Megatron engine (TP/PP/EP, DistributedOptimizer).  This module lets it
join a full-parameter syncer session the way an FSDP2 island does: the
engine-neutral role stream, the verl chunked adapter, the cuts, the lazy
capture, local INIT, bf16 broadcast deltas and the final cut are the FSDP2
island's, unchanged (:mod:`.full_parameter_stream`).  Only the trainer group
differs: :class:`MegatronFullParameterGroup` reads and writes the trainer
through the backend-neutral Megatron HF boundary
(:mod:`yeto.rl.megatron_hf_boundary`) instead of FSDP2 ``DTensor`` gathers.

The layout is the FSDP2 island's: the HF model's parameters (tied weights
once) as FP32 specs, planned by ``full_parameter.plan_layout``.  So a Megatron
island and an FSDP2 island of one model have the same layout hash, and the
launcher's planner (``full_parameter_plan``) gives it for both.  They still do
not share one session: the backend identity (parameter-name map) differs.

What a Megatron island reads is the FP32 optimizer master of every
parameter, not the bf16 model copy; an apply writes the master and the model
copy, keeps the Adam moments, and the adapter proves it by reading the
fragments back and comparing their hashes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl import megatron_hf_boundary as boundary

from . import full_parameter as fp
from .full_parameter_stream import VerlFullParameterGroup


class VerlMegatronSyncError(RuntimeError):
    """The verl Megatron island cannot run the full-parameter sync as configured."""


# ----------------------------------------------------------------------------- worker side
def _engine(worker_dict) -> tuple[Any, list, Any]:
    """``(bridge, model chunks, optimizer)`` of a verl Megatron actor worker."""

    from .ports_impl import _inner_worker

    engine = _inner_worker(worker_dict).actor.engine
    offload = [name for name in ("is_param_offload_enabled", "is_optimizer_offload_enabled")
               if callable(getattr(engine, name, None)) and getattr(engine, name)()]
    if offload:
        raise VerlMegatronSyncError(f"the full-parameter sync reads the trainer on its cards; offload is on "
                                    f"({', '.join(offload)})")
    module = engine.module
    chunks = list(module) if isinstance(module, (list, tuple)) else [module]
    return engine.bridge, chunks, engine.optimizer


def _rank() -> int:
    import torch.distributed as dist

    return int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0


def remote_megatron_manifest(worker_dict) -> dict:
    """This rank's conversion tasks and master dtypes (no tensor moves)."""

    bridge, chunks, optimizer = _engine(worker_dict)
    record = boundary.manifest(bridge, chunks, optimizer)
    record["rank"] = _rank()
    return record


def remote_megatron_export_flat(worker_dict, indices: Sequence[int], units: Sequence[Sequence], numel: int,
                                want_payload: bool, want_hash: bool) -> dict:
    """One fragment's flat FP32 payload (``full_parameter.remote_export_flat`` on
    Megatron): every rank gathers the FP32 masters of tasks ``indices`` into whole
    HF tensors (collective); rank 0 copies the fragment's units, checks them
    finite and hashes them (:class:`.full_parameter.FlatFragment`)."""

    import torch

    started = time.perf_counter()
    fragment = fp.FlatFragment(units, numel, reader=fp._is_reader())
    wanted = set(fragment.names)
    bridge, chunks, optimizer = _engine(worker_dict)

    def keep(name: str, tensor) -> None:
        if name in wanted:
            fragment.add(name, tensor)

    with torch.no_grad():
        boundary.export_hf(bridge, chunks, optimizer, [int(i) for i in indices], keep)
    gather_s = time.perf_counter() - started - fragment.copy_s
    return fragment.result({"gather_s": gather_s}, want_payload=want_payload, want_hash=want_hash,
                           started=started)


def remote_megatron_apply(worker_dict, arrays: Mapping[str, Any], write_indices: Sequence[int],
                          read_indices: Sequence[int]) -> dict:
    """Write whole FP32 HF tensors into every rank's master shard and model copy
    (collective; :func:`yeto.rl.megatron_hf_boundary.apply_hf`)."""

    import torch

    bridge, chunks, optimizer = _engine(worker_dict)
    device = next(chunks[0].parameters()).device
    tensors = {name: fp._as_tensor(value).to(device=device, dtype=torch.float32) for name, value in arrays.items()}
    started = time.perf_counter()
    with torch.no_grad():
        result = boundary.apply_hf(bridge, chunks, optimizer, tensors, [int(i) for i in write_indices],
                                   [int(i) for i in read_indices])
    result["seconds"] = round(time.perf_counter() - started, 4)
    return result


# ----------------------------------------------------------------------------- island side
def check_hybrid_rules(shapes: Mapping[str, Sequence[int]], master_dtypes: Mapping[str, str],
                       config: Mapping[str, Any]) -> None:
    """The hybrid-model rules (:mod:`yeto.rl.hybrid_params`) on the boundary:
    no frozen vision parameter in it, every f32-required tensor read as f32
    (Megatron reads FP32 masters, so this holds by construction), and every
    routed expert present when the model has routed experts."""

    from yeto.rl import hybrid_params

    hybrid_params.check_frozen_excluded(shapes)
    hybrid_params.check_f32(master_dtypes)
    llm = hybrid_params.language_model_config(config)
    experts = llm.get("n_routed_experts") or llm.get("num_experts")
    if experts:
        hybrid_params.check_expert_coverage(shapes, num_experts=int(experts))


class MegatronFullParameterGroup(VerlFullParameterGroup):
    """The trainer group of a verl Megatron island, as the role stream sees it:
    the FSDP2 group's records and receipts, with the worker reads and writes of
    the Megatron HF boundary.

    ``shapes`` are the boundary's HF parameter shapes (``full_parameter_plan.model_shapes``
    of the model, the FSDP2 island's); ``config`` is the model's ``config.json``."""

    def __init__(self, island: Any, *, shapes: Mapping[str, Sequence[int]], config: Mapping[str, Any]) -> None:
        super().__init__(island)
        self.shapes = {str(name): tuple(int(d) for d in shape) for name, shape in shapes.items()}
        self.config = dict(config)
        self.index: boundary.HfTaskIndex | None = None

    async def manifests(self) -> list[dict]:
        """Index every rank's conversion tasks against the HF shapes (refused
        unless Bridge exports exactly these names), check the hybrid rules and
        return the shapes as one manifest (the adapter plans the layout from it)."""

        records = await asyncio.to_thread(self.island.call_workers, remote_megatron_manifest)
        self.index = boundary.HfTaskIndex.from_manifests(records, self.shapes)
        check_hybrid_rules(self.shapes, self.index.master_dtypes(records), self.config)
        return [{"shapes": {name: list(shape) for name, shape in self.shapes.items()}}]

    def _task_index(self) -> boundary.HfTaskIndex:
        if self.index is None:
            raise VerlMegatronSyncError("the Megatron group reads no fragment before its manifest")
        return self.index

    def _export_on_workers(self, units: list, numel: int, payload: bool, sha256: bool) -> list[dict]:
        indices = self._task_index().tasks_for({str(unit[0]) for unit in units})
        return self.island.call_workers(remote_megatron_export_flat, indices, units, numel, payload, sha256)

    def _apply_on_workers(self, arrays: Mapping[str, Any]) -> list[dict]:
        write, read = self._task_index().apply_plan(arrays)
        return self.island.call_workers_shared(remote_megatron_apply, arrays, write, read)

    async def export_parameters(self, names: Sequence[str]):
        raise VerlMegatronSyncError("the Megatron island exports HF files with Megatron-Bridge "
                                    "(megatron_ports.remote_hf_export) and has no peer recovery")


def megatron_group(island: Any, model_path: str) -> MegatronFullParameterGroup:
    """The group of a synced Megatron island: shapes and config of ``model_path``."""

    import json
    from pathlib import Path

    from .full_parameter_plan import model_shapes

    config = json.loads((Path(model_path) / "config.json").read_text())
    return MegatronFullParameterGroup(island, shapes=model_shapes(model_path), config=config)
