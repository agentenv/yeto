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
import math
import time
from collections.abc import Iterable, Mapping, Sequence
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


_TRACED = False


def _trace(step: str) -> None:
    """One line per step of the first manifest, with Python fault frames armed.

    S27 G-FL: under PP > 1 the first manifest is also the first cross-stage collective
    (Megatron-Bridge gathers the global parameter names over the pipeline group), and a
    crash there reaches the log as Ray's C++ fiber stack with no Python frame. The lines
    below say how far the rank got; ``faulthandler`` adds the Python stack of every
    thread if a fatal signal arrives."""

    global _TRACED

    if _TRACED:
        return
    import faulthandler

    try:
        faulthandler.enable()
    except (ValueError, OSError):  # a closed or redirected stderr
        pass
    print(f"[yeto-megatron-boundary] rank {_rank()} manifest: {step}", flush=True)


def _pp_probe() -> None:
    """S27 G-FL diagnostic, off unless ``YETO_MEGATRON_PP_PROBE=1``.

    Runs the collective Megatron-Bridge runs first under PP > 1 -- an object all-gather
    over the pipeline group (``_megatron_global_param_names_all_pp_ranks``) -- on its
    own, so a crash there is attributed without Bridge in the frame."""

    import os

    if os.environ.get("YETO_MEGATRON_PP_PROBE") != "1":
        return
    import torch.distributed as dist
    from megatron.core import parallel_state as mpu

    rank = _rank()
    group = mpu.get_pipeline_model_parallel_group()
    size = int(dist.get_world_size(group=group))
    print(f"[yeto-pp-probe] rank {rank} pp_group size={size} backend={dist.get_backend(group)} "
          f"pp_rank={mpu.get_pipeline_model_parallel_rank()}", flush=True)
    gathered: list = [None] * size
    dist.all_gather_object(gathered, f"rank-{rank}", group=group)
    print(f"[yeto-pp-probe] rank {rank} all_gather_object ok -> {gathered}", flush=True)
    dist.barrier(group=group)
    print(f"[yeto-pp-probe] rank {rank} barrier ok", flush=True)


def router_bias_dump(chunks: Sequence[Any], tag: str) -> None:
    """S27 G-FT diagnostic, off unless ``YETO_MEGATRON_ROUTER_BIAS_DUMP`` names a path.

    The MoE router bias (``mlp.router.expert_bias``) is a persistent buffer, so no
    optimizer master backs it and the full-parameter layout does not hold it
    (:func:`layout_map`).  Whether leaving it outside the outer sync costs anything
    depends on one fact -- does training change it? -- and that needs a run that
    reaches a training step.  This writes a per-rank summary of every such buffer
    (no tensor leaves the card beyond a scalar reduction), so the same run answers it."""

    import json
    import os

    path = os.environ.get("YETO_MEGATRON_ROUTER_BIAS_DUMP")
    if not path:
        return
    import torch

    rows = []
    with torch.no_grad():
        for chunk in chunks:
            for name, buffer in chunk.named_buffers():
                if "expert_bias" not in name:
                    continue
                values = buffer.detach().to(dtype=torch.float32)
                rows.append({"name": str(name), "dtype": str(buffer.dtype), "numel": int(values.numel()),
                             "sum": float(values.sum().item()), "absmax": float(values.abs().max().item()),
                             "l2": float(values.norm().item()),
                             "head": [float(v) for v in values.reshape(-1)[:4].tolist()]})
    try:
        with open(f"{path}-rank{_rank()}-{tag}.json", "w") as handle:
            json.dump(rows, handle)
    except OSError as error:  # a diagnostic must not fail the island
        print(f"[yeto-megatron-boundary] router bias dump failed: {error}", flush=True)
        return
    print(f"[yeto-megatron-boundary] rank {_rank()} router bias dump {tag}: {len(rows)} buffer(s)", flush=True)


def remote_megatron_manifest(worker_dict) -> dict:
    """This rank's conversion tasks and master dtypes (no tensor moves)."""

    global _TRACED

    _trace("engine")
    bridge, chunks, optimizer = _engine(worker_dict)
    _pp_probe()
    router_bias_dump(chunks, "manifest")
    _trace(f"conversion tasks ({len(chunks)} chunk(s))")
    record = boundary.manifest(bridge, chunks, optimizer)
    _trace(f"built {len(record['tasks'])} task(s), {len(record['refused'])} refused")
    _TRACED = True
    record["rank"] = _rank()
    return record


def remote_megatron_export_flat(worker_dict, indices: Sequence[int], units: Sequence[Sequence], numel: int,
                                want_payload: bool, want_hash: bool,
                                layout: boundary.LayoutMap | None = None) -> dict:
    """One fragment's flat FP32 payload (``full_parameter.remote_export_flat`` on
    Megatron): every rank gathers the FP32 masters of tasks ``indices`` into whole
    HF tensors (collective); rank 0 copies the fragment's units, checks them
    finite and hashes them (:class:`.full_parameter.FlatFragment`).

    ``layout`` translates Bridge's HF names into the layout's names and offsets
    (S27 G-FT, blocker B15): a per-expert tensor lands at its row's offset
    inside the stacked layout tensor, so the fragment buffer does the stacking
    (:class:`yeto.rl.megatron_hf_boundary.LayoutSink`)."""

    import torch

    started = time.perf_counter()
    fragment = fp.FlatFragment(units, numel, reader=fp._is_reader())
    bridge, chunks, optimizer = _engine(worker_dict)
    router_bias_dump(chunks, "export")
    sink = boundary.LayoutSink(layout or boundary.IDENTITY, fragment.add_part, want=set(fragment.names))
    with torch.no_grad():
        boundary.export_hf(bridge, chunks, optimizer, [int(i) for i in indices], sink)
    gather_s = time.perf_counter() - started - fragment.copy_s
    return fragment.result({"gather_s": gather_s, "kept": sink.kept, "dropped": sink.dropped},
                           want_payload=want_payload, want_hash=want_hash, started=started)


def bridge_inputs(layout: boundary.LayoutMap | None, whole: Mapping[str, Any]) -> dict[str, Any]:
    """``whole`` (whole layout tensors) under Bridge's own HF names.

    One row of a stacked layout tensor is a view of it, so this costs no copy
    (S27 G-FT).  Without a map the two namespaces are the same."""

    if layout is None or not layout.slots:
        return dict(whole)
    out: dict[str, Any] = {}
    for bridge_name, slot in layout.slots.items():
        if slot is None or slot.name not in whole:
            continue
        tensor = whole[slot.name]
        out[str(bridge_name)] = tensor if slot.row is None else tensor[int(slot.row)]
    return out


def remote_megatron_apply(worker_dict, arrays: Mapping[str, Any], write_indices: Sequence[int],
                          read_indices: Sequence[int],
                          layout: boundary.LayoutMap | None = None) -> dict:
    """Write whole FP32 HF tensors into every rank's master shard and model copy
    (collective; :func:`yeto.rl.megatron_hf_boundary.apply_hf`).

    ``arrays`` are named by the layout, ``apply_hf`` reads Bridge's names, so
    ``layout`` translates between them (:func:`bridge_inputs`)."""

    import torch

    bridge, chunks, optimizer = _engine(worker_dict)
    device = next(chunks[0].parameters()).device
    whole = {name: fp._as_tensor(value).to(device=device, dtype=torch.float32) for name, value in arrays.items()}
    tensors = bridge_inputs(layout, whole)
    started = time.perf_counter()
    with torch.no_grad():
        result = boundary.apply_hf(bridge, chunks, optimizer, tensors, [int(i) for i in write_indices],
                                   [int(i) for i in read_indices])
    result["seconds"] = round(time.perf_counter() - started, 4)
    return result


# ----------------------------------------------------------------------------- island side
def layout_map(model_path: str | None, bridge_names: Iterable[str],
               shapes: Mapping[str, Sequence[int]]) -> boundary.LayoutMap:
    """Where Megatron-Bridge's HF names go in the island's layout (S27 G-FT, B15).

    Megatron-Bridge names the tensors of the *base checkpoint*; the layout names
    the HF model's own trainable parameters.  For Qwen3 the two are the same set
    and this returns :data:`~yeto.rl.megatron_hf_boundary.IDENTITY` without
    building anything.  For NemotronH they are not: ``backbone.`` against
    ``model.``, and 128 per-expert checkpoint tensors against one layout tensor
    stacked along dim 0.  ``transformers`` itself owns that translation, and
    :mod:`.hf_checkpoint_keys` runs the loader's own key loop over the names
    (no tensor read, meta-device build from ``config.json``), so no rule is
    written twice and no model family is named here.

    A Bridge name whose target is in the model's ``state_dict`` but not in the
    layout gets slot None: the layout is ``named_parameters()`` with
    ``requires_grad``, and Bridge also exports the persistent buffers (NemotronH:
    23 f32 router biases).  Those are left out of the boundary, the way the FSDP2
    island's layout already leaves them out, so a Megatron island and an FSDP2
    island of one model keep one layout hash (module doc)."""

    names = sorted({str(name) for name in bridge_names})
    if model_path is None or set(names) <= set(shapes):
        return boundary.IDENTITY
    from . import hf_checkpoint_keys as keys

    rules, prefix, state_names = keys.model_state_conversion(model_path)
    key_map = keys.checkpoint_key_map(names, state_names, rules=rules, prefix=prefix)
    slots: dict[str, boundary.LayoutSlot | None] = {}
    for name, key in key_map.keys.items():
        slots[str(name)] = (boundary.LayoutSlot(str(key.model_name), key.index)
                            if str(key.model_name) in shapes else None)
    row_numel: dict[str, int] = {}
    for slot in slots.values():
        if slot is None or slot.row is None or slot.name in row_numel:
            continue
        row_numel[slot.name] = math.prod(int(d) for d in shapes[slot.name][1:])
    return boundary.LayoutMap(slots, row_numel)


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

    def __init__(self, island: Any, *, shapes: Mapping[str, Sequence[int]], config: Mapping[str, Any],
                 model_path: str | None = None) -> None:
        super().__init__(island)
        self.shapes = {str(name): tuple(int(d) for d in shape) for name, shape in shapes.items()}
        self.config = dict(config)
        self.model_path = model_path
        self.index: boundary.HfTaskIndex | None = None

    async def manifests(self) -> list[dict]:
        """Index every rank's conversion tasks against the HF shapes (refused
        unless Bridge's names cover exactly these, through the loader's own
        conversion rules), check the hybrid rules and return the shapes as one
        manifest (the adapter plans the layout from it)."""

        records = await asyncio.to_thread(self.island.call_workers, remote_megatron_manifest)
        bridge_names = {str(name) for record in records for _i, hf_names, _d in record["tasks"]
                        for name in hf_names}
        layout = layout_map(self.model_path, bridge_names, self.shapes)
        self.index = boundary.HfTaskIndex.from_manifests(records, self.shapes, layout=layout)
        self.index.check_stack_heights(self.shapes)
        check_hybrid_rules(self.shapes, self.index.master_dtypes(records), self.config)
        print(f"[yeto-megatron-boundary] island index: {len(self.index.names)} task(s) -> "
              f"{len(self.index.by_name)} layout name(s), {len(self.index.stacked)} stacked, "
              f"{sum(1 for slot in layout.slots.values() if slot is None)} name(s) outside the layout",
              flush=True)
        return [{"shapes": {name: list(shape) for name, shape in self.shapes.items()}}]

    def _task_index(self) -> boundary.HfTaskIndex:
        if self.index is None:
            raise VerlMegatronSyncError("the Megatron group reads no fragment before its manifest")
        return self.index

    def _export_on_workers(self, units: list, numel: int, payload: bool, sha256: bool) -> list[dict]:
        index = self._task_index()
        indices = index.tasks_for({str(unit[0]) for unit in units})
        return self.island.call_workers(remote_megatron_export_flat, indices, units, numel, payload, sha256,
                                        index.layout_for(indices))

    def _apply_on_workers(self, arrays: Mapping[str, Any]) -> list[dict]:
        index = self._task_index()
        write, read = index.apply_plan(arrays)
        return self.island.call_workers_shared(remote_megatron_apply, arrays, write, read,
                                               index.layout_for([*write, *read]))

    async def export_parameters(self, names: Sequence[str]):
        raise VerlMegatronSyncError("the Megatron island exports HF files with Megatron-Bridge "
                                    "(megatron_ports.remote_hf_export) and has no peer recovery")


def megatron_group(island: Any, model_path: str) -> MegatronFullParameterGroup:
    """The group of a synced Megatron island: shapes and config of ``model_path``."""

    import json
    from pathlib import Path

    from .full_parameter_plan import model_shapes

    config = json.loads((Path(model_path) / "config.json").read_text())
    return MegatronFullParameterGroup(island, shapes=model_shapes(model_path), config=config,
                                      model_path=str(model_path))
