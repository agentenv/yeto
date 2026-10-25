"""Full-parameter boundary of a verl FSDP2 island (S27 B10, phase 1).

The LoRA island (:mod:`.ports_impl`) exports adapter tensors from one card.
A full-parameter island trains every parameter of the model on ``n_gpus``
cards with FSDP2 data parallelism, and the synchronization boundary is the
whole model instead of an adapter.

What this module provides:

* the island's canonical parameter names (:func:`base_to_canonical`): the HF
  parameter names of the unwrapped module, so the names do not depend on how
  FSDP2 wrapped the model;
* the FP32 parameter specs of the boundary (:func:`island_specs`) and the
  owner-affine fragment layout over them (:func:`plan_layout`), both built
  with the engine-neutral planner (:mod:`yeto.rl.local_learner`) that the
  Miles islands use, so one syncer profile covers both backends' fragments;
* the worker-side operations a fragment needs
  (:func:`remote_manifest`, :func:`remote_export_fragment`,
  :func:`remote_export_flat`, :func:`remote_apply_fragment`), run on every
  actor worker through ``VerlIsland.call_workers``.  Since S27 G-EC a
  fragment is read with :func:`remote_export_flat` (rank 0 builds the flat
  payload, checks it finite on the device, hashes it) and every payload
  crosses Ray as a numpy array, not a torch tensor (Ray pickles torch tensors
  through ``torch.save``; numpy goes out of band).

Two properties of the verl FSDP2 boundary, both different from Miles:

* **FP32 parameters are their own masters.** The island runs
  ``actor.fsdp_config.model_dtype=fp32``, so the parameter FSDP2 shards *is*
  the optimizer master; there is no separate FP32 copy to read (Miles reads
  Megatron's ``main_param``). :func:`remote_manifest` refuses any other
  parameter dtype instead of exporting bf16 values as if they were masters.
* **The layout does not depend on the card count.** Every FSDP2 parameter is
  a ``DTensor`` that any rank can gather whole (``full_tensor()``), so the
  specs carry no shard identity (``shard_id`` stays ``"global"``) and a 2-card
  and a 4-card island of the same model produce the same layout hash. A Miles
  layout instead names every rank's TP/PP/EP/CP/DP shard, so its hash changes
  with the parallel shape (``full_parameter_state.FullParameterShardSpec``).

The fragment-streaming role stream that answers the unified syncer's PULL
permits is the engine-neutral ``yeto.rl.engine.full_parameter_stream``; its
verl binding (:mod:`.full_parameter_stream`) calls the worker operations here
through ``VerlIsland.call_workers``.

Every export and apply is collective: all ranks must call it with the same
fragment, because ``DTensor.full_tensor`` and ``distribute_tensor`` are
collectives over the device mesh.

Two additions for the HF export (S27 G-EQ3):

* the manifest also lists the module's **persistent buffers** (``buffers``),
  and :func:`remote_export_fragment` reads them by name.  A buffer is part of
  the checkpoint (a MoE router's ``e_score_correction_bias`` is one: the
  loader fills it, no optimizer updates it); an export without it loads with
  the buffer's initial value.  The sync layout stays parameters only.
* a parameter whose local shard lives on the host (FSDP2 ``CPUOffloadPolicy``,
  ``--rl-fsdp-offload``) is moved to the mesh's device before the gather
  (:func:`_whole`, as ``torch.distributed._state_dict_utils`` does), so the
  all-gather runs on the accelerator's collective backend, not on the host.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from yeto.rl.local_learner import ComponentIdentity, ParameterLayout, ParameterSpec, fit_owner_affine_layout

# FSDP2 / activation-checkpoint wrappers add these segments to parameter names.
WRAPPER_SEGMENTS = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")
# The dtype a full-parameter boundary reads (``fsdp_config.model_dtype=fp32``).
MASTER_DTYPE = "torch.float32"
# One fragment's FP32 bytes; the syncer frame bound of the legacy syncer.
DEFAULT_MAX_FRAGMENT_BYTES = 2 << 30


class VerlFullParameterError(RuntimeError):
    """The verl island cannot export or apply a full-parameter boundary."""


def base_to_canonical(name: str) -> str:
    """FSDP2 parameter name -> canonical name (the HF name of the module).

    LoRA names belong to :mod:`.param_names`; a full-parameter island has no
    adapter, so an adapter name here means the island was built in LoRA mode.
    """

    if ".lora_" in name:
        raise VerlFullParameterError(
            f"LoRA 参数出现在全参数边界上: {name!r}（全参数岛必须 lora_rank=0）")
    canonical = name
    for segment in WRAPPER_SEGMENTS:
        canonical = canonical.replace(segment, "")
    if not canonical or canonical.startswith(".") or ".." in canonical:
        raise VerlFullParameterError(f"无法规范化的参数名: {name!r}")
    return canonical


def island_specs(shapes: Mapping[str, Sequence[int]], *, role: str = "actor") -> tuple[ParameterSpec, ...]:
    """The boundary's FP32 specs, sorted, from ``{canonical name: shape}``."""

    if not shapes:
        raise VerlFullParameterError("全参数边界没有任何参数")
    specs = tuple(
        ParameterSpec(role, name, tuple(int(dimension) for dimension in shape), "float32",
                      _numel(shape))
        for name, shape in shapes.items()
    )
    return tuple(sorted(specs))


def plan_layout(
    specs: Sequence[ParameterSpec],
    *,
    component: ComponentIdentity,
    minimum_fragments: int,
    max_fragment_bytes: int = DEFAULT_MAX_FRAGMENT_BYTES,
    matrix_merge: str = "avg",
) -> ParameterLayout:
    """The island's owner-affine fragment layout (the neutral planner, GRPO)."""

    return fit_owner_affine_layout(
        algorithm="grpo",
        components=(component,),
        specs=tuple(specs),
        minimum_fragments=minimum_fragments,
        max_fragment_bytes=max_fragment_bytes,
        matrix_merge=matrix_merge,
    )


def fragment_tensor_names(layout: ParameterLayout, fragment_id: int) -> tuple[str, ...]:
    """Canonical parameter names of one fragment, in the layout's order.

    A fragment of a split parameter names that parameter once; the export
    reads the whole parameter and the layout slices it.
    """

    names: list[str] = []
    for unit in layout.fragment_slices(fragment_id):
        if unit.spec.name not in names:
            names.append(unit.spec.name)
    return tuple(names)


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dimension in shape:
        total *= int(dimension)
    return total


# ----------------------------------------------------------------------------- worker side
def _trainable_module(worker_dict):
    """The unwrapped training module of a verl actor worker."""

    from .ports_impl import _inner_worker

    module = _inner_worker(worker_dict).actor.engine.module
    return getattr(module, "_fsdp_wrapped_module", module)


def _named_full_parameters(module):
    """``[(canonical name, parameter)]`` of every trainable parameter, sorted."""

    named = [(base_to_canonical(name), parameter)
             for name, parameter in module.named_parameters()
             if parameter.requires_grad]
    if len({name for name, _parameter in named}) != len(named):
        raise VerlFullParameterError("规范化后出现重名参数")
    return sorted(named, key=lambda item: item[0])


def _named_persistent_buffers(module):
    """``[(canonical name, buffer)]`` of every persistent buffer (the ones in
    ``state_dict``), sorted; a buffer shared by two modules is listed once."""

    named, seen = [], set()
    for prefix, submodule in module.named_modules():
        skip = getattr(submodule, "_non_persistent_buffers_set", set())
        for name, buffer in submodule._buffers.items():
            if buffer is None or name in skip or id(buffer) in seen:
                continue
            seen.add(id(buffer))
            named.append((base_to_canonical(f"{prefix}.{name}" if prefix else name), buffer))
    if len({name for name, _buffer in named}) != len(named):
        raise VerlFullParameterError("规范化后出现重名 buffer")
    return sorted(named, key=lambda item: item[0])


def _exportable(module) -> dict:
    """Parameters and persistent buffers by canonical name (a name is never both)."""

    by_name = dict(_named_full_parameters(module))
    buffers = dict(_named_persistent_buffers(module))
    clash = sorted(set(by_name) & set(buffers))
    if clash:
        raise VerlFullParameterError(f"参数与 buffer 同名: {clash[:4]}")
    by_name.update(buffers)
    return by_name


def _whole(tensor):
    """The whole value of a parameter or buffer (collective for a ``DTensor``).
    A host-resident shard (FSDP2 CPU offload) goes to the mesh's device first,
    so the all-gather uses the device backend (module doc)."""

    if not hasattr(tensor, "full_tensor"):
        return tensor
    device_type = tensor.device_mesh.device_type
    if tensor.device.type != device_type:
        tensor = tensor.to(device_type)
    return tensor.full_tensor()


def remote_manifest(worker_dict) -> dict:
    """This worker's full-parameter manifest: shapes, dtype and mesh facts.

    Collective-free (it reads metadata only), so every rank returns the same
    shapes; the island compares them before planning a layout.
    """

    import torch.distributed as dist

    module = _trainable_module(worker_dict)
    named = _named_full_parameters(module)
    buffers = _named_persistent_buffers(module)
    dtypes = sorted({str(parameter.dtype) for _name, parameter in named})
    if dtypes != [MASTER_DTYPE]:
        raise VerlFullParameterError(
            f"全参数同步需要 FP32 主权重（actor.fsdp_config.model_dtype=fp32）；"
            f"看到的参数 dtype: {dtypes}")
    initialized = dist.is_available() and dist.is_initialized()
    return {
        "shapes": {name: list(_global_shape(parameter)) for name, parameter in named},
        "param_dtypes": dtypes,
        "buffers": {name: list(_global_shape(buffer)) for name, buffer in buffers},
        "buffer_dtypes": sorted({str(buffer.dtype) for _name, buffer in buffers}),
        # S27 G-EQ3: a fingerprint per buffer, so the island can see that every rank
        # holds the same buffer values (:func:`agree_on_buffers`)
        "buffer_digests": {name: _buffer_digest(buffer) for name, buffer in buffers},
        "rank": int(dist.get_rank()) if initialized else 0,
        "world_size": int(dist.get_world_size()) if initialized else 1,
        "sharded": any(hasattr(parameter, "device_mesh") for _name, parameter in named),
    }


def _buffer_digest(buffer) -> list[float]:
    """``[sum, sum of absolute values, number of values]`` of one buffer, as floats.

    Why (S27 G-EQ3): a persistent buffer is **not sharded**, so every rank keeps
    its own copy and each rank's forward reads its own. A rank whose copy is
    wrong computes wrong log-probs while the parameters are perfectly fine -- for
    a MoE model the router's correction bias decides which experts a token goes
    to. The values come from the checkpoint through rank 0 (verl's
    ``fsdp2_load_full_state_dict`` broadcasts every buffer after the parameters),
    and this fingerprint is how the island checks that it happened.
    """

    import torch

    with torch.no_grad():
        values = buffer.detach()
        if hasattr(values, "full_tensor"):  # a sharded buffer: its local part is enough here
            values = values.to_local() if hasattr(values, "to_local") else values
        values = values.to("cpu", torch.float64).reshape(-1)
        return [float(values.sum()), float(values.abs().sum()), float(values.numel())]


def _global_shape(parameter) -> tuple[int, ...]:
    """The parameter's whole shape (a ``DTensor`` reports it; a plain tensor is whole)."""

    return tuple(int(dimension) for dimension in parameter.shape)


def remote_export_fragment(worker_dict, names: Sequence[str], return_tensors: bool) -> dict:
    """Gather ``names`` whole on every rank; only ``return_tensors`` ranks keep them.

    Collective: ``DTensor.full_tensor`` all-gathers, so every rank calls it for
    every name in the same order. ``return_tensors`` is True on the one rank
    whose result the island reads, which keeps the fragment's bytes off the
    other ranks' Ray replies.
    """

    import torch

    module = _trainable_module(worker_dict)
    by_name = _exportable(module)
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise VerlFullParameterError(f"分片引用了模型里没有的参数: {missing[:4]}")
    tensors = {}
    with torch.no_grad():
        for name in names:  # the island's order, the same on every rank
            whole = _whole(by_name[name])
            if return_tensors:
                tensors[name] = whole.detach().to("cpu", torch.float32).clone()
            del whole
    return {"tensors": tensors, "names": list(names)} if return_tensors else {"names": list(names)}


def remote_export_for_reader(worker_dict, names: Sequence[str]) -> dict:
    """:func:`remote_export_fragment` where only rank 0 (the reader) keeps the tensors."""

    return remote_export_fragment(worker_dict, names, _is_reader())


def remote_export_arrays_for_reader(worker_dict, names: Sequence[str]) -> dict:
    """:func:`remote_export_for_reader` with the reader's tensors as numpy arrays
    (S27 G-EC): Ray moves a numpy array out of band into its object store,
    while a torch tensor goes through ``torch.save`` pickling (two extra
    copies of every byte at about 0.5 GB/s each way, measured on CPU)."""

    result = remote_export_fragment(worker_dict, names, _is_reader())
    if "tensors" in result:
        result["arrays"] = {name: tensor.numpy() for name, tensor in result.pop("tensors").items()}
    return result


def _is_reader() -> bool:
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def fragment_units(layout: ParameterLayout, fragment_id: int) -> tuple[tuple[str, int, int], ...]:
    """``(canonical name, flat start, flat stop)`` of every slice of one fragment,
    in the layout's order: the fragment's flat FP32 payload is these element
    ranges of the whole parameters, one after another."""

    return tuple((unit.spec.name, *unit.flat_range) for unit in layout.fragment_slices(fragment_id))


def remote_export_flat(worker_dict, units: Sequence[Sequence], numel: int, want_payload: bool,
                       want_hash: bool) -> dict:
    """One fragment's flat FP32 payload, built on the reading rank (S27 G-EC).

    Collective like :func:`remote_export_fragment`: every rank gathers each
    parameter of ``units`` whole, in the order the parameters first appear.
    Rank 0 (the reader) copies each unit's element range into one CPU FP32
    buffer in ``units`` order and checks on the parameter's device that every
    value is finite.  It returns ``finite`` and, when asked, the buffer as a
    numpy array (``want_payload``; Ray moves it without torch pickling) and
    its SHA256 (``want_hash``; the fragment's wire hash, so an apply proof
    needs no bytes back).  Every rank returns its seconds per step."""

    import hashlib
    import time

    import torch

    started = time.perf_counter()
    reader = _is_reader()
    module = _trainable_module(worker_dict)
    by_name = dict(_named_full_parameters(module))
    names = list(dict.fromkeys(str(unit[0]) for unit in units))
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise VerlFullParameterError(f"分片引用了模型里没有的参数: {missing[:4]}")
    offsets, offset = [], 0
    for _name, start, stop in units:
        offsets.append(offset)
        offset += int(stop) - int(start)
    if offset != int(numel):
        raise VerlFullParameterError(f"分片元素数不符: 切片合计 {offset}，布局是 {numel}")
    flat = torch.empty(int(numel), dtype=torch.float32) if reader else None
    finite = None
    gather_s = copy_s = 0.0
    with torch.no_grad():
        for name in names:  # the island's order, the same on every rank
            mark = time.perf_counter()
            whole = _whole(by_name[name])
            gather_s += time.perf_counter() - mark
            if reader:
                mark = time.perf_counter()
                values = whole.detach().reshape(-1)
                for (unit_name, start, stop), at in zip(units, offsets, strict=True):
                    if unit_name != name:
                        continue
                    part = values[int(start):int(stop)]
                    ok = torch.isfinite(part).all()
                    finite = ok if finite is None else finite & ok.to(finite.device)
                    flat[at:at + part.numel()].copy_(part)
                copy_s += time.perf_counter() - mark
            del whole
    timings = {"gather_s": gather_s, "copy_s": copy_s}
    if not reader:
        timings["total_s"] = time.perf_counter() - started
        return {"names": names, "timings": timings}
    result: dict = {"names": names, "finite": bool(finite.item()) if finite is not None else True}
    if want_hash:
        mark = time.perf_counter()
        result["sha256"] = hashlib.sha256(memoryview(flat.numpy()).cast("B")).hexdigest()
        timings["hash_s"] = time.perf_counter() - mark
    if want_payload:
        result["flat"] = flat.numpy()
    timings["total_s"] = time.perf_counter() - started
    result["timings"] = timings
    return result


def remote_release_memory(worker_dict, host: bool = True) -> dict:
    """Give the device allocator's cached blocks back (S27 G-EQ3).

    Why: a full export gathers every parameter whole on the device
    (:func:`_whole`), and the biggest ones are large -- a Nemotron Nano MoE
    layer's stacked expert tensor is 1.28e9 values. The allocator keeps those
    blocks cached, so the step *after* an export starts with a much smaller
    free pool: on npu1 the first step peaked at 63.1 of 65.5 GiB and the step
    after the first release died asking for 9.00 GiB. Returns the allocator's
    numbers before and after, for the island's tape.
    """

    del worker_dict  # the allocator is per process, not per parameter
    import torch

    from yeto.rl.engine import host_memory

    device = torch.npu if hasattr(torch, "npu") and torch.npu.is_available() else torch.cuda
    before = {"allocated": int(device.memory_allocated()), "reserved": int(device.memory_reserved())}
    device.synchronize()  # blocks still referenced by pending work are not freed
    device.empty_cache()
    result = {"before": before,
              "after": {"allocated": int(device.memory_allocated()), "reserved": int(device.memory_reserved())}}
    if host and host_memory.host_memory_mode() == "trim":  # S27 G-EZ3: the export's host copies too
        result["host"] = host_memory.release_host_memory()
    return result


def remote_host_memory(worker_dict, release: bool) -> dict:
    """This worker's host memory (S27 G-EZ3): its RSS by kind and its node's
    ``/proc/meminfo`` (``yeto.rl.engine.host_memory``), after a full garbage
    collection and ``malloc_trim(0)`` when ``release`` (the trim's own result
    is in ``release``). Not collective: each worker reads its own process."""

    del worker_dict  # per process
    import os
    import socket

    from yeto.rl.engine import host_memory

    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else None
    except Exception:  # noqa: BLE001 - a report, not a check
        rank = None
    out = {"host": socket.gethostname(), "pid": os.getpid(), "rank": rank}
    if release:
        out["release"] = host_memory.release_host_memory()
    out["process"] = host_memory.process_memory()
    out["node"] = host_memory.node_memory()
    return out


def remote_apply_fragment(worker_dict, tensors: Mapping[str, object], reset_optimizer: bool) -> dict:
    """Write ``tensors`` into the parameters this worker holds a shard of.

    Collective: ``distribute_tensor`` shards each whole tensor over the mesh,
    so every rank calls it for every name in the same order.
    """

    import torch

    module = _trainable_module(worker_dict)
    by_name = dict(_named_full_parameters(module))
    missing = sorted(set(tensors) - set(by_name))
    if missing:
        raise VerlFullParameterError(f"要写入的参数不在模型里: {missing[:4]}")
    applied = 0
    with torch.no_grad():
        for name in sorted(tensors):
            parameter = by_name[name]
            value = _as_tensor(tensors[name])
            if tuple(value.shape) != _global_shape(parameter):
                raise VerlFullParameterError(
                    f"{name!r} 形状不符: 收到 {tuple(value.shape)}，模型是 {_global_shape(parameter)}")
            if hasattr(parameter, "device_mesh"):
                from torch.distributed.tensor import distribute_tensor

                parameter.copy_(distribute_tensor(value.to(parameter.device, parameter.dtype),
                                                  parameter.device_mesh, parameter.placements))
            else:
                parameter.copy_(value.to(parameter.device, parameter.dtype))
            applied += 1
    zeroed = _reset_moments(worker_dict) if reset_optimizer else 0
    return {"applied": applied, "moments_zeroed": zeroed}


def _as_tensor(value):
    """A torch view of one value to write: a numpy array (Ray hands one over
    read-only, out of its object store) is wrapped without a copy; the write
    only reads it."""

    import warnings

    import numpy as np
    import torch

    if isinstance(value, np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # read-only buffer: never written here
            return torch.from_numpy(value)
    return value


def _reset_moments(worker_dict) -> int:
    from .ports_impl import _inner_worker

    optimizer = getattr(_inner_worker(worker_dict).actor.engine, "optimizer", None)
    zeroed = 0
    for state in (optimizer.state.values() if optimizer is not None else ()):
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and hasattr(state[key], "zero_"):
                state[key].zero_()
                zeroed += 1
    return zeroed


def agree_on_buffers(manifests: Sequence[Mapping[str, object]]) -> dict[str, list[float]]:
    """One ``{buffer name: fingerprint}`` from every worker's manifest, or an error
    naming the first buffer whose ranks disagree (:func:`_buffer_digest`).

    A manifest without ``buffer_digests`` (an older worker) is skipped, so this
    never turns a missing field into a failure.
    """

    # "absent" (an older worker) is not "empty" (a worker that reports no buffer)
    digests = [dict(m.get("buffer_digests") or {}) for m in manifests if "buffer_digests" in m]
    if not digests or not any(digests):
        return {}
    first = digests[0]
    for index, other in enumerate(digests[1:], start=1):
        if sorted(other) != sorted(first):
            only_first = sorted(set(first) - set(other))
            only_other = sorted(set(other) - set(first))
            raise VerlFullParameterError(
                f"worker 0 与 worker {index} 的 buffer 集合不一致: 仅 0 有 {only_first[:4]}, "
                f"仅 {index} 有 {only_other[:4]}")
        for name in sorted(first):
            if other[name] != first[name]:
                raise VerlFullParameterError(
                    f"buffer {name!r} 在 worker 0 与 worker {index} 上的值不同（和/绝对值和/元素数: "
                    f"{first[name]} 对 {other[name]}）。持久 buffer 不分片，每个 rank 用自己那份；"
                    f"值不同说明加载没有把它广播到所有 rank，这个 rank 的前向会算错（MoE 路由偏置就是一例）")
    return first


def agree_on_shapes(manifests: Sequence[Mapping[str, object]], *,
                    key: str = "shapes") -> dict[str, tuple[int, ...]]:
    """One ``{name: shape}`` from every worker's manifest, or an error naming the
    drift. ``key``: ``"shapes"`` (parameters) or ``"buffers"`` (persistent
    buffers; a manifest without the key has none)."""

    if not manifests:
        raise VerlFullParameterError("没有收到任何 worker 的全参数清单")
    first = {name: tuple(shape) for name, shape in dict(manifests[0].get(key) or {}).items()}
    for index, manifest in enumerate(manifests[1:], start=1):
        other = {name: tuple(shape) for name, shape in dict(manifest.get(key) or {}).items()}
        if other != first:
            only_first = sorted(set(first) - set(other))
            only_other = sorted(set(other) - set(first))
            raise VerlFullParameterError(
                f"worker 0 与 worker {index} 的参数集合不一致: "
                f"仅 0 有 {only_first[:4]}, 仅 {index} 有 {only_other[:4]}")
    return first
