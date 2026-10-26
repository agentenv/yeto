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


def remote_manifest(worker_dict) -> dict:
    """This worker's full-parameter manifest: shapes, dtype and mesh facts.

    Collective-free (it reads metadata only), so every rank returns the same
    shapes; the island compares them before planning a layout.
    """

    import torch.distributed as dist

    named = _named_full_parameters(_trainable_module(worker_dict))
    dtypes = sorted({str(parameter.dtype) for _name, parameter in named})
    if dtypes != [MASTER_DTYPE]:
        raise VerlFullParameterError(
            f"全参数同步需要 FP32 主权重（actor.fsdp_config.model_dtype=fp32）；"
            f"看到的参数 dtype: {dtypes}")
    initialized = dist.is_available() and dist.is_initialized()
    return {
        "shapes": {name: list(_global_shape(parameter)) for name, parameter in named},
        "param_dtypes": dtypes,
        "rank": int(dist.get_rank()) if initialized else 0,
        "world_size": int(dist.get_world_size()) if initialized else 1,
        "sharded": any(hasattr(parameter, "device_mesh") for _name, parameter in named),
    }


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
    by_name = dict(_named_full_parameters(module))
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise VerlFullParameterError(f"分片引用了模型里没有的参数: {missing[:4]}")
    tensors = {}
    with torch.no_grad():
        for name in names:  # the island's order, the same on every rank
            parameter = by_name[name]
            whole = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
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
            parameter = by_name[name]
            whole = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
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


def agree_on_shapes(manifests: Sequence[Mapping[str, object]]) -> dict[str, tuple[int, ...]]:
    """One ``{name: shape}`` from every worker's manifest, or an error naming the drift."""

    if not manifests:
        raise VerlFullParameterError("没有收到任何 worker 的全参数清单")
    first = {name: tuple(shape) for name, shape in dict(manifests[0]["shapes"]).items()}
    for index, manifest in enumerate(manifests[1:], start=1):
        other = {name: tuple(shape) for name, shape in dict(manifest["shapes"]).items()}
        if other != first:
            only_first = sorted(set(first) - set(other))
            only_other = sorted(set(other) - set(first))
            raise VerlFullParameterError(
                f"worker 0 与 worker {index} 的参数集合不一致: "
                f"仅 0 有 {only_first[:4]}, 仅 {index} 有 {only_other[:4]}")
    return first
