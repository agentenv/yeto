"""Full-parameter boundary of a Megatron model in HF parameter names (S27 G-FG).

A full-parameter island synchronizes whole FP32 parameters with the unified
syncer.  The verl FSDP2 island names them by the HF parameter names of the
model (``adapters/verl/full_parameter.py``), so its layout carries no shard
identity.  This module gives a Megatron model (any TP/PP/EP shape) the same
boundary, so a Megatron island plans the same layout (same hash) as an FSDP2
island of the same model.

Nothing here is new conversion code: Megatron-Bridge already maps every
Megatron parameter to its HF tensors and back (the same conversion tasks verl
uses to send weights to vLLM and to load the checkpoint):

* **export** -- the FP32 optimizer masters (:mod:`yeto.rl.megatron_masters`,
  complete or DistributedOptimizer-sharded) are swapped in as the module
  parameters, and Bridge's ``export_hf_weights`` gathers them over TP, PP and
  EP into whole HF tensors.  Bridge conversions are reshapes, splits and
  concatenations, so the FP32 values pass through unchanged;
* **apply** -- Bridge's ``hf_to_megatron`` (with the masters swapped in, so
  its dtype cast is a no-op) cuts each rank's FP32 shard out of whole HF
  tensors, and :func:`yeto.rl.megatron_masters.write_masters` writes the
  master shard and the bf16 model copy in place.

Every function that touches tensors is collective: all ranks of the model call
it with the same task indices in the same order (Bridge gathers over the TP,
PP and EP groups; the master gather reduces over each data-parallel group).
The island decides the indices once from every rank's manifest
(:class:`HfTaskIndex`), so ranks whose tasks name different experts still run
the same tasks.

Pipeline parallelism (S27 G-FL): Bridge's task list covers the global parameter
names of every stage, so a stage holds ``param_weight = None`` for another
stage's tasks and receives the whole tensor over the PP group.  The one
parameter outside that list is the tied embedding's second copy on the last
stage (:func:`tied_output_copies`), which an apply writes explicitly.

Megatron, Megatron-Bridge and torch are imported lazily by the caller's
objects; importing this module is cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .megatron_masters import (
    distributed_ranges,
    full_masters,
    has_complete_master,
    masters_as_module_parameters,
    parameter_owners,
    write_masters,
)


class MegatronHfBoundaryError(RuntimeError):
    """The Megatron model cannot export or apply an HF-named full-parameter boundary."""


# ----------------------------------------------------------------------------- tasks (worker side)
def conversion_tasks(bridge: Any, modules: Sequence[Any]) -> list[Any]:
    """Bridge's conversion tasks of ``modules``, one per global parameter (None
    where Bridge has no mapping).  The same list on every rank: Bridge orders it
    by the global parameter names of all pipeline stages."""

    return list(bridge.get_conversion_tasks(list(modules)))


def task_hf_names(task: Any) -> tuple[str, ...]:
    """The HF tensor names one task writes (a fused Megatron parameter, e.g. QKV
    or gate/up, names several)."""

    hf_param = task.mapping.hf_param
    if isinstance(hf_param, str):
        return (hf_param,)
    return tuple(str(name) for name in hf_param.values())


def _is_parameter(tensor: Any) -> bool:
    import torch

    return isinstance(tensor, torch.nn.Parameter)


def _local_parameters(tasks: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    """The parameters (not buffers) this rank holds for ``indices``, task order,
    each once (a tied parameter can serve two tasks)."""

    out, seen = [], set()
    for index in indices:
        weight = tasks[index].param_weight
        if weight is not None and _is_parameter(weight) and id(weight) not in seen:
            seen.add(id(weight))
            out.append(weight)
    return out


# ----------------------------------------------------------------------------- the tied embedding's second copy
EMBEDDING_SUFFIX = "embedding.word_embeddings.weight"


def embedding_task_index(tasks: Sequence[Any]) -> int | None:
    """The index of the input embedding's conversion task (Bridge's own criterion,
    ``MegatronModelBridge._get_tied_output_hf_name``), or None."""

    for index, task in enumerate(tasks):
        if task is not None and str(task.global_param_name).endswith(EMBEDDING_SUFFIX):
            return index
    return None


def tied_output_copies(modules: Sequence[Any], tasks: Sequence[Any]) -> list[Any]:
    """The tied embedding's second copy on this rank, which no conversion task writes
    (S27 G-FL; empty for PP == 1 and for an untied model).

    With ``share_embeddings_and_output_weights`` and PP > 1 the last pipeline stage
    owns its own ``output_layer.weight``: Megatron fills it with the first stage's
    embedding over the embedding group at start-up and keeps the two equal by
    all-reducing both gradients (``LanguageModule.setup_embeddings_and_output_layer``).
    Megatron-Bridge drops every ``output_layer`` global name when the embeddings are
    tied, so no task carries that copy, and an export always reads the first stage's
    embedding (the lowest-ranked owner, ``broadcast_from_pp_rank``).  An apply that
    wrote only the first stage would leave the two apart with no symptom in the
    export, so :func:`apply_hf` writes this copy from the same whole HF tensor.
    """

    covered = {id(task.param_weight) for task in tasks
               if task is not None and task.param_weight is not None}
    out: list[Any] = []
    for chunk in modules:
        inner = getattr(chunk, "language_model", None)
        if inner is None:
            inner = chunk
        if not getattr(inner, "share_embeddings_and_output_weights", False):
            continue
        weight = getattr(getattr(inner, "output_layer", None), "weight", None)
        if weight is None or not _is_parameter(weight):
            continue
        if id(weight) in covered or any(weight is kept for kept in out):
            continue
        out.append(weight)
    return out


def manifest(bridge: Any, modules: Sequence[Any], optimizer: Any) -> dict[str, Any]:
    """This rank's boundary metadata (collective-free).

    ``tasks`` lists ``[index, [HF names], master dtype]`` for every task Bridge
    maps.  The master dtype is the dtype the sync reads: ``float32`` for a
    parameter (FP32 itself, a complete ``main_param``, or a DistributedOptimizer
    shard), the buffer's own dtype for a persistent buffer, None when another
    pipeline stage holds the parameter.  ``refused`` lists the parameters
    without an FP32 master, the tied embedding's second copy included
    (:func:`tied_output_copies`), so the island refuses at start instead of
    mid-run."""

    import torch

    tasks = conversion_tasks(bridge, modules)
    has_dist, _owned = distributed_ranges(optimizer)
    rows, refused = [], []
    for weight in tied_output_copies(modules, tasks):
        if not (weight.dtype == torch.float32 or has_complete_master(weight) or has_dist):
            refused.append("output_layer.weight (the tied embedding's pipeline copy)")
    for index, task in enumerate(tasks):
        if task is None:
            continue
        weight, dtype = task.param_weight, None
        if weight is not None and not _is_parameter(weight):
            dtype = str(weight.dtype).replace("torch.", "")
        elif weight is not None:
            if weight.dtype == torch.float32 or has_complete_master(weight) or has_dist:
                dtype = "float32"
            else:
                dtype = str(weight.dtype).replace("torch.", "")
                refused.append(str(task.global_param_name))
        rows.append([index, list(task_hf_names(task)), dtype])
    return {"tasks": rows, "refused": refused}


# ----------------------------------------------------------------------------- export and apply (worker side)
def _export_swapped(bridge: Any, modules: Sequence[Any], swapped: Sequence[Any], indices: Sequence[int],
                    sink: Callable[[str, Any], None]) -> int:
    count = 0
    for name, tensor in bridge.export_hf_weights(list(modules), cpu=False, show_progress=False,
                                                  conversion_tasks=[swapped[index] for index in indices],
                                                  merge_adapter_weights=False):
        sink(str(name), tensor)
        count += 1
    return count


def export_hf(bridge: Any, modules: Sequence[Any], optimizer: Any, indices: Sequence[int],
              sink: Callable[[str, Any], None]) -> int:
    """Gather the FP32 HF tensors of tasks ``indices`` and hand each to
    ``sink(name, tensor)`` (whole tensor on this rank's device; every rank sees
    every tensor).  Collective.  Returns the tensors handed over.

    A tied output layer (Bridge may emit ``lm_head.weight`` next to the
    embedding) arrives like any other name; the caller keeps the names it asked for."""

    tasks = conversion_tasks(bridge, modules)
    params = _local_parameters(tasks, indices)
    masters = full_masters(optimizer, params)
    with masters_as_module_parameters(list(modules), params, masters):
        swapped = conversion_tasks(bridge, modules)  # param_weight is now each FP32 master
        return _export_swapped(bridge, modules, swapped, indices, sink)


def _hf_input(hf_param: Any, state: Mapping[str, Any]) -> Any:
    """Bridge's default HF fetch (``MegatronModelBridge.maybe_modify_loaded_hf_weight``):
    one tensor, or ``{role: tensor}`` for a fused parameter."""

    if isinstance(hf_param, str):
        return state[hf_param]
    return {role: state[name] for role, name in hf_param.items()}


def _plan_shard(shards: list, original: Any, converted: Any, what: str) -> None:
    """Keep ``converted`` as the FP32 target of ``original`` (one rank's Megatron shard)."""

    import torch

    if converted is None or original is None:
        return
    if tuple(converted.shape) != tuple(original.shape):
        raise MegatronHfBoundaryError(
            f"{what}: Bridge gave shape {tuple(converted.shape)}, the rank holds "
            f"{tuple(original.shape)} (a padded vocabulary is not supported)")
    shards.append((original, converted.to(dtype=torch.float32)))


def apply_hf(bridge: Any, modules: Sequence[Any], optimizer: Any, tensors: Mapping[str, Any],
             write_indices: Sequence[int], read_indices: Sequence[int]) -> dict[str, int]:
    """Make the model hold ``tensors`` (whole FP32 HF tensors): the FP32 master
    shard and the model copy of every parameter of tasks ``write_indices``.

    A fused parameter needs all its HF inputs; tasks ``read_indices`` (a subset of
    ``write_indices`` decided by the island) first export the current values,
    and ``tensors`` overrides them.  When the embedding task is written and this
    rank also holds the tied embedding's pipeline copy, that copy is written from
    the same whole HF tensor (:func:`tied_output_copies`).  Collective.  Returns
    counts."""

    import torch

    tasks = conversion_tasks(bridge, modules)
    params = _local_parameters(tasks, write_indices)
    embedding = embedding_task_index(tasks)
    tied: list[Any] = []
    embedding_name: str | None = None
    if embedding is not None and embedding in {int(index) for index in write_indices}:
        hf_param = tasks[embedding].mapping.hf_param
        if isinstance(hf_param, str):
            tied = tied_output_copies(modules, tasks)
            embedding_name = str(hf_param)
    # One list, the same order on every rank of a data-parallel group (the gather in
    # ``full_masters`` is collective over it; pipeline stages have their own groups).
    gathered = [*params, *tied]
    # Before the swap: inside it the module entry is the master, not this object.
    tied_owners = parameter_owners(list(modules), tied) if tied else {}
    masters = full_masters(optimizer, gathered)
    state: dict[str, Any] = {}
    shards: list[tuple[Any, Any]] = []
    with masters_as_module_parameters(list(modules), gathered, masters):
        swapped = conversion_tasks(bridge, modules)  # param_weight is now each FP32 master
        if read_indices:
            _export_swapped(bridge, modules, swapped, read_indices, state.__setitem__)
        state.update(tensors)
        for index in write_indices:
            task = swapped[index]
            if task.megatron_module is None:  # another pipeline stage holds it (as Bridge's load)
                continue
            try:
                hf_weights = _hf_input(task.mapping.hf_param, state)
            except KeyError as error:
                raise MegatronHfBoundaryError(f"no HF input {error} for {task.global_param_name}") from None
            _plan_shard(shards, tasks[index].param_weight,
                        task.mapping.hf_to_megatron(hf_weights, task.megatron_module),
                        str(task.global_param_name))
        if tied:
            if embedding_name not in state:
                raise MegatronHfBoundaryError(f"no HF input {embedding_name!r} for the tied output layer")
            mapping, whole = swapped[embedding].mapping, state[embedding_name]
            for weight in tied:
                module, _attr = tied_owners[id(weight)]
                # The same column-parallel conversion as the embedding, with the
                # output layer as the target module (its TP shard and dtype).
                _plan_shard(shards, weight, mapping.hf_to_megatron(whole, module),
                            "output_layer.weight (the tied embedding's pipeline copy)")
    written_params, written_buffers = [], 0
    with torch.no_grad():
        for original, target in shards:
            if _is_parameter(original):
                written_params.append((original, target))
            else:  # a persistent buffer (no optimizer master): written as it is
                original.copy_(target.to(original.device, original.dtype))
                written_buffers += 1
        plist = [p for p, _ in written_params]
        tlist = [t.to(p.device) for p, t in written_params]
        write_masters(optimizer, plist, tlist)
        for param, target in zip(plist, tlist, strict=True):
            if param.dtype != torch.float32:  # a complete master: write_masters set the master only
                param.data.copy_(target.to(dtype=param.dtype))
    return {"parameters_written": len(written_params), "buffers_written": written_buffers,
            "tasks": len(write_indices), "tasks_read": len(read_indices), "tied_outputs": len(tied)}


# ----------------------------------------------------------------------------- the island's task index
@dataclass(frozen=True)
class HfTaskIndex:
    """Which conversion tasks produce which HF names, over all ranks.

    ``names[i]`` is the union of task ``i``'s HF names on every rank (ranks of
    different expert-parallel coordinates name different experts at one
    index); ``by_name`` inverts it.  Built once from every rank's
    :func:`manifest` and checked against the boundary's layout names."""

    names: Mapping[int, frozenset[str]]
    by_name: Mapping[str, tuple[int, ...]]

    @classmethod
    def from_manifests(cls, records: Sequence[Mapping[str, Any]], layout_names: Iterable[str]
                       ) -> HfTaskIndex:
        refused = sorted({name for record in records for name in record.get("refused", ())})
        if refused:
            raise MegatronHfBoundaryError(f"parameters without an FP32 optimizer master: {refused[:4]}")
        names: dict[int, set[str]] = {}
        for record in records:
            for index, hf_names, _dtype in record["tasks"]:
                names.setdefault(int(index), set()).update(str(name) for name in hf_names)
        layout = set(layout_names)
        produced = set().union(*names.values()) if names else set()
        missing = sorted(layout - produced)
        if missing:
            raise MegatronHfBoundaryError(f"Megatron-Bridge exports no tensor for layout names {missing[:4]}"
                                          + (f" ... and {len(missing) - 4} more" if len(missing) > 4 else ""))
        extra = sorted(produced - layout)
        if extra:
            raise MegatronHfBoundaryError(f"Megatron-Bridge exports names the layout does not have: {extra[:4]}")
        by_name: dict[str, list[int]] = {}
        for index in sorted(names):
            for name in names[index]:
                by_name.setdefault(name, []).append(index)
        doubled = sorted(name for name, indices in by_name.items() if len(indices) > 1)
        if doubled:
            raise MegatronHfBoundaryError(f"two conversion tasks write {doubled[:4]}")
        return cls(names={index: frozenset(value) for index, value in names.items()},
                   by_name={name: tuple(indices) for name, indices in by_name.items()})

    def tasks_for(self, hf_names: Iterable[str]) -> list[int]:
        """The task indices that produce any of ``hf_names``, ascending."""

        out: set[int] = set()
        for name in hf_names:
            indices = self.by_name.get(name)
            if indices is None:
                raise MegatronHfBoundaryError(f"no conversion task produces {name!r}")
            out.update(indices)
        return sorted(out)

    def apply_plan(self, given: Iterable[str]) -> tuple[list[int], list[int]]:
        """``(write, read)`` task indices of an apply of ``given``: every task
        that writes one of them, and those of it that also need inputs not given."""

        given = set(given)
        write = self.tasks_for(given)
        read = [index for index in write if not self.names[index] <= given]
        return write, read

    def master_dtypes(self, records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        """``{HF name: dtype of the master the sync reads}`` over all ranks."""

        out: dict[str, str] = {}
        for record in records:
            for _index, hf_names, dtype in record["tasks"]:
                if dtype is not None:
                    for name in hf_names:
                        out.setdefault(str(name), str(dtype))
        return out
