"""FP32 optimizer masters of a Megatron model, read and written in place.

Moved unchanged from the Miles adapter's ``state_plugin`` (S27 G-FG) so that
every backend that trains on Megatron reads the same masters: the Miles
islands (LoRA state, critic write-back, cuts) and the verl Megatron engine
island (:mod:`yeto.rl.megatron_hf_boundary`).  ``state_plugin`` re-exports
every name here; its ``StatePluginError`` is :class:`MegatronMasterError`.

Two optimizer layouts are covered:

* a complete master: the parameter is FP32 itself, or ``main_param`` covers
  the whole parameter (Megatron's ``Float16OptimizerWithFloat16Params``);
* Megatron's ``DistributedOptimizer``: each data-parallel rank owns the FP32
  master of a sub-range of the flattened parameter (``main_param_sharded``).

Gradient-flow invariant (#64): ``Parameter.data`` is never reassigned across
dtypes; :func:`masters_as_module_parameters` swaps a new ``Parameter`` that
shares the master storage into ``module._parameters`` and restores the
original object afterwards.

torch is imported lazily; importing this module is cheap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any


class MegatronMasterError(RuntimeError):
    """A parameter has no FP32 master this module can read or write."""


# --------------------------------------------------------------------------
# Gradient-flow-safe master exposure (#64 / D4)
# --------------------------------------------------------------------------


def master_of(parameter: Any) -> Any:
    """The FP32 optimizer master viewed in the parameter's shape."""

    import torch

    main = getattr(parameter, "main_param", None)
    if main is None:
        if parameter.dtype != torch.float32:
            raise MegatronMasterError("low-precision adapter parameter has no FP32 optimizer master")
        return parameter.detach()
    if main.dtype != torch.float32 or main.numel() != parameter.numel():
        raise MegatronMasterError(
            "adapter parameter has no complete FP32 optimizer master "
            "(sharded distributed-optimizer masters are not supported)"
        )
    return main.view(parameter.shape)


def has_complete_master(parameter: Any) -> bool:
    """FP32 itself, or a ``main_param`` covering the whole parameter.

    Megatron's DistributedOptimizer sets ``main_param`` to this rank's FP32
    *shard* (``main_param_sharded=True``) on owned ranges and leaves it unset
    elsewhere; both are DP-sharded masters, not complete ones.
    """

    import torch

    if parameter.dtype == torch.float32:
        return True
    main = getattr(parameter, "main_param", None)
    return (
        main is not None
        and not getattr(parameter, "main_param_sharded", False)
        and main.numel() == parameter.numel()
    )


# --------------------------------------------------------------------------
# DistributedOptimizer: DP-sharded FP32 masters (rl-infra-spec 2.4, decision (a))
# --------------------------------------------------------------------------
# Under Megatron's DistributedOptimizer with DP>1 a bf16 adapter parameter has
# no ``main_param``: each DP rank's optimizer owns the FP32 main copy of a
# sub-range ``gbuf_ranges[..]["param_map"][param]["param"]`` of the flattened
# parameter (same layout fork-M5 v2 reads). Export gathers the full master by
# filling each rank's range into zeros and summing over the DP group (every
# element is owned by exactly one rank); apply writes each rank's range from
# the full target and sets the bf16 model copy on every rank.


def _optimizer_leaves(optimizer: Any) -> list[Any]:
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        return [leaf for member in chained for leaf in _optimizer_leaves(member)]
    return [optimizer]


def _is_distributed(leaf: Any) -> bool:
    return hasattr(leaf, "gbuf_ranges") and hasattr(leaf, "model_param_group_index_map")


def distributed_ranges(optimizer: Any) -> tuple[bool, dict[int, tuple[Any, Any, int, int]]]:
    """(has a DistributedOptimizer, {id(model_param): (leaf, param, start, end)} owned here)."""
    found, owned = False, {}
    for leaf in _optimizer_leaves(optimizer) if optimizer is not None else ():
        if not _is_distributed(leaf):
            continue
        found = True
        if getattr(getattr(leaf, "config", None), "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
            raise MegatronMasterError("sharded masters under the precision-aware optimizer are not supported")
        for gbuf_range_maps in leaf.gbuf_ranges:
            for per_bucket in gbuf_range_maps.values():
                for bucket_range_map in per_bucket:
                    for param, range_map in bucket_range_map["param_map"].items():
                        r = range_map["param"]
                        if r.end > r.start:
                            if id(param) in owned:
                                raise MegatronMasterError("a parameter range appears twice on one rank")
                            owned[id(param)] = (leaf, param, int(r.start), int(r.end))
    return found, owned


def _leaf_members(leaf: Any) -> set[int]:
    """ids of every model parameter this distributed leaf shards (owned here or not).

    ``gbuf_ranges``/``model_param_group_index_map`` list only the locally owned
    ranges; the grad buffers (``leaf.buffers[..].param_index_map``) list all.
    """

    members: set[int] = set()
    for buffer in getattr(leaf, "buffers", None) or ():
        members.update(id(p) for p in getattr(buffer, "param_index_map", {}) or {})
    for gbuf_range_maps in getattr(leaf, "gbuf_ranges", None) or ():
        for per_bucket in gbuf_range_maps.values():
            for bucket_range_map in per_bucket:
                members.update(id(p) for p in bucket_range_map["param_map"])
    return members


def _leaf_for(optimizer: Any, param: Any, owned: Mapping[int, tuple[Any, Any, int, int]]) -> Any:
    """The distributed leaf (hence DP group) that shards ``param``.

    ChainedOptimizer: dense params belong to the dense leaf (dense DP group),
    expert params to the expert leaf (expert DP group).
    """

    if id(param) in owned:
        return owned[id(param)][0]
    leaves = [leaf for leaf in _optimizer_leaves(optimizer) if _is_distributed(leaf)]
    hits = [leaf for leaf in leaves if id(param) in _leaf_members(leaf)]
    if len(hits) == 1:
        return hits[0]
    if not hits and len(leaves) == 1:
        return leaves[0]
    raise MegatronMasterError("cannot tell which distributed optimizer shards an adapter parameter")


def _dp_all_reduce_sum(leaf: Any, flat: Any) -> None:
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=getattr(leaf, "data_parallel_group", None))


def needs_gather(optimizer: Any, parameters: Sequence[Any]) -> bool:
    """Whether some low-precision parameter's master lives only in DistributedOptimizer shards."""

    if optimizer is None or not any(_is_distributed(leaf) for leaf in _optimizer_leaves(optimizer)):
        return False
    return any(not has_complete_master(p) for p in parameters)


def _single_rank_dp(leaf: Any) -> bool:
    """True only when torch.distributed is up and the leaf's DP group has one rank
    (no collective needed). Not initialised: False, the caller keeps the
    ``_dp_all_reduce_sum`` path (a no-op there, patched in tests)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return False
    try:
        return int(dist.get_world_size(group=getattr(leaf, "data_parallel_group", None))) == 1
    except (RuntimeError, ValueError, TypeError, AttributeError):
        return False  # unknown group: keep the collective path


def full_masters(optimizer: Any, parameters: Sequence[Any], *, all_reduce_sum=None, reduce=None,
                 device: Any = None) -> list[Any]:
    """FP32 masters in each parameter's shape; DP-sharded ones are gathered.

    Collective over each owning leaf's DP group when any parameter is
    DP-sharded (every rank must call it with the same parameters in the same
    order). ``reduce(flat, leaf)`` / ``all_reduce_sum(flat)`` replace the
    all-reduce (tests).

    One flat FP32 buffer per leaf holds the gathered masters (each returned
    master is a view of it); no second concatenated copy is made
    (s19-compaction-g3-20261010a: the critic export ran out of GPU memory on
    that copy). ``device``: where the gather buffers live (default: each
    parameter's device). A leaf whose DP group has one rank needs no
    collective, so a CPU buffer is allowed there; a CPU buffer with a
    multi-rank group is refused (NCCL reduces GPU tensors only).
    """
    import torch

    has_dist, owned = distributed_ranges(optimizer)
    out: list[Any] = [None] * len(parameters)
    pending: dict[int, tuple[Any, list[int]]] = {}
    for i, param in enumerate(parameters):
        if has_complete_master(param):
            out[i] = master_of(param)
            continue
        if not has_dist:
            out[i] = master_of(param)  # raises: no master anywhere
            continue
        leaf = _leaf_for(optimizer, param, owned)
        pending.setdefault(id(leaf), (leaf, []))[1].append(i)
    # Leaf order is the same on every rank of a DP group, so the per-group
    # collectives are issued in the same order.
    order = {id(leaf): k for k, leaf in enumerate(_optimizer_leaves(optimizer))} if pending else {}
    for _, (leaf, items) in sorted(pending.items(), key=lambda kv: order.get(kv[0], 0)):
        hooked = reduce is not None or all_reduce_sum is not None
        where = device if device is not None else parameters[items[0]].device
        if (not hooked and torch.device(where).type == "cpu"
                and parameters[items[0]].device.type != "cpu" and not _single_rank_dp(leaf)):
            raise MegatronMasterError("CPU gather buffer with a multi-rank DP group")
        flat = torch.zeros(sum(parameters[i].numel() for i in items), dtype=torch.float32, device=where)
        offset = 0
        for i in items:
            param = parameters[i]
            n = param.numel()
            full = flat[offset : offset + n]
            if id(param) in owned:
                _, _, start, end = owned[id(param)]
                shard = owned[id(param)][0]._get_main_param_and_optimizer_states(param)["param"]
                if shard.dtype != torch.float32 or shard.numel() != end - start:
                    raise MegatronMasterError("distributed-optimizer main shard does not match its range")
                full[start:end].copy_(shard.detach().reshape(-1))
            out[i] = full.view(param.shape)
            offset += n
        if reduce is not None:
            reduce(flat, leaf)
        elif all_reduce_sum is not None:
            all_reduce_sum(flat)
        elif not _single_rank_dp(leaf):
            _dp_all_reduce_sum(leaf, flat)
    return out


def write_masters(optimizer: Any, parameters: Sequence[Any], targets: Sequence[Any]) -> bool:
    """Write full FP32 targets into the masters; returns whether model copies were set here.

    For DP-sharded masters each rank writes its own range and sets the bf16
    model copy directly (in place; ``Parameter.data`` is not reassigned).
    """
    import torch

    has_dist, owned = distributed_ranges(optimizer)
    wrote_model = False
    for param, target in zip(parameters, targets, strict=True):
        if has_complete_master(param):
            master_of(param).copy_(target)
            continue
        if not has_dist:
            master_of(param)  # raises
        if id(param) in owned:
            leaf, _, start, end = owned[id(param)]
            shard = leaf._get_main_param_and_optimizer_states(param)["param"]
            shard.data.copy_(target.reshape(-1)[start:end].reshape(shard.shape))
        param.data.copy_(target.to(dtype=param.dtype))
        wrote_model = True
    return wrote_model


def parameter_owners(modules: Iterable[Any], parameters: Iterable[Any]) -> dict[int, tuple[Any, str]]:
    wanted = {id(p) for p in parameters}
    owners: dict[int, tuple[Any, str]] = {}
    for root in modules:
        for module in root.modules():
            for attr, param in module._parameters.items():
                if param is not None and id(param) in wanted:
                    owners.setdefault(id(param), (module, attr))
    missing = wanted - set(owners)
    if missing:
        raise MegatronMasterError(f"{len(missing)} adapter parameters are not registered in the model")
    return owners


@contextmanager
def masters_as_module_parameters(modules: Sequence[Any], parameters: Sequence[Any], masters: Sequence[Any] | None = None):
    """Temporarily register FP32 masters as the module parameters.

    A fresh ``Parameter`` that shares the master's storage and has
    ``requires_grad=False`` replaces the entry in ``module._parameters``; the
    original Parameter object (with its grad-accumulation hooks and
    ``main_grad``) is untouched and restored on exit. ``Parameter.data`` is
    never reassigned. ``masters`` (e.g. gathered DistributedOptimizer shards,
    see ``full_masters``) overrides ``master_of``.
    """

    import torch

    owners = parameter_owners(modules, parameters)
    swapped: list[tuple[Any, str, Any]] = []
    try:
        if masters is None:
            masters = [master_of(p) for p in parameters]
        for param, master in zip(parameters, masters, strict=True):
            if master.data_ptr() == param.data_ptr() and master.dtype == param.dtype:
                continue  # already FP32 and self-mastered
            module, attr = owners[id(param)]
            module._parameters[attr] = torch.nn.Parameter(master, requires_grad=False)
            swapped.append((module, attr, param))
        yield
    finally:
        for module, attr, param in reversed(swapped):
            module._parameters[attr] = param


def assert_grad_flow_intact(modules: Sequence[Any], parameters: Sequence[Any]) -> None:
    owners = parameter_owners(modules, parameters)
    for param in parameters:
        module, attr = owners[id(param)]
        if module._parameters[attr] is not param:
            raise MegatronMasterError(f"adapter parameter {attr!r} was replaced")
        if not param.requires_grad:
            raise MegatronMasterError(f"adapter parameter {attr!r} no longer requires grad")
