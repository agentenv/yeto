"""How a trainer rank's parameters sit in the whole model, and the offline merge (S27 R-5).

A full-parameter cut shard holds the rank's local model copy keyed by local
Megatron names (``cut_plugin``). With TP/PP/EP > 1 those tensors are slices
under rank-local layer and expert indices, so a reader cannot rebuild the
model from the tensors alone. At save time the writer therefore records, per
local parameter, what the live trainer knows (:func:`build_param_layout`):

* ``global_name``: the name with pipeline-layer and expert offsets folded in
  (the trainer's weight-sync naming, Miles ``named_params_and_buffers``);
* ``tensor_parallel``, ``partition_dim``, ``partition_stride``: the TP split of
  the parameter as the trainer's weight sync gathers it;
* ``expert``: a routed-expert parameter (gathered over expert TP, owned by one
  EP rank) or a dense one (gathered over TP, replicated over EP).

:func:`merge_shards` rebuilds every whole tensor from the shards of one cut:
dense parameters are gathered over TP ranks, routed-expert parameters over
expert-TP ranks; a replicated parameter must be bitwise equal on every rank
that holds it (checked, not assumed).

Import-light: torch only (lazily); no Miles or Megatron import.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

PARAM_LAYOUT_SCHEMA = "yeto.param_layout/v1"
RANK_KEYS = ("tp_rank", "tp_size", "pp_rank", "pp_size", "ep_rank", "ep_size", "etp_rank", "etp_size")

# Megatron SequentialMLP names its experts by local index; the global index is
# ep_rank * num_local_experts + local index (megatron/core/transformer/moe/moe_layer.py,
# ``local_expert_indices_offset``). Rewritten to the grouped-GEMM form the trainer's
# weight sync and the HF converters use: ``mlp.experts.linear_fc1.weight<global>``.
_SEQUENTIAL_EXPERT = re.compile(r"^(.*\.mlp\.experts)\.local_experts\.(\d+)\.(linear_fc[12])\.weight$")

# (local name, parameter) -> (tensor_parallel, partition_dim, partition_stride)
PartitionOf = Callable[[str, Any], tuple[bool, int, int]]


class ParamLayoutError(RuntimeError):
    pass


def sequential_expert_global_name(name: str, *, ep_rank: int, num_local_experts: int) -> str:
    match = _SEQUENTIAL_EXPERT.match(name)
    if match is None:
        return name
    prefix, local, proj = match.groups()
    return f"{prefix}.{proj}.weight{ep_rank * num_local_experts + int(local)}"


def build_param_layout(named: Sequence[tuple[str, Any]], *, global_names: Mapping[str, str],
                       partition_of: PartitionOf, is_expert: Callable[[str], bool],
                       ranks: Mapping[str, int], num_experts: int | None = None) -> dict[str, Any]:
    """The layout record of one rank (stored in the cut shard as ``param_layout``)."""
    missing = [k for k in RANK_KEYS if k not in ranks]
    if missing:
        raise ParamLayoutError(f"rank coordinates missing: {missing}")
    num_local = (int(num_experts) // int(ranks["ep_size"])) if num_experts else 0
    params: dict[str, dict[str, Any]] = {}
    for local, param in named:
        if local not in global_names:
            raise ParamLayoutError(f"{local}: no global name")
        global_name = sequential_expert_global_name(global_names[local], ep_rank=int(ranks["ep_rank"]),
                                                    num_local_experts=num_local)
        tp, dim, stride = partition_of(global_name, param)
        params[local] = {"global_name": global_name, "tensor_parallel": bool(tp),
                         "partition_dim": int(dim), "partition_stride": int(stride),
                         "expert": bool(is_expert(global_name))}
    globals_seen = [p["global_name"] for p in params.values()]
    if len(set(globals_seen)) != len(globals_seen):
        raise ParamLayoutError("two local parameters share one global name")
    return {"schema": PARAM_LAYOUT_SCHEMA, **{k: int(ranks[k]) for k in RANK_KEYS}, "params": params}


def gather_with_stride(parts: Sequence[Any], dim: int, stride: int) -> Any:
    """Concatenate TP slices; ``stride`` > 1 interleaves them (e.g. GLU ``[gate, up]``).

    Same rule as the trainer's weight-sync gather (Miles
    ``hf_weight_iterator_direct._gather_with_stride``)."""
    import torch

    if stride == 1:
        return torch.cat(list(parts), dim=dim)
    chunks = [p.chunk(stride, dim=dim) for p in parts]
    return torch.cat([chunks[r][s] for s in range(stride) for r in range(len(parts))], dim=dim)


def _check_sizes(layouts: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    sizes = {(lay["tp_size"], lay["pp_size"], lay["ep_size"], lay["etp_size"]) for lay in layouts}
    if len(sizes) != 1:
        raise ParamLayoutError(f"shards disagree on parallel sizes: {sorted(sizes)}")
    tp, pp, ep, etp = next(iter(sizes))
    return {"tp_size": tp, "pp_size": pp, "ep_size": ep, "etp_size": etp}


def merge_shards(shards: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, Any]:
    """Whole tensors by global name from ``(param_layout, local tensors)`` of several ranks.

    The shards must cover every TP rank of every pipeline stage, and every
    (EP, expert-TP) rank pair for routed experts. Extra replicas are allowed and
    must be bitwise equal."""
    import torch

    if not shards:
        raise ParamLayoutError("no shards to merge")
    for layout, _ in shards:
        if layout.get("schema") != PARAM_LAYOUT_SCHEMA:
            raise ParamLayoutError(f"unknown param layout schema {layout.get('schema')!r}")
    sizes = _check_sizes(lay for lay, _ in shards)
    # global name -> slice rank -> tensor (+ the parameter's split)
    slices: dict[str, dict[int, Any]] = defaultdict(dict)
    split: dict[str, tuple[bool, int, int, bool]] = {}
    for layout, tensors in shards:
        for local, meta in layout["params"].items():
            if local not in tensors:
                raise ParamLayoutError(f"{local}: in the layout but not in the shard")
            name = meta["global_name"]
            info = (meta["tensor_parallel"], meta["partition_dim"], meta["partition_stride"], meta["expert"])
            if split.setdefault(name, info) != info:
                raise ParamLayoutError(f"{name}: ranks disagree on its split")
            slot = (layout["etp_rank"] if meta["expert"] else layout["tp_rank"]) if meta["tensor_parallel"] else 0
            held = slices[name].get(slot)
            tensor = tensors[local]
            if held is None:
                slices[name][slot] = tensor
            elif not torch.equal(held, tensor):
                raise ParamLayoutError(f"{name}: replicas differ (slot {slot})")
    out: dict[str, Any] = {}
    for name, by_slot in slices.items():
        tp, dim, stride, expert = split[name]
        width = (sizes["etp_size"] if expert else sizes["tp_size"]) if tp else 1
        if sorted(by_slot) != list(range(width)):
            raise ParamLayoutError(f"{name}: have TP slices {sorted(by_slot)} of {width}")
        out[name] = by_slot[0] if width == 1 else gather_with_stride([by_slot[r] for r in range(width)], dim, stride)
    return out


def required_coverage(layouts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Rank coordinates missing from a shard set (empty = every slice is present)."""
    sizes = _check_sizes(layouts)
    have = {(lay["pp_rank"], lay["tp_rank"], lay["ep_rank"], lay["etp_rank"]) for lay in layouts}
    missing = []
    for pp in range(sizes["pp_size"]):
        for tp in range(sizes["tp_size"]):
            if not any(h[0] == pp and h[1] == tp for h in have):
                missing.append(f"pp{pp}/tp{tp}")
        for ep in range(sizes["ep_size"]):
            for etp in range(sizes["etp_size"]):
                if not any(h[0] == pp and h[2] == ep and h[3] == etp for h in have):
                    missing.append(f"pp{pp}/ep{ep}/etp{etp}")
    return missing
