"""Full-parameter layout that does not depend on an island's parallel shape.

The shape-bound layout (``yeto.rl.local_learner.fit_owner_affine_layout`` over
the specs each trainer rank owns) names every tensor by the rank that owns it
(``role::<tp/pp/ep/cp/dp ranks and sizes>::name``) and uses that rank's local
TP slice as the tensor. Two islands with different TP/EP/DP shapes therefore
have different layouts and cannot join one syncer session.

This module builds a layout from global tensors instead. Plain rules:

* Unit. Every global parameter is cut into units. A parameter that tensor
  parallelism splits along ``partition_dim`` is cut into ``tp_units`` (G)
  units: unit ``c`` holds, for each block the dimension packs (Megatron
  ``partition_sizes``, e.g. Mamba ``in_proj`` z/x/B/C/dt), the ``c``-th of G
  equal parts of that block. So unit ``c`` is exactly the local tensor of TP
  rank ``c`` in an island with TP = G. Every other parameter (an expert, a
  norm, a router) is one unit: the whole tensor.
* Name. A split unit is ``tp<c>-of-<G>.<parameter name>``; a whole unit keeps
  the parameter name (Megatron global names: PP layer and EP expert offsets
  already folded in).
* Class. Every unit has a placement class, used as the spec's ``shard_id``:
  ``x<k>`` for experts (``k = expert * expert_classes // experts in the
  layer``), ``d<g>.t<c>`` for unit ``c`` of a split dense parameter of dense
  group ``g``, ``r<g>`` for a whole dense parameter of group ``g``. The
  ``dense_groups`` groups come from a greedy balance of the global dense
  parameters. With ``split_layers`` every class also names the layer, so no
  class spans two pipeline stages. The owner-affine planner never mixes two
  classes in one fragment.
* Layout. The layout is ``fit_owner_affine_layout`` over the unit specs. It
  depends only on the global parameters and the granularity, so islands of
  any compatible shape compute the same layout and the same hash.
* Owners (island-local, not in the hash). The ranks holding a unit are its
  holders. All units of a class must have the same holders (else the shape is
  refused). Each (class without the TP column, holders) key gets one owner
  rank among its holders, balanced by bytes; every rank must own at least
  one key. A local parameter's owner is its key's owner; its writers are all
  holders: its data-parallel group, and for a whole dense parameter also the
  groups of the other TP ranks, which write the owner's bytes.
* Members. The owner reads a unit from its local tensor as flat segments in
  unit order: one segment per block per outer index (one per block when the
  partition dimension is 0). An island with TP = G reads each unit as its
  whole local tensor.

Compatible shapes: island TP divides ``tp_units``; every expert class lies on
one EP rank (``expert_classes`` a multiple of the island EP and a divisor of
the expert count); PP > 1 needs ``split_layers``. Expert tensor parallelism
must be 1. A shape outside these rules is refused with the reason.

This module is engine-neutral: a backend adapter turns its rank reports into
:class:`RankParameters` and the plan's members into its own owner plans.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.rl.local_learner import (
    ComponentIdentity,
    ParameterLayout,
    ParameterSpec,
    fit_owner_affine_layout,
)
from yeto.rl.moe_layout import expert_of

LAYOUT_SHAPE_BOUND = "shape-bound"
LAYOUT_SHAPE_FREE = "shape-free"
LAYOUT_MODES = (LAYOUT_SHAPE_BOUND, LAYOUT_SHAPE_FREE)

_LAYER = re.compile(r"(?:^|\.)(decoder|mtp)\.layers\.(\d+)\.")


class ShapeFreeLayoutError(ValueError):
    """An island shape (or a model) the shape-free layout cannot serve."""


@dataclass(frozen=True)
class Granularity:
    """Session constants of a shape-free layout (every island must use the same ones)."""

    tp_units: int = 1
    expert_classes: int = 8
    dense_groups: int = 8
    split_layers: bool = False

    def __post_init__(self) -> None:
        for name in ("tp_units", "expert_classes", "dense_groups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"shape-free {name} must be a positive integer, got {value!r}")
        if not isinstance(self.split_layers, bool):
            raise ValueError("shape-free split_layers must be a bool")


@dataclass(frozen=True, order=True)
class LocalParameter:
    """One local trainable parameter of one rank (see Miles ``FullParameterPartition``)."""

    name: str
    shape: tuple[int, ...]
    expert: bool
    partition_dim: int | None = None
    partition_sizes: tuple[int, ...] = ()

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class RankParameters:
    """One trainer rank: its parameters, its TP coordinates and its replica groups.

    ``shard_id`` is the rank's own (shape-bound) owner identity, which its
    backend uses in local wire names; ``key`` is the backend's rank key
    (e.g. a Miles topology), returned unchanged in the plan."""

    key: Any
    shard_id: str
    rank: int
    tp_rank: int
    tp_size: int
    dense_ranks: tuple[int, ...]
    expert_ranks: tuple[int, ...] | None
    tensor_parallel_ranks: tuple[int, ...]
    parameters: tuple[LocalParameter, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.tp_rank < self.tp_size or len(self.tensor_parallel_ranks) != self.tp_size:
            raise ShapeFreeLayoutError(f"rank {self.rank}: tensor-parallel coordinates are inconsistent")
        if self.tensor_parallel_ranks[self.tp_rank] != self.rank or self.rank not in self.dense_ranks:
            raise ShapeFreeLayoutError(f"rank {self.rank}: replica groups do not contain the rank")


@dataclass(frozen=True)
class GlobalParameter:
    """One global parameter: its shape and how tensor parallelism splits it.

    ``block_sizes`` are the global sizes of the blocks along
    ``partition_dim`` (one block for a plain split); empty when tensor
    parallelism does not split the parameter."""

    name: str
    shape: tuple[int, ...]
    expert: bool
    partition_dim: int | None = None
    block_sizes: tuple[int, ...] = ()

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def split(self) -> bool:
        return self.partition_dim is not None


def global_parameter(local: LocalParameter, tp_size: int) -> GlobalParameter:
    """The global parameter a rank with ``tp_size`` TP ranks holds part of."""

    if local.partition_dim is None:
        return GlobalParameter(local.name, tuple(local.shape), local.expert)
    if local.expert:
        raise ShapeFreeLayoutError(f"{local.name!r}: a split expert needs expert tensor parallelism 1")
    shape = list(local.shape)
    shape[local.partition_dim] *= tp_size
    return GlobalParameter(
        local.name,
        tuple(shape),
        False,
        local.partition_dim,
        tuple(size * tp_size for size in local.partition_sizes),
    )


@dataclass(frozen=True)
class Unit:
    """One canonical tensor of the shape-free layout (see the module rules)."""

    parameter: GlobalParameter
    index: int
    count: int
    label: str
    owner_class: tuple

    @property
    def name(self) -> str:
        if self.count == 1:
            return self.parameter.name
        return f"tp{self.index}-of-{self.count}.{self.parameter.name}"

    @property
    def shape(self) -> tuple[int, ...]:
        if self.count == 1:
            return self.parameter.shape
        shape = list(self.parameter.shape)
        shape[self.parameter.partition_dim] = sum(size // self.count for size in self.parameter.block_sizes)
        return tuple(shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def spec(self, role: str) -> ParameterSpec:
        return ParameterSpec(role, self.name, self.shape, "float32", self.numel, self.label)

    def local_units(self, tp_rank: int, tp_size: int) -> bool:
        """Whether TP rank ``tp_rank`` of ``tp_size`` holds this unit."""

        if self.count == 1:
            return True
        return self.index // (self.count // tp_size) == tp_rank

    def local_segments(self, local: LocalParameter, tp_rank: int, tp_size: int) -> tuple[tuple[int, int], ...]:
        """Flat ``[start, stop)`` segments of ``local`` whose concatenation is this unit.

        ``local`` is the parameter on TP rank ``tp_rank`` of ``tp_size``,
        which must hold the unit."""

        if not self.local_units(tp_rank, tp_size):
            raise ShapeFreeLayoutError(f"TP rank {tp_rank} of {tp_size} does not hold {self.name!r}")
        if self.count == 1:
            return ((0, local.numel),)
        dim = self.parameter.partition_dim
        per_rank = self.count // tp_size
        position = self.index % per_rank
        outer = math.prod(local.shape[:dim])
        inner = math.prod(local.shape[dim + 1 :])
        length = local.shape[dim]
        segments: list[tuple[int, int]] = []
        for row in range(outer):
            block_start = 0
            for local_size in local.partition_sizes:
                part = local_size // per_rank
                start = (row * length + block_start + position * part) * inner
                stop = start + part * inner
                if segments and segments[-1][1] == start:
                    segments[-1] = (segments[-1][0], stop)
                else:
                    segments.append((start, stop))
                block_start += local_size
        return tuple(segments)


@dataclass(frozen=True)
class OwnedFragment:
    """One fragment of the shape-free layout as its owner rank reads it.

    ``members`` are ``(local parameter name, start, stop)`` flat segments of
    the owner's local tensors, in fragment payload order."""

    fragment_id: int
    owner: int
    numel: int
    members: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ShapeFreePlan:
    """The shape-free layout and one island's owners and fragment members."""

    layout: ParameterLayout
    granularity: Granularity
    units: Mapping[str, Unit]
    ranks: Mapping[int, RankParameters]
    # rank -> {local parameter name: (owner rank, writer ranks)}
    owner_maps: Mapping[int, Mapping[str, tuple[int, tuple[int, ...]]]]
    fragments: tuple[OwnedFragment, ...]
    holders: Mapping[str, tuple[int, ...]] = field(repr=False)

    def owned_fragments(self, rank: int) -> tuple[OwnedFragment, ...]:
        return tuple(fragment for fragment in self.fragments if fragment.owner == rank)

    def fragment_writers(self, fragment_id: int) -> tuple[int, ...]:
        """Every rank that holds the units of one fragment (owner and replicas)."""

        names = {unit_slice.spec.name for unit_slice in self.layout.fragment_slices(fragment_id)}
        holders = {self.holders[name] for name in names}
        if len(holders) != 1:
            raise RuntimeError(f"fragment {fragment_id} mixes holder sets")
        return next(iter(holders))


def global_parameters(ranks: Sequence[RankParameters]) -> dict[str, GlobalParameter]:
    """The global parameters every rank holds part of; every rank must agree."""

    found: dict[str, GlobalParameter] = {}
    for rank in ranks:
        for local in rank.parameters:
            value = global_parameter(local, rank.tp_size)
            seen = found.setdefault(value.name, value)
            if seen != value:
                raise ShapeFreeLayoutError(
                    f"{value.name!r}: ranks disagree on the global parameter ({seen} vs {value})"
                )
    if not found:
        raise ShapeFreeLayoutError("no trainable parameter")
    return found


def _layer_suffix(name: str, split_layers: bool) -> str:
    if not split_layers:
        return ""
    match = _LAYER.search(name)
    if match is not None:
        return f".{'l' if match.group(1) == 'decoder' else 'm'}{int(match.group(2))}"
    return f".p{hashlib.sha256(name.encode()).hexdigest()[:10]}"


def dense_groups(parameters: Sequence[GlobalParameter], groups: int) -> dict[str, int]:
    """``{dense parameter name: group}``: largest first to the lightest group."""

    load = [0] * groups
    assignment = {}
    for parameter in sorted(parameters, key=lambda value: (-value.numel, value.name)):
        index = min(range(groups), key=lambda position: (load[position], position))
        assignment[parameter.name] = index
        load[index] += parameter.numel
    return assignment


def canonical_units(parameters: Mapping[str, GlobalParameter], granularity: Granularity) -> dict[str, Unit]:
    """Every unit of the shape-free layout, by unit name."""

    experts_per_layer: dict[int, set[int]] = {}
    for parameter in parameters.values():
        owner = expert_of(parameter.name)
        if parameter.expert and owner is None:
            raise ShapeFreeLayoutError(f"{parameter.name!r} is an expert parameter without a global expert name")
        if owner is not None:
            experts_per_layer.setdefault(owner[0], set()).add(owner[1])
    for layer, experts in experts_per_layer.items():
        if experts != set(range(len(experts))):
            raise ShapeFreeLayoutError(f"MoE layer {layer} does not hold experts 0..{len(experts) - 1}")
    groups = dense_groups(
        [parameter for parameter in parameters.values() if expert_of(parameter.name) is None],
        granularity.dense_groups,
    )
    units: dict[str, Unit] = {}
    for parameter in parameters.values():
        suffix = _layer_suffix(parameter.name, granularity.split_layers)
        owner = expert_of(parameter.name)
        if owner is not None:
            layer, expert = owner
            if parameter.split:
                raise ShapeFreeLayoutError(f"{parameter.name!r}: experts must not be tensor-parallel split")
            klass = expert * granularity.expert_classes // len(experts_per_layer[layer])
            produced = [Unit(parameter, 0, 1, f"x{klass}{suffix}", ("x", klass, suffix))]
        elif parameter.split:
            count = granularity.tp_units
            if any(size % count for size in parameter.block_sizes):
                raise ShapeFreeLayoutError(
                    f"{parameter.name!r}: blocks {parameter.block_sizes} along dimension "
                    f"{parameter.partition_dim} do not split into {count} TP units"
                )
            group = groups[parameter.name]
            produced = [
                Unit(parameter, index, count, f"d{group}.t{index}{suffix}", ("d", group, suffix))
                for index in range(count)
            ]
        else:
            group = groups[parameter.name]
            produced = [Unit(parameter, 0, 1, f"r{group}{suffix}", ("r", group, suffix))]
        for unit in produced:
            units[unit.name] = unit
    return units


def _local_units(rank: RankParameters, units_by_parameter: Mapping[str, Sequence[Unit]]):
    for local in rank.parameters:
        for unit in units_by_parameter[local.name]:
            if unit.local_units(rank.tp_rank, rank.tp_size):
                yield local, unit


def _assign_owners(
    keys: Mapping[tuple, int],
    ranks: Sequence[int],
) -> dict[tuple, int]:
    """One owner per key (``{(owner class, holders): bytes}``), balanced by bytes.

    Within each holder set the largest keys first go to distinct ranks (so a
    holder set with at least as many keys as ranks gives every rank one),
    then every further key to the least-loaded holder."""

    load = dict.fromkeys(ranks, 0)
    owners: dict[tuple, int] = {}
    by_holders: dict[tuple[int, ...], list[tuple]] = {}
    for key in keys:
        by_holders.setdefault(key[1], []).append(key)
    for holders, members in sorted(by_holders.items()):
        members.sort(key=lambda key: (-keys[key], repr(key[0])))
        free = list(holders)
        for key in members:
            pool = free or list(holders)
            owner = min(pool, key=lambda rank: (load[rank], rank))
            if free:
                free.remove(owner)
            owners[key] = owner
            load[owner] += keys[key]
    return owners


def plan_shape_free_layout(
    ranks: Sequence[RankParameters],
    *,
    algorithm: str,
    components: Sequence[ComponentIdentity],
    role: str,
    granularity: Granularity,
    minimum_fragments: int,
    max_fragment_bytes: int,
    stream_role: str | None = None,
    matrix_merge: str = "avg",
) -> ShapeFreePlan:
    """The shape-free layout of one island and the island's owners and members."""

    ordered = tuple(sorted(ranks, key=lambda value: value.rank))
    if not ordered or len({value.rank for value in ordered}) != len(ordered):
        raise ShapeFreeLayoutError("rank reports are empty or repeat a rank")
    for rank in ordered:
        if granularity.tp_units % rank.tp_size:
            raise ShapeFreeLayoutError(
                f"island TP {rank.tp_size} does not divide the shape-free tp_units {granularity.tp_units}"
            )
    parameters = global_parameters(ordered)
    units = canonical_units(parameters, granularity)
    units_by_parameter: dict[str, list[Unit]] = {}
    for unit in units.values():
        units_by_parameter.setdefault(unit.parameter.name, []).append(unit)

    holders_sets: dict[str, set[int]] = {name: set() for name in units}
    for rank in ordered:
        for _local, unit in _local_units(rank, units_by_parameter):
            holders_sets[unit.name].add(rank.rank)
    missing = sorted(name for name, holders in holders_sets.items() if not holders)
    if missing:
        raise ShapeFreeLayoutError(f"no rank holds {missing[0]!r} ({len(missing)} units)")
    holders = {name: tuple(sorted(value)) for name, value in holders_sets.items()}

    # Every class must have one holder set: its fragments then have one owner.
    by_label: dict[str, set[tuple]] = {}
    for unit in units.values():
        by_label.setdefault(unit.label, set()).add((unit.owner_class, holders[unit.name]))
    for label, keys in sorted(by_label.items()):
        if len(keys) != 1:
            raise ShapeFreeLayoutError(
                f"placement class {label!r} is held by {len(keys)} different rank sets: this island's "
                "shape does not fit the shape-free granularity (EP must divide expert_classes; "
                "PP > 1 needs split_layers)"
            )
    key_bytes: dict[tuple, int] = {}
    for unit in units.values():
        key = (unit.owner_class, holders[unit.name])
        key_bytes[key] = key_bytes.get(key, 0) + unit.numel * 4
    key_owner = _assign_owners(key_bytes, [rank.rank for rank in ordered])
    owned_ranks = set(key_owner.values())
    idle = [rank.rank for rank in ordered if rank.rank not in owned_ranks]
    if idle:
        raise ShapeFreeLayoutError(
            f"ranks {idle} own no shape-free class; raise dense_groups or expert_classes"
        )

    owner_maps: dict[int, dict[str, tuple[int, tuple[int, ...]]]] = {}
    for rank in ordered:
        entries: dict[str, tuple[int, tuple[int, ...]]] = {}
        for local, unit in _local_units(rank, units_by_parameter):
            entry = (key_owner[(unit.owner_class, holders[unit.name])], holders[unit.name])
            if entries.setdefault(local.name, entry) != entry:
                raise ShapeFreeLayoutError(f"{local.name!r} on rank {rank.rank} would have two owners")
            if not set(rank.dense_ranks if not local.expert else rank.expert_ranks or ()) <= set(entry[1]):
                raise ShapeFreeLayoutError(f"{local.name!r} on rank {rank.rank}: holders miss its replica group")
        owner_maps[rank.rank] = entries

    role_specs = [unit.spec(role) for unit in units.values()]
    layout = fit_owner_affine_layout(
        algorithm=algorithm,
        components=components,
        specs=role_specs,
        minimum_fragments=minimum_fragments,
        max_fragment_bytes=max_fragment_bytes,
        stream_role=stream_role,
        matrix_merge=matrix_merge,
    )
    by_rank = {rank.rank: rank for rank in ordered}
    locals_by_rank = {rank.rank: {local.name: local for local in rank.parameters} for rank in ordered}
    fragments = []
    for fragment_id in range(layout.fragments.num_fragments):
        slices = layout.fragment_slices(fragment_id)
        first = units[slices[0].spec.name]
        owner = key_owner[(first.owner_class, holders[first.name])]
        owner_rank = by_rank[owner]
        members: list[tuple[str, int, int]] = []
        for unit_slice in slices:
            unit = units[unit_slice.spec.name]
            if key_owner[(unit.owner_class, holders[unit.name])] != owner:
                raise RuntimeError(f"fragment {fragment_id} mixes owners")
            local = locals_by_rank[owner][unit.parameter.name]
            segments = unit.local_segments(local, owner_rank.tp_rank, owner_rank.tp_size)
            low, high = unit_slice.flat_range
            for start, stop in _clip(segments, low, high):
                if members and members[-1][0] == local.name and members[-1][2] == start:
                    members[-1] = (local.name, members[-1][1], stop)
                else:
                    members.append((local.name, start, stop))
        numel = layout.fragments.fragments[fragment_id].numel
        if sum(stop - start for _name, start, stop in members) != numel:
            raise RuntimeError(f"fragment {fragment_id} members do not add up to its scalar count")
        fragments.append(OwnedFragment(fragment_id, owner, numel, tuple(members)))
    return ShapeFreePlan(
        layout=layout,
        granularity=granularity,
        units=units,
        ranks=by_rank,
        owner_maps=owner_maps,
        fragments=tuple(fragments),
        holders=holders,
    )


def _clip(segments: Sequence[tuple[int, int]], low: int, high: int):
    """The parts of ``segments`` at unit offsets ``[low, high)`` (unit offsets
    count the segments' elements in order)."""

    offset = 0
    for start, stop in segments:
        length = stop - start
        begin, end = max(low, offset), min(high, offset + length)
        if begin < end:
            yield start + begin - offset, start + end - offset
        offset += length
        if offset >= high:
            return


def segments_numel(segments: Sequence[tuple[int, int]]) -> int:
    return sum(stop - start for start, stop in segments)


def unit_from_global(unit: Unit, values):
    """Reference: the unit's values from the global tensor (numpy, for tests and tools)."""

    import numpy as np

    if unit.count == 1:
        return np.asarray(values).reshape(-1).copy()
    dim = unit.parameter.partition_dim
    pieces = []
    offset = 0
    for size in unit.parameter.block_sizes:
        part = size // unit.count
        index = [slice(None)] * values.ndim
        index[dim] = slice(offset + unit.index * part, offset + (unit.index + 1) * part)
        pieces.append(values[tuple(index)])
        offset += size
    return np.concatenate(pieces, axis=dim).reshape(-1).copy()


def local_from_global(parameter: GlobalParameter, values, tp_rank: int, tp_size: int):
    """Reference: what TP rank ``tp_rank`` of ``tp_size`` holds of a global tensor
    (Megatron's split: the ``tp_rank``-th part of every block)."""

    import numpy as np

    if not parameter.split:
        return np.array(values, copy=True)
    dim = parameter.partition_dim
    pieces = []
    offset = 0
    for size in parameter.block_sizes:
        part = size // tp_size
        index = [slice(None)] * values.ndim
        index[dim] = slice(offset + tp_rank * part, offset + (tp_rank + 1) * part)
        pieces.append(values[tuple(index)])
        offset += size
    return np.concatenate(pieces, axis=dim).copy()
