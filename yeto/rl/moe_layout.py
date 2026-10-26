"""Mixture-of-experts parameters at the full-parameter sync boundary.

Each Megatron rank exports its own FP32 master shard (Miles
``full_parameter_state``). With expert parallelism (EP) a rank holds only its
local experts, and Miles names them by their global expert index
(``decoder.layers.L.mlp.experts.linear_fc1.weightE``). Each rank therefore
pushes its own experts and owner-affine fragments never gather experts
across ranks: no extra collective, no extra device memory, and the bytes on
the wire are exactly the local masters.

This module checks that an island's manifests partition the experts exactly
once. Expert tensor parallelism is pinned to 1, so with data parallelism 1
the experts are replicated on ``TP x CP / EP`` ranks when EP is smaller than
``TP x CP``. Replicated experts would be pushed once per replica and could
drift apart under block-quantized pushes, so they are refused.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# Megatron global names (Miles ``named_params_and_buffers`` with
# ``convert_to_global_name=True``, TE grouped GEMM experts):
# ``decoder.layers.L.mlp.experts.linear_fc1.weightE`` with E the global expert
# index. ``global_expert_name`` writes this form; ``_EXPERT`` reads it, also
# inside a wire name (``role::shard::name``).
_EXPERT = re.compile(
    r"(?:^|[.:])decoder\.layers\.(\d+)\.mlp\.experts\.(linear_fc1|linear_fc2)\.weight(\d+)\Z"
)
# Megatron SequentialMLP names its experts by local index; the global index is
# ep_rank * num_local_experts + local index (megatron/core/transformer/moe/moe_layer.py,
# ``local_expert_indices_offset``).
_SEQUENTIAL_EXPERT = re.compile(r"^(.*\.mlp\.experts)\.local_experts\.(\d+)\.(linear_fc[12])\.weight\Z")
_ROUTER = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.mlp\.router\.weight\Z")
# Latent MoE (Nemotron-3 Super): the router reads the hidden state and the
# experts read its projection ``fc1_latent_proj`` ([latent, hidden], duplicated
# on every TP rank, megatron/core/transformer/moe/moe_layer.py).
_LATENT_FC1 = re.compile(r"(?:^|\.)decoder\.layers\.(\d+)\.mlp\.fc1_latent_proj\.weight\Z")
_SHARD_SIZES = re.compile(r"([a-z]+)\d+-of-(\d+)")


@dataclass(frozen=True)
class ExpertPartition:
    """How one island's experts are spread over its topology owners."""

    num_experts: int
    moe_layers: tuple[int, ...]
    # Owner (role, shard ID) -> number of (layer, expert) pairs it holds.
    experts_per_owner: tuple[tuple[tuple[str, str], int], ...]


def global_expert_name(experts_prefix: str, projection: str, expert: int) -> str:
    """The grouped-GEMM global name of one expert projection (see ``_EXPERT``)."""
    return f"{experts_prefix}.{projection}.weight{expert}"


def sequential_expert_global_name(name: str, *, ep_rank: int, num_local_experts: int) -> str:
    """Rewrite a SequentialMLP local expert name to its global grouped-GEMM name,
    the form the trainer's weight sync and the HF converters use. Other names
    are returned unchanged."""
    match = _SEQUENTIAL_EXPERT.match(name)
    if match is None:
        return name
    prefix, local, projection = match.groups()
    return global_expert_name(prefix, projection, ep_rank * num_local_experts + int(local))


def is_expert_parameter(name: str) -> bool:
    return _EXPERT.search(name) is not None


def parallel_sizes(shard_id: str) -> dict[str, int]:
    """Parallel sizes encoded in a Miles shard ID, e.g. ``tp0-of-2.ep1-of-2``."""

    return {axis: int(size) for axis, size in _SHARD_SIZES.findall(shard_id)}


def check_expert_partition(specs: Iterable) -> ExpertPartition | None:
    """Check that every expert of every MoE layer has exactly one owner.

    ``specs`` are objects with ``role``, ``shard_id``, ``name`` and ``shape``
    (yeto ``ParameterSpec`` or Miles ``FullParameterShardSpec``). Returns
    ``None`` for a dense model. Raises ``ValueError`` when an expert is
    missing, is held by more than one owner, has its two projections on
    different owners, or disagrees with its router or its peers in shape.
    With a latent MoE layer the router reads the hidden size and the experts
    the latent size of the layer's ``fc1_latent_proj``.
    """

    owners: dict[tuple[str, int, str, int], tuple[str, str]] = {}
    shapes: dict[str, set[tuple[int, ...]]] = {"linear_fc1": set(), "linear_fc2": set()}
    routers: dict[tuple[str, int], tuple[int, ...]] = {}
    latents: dict[tuple[str, int], set[tuple[int, ...]]] = {}
    for spec in specs:
        owner = (spec.role, spec.shard_id)
        router = _ROUTER.search(spec.name)
        if router is not None:
            routers[(spec.role, int(router.group(1)))] = tuple(spec.shape)
            continue
        latent = _LATENT_FC1.search(spec.name)
        if latent is not None:
            latents.setdefault((spec.role, int(latent.group(1))), set()).add(tuple(spec.shape))
            continue
        match = _EXPERT.search(spec.name)
        if match is None:
            continue
        layer, projection, expert = int(match.group(1)), match.group(2), int(match.group(3))
        key = (spec.role, layer, projection, expert)
        if key in owners:
            sizes = parallel_sizes(spec.shard_id)
            raise ValueError(
                f"expert {expert} of layer {layer} ({projection}) is listed by both "
                f"{owners[key][1]!r} and {spec.shard_id!r}: full-parameter sync needs "
                "each expert listed by exactly one rank. A Miles pin with data-parallel "
                "replicas gives each expert one owner; an older one needs "
                "expert-model-parallel size = tensor-parallel x context-parallel size "
                f"(this island: {sizes})"
            )
        owners[key] = owner
        shapes[projection].add(tuple(spec.shape))
    if not owners:
        return None

    for projection, seen in shapes.items():
        if len(seen) != 1:
            raise ValueError(f"expert {projection} shapes differ across experts: {sorted(seen)}")
    (fc1,), (fc2,) = shapes["linear_fc1"], shapes["linear_fc2"]
    # Gated experts (SwiGLU) stack gate and up in fc1; ungated ones (Nemotron-H
    # squared ReLU) hold only up.
    if (
        len(fc1) != 2 or len(fc2) != 2
        or fc1[0] not in (fc2[1], 2 * fc2[1]) or fc1[1] != fc2[0]
    ):
        raise ValueError(
            f"expert shapes are not a gated or ungated MLP: linear_fc1 {fc1}, linear_fc2 {fc2}"
        )

    by_layer: dict[tuple[str, int], dict[int, tuple[str, str]]] = {}
    for (role, layer, projection, expert), owner in owners.items():
        experts = by_layer.setdefault((role, layer), {})
        if experts.setdefault(expert, owner) != owner:
            raise ValueError(
                f"expert {expert} of layer {layer} has its projections on two owners"
            )
    num_experts = None
    for (role, layer), experts in sorted(by_layer.items()):
        count = len(experts)
        for projection in ("linear_fc1", "linear_fc2"):
            ids = {e for (r, lay, p, e) in owners if (r, lay, p) == (role, layer, projection)}
            if ids != set(range(count)):
                raise ValueError(
                    f"layer {layer} {projection} experts are not 0..{count - 1}: {sorted(ids)}"
                )
        if num_experts is None:
            num_experts = count
        elif count != num_experts:
            raise ValueError(f"layer {layer} has {count} experts, others have {num_experts}")
        router_input = fc1[1]
        latent = latents.get((role, layer))
        if latent is not None:
            shape = next(iter(latent))
            if len(latent) != 1 or len(shape) != 2 or shape[0] != fc1[1]:
                raise ValueError(
                    f"layer {layer} latent projection shapes {sorted(latent)} do not feed "
                    f"experts of input size {fc1[1]}"
                )
            router_input = shape[1]
        router = routers.get((role, layer))
        if router is not None and (len(router) != 2 or router != (count, router_input)):
            raise ValueError(
                f"layer {layer} router shape {router} does not match {count} experts "
                f"of hidden size {router_input}"
            )

    per_owner: dict[tuple[str, str], int] = {}
    for experts in by_layer.values():
        for owner in experts.values():
            per_owner[owner] = per_owner.get(owner, 0) + 1
    return ExpertPartition(
        num_experts=num_experts,
        moe_layers=tuple(sorted({layer for _, layer in by_layer})),
        experts_per_owner=tuple(sorted(per_owner.items())),
    )


def describe_topology(shard_ids: Sequence[str]) -> str:
    """A short label of an island's parallel sizes for error messages."""

    sizes = {tuple(sorted(parallel_sizes(shard).items())) for shard in shard_ids}
    if len(sizes) != 1:
        return f"mixed {sorted(sizes)}"
    return ",".join(f"{axis}={size}" for axis, size in next(iter(sizes)))


# --- routed-token counts for expert token weights ---------------------------


def expert_of(name: str) -> tuple[int, int] | None:
    """``(layer, expert)`` of an expert projection (global or wire name), else
    ``None``."""

    match = _EXPERT.search(name)
    return None if match is None else (int(match.group(1)), int(match.group(3)))


@dataclass(frozen=True)
class RoutedTokens:
    """Tokens routed to each expert of each MoE layer, summed over the
    island's ranks.

    Island ranks that see the same tokens (tensor parallelism without
    sequence parallelism) count them more than once, but every expert of a
    layer by the same factor, so the share of an expert within its layer, the
    only quantity the syncer uses, is exact.
    """

    # layer -> per-expert counts; every layer has the same number of experts.
    counts: tuple[tuple[int, tuple[int, ...]], ...] = ()

    @classmethod
    def from_mapping(cls, counts: Mapping[int, Sequence[int]]) -> RoutedTokens:
        rows = []
        for layer, row in sorted(counts.items()):
            row = tuple(int(value) for value in row)
            if int(layer) < 0 or not row or any(value < 0 for value in row):
                raise ValueError(f"invalid routed-token counts for layer {layer}")
            rows.append((int(layer), row))
        if len({len(row) for _, row in rows}) > 1:
            raise ValueError("MoE layers report different expert counts")
        return cls(tuple(rows))

    def as_dict(self) -> dict[int, tuple[int, ...]]:
        return dict(self.counts)

    def __add__(self, other: RoutedTokens) -> RoutedTokens:
        merged = self.as_dict()
        for layer, row in other.counts:
            mine = merged.get(layer)
            if mine is not None and len(mine) != len(row):
                raise ValueError(f"layer {layer} changed its expert count")
            merged[layer] = row if mine is None else tuple(a + b for a, b in zip(mine, row))
        return RoutedTokens.from_mapping(merged)

    def __sub__(self, earlier: RoutedTokens) -> RoutedTokens:
        """Counts since ``earlier`` (a snapshot of the same cumulative counter)."""

        before = earlier.as_dict()
        rows = {}
        for layer, row in self.counts:
            old = before.pop(layer, (0,) * len(row))
            if len(old) != len(row) or any(b > a for a, b in zip(row, old)):
                raise ValueError(f"routed-token counter of layer {layer} moved backwards")
            rows[layer] = tuple(a - b for a, b in zip(row, old))
        if before:
            raise ValueError(f"routed-token counts lost layers {sorted(before)}")
        return RoutedTokens.from_mapping(rows)


def fragment_expert_tags(
    tensor_names: Sequence[str], interval: RoutedTokens
) -> tuple[tuple[int, int, int], ...]:
    """Per-tensor routed-token tags of one push (``yeto.protocol.encode_expert_tags``).

    ``interval`` holds the tokens routed during the push's outer interval. A
    dense tensor is tagged ``(0, 0, 0)``; an expert tensor
    ``(routed to the expert, routed in its layer, experts in its layer)``.
    """

    counts = interval.as_dict()
    tags = []
    for name in tensor_names:
        owner = expert_of(name)
        if owner is None:
            tags.append((0, 0, 0))
            continue
        layer, expert = owner
        row = counts.get(layer)
        if row is None or expert >= len(row):
            raise ValueError(
                f"no routed-token count for expert {expert} of MoE layer {layer}; "
                "is the router counter installed on every rank?"
            )
        tags.append((row[expert], sum(row), len(row)))
    return tuple(tags)
