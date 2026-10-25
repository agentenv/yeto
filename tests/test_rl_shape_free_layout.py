"""Shape-free full-parameter layout (yeto.rl.engine.shape_free_layout), on CPU.

A small Nemotron-H-like model (Mamba mixer with packed in_proj/conv1d blocks,
attention, MoE with a router, a shared expert and 8 experts, vocabulary
tables) is split the way Megatron splits it on islands of different shapes:

* A: 8 ranks, TP1 / EP8 / DP8 (expert data-parallel 1);
* B: 8 ranks, TP2 / EP4 (dense data-parallel 4, expert data-parallel 2);
* C: 4 ranks, TP1 / EP4 / DP4;
* D: 8 ranks, TP1 / PP2 / EP4 (with ``split_layers``).

The tests check that all shapes produce one layout hash, that the owners'
exports are bit-identical canonical bytes, and that export -> merge -> apply
leaves every rank of every island holding exactly its slice of the merged
global model, bit for bit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from yeto.rl.engine.shape_free_layout import (
    Granularity,
    LocalParameter,
    RankParameters,
    ShapeFreeLayoutError,
    global_parameter,
    local_from_global,
    plan_shape_free_layout,
    unit_from_global,
)
from yeto.rl.local_learner import ComponentIdentity

COMPONENT = ComponentIdentity("actor", "a" * 40, "c" * 64)
EXPERTS = 8
HIDDEN = 8
L = "module.module.decoder.layers"

# name -> (global shape, partition dim or None, global block sizes or None)
DENSE = {
    "module.module.embedding.word_embeddings.weight": ((16, HIDDEN), 0, None),
    "module.module.output_layer.weight": ((16, HIDDEN), 0, None),
    "module.module.decoder.final_layernorm.weight": ((HIDDEN,), None, None),
    # layer 0: Mamba (in_proj packs z, x, B, C, dt; conv1d packs x, B, C)
    f"{L}.0.mixer.in_proj.weight": ((28, HIDDEN), 0, (8, 8, 4, 4, 4)),
    f"{L}.0.mixer.in_proj.layer_norm_weight": ((HIDDEN,), None, None),
    f"{L}.0.mixer.conv1d_weight": ((16, 1, 4), 0, (8, 4, 4)),
    f"{L}.0.mixer.conv1d_bias": ((16,), 0, (8, 4, 4)),
    f"{L}.0.mixer.A_log": ((4,), 0, None),
    f"{L}.0.mixer.D": ((4,), 0, None),
    f"{L}.0.mixer.dt_bias": ((4,), 0, None),
    f"{L}.0.mixer.norm.weight": ((8,), 0, None),
    f"{L}.0.mixer.out_proj.weight": ((HIDDEN, 8), 1, None),
    # layer 1: attention
    f"{L}.1.self_attention.linear_qkv.weight": ((24, HIDDEN), 0, None),
    f"{L}.1.self_attention.linear_qkv.layer_norm_weight": ((HIDDEN,), None, None),
    f"{L}.1.self_attention.linear_proj.weight": ((HIDDEN, 16), 1, None),
    # layer 2: MoE
    f"{L}.2.mlp.router.weight": ((EXPERTS, HIDDEN), None, None),
    f"{L}.2.pre_mlp_layernorm.weight": ((HIDDEN,), None, None),
    f"{L}.2.mlp.shared_experts.linear_fc1.weight": ((16, HIDDEN), 0, None),
    f"{L}.2.mlp.shared_experts.linear_fc2.weight": ((HIDDEN, 16), 1, None),
}
EXPERT_SHAPES = {"linear_fc1": (12, HIDDEN), "linear_fc2": (HIDDEN, 6)}


def expert_name(projection: str, expert: int) -> str:
    return f"{L}.2.mlp.experts.{projection}.weight{expert}"


def layer_of(name: str) -> int | None:
    if not name.startswith(L + "."):
        return None
    return int(name[len(L) + 1 :].split(".")[0])


def global_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _d, _b) in DENSE.items()}
    for expert in range(EXPERTS):
        for projection, shape in EXPERT_SHAPES.items():
            weights[expert_name(projection, expert)] = rng.standard_normal(shape).astype(np.float32)
    return weights


class Island:
    """One island: per-rank reports and local tensors of the same global model."""

    def __init__(self, world: int, *, tp: int, ep: int, pp: int = 1) -> None:
        self.world, self.tp, self.ep, self.pp = world, tp, ep, pp
        per_stage = world // pp
        dp = per_stage // tp
        expert_dp = per_stage // ep
        self.ranks: list[RankParameters] = []
        self.local_keys: dict[int, list[tuple[str, object, int]]] = {}
        for rank in range(world):
            stage, inner = divmod(rank, per_stage)
            tp_rank, dp_rank = inner % tp, inner // tp
            ep_rank = inner % ep
            base = stage * per_stage
            tp_ranks = tuple(base + dp_rank * tp + index for index in range(tp))
            dense_ranks = tuple(base + index * tp + tp_rank for index in range(dp))
            expert_ranks = tuple(base + ep_rank + index * ep for index in range(expert_dp))
            parameters = []
            for name, (shape, dim, blocks) in DENSE.items():
                if not self._on_stage(name, stage):
                    continue
                if dim is None:
                    parameters.append(LocalParameter(name, shape, False))
                    continue
                blocks = blocks or (shape[dim],)
                local_shape = list(shape)
                local_shape[dim] //= tp
                parameters.append(
                    LocalParameter(name, tuple(local_shape), False, dim, tuple(size // tp for size in blocks))
                )
            if self._on_stage(expert_name("linear_fc1", 0), stage):
                per_rank = EXPERTS // ep
                for expert in range(ep_rank * per_rank, (ep_rank + 1) * per_rank):
                    for projection, shape in EXPERT_SHAPES.items():
                        parameters.append(LocalParameter(expert_name(projection, expert), shape, True))
            self.ranks.append(
                RankParameters(
                    key=rank,
                    shard_id=f"tp{tp_rank}-of-{tp}.pp{stage}-of-{pp}.ep{ep_rank}-of-{ep}.cp0-of-1.dp{dp_rank}-of-{dp}",
                    rank=rank,
                    tp_rank=tp_rank,
                    tp_size=tp,
                    dense_ranks=dense_ranks,
                    expert_ranks=expert_ranks,
                    tensor_parallel_ranks=tp_ranks,
                    parameters=tuple(sorted(parameters)),
                )
            )

    def _on_stage(self, name: str, stage: int) -> bool:
        if self.pp == 1:
            return True
        layer = layer_of(name)
        if layer is None:  # embedding on the first stage, final norm and output on the last
            return stage == (0 if "embedding" in name else self.pp - 1)
        return stage == (0 if layer < 2 else 1)

    def locals_of(self, weights: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
        result = {}
        for rank in self.ranks:
            result[rank.rank] = {
                local.name: local_from_global(
                    global_parameter(local, rank.tp_size), weights[local.name], rank.tp_rank, rank.tp_size
                )
                for local in rank.parameters
            }
        return result

    def plan(self, granularity: Granularity, **kwargs):
        return plan_shape_free_layout(
            self.ranks,
            algorithm="grpo",
            components=(COMPONENT,),
            role="actor",
            granularity=granularity,
            minimum_fragments=kwargs.pop("minimum_fragments", 1),
            max_fragment_bytes=kwargs.pop("max_fragment_bytes", 1024),
            **kwargs,
        )


def export(plan, locals_by_rank) -> list[np.ndarray]:
    """What every owner pushes: its members' flat segments, in order."""

    payloads = []
    for fragment in plan.fragments:
        tensors = locals_by_rank[fragment.owner]
        parts = [tensors[name].reshape(-1)[start:stop] for name, start, stop in fragment.members]
        payloads.append(np.concatenate(parts))
    return payloads


def apply(plan, locals_by_rank, payloads) -> None:
    """What every holder writes: the owner's members, into its own copy."""

    for fragment, payload in zip(plan.fragments, payloads, strict=True):
        for writer in plan.fragment_writers(fragment.fragment_id):
            tensors = locals_by_rank[writer]
            offset = 0
            for name, start, stop in fragment.members:
                flat = tensors[name].reshape(-1)
                flat[start:stop] = payload[offset : offset + stop - start]
                offset += stop - start


def reference_payloads(plan, weights) -> list[np.ndarray]:
    """Every fragment's bytes from the global model (no island involved)."""

    payloads = []
    for fragment_id in range(plan.layout.fragments.num_fragments):
        parts = []
        for unit_slice in plan.layout.fragment_slices(fragment_id):
            unit = plan.units[unit_slice.spec.name]
            low, high = unit_slice.flat_range
            parts.append(unit_from_global(unit, weights[unit.parameter.name])[low:high])
        payloads.append(np.concatenate(parts))
    return payloads


GRANULARITY = Granularity(tp_units=2, expert_classes=8, dense_groups=4)
SHAPES = {
    "A": lambda: Island(8, tp=1, ep=8),
    "B": lambda: Island(8, tp=2, ep=4),
    "C": lambda: Island(4, tp=1, ep=4),
}


def test_every_shape_has_the_same_layout_hash():
    plans = {key: make().plan(GRANULARITY) for key, make in SHAPES.items()}
    hashes = {plan.layout.layout_hash for plan in plans.values()}
    assert len(hashes) == 1
    specs = {tuple(plan.layout.specs) for plan in plans.values()}
    assert len(specs) == 1
    # Units carry no rank identity: only placement classes.
    assert all("-of-" not in spec.shard_id for spec in plans["A"].layout.specs)


def test_layout_hash_depends_on_the_granularity():
    island = SHAPES["A"]()
    hashes = {
        island.plan(GRANULARITY).layout.layout_hash,
        island.plan(Granularity(tp_units=4, expert_classes=8, dense_groups=4)).layout.layout_hash,
        island.plan(Granularity(tp_units=2, expert_classes=8, dense_groups=2)).layout.layout_hash,
    }
    assert len(hashes) == 3


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_owner_exports_are_the_canonical_bytes(shape):
    island = SHAPES[shape]()
    plan = island.plan(GRANULARITY)
    weights = global_weights(1)
    exported = export(plan, island.locals_of(weights))
    expected = reference_payloads(plan, weights)
    assert len(exported) == len(expected) == plan.layout.fragments.num_fragments
    for got, want in zip(exported, expected, strict=True):
        assert got.dtype == np.float32
        assert got.tobytes() == want.tobytes()


def test_every_parameter_has_one_owner_and_every_rank_owns_some():
    for make in SHAPES.values():
        island = make()
        plan = island.plan(GRANULARITY)
        assert {fragment.owner for fragment in plan.fragments} == {rank.rank for rank in island.ranks}
        for rank in island.ranks:
            entries = plan.owner_maps[rank.rank]
            assert set(entries) == {local.name for local in rank.parameters}
            for name, (owner, writers) in entries.items():
                assert rank.rank in writers and owner in writers
                assert plan.owner_maps[owner][name] == (owner, writers)


def test_owner_reads_cover_each_owned_parameter_exactly_once():
    island = SHAPES["A"]()
    plan = island.plan(GRANULARITY)
    for rank in island.ranks:
        owned = {name for name, (owner, _w) in plan.owner_maps[rank.rank].items() if owner == rank.rank}
        cover = {name: np.zeros(math.prod(local.shape), dtype=int) for local in rank.parameters
                 if (name := local.name) in owned}
        for fragment in plan.owned_fragments(rank.rank):
            for name, start, stop in fragment.members:
                cover[name][start:stop] += 1
        assert all((value == 1).all() for value in cover.values())


def test_tp_equal_to_units_reads_whole_local_tensors():
    """With TP = tp_units (and no row slicing) each member is one whole local tensor."""

    island = SHAPES["B"]()  # TP2 = tp_units
    plan = island.plan(GRANULARITY, max_fragment_bytes=1 << 20)
    for fragment in plan.fragments:
        names = [name for name, _start, _stop in fragment.members]
        assert len(names) == len(set(names))
        for name, start, stop in fragment.members:
            local = next(item for item in plan.ranks[fragment.owner].parameters if item.name == name)
            assert (start, stop) == (0, math.prod(local.shape))


def test_tp1_reads_row_parallel_units_as_row_segments():
    """TP1 < tp_units: a dimension-1 split is read row by row (one segment per row)."""

    island = SHAPES["A"]()
    plan = island.plan(GRANULARITY, max_fragment_bytes=1 << 20)
    name = f"{L}.0.mixer.out_proj.weight"
    spans = sorted(
        (start, stop) for fragment in plan.fragments for member, start, stop in fragment.members if member == name
    )
    rows, columns = DENSE[name][0]
    assert spans == sorted((row * columns + half * columns // 2, row * columns + (half + 1) * columns // 2)
                           for row in range(rows) for half in range(2))


@pytest.mark.parametrize("pair", [("A", "B"), ("A", "C"), ("B", "C")])
def test_export_merge_apply_is_bit_exact_across_shapes(pair):
    islands = {key: SHAPES[key]() for key in pair}
    plans = {key: island.plan(GRANULARITY) for key, island in islands.items()}
    assert len({plan.layout.layout_hash for plan in plans.values()}) == 1
    trained = {key: global_weights(10 + index) for index, key in enumerate(pair)}
    exported = {key: export(plans[key], islands[key].locals_of(trained[key])) for key in pair}
    merged = [(a + b) / np.float32(2) for a, b in zip(*(exported[key] for key in pair), strict=True)]
    merged_global = {name: (trained[pair[0]][name] + trained[pair[1]][name]) / np.float32(2) for name in trained[pair[0]]}
    for key in pair:
        stale = islands[key].locals_of({name: np.full_like(value, np.nan) for name, value in merged_global.items()})
        apply(plans[key], stale, merged)
        expected = islands[key].locals_of(merged_global)
        for rank, tensors in stale.items():
            for name, value in tensors.items():
                assert not np.isnan(value).any(), (key, rank, name)
                assert value.tobytes() == expected[rank][name].tobytes(), (key, rank, name)


def test_pipeline_parallel_island_joins_with_split_layers():
    granularity = Granularity(tp_units=2, expert_classes=8, dense_groups=4, split_layers=True)
    flat, staged = Island(8, tp=1, ep=8), Island(8, tp=1, ep=4, pp=2)
    plans = [flat.plan(granularity), staged.plan(granularity)]
    assert plans[0].layout.layout_hash == plans[1].layout.layout_hash
    weights = global_weights(3)
    for island, plan in zip((flat, staged), plans, strict=True):
        exported = export(plan, island.locals_of(weights))
        for got, want in zip(exported, reference_payloads(plan, weights), strict=True):
            assert got.tobytes() == want.tobytes()


def test_pipeline_parallel_without_split_layers_is_refused():
    with pytest.raises(ShapeFreeLayoutError, match="split_layers"):
        Island(8, tp=1, ep=4, pp=2).plan(GRANULARITY)


def test_tp_that_does_not_divide_the_units_is_refused():
    with pytest.raises(ShapeFreeLayoutError, match="does not divide"):
        Island(8, tp=2, ep=4).plan(Granularity(tp_units=1, expert_classes=8, dense_groups=4))


def test_ep_that_does_not_divide_the_expert_classes_is_refused():
    with pytest.raises(ShapeFreeLayoutError, match="expert_classes"):
        Island(8, tp=1, ep=8).plan(Granularity(tp_units=2, expert_classes=4, dense_groups=4))


def test_a_rank_without_a_class_is_refused():
    with pytest.raises(ShapeFreeLayoutError, match="own no shape-free class"):
        Island(8, tp=1, ep=1).plan(Granularity(tp_units=2, expert_classes=1, dense_groups=1))
