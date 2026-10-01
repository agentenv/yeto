"""rl-multinode-island tasks 1.3-1.5: node rules on PlacementRequest, cell chunking, StartupBundles blocks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from yeto.rl.engine.miles_adapter.bundles import BundleMapError, StartupBundles
from yeto.rl.engine.miles_adapter.placement import PlacementRequest
from yeto.rl.engine import multinode as mn


def _req(**kw):
    base = dict(kind="fixed-partition", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8,
                gpus_per_node=8, model_parallel=2)
    base.update(kw)
    return PlacementRequest(**base)


def test_two_node_partition_legal_and_trainer_shape():
    r = _req()
    assert r.topology == mn.Topology(2, 8)
    assert r.trainer_shape() == (1, 8)
    assert r.role_bundles()["rollout"] == tuple(range(8, 16))
    single = PlacementRequest("fixed-partition", trainer_gpus=2, rollout_gpus=2, gpus_per_engine=1)
    assert single.topology is None and single.trainer_shape() == (1, 2)


def test_engine_spanning_nodes_rejected():
    with pytest.raises(ValueError, match="spans nodes"):
        _req(trainer_gpus=6, rollout_gpus=8, gpus_per_engine=4, standby_gpus=2, model_parallel=1)


def test_tp_group_spanning_nodes_rejected():
    # explicit map: trainer (7, 8) straddles n0/n1; checked before the rectangle rule
    bm = {"trainer": (7, 8, 9, 10), "rollout": tuple(range(7)) + tuple(range(11, 16)), "standby": ()}
    with pytest.raises(ValueError, match="model-parallel group .* spans nodes"):
        _req(trainer_gpus=4, rollout_gpus=12, gpus_per_engine=1, model_parallel=2, bundle_map=bm)


def test_trainer_not_rectangular_rejected():
    with pytest.raises(ValueError, match="GPUs per node must be equal"):
        _req(trainer_gpus=12, rollout_gpus=4, gpus_per_engine=4, model_parallel=2)


def test_ep_alignment_in_request():
    assert _req(expert_parallel=4).trainer_shape() == (1, 8)
    with pytest.raises(ValueError, match="expert parallel 8"):
        _req(expert_parallel=8)


def test_island_not_whole_nodes_rejected():
    with pytest.raises(ValueError, match="not whole 8-GPU nodes"):
        _req(trainer_gpus=6, rollout_gpus=6, gpus_per_engine=6, model_parallel=1)


def test_colocated_multinode_allowed_when_tp_fits_node():
    r = PlacementRequest("colocated", trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8,
                         gpus_per_node=8, model_parallel=8)
    assert r.trainer_shape() == (2, 8)
    with pytest.raises(ValueError, match="spans nodes"):
        PlacementRequest("colocated", trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8,
                         gpus_per_node=8, model_parallel=16)


# ---- 1.4 cell chunking by node

def test_cells_T8R8_engine8_one_cell_on_n1():
    r = _req(rollout_cell_names=("c0", "c1"))
    cells = r.placement_map_arg["rollout_cells"]
    assert cells == [{"name": "c0", "bundles": list(range(8, 16)), "start": True},
                     {"name": "c1", "bundles": [], "start": False}]


def test_cells_T8R4S4_engine2_never_cross_nodes():
    r = _req(rollout_gpus=4, standby_gpus=4, gpus_per_engine=2,
             rollout_cell_names=("a", "b", "c", "d", "e"))
    cells = r.placement_map_arg["rollout_cells"]
    assert [(c["bundles"], c["start"]) for c in cells] == [
        ([8, 9], True), ([10, 11], True), ([12, 13], False), ([14, 15], False), ([], False)]
    for c in cells:
        assert len({b // 8 for b in c["bundles"]}) <= 1


def test_chunk_by_node_leftover_and_legacy():
    runs, rest = mn.chunk_by_node([6, 7, 8, 9, 10], mn.Topology(2, 8), 2)
    assert runs == [[6, 7], [8, 9]] and rest == [10]
    runs, rest = mn.chunk_by_node([5, 6, 7, 8], mn.Topology(2, 8), 2)
    assert runs == [[5, 6]] and rest == [7, 8]  # 7|8 straddles n0/n1
    assert mn.chunk_by_node([0, 1, 2], None, 2) == ([[0, 1]], [2])


def test_T12R4_with_bundle_map_is_rejected_as_non_rectangular():
    bm = {"trainer": tuple(range(12)), "rollout": tuple(range(12, 16)), "standby": ()}
    with pytest.raises(ValueError, match="GPUs per node must be equal"):
        PlacementRequest("fixed-partition", trainer_gpus=12, rollout_gpus=4, gpus_per_engine=4,
                         bundle_map=bm, gpus_per_node=8, model_parallel=2)


# ---- 1.5 StartupBundles node blocks

@dataclass
class _Info:
    pg: object
    pg_reordered_bundle_indices: list
    pg_reordered_gpu_ids: list
    pg_reordered_node_ids: list | None = None


PG = object()
POOL = tuple(f"g{i}" for i in range(4))
MAP = {"trainer": (0, 1), "rollout": (2, 3), "standby": ()}


def _views(nodes):
    return {"actor": _Info(PG, [0, 1], [0, 1], nodes[:2]), "rollout": _Info(PG, [2, 3], [0, 1], nodes[2:])}


def test_startup_bundles_blocks_ok_and_node_of():
    b = StartupBundles(pool_gpus=POOL, views=_views(["A", "A", "B", "B"]), placement_map=MAP, gpus_per_node=2)
    assert b.node_blocks == ("A", "B")
    assert b.node_of("g1") == "A" and b.node_of("g2") == "B"
    assert b.same_node(("g2", "g3")) and not b.same_node(("g1", "g2"))


def test_startup_bundles_unblocked_rejected():
    with pytest.raises(BundleMapError, match="not node-blocked"):
        StartupBundles(pool_gpus=POOL, views=_views(["A", "B", "A", "B"]), placement_map=MAP, gpus_per_node=2)
    with pytest.raises(BundleMapError, match="node repeats"):
        StartupBundles(pool_gpus=POOL, views=_views(["A", "A", "A", "A"]), placement_map=MAP, gpus_per_node=2)


def test_startup_bundles_missing_node_info_fails_closed():
    with pytest.raises(BundleMapError, match="no node id"):
        StartupBundles(pool_gpus=POOL, views=_views([None] * 4) | {}, placement_map=MAP, gpus_per_node=2,
                       node_ids=None) if False else _no_nodes()


def _no_nodes():
    views = {"actor": _Info(PG, [0, 1], [0, 1]), "rollout": _Info(PG, [2, 3], [0, 1])}
    StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP, gpus_per_node=2)


def test_startup_bundles_node_resolver_and_single_node_untouched():
    views = {"actor": _Info(PG, [0, 1], [0, 1]), "rollout": _Info(PG, [2, 3], [0, 1])}
    resolver = lambda pg, bundle: {0: "A", 1: "A", 2: "B", 3: "B"}[bundle]  # noqa: E731
    b = StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP, gpus_per_node=2, node_resolver=resolver)
    assert b.node_blocks == ("A", "B")
    plain = StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP)
    assert plain.node_blocks is None and plain.same_node(("g0", "g3"))
    with pytest.raises(BundleMapError, match="no node information"):
        plain.node_of("g0")
