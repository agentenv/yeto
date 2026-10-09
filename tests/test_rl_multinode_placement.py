"""rl-multinode-island tasks 1.3-1.5: node rules on PlacementRequest, cell chunking, StartupBundles blocks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from yeto.rl.adapters.miles.bundles import BundleMapError, StartupBundles
from yeto.rl.adapters.miles.placement import PlacementRequest
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
    with pytest.raises(ValueError, match="in-node .* group .* spans nodes"):
        _req(trainer_gpus=4, rollout_gpus=12, gpus_per_engine=1, model_parallel=2, bundle_map=bm)
    # Q1/Q3 ruling: node_parallel (tp*cp) is the in-node group; tp=16 > 8-GPU node -> refused
    with pytest.raises(ValueError, match="in-node .* group .* spans nodes"):
        _req(trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8, node_parallel=16, model_parallel=16)
    with pytest.raises(ValueError, match="in-node .* group .* spans nodes"):
        PlacementRequest("colocated", trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8,
                         gpus_per_node=8, node_parallel=16)


def test_trainer_not_rectangular_rejected():
    with pytest.raises(ValueError, match="GPUs per node must be equal"):
        _req(trainer_gpus=12, rollout_gpus=4, gpus_per_engine=4, model_parallel=2)


def test_pp_group_spanning_nodes_legal():
    # Q3 ruling 2026-10-04: PP may cross nodes. tp=2,pp=2 on a 2x8 island with a
    # 16-GPU trainer: model_parallel (tp*pp*cp) = 4 is irrelevant to node rules;
    # node_parallel = tp*cp = 2 fits a node.
    r = _req(trainer_gpus=16, rollout_gpus=8, gpus_per_engine=8, gpus_per_node=8,
             model_parallel=4, node_parallel=2)
    assert r.trainer_shape() == (2, 8) and r.topology == mn.Topology(3, 8)
    # tp=8,pp=2: each TP group fills a node, the PP group spans the two nodes
    r = _req(trainer_gpus=16, rollout_gpus=8, gpus_per_engine=8, model_parallel=16, node_parallel=8)
    assert r.trainer_shape() == (2, 8)
    # legacy spelling (no node_parallel): model_parallel is still the in-node group
    with pytest.raises(ValueError, match="spans nodes"):
        _req(trainer_gpus=16, rollout_gpus=8, gpus_per_engine=8, model_parallel=16)


def test_ep_group_spanning_nodes_legal():
    # Q1 ruling 2026-10-04: EP may cross nodes. tp=2, ep=8 over 16 trainer GPUs on
    # 2 nodes: the EP group (8 TP groups) spans both nodes and is legal.
    r = _req(trainer_gpus=16, rollout_gpus=8, gpus_per_engine=8, node_parallel=2, expert_parallel=8)
    assert r.trainer_shape() == (2, 8)
    assert _req(expert_parallel=4).trainer_shape() == (1, 8)
    # ep=8 with tp=2 on 8 trainer GPUs is legal (one replica = tp*cp*pp*ceil(ep/tp) = 8);
    # ep=16 on 8 GPUs is refused (not a node rule)
    assert _req(expert_parallel=8).trainer_shape() == (1, 8)
    with pytest.raises(ValueError, match=r"expert parallel 16 needs trainer GPUs / pp"):
        _req(expert_parallel=16)


def test_tp_cp_not_dividing_node_rejected():
    # node_parallel=3 on 8-GPU nodes: 9 trainer GPUs is divisible by 3 but a group
    # (n0:6, n0:7, n1:0) straddles nodes; and per-node share 8 % 3 != 0
    bm = {"trainer": tuple(range(9)), "rollout": tuple(range(9, 16)), "standby": ()}
    with pytest.raises(ValueError, match="spans nodes|does not divide"):
        _req(trainer_gpus=9, rollout_gpus=7, gpus_per_engine=1, node_parallel=3, bundle_map=bm)
    with pytest.raises(ValueError, match=r"not divisible by in-node parallel tp\*cp 3"):
        _req(trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8, node_parallel=3)
    # 2x4 island, trainer n0:0..2 + n1:0..2 (3 per node) with tp*cp=2: 6 % 2 == 0 but 3 % 2 != 0
    bm = {"trainer": (0, 1, 2, 4, 5, 6), "rollout": (3, 7), "standby": ()}
    with pytest.raises(ValueError, match="spans nodes|does not divide the 3 trainer GPUs"):
        _req(trainer_gpus=6, rollout_gpus=2, gpus_per_engine=1, gpus_per_node=4, node_parallel=2,
             bundle_map=bm)
    slots = {"trainer": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], "rollout": [], "standby": []}
    assert mn.node_placement_rejection(slots, node_parallel=1, expert_parallel=1) is None
    # 3 trainer GPUs per node with tp*cp=2: some pair must straddle a node -> refused
    # whatever the order (the per-node divisibility rule is the fallback)
    assert "spans nodes" in mn.node_placement_rejection(slots, node_parallel=2, expert_parallel=1)
    slots = {"trainer": [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2)], "rollout": [], "standby": []}
    assert mn.node_placement_rejection(slots, node_parallel=2, expert_parallel=1) is not None


def test_min_nodes_uses_smallest_replica_tp_cp_ep_pp():
    # tp2 cp1 ep4 pp2 = 16-GPU replica spanning two 8-GPU nodes + one 8-GPU engine -> 3 nodes
    assert mn.min_nodes(trainer_min_gpus=16, rollout_min_gpus=8, standby_gpus=0, gpus_per_node=8,
                        node_parallel=2) == 3
    # the replica may be smaller than a node (tp2 pp1 ep1 -> 2 GPUs)
    assert mn.min_nodes(trainer_min_gpus=2, rollout_min_gpus=8, standby_gpus=0, gpus_per_node=8,
                        node_parallel=2) == 2
    with pytest.raises(mn.TopologyError, match=r"tp\*cp 16 must fit and divide"):
        mn.min_nodes(trainer_min_gpus=16, rollout_min_gpus=8, standby_gpus=0, gpus_per_node=8,
                     node_parallel=16)
    with pytest.raises(mn.TopologyError, match="must fit and divide"):
        mn.min_nodes(trainer_min_gpus=6, rollout_min_gpus=1, standby_gpus=0, gpus_per_node=8,
                     node_parallel=3)
    with pytest.raises(mn.TopologyError, match="not a whole number"):
        mn.min_nodes(trainer_min_gpus=12, rollout_min_gpus=1, standby_gpus=0, gpus_per_node=8,
                     node_parallel=2)


def test_island_not_whole_nodes_rejected():
    with pytest.raises(ValueError, match="not whole 8-GPU nodes"):
        _req(trainer_gpus=6, rollout_gpus=6, gpus_per_engine=6, model_parallel=1)


def test_colocated_multinode_allowed_when_tp_fits_node():
    r = PlacementRequest("colocated", trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8,
                         gpus_per_node=8, model_parallel=8)
    assert r.trainer_shape() == (2, 8)
    # legacy spelling: model_parallel=16 alone is the in-node group -> refused
    with pytest.raises(ValueError, match="spans nodes"):
        PlacementRequest("colocated", trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8,
                         gpus_per_node=8, model_parallel=16)
    # new spelling: tp8 x pp2 (model_parallel 16, node_parallel 8) spans nodes via PP -> legal
    r = PlacementRequest("colocated", trainer_gpus=16, rollout_gpus=16, gpus_per_engine=8,
                         gpus_per_node=8, model_parallel=16, node_parallel=8)
    assert r.trainer_shape() == (2, 8) and r.in_node_parallel == 8


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
    b = StartupBundles(pool_gpus=POOL, views=_views(["A", "A", "B", "B"]), placement_map=MAP, gpus_per_node=2,
                       head_node="A")
    assert b.node_blocks == ("A", "B") and b.head_node == "A"
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
    b = StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP, gpus_per_node=2, node_resolver=resolver,
                       head_node="A")
    assert b.node_blocks == ("A", "B")
    plain = StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP)
    assert plain.node_blocks is None and plain.same_node(("g0", "g3"))
    with pytest.raises(BundleMapError, match="no node information"):
        plain.node_of("g0")


# ---- D3 head pin (ruling 2026-10-03): node 0 = Ray head = trainer, fail closed

def test_startup_bundles_head_pin_fails_closed():
    views = _views(["A", "A", "B", "B"])
    # 2x1 L40S finding (s1-mn-20261003g): the head's block was logical node 1 -> refused
    with pytest.raises(BundleMapError, match="node 0 is not the head"):
        StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP, gpus_per_node=2, head_node="B")
    # more than one node without a head id: the block assertion alone cannot tell -> refused
    with pytest.raises(BundleMapError, match="without the Ray head node id"):
        StartupBundles(pool_gpus=POOL, views=views, placement_map=MAP, gpus_per_node=2)
    # one node per GPU (1 GPU/node): blocks are trivially fine, the head pin still decides
    one = {"actor": _Info(PG, [0], [0], ["W"]), "rollout": _Info(PG, [1], [0], ["H"])}
    with pytest.raises(BundleMapError, match="node 0 is not the head"):
        StartupBundles(pool_gpus=("g0", "g1"), views=one, placement_map={"trainer": (0,), "rollout": (1,)},
                       gpus_per_node=1, head_node="H")
    ok = {"actor": _Info(PG, [1], [0], ["H"]), "rollout": _Info(PG, [0], [0], ["W"])}
    b = StartupBundles(pool_gpus=("g0", "g1"), views=ok, placement_map={"trainer": (0,), "rollout": (1,)},
                       gpus_per_node=1, head_node="H")
    assert b.node_blocks == ("H", "W")
    # single node island: a head id is optional and checked when given
    single = StartupBundles(pool_gpus=POOL, views=_views(["A"] * 4), placement_map=MAP, gpus_per_node=4)
    assert single.node_blocks == ("A",)


def test_head_pinned_bundles_and_sort_key():
    from yeto.rl.engine.multinode import (HEAD_RESOURCE, TopologyError, assert_head_block,
                                          head_first_sort_key, head_pinned_bundles)

    bundles = head_pinned_bundles(4, 2)
    assert bundles[0] == bundles[1] == {"GPU": 1, "CPU": 1, HEAD_RESOURCE: 0.001}
    assert bundles[2] == bundles[3] == {"GPU": 1, "CPU": 1}
    assert head_pinned_bundles(2, 1)[1] == {"GPU": 1, "CPU": 1}
    with pytest.raises(TopologyError):
        head_pinned_bundles(3, 2)
    with pytest.raises(TopologyError):
        head_pinned_bundles(8, 8, share=0.5)
    # the fork sorts by IP: worker 10.0.0.14 < head 10.0.0.22 put the trainer on the worker
    base = lambda x: (list(map(int, x[1].split("."))), x[2])  # noqa: E731
    infos = [(0, "10.0.0.22", 0), (1, "10.0.0.14", 0)]
    assert sorted(infos, key=base)[0][1] == "10.0.0.14"
    assert [i[1] for i in sorted(infos, key=head_first_sort_key("10.0.0.22", base))] == ["10.0.0.22", "10.0.0.14"]
    assert_head_block(("H", "W"), "H")
    with pytest.raises(TopologyError, match="not the Ray head"):
        assert_head_block(("W", "H"), "H")


def test_pin_placement_group_to_head_patches_fork_and_checks(monkeypatch):
    import types

    from yeto.rl.adapters.miles.entry import _ray_head_node, pin_placement_group_to_head
    from yeto.rl.engine.multinode import HEAD_RESOURCE

    nodes = [{"NodeID": "W", "Alive": True, "NodeManagerAddress": "10.0.0.14", "Resources": {"GPU": 1}},
             {"NodeID": "H", "Alive": True, "NodeManagerAddress": "10.0.0.22",
              "Resources": {"GPU": 1, HEAD_RESOURCE: 1.0}},
             {"NodeID": "D", "Alive": False, "NodeManagerAddress": "10.0.0.9",
              "Resources": {HEAD_RESOURCE: 1.0}}]
    assert _ray_head_node(nodes) == ("H", "10.0.0.22")
    with pytest.raises(RuntimeError, match="exactly one alive Ray head"):
        _ray_head_node(nodes[:1])

    seen = {}
    placement = {0: "H", 1: "W"}  # Ray's physical bundle -> node (bundle 0 = 10.0.0.22 = head)

    def fake_ray_pg(bundles, strategy="PACK"):
        seen["bundles"], seen["strategy"] = bundles, strategy
        return "PG"

    mod = types.SimpleNamespace(placement_group=fake_ray_pg,
                                sort_key=lambda x: (list(map(int, x[1].split("."))), x[2]))

    def create(num_gpus):
        if num_gpus == 0:
            return None, [], []
        pg = mod.placement_group([{"GPU": 1, "CPU": 1}] * num_gpus, strategy="PACK")
        infos = [(0, "10.0.0.22", 0), (1, "10.0.0.14", 0)]
        order = [i[0] for i in sorted(infos, key=mod.sort_key)]
        return pg, order, [0] * num_gpus

    mod._create_placement_group = create
    head = pin_placement_group_to_head(1, pg_module=mod, head=("H", "10.0.0.22"), table=lambda pg: placement)
    assert head == "H"
    pg, order, _ = mod._create_placement_group(2)
    assert pg == "PG" and order == [0, 1] and seen["strategy"] == "PACK"
    assert seen["bundles"] == [{"GPU": 1, "CPU": 1, HEAD_RESOURCE: 0.001}, {"GPU": 1, "CPU": 1}]
    assert mod._create_placement_group(0) == (None, [], [])
    # idempotent (one PG per driver)
    assert pin_placement_group_to_head(1, pg_module=mod, head=("H", "10.0.0.22")) == "H"
    # Ray ignored the pin (block 0 landed on the worker): startup is refused
    placement.update({0: "W", 1: "H"})
    with pytest.raises(Exception, match="not the Ray head"):
        mod._create_placement_group(2)


# ---- Q2 (2026-10-04 ruling): mixed rollout/trainer nodes via an explicit bundle map

def test_mixed_2x2_bundle_map_legal_and_shape():
    bm = {"trainer": (0, 2), "rollout": (1,), "standby": (3,)}
    r = PlacementRequest("fixed-partition", trainer_gpus=2, rollout_gpus=1, gpus_per_engine=1,
                         standby_gpus=1, gpus_per_node=2, model_parallel=1, bundle_map=bm)
    assert r.trainer_shape() == (2, 1)
    assert r.role_bundles() == bm


def test_mixed_non_rectangular_bundle_map_rejected():
    bm = {"trainer": (0, 1, 2), "rollout": (3,), "standby": ()}
    with pytest.raises(ValueError, match="GPUs per node must be equal"):
        PlacementRequest("fixed-partition", trainer_gpus=3, rollout_gpus=1, gpus_per_engine=1,
                         gpus_per_node=2, model_parallel=1, bundle_map=bm)


def test_placement_request_shape_must_match_actor_args():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import config as mc

    def _cfg(nodes, per_node, bundle_map=None):
        return SimpleNamespace(parallel=SimpleNamespace(
            actor_num_nodes=nodes, actor_num_gpus_per_node=per_node, tensor_parallel=1,
            pipeline_parallel=1, expert_parallel=1, rollout_num_gpus_per_engine=1,
            dedicated_rollout_gpus=1, standby_gpus=1, rollout_cell_names=(), island_gpus_per_node=2,
            bundle_map=bundle_map, colocated=False))

    bm = {"trainer": (0, 2), "rollout": (1,), "standby": (3,)}
    assert mc.placement_request(_cfg(2, 1, bm)).trainer_shape() == (2, 1)
    with pytest.raises(mc.MilesConfigError, match=r"trainer shape \(2, 1\) .* disagrees .* \(1, 2\)"):
        mc.placement_request(_cfg(1, 2, bm))
    # leading layout: trainer = bundles 0,1 on n0 -> (1, 2); told (2, 1) is refused
    with pytest.raises(mc.MilesConfigError, match="disagrees"):
        mc.placement_request(_cfg(2, 1))
