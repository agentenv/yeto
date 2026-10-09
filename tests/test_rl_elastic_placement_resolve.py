"""S8 M3 regression: elastic config placements (nested per-engine, n<k>:<g> slots)
resolve to the startup description's GPU ids instead of crashing after the commit."""

from __future__ import annotations

import pytest

from yeto.rl.adapters.miles.elastic_placement import ElasticPlacement, PlacementPlanError
from yeto.rl.engine.ports import PlacementDescription


class _Base:
    def __init__(self, trainer, rollout, standby):
        self._d = PlacementDescription("fixed-partition", tuple(trainer), tuple(rollout),
                                       {"standby_gpus": tuple(standby)})

    def describe(self):
        return self._d


def _m3(**kw):
    # M3 2x2 layout: trainer bundles [0,2], rollout [1], standby [3] (logical ids)
    return ElasticPlacement(_Base(["bundle0", "bundle2"], ["bundle1"], ["bundle3"]), **kw)


def test_up_edge_with_nested_slot_config_resolves_to_bundles():
    p = _m3(gpus_per_node=2)
    cur = p.describe()
    plan = PlacementDescription(cur.kind, cur.trainer_gpus, [["n0:1"], ["n1:1"]], dict(cur.extra))
    out = p.reconfigure(plan, epoch=1)
    assert out.rollout_gpus == ("bundle1", "bundle3")
    assert out.extra["standby_gpus"] == ()
    down = PlacementDescription(cur.kind, cur.trainer_gpus, [["n0:1"]], dict(cur.extra))
    out = p.reconfigure(down, epoch=2)
    assert out.rollout_gpus == ("bundle1",) and out.extra["standby_gpus"] == ("bundle3",)


def test_restore_committed_accepts_nested_config_and_ints():
    p = _m3(gpus_per_node=2)
    assert p.restore_committed((["n0:1"], ["n1:1"]), epoch=1).rollout_gpus == ("bundle1", "bundle3")
    q = _m3()
    assert q.restore_committed(([1], [3]), epoch=1).rollout_gpus == ("bundle1", "bundle3")


def test_pool_uuid_spelling_maps_slots_through_bundle_order():
    base = _Base(["u0", "u2"], ["u1"], ["u3"])
    p = ElasticPlacement(base, pool_gpus=("u0", "u1", "u2", "u3"), gpus_per_node=2)
    cur = p.describe()
    out = p.reconfigure(PlacementDescription(cur.kind, cur.trainer_gpus, [["n1:1"], ["u1"]],
                                             dict(cur.extra)), epoch=1)
    assert out.rollout_gpus == ("u3", "u1")


def test_unresolvable_or_overlapping_entries_are_refused_cleanly():
    p = _m3()  # no topology: slots cannot resolve -> outside the pool, a PlacementPlanError
    cur = p.describe()
    with pytest.raises(PlacementPlanError, match="outside the pool"):
        p.reconfigure(PlacementDescription(cur.kind, cur.trainer_gpus, [["n1:1"]], {}), epoch=1)
    q = _m3(gpus_per_node=2)
    with pytest.raises(PlacementPlanError, match="duplicate"):
        q.reconfigure(PlacementDescription(cur.kind, cur.trainer_gpus, [["n0:1"], [1]], {}), epoch=1)
    with pytest.raises(PlacementPlanError, match="overlap"):
        q.reconfigure(PlacementDescription(cur.kind, cur.trainer_gpus, [["n1:0"]], {}), epoch=1)


def test_single_node_island_resolves_n0_slots_without_topology():
    """S11 H100 e1 regression: single node, 8 physical GPUs with 4 allocated
    (--rl-island-use-gpus-per-node 4, no --rl-island-gpus-per-node -> topology None).
    T2R1S1 -> T2R2S0 from resources-1x4-h200.json must pass the post-commit
    bookkeeping instead of "rollout GPUs ['n0:2', 'n0:3'] are outside the pool"."""
    import json
    from pathlib import Path

    res = json.loads((Path(__file__).parent / "multinode_gpu" / "resources-1x4-h200.json").read_text())
    p = ElasticPlacement(_Base(["bundle0", "bundle1"], ["bundle2"], ["bundle3"]), gpus_per_node=None)
    cur = p.describe()
    up = res["configs"]["T2R2S0"]["placement"]["rollout"]
    out = p.reconfigure(PlacementDescription(cur.kind, cur.trainer_gpus, up, dict(cur.extra)), epoch=1)
    assert out.rollout_gpus == ("bundle2", "bundle3") and out.extra["standby_gpus"] == ()
    down = res["configs"]["T2R1S1"]["placement"]["rollout"]
    out = p.reconfigure(PlacementDescription(cur.kind, cur.trainer_gpus, down, dict(cur.extra)), epoch=2)
    assert out.rollout_gpus == ("bundle2",) and out.extra["standby_gpus"] == ("bundle3",)
    assert p.restore_committed(tuple(up), epoch=3).rollout_gpus == ("bundle2", "bundle3")
