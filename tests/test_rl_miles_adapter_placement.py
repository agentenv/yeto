"""CPU tests for miles_adapter.placement (task 3.6)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.placement import (
    MilesPlacement,
    PlacementRequest,
    PlacementRewriteError,
    check_placement_not_rewritten,
)
from yeto.rl.engine.ports import Placement

COLOCATED = PlacementRequest("colocated", trainer_gpus=4, rollout_gpus=4, gpus_per_engine=2)
PARTITION = PlacementRequest("fixed-partition", trainer_gpus=2, rollout_gpus=2, gpus_per_engine=1)


def args(**kw):
    base = dict(
        colocate=True, actor_num_nodes=1, actor_num_gpus_per_node=4, rollout_num_gpus=4,
        rollout_num_gpus_per_engine=2, debug_rollout_only=False, debug_train_only=False,
        rollout_external=False, eval_num_gpus=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def pg(bundles, gpus):
    return (object(), list(bundles), list(gpus))


def test_colocated_description():
    groups = {"actor": pg(range(4), [3, 2, 1, 0]), "rollout": pg(range(4), [3, 2, 1, 0])}
    placement = MilesPlacement(COLOCATED, groups)
    assert isinstance(placement, Placement)
    d = placement.describe()
    assert d.kind == "colocated"
    assert d.trainer_gpus == d.rollout_gpus == ("bundle0:gpu3", "bundle1:gpu2", "bundle2:gpu1", "bundle3:gpu0")


def test_fixed_partition_description():
    groups = {"actor": pg(range(4), range(4)), "rollout": pg([2, 3], [2, 3])}
    d = MilesPlacement(PARTITION, groups).describe()
    assert d.kind == "fixed-partition"
    assert d.trainer_gpus == ("bundle0:gpu0", "bundle1:gpu1")
    assert d.rollout_gpus == ("bundle2:gpu2", "bundle3:gpu3")
    assert not set(d.trainer_gpus) & set(d.rollout_gpus)


def test_description_from_parsed_args_both_kinds():
    d = MilesPlacement.from_parsed_args(COLOCATED, args()).describe()
    assert d.trainer_gpus == d.rollout_gpus and d.extra["logical"]
    p = args(colocate=False, actor_num_gpus_per_node=2, rollout_num_gpus=2, rollout_num_gpus_per_engine=1)
    d = MilesPlacement.from_parsed_args(PARTITION, p).describe()
    assert d.trainer_gpus == ("bundle0", "bundle1") and d.rollout_gpus == ("bundle2", "bundle3")


def test_unrewritten_args_pass():
    check_placement_not_rewritten(COLOCATED, args())
    check_placement_not_rewritten(
        PARTITION,
        args(colocate=False, actor_num_gpus_per_node=2, rollout_num_gpus=2, rollout_num_gpus_per_engine=1),
    )


@pytest.mark.parametrize(
    "request_, overrides, needle",
    [
        # partition requested but normalization produced colocate
        (PARTITION, dict(colocate=True, actor_num_gpus_per_node=2, rollout_num_gpus=2,
                         rollout_num_gpus_per_engine=1), "colocate=True"),
        # colocate requested but debug-rollout-only turned it off
        (COLOCATED, dict(colocate=False, debug_rollout_only=True), "debug_rollout_only"),
        # partition size rewritten
        (PARTITION, dict(colocate=False, actor_num_gpus_per_node=2, rollout_num_gpus=0,
                         rollout_num_gpus_per_engine=1), "rollout GPUs 0"),
        (COLOCATED, dict(rollout_external=True), "rollout_external"),
        (COLOCATED, dict(eval_num_gpus=2), "eval_num_gpus"),
        (COLOCATED, dict(rollout_num_gpus_per_engine=4), "GPUs per engine"),
    ],
)
def test_rewrite_detected(request_, overrides, needle):
    with pytest.raises(PlacementRewriteError, match=needle):
        check_placement_not_rewritten(request_, args(**overrides))


def test_rewritten_placement_group_rejected():
    overlap = {"actor": pg(range(4), range(4)), "rollout": pg([1, 2], [1, 2])}
    with pytest.raises(PlacementRewriteError):
        MilesPlacement(PARTITION, overlap)
    short = {"actor": pg(range(2), range(2)), "rollout": pg(range(2), range(2))}
    with pytest.raises(PlacementRewriteError):
        MilesPlacement(COLOCATED, short)


def test_request_validation():
    with pytest.raises(ValueError):
        PlacementRequest("colocated", trainer_gpus=2, rollout_gpus=4, gpus_per_engine=1)
    with pytest.raises(ValueError):
        PlacementRequest("elastic", trainer_gpus=2, rollout_gpus=2, gpus_per_engine=1)
    with pytest.raises(ValueError):
        PlacementRequest("fixed-partition", trainer_gpus=2, rollout_gpus=3, gpus_per_engine=2)


# -- rl-infra-spec 2.1 / 2.1a: standby and explicit bundle map (fork-M1) -------


def _part_args(**kw):
    return args(colocate=False, actor_num_gpus_per_node=2, rollout_num_gpus=2,
                rollout_num_gpus_per_engine=1, **kw)


def test_bundle_map_is_validated_like_fork_m1():
    base = dict(kind="fixed-partition", trainer_gpus=2, rollout_gpus=2, gpus_per_engine=1,
                standby_gpus=1)
    ok = PlacementRequest(**base, bundle_map={"trainer": [4, 0], "rollout": [1, 2], "standby": [3]})
    assert ok.role_bundles()["trainer"] == (4, 0)
    for bad, why in (
        ({"trainer": [0, 0], "rollout": [1, 2], "standby": [3]}, "repeats"),
        ({"trainer": [0, 9], "rollout": [1, 2], "standby": [3]}, "outside"),
        ({"trainer": [0, 1], "rollout": [1, 2], "standby": [3]}, "overlaps"),
        ({"trainer": [0], "rollout": [1, 2], "standby": [3]}, "requested"),
        ({"trainer": [0, 1], "rollout": [2, 3], "spare": [4]}, "unknown"),
    ):
        with pytest.raises(ValueError, match=why):
            PlacementRequest(**base, bundle_map=bad)
    with pytest.raises(ValueError, match="colocated"):
        PlacementRequest("colocated", 2, 2, 1, standby_gpus=1)
    # default: no standby -> no map -> upstream offset layout (unchanged)
    assert PARTITION.placement_map is None


def test_standby_placement_describes_m1_roles_and_detects_rewrites():
    req = PlacementRequest("fixed-partition", 2, 2, 1, standby_gpus=2)
    pm = req.placement_map
    assert pm == {"trainer": [0, 1], "rollout": [2, 3], "standby": [4, 5]}
    import json

    d = MilesPlacement.from_parsed_args(req, _part_args(yeto_placement_map=json.dumps(pm))).describe()
    assert d.trainer_gpus == ("bundle0", "bundle1")
    assert d.rollout_gpus == ("bundle2", "bundle3")
    assert d.extra["standby_gpus"] == ("bundle4", "bundle5")
    assert d.extra["weight_transport"] == "collective-broadcast"  # neutral (decoupling 2.6)
    with pytest.raises(PlacementRewriteError, match="placement map"):
        check_placement_not_rewritten(req, _part_args())  # Miles dropped the map
    # physical M1 output: actor holds trainer bundles only, standby separate
    groups = {
        "actor": (None, [5, 1], [5, 1]),
        "rollout": (None, [2, 3], [2, 3]),
        "standby": (None, [0, 4], [0, 4]),
    }
    phys = MilesPlacement(req, groups).describe()
    assert phys.trainer_gpus == ("bundle5:gpu5", "bundle1:gpu1")
    groups["standby"] = (None, [0, 3], [0, 3])
    with pytest.raises(PlacementRewriteError, match="overlapping"):
        MilesPlacement(req, groups)
