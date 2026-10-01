"""rl-multinode-island task 1.6: launcher shape derivation, min nodes, island flags."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec


def _args(**kw):
    base = dict(rl_placement="fixed-partition", rollout_num_gpus=8, rl_standby_gpus=0,
                tensor_parallel=2, pipeline_parallel=1, rollout_num_gpus_per_engine=8,
                rl_min_nodes_per_learner=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_trainer_shape_two_nodes():
    (spec,) = parse_gpu_spec("nebius:2x8xh100")
    assert launcher.rl_trainer_shape(_args(), spec) == (1, 8)
    assert launcher.rl_actor_gpus_per_node(_args(), spec) == 8
    (four,) = parse_gpu_spec("nebius:4x8xh100")
    assert launcher.rl_trainer_shape(_args(rollout_num_gpus=16), four) == (2, 8)
    with pytest.raises(ValueError, match="GPUs per node must be equal"):
        launcher.rl_trainer_shape(_args(rollout_num_gpus=4), spec)
    with pytest.raises(ValueError, match="< island GPUs"):
        launcher.rl_trainer_shape(_args(rollout_num_gpus=16), spec)


def test_single_node_shape_unchanged():
    (spec,) = parse_gpu_spec("nebius:8xh100")
    assert launcher.rl_trainer_shape(_args(rollout_num_gpus=2, rl_standby_gpus=2), spec) == (1, 4)
    assert launcher.rl_actor_gpus_per_node(_args(rollout_num_gpus=2, rl_standby_gpus=2), spec) == 4
    assert launcher.rl_trainer_shape(SimpleNamespace(rl_placement="colocated"), spec) == (1, 8)


def test_min_nodes_from_recipe_and_flag():
    (spec,) = parse_gpu_spec("nebius:2x8xh100")
    assert launcher.rl_min_nodes(_args(), spec) == 2
    assert launcher.rl_min_nodes(_args(rl_min_nodes_per_learner=3), spec) == 3
    assert launcher.rl_min_nodes(_args(rollout_num_gpus_per_engine=1, tensor_parallel=1), spec) == 1
    assert launcher.rl_min_nodes(SimpleNamespace(rl_placement="colocated", rl_min_nodes_per_learner=0), spec) == 1
    with pytest.raises(ValueError, match="does not fit"):
        launcher.rl_min_nodes(_args(rollout_num_gpus_per_engine=16), spec)
