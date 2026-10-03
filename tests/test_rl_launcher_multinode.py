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


# ---- 1.6 task level (D1/D6) and 1.8 teardown (D10)

def test_multinode_env_prelude():
    assert launcher.multinode_env_prelude("nebius", 1) == ""
    nebius = launcher.multinode_env_prelude("nebius", 2)
    assert "NCCL_SOCKET_IFNAME=eth0" in nebius and "GLOO_SOCKET_IFNAME=eth0" in nebius
    assert "NCCL_IB_DISABLE" in nebius
    assert "SOCKET_IFNAME" not in launcher.multinode_env_prelude("aws", 2)


def test_two_node_island_task(monkeypatch):
    import sys
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _args(("--gpu", "nebius:2x8xh100", "--rl-engine", "ports", "--rl-placement", "fixed-partition",
                  "--rl-rollout-gpus", "8", "--rollout-num-gpus-per-engine", "8", "--tensor-parallel", "2"))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    task = launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")
    assert task.num_nodes == 2 and task.resources.network_tier == "best"
    assert "--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rl-island-gpus-per-node 8" in task.run
    assert "export NCCL_SOCKET_IFNAME=eth0" in task.run
    head, _, worker = task.run.partition("\nelse\n")
    assert "ray start --head" in head
    # G1 (2026-10-03): the dashboard/state API must be reachable from worker nodes
    assert "--include-dashboard=true" in head and "--dashboard-host=0.0.0.0" in head
    assert worker.index("trap stop_miles_ray EXIT") < worker.index('ray start --address="$MASTER_ADDR:6379"')
    assert "worker could not join the Ray head" in worker


def test_single_node_task_has_no_multinode_additions(monkeypatch):
    import sys
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _args(("--gpu", "nebius:8xh100", "--rl-engine", "ports", "--rl-placement", "fixed-partition",
                  "--rl-rollout-gpus", "4", "--tensor-parallel", "2"))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    task = launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")
    assert task.num_nodes == 1
    assert "--actor-num-nodes 1 --actor-num-gpus-per-node 4" in task.run
    assert "--rl-island-gpus-per-node" not in task.run and "NCCL_SOCKET_IFNAME" not in task.run


def test_below_min_nodes_refused_at_prepare():
    from test_rl_launcher import _args, _prepare_rl_args

    args = _args(("--gpu", "nebius:8xh100", "--rl-engine", "ports", "--rl-placement", "fixed-partition",
                  "--rl-rollout-gpus", "4", "--rollout-num-gpus-per-engine", "4", "--tensor-parallel", "2",
                  "--rl-min-nodes-per-learner", "2"))
    with pytest.raises(ValueError, match="at least 2 node"):
        _prepare_rl_args(args)


class _Down:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


def test_teardown_two_nodes_confirms_each_instance(capsys):
    live = [["i-a", "i-b"], []]
    down = _Down()
    ok = launcher.terminate_and_verify(None, "isl", probe=lambda: live.pop(0), down=down,
                                       sleep_fn=lambda s: None, num_nodes=2)
    assert ok and down.calls == 1
    out = capsys.readouterr().out
    assert "node instance i-a confirmed terminated" in out and "node instance i-b confirmed terminated" in out


def test_teardown_two_nodes_one_unconfirmed_is_false(capsys):
    live = [["i-a", "i-b"], ["i-b"], ["i-b"], ["i-b"]]
    down = _Down()
    ok = launcher.terminate_and_verify(None, "isl", probe=lambda: live[min(down.calls, 3)], down=down,
                                       sleep_fn=lambda s: None, attempts=2, num_nodes=2)
    assert ok is False and down.calls == 3
    err = capsys.readouterr().err
    assert "UNCONFIRMED node instance(s) i-b" in err


def test_teardown_two_nodes_without_probe_is_not_trusted(capsys):
    down = _Down()
    assert launcher.terminate_and_verify(None, "isl", probe=None, down=down, sleep_fn=lambda s: None,
                                         num_nodes=2) is False
    assert "cannot be cloud-verified" in capsys.readouterr().err
    # single node keeps the legacy "trust down" behaviour
    assert launcher.terminate_and_verify(None, "isl", probe=None, down=_Down(), sleep_fn=lambda s: None) is True


def test_teardown_two_nodes_probe_failure_is_false():
    def boom():
        raise RuntimeError("cloud api down")
    assert launcher.terminate_and_verify(None, "isl", probe=boom, down=_Down(), sleep_fn=lambda s: None,
                                         num_nodes=2) is False


def test_network_tier_best_only_where_the_cloud_honors_it():
    # G1 (2026-10-03): Nebius rejects network_tier=best for anything but H100:8/H200:8
    # ("Catalog does not contain any instances"), so a 2x1xL40S island asks for none.
    assert launcher.multinode_network_tier("nebius", "H100", 8) == "best"
    assert launcher.multinode_network_tier("nebius", "L40S", 1) is None
    assert launcher.multinode_network_tier("nebius", "h100", 1) is None
    assert launcher.multinode_network_tier("aws", "A10G", 1) == "best"
