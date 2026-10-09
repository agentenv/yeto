"""rl-multinode-island task 1.6: launcher shape derivation, min nodes, island flags."""

from __future__ import annotations

import json
from pathlib import Path
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
    # Q1/Q3 ruling 2026-10-04: smallest replica tp*cp*ep*pp may span nodes
    (four,) = parse_gpu_spec("nebius:4x8xh100")
    # EP shares ranks with TP x DP (etp = 1): tp2 pp2 ep4 -> 2*2*max(1, 4/2) = 8 trainer GPUs (+ engine 8)
    assert launcher.rl_min_nodes(_args(pipeline_parallel=2, expert_parallel=4), four) == 2
    assert launcher.rl_min_nodes(_args(pipeline_parallel=2, expert_parallel=8), four) == 3  # 16 + 8
    assert launcher.rl_min_nodes(_args(tensor_parallel=1, expert_parallel=2, rollout_num_gpus_per_engine=1),
                                 parse_gpu_spec("nebius:2x2xl40s")[0]) == 2  # M2: EP2 = 2 DP ranks (2 + engine 1 on 2-GPU nodes)
    assert launcher.rl_min_nodes(_args(tensor_parallel=2, expert_parallel=2), spec) == 2  # ep <= tp: no extra GPU
    assert launcher.rl_min_nodes(_args(tensor_parallel=8, pipeline_parallel=2), four) == 3
    with pytest.raises(ValueError, match="TP stays inside a node"):
        launcher.rl_min_nodes(_args(tensor_parallel=16), four)
    with pytest.raises(ValueError, match="not a whole number"):
        launcher.rl_min_nodes(_args(tensor_parallel=2, pipeline_parallel=3, expert_parallel=4), four)  # replica 12


# ---- 1.6 task level (D1/D6) and 1.8 teardown (D10)

def test_multinode_env_prelude():
    assert launcher.multinode_env_prelude("nebius", 1) == ""
    nebius = launcher.multinode_env_prelude("nebius", 2)
    assert "/sys/class/net" in nebius and "GLOO_SOCKET_IFNAME=\"$YETO_IFACE\"" in nebius  # D6: detected on the node (Nebius NIC is network-interface-0)
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
    assert "/sys/class/net" in task.run and 'GLOO_SOCKET_IFNAME="$YETO_IFACE"' in task.run  # D6: detected on the node
    head, _, worker = task.run.partition("\nelse\n")
    assert "ray start --head" in head
    # G1 (2026-10-03): the dashboard/state API must be reachable from worker nodes
    assert "--include-dashboard=true" in head and "--dashboard-host=0.0.0.0" in head
    # G1 (2026-10-03): worker nodes fetch the pinned model snapshot before joining the Ray
    assert "snapshot_download" in worker and "a" * 40 in worker and "exit 1" in worker
    assert worker.index("snapshot_download") < worker.index("ray start --address")
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


def test_failure_path_teardown_verifies_every_node(monkeypatch):
    """rl-multinode-island D10 (s1-mn-20261004d G4): FleetController._down -> SkySDKOps.down
    must take the island's node count, not the single-node default."""
    seen = {}

    def fake_tv(sky, cluster, **kw):
        seen[cluster] = kw.get("num_nodes")
        return True

    monkeypatch.setattr(launcher, "terminate_and_verify", fake_tv)
    monkeypatch.setitem(__import__("sys").modules, "sky", object())
    launcher.SkySDKOps(nodes_by_cluster={"isl-l0": 2}).down("isl-l0")
    launcher.SkySDKOps().down("other")
    assert seen == {"isl-l0": 2, "other": 1}


# ---- Q2 (2026-10-04 ruling): rollout/trainer mixed on one node, trainer shape from the cfg placement

_GPU_DIR = Path(__file__).resolve().parent / "multinode_gpu"


def _elastic_task(monkeypatch, tmp_path, gpu, resources, config, *, rollout, standby=0, extra=()):
    import sys
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    flags = ["--gpu", gpu, "--rl-engine", "ports", "--rl-placement", "fixed-partition",
             "--rl-rollout-gpus", str(rollout), "--rollout-num-gpus-per-engine", "1",
             "--tensor-parallel", "1", "--rl-elastic", "--rl-elastic-resources", str(resources),
             "--rl-elastic-state-dir", str(tmp_path / "state"), "--rl-elastic-initial-config", config,
             *extra]
    if standby:
        flags += ["--rl-standby-gpus", str(standby)]
    args = _args(tuple(flags))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    spec = parse_gpu_spec(args.gpu)[0]
    return args, spec, launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")


def test_mixed_2x2_cfg_placement_drives_actor_shape_and_bundle_map(monkeypatch, tmp_path):
    """T2(n0:0,n1:0) R1(n0:1) S1(n1:1): trainer 2 nodes x 1 GPU, rollout shares n0 with the trainer."""
    args, spec, task = _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s",
                                     _GPU_DIR / "resources-2x2.json", "T2R1S1", rollout=1, standby=1)
    assert launcher.rl_trainer_shape(args, spec) == (2, 1)
    assert launcher.rl_actor_gpus_per_node(args, spec) == 1
    assert launcher.rl_island_layout(args, spec) == (2, 1, {"trainer": (0, 2), "rollout": (1,), "standby": (3,)})
    assert task.num_nodes == 2
    assert ("--actor-num-nodes 2 --actor-num-gpus-per-node 1 --rl-island-gpus-per-node 2"
            " --rl-island-bundle-map '{\"rollout\":[1],\"standby\":[3],\"trainer\":[0,2]}'") in task.run
    assert " --rollout-num-gpus 1" in task.run and " --rl-standby-gpus 1" in task.run


def test_mixed_2x2_pp2_trainer_spans_nodes(monkeypatch, tmp_path):
    """M1: --pipeline-parallel 2 on T2(n0:0,n1:0): the PP group spans the nodes (Q3), only
    tp*cp = 1 must stay in-node; the cfg layout is accepted and the shape is 2 x 1."""
    args, spec, task = _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s",
                                     _GPU_DIR / "resources-2x2.json", "T2R1S1", rollout=1, standby=1,
                                     extra=("--pipeline-parallel", "2"))
    assert launcher.rl_island_layout(args, spec)[:2] == (2, 1)
    assert "--actor-num-nodes 2 --actor-num-gpus-per-node 1" in task.run and "--pipeline-parallel 2" in task.run
    # M2: EP2 with tp1 pp1 -> the EP group = the two DP ranks on n0:0 / n1:0 (Q1)
    args, spec, task = _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s",
                                     _GPU_DIR / "resources-2x2.json", "T2R1S1", rollout=1, standby=1,
                                     extra=("--expert-parallel", "2"))
    assert launcher.rl_island_layout(args, spec)[:2] == (2, 1) and "--expert-parallel 2" in task.run
    # TP2 across n0:0 / n1:0 stays refused (TP inside a node)
    with pytest.raises(ValueError, match=r"in-node \(tp\*cp\) group .* spans nodes"):
        _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s", _GPU_DIR / "resources-2x2.json", "T2R1S1",
                      rollout=1, standby=1, extra=("--tensor-parallel", "2"))


def test_mixed_2x2_bundle_map_round_trips_into_the_learner_placement(monkeypatch, tmp_path):
    from yeto.rl.adapters.miles.placement import PlacementRequest

    args, spec, _ = _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s",
                                  _GPU_DIR / "resources-2x2.json", "T2R1S1", rollout=1, standby=1)
    bundle_map = launcher.rl_island_layout(args, spec)[2]
    request = PlacementRequest("fixed-partition", trainer_gpus=2, rollout_gpus=1, gpus_per_engine=1,
                               standby_gpus=1, gpus_per_node=2, model_parallel=1, bundle_map=bundle_map)
    assert request.trainer_shape() == (2, 1)
    assert request.placement_map_arg == {"trainer": [0, 2], "rollout": [1], "standby": [3]}


def _write_cfg(tmp_path, name, *, nodes, gpus_per_node, trainer, rollout, standby, placement):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "nodes": nodes, "gpus_per_node": gpus_per_node,
        "configs": {name: {"trainer": trainer, "rollout": rollout, "standby": standby,
                           "rollout_engine_gpus": 1, "placement": placement}},
        "edges": []}), encoding="utf-8")
    return path


def test_non_rectangular_cfg_placement_rejected_before_any_cloud_work(monkeypatch, tmp_path):
    # n0 holds two trainer GPUs, n1 one: Miles cannot express it as nodes x gpus_per_node
    path = _write_cfg(tmp_path, "T3R1S0", nodes=2, gpus_per_node=2, trainer=3, rollout=1, standby=0,
                      placement={"trainer": ["n0:0", "n0:1", "n1:0"], "rollout": [["n1:1"]], "standby": []})
    with pytest.raises(ValueError, match="GPUs per node must be equal"):
        _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s", path, "T3R1S0", rollout=1)


def test_cfg_placement_must_match_launch_totals(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="placement is T2 R1 S1 but the launch asks for"):
        _elastic_task(monkeypatch, tmp_path, "nebius:2x2xl40s", _GPU_DIR / "resources-2x2.json",
                      "T2R1S1", rollout=2)  # --rl-rollout-gpus 2 disagrees with the cfg's R1


def test_trainer_layout_rules():
    from yeto.rl.engine import multinode as mn

    topo = mn.Topology(2, 4)
    slots = {"trainer": [(0, 1), (0, 2), (1, 1), (1, 2)], "rollout": [[(0, 0)], [(1, 0)]],
             "standby": [(0, 3), (1, 3)]}
    assert mn.trainer_layout(slots, topo) == (2, 2, {"trainer": (1, 2, 5, 6), "rollout": (0, 4),
                                                   "standby": (3, 7)})
    assert mn.leading_bundle_map(2, 1, 1) == {"trainer": (0, 1), "rollout": (2,), "standby": (3,)}
    # a node's run must be contiguous local GPUs (local_rank = rank % per_node)
    with pytest.raises(mn.TopologyError, match="contiguous ascending run"):
        mn.trainer_layout({"trainer": [(0, 0), (0, 2), (1, 0), (1, 1)]}, topo)
    # ranks of one node must be listed together
    with pytest.raises(mn.TopologyError, match="must sit on one node"):
        mn.trainer_layout({"trainer": [(0, 0), (1, 0), (0, 1), (1, 1)]}, topo)
    with pytest.raises(mn.TopologyError, match="GPUs per node must be equal"):
        mn.trainer_layout({"trainer": [(0, 0), (0, 1), (1, 0)]}, topo)


def test_old_2x1_cfg_and_single_node_outputs_unchanged(monkeypatch, tmp_path):
    """Snapshot: a whole-node trainer cfg (tests/multinode_gpu/resources-2x1.json) and a
    single-node launch emit exactly what the pre-Q2 launcher did (no bundle map)."""
    args, spec, task = _elastic_task(monkeypatch, tmp_path, "nebius:2x1xl40s",
                                     _GPU_DIR / "resources-2x1.json", "T1R1S0", rollout=1)
    assert args.rl_elastic_initial_placement_slots == {"trainer": [(0, 0)], "rollout": [[(1, 0)]], "standby": []}
    assert launcher.rl_island_layout(args, spec) == (1, 1, {"trainer": (0,), "rollout": (1,), "standby": ()})
    assert launcher.rl_island_bundle_map_flag(args, spec) == ""
    assert "--actor-num-nodes 1 --actor-num-gpus-per-node 1 --rl-island-gpus-per-node 1 --tensor-parallel" in task.run
    assert "--rl-island-bundle-map" not in task.run
    # the same launch with the cfg placement ignored (= the pre-Q2 derivation) is byte-identical
    args.rl_elastic_initial_placement_slots = None
    legacy = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert legacy.run == task.run

    single = tmp_path / "single.json"
    single.write_text(json.dumps({"configs": {"T2R2S0": {"trainer": 2, "rollout": 2}}, "edges": []}))
    args, spec, task = _elastic_task(monkeypatch, tmp_path, "nebius:4xl40s", single, "T2R2S0", rollout=2)
    assert args.rl_elastic_initial_placement_slots is None
    assert launcher.rl_island_layout(args, spec) is None
    assert "--actor-num-nodes 1 --actor-num-gpus-per-node 2 --tensor-parallel" in task.run
    assert "--rl-island-bundle-map" not in task.run and "--rl-island-gpus-per-node" not in task.run


def test_learner_parses_bundle_map_into_the_run_config_layout():
    from yeto.rl import learner
    from yeto.rl.engine.run_config import _island_bundle_map

    payload = '{"rollout":[1],"standby":[3],"trainer":[0,2]}'
    from test_rl_engine_selection import _learner_argv

    parsed = learner.parse_args(_learner_argv(("--rl-island-gpus-per-node", "2",
                                               "--rl-island-bundle-map", payload)))
    assert parsed.rl_island_bundle_map == payload
    assert _island_bundle_map(parsed, 1) == {"trainer": (0, 2), "rollout": (1,), "standby": (3,)}
    assert _island_bundle_map(SimpleNamespace(rl_island_bundle_map=None), 1) is None
    with pytest.raises(ValueError, match="needs a multi-node fixed partition"):
        _island_bundle_map(SimpleNamespace(rl_island_bundle_map=payload, rl_island_gpus_per_node=None), 1)
    with pytest.raises(ValueError, match="needs a multi-node fixed partition"):
        _island_bundle_map(SimpleNamespace(rl_island_bundle_map=payload, rl_island_gpus_per_node=2), None)
    with pytest.raises(ValueError, match="list of ints"):
        _island_bundle_map(SimpleNamespace(rl_island_bundle_map='{"trainer":["a"]}', rl_island_gpus_per_node=2), 1)
