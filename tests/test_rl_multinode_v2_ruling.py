"""rl-multinode-island, user ruling 2026-10-04 v2 (MULTINODE-GAP-S8.md §8):
TP-in-node is a default preference with explicit opt-ins (trainer tp*cp / rollout engine TP),
cross-node engines scale as whole replicas, mixed placement never overlaps, GPU uuid rebind
is guarded against duplicate occupation / role conflict / stale re-entry."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.rl.elastic_benchmark import capabilities as caps
from yeto.rl.elastic_benchmark.capabilities import ManifestError
from yeto.rl.engine import multinode as mn
from yeto.rl.engine.miles_adapter.placement import PlacementRequest

T = mn.Topology(2, 4)


def _slots(trainer, rollout, standby=()):
    return {"trainer": [T.slot_of(b) for b in trainer],
            "rollout": [[T.slot_of(b) for b in e] for e in rollout],
            "standby": [T.slot_of(b) for b in standby]}


# ------------------------------------------------------------ 8.1 trainer TP: default vs opt-in
def test_trainer_tp_group_across_nodes_is_refused_by_default_and_allowed_explicitly():
    slots = _slots(range(8), [])  # tp*cp = 8 over 2x4: the TP group spans nodes
    why = mn.node_placement_rejection(slots, node_parallel=8, model_parallel=8)
    assert why and "spans nodes" in why and "--rl-allow-cross-node-tp" in why
    assert mn.node_placement_rejection(slots, node_parallel=8, model_parallel=8, allow_cross_node_tp=True) is None
    # divisibility of the trainer total still holds with the opt-in
    assert "not divisible" in mn.node_placement_rejection(_slots(range(6), []), node_parallel=4,
                                                         allow_cross_node_tp=True)
    # PlacementRequest: tp=8 trainer over 2x4 (rollout none -> use a 4x4 island with rollout on n2/n3)
    with pytest.raises(ValueError, match="spans nodes"):
        PlacementRequest("fixed-partition", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=4,
                         gpus_per_node=4, node_parallel=8, model_parallel=8)
    r = PlacementRequest("fixed-partition", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=4,
                         gpus_per_node=4, node_parallel=8, model_parallel=8, allow_cross_node_tp=True)
    assert r.trainer_shape() == (2, 4)
    # min_nodes: tp 8 on 4-GPU nodes
    with pytest.raises(mn.TopologyError, match="TP stays inside a node by default"):
        mn.min_nodes(trainer_min_gpus=8, rollout_min_gpus=4, standby_gpus=0, gpus_per_node=4, node_parallel=8)
    assert mn.min_nodes(trainer_min_gpus=8, rollout_min_gpus=4, standby_gpus=0, gpus_per_node=4,
                        node_parallel=8, allow_cross_node_tp=True) == 3


# ------------------------------------------------------------ 8.2 rollout engine TP across nodes
def test_rollout_engine_across_nodes_needs_the_opt_in_and_whole_nodes():
    slots = _slots([], [range(8)])  # one TP8 engine over 2x4
    why = mn.node_placement_rejection(slots, gpus_per_engine=8)
    assert why and "rollout engine" in why and "--rl-allow-cross-node-engine-tp" in why
    assert mn.node_placement_rejection(slots, gpus_per_engine=8, allow_cross_node_engine=True,
                                       gpus_per_node=4) is None
    # a cross-node engine taking a PART of a node is never a replica
    part = _slots([0, 1], [[2, 3, 4, 5]], [6, 7])
    why = mn.node_placement_rejection(part, gpus_per_engine=4, allow_cross_node_engine=True, gpus_per_node=4)
    assert why and "whole 4-GPU nodes" in why
    # min_nodes with a cross-node engine: 8-GPU engine on 4-GPU nodes
    with pytest.raises(mn.TopologyError, match="--rl-allow-cross-node-engine-tp"):
        mn.min_nodes(trainer_min_gpus=4, rollout_min_gpus=8, standby_gpus=0, gpus_per_node=4)
    assert mn.min_nodes(trainer_min_gpus=4, rollout_min_gpus=8, standby_gpus=0, gpus_per_node=4,
                        allow_cross_node_engine=True) == 3
    with pytest.raises(mn.TopologyError, match="not a whole number"):
        mn.min_nodes(trainer_min_gpus=4, rollout_min_gpus=6, standby_gpus=0, gpus_per_node=4,
                     allow_cross_node_engine=True)


def test_placement_request_cells_chunk_a_cross_node_engine_as_whole_nodes():
    # 4x4 island: trainer n0+n1 (tp 2), one TP8 engine over n2+n3 declared as cell c0
    with pytest.raises(ValueError, match="spans nodes"):
        PlacementRequest("fixed-partition", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8,
                         gpus_per_node=4, node_parallel=2, model_parallel=2, rollout_cell_names=("c0",))
    r = PlacementRequest("fixed-partition", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8,
                         gpus_per_node=4, node_parallel=2, model_parallel=2, rollout_cell_names=("c0",),
                         allow_cross_node_engine_tp=True)
    cells = r.placement_map_arg["rollout_cells"]
    assert cells == [{"name": "c0", "bundles": list(range(8, 16)), "start": True}]
    # chunk_by_node: partial nodes stay leftover; the default path is unchanged
    assert mn.chunk_by_node([4, 5, 6, 7, 8, 9, 10, 11], mn.Topology(3, 4), 8, allow_cross_node=True) == (
        [[4, 5, 6, 7, 8, 9, 10, 11]], [])
    assert mn.chunk_by_node([5, 6, 7, 8, 9, 10, 11], mn.Topology(3, 4), 8, allow_cross_node=True) == (
        [], [5, 6, 7, 8, 9, 10, 11])
    assert mn.chunk_by_node(list(range(8)), T, 4) == ([[0, 1, 2, 3], [4, 5, 6, 7]], [])


def test_cfg_parallel_switches_lift_the_defaults():
    pool = [{"uuid": f"GPU-n{n}-{g}", "node": n, "index": g} for n in range(4) for g in range(4)]
    cfg = {"nodes": 4, "gpus_per_node": 4, "gpus": pool, "edges": [], "configs": {
        "T8R8S0": {"trainer": 8, "rollout": 8, "standby": 0, "rollout_engine_gpus": 8,
                   "parallel": {"tp": 8, "pp": 1},
                   "placement": {"trainer": [f"n{n}:{g}" for n in (0, 1) for g in range(4)],
                                 "rollout": [[f"n{n}:{g}" for n in (2, 3) for g in range(4)]],
                                 "standby": []}}}}
    with pytest.raises(ManifestError, match="spans nodes"):
        caps.parse_configs(copy.deepcopy(cfg))
    on = copy.deepcopy(cfg)
    on["configs"]["T8R8S0"]["parallel"].update(allow_cross_node_tp=True, allow_cross_node_engine_tp=True)
    c = caps.parse_configs(on)["T8R8S0"]
    assert c.allow_cross_node_tp and c.allow_cross_node_engine_tp and c.dims == {"tp": 8, "pp": 1, "cp": 1, "ep": 1}
    assert caps.placement_rejection(c, caps.pool_gpus(on)) is None
    bad = copy.deepcopy(cfg)
    bad["configs"]["T8R8S0"]["parallel"]["allow_cross_node_tp"] = 1
    with pytest.raises(ManifestError, match="must be true/false"):
        caps.parse_configs(bad)
    # a cross-node engine scales as a whole replica: rollout must be a multiple of the engine
    half = copy.deepcopy(on)
    half["configs"]["T8R8S0"].update(rollout=4, standby=4)
    half["configs"]["T8R8S0"]["placement"] = {"trainer": on["configs"]["T8R8S0"]["placement"]["trainer"],
                                              "rollout": [[f"n2:{g}" for g in range(4)]],
                                              "standby": [f"n3:{g}" for g in range(4)]}
    with pytest.raises(ManifestError, match="whole number of 8-GPU cross-node engine replicas"):
        caps.parse_configs(half)


def test_launcher_switches_merge_cli_and_cfg_and_forward_to_the_learner(monkeypatch, tmp_path):
    from test_rl_launcher_multinode import _elastic_task

    pool = []
    cfg = {"nodes": 4, "gpus_per_node": 4, "gpus": pool, "configs": {
        "T8R8S0": {"trainer": 8, "rollout": 8, "standby": 0, "rollout_engine_gpus": 8,
                   "parallel": {"tp": 1, "pp": 1, "allow_cross_node_engine_tp": True},
                   "placement": {"trainer": [f"n{n}:{g}" for n in (0, 1) for g in range(4)],
                                 "rollout": [[f"n{n}:{g}" for n in (2, 3) for g in range(4)]],
                                 "standby": []}}},
           "edges": []}
    res = tmp_path / "res.json"
    res.write_text(json.dumps(cfg))
    args, spec, task = _elastic_task(monkeypatch, tmp_path, "nebius:4x4xl40s", res, "T8R8S0", rollout=8,
                                     extra=["--rollout-num-gpus-per-engine", "8"])
    assert launcher.rl_cross_node_switches(args) == (False, True)  # cfg spelling
    assert " --rl-allow-cross-node-engine-tp" in task.run and "--rl-allow-cross-node-tp" not in task.run
    assert launcher.rl_island_layout(args, spec)[:2] == (2, 4)
    # CLI spelling for the trainer TP
    a = SimpleNamespace(rl_allow_cross_node_tp=True)
    assert launcher.rl_cross_node_switches(a) == (True, False)
    assert launcher.rl_cross_node_flags(a) == " --rl-allow-cross-node-tp"
    base = dict(rl_placement="fixed-partition", rollout_num_gpus=8, rl_standby_gpus=0, tensor_parallel=16,
                pipeline_parallel=1, rollout_num_gpus_per_engine=8, rl_min_nodes_per_learner=0)
    (four,) = parse_gpu_spec("nebius:4x8xh100")
    with pytest.raises(ValueError, match="TP stays inside a node"):
        launcher.rl_min_nodes(SimpleNamespace(**base), four)
    assert launcher.rl_min_nodes(SimpleNamespace(**base, rl_allow_cross_node_tp=True), four) == 3


def test_learner_switches_reach_the_placement_request_and_the_layout():
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.config import placement_request

    parallel = SimpleNamespace(colocated=False, actor_num_nodes=2, actor_num_gpus_per_node=4,
                               dedicated_rollout_gpus=8, rollout_num_gpus_per_engine=8, standby_gpus=0,
                               rollout_cell_names=(), island_gpus_per_node=4, tensor_parallel=8,
                               pipeline_parallel=1, context_parallel=1, expert_parallel=1, bundle_map=None,
                               allow_cross_node_tp=True, allow_cross_node_engine_tp=True)
    req = placement_request(SimpleNamespace(parallel=parallel))
    assert req.allow_cross_node_tp and req.allow_cross_node_engine_tp and req.trainer_shape() == (2, 4)
    layout = entry.island_layout_of(SimpleNamespace(tensor_model_parallel_size=8, actor_num_nodes=2,
                                                    actor_num_gpus_per_node=4),
                                    SimpleNamespace(nodes=4, gpus_per_node=4), req)
    assert layout["cross_node_tp"] == 1 and layout["cross_node_engine_tp"] == 1
    from yeto.rl.engine import controller as ctl

    assert "cross_node_tp" in ctl.LAYOUT_KEYS
    base = ctl.island_layout(SimpleNamespace(dims={"tp": 8}, trainer=8), (4, 4))
    assert base["cross_node_tp"] == 0 and ctl.layout_diff(base, layout)["cross_node_tp"] == (0, 1)


# ------------------------------------------------------------ 8.3 whole-replica scaling
def test_rollout_edge_scales_by_whole_engine_replicas():
    assert mn.engine_replica_delta_rejection(8, 16, 8) is None
    why = mn.engine_replica_delta_rejection(8, 4, 8)
    assert why and "single node of it cannot be removed" in why
    import inspect

    from yeto.rl.engine import controller

    src = inspect.getsource(controller.IslandController.plan)
    assert "engine_replica_delta_rejection(src.rollout, dst.rollout, src.rollout_engine_gpus)" in src


def test_bind_members_refuses_a_partial_node_cross_node_cell():
    from yeto.rl.engine.miles_adapter.rollout import MembershipPlanError
    import yeto.rl.engine.miles_adapter.rollout as ro

    class _B:
        gpus_per_node = 4

        def same_node(self, gpus):
            return len({self.node_of(g) for g in gpus}) <= 1

        def node_of(self, g):
            return int(g[1])  # "g<node><local>"

        def view_for(self, gpus):
            return {"gpus": list(gpus)}

    class _Ctl:
        def get_cell_statuses(self):
            return {}

    class _Ops:
        def __init__(self, cross):
            self._args = SimpleNamespace(rollout_num_gpus_per_engine=8, yeto_rl_allow_cross_node_engine_tp=cross)
            self._gpus_per_engine = 8
            self._bundles = _B()
            self._controller = _Ctl()
            self._require_declared = lambda: ["c0"]
            self._bind_seq = 0
            self._run = lambda coro: coro.close() if hasattr(coro, "close") else coro
            self._manager = lambda: SimpleNamespace(set_pg_view=SimpleNamespace(remote=lambda *a: None),
                                                     rebind_cell=SimpleNamespace(remote=lambda *a, **k: None))

    gpus = [f"g2{g}" for g in range(4)] + [f"g3{g}" for g in range(4)]
    with pytest.raises(MembershipPlanError, match="span nodes"):
        ro.MilesRolloutPool.bind_members(_Ops(False), frozenset({"engine:c0"}), tuple(gpus))
    ro.MilesRolloutPool.bind_members(_Ops(True), frozenset({"engine:c0"}), tuple(gpus))  # whole nodes: accepted
    with pytest.raises(MembershipPlanError, match="whole 4-GPU nodes"):
        ro.MilesRolloutPool.bind_members(_Ops(True), frozenset({"engine:c0"}),
                                         tuple([f"g2{g}" for g in range(4)] + [f"g3{g}" for g in range(2)] + ["g10", "g11"]))


# ------------------------------------------------------------ 8.4 mixed placement never overlaps
def test_mixed_placement_slots_are_exclusive():
    with pytest.raises(mn.TopologyError, match="n0:1 more than once"):
        mn.normalize_placement({"trainer": ["n0:0", "n1:0"], "rollout": [["n0:1"]], "standby": ["n0:1"]}, mn.Topology(2, 2))
    with pytest.raises(mn.TopologyError, match="more than once"):
        mn.normalize_placement({"trainer": [0, 2], "rollout": [[2]], "standby": [3]}, mn.Topology(2, 2))
    cfg = json.loads((Path(__file__).parent / "multinode_gpu" / "resources-2x2.json").read_text())
    cfg["configs"]["T2R1S1"]["placement"]["standby"] = ["n0:1"]
    with pytest.raises(ManifestError, match="more than once"):
        caps.parse_configs(cfg)
    assert mn.placement_overlap(_slots([0, 4], [[1]], [5])) is None


# ------------------------------------------------------------ 8.5 uuid rebind safety
A, B, C, D = "GPU-a", "GPU-b", "GPU-c", "GPU-d"


def test_role_map_and_occupation_and_stale_incarnation_rules():
    roles = mn.role_uuid_map({"trainer": (0, 2), "rollout": (1,), "standby": (3,)}, [A, B, C, D])
    assert roles == {A: "trainer", C: "trainer", B: "rollout", D: "standby"}
    assert mn.role_uuid_map(None, [A, B, C, D], counts=(2, 1, 1)) == {A: "trainer", B: "trainer", C: "rollout", D: "standby"}
    with pytest.raises(mn.TopologyError, match="both trainer and rollout"):
        mn.role_uuid_map({"trainer": (0, 1), "rollout": (1,), "standby": ()}, [A, B])
    assert mn.occupation_rejection([A, B, C, D], roles) is None
    assert "more than once" in mn.occupation_rejection([A, A, C, D], roles)
    assert "another active island 'isl-2'" in mn.occupation_rejection([A, B, C, D], roles, {"isl-2": [C]})
    assert "no role" in mn.occupation_rejection([A, B, C, D], {A: "trainer"})
    assert mn.stale_incarnation_rejection([A, B], [A, B], {}, "inc-new") is None
    assert mn.stale_incarnation_rejection([A, B], [A, B], {A: "inc-new"}, "inc-new") is None
    why = mn.stale_incarnation_rejection([A, B], [A, C], {B: "inc-old"}, "inc-new")
    assert why and "GPU-b@inc-old" in why and "yeto down" in why


def test_entry_preflight_journals_roles_and_refuses_stale_incarnation(tmp_path, monkeypatch):
    from test_rl_multinode_recovery import _journal
    from test_rl_reconfig_recovery import _ctl
    from yeto.rl.engine.miles_adapter import entry

    topology = SimpleNamespace(nodes=2, gpus_per_node=2)
    miles_args = SimpleNamespace(yeto_rl_event_tape=str(tmp_path / "events.jsonl"), yeto_rl_learner_id=0,
                                 yeto_rl_elastic={"resources": {"nodes": 2, "gpus_per_node": 2, "gpus": []}})
    placement = SimpleNamespace(trainer_gpus=2, rollout_gpus=1, standby_gpus=1,
                                bundle_map={"trainer": (0, 2), "rollout": (1,), "standby": (3,)})
    live = [[(0, A), (1, B)], [(0, C), (1, D)]]
    monkeypatch.setattr(entry, "_ray_gpu_uuids", lambda _t: live)
    written = []
    ctl = _ctl(tmp_path / "state", {"t": 1000.0})
    res = entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args,
                                             placement=placement, incarnation_probe=lambda t, f: {},
                                             marker_writer=lambda t, obs, inc: written.append((obs, inc)))
    assert res.ok
    (rec,) = _journal(tmp_path, "gpu_pool")
    assert rec["roles"] == {A: "trainer", C: "trainer", B: "rollout", D: "standby"}
    assert written == [(((A, B), (C, D)), ctl.incarnation["id"])]
    old = ctl.incarnation["id"]
    ctl.close()
    # (c) the old incarnation still holds n1's GPUs: refused, operator must `yeto down`
    ctl = _ctl(tmp_path / "state", {"t": 1001.0})
    with pytest.raises(RuntimeError, match="old incarnation.*GPU-c@" + old + ".*yeto down"):
        entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args,
                                           placement=placement, incarnation_probe=lambda t, f: {C: old},
                                           marker_writer=lambda *a: None)
    recs = _journal(tmp_path, "gpu_pool")
    assert not recs[-1]["accepted"] and "yeto down" in recs[-1]["error"]
    assert ctl.recovery_required and not [r for r in _journal(tmp_path, "phase") if r.get("phase") == "RECOVERY_REQUIRED"]
    ctl.close()
    # (a) duplicate occupation by another active island
    ctl = _ctl(tmp_path / "state", {"t": 1002.0})
    with pytest.raises(RuntimeError, match="another active island"):
        entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args,
                                           placement=placement, incarnation_probe=lambda t, f: {},
                                           other_islands={"isl-2": [D]}, marker_writer=lambda *a: None)
    ctl.close()
    # (b) role conflict: a bundle map giving one GPU two roles
    ctl = _ctl(tmp_path / "state", {"t": 1003.0})
    clash = SimpleNamespace(trainer_gpus=2, rollout_gpus=1, standby_gpus=1,
                            bundle_map={"trainer": (0, 2), "rollout": (2,), "standby": (3,)})
    with pytest.raises(RuntimeError, match="both trainer and rollout"):
        entry.reconcile_gpu_pool_preflight(SimpleNamespace(controller=ctl), topology, miles_args,
                                           placement=clash, incarnation_probe=lambda t, f: {},
                                           marker_writer=lambda *a: None)
    ctl.close()


def test_incarnation_marker_files_track_live_pids(tmp_path):
    from yeto.rl.engine.miles_adapter import entry

    d = str(tmp_path / "markers")
    entry._write_markers([A, B], "inc-1", os.getpid(), marker_dir=d)
    entry._write_markers([C], "inc-0", 2 ** 22 + 12345, marker_dir=d)  # almost surely dead
    rows = entry._marker_rows([A, B, C, D], marker_dir=d)
    assert rows == {A: "inc-1", B: "inc-1"}
