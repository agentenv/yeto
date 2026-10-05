"""m5 (ruling 2026-10-04 v2): rollout SGLang TP8 engine across two 4-GPU nodes.

Colocated 2x4 island: the trainer (DP8, tp1) and one TP8 sglang engine share all 8
GPUs; the engine spans both nodes (sglang nnodes=2) and needs
--rl-allow-cross-node-engine-tp. CPU checks: placement rules, sglang TP rank map,
launcher argv, learner PlacementRequest, s1run m5 dry-run args and the m5 judge."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.rl.engine import multinode as mn
from yeto.rl.engine.miles_adapter.placement import PlacementRequest

GPU_DIR = Path(__file__).parent / "multinode_gpu"
T24 = mn.Topology(2, 4)


# ---------------------------------------------------------------- rank map
def test_sglang_tp8_rank_map_over_two_4gpu_nodes():
    engine = mn.colocated_engine_slots(8, 8, T24)
    assert engine == [[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]]
    ranks = mn.sglang_tp_rank_map(engine[0])
    assert [(r["tp_rank"], r["node_rank"], r["node"], r["gpu"]) for r in ranks] == [
        (0, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 2), (3, 0, 0, 3),
        (4, 1, 1, 0), (5, 1, 1, 1), (6, 1, 1, 2), (7, 1, 1, 3)]


def test_rank_map_base_gpu_and_single_node_engines():
    t = mn.Topology(1, 8)
    engines = mn.colocated_engine_slots(8, 4, t)
    assert [r["gpu"] for r in mn.sglang_tp_rank_map(engines[1])] == [4, 5, 6, 7]
    assert {r["node_rank"] for r in mn.sglang_tp_rank_map(engines[1])} == {0}


@pytest.mark.parametrize("engine", [
    [(0, 0), (0, 1), (0, 2), (1, 0)],          # uneven per-node runs
    [(0, 0), (0, 2), (1, 0), (1, 2)],          # non-consecutive local GPUs
    [(0, 0), (1, 0), (0, 1), (1, 1)],          # revisits a node
    [],
])
def test_rank_map_refuses_shapes_sglang_cannot_launch(engine):
    with pytest.raises(mn.TopologyError):
        mn.sglang_tp_rank_map(engine)


def test_colocated_engine_slots_must_divide():
    with pytest.raises(mn.TopologyError, match="multiple"):
        mn.colocated_engine_slots(8, 3, T24)


# ---------------------------------------------------------------- placement rules (learner side)
def _colocated(**kw):
    base = dict(trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8, gpus_per_node=4)
    base.update(kw)
    return PlacementRequest("colocated", **base)


def test_colocated_cross_node_engine_needs_the_opt_in():
    with pytest.raises(ValueError, match="--rl-allow-cross-node-engine-tp"):
        _colocated()
    assert _colocated(allow_cross_node_engine_tp=True).trainer_shape() == (2, 4)


def test_colocated_engine_must_take_whole_nodes():
    # 12 GPUs on 3x4 with TP6 engines: engine 0 = n0 x4 + n1 x2 -> a partial node
    with pytest.raises(ValueError, match="whole 4-GPU nodes"):
        _colocated(trainer_gpus=12, rollout_gpus=12, gpus_per_engine=6, allow_cross_node_engine_tp=True)


def test_colocated_in_node_engines_unchanged():
    assert _colocated(gpus_per_engine=4).trainer_shape() == (2, 4)
    assert PlacementRequest("colocated", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8).trainer_shape() == (1, 8)


def test_fixed_partition_3x4_fallback_layout():
    """Fallback (if colocated cross-node weight sync fails): trainer n0 x4, TP8 engine n1+n2."""
    r = PlacementRequest("fixed-partition", trainer_gpus=4, rollout_gpus=8, gpus_per_engine=8,
                         gpus_per_node=4, allow_cross_node_engine_tp=True)
    assert r.trainer_shape() == (1, 4)
    rollout = [r.topology.slot_of(b) for b in r.role_bundles()["rollout"]]
    assert [(x["tp_rank"], x["node"], x["gpu"]) for x in mn.sglang_tp_rank_map(rollout)][::4] == [(0, 1, 0), (4, 2, 0)]
    with pytest.raises(ValueError, match="spans nodes"):
        PlacementRequest("fixed-partition", trainer_gpus=4, rollout_gpus=8, gpus_per_engine=8, gpus_per_node=4)


def test_learner_run_config_keeps_island_gpus_per_node_when_colocated():
    from yeto.rl.engine.miles_adapter.config import placement_request
    from types import SimpleNamespace as NS

    par = NS(actor_num_nodes=2, actor_num_gpus_per_node=4, colocated=True, rollout_num_gpus_per_engine=8,
             island_gpus_per_node=4, tensor_parallel=1, pipeline_parallel=1, context_parallel=1,
             expert_parallel=1, allow_cross_node_tp=False, allow_cross_node_engine_tp=True)
    req = placement_request(NS(parallel=par))
    assert req.kind == "colocated" and req.gpus_per_node == 4 and req.trainer_shape() == (2, 4)
    par.allow_cross_node_engine_tp = False
    with pytest.raises(ValueError, match="spans nodes"):
        placement_request(NS(parallel=par))


# ---------------------------------------------------------------- launcher
def _m5_args(*extra):
    from test_rl_launcher import _args

    return _args(("--gpu", "nebius:2x4xl40s@eu-north1", "--rl-engine", "ports",
                  "--rollout-num-gpus-per-engine", "8", "--tensor-parallel", "1",
                  "--rollout-batch-size", "4", "--n-samples-per-prompt", "8", *extra))


def test_launcher_refuses_colocated_cross_node_engine_without_opt_in():
    from test_rl_launcher import _prepare_rl_args

    with pytest.raises(ValueError, match="spans nodes"):
        _prepare_rl_args(_m5_args())


def test_launcher_colocated_cross_node_engine_task(monkeypatch, capsys):
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _m5_args("--rl-allow-cross-node-engine-tp")
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    assert "spans 2 whole nodes" in capsys.readouterr().out
    spec = parse_gpu_spec(args.gpu)[0]
    assert [(r["tp_rank"], r["node"], r["gpu"]) for r in launcher.rl_colocated_engine_check(args, spec)][3:5] == [(3, 0, 3), (4, 1, 0)]
    task = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert task.num_nodes == 2
    assert "--actor-num-nodes 2 --actor-num-gpus-per-node 4 --rl-island-gpus-per-node 4" in task.run
    assert "--rollout-num-gpus-per-engine 8" in task.run and "--rl-allow-cross-node-engine-tp" in task.run
    assert "--rl-placement" not in task.run  # colocated stays the default argv


# ---------------------------------------------------------------- s1run / s1judge m5
def test_s1run_m5_dry_args():
    out = subprocess.run(["bash", str(GPU_DIR / "s1run.sh"), "m5", "m5-dry", "3600"], capture_output=True,
                         text=True, env={**os.environ, "DRY": "1", "IMAGE": "docker:x"}, check=True).stdout
    assert "nebius:2x4xl40s@eu-north1" in out and "nodes=2" in out
    for flag in ("--rollout-num-gpus-per-engine 8", "--rl-allow-cross-node-engine-tp", "--rl-observe-timeline",
                 "--tensor-parallel 1 --pipeline-parallel 1", "--keep"):
        assert flag in out
    assert "--rl-placement" not in out


def _uuid(n, g):
    return f"GPU-{n:08x}-0000-0000-0000-{g:012x}"


def _fake_m5_run(tmp, *, gen_ref=None, ranks=range(8), allreduce_ok=True, phys=4, use=None, gpu_name="NVIDIA L40S",
                 rank_gpu=None, stray_apps=()):
    cl = "m5-l0-eu-north1"
    p = tmp / "pulled"
    p.mkdir(parents=True)
    (tmp / "cluster.txt").write_text(cl)
    (tmp / "rc.txt").write_text("rc=0")
    (tmp / "args.txt").write_text("--rollout-num-gpus-per-engine 8 --rl-allow-cross-node-engine-tp")
    for n, name in enumerate((cl, cl + "-worker1")):
        (p / f"gpu-{name}.txt").write_text(f"h{n}\n" + "".join(f"{g}, {_uuid(n, g)}, {gpu_name}, 570\n" for g in range(phys)))
        apps = "2026-10-05T10:00:00Z\n" + "".join(f"{_uuid(n, g)}, {200 + g}, sglang::scheduler, 9000 MiB\n" for g in range(4))
        apps += "".join(f"{_uuid(sn, sg)}, 300, {what}, 500 MiB\n" for sn, sg, what in stray_apps if sn == n)
        (p / f"apps-{name}.txt").write_text(apps + "   201 sglang::scheduler_TP1\n")
    if use is not None:
        (tmp / "gpus_per_node.txt").write_text(f"{phys} {use}\n")
    events = []
    for rid in range(2):
        events += [{"event": "rl_publication", "policy_version": rid, "rl/policy_token": f"v{rid}",
                    "sync/publication_members": ["c0"], "sync/publication_payload_hash": f"h{rid}"},
                   {"event": "rl_driver_phase", "phase": "generate", "rollout_id": rid},
                   {"event": "rl_timeline_span", "task": "generate", "start": 10.0 * rid, "end": 10.0 * rid + 4,
                    "rollout_id": rid},
                   {"event": "rl_local_round", "local_round_id": rid, "action_tokens": 4000},
                   {"event": "rl_driver_phase", "phase": "train", "rollout_id": rid}]
    (p / "rl-island-0.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    rows = []
    for r in ranks:
        n, g = (rank_gpu or {}).get(r, divmod(r, 4))
        rows.append({"node": n, "tp_rank": r, "pid": 100 + r, "gpu_uuid": _uuid(n, g), "title": f"sglang::scheduler_TP{r}"})
    (p / "m5ranks.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    nccl = {"ok": allreduce_ok, "world": 8, "nodes": 2, "results": [{"bytes": 1 << 20, "lat_us": 900.0, "busbw_gbps": 2.0}]}
    (p / "m5nccl.json").write_text(json.dumps(nccl))
    gen = {"tp8": {"tokens_per_s": 900.0, "outputs": [[1, 2, 3, 4] * 8] * 4, "nnodes": 2, "tp": 8},
           "ref": {"tokens_per_s": 1500.0, "outputs": gen_ref if gen_ref is not None else [[1, 2, 3, 4] * 8] * 4,
                   "nnodes": 1, "tp": 4}}
    (p / "m5gen.json").write_text(json.dumps(gen))
    return tmp


def _judge(run):
    r = subprocess.run([sys.executable, str(GPU_DIR / "s1judge.py"), str(run), "m5"], capture_output=True, text=True)
    return r.returncode, json.loads((run / "judgment-m5.json").read_text())


def test_judge_m5_pass_and_perf_numbers(tmp_path):
    rc, j = _judge(_fake_m5_run(tmp_path))
    assert rc == 0 and j["verdict"] == "PASS", j
    perf = j["perf"]
    assert perf["rl_generate_tokens_per_s"] == [1000.0, 1000.0]
    assert perf["gen_only_tp8_tokens_per_s"] == 900.0 and perf["allreduce"][0]["lat_us"] == 900.0


def test_judge_m5_fails_on_rank_map_gap(tmp_path):
    rc, j = _judge(_fake_m5_run(tmp_path, ranks=range(7)))
    assert rc == 1 and j["verdict"] == "FAIL" and not j["checks"]["tp_ranks_0_7_match_gpu_uuids"]


def test_judge_m5_generation_tolerance(tmp_path):
    diverged = [[1, 2, 3, 4] * 8, [1, 2, 3, 4] * 8, [1, 2, 3, 4] * 8, [9] * 32]  # 3/4 prompts share the prefix
    rc, j = _judge(_fake_m5_run(tmp_path, gen_ref=diverged))
    assert j["checks"]["gen_matches_single_node_ref_within_tolerance"] is True
    rc, j = _judge(_fake_m5_run(tmp_path / "b", gen_ref=[[9] * 32] * 4))
    assert j["verdict"] == "FAIL" and j["checks"]["gen_matches_single_node_ref_within_tolerance"] is False


def test_judge_m5_nccl_failure(tmp_path):
    rc, j = _judge(_fake_m5_run(tmp_path, allreduce_ok=False))
    assert j["verdict"] == "FAIL" and not j["checks"]["nccl_cross_node_init_ok"]
