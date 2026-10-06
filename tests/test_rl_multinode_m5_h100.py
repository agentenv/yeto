"""m5 on 2x8xH100 machines with the island limited to GPUs 0..3 per node
(--rl-island-use-gpus-per-node 4): launcher island view, ray start / CUDA_VISIBLE_DEVICES
script text, sky resources stay physical, runtime gpu_pool filter, s1run m5 dry args,
s1m5post prelude and the m5 judge (allocated-card rank map + unallocated-card check)."""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.rl.engine import multinode as mn

from test_rl_multinode_m5 import GPU_DIR, _fake_m5_run, _judge, _uuid

H100 = "nebius:2x8xh100@eu-north1"


def _args(*extra, gpu=H100):
    from test_rl_launcher import _args as base

    return base(("--gpu", gpu, "--rl-engine", "ports", "--rollout-num-gpus-per-engine", "8",
                 "--tensor-parallel", "1", "--rollout-batch-size", "4", "--n-samples-per-prompt", "8",
                 "--rl-allow-cross-node-engine-tp", *extra))


def _task(monkeypatch, *extra, gpu=H100):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _args(*extra, gpu=gpu)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    spec = parse_gpu_spec(args.gpu)[0]
    return args, spec, launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")


def test_island_spec_view():
    spec = parse_gpu_spec(H100)[0]
    ns = types.SimpleNamespace
    assert launcher.rl_island_spec(ns(), spec) is spec
    assert launcher.rl_island_spec(ns(rl_island_use_gpus_per_node=8), spec) is spec
    island = launcher.rl_island_spec(ns(rl_island_use_gpus_per_node=4), spec)
    assert (island.num_nodes, island.gpus_per_node, island.total_gpus, island.gpu) == (2, 4, 8, "H100")
    for bad in (0, 9):
        with pytest.raises(ValueError, match="1..8"):
            launcher.rl_island_spec(ns(rl_island_use_gpus_per_node=bad), spec)
    with pytest.raises(ValueError, match="Modal"):
        launcher.rl_island_spec(ns(rl_island_use_gpus_per_node=4), parse_gpu_spec("modal:2x8xh100")[0])


def test_without_use_gpus_tp8_does_not_span_nodes_on_8gpu_machines(monkeypatch):
    """the bug this flag fixes: 8 GPUs/node -> TP8 engine stays on one node, the trainer DP16."""
    _args_, spec, task = _task(monkeypatch)
    assert "--actor-num-nodes 2 --actor-num-gpus-per-node 8 --rl-island-gpus-per-node 8" in task.run
    assert "--num-gpus=" not in task.run and "CUDA_VISIBLE_DEVICES" not in task.run
    assert [r["node"] for r in launcher.rl_colocated_engine_check(_args_, spec)] == [0] * 8
    assert vars(task.resources).get("network_tier") == "best"  # auto: unchanged default


def test_use_gpus_4_island_task(monkeypatch, capsys):
    args, spec, task = _task(monkeypatch, "--rl-island-use-gpus-per-node", "4", "--rl-island-network-tier", "none")
    out = capsys.readouterr().out
    assert "spans 2 whole nodes" in out
    run = task.run
    # learner argv: trainer 2x4, island 4 GPUs/node -> Miles --num-gpus-per-node 4 -> sglang nnodes = 8 // 4 = 2
    assert "--actor-num-nodes 2 --actor-num-gpus-per-node 4 --rl-island-gpus-per-node 4" in run
    assert "--rollout-num-gpus-per-engine 8" in run and "--rl-allow-cross-node-engine-tp" in run
    # CUDA_VISIBLE_DEVICES + marker exported before any ray start; both ray starts carry --num-gpus=4
    env_at = run.index("export CUDA_VISIBLE_DEVICES=0,1,2,3 YETO_ISLAND_USE_GPUS_PER_NODE=4")
    head_at = run.index("ray start --head")
    worker_at = run.index('ray start --address="$MASTER_ADDR:6379"')
    assert env_at < head_at < worker_at
    head_line = run[head_at:run.index("\n", head_at)]
    worker_line = run[worker_at:run.index("\n", worker_at)]
    assert "--num-gpus=4" in head_line and "--num-gpus=4 && break" in worker_line
    assert "physical 8/node, allocated 4/node" in run
    # sky resources stay physical: H100:8 machines (gpu-h100-sxm_8gpu-128vcpu-1600gb), no IB GPU cluster
    kw = vars(task.resources)
    assert kw["accelerators"] == "H100:8" and kw["infra"] == "nebius/eu-north1" and "network_tier" not in kw
    assert task.num_nodes == 2
    # rank map reference: TP0-3 on n0 GPUs 0-3, TP4-7 on n1 GPUs 0-3
    island = launcher.rl_island_spec(args, spec)
    ranks = launcher.rl_colocated_engine_check(args, island)
    assert [(r["tp_rank"], r["node"], r["gpu"]) for r in ranks] == [
        (t, t // 4, t % 4) for t in range(8)]


def test_use_gpus_colocated_placement_packs_4_plus_4():
    from yeto.rl.engine.miles_adapter.placement import PlacementRequest

    req = PlacementRequest("colocated", trainer_gpus=8, rollout_gpus=8, gpus_per_engine=8, gpus_per_node=4,
                           allow_cross_node_engine_tp=True)
    assert req.trainer_shape() == (2, 4)
    eng = mn.colocated_engine_slots(8, 8, mn.Topology(2, 4))[0]
    assert sorted({n for n, _g in eng}) == [0, 1] and {g for _n, g in eng} == {0, 1, 2, 3}


def test_dry_run_plan_reports_physical_and_allocated(monkeypatch):
    args, spec, _task_ = _task(monkeypatch, "--rl-island-use-gpus-per-node", "4")
    plan = launcher.dry_run_plan(args)
    isl = plan["island_requests"][0]
    assert isl["gpus_per_node"] == 8 and isl["allocated_gpus_per_node"] == 4


def test_use_gpus_refused_above_physical():
    from test_rl_launcher import _prepare_rl_args

    with pytest.raises(ValueError, match="must be in 1..8"):
        _prepare_rl_args(_args("--rl-island-use-gpus-per-node", "16"))


def test_use_gpus_3_breaks_tp8_whole_node_rule():
    from test_rl_launcher import _prepare_rl_args

    with pytest.raises(ValueError):
        _prepare_rl_args(_args("--rl-island-use-gpus-per-node", "3"))


def test_runtime_gpu_pool_rows_only_allocated():
    from yeto.rl.engine.miles_adapter.entry import island_smi_rows

    out = "".join(f"{g}, GPU-{g:08x}-0000-0000-0000-000000000000\n" for g in range(8))
    assert [i for i, _u in island_smi_rows(out, "4")] == [0, 1, 2, 3]
    assert len(island_smi_rows(out, "")) == 8
    rows = island_smi_rows(out, "4")
    other = [(i, u.replace("GPU-", "GPU-f")) for i, u in rows]
    pool = mn.observed_gpu_pool([rows, other], mn.Topology(2, 4))
    assert [len(n) for n in pool] == [4, 4] and pool[0][3].startswith("GPU-00000003")
    with pytest.raises(mn.TopologyError):  # unfiltered 8 rows vs a 4-GPU island fail closed
        mn.observed_gpu_pool([island_smi_rows(out, ""), other], mn.Topology(2, 4))


# ---------------------------------------------------------------- s1run / s1m5post
def _dry(**env):
    return subprocess.run(["bash", str(GPU_DIR / "s1run.sh"), "m5", "m5h-dry", "3300"], capture_output=True, text=True,
                          env={**os.environ, "DRY": "1", "IMAGE": "docker:x", **env})


def test_s1run_m5_h100_dry_args():
    r = _dry(M5_GPU=H100, M5_USE_GPUS="4", M5_POST_HARD="1200")
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "--gpu nebius:2x8xh100@eu-north1" in out and "nodes=2" in out
    assert "physical=8 use=4" in out and "hard=3300 wd=4800" in out
    for flag in ("--rollout-num-gpus-per-engine 8", "--rl-allow-cross-node-engine-tp", "--rl-island-use-gpus-per-node 4",
                 "--rl-island-network-tier none", "--tensor-parallel 1 --pipeline-parallel 1", "--keep",
                 "--model Qwen/Qwen3-0.6B"):
        assert flag in out, flag


def test_s1run_m5_refuses_tp8_mismatch():
    assert _dry(M5_GPU=H100).returncode == 68             # 8 per node without M5_USE_GPUS: TP8 would not span
    assert _dry(M5_GPU=H100, M5_USE_GPUS="2").returncode == 68
    assert _dry(M5_GPU="nebius:1x8xh100@eu-north1", M5_USE_GPUS="4").returncode == 68
    r = _dry()  # L40S default unchanged (plus the no-op network tier none)
    assert r.returncode == 0 and "nebius:2x4xl40s@eu-north1" in r.stdout and "use-gpus" not in r.stdout


def test_s1m5post_prelude_limits_visible_gpus(tmp_path):
    """the PRE prelude the post step sends to each node: CUDA_VISIBLE_DEVICES=0,1,2,3 appended (no ';;')."""
    (tmp_path / "gpus_per_node.txt").write_text("8 4\n")
    script = (GPU_DIR / "s1m5post.sh").read_text()
    head = script[:script.index("mkdir -p $R/pulled")]
    probe = head.replace("set -u", "set -u; R=$1") + '\necho "PRE<<$PRE>>"\n'
    out = subprocess.run(["bash", "-c", probe, "x", str(tmp_path)], capture_output=True, text=True).stdout
    pre = out.split("PRE<<", 1)[1].split(">>", 1)[0]
    assert pre.rstrip().endswith("cd ~/yeto-rl; export CUDA_VISIBLE_DEVICES=0,1,2,3")
    # composed remote command is valid bash
    assert subprocess.run(["bash", "-n", "-c", pre + "; (true)"]).returncode == 0
    (tmp_path / "gpus_per_node.txt").unlink()
    out = subprocess.run(["bash", "-c", probe, "x", str(tmp_path)], capture_output=True, text=True).stdout
    pre = out.split("PRE<<", 1)[1].split(">>", 1)[0]
    assert pre.rstrip().endswith("cd ~/yeto-rl") and subprocess.run(["bash", "-n", "-c", pre + "; (true)"]).returncode == 0


# ---------------------------------------------------------------- judge (2x8 physical, 4 allocated)
def _h100_run(tmp, **kw):
    return _fake_m5_run(tmp, phys=8, use=4, gpu_name="NVIDIA H100 80GB HBM3", **kw)


def test_judge_m5_h100_pass(tmp_path):
    rc, j = _judge(_h100_run(tmp_path))
    assert rc == 0 and j["verdict"] == "PASS", j
    assert j["checks"]["no_process_on_unallocated_gpus"] is True
    assert any("physical=8 use=4 unallocated=8" in n for n in j["notes"])


def test_judge_m5_h100_fails_on_rank_on_unallocated_card(tmp_path):
    rc, j = _judge(_h100_run(tmp_path, rank_gpu={7: (1, 4)}))  # TP7 on n1 GPU 4 (not allocated)
    assert rc == 1 and not j["checks"]["tp_ranks_0_7_match_gpu_uuids"]


def test_judge_m5_h100_fails_on_process_on_unallocated_card(tmp_path):
    rc, j = _judge(_h100_run(tmp_path, stray_apps=[(0, 6, "ray::MegatronTrainRayActor")]))
    assert rc == 1 and j["verdict"] == "FAIL" and not j["checks"]["no_process_on_unallocated_gpus"]
    assert j["checks"]["tp_ranks_0_7_match_gpu_uuids"]  # the rank map alone would have passed


def test_judge_m5_h100_fails_on_wrong_node(tmp_path):
    rc, j = _judge(_h100_run(tmp_path, rank_gpu={3: (1, 3), 4: (0, 3)}))
    assert rc == 1 and not j["checks"]["tp_ranks_0_7_match_gpu_uuids"]


def test_judge_m5_prefers_resolved_probe_rows(tmp_path):
    import json

    run = _h100_run(tmp_path)
    f = run / "pulled" / "m5ranks.jsonl"
    f.write_text(f.read_text() + "\n" + json.dumps({"node": 0, "tp_rank": 0, "gpu_uuid": None, "method": "unresolved"}))
    rc, j = _judge(run)
    assert j["checks"]["tp_ranks_0_7_match_gpu_uuids"] is True
