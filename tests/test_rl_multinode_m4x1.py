"""M4 2x1 variant (m4a1/m4b1): g3 topology (fixed-partition resources-2x1.json T1R1S0,
trainer n0:0, rollout n1:0), TP1 PP1, + checkpoint store.

CPU checks: s1run dry-run argv (A/B identical except --rl-elastic-accept-rebind and the
instance type, steps 8, GPU spec switch), the launcher's pre-provision validation accepts
both argvs, and the m4 judge's 2-uuid rule."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from yeto import launcher
from yeto.cli import build_parser
from yeto.gpu_spec import parse_gpu_spec

GPU_DIR = Path(__file__).parent / "multinode_gpu"
A_ITYPE, B_ITYPE = "gpu-l40s-a_1gpu-16vcpu-64gb", "gpu-l40s-d_1gpu-16vcpu-96gb"


def _dry(case, **env):
    out = subprocess.run(["bash", str(GPU_DIR / "s1run.sh"), case, f"p-{case}", "100"], capture_output=True, text=True,
                         env={**os.environ, "DRY": "1", "IMAGE": "docker:x", "STORE": "s3://b/x", **env},
                         check=True).stdout.splitlines()
    return out[0], shlex.split(out[1])


def test_s1run_m4x1_argv_identical_except_rebind_and_itype():
    ha, a = _dry("m4a1", ITYPE=A_ITYPE, M4B_ITYPE=B_ITYPE)
    hb, b = _dry("m4b1", ITYPE=A_ITYPE, M4B_ITYPE=B_ITYPE)
    assert "nodes=2" in ha and "case=m4b1" in hb
    for argv in (a, b):
        assert argv[argv.index("--gpu") + 1] == "nebius:2x1xl40s@eu-north1"
        assert argv[argv.index("--total-steps") + 1] == "8"
        assert argv.count("--learner-instance-type") == 1
        assert argv[argv.index("--rl-placement") + 1] == "fixed-partition"
        assert argv[argv.index("--rl-elastic-resources") + 1].endswith("/resources-2x1.json")
        assert argv[argv.index("--rl-elastic-initial-config") + 1] == "T1R1S0"
        assert argv[argv.index("--pipeline-parallel") + 1] == "1" and argv[argv.index("--tensor-parallel") + 1] == "1"
        assert argv[argv.index("--rl-checkpoint-store") + 1] == "s3://b/x"
    assert a[a.index("--learner-instance-type") + 1] == A_ITYPE
    assert b[b.index("--learner-instance-type") + 1] == B_ITYPE
    strip = lambda v: [x for i, x in enumerate(v) if x not in ("--rl-elastic-accept-rebind", "--learner-instance-type")
                       and v[i - 1] not in ("--learner-instance-type", "--cluster-prefix")]
    assert "--rl-elastic-accept-rebind" in b and "--rl-elastic-accept-rebind" not in a
    assert strip(a) == strip(b)


def test_s1run_m4x1_h100_and_store_required():
    _, b = _dry("m4b1", M4X1_GPU="nebius:2x1xh100@eu-north1")
    spec = b[b.index("--gpu") + 1]
    assert [(s.num_nodes, s.gpus_per_node, s.gpu) for s in parse_gpu_spec(spec)] == [(2, 1, "H100")]
    assert "--learner-instance-type" not in b
    r = subprocess.run(["bash", str(GPU_DIR / "s1run.sh"), "m4a1", "p", "1"], capture_output=True, text=True,
                       env={k: v for k, v in {**os.environ, "DRY": "1", "IMAGE": "docker:x"}.items() if k != "STORE"})
    assert r.returncode == 66


@pytest.mark.parametrize("case", ["m4a1", "m4b1"])
def test_launcher_accepts_m4x1_argv(case):
    _, argv = _dry(case, ITYPE=A_ITYPE, M4B_ITYPE=B_ITYPE)
    launcher._check_ports_infra_switches(build_parser().parse_args(argv), "ports")


def _u(i):
    return f"GPU-{i:08x}-0000-0000-0000-000000000000"


def _w(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _fake(tmp, *, b_uuids, b_case="m4b1"):
    a, b = tmp / "a", tmp / "b"
    for d in (a, b):
        (d / "pulled").mkdir(parents=True)
    inc = {"incarnation": "incA"}
    _w(a / "pulled/journal.jsonl", [{"kind": "topology", "layout": {"pp": 2}},
                                    {"kind": "gpu_pool", "accepted": True, "uuids": [[_u(1)], [_u(2)]], "incarnation": "incA"},
                                    {"kind": "node_lost"}])
    _w(a / "pulled/rl-island-0.jsonl", [{"event": "rl_driver_phase", "phase": "train", "rollout_id": r} for r in (0, 1, 2)]
       + [{"event": "rl_round_cut", "ok": True, "store_synced": True, "cut_id": "r000002-incA"},
          {"event": "rl_reconfiguration", "result": "RECOVERY_REQUIRED", "error": "node_lost: n1"}])
    (a / "pulled/gpu-a.txt").write_text(f"h\n0, {_u(1)}\n")
    _w(b / "pulled/journal.jsonl", [{"kind": "checkpoint_store", "action": "restore", "restored_from": inc},
                                    {"kind": "topology", "layout": {"pp": 2}, "layout_accepted": True},
                                    {"kind": "gpu_pool", "accepted": True, "rebind": True, "mapping": {"x": "y"},
                                     "uuids": b_uuids},
                                    {"kind": "round_cut", "action": "restore", "cut_incarnation": "incA"}])
    _w(b / "pulled/rl-island-0.jsonl", [{"event": "rl_round_cut_restored", "rollout_id": 2}]
       + [{"event": "rl_driver_phase", "phase": "train", "rollout_id": r} for r in (2, 3)])
    (b / "pulled/gpu-b.txt").write_text("".join(f"{i}, {u}\n" for i, u in enumerate(x for n in b_uuids for x in n)))
    (b / "rc.txt").write_text("rc=0\n")
    (b / "case.txt").write_text(b_case + "\n")
    return a, b


def _judge(a, b):
    subprocess.run([sys.executable, str(GPU_DIR / "s1judge.py"), str(a), "m4", str(b)], capture_output=True, text=True)
    return json.loads((b / "judgment-m4.json").read_text())


def test_judge_m4b1_two_new_uuids_pass(tmp_path):
    j = _judge(*_fake(tmp_path, b_uuids=[[_u(3)], [_u(4)]]))
    assert j["verdict"] == "PASS", j


@pytest.mark.parametrize("uuids", [[[_u(3), _u(4)], [_u(5), _u(6)]],   # 2x2 shape on a 2x1 case
                                   [[_u(3), _u(4)]],                   # both on one node
                                   [[_u(1)], [_u(4)]]])                # A's uuid reused (no machine replacement)
def test_judge_m4b1_refuses_wrong_pool(tmp_path, uuids):
    j = _judge(*_fake(tmp_path, b_uuids=uuids))
    assert j["verdict"] == "FAIL" and not j["checks"]["b_uuids_new_and_on_b_nodes"], j


def test_judge_m4b_keeps_four_uuids(tmp_path):
    j = _judge(*_fake(tmp_path, b_uuids=[[_u(3)], [_u(4)]], b_case="m4b"))
    assert not j["checks"]["b_uuids_new_and_on_b_nodes"]
