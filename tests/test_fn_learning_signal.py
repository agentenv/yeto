"""fn8r (Flash-Next 4-layer stage-A rerun with a learning signal): dense length reward,
learning-signal judge, fnrun/s1run/fp_fn case and the s11h200chain fn switches (CPU only)."""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
MG = REPO / "tests" / "multinode_gpu"
TRY27 = (REPO / "openspec/changes/rl-infra-spec/evidence/fn/try27-s12-h100-20261006a/fna/pulled"
         / "rl-island-0.jsonl")
sys.path.insert(0, str(REPO / "scripts"))
import judge_fn_learning_signal as jfs  # noqa: E402

from yeto.rl import length_reward as lr  # noqa: E402


# ------------------------------------------------------------------ reward
def _sample(text, status):
    return SimpleNamespace(response=text, label="7", status=SimpleNamespace(name=status))


def test_length_reward_is_dense_and_prefers_short_finished():
    r = lambda t, s: asyncio.run(lr.reward_func(None, _sample(t, s)))  # noqa: E731
    assert r("", "COMPLETED") == 1.0
    assert r("x" * 5000, "TRUNCATED") == 0.0
    assert 0.0 < r("x" * 400, "TRUNCATED") < r("x" * 300, "TRUNCATED") < 0.5
    assert r("x" * 300, "COMPLETED") == pytest.approx(r("x" * 300, "TRUNCATED") + 0.5)
    # all-truncated group still has variance (try27: 95% truncated -> gsm8k all 0)
    group = [r("y" * n, "TRUNCATED") for n in (310, 355, 402, 290, 377, 333, 368, 345)]
    assert max(group) - min(group) > 0
    assert asyncio.run(lr.reward_func(None, SimpleNamespace(response=None))) == 0.5


def test_length_reward_spec_is_a_valid_callable_in_repo():
    from yeto.provenance import python_spec_path
    from yeto.rl.engine.run_config import _validate_callable_spec

    _validate_callable_spec("yeto.rl.length_reward:reward_func")
    assert python_spec_path("yeto.rl.length_reward:reward_func", base_dir=REPO) == \
        REPO / "yeto/rl/length_reward.py"


# ------------------------------------------------------------------ judge
def test_signal_judge_fails_try27():
    r = subprocess.run([sys.executable, str(REPO / "scripts/judge_fn_learning_signal.py"), str(TRY27)],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "verdict: FAIL" in r.stdout and "nonzero_grad: FAIL" in r.stdout
    res = jfs.judge(jfs.load(str(TRY27)))
    assert res["checks"]["rounds"]["count"] == 6
    assert not res["checks"]["group_variance"]["ok"]
    assert res["checks"]["export_hash_change"]["status"] == "unchanged"


def _synthetic(tmp_path, *, grads=(0.0, 0.12, 0.09), zv=(1.0, 0.25, 0.0), hashes=("aa", "aa", "bb", "cc")):
    ev = [{"event": "rl_local_round", "grad_norm": g, "zero_variance_group_ratio": z, "reward_std": 0.1}
          for g, z in zip(grads, zv)]
    ev += [{"event": "rl_publication", "sync/publication_payload_hash": h} for h in hashes]
    p = tmp_path / "rl-island-0.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in ev) + "\nnot json\n")
    return p


def test_signal_judge_passes_synthetic_nonzero_grad(tmp_path):
    p = _synthetic(tmp_path)
    out = tmp_path / "j.json"
    r = subprocess.run([sys.executable, str(REPO / "scripts/judge_fn_learning_signal.py"), str(p),
                        "--min-rounds", "3", "--json", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    res = json.loads(out.read_text())
    assert res["verdict"] == "PASS"
    assert res["checks"]["nonzero_grad"]["rounds_with_grad"] == 2
    assert res["checks"]["export_hash_change"]["status"] == "changed"


def test_signal_judge_edge_cases(tmp_path):
    # no publications: export check unavailable -> PASS unless required
    p = _synthetic(tmp_path, hashes=())
    assert jfs.judge(jfs.load(str(p)))["verdict"] == "PASS"
    assert jfs.judge(jfs.load(str(p)), require_export_hash=True)["verdict"] == "FAIL"
    # grad > 0 but every group zero-variance -> FAIL; NaN grad does not count
    p = _synthetic(tmp_path, grads=(float("nan"), 0.0), zv=(1.0, 1.0))
    res = jfs.judge(jfs.load(str(p)))
    assert res["verdict"] == "FAIL" and not res["checks"]["nonzero_grad"]["ok"]
    assert jfs.main([str(tmp_path / "missing.jsonl")]) == 2


# ------------------------------------------------------------------ fnrun / s1run / fp_fn
def _fnrun(case, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("BOOT_ONLY", "FN_GPU")} | env
    return shlex.split(subprocess.run(["bash", str(MG / "fnrun.sh"), case], env=e, check=True,
                                      capture_output=True, text=True).stdout)


def _val(toks, flag):
    return toks[toks.index(flag) + 1]


def test_fnrun_fn8r_differs_from_fn8s_only_in_reward_length_and_sampling():
    s, r = _fnrun("fn8s"), _fnrun("fn8r")
    assert _val(s, "--reward-function") == "gsm8k_reward:score"
    assert _val(s, "--rollout-max-response-len") == "1024" and _val(s, "--seq-len") == "2048"
    assert "--rl-resource-sample-interval" not in s
    assert _val(r, "--reward-function") == "yeto.rl.length_reward:reward_func"
    assert _val(r, "--rollout-max-response-len") == "128" and _val(r, "--seq-len") == "1024"
    assert float(_val(r, "--rl-resource-sample-interval")) <= 10
    changed = {"--reward-function", "--rollout-max-response-len", "--seq-len", "--rl-resource-sample-interval"}

    def strip(t):
        out, i = [], 0
        while i < len(t):
            if t[i] in changed:
                i += 2
                continue
            out.append(t[i]); i += 1
        return out
    assert strip(s) == strip(r)
    assert "--rl-boot-only" in _fnrun("fn8r", BOOT_ONLY="1")
    assert "nebius:1x8xh100@eu-north1" in _fnrun("fn8r", FN_GPU="h100")


def test_s1run_dry_fn8r():
    env = {**os.environ, "DRY": "1", "BOOT_ONLY": "0", "THREAD_MAX": "99999"}
    r = subprocess.run(["bash", str(MG / "s1run.sh"), "fn8r", "pfx", "100", "200"],
                       env=env, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    toks = shlex.split(r.stdout.splitlines()[-1])
    assert _val(toks, "--reward-function") == "yeto.rl.length_reward:reward_func" and "--keep" in toks


def test_launcher_forwards_resource_sample_interval():
    from yeto import launcher
    base = dict(rl_observe_timeline=True)
    _, flags = launcher._ports_infra_flags(SimpleNamespace(**base, rl_resource_sample_interval=10.0))
    assert " --rl-resource-sample-interval 10.0" in flags
    _, flags = launcher._ports_infra_flags(SimpleNamespace(**base))
    assert "--rl-resource-sample-interval" not in flags


def test_fp_fn_fn8r_fingerprint():
    sys.path.insert(0, str(MG))
    try:
        import fp_fn
    finally:
        sys.path.pop(0)
    fp = fp_fn.fn_fingerprint(str(REPO), "fn8r", seed=17, total_steps=6)
    a = fp["argv"]
    assert fp["recipe"] == "qwen3_8_next" and fp["fp"].startswith("sha256:")
    assert _val(a, "--custom-rm-path") == "yeto.rl.length_reward.reward_func"
    assert _val(a, "--rollout-max-response-len") == "128" and _val(a, "--num-layers") == "4"
    assert fp["fp"] != fp_fn.fn_fingerprint(str(REPO), "fn8s", seed=17, total_steps=6)["fp"]


# ------------------------------------------------------------------ chain switches
def _plan(tmp_path, **env):
    e = {k: v for k, v in os.environ.items() if k not in ("FNA_CASE", "FNA_SKIP", "FN_CONV", "FN_GPU")}
    e |= {"FN_PLAN_ONLY": "1", "RUN_ROOT": str(tmp_path)} | env
    return subprocess.run(["bash", str(MG / "s11h200chain.sh"), "pfx"], env=e, capture_output=True, text=True)


def test_chain_plan_defaults_unchanged(tmp_path):
    r = _plan(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "case=fn8s skip_fna=0 fnconv=1" in r.stdout and "segs=fnboot,fnprep,fna,fnconv" in r.stdout
    assert "signal=" not in r.stdout
    assert "segs=fnboot,fnprep,fna " in _plan(tmp_path, FN_GPU="h100").stdout


def test_chain_plan_switches(tmp_path):
    r = _plan(tmp_path, FNA_CASE="fn8r")
    assert "case=fn8r" in r.stdout and "signal=judge_fn_learning_signal.py" in r.stdout
    r = _plan(tmp_path, FNA_SKIP="1")
    assert r.returncode == 0 and "segs=fnboot,fnprep,fnconv " in r.stdout
    assert _plan(tmp_path, FNA_SKIP="1", FN_GPU="h100").returncode == 64
    assert _plan(tmp_path, FNA_CASE="fn32s").returncode == 64
    assert _plan(tmp_path, FNA_SKIP="2").returncode == 64


def test_chain_judge_args_and_signal_judge():
    src = (MG / "s11h200chain.sh").read_text()
    assert subprocess.run(["bash", "-n", str(MG / "s11h200chain.sh")]).returncode == 0
    assert 'FNJ="--num-gpus 8 --rollouts 6 --rank 16 --expert-rank 8"' in src
    assert "judge_qwen3_8_next_lora_log.py $B/$P-$1/pulled/run.log $FNJ " in src
    assert "judge_fn_learning_signal.py $B/$P-$1/pulled/rl-island-0.jsonl" in src
    assert "$D/s1run.sh $FC $P-$1" in src and "fp_fn.py $REPO $FC " in src
    assert 'if [ $FSKIP = 1 ]; then log "fna skipped' in src


def test_fnprep_records_full_snapshot_marker():
    src = (MG / "s11fnprep.sh").read_text()
    assert "yeto-complete/Qwen--Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540.json" in src
    assert '"full_snapshot": g("FULL_SNAPSHOT")' in src
    assert subprocess.run(["bash", "-n", str(MG / "s11fnprep.sh")]).returncode == 0
