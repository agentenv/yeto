"""FN-A alignment: the ports Flash-Next argv matches scripts/run_qwen3_8_next.py
(ref-load torch_dist, expert rank, perf/health flags), the legacy path refuses the
recipe, and the stage-A small slice (fnrun.sh fn8s) renders the 4-layer layout."""
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.engine import run_config as rc
from yeto.rl.profiles import qwen3_8_next as q

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "multinode_gpu"))
import fp_fn  # noqa: E402

# perf_args / sglang_args / misc_args of scripts/run_qwen3_8_next.py (miles 5fb286567)
HISTORICAL = shlex.split(
    "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    "--micro-batch-size 1 --max-tokens-per-gpu 8192 --sglang-chunked-prefill-size 8192 "
    "--router-health-success-threshold 1 --router-health-check-interval-secs 15 "
    "--router-health-failure-threshold 40 --update-weight-buffer-size 1073741824 "
    "--train-memory-margin-bytes 3221225472 --rollout-health-check-interval 300 "
    "--rollout-health-check-timeout 300 --distributed-timeout-minutes 60 "
    "--model-name qwen4_exp --qkv-format thd --linear-attention-backend flashqla "
    "--sglang-linear-attn-prefill-backend flashinfer --sglang-moe-runner-backend triton")


def _get(argv, flag):
    assert argv.count(flag) == 1, (flag, argv.count(flag))
    return argv[argv.index(flag) + 1]


@pytest.fixture(scope="module")
def runs():
    return {c: fp_fn.fn_fingerprint(str(REPO), c) for c in ("fn32s", "fn8s")}


def test_historical_runtime_flags(runs):
    for case, run in runs.items():
        argv = run["argv"]
        for i in range(0, len(HISTORICAL), 2):
            assert _get(argv, HISTORICAL[i]) == HISTORICAL[i + 1], (case, HISTORICAL[i])
        assert "--sglang-disable-radix-cache" in argv
        assert _get(argv, "--attention-backend") == "auto"  # script sets none


def test_ref_load_is_torch_dist_and_expert_rank_from_profile(runs):
    for case, variant in (("fn32s", "full"), ("fn8s", "4layer")):
        argv = runs[case]["argv"]
        assert _get(argv, "--ref-load") == (
            f"/mnt/yeto-models/torch_dist/{q.MEGATRON_MODEL_TYPES[variant]}_torch_dist")
        assert _get(argv, "--ref-load") != _get(argv, "--hf-checkpoint")
        assert _get(argv, "--megatron-to-hf-mode") == "raw"
        assert _get(argv, "--lora-expert-rank") == str(q.Qwen38NextLoraProfile.lora_expert_rank)


def test_fn8s_matches_4layer_8gpu_layout(runs):
    argv = runs["fn8s"]["argv"]
    assert runs["fn8s"]["recipe"] == rc.RECIPE_QWEN3_8_NEXT
    flat = " " + " ".join(argv) + " "
    for group in q.model_args("4layer"):
        assert f" {group.strip()} " in flat, group
    prof = q.Qwen38NextLoraProfile(variant="4layer").parallel
    assert (_get(argv, "--tensor-model-parallel-size"), _get(argv, "--pipeline-model-parallel-size"),
            _get(argv, "--expert-model-parallel-size")) == (
        str(prof["tp"]), str(prof["pp"]), str(prof["ep"]))
    assert _get(argv, "--rollout-num-gpus-per-engine") == str(prof["rollout_num_gpus_per_engine"])
    assert _get(argv, "--sglang-tp-size") == _get(argv, "--sglang-ep-size") == "4"
    assert "--colocate" in argv and _get(argv, "--sglang-mem-fraction-static") == "0.7"
    assert _get(argv, "--actor-num-nodes") == "1" and _get(argv, "--actor-num-gpus-per-node") == "8"


def test_expert_rank_override_and_bounds():
    fp = fp_fn.fn_fingerprint(str(REPO), "fn8s", extra=("--rl-lora-expert-rank", "16"))
    assert _get(fp["argv"], "--lora-expert-rank") == "16"
    with pytest.raises(ValueError, match="r_e <= --lora-r"):
        rc._lora_expert_rank(SimpleNamespace(rl_lora_expert_rank=32, lora_r=16),
                             rc.RECIPE_QWEN3_8_NEXT)
    with pytest.raises(ValueError, match="only applies"):
        rc._lora_expert_rank(SimpleNamespace(rl_lora_expert_rank=8, lora_r=16), rc.RECIPE_QWEN3_5)
    assert rc._lora_expert_rank(SimpleNamespace(lora_r=16), rc.RECIPE_GENERIC) is None


def test_missing_ref_load_refused(monkeypatch):
    orig = fp_fn.fnrun_cli

    def without_ref(case, **kw):
        toks = orig(case, **kw)
        i = toks.index("--rl-megatron-ref-load")
        return toks[:i] + toks[i + 2:]

    monkeypatch.setattr(fp_fn, "fnrun_cli", without_ref)
    with pytest.raises(ValueError, match="needs --megatron-ref-load"):
        fp_fn.fn_fingerprint(str(REPO), "fn8s")


def test_legacy_path_refuses_flash_next(monkeypatch):
    from yeto.rl import learner

    cfg = SimpleNamespace(model_recipe=SimpleNamespace(name=rc.RECIPE_QWEN3_8_NEXT))
    with pytest.raises(ValueError, match="ports-only"):
        learner._legacy_miles_argv(cfg)


def test_launcher_forwards_and_exports_fn_env(tmp_path):
    import _pytest.monkeypatch as _m
    from rl_e2e_launch import learner_from_run
    from test_rl_engine_selection import _cli, _island_task
    from yeto.launcher import _prepare_rl_args

    mp = _m.MonkeyPatch()
    try:
        cli = _cli(tuple(fp_fn.fnrun_cli("fn8s")))
        _prepare_rl_args(cli)
        task = _island_task(cli, mp)
        args, _env = learner_from_run(task.run, tmp_path / "home")
    finally:
        mp.undo()
    assert args.megatron_ref_load.endswith("qwen3.8-flash-next-4layer_torch_dist")
    assert args.rl_lora_expert_rank == 8
    for k, v in q.PORTS_RUNTIME_ENV.items():
        assert task.envs.get(k) == v, k
