"""fn-train: the formal Flash-Next RL launch (tests/multinode_gpu/fntrain.sh), CPU only.

The printed argv must (1) parse with the real launcher CLI, (2) shape the 4x8 island
for both fixed configs, (3) reach the learner with recommend-mode, the checkpoint store,
torch_dist ref-load, expert rank 8, dapo-math data and the math reward, and (4) render the
full Miles argv through the FN recipe (same pipeline as fp_fn.py). No cloud call.
"""
import os
import shlex
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tests" / "multinode_gpu" / "fntrain.sh"
sys.path.insert(0, str(REPO / "tests" / "multinode_gpu"))
import fp_fn  # noqa: E402

REF = "/mnt/yeto-models/torch_dist/qwen3.8-flash-next_torch_dist"


def _argv(shape="b", **env):
    base = {"PATH": "/usr/bin:/bin", "RUN": "fnt-test"}
    base.update(env)
    out = subprocess.run(["bash", str(SCRIPT), "print", shape], check=True, capture_output=True,
                         text=True, env=base).stdout
    toks = shlex.split(out.strip())
    assert toks[0] == "launch"
    return toks[1:]


def _get(argv, flag):
    assert argv.count(flag) == 1, (flag, argv.count(flag))
    return argv[argv.index(flag) + 1]


def test_print_carries_the_training_contract():
    a = _argv()
    assert _get(a, "--rl-megatron-ref-load") == REF
    assert _get(a, "--rl-lora-expert-rank") == "8" and _get(a, "--lora-r") == "16"
    assert _get(a, "--data") == "zhuzilin/dapo-math-17k"
    assert len(_get(a, "--data-revision")) == 40 and len(_get(a, "--model-revision")) == 40
    assert _get(a, "--reward-function") == "yeto.rl.math_reward:reward_func"
    assert _get(a, "--rl-recommend-mode") == "recommend" and "--rl-observe-timeline" in a
    assert _get(a, "--rl-checkpoint-store") == "s3://yeto-rl-ckpt-ddde6f79/fn-train/fnt-test"
    assert _get(a, "--gpu") == "nebius:4x8xh200@eu-north1"
    assert _get(a, "--total-steps") == "200" and _get(a, "--inner-lr") == "1e-6"
    for f in ("--rl-elastic-attestation", "--rl-edge-costs-path", "--rl-elastic-window-s"):
        assert f not in a  # nothing certified / no cost table / window unset by default
    a = _argv(COSTS="/x/c.json", WINDOW="900", STEPS="40")
    assert _get(a, "--rl-edge-costs-path") == "/x/c.json"
    assert _get(a, "--rl-elastic-window-s") == "900" and _get(a, "--total-steps") == "40"


@pytest.mark.parametrize("shape,config,rollout", [("s", "FN-T16R8S8", 8), ("b", "FN-T16R16S0", 16)])
def test_launcher_dry_run(monkeypatch, tmp_path, shape, config, rollout):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    from yeto import launcher
    from yeto.cli import parse_args
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = parse_args(_argv(shape, COSTS=str(tmp_path / "c.json"), WINDOW="900")
                      + ["--rl-elastic-state-dir", str(tmp_path / "state")])
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    spec = parse_gpu_spec(args.gpu)[0]
    assert (spec.num_nodes, spec.gpus_per_node) == (4, 8)
    task = launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")
    assert launcher.rl_island_layout(args, spec)[:2] == (2, 8)
    for s in ("--rl-recommend-mode recommend", f"--rl-elastic-initial-config {config}",
              f"--rollout-num-gpus {rollout}", "--rl-elastic-window-s 900.0",
              "--actor-num-nodes 2 --actor-num-gpus-per-node 8"):
        assert s in task.run, s
    assert launcher.rl_checkpoint_store_plan(args) == (
        launcher.ELASTIC_CHECKPOINT_STORE_MOUNT + "/fn-train/fnt-test", "s3://yeto-rl-ckpt-ddde6f79")
    assert "yeto-checkpoint-store/fn-train/fnt-test" in task.run
    assert REF in task.run


@pytest.fixture(scope="module")
def miles_argv():
    for p in (str(REPO), str(REPO / "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import _pytest.monkeypatch as _m

    from rl_e2e_launch import island_run, learner_from_run
    from yeto.rl import learner
    from yeto.rl.engine import run_config
    from yeto.rl.engine.run_config import resolve_rl_run_config
    from yeto.rl.profiles import qwen3_8_next as q

    mp = _m.MonkeyPatch()
    try:
        run = island_run(tuple(_argv("b")), mp)
        args, _env = learner_from_run(run, Path(tempfile.mkdtemp()) / "home")
        mp.setattr(run_config, "_resolve_ref_load",
                   lambda a, model_path: a.megatron_ref_load or str(model_path))
        rc = resolve_rl_run_config(
            args, model_path=fp_fn.FN_SNAPSHOT, rollout_model_path=None,
            prompt_path="/root/yeto-rl/prompts.jsonl", eval_prompt_path=None,
            provider=fp_fn.fn_provider(48),
            target_modules=sorted({m.rsplit(".", 1)[-1] for m in q.LORA_TARGET_MODULES}),
            yeto_policy_sync=False)
    finally:
        mp.undo()
    return list(learner.build_ports_launch(args, rc, ()).argv)


def test_full_recipe_miles_argv(miles_argv):
    a = miles_argv
    if os.environ.get("FNTRAIN_DUMP"):  # archive the rendered Miles argv for the plan
        Path(os.environ["FNTRAIN_DUMP"]).write_text(shlex.join(a) + "\n")
    assert _get(a, "--ref-load") == REF and _get(a, "--megatron-to-hf-mode") == "raw"
    assert _get(a, "--lora-expert-rank") == "8"
    assert _get(a, "--num-layers") == "48" and _get(a, "--num-experts") == "512"
    assert _get(a, "--tensor-model-parallel-size") == "2"
    assert _get(a, "--pipeline-model-parallel-size") == "8"
    assert _get(a, "--expert-model-parallel-size") == "2"
    assert _get(a, "--n-samples-per-prompt") == "8"
    assert _get(a, "--rollout-max-response-len") == "4096"
    assert _get(a, "--max-tokens-per-gpu") == "8192"


def test_convert_renders_full_b0_command():
    out = subprocess.run(["bash", str(SCRIPT), "convert"], check=True, capture_output=True, text=True,
                         env={"PATH": os.path.dirname(sys.executable) + ":/usr/bin:/bin"}).stdout
    cmd = shlex.split(out.strip().splitlines()[-1])
    assert cmd[0] == "torchrun" and _get(cmd, "--save") == REF
    assert _get(cmd, "--num-layers") == "48" and _get(cmd, "--nproc-per-node") == "8"
    # PP4 x TP2 on one 8-GPU node (TP2 PP1 would need ~180 GB/rank, estimate)
    assert _get(cmd, "--tensor-model-parallel-size") == "2"
    assert _get(cmd, "--pipeline-model-parallel-size") == "4"
