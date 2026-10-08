"""S11 try26: the Flash-Next recipe (FlashInfer GDN prefill) must run non-deterministic SGLang inference."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GPU = os.path.join(HERE, "multinode_gpu")
REPO = os.path.dirname(HERE)


def _fp(extra=()):
    env = dict(os.environ, FN_GPU="h100")
    return subprocess.run(
        [sys.executable, os.path.join(GPU, "fp_fn.py"), REPO, "fn8s", "--seed", "17", "--total-steps", "6", *extra],
        capture_output=True, text=True, env=env, cwd=REPO,
    )


def test_fnrun_cases_disable_deterministic_inference():
    for case in ("fn8s", "fn32s", "fn32b"):
        out = subprocess.run(["bash", os.path.join(GPU, "fnrun.sh"), case], capture_output=True, text=True,
                             env=dict(os.environ, FN_GPU="h100"), check=True).stdout
        assert "--no-sglang-deterministic-inference" in out.split(), case


def test_fntrain_disables_deterministic_inference():
    text = open(os.path.join(GPU, "fntrain.sh")).read()
    assert "--no-sglang-deterministic-inference" in text


def test_fn8s_argv_has_flashinfer_prefill_and_no_deterministic():
    import json

    r = _fp()
    assert r.returncode == 0, r.stderr[-2000:]
    argv = json.loads(r.stdout.strip().splitlines()[-1])["argv"]
    assert "--sglang-enable-deterministic-inference" not in argv
    i = argv.index("--sglang-linear-attn-prefill-backend")
    assert argv[i + 1] == "flashinfer"


def test_recipe_rejects_deterministic_inference():
    from types import SimpleNamespace

    from yeto.rl.engine import run_config

    src = open(run_config.__file__).read()
    assert "needs --no-sglang-deterministic-inference" in src
    r = _fp(["--sglang-deterministic-inference"])
    assert r.returncode != 0
    assert "no-sglang-deterministic-inference" in (r.stderr + r.stdout)
