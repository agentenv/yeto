"""G3: FN attestation fingerprint (tests/multinode_gpu/fp_fn.py) is reproducible
for fnrun.sh fn32s / fn32b and pins the exact run parameters (CPU only)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests" / "multinode_gpu"))
import fp_fn  # noqa: E402


@pytest.fixture(scope="module")
def fps():
    out = {}
    for case in ("fn32s", "fn32b"):
        for seed, steps in ((17, 12), (17, 12), (18, 12), (17, 20)):
            out.setdefault((case, seed, steps), []).append(
                fp_fn.fn_fingerprint(str(REPO), case, seed=seed, total_steps=steps))
    return out


def test_recipe_and_reproducible(fps):
    for key, runs in fps.items():
        assert all(r["recipe"] == "qwen3_8_next" for r in runs)
        assert len({r["fp"] for r in runs}) == 1, key  # recomputation gives the same value
        assert all(r["fp"].startswith("sha256:") for r in runs)


def test_fingerprint_pins_case_seed_and_steps(fps):
    values = {key: runs[0]["fp"] for key, runs in fps.items()}
    assert len(set(values.values())) == len(values) == 6


def test_fn_argv_layout(fps):
    for case, rollout in (("fn32s", "8"), ("fn32b", "16")):
        argv = fps[(case, 17, 12)][0]["argv"]
        get = lambda f: argv[argv.index(f) + 1]  # noqa: E731
        assert get("--model-name") == "qwen4_exp" and get("--num-layers") == "48"
        assert get("--rollout-num-gpus") == rollout and get("--actor-num-nodes") == "2"
        assert (get("--tensor-model-parallel-size"), get("--pipeline-model-parallel-size"),
                get("--expert-model-parallel-size")) == ("2", "8", "2")
        assert get("--sglang-tp-size") == "8" and get("--num-rollout") == "12"
