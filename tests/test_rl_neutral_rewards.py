"""Neutral rewards, filters and their Miles wrappers (yeto-framework-decoupling group 3).

CPU only; Miles is never imported (math grading uses the test-only copy in
``tests/vendor/``, task 3.9).
"""

from __future__ import annotations

import asyncio
import ast
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import reward_equivalence_cases as cases
from yeto.rl.engine.miles_adapter import rewards as miles_rewards
from yeto.rl.rewards import FilterDecision, RewardResult, Trajectory, builtin, registry

REPO = Path(__file__).resolve().parents[1]


def _run(coro):
    return asyncio.run(coro)


# -- 3.1 types -----------------------------------------------------------------

def test_types_defaults_and_fields():
    t = Trajectory()
    assert t.response == "" and t.metadata == {} and t.status is None
    assert Trajectory().metadata is not t.metadata
    fields = set(Trajectory.__dataclass_fields__)
    assert {"group_index", "index", "prompt", "response", "label", "tokens", "rollout_logprobs",
            "loss_mask", "policy_token", "segments", "status", "metadata"} <= fields
    r = RewardResult(1.0)
    assert (r.aborted, r.reason, r.metadata, r.extra) == (False, None, None, None)
    with pytest.raises(Exception):
        r.value = 2.0  # frozen
    assert FilterDecision(False, "x") == FilterDecision(keep=False, reason="x")


def test_boundary_check_covers_rewards_dir():
    import test_import_boundaries as tib

    assert "yeto/rl/rewards" in tib.CORE_DIRS
    scanned = {str(p.relative_to(REPO)) for p in (REPO / "yeto/rl/rewards").glob("*.py")}
    assert {"yeto/rl/rewards/types.py", "yeto/rl/rewards/builtin.py",
            "yeto/rl/rewards/registry.py"} <= scanned


def test_rewards_package_imports_no_framework():
    for path in (REPO / "yeto/rl/rewards").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in {"miles", "verl", "megatron", "sglang", "torch"}, path
                assert not name.startswith(("yeto.rl.engine.miles_adapter", "yeto.rl.miles")), path


# -- 3.2 equivalence with the pre-refactor entry points --------------------------

def test_entry_points_match_recorded_baseline():
    """Values, statuses, metadata, filter keep/reason and state equal the recording."""
    rendered = cases.dumps(cases.render())
    expected = cases.BASELINE.read_text()
    if rendered != expected:
        old, new = json.loads(expected), json.loads(rendered)
        bad = [k for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]
        pytest.fail(f"neutral refactor changed {bad}")


def test_neutral_forms_without_miles():
    assert "miles" not in sys.modules or not hasattr(sys.modules["miles"], "rollout")
    r = builtin.gsm8k_reward(Trajectory(response="\\boxed{18}", label="#### 18"))
    assert (r.value, dict(r.metadata)) == (1.0, {"success": True})
    r = builtin.length_reward(Trajectory(response="ab", status="completed"))
    assert r.value == 0.5 + 0.5 * (1 - 2 / 1024) and r.metadata is None
    assert builtin.length_reward(Trajectory(response="ab", status="truncated")).value < 0.5
    flt = builtin.BoundedNonzeroStdFilter()
    group = [Trajectory(index=0, reward=0.0), Trajectory(index=1, reward=0.0)]
    assert flt(group, round_id=1, max_replacements=1) == FilterDecision(False, "zero_std_0.0")
    assert flt(group, round_id=1, max_replacements=1) == FilterDecision(False, "zero_std_0.0")
    other = [Trajectory(index=2, reward=1.0), Trajectory(index=3, reward=1.0)]
    assert flt(other, round_id=1, max_replacements=1).reason == "bounded_fallback_after_1_replacements"
    assert flt.state["rejections"] == 1 and flt.state["forced"] == 1
    assert flt(other, round_id=2, max_replacements=1).keep is False  # new round resets


def test_math_neutral_form_needs_miles_graders():
    with pytest.raises(ImportError):
        builtin.math_reward(Trajectory(response="\\boxed{1}", label="1"))


def test_filter_state_reset_by_round_hook_is_honoured():
    from yeto.rl import filters

    args = SimpleNamespace(yeto_rl_policy_version=1, yeto_rl_dynamic_sampling_max_replacements=1)
    group = [cases.FakeSample("r", None, index=0, reward=0.0)]
    assert filters.bounded_nonzero_reward_std(args, group).keep is False
    assert filters.bounded_nonzero_reward_std(args, group).keep is False  # memoized
    args._yeto_bounded_filter_state = None  # rollout_meta_hook per-round reset
    assert filters.bounded_nonzero_reward_std(args, group).keep is False
    assert args._yeto_bounded_filter_state["rejections"] == 1
    other = [cases.FakeSample("r", None, index=1, reward=0.0)]
    out = filters.bounded_nonzero_reward_std(args, other)
    assert (out.keep, out.reason) == (True, "bounded_fallback_after_1_replacements")


@pytest.mark.parametrize("name,entry", [
    ("gsm8k", "yeto.rl.gsm8k_reward:score"),
    ("length", "yeto.rl.length_reward:reward_func"),
    ("gdpo", "yeto.rl.algos.gdpo_reward:reward_func"),
    ("gdpo_correctness", "yeto.rl.algos.gdpo_reward:correctness_reward"),
    ("math", "yeto.rl.math_reward:reward_func"),
])
def test_generic_miles_wrapper_equals_original_entry_point(name, entry):
    module, _, attr = entry.partition(":")
    original = getattr(importlib.import_module(module), attr)
    wrapped = getattr(miles_rewards, name)
    source = {"gsm8k": cases.GSM8K_CASES, "math": cases.MATH_CASES}.get(name, cases.GDPO_CASES)
    rows = [(r, l, cases.Status.COMPLETED) for r, l in source]
    if name == "length":
        rows = [(r, None, s) for r, s in cases.LENGTH_CASES]
    with cases.vendor_math_utils():
        for i, (resp, label, status) in enumerate(rows):
            a = cases.FakeSample(resp, label, status=status, metadata=cases._metadata_seed(i))
            b = cases.FakeSample(resp, label, status=status, metadata=cases._metadata_seed(i))
            assert _run(original(None, a)) == _run(wrapped(None, b))
            assert a.snapshot() == b.snapshot(), (name, i)


# -- 3.3 codex: abort as a reward result --------------------------------------------

def _signed(status: str, reward: float, monkeypatch) -> dict:
    from yeto.rl.harness.codex import reward as codex_reward

    monkeypatch.setenv("SECRLENV_REWARD_HMAC_KEY", "k" * 48)
    outcome = {"schema": 1, "status": status, "episode_id": "ep", "task_id": "t", "reward": reward,
               "passed": reward == 1.0, "class": None,
               codex_reward.INFRASTRUCTURE_503_RETRIES_KEY: 0}
    return {codex_reward.OUTCOME_KEY: outcome, codex_reward.MAC_KEY: codex_reward.sign_outcome(outcome)}


@pytest.fixture
def fake_miles_sample(monkeypatch):
    names = ("miles", "miles.utils", "miles.utils.types")
    for n in names[:-1]:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    mod = types.ModuleType(names[-1])
    mod.Sample = cases.FakeSample
    monkeypatch.setitem(sys.modules, names[-1], mod)


def test_codex_infrastructure_outcome_aborts_in_neutral_and_miles(monkeypatch, fake_miles_sample):
    from yeto.rl.harness.codex import reward as codex_reward

    metadata = _signed("infrastructure_error", 0.0, monkeypatch)
    result = codex_reward.secrlenv_reward(Trajectory(metadata=metadata))
    assert (result.value, result.aborted, result.reason) == (0.0, True, "secrlenv_infrastructure_failure")
    sample = cases.FakeSample(metadata=dict(metadata))
    assert _run(codex_reward.reward_func(None, sample)) == 0.0
    assert sample.status is cases.Status.ABORTED
    generic = cases.FakeSample(metadata=dict(metadata))
    assert _run(miles_rewards.miles_reward(codex_reward.secrlenv_reward)(None, generic)) == 0.0
    assert generic.status is cases.Status.ABORTED


def test_codex_verdict_and_failed_signature(monkeypatch, fake_miles_sample):
    from yeto.rl.harness.codex import reward as codex_reward

    metadata = _signed("completed", 1.0, monkeypatch)
    result = codex_reward.secrlenv_reward(Trajectory(metadata=metadata))
    assert (result.value, result.aborted) == (1.0, False)
    sample = cases.FakeSample(metadata=dict(metadata))
    assert _run(codex_reward.reward_func(None, [sample])) == [1.0]
    assert sample.status is cases.Status.COMPLETED
    tampered = dict(metadata)
    tampered[codex_reward.MAC_KEY] = "0" * 64
    with pytest.raises(codex_reward.UntrustedOutcome):
        codex_reward.secrlenv_reward(Trajectory(metadata=tampered))
    with pytest.raises(codex_reward.UntrustedOutcome):
        _run(codex_reward.reward_func(None, cases.FakeSample(metadata=tampered)))


def test_tbench_infrastructure_marker_aborts(monkeypatch):
    from yeto.rl.harness.codex import tbench_reward

    metadata = {tbench_reward.INFRASTRUCTURE_KEY: "x", "success": True}
    result = tbench_reward.tbench_reward(Trajectory(metadata=metadata))
    assert result.aborted and result.value == 0.0 and "success" not in metadata
    sample = SimpleNamespace(metadata={tbench_reward.INFRASTRUCTURE_KEY: "x", "success": True},
                             status=None)
    assert _run(tbench_reward.reward_func(None, sample)) == 0.0
    assert sample.status == "ABORTED" and "success" not in sample.metadata


# -- 3.10 registry and user rewards ------------------------------------------------

def test_custom_reward_via_miles_equals_direct_call():
    from examples.custom_reward import reward as example

    assert "ends_with_label" in registry.registered_names()
    path = miles_rewards.miles_custom_rm_path("ends_with_label")
    assert path == "yeto.rl.engine.miles_adapter.rewards.ends_with_label"
    by_ref = miles_rewards.miles_custom_rm_path("examples.custom_reward.reward:ends_with_label")
    for p in (path, by_ref):
        module, _, attr = p.rpartition(".")  # what Miles' load_function does
        fn = getattr(importlib.import_module(module), attr)
        sample = cases.FakeSample("the answer is 7", "7", metadata=None)
        direct = example.ends_with_label(Trajectory(response="the answer is 7", label="7"))
        assert _run(fn(None, sample)) == direct.value == 1.0
        assert sample.metadata == dict(direct.metadata)
    assert not {m.split(".")[0] for m in sys.modules} & {"verl", "megatron"}


def test_unregistered_name_lists_available():
    with pytest.raises(registry.RewardRegistryError, match="not registered.*gsm8k"):
        miles_rewards.miles_custom_rm_path("no_such_reward")
    with pytest.raises(AttributeError):
        getattr(miles_rewards, "no_such_reward")


def test_identity_tracks_source(tmp_path, monkeypatch):
    pkg = tmp_path / "user_rewards_pkg"
    pkg.mkdir()
    src = pkg / "mine.py"
    body = ("from yeto.rl.rewards.types import RewardResult\n"
            "def r(t):\n    return RewardResult(1.0)\n")
    src.write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    first = registry.load_reward("user_rewards_pkg.mine:r")
    assert first.identity()["module"] == "user_rewards_pkg.mine"
    assert first.identity()["qualname"] == "r"
    src.write_text(body + "# changed\n")
    second = registry.load_reward("user_rewards_pkg.mine:r")
    assert first.source_sha256 != second.source_sha256


def test_register_conflict_rejected():
    def a(t):
        return RewardResult(0.0)

    def b(t):
        return RewardResult(0.0)

    registry.register_reward("conflict_test_name", a)
    registry.register_reward("conflict_test_name", a)  # same function: fine
    with pytest.raises(registry.RewardRegistryError, match="already registered"):
        registry.register_reward("conflict_test_name", b)
    with pytest.raises(registry.RewardRegistryError):
        registry.register_reward("bad name!", a)


# -- 3.9 math graders: test-only copy -------------------------------------------------

MILES_PIN_CHECKOUT = Path("/home/michael/work/miles-fr1")


def test_runtime_code_never_imports_the_vendor_copy():
    for path in (REPO / "yeto").rglob("*.py"):
        text = path.read_text(errors="replace")
        if "miles_math_utils" not in text and "tests.vendor" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                assert not any("miles_math_utils" in n or n.startswith(("tests", "vendor"))
                               for n in names), path


def test_vendor_copy_is_marked_and_matches_the_pinned_source():
    import subprocess

    text = (REPO / "tests/vendor/miles_math_utils.py").read_text()
    assert "许可证待核实、不进 main" in text.splitlines()[0]
    from yeto.rl import MILES_COMMIT

    if not (MILES_PIN_CHECKOUT / ".git").exists():
        pytest.skip("no local Miles checkout to compare the copy with")
    pinned = subprocess.run(
        ["git", "-C", str(MILES_PIN_CHECKOUT), "show",
         f"{MILES_COMMIT}:miles/rollout/rm_hub/math_utils.py"],
        capture_output=True, text=True)
    if pinned.returncode != 0:
        pytest.skip("pinned Miles commit not in the local checkout")
    body = text.split("# 以下自下一行起与 pin 提交中的文件逐字节相同。\n", 1)[1]
    assert body == pinned.stdout


def test_math_reward_with_vendor_copy_matches_pinned_miles_graders(tmp_path):
    """Case by case, yeto.rl.math_reward with the copy == with the pinned Miles file.

    The Miles package itself cannot be imported on this host, so the pinned
    commit's math_utils.py (read from git) stands in for an installed Miles.
    """
    import subprocess

    from yeto.rl import MILES_COMMIT, math_reward

    pinned = subprocess.run(
        ["git", "-C", str(MILES_PIN_CHECKOUT), "show",
         f"{MILES_COMMIT}:miles/rollout/rm_hub/math_utils.py"],
        capture_output=True, text=True)
    if pinned.returncode != 0:
        pytest.skip("pinned Miles commit not available locally")
    pinned_file = tmp_path / "pinned_math_utils.py"
    pinned_file.write_text(pinned.stdout)
    spec = importlib.util.spec_from_file_location("pinned_math_utils", pinned_file)
    real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real)
    with cases.vendor_math_utils():
        copy_scores = [math_reward.score(r, l) for r, l in cases.MATH_CASES]
    saved = {n: sys.modules.get(n) for n in ("miles", "miles.rollout", "miles.rollout.rm_hub",
                                             "miles.rollout.rm_hub.math_utils")}
    try:
        for n in list(saved)[:-1]:
            sys.modules[n] = types.ModuleType(n)
        sys.modules["miles.rollout.rm_hub.math_utils"] = real
        real_scores = [math_reward.score(r, l) for r, l in cases.MATH_CASES]
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
    assert copy_scores == real_scores
