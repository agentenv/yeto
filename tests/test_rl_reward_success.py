"""Reward functions write the boolean full-success flag (VAPO positives) without changing scores."""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(coro):
    return asyncio.run(coro)


def _fake_miles(monkeypatch):
    mod = types.ModuleType("miles.rollout.rm_hub.math_utils")

    def extract_answer(text):
        i = text.rfind("\\boxed{")
        return text[i + len("\\boxed{"):].split("}")[0] if i >= 0 else None

    mod.extract_answer = extract_answer
    mod.grade_answer_mathd = lambda a, b: a == b
    mod.grade_answer_sympy = lambda a, b: False
    for name in ("miles", "miles.rollout", "miles.rollout.rm_hub"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.rollout.rm_hub.math_utils", mod)


def _fork_flag(metadata):
    """Independent restatement of the fork's sample_success_flag contract."""
    found = [int(metadata[k]) for k in ("success", "is_correct") if metadata.get(k) is not None]
    assert all(type(metadata[k]) is bool for k in ("success", "is_correct") if metadata.get(k) is not None)
    assert len(set(found)) <= 1
    return found[0] if found else -1


@pytest.mark.parametrize("response,label,want", [
    ("so \\boxed{42}", "42", 1.0),
    ("<think>\\boxed{42}</think> \\boxed{41}", "42", 0.0),
    ("no box", "42", 0.0),
    ("\\boxed{3}", "", 0.0),
])
def test_math_reward_success(monkeypatch, response, label, want):
    _fake_miles(monkeypatch)
    from yeto.rl import math_reward

    for metadata in ({}, None, {"other": 1}):
        sample = SimpleNamespace(response=response, label=label, metadata=metadata)
        assert _run(math_reward.reward_func(None, sample)) == want == math_reward.score(response, label)
        assert sample.metadata["success"] is (want == 1.0)
        assert _fork_flag(sample.metadata) == int(want)
        if metadata is not None and "other" in metadata:
            assert sample.metadata["other"] == 1


def _reference_gsm8k():
    # frozen GPU-run copy; not imported from yeto
    path = REPO / "openspec/changes/rl-algo-seq-and-adv/evidence/g1/gsm8k_reward.py"
    spec = importlib.util.spec_from_file_location("ref_gsm8k_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("response,label", [
    ("The answer is \\boxed{1,234}", "work #### 1234"),
    ("... so 17 apples", "#### 17"),
    ("\\boxed{17} then 18", "#### 18"),
    ("\\boxed{-3.5}", "#### -3.5"),
    ("nothing", "#### 5"),
    ("\\boxed{5}", None),
    ("", "#### 0"),
])
def test_gsm8k_reward_matches_reference_and_sets_success(response, label):
    from yeto.rl import gsm8k_reward

    ref = _reference_gsm8k()
    want = _run(ref.score(None, SimpleNamespace(response=response, label=label)))
    sample = SimpleNamespace(response=response, label=label, metadata={})
    assert _run(gsm8k_reward.score(None, sample)) == want
    assert sample.metadata["success"] is (want == 1.0)


def test_gdpo_success_is_correctness_not_format(monkeypatch):
    _fake_miles(monkeypatch)
    from yeto.rl.algos import gdpo_reward

    sample = SimpleNamespace(response="\\boxed{4}", label="#### 5", metadata={})
    assert _run(gdpo_reward.reward_func(None, sample)) == 0.0
    assert sample.metadata[gdpo_reward.REWARD_COMPONENTS_KEY] == {"correctness": 0.0, "format": 1.0}
    assert sample.metadata["success"] is False
    sample = SimpleNamespace(response="\\boxed{5}", label="#### 5", metadata={})
    assert _run(gdpo_reward.correctness_reward(None, sample)) == 1.0
    assert sample.metadata["success"] is True


def _tb_sample(monkeypatch, reward, status="completed", verifier=None, rc=None):
    from yeto.rl import tbench_outcome

    verifier = verifier or tbench_outcome.NATIVE_VERIFIER
    meta = tbench_outcome.build_outcome_metadata(
        task_id="t1", sample_id="s1", episode_id="e1", status=status, reward=reward,
        verifier=verifier, testsh_rc=rc)
    return SimpleNamespace(metadata=meta, status=None)


