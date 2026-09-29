"""examples/math_reward.py and the RL dataset column flags it relies on."""

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_example():
    spec = importlib.util.spec_from_file_location(
        "examples_math_reward", REPO_ROOT / "examples" / "math_reward.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_miles(monkeypatch):
    """Stand-in for miles.rollout.rm_hub.math_utils: boxed extraction plus
    a string-equality grader (the real graders normalise LaTeX)."""
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


def test_math_reward_grades_the_boxed_answer_after_thinking(monkeypatch):
    _fake_miles(monkeypatch)
    example = _load_example()

    assert example.score("so \\boxed{42}", "42") == 1.0
    assert example.score("<think>\\boxed{42}</think> it is \\boxed{41}", "42") == 0.0
    assert example.score("no box", "42") == 0.0
    assert example.score("\\boxed{3}", "") == 0.0
    sample = SimpleNamespace(response="\\boxed{7}", label="7")
    assert asyncio.run(example.reward_func(None, sample)) == 1.0


def test_example_reward_is_a_valid_reward_spec_inside_the_repo():
    from yeto.launcher import REPO_ROOT as launcher_root
    from yeto.provenance import python_spec_path

    path = python_spec_path("examples/math_reward.py:reward_func", base_dir=launcher_root)
    assert path == launcher_root / "examples" / "math_reward.py"
    path.relative_to(launcher_root.resolve())  # the island-sync requirement


def test_named_columns_map_math_500_rows_and_never_guess(tmp_path, monkeypatch):
    from yeto import data as yeto_data
    from yeto.rl.learner import prepare_prompt_data

    rows = [{"problem": "What is 6*7?", "answer": "42", "level": 1, "unique_id": "t/1"}]
    monkeypatch.setattr(yeto_data, "load_rows", lambda source, revision=None: rows)
    row = json.loads(
        prepare_prompt_data(
            "unused", None, tmp_path / "p.jsonl", prompt_column="problem", label_column="answer"
        ).read_text()
    )
    assert row["messages"] == [{"role": "user", "content": "What is 6*7?"}]
    assert row["label"] == "42"
    assert row["metadata"]["unique_id"] == "t/1"
    # Without the flags the default columns apply and MATH-500 rows are rejected.
    with pytest.raises(ValueError, match="messages or a string prompt/input"):
        prepare_prompt_data("unused", None, tmp_path / "q.jsonl")
    # A named column that is missing is an error naming the flag, not a fallback.
    with pytest.raises(ValueError, match="no column 'question' \\(--rl-prompt-column\\)"):
        prepare_prompt_data("unused", None, tmp_path / "r.jsonl", prompt_column="question")
    with pytest.raises(ValueError, match="no column 'gold' \\(--rl-label-column\\)"):
        prepare_prompt_data(
            "unused", None, tmp_path / "s.jsonl", prompt_column="problem", label_column="gold"
        )


def test_default_columns_are_unchanged(tmp_path, monkeypatch):
    from yeto import data as yeto_data
    from yeto.rl.learner import prepare_prompt_data

    rows = [{"prompt": "hi", "label": "x"}, {"messages": [{"role": "user", "content": "yo"}]}]
    monkeypatch.setattr(yeto_data, "load_rows", lambda source, revision=None: rows)
    out = [json.loads(l) for l in prepare_prompt_data("unused", None, tmp_path / "p.jsonl").read_text().splitlines()]
    assert out[0]["messages"] == [{"role": "user", "content": "hi"}] and out[0]["label"] == "x"
    assert out[1]["messages"] == [{"role": "user", "content": "yo"}] and out[1]["label"] is None
