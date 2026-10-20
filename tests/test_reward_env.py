"""CPU tests for the benchmark-neutral reward-env helpers (rl-agentic-reward-env).

The TB2 adapter and its prebake / judge tests left with the self-written Codex
harness (S26 E0.5).
"""

from __future__ import annotations

import pytest

from yeto.rl.harness.reward_env import benchmark as bm


def test_unknown_benchmark_is_refused():
    with pytest.raises(KeyError, match="unknown benchmark"):
        bm.get_adapter("no_such_bench")


def test_rows_split_and_leak_check():
    rows = [{"metadata": {"task_id": "a"}}, {"metadata": {"task_id": "c"}}, {"task_id": "b"}]
    train, ev = bm.split_rows(rows, ["a", "b"])
    assert [bm.task_id_of(r) for r in train] == ["c"] and len(ev) == 2
    with pytest.raises(ValueError):
        bm.split_rows([{"prompt": "x"}], ["a"])
    bm.assert_disjoint(["c"], ["a"])
    with pytest.raises(ValueError, match="leak"):
        bm.assert_disjoint(["a"], ["a"])
    with pytest.raises(ValueError, match="duplicate"):
        bm.holdout_ids({"schema": bm.HOLDOUT_SCHEMA, "items": [{"task_id": "a"}, {"task_id": "a"}]})
