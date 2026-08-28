"""CPU-only contracts for the reward-stratified five-island replay plan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reorder_miles_value_five_islands",
    ROOT / "scripts" / "reorder_miles_value_five_islands.py",
)
assert SPEC is not None and SPEC.loader is not None
reorder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reorder
SPEC.loader.exec_module(reorder)


def _context(
    uid: str,
    reward: int,
    *,
    thread: str,
    epoch: int = 0,
    count: int = 1,
    length: int = 100,
) -> object:
    return reorder.Context(
        {
            "source_uid": uid,
            "reward": float(reward),
            "token_length": length,
            "supervised_tokens": length // 2,
            "attention_cost": length * length,
            "output_path": "island_0/data_0.pt",
            "output_position": 0,
            "split": "train",
            "island_id": 0,
            "source_corpus": "synthetic",
            "thread_id": thread,
            "trace_context_index": epoch,
            "compaction_epoch": epoch,
            "trace_context_count": count,
        }
    )


def test_pack_is_atomic_deterministic_and_reward_balanced_per_h_window() -> None:
    contexts = [
        _context("compact-fail-0", 0, thread="compact-fail", epoch=0, count=2),
        _context("compact-fail-1", 0, thread="compact-fail", epoch=1, count=2),
        _context("compact-pass-0", 1, thread="compact-pass", epoch=0, count=2),
        _context("compact-pass-1", 1, thread="compact-pass", epoch=1, count=2),
    ]
    contexts.extend(
        _context(
            f"single-{reward}-{index}",
            reward,
            thread=f"single-{reward}-{index}",
            length=101 + index,
        )
        for reward in (0, 1)
        for index in range(6)
    )
    groups = reorder._groups(contexts)

    first = reorder.pack_groups(groups, num_buckets=6, window_size=3, seed=17)
    second = reorder.pack_groups(groups, num_buckets=6, window_size=3, seed=17)

    assert [[bucket.stable_id for bucket in first]] == [
        [bucket.stable_id for bucket in second]
    ]
    assert len(first) == 6
    assert sum(bucket.size for bucket in first) == len(contexts)
    assert max(bucket.size for bucket in first) <= 4
    assert len({context.uid for bucket in first for context in bucket.contexts}) == len(
        contexts
    )
    for thread in ("compact-fail", "compact-pass"):
        destinations = [
            index
            for index, bucket in enumerate(first)
            if any(group.key[-1] == thread for group in bucket.groups)
        ]
        assert len(destinations) == 1
        assert len(destinations) == 1

    assert all({group.reward for group in bucket.groups} == {0, 1} for bucket in first)
    for bucket in first:
        weights = [
            bucket.sample_weight(group)
            for group in bucket.groups
            for _ in group.contexts
        ]
        assert sum(weights) == pytest.approx(bucket.size)
        group_totals = [
            bucket.sample_weight(group) * group.size for group in bucket.groups
        ]
        assert max(group_totals) == pytest.approx(min(group_totals))

    metrics = reorder._window_metrics(first, 3)
    assert metrics["max_step_positive_rate_deviation"] == pytest.approx(0.0)
    for window in metrics["windows"]:
        selected = first[
            window["start_rollout_id"] : window["end_rollout_id_exclusive"]
        ]
        assert {
            context.reward for bucket in selected for context in bucket.contexts
        } == {
            0,
            1,
        }


def test_pack_refuses_an_atomic_group_larger_than_one_rollout() -> None:
    contexts = [
        _context(f"oversized-{index}", 0, thread="oversized", epoch=index, count=6)
        for index in range(6)
    ]

    with pytest.raises(ValueError, match="exceeding the bucket limit"):
        reorder._groups(contexts)
