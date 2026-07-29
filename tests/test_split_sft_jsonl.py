import pytest

from scripts.split_sft_jsonl import (
    split_grouped_rows,
    split_grouped_rows_near_target,
    split_rows,
)


def test_split_rows_is_deterministic_and_preserves_source_order():
    rows = [{"id": idx} for idx in range(10)]

    train, evaluation = split_rows(rows, eval_rows=3, seed=27)

    assert train == [{"id": idx} for idx in [0, 1, 2, 4, 5, 7, 8]]
    assert evaluation == [{"id": idx} for idx in [3, 6, 9]]
    assert split_rows(rows, eval_rows=3, seed=27) == (train, evaluation)


@pytest.mark.parametrize("eval_rows", [0, 3])
def test_split_rows_rejects_empty_train_or_eval(eval_rows):
    with pytest.raises(ValueError):
        split_rows([{"id": 0}, {"id": 1}, {"id": 2}], eval_rows, seed=27)


def test_split_grouped_rows_keeps_sessions_disjoint_and_source_ordered():
    rows = [
        {"id": idx, "metadata": {"session_id": session}}
        for idx, session in enumerate(["a", "a", "b", "c", "c", "d"])
    ]

    train, evaluation = split_grouped_rows(
        rows, "metadata.session_id", eval_groups=2, seed=27
    )

    train_sessions = {row["metadata"]["session_id"] for row in train}
    eval_sessions = {row["metadata"]["session_id"] for row in evaluation}
    assert train_sessions.isdisjoint(eval_sessions)
    assert [row["id"] for row in train] == sorted(row["id"] for row in train)
    assert [row["id"] for row in evaluation] == sorted(row["id"] for row in evaluation)


def test_split_grouped_rows_requires_group_key():
    with pytest.raises(ValueError, match="missing group key"):
        split_grouped_rows([{"id": 1}, {"id": 2}], "metadata.session_id", 1, 27)


def test_grouped_target_split_approximates_rows_without_session_leakage():
    rows = [
        {"id": f"{session}-{idx}", "metadata": {"session_id": session}}
        for session, count in [("large", 7), ("medium", 4), ("small", 2), ("tiny", 1)]
        for idx in range(count)
    ]

    train, evaluation = split_grouped_rows_near_target(
        rows, "metadata.session_id", eval_rows=6, seed=42
    )

    train_sessions = {row["metadata"]["session_id"] for row in train}
    eval_sessions = {row["metadata"]["session_id"] for row in evaluation}
    assert train_sessions.isdisjoint(eval_sessions)
    assert len(evaluation) == 6
