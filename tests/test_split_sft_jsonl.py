import pytest

from scripts.split_sft_jsonl import split_rows


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
