"""The verl island reads ``--data`` in the forms the Miles island reads.

Before, ``prepare_data`` only downloaded ``train.parquet`` / ``test.parquet`` from the
root of a Hub dataset with ``messages`` / ``label`` columns: ``--data openai/gsm8k:main
--rl-prompt-column question --rl-label-column answer`` (accepted by the launcher, as
on Miles) crashed on the island in ``hf_hub_download`` with ``HFValidationError``,
after the cards had started, and the island dropped the column flags. Now a
``repo:config`` id, a dataset without root parquet files and a local directory are
read as the Miles island reads them (``yeto.data.load_rows``), and the prompt / label
columns follow the rules both backends share (``yeto.rl.prompt_rows``). yeto's
prepared datasets keep their root parquet files. CPU only (the Hub is faked).
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl import prompt_rows
from yeto.rl.adapters.verl import island_entry

GSM8K = [{"question": f"q{i}", "answer": f"a{i} #### {i}"} for i in range(5)]


def _rows(path) -> list[dict]:
    return pd.read_parquet(path).to_dict("records")


def _prompts(path) -> list[tuple[str, str]]:
    return [(list(row["prompt"])[0]["content"], row["reward_model"]["ground_truth"]) for row in _rows(path)]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    """A fake Hub: ``files`` maps ``(repo, filename)`` to a local parquet file
    (others raise EntryNotFoundError); ``datasets`` maps an id to its rows."""
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    import yeto.data

    state = {"files": {}, "datasets": {}, "downloads": [], "loads": []}

    def download(repo, filename, *, repo_type, revision=None):
        assert repo_type == "dataset" and ":" not in repo, repo  # a repo:config id never reaches the Hub API
        state["downloads"].append((repo, filename))
        if (repo, filename) not in state["files"]:
            raise EntryNotFoundError(f"{filename} not in {repo}")
        return str(state["files"][repo, filename])

    def load_rows(name, split="train", revision=None):
        state["loads"].append((name, split, revision))
        return state["datasets"][name]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(yeto.data, "load_rows", load_rows)
    monkeypatch.setattr(yeto.data, "prompt_split", lambda source, revision: "train")
    return state


def test_a_repo_config_id_with_named_columns_is_read_as_on_miles(hub, tmp_path):
    hub["datasets"]["openai/gsm8k:main"] = GSM8K
    train, test = island_entry.prepare_data("openai/gsm8k:main", "c" * 40, tmp_path / "out", val_rows=2,
                                            prompt_column="question", label_column="answer")
    assert hub["downloads"] == [] and hub["loads"] == [("openai/gsm8k:main", "train", "c" * 40)]
    assert _prompts(train) == [(row["question"], row["answer"]) for row in GSM8K]
    assert _prompts(test) == _prompts(train)[:2]  # the head of the train rows, as for a local file
    assert {row["data_source"] for row in _rows(train)} == {"yeto/openai/gsm8k:main"}
    assert list(_rows(train)[0]["prompt"])[0]["role"] == "user"


def test_a_dataset_without_root_parquet_files_is_read_as_on_miles(hub, tmp_path):
    hub["datasets"]["org/plain"] = [{"prompt": "p0", "label": "7"}, {"prompt": "p1", "label": "8"}]
    train, _ = island_entry.prepare_data("org/plain", None, tmp_path / "out")
    assert hub["downloads"] == [("org/plain", "train.parquet")]
    assert _prompts(train) == [("p0", "7"), ("p1", "8")]


def test_a_prepared_dataset_keeps_its_root_parquet_files(hub, tmp_path):
    for split, n in (("train", 3), ("test", 2)):
        path = tmp_path / f"src-{split}.parquet"
        pd.DataFrame([{"messages": [{"role": "user", "content": f"{split}{i}"}], "label": str(i)}
                      for i in range(n)]).to_parquet(path)
        hub["files"]["org/prepared", f"{split}.parquet"] = path
    train, test = island_entry.prepare_data("org/prepared", None, tmp_path / "out")
    assert hub["loads"] == []
    assert _prompts(train) == [(f"train{i}", str(i)) for i in range(3)]
    assert _prompts(test) == [(f"test{i}", str(i)) for i in range(2)]
    # named columns apply to those files too
    renamed = tmp_path / "renamed.parquet"
    pd.DataFrame([{"problem": "x", "solution": "y"}]).to_parquet(renamed)
    hub["files"]["org/renamed", "train.parquet"] = renamed
    train, test = island_entry.prepare_data("org/renamed", None, tmp_path / "out2", prompt_column="problem",
                                            label_column="solution")
    assert _prompts(train) == _prompts(test) == [("x", "y")]


def test_a_local_task_file_takes_the_named_columns(tmp_path):
    data = tmp_path / "tasks.jsonl"
    data.write_text("".join(json.dumps({"problem": f"p{i}", "answer": "34"}) + "\n" for i in range(3)))
    train, _ = island_entry.prepare_data(str(data), None, tmp_path / "out", prompt_column="problem",
                                         label_column="answer")
    assert _prompts(train) == [(f"p{i}", "34") for i in range(3)]


def test_a_missing_named_column_names_the_flag(hub, tmp_path):
    hub["datasets"]["openai/gsm8k:main"] = GSM8K
    with pytest.raises(ValueError, match=r"no column 'problem' \(--rl-prompt-column\)"):
        island_entry.prepare_data("openai/gsm8k:main", None, tmp_path / "out", prompt_column="problem",
                                  label_column="answer")
    with pytest.raises(ValueError, match="string prompt/input"):  # gsm8k has no messages/prompt/input
        island_entry.prepare_data("openai/gsm8k:main", None, tmp_path / "out", label_column="answer")
    with pytest.raises(ValueError, match="--rl-label-column"):
        island_entry.prepare_data("openai/gsm8k:main", None, tmp_path / "out", prompt_column="question")


def test_the_launcher_forwards_the_columns_and_the_island_parses_them():
    from test_verl_decoupled import _island_argv, _prepared

    argv = _island_argv(_prepared("--rl-prompt-column", "question", "--rl-label-column", "answer"), 0,
                        "127.0.0.1:9")
    parsed, ignored = island_entry.parse_args(argv)
    assert (parsed.rl_prompt_column, parsed.rl_label_column) == ("question", "answer")
    assert not {"--rl-prompt-column", "--rl-label-column"} & set(ignored)


def test_both_backends_share_the_prompt_rules():
    from yeto.rl.adapters.miles import island_entry as miles_island

    assert miles_island._messages is prompt_rows.messages and miles_island._column is prompt_rows.column
