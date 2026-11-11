"""--data REPO:CONFIG selects the config of a multi-config Hub dataset."""

import sys
import types

import pytest

from yeto import data, provenance

PIN = "740312add88f781978c0658806c59bc2815b9866"


@pytest.mark.parametrize("name,expected", [
    ("openai/gsm8k:main", ("openai/gsm8k", "main")),
    ("openai/gsm8k", ("openai/gsm8k", None)),
    ("s3://bucket/a:b", ("s3://bucket/a:b", None)),
    ("repo:", ("repo:", None)),
    (":cfg", (":cfg", None)),
    ("C:/data/x.jsonl", ("C:/data/x.jsonl", None)),
])
def test_split_hub_config(name, expected):
    assert data.split_hub_config(name) == expected


def test_load_rows_passes_config_name(monkeypatch):
    calls = []
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda path, split=None, revision=None, name=None: calls.append(
        (path, split, revision, name)) or ["row"]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    assert data.load_rows("openai/gsm8k:main", split="train", revision=PIN) == ["row"]
    assert data.load_rows("openai/gsm8k", split="train", revision=PIN) == ["row"]
    assert calls == [("openai/gsm8k", "train", PIN, "main"), ("openai/gsm8k", "train", PIN, None)]


def test_prompt_split_uses_config(monkeypatch):
    seen = []
    fake = types.ModuleType("datasets")

    def names(path, config=None, revision=None):
        seen.append((path, config))
        return ["test"]

    fake.get_dataset_split_names = names
    monkeypatch.setitem(sys.modules, "datasets", fake)
    assert data.prompt_split("org/only-test:cfg", PIN) == "test"
    assert seen == [("org/only-test", "cfg")]


def test_moving_revision_resolves_the_repo_not_the_config():
    class Api:
        def dataset_info(self, repo_id, revision=None):
            assert repo_id == "openai/gsm8k"
            return types.SimpleNamespace(sha=PIN)

    assert provenance.resolve_hub_revision("openai/gsm8k:main", "main", repo_type="dataset",
                                           api=Api()) == ("main", PIN)
