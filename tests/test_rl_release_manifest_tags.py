"""Release manifest of an island: the kimi_k31 fields for every island (also the
single-island no-sync trainer) and the --rl-release-tag labels."""

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import island_entry


def test_tags_parse_and_refuse_bad_values():
    assert island_entry.release_tags(["reward=synthetic", "note=a=b"]) == {"reward": "synthetic", "note": "a=b"}
    assert island_entry.release_tags(None) == {}
    with pytest.raises(ValueError, match="KEY=VALUE"):
        island_entry.release_tags(["synthetic"])
    with pytest.raises(ValueError, match="twice"):
        island_entry.release_tags(["a=1", "a=2"])


def test_manifest_without_recipe_fields_is_the_tags_or_none():
    args = SimpleNamespace(rl_model_recipe="generic", rl_release_tag=None)
    assert island_entry.release_manifest(args, provider=None) is None
    args.rl_release_tag = ["reward=synthetic"]
    assert island_entry.release_manifest(args, provider=None) == {"tags": {"reward": "synthetic"}}


def test_kimi_k31_fields_and_tags_merge(monkeypatch):
    fields = {"expert_selection_sha256": "ab" * 32, "partial_experts": {"num_trainable": 128}}
    monkeypatch.setattr(island_entry, "_kimi_k31_release_manifest", lambda args, provider: dict(fields))
    args = SimpleNamespace(rl_model_recipe="kimi_k31", rl_release_tag=["reward=synthetic"])
    assert island_entry.release_manifest(args, provider=None) == {**fields, "tags": {"reward": "synthetic"}}


def test_the_parser_takes_repeated_tags():
    import argparse

    (action,) = [a for a in island_entry.build_parser()._actions if a.dest == "rl_release_tag"]
    assert isinstance(action, argparse._AppendAction)
