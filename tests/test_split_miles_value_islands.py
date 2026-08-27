"""CPU-only tests for the immutable Miles five-island data splitter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "split_miles_value_islands",
    ROOT / "scripts" / "split_miles_value_islands.py",
)
assert SPEC is not None and SPEC.loader is not None
splitter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = splitter
SPEC.loader.exec_module(splitter)


def _sample(position: int, *, response_length: int = 4) -> dict:
    prompt_length = 3
    mask_patterns = (
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    )
    loss_mask = mask_patterns[position % len(mask_patterns)][:response_length]
    token_count = prompt_length + response_length
    # Deliberately unrelated to source-list ordinal.  The splitter must not
    # assign islands using this identity field.
    sample_index = 10_003 - position * 37
    return {
        "group_index": sample_index,
        "index": sample_index,
        "prompt": [{"role": "user", "content": f"opaque prompt {position}"}],
        "tokens": [1000 + position * 10 + offset for offset in range(token_count)],
        "multimodal_inputs": None,
        "multimodal_train_inputs": None,
        "response": f"opaque response {position}",
        "response_length": response_length,
        "label": f"label-{position % 2}",
        "reward": float(position % 2),
        "loss_mask": loss_mask,
        "weight_versions": [],
        "rollout_log_probs": None,
        "rollout_routed_experts": None,
        "rollout_indexer_topk": None,
        "remove_sample": False,
        "teacher_log_probs": None,
        "opd_reverse_kl": None,
        "status": "completed",
        "metadata": {
            "compaction_epoch": position % 3,
            "compaction_epoch_id": f"epoch-{position % 3}",
            "trace_context_index": position,
            "trace_context_count": 10,
            "native_compaction_replacement_history": position > 0,
            "response_continues": position % 2 == 0,
            "response_continuation": position % 2 == 1,
            "eos_semantics": {
                "intermediate_supervised_eos": 0,
                "final_supervised_eos": 1,
            },
            "prompt_token_length": prompt_length,
            "token_length": token_count,
        },
        "generate_function_path": None,
        "train_metadata": None,
        "session_id": None,
        "non_generation_time": 0.0,
        "spec_info": {
            "spec_accept_token_num": 0,
            "spec_draft_token_num": 0,
            "spec_verify_ct": 0,
            "completion_token_num": 0,
        },
        "prefix_cache_info": {"cached_tokens": 0, "total_prompt_tokens": 0},
        # Future/critic fields must remain opaque rather than being rebuilt.
        "returns": [position + 0.25],
        "custom_eos_boundaries": ["commentary", "tool_call", "final"],
    }


def _write_bucket(source: Path, rollout_id: int, size: int = 5) -> dict:
    payload = {
        "rollout_id": rollout_id,
        "samples": [_sample(position + rollout_id * 100) for position in range(size)],
        "top_level_metadata": {
            "producer": "synthetic-miles",
            "continuation_contract": ["response_continues", "response_continuation"],
        },
        # Its length can equal the sample count; it is still top-level opaque
        # metadata and must not be mistaken for a parallel sample field.
        "top_level_opaque_list": [f"metadata-{i}" for i in range(size)],
    }
    torch.save(payload, source / f"data_{rollout_id}.pt")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(source: Path, output: Path, *, overwrite: bool = False) -> dict:
    return splitter.split_dataset(
        source,
        output,
        train_rollout_ids=(0,),
        validation_rollout_ids=(1,),
        overwrite=overwrite,
    )


def _counts(samples: list[dict]) -> tuple[int, int, int]:
    return (
        len(samples),
        sum(sum(sample["loss_mask"]) for sample in samples),
        sum(len(sample["tokens"]) for sample in samples),
    )


def test_split_preserves_opaque_samples_metadata_and_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    original = {
        0: _write_bucket(source, 0, size=10),
        1: _write_bucket(source, 1, size=5),
    }
    hashes = {rollout_id: _sha256(source / f"data_{rollout_id}.pt") for rollout_id in (0, 1)}

    manifest = _run(source, output)

    assert manifest == json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_islands"] == 5
    assert manifest["nominal_global_batch_size_per_island"] == 5
    assert manifest["dynamic_global_batch_size_required"] is True
    assert manifest["strategy"] == "source_sample_position_modulo_5"
    assert manifest["source_hashes"] == {
        "data_0.pt": hashes[0],
        "data_1.pt": hashes[1],
    }
    assert manifest["source_counts"]["trajectory_counts"] == {
        "train": 10,
        "validation": 5,
        "total": 15,
    }

    for island_id in range(5):
        island_manifest = manifest["islands"][island_id]
        assert island_manifest["island_id"] == island_id
        assert island_manifest["nominal_global_batch_size"] == 5
        assert island_manifest["trajectory_counts"] == {
            "train": 2,
            "validation": 1,
            "total": 3,
        }
        for rollout_id, positions in ((0, [island_id, island_id + 5]), (1, [island_id])):
            saved = torch.load(
                output / f"island_{island_id}" / f"data_{rollout_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            expected = original[rollout_id]
            assert saved["rollout_id"] == rollout_id
            assert saved["top_level_metadata"] == expected["top_level_metadata"]
            assert saved["top_level_opaque_list"] == expected["top_level_opaque_list"]
            assert saved["samples"] == [expected["samples"][position] for position in positions]
            # Explicitly cover compaction and assistant-response/EOS metadata.
            for actual, source_position in zip(saved["samples"], positions, strict=True):
                assert actual["metadata"] == expected["samples"][source_position]["metadata"]
                assert actual["tokens"] == expected["samples"][source_position]["tokens"]
                assert actual["loss_mask"] == expected["samples"][source_position]["loss_mask"]
                assert actual["returns"] == expected["samples"][source_position]["returns"]

    for rollout_id, expected_split in ((0, "train"), (1, "validation")):
        mapping = manifest["rollout_mapping"][rollout_id]
        assert mapping["rollout_id"] == rollout_id
        assert mapping["split"] == expected_split
        assert mapping["source_sha256"] == hashes[rollout_id]
        all_positions = sorted(
            position
            for island in mapping["islands"]
            for position in island["source_sample_positions"]
        )
        assert all_positions == list(range(len(original[rollout_id]["samples"])))
        for island in mapping["islands"]:
            positions = island["source_sample_positions"]
            selected = [original[rollout_id]["samples"][position] for position in positions]
            trajectories, supervised, total = _counts(selected)
            assert island["trajectories"] == trajectories
            assert island["supervised_tokens"] == supervised
            assert island["total_tokens"] == total


def test_split_is_deterministic_and_uses_list_position_not_sample_index(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = _write_bucket(source, 0, size=10)
    _write_bucket(source, 1, size=5)

    first = _run(source, tmp_path / "first")
    second = _run(source, tmp_path / "second")

    assert first == second
    island_zero = torch.load(
        tmp_path / "first" / "island_0" / "data_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert island_zero["samples"] == [train["samples"][0], train["samples"][5]]
    assert [sample["index"] % 5 for sample in island_zero["samples"]] != [0, 0]


@pytest.mark.parametrize(
    "break_sample, expected",
    [
        (lambda sample: sample.pop("loss_mask"), "missing loss_mask"),
        (lambda sample: sample.__setitem__("loss_mask", [1, 0]), "loss_mask length"),
        (
            lambda sample: sample.__setitem__("rollout_log_probs", [0.1, 0.2]),
            "rollout_log_probs length",
        ),
        (
            lambda sample: sample["metadata"].__setitem__("token_length", 999),
            "metadata.token_length",
        ),
        (lambda sample: sample.__setitem__("remove_sample", True), "remove_sample is true"),
    ],
)
def test_invalid_sample_alignment_fails_without_publishing(
    tmp_path: Path, break_sample, expected: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    train = _write_bucket(source, 0)
    _write_bucket(source, 1)
    break_sample(train["samples"][2])
    torch.save(train, source / "data_0.pt")
    output = tmp_path / "output"

    with pytest.raises(splitter.SplitValidationError, match=expected):
        _run(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".output.building-*"))


def test_non_dp5_bucket_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_bucket(source, 0, size=6)
    _write_bucket(source, 1)

    with pytest.raises(splitter.SplitValidationError, match="must be divisible by 5"):
        _run(source, tmp_path / "output")


def test_missing_source_is_reported_before_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_bucket(source, 0)

    with pytest.raises(splitter.SplitValidationError, match="missing source rollout files"):
        _run(source, tmp_path / "output")

    assert not (tmp_path / "output").exists()


def test_overwrite_is_explicit_and_replaces_only_after_success(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    _write_bucket(source, 0)
    _write_bucket(source, 1)
    _run(source, output)
    sentinel = output / "old-sentinel"
    sentinel.write_text("old tree", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(source, output)
    assert sentinel.read_text(encoding="utf-8") == "old tree"

    _run(source, output, overwrite=True)
    assert not sentinel.exists()
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".output.building-*"))
    assert not list(tmp_path.glob(".output.replaced-*"))


def test_failed_overwrite_keeps_existing_tree(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    train = _write_bucket(source, 0)
    _write_bucket(source, 1)
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("must survive", encoding="utf-8")
    train["samples"][0].pop("loss_mask")
    torch.save(train, source / "data_0.pt")

    with pytest.raises(splitter.SplitValidationError, match="missing loss_mask"):
        _run(source, output, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "must survive"
    assert not list(tmp_path.glob(".output.building-*"))
