"""CPU-only contracts for the topology-comparable 128K replay views."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_miles_value_128k_ab_views",
    ROOT / "scripts" / "build_miles_value_128k_ab_views.py",
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def _sample(uid: str, reward: int, length: int) -> dict[str, object]:
    thread_id = f"thread-{uid}"
    return {
        "tokens": [10 + reward] * length,
        "loss_mask": [1] * length,
        "reward": float(reward),
        "response_length": length,
        "label": reward,
        "metadata": {
            "thread_id": thread_id,
            "compaction_epoch": 0,
            "trace_context_index": 0,
            "trace_context_count": 1,
        },
        "train_metadata": {
            "value_sample_weight": 1.0,
            "value_atomic_group_size": 1,
            "value_atomic_group_id": f"train:1:synthetic:{thread_id}",
            "preserve_me": uid,
        },
    }


def _parent_bundle(root: Path, *, buckets: int = 66) -> Path:
    island = root / "island_1"
    island.mkdir(parents=True)
    rows = []
    for rollout_id in range(buckets):
        # Bucket zero is mixed-label but ineligible, proving that selection is
        # the first 64 eligible buckets rather than simply source IDs 0..63.
        lengths = (131_073, 8) if rollout_id == 0 else (8, 9)
        samples = [
            _sample(f"{rollout_id}-{reward}", reward, lengths[reward])
            for reward in (0, 1)
        ]
        path = island / f"data_{rollout_id}.pt"
        torch.save({"rollout_id": rollout_id, "samples": samples}, path)
        file_hash = builder.sha256_file(path)
        for position, sample in enumerate(samples):
            reward = int(sample["reward"])
            length = len(sample["tokens"])
            uid = f"uid-{rollout_id}-{reward}"
            rows.append(
                {
                    "schema_version": 2,
                    "source_uid": uid,
                    "source_corpus": "synthetic",
                    "thread_id": sample["metadata"]["thread_id"],
                    "trace_context_index": 0,
                    "trace_context_count": 1,
                    "compaction_epoch": 0,
                    "reward": float(reward),
                    "token_length": length,
                    "supervised_tokens": length,
                    "attention_cost": length * length,
                    "split": "train",
                    "island_id": 1,
                    "rollout_id": rollout_id,
                    "output_path": f"island_1/data_{rollout_id}.pt",
                    "output_file_sha256": file_hash,
                    "output_position": position,
                    "destination_sample_semantic_sha256": builder.stable_sample_hash(
                        sample
                    ),
                    "value_sample_weight": 1.0,
                }
            )
    manifest = {
        "schema_version": 3,
        "dataset_version": "synthetic-contrastive-v1",
        "strategy": "atomic-thread-reward-contrastive-window-balanced-v2",
        "seed": 17,
        "sync_window_size": 12,
        "num_islands": 5,
        "max_sequence_length": 262_144,
        "max_contexts_per_rollout_file": 5,
        "train_rollout_ids": list(range(buckets)),
        "validation_rollout_ids": [buckets],
        "critic_recipe": {
            "value_loss_type": "classification",
            "value_num_bins": 51,
            "value_reward_range": [0.0, 1.0],
            "value_target_type": "hl_gauss",
            "hl_gauss_sigma_ratio": 0.75,
            "sample_weighting": "atomic-group-equal-within-step-v1",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return root


def _source_rollout_ids(root: Path, split: str) -> set[int]:
    result = set()
    with (root / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                result.add(int(row["parent_rollout_id"]))
    return result


def test_builds_exact_128k_baseline_and_two_island_diloco_views(
    tmp_path: Path,
) -> None:
    parent = _parent_bundle(tmp_path / "parent")
    output = tmp_path / "ab-views"

    result = builder.build_ab_views(parent, output, seed=23)

    assert result["verification"]["launch_ready"] is True
    assert result["selection"]["selected_source_rollout_ids"] == list(range(1, 65))
    assert result["selection"]["train_source_rollout_ids"] == list(range(1, 49))
    assert result["selection"]["heldout_source_rollout_ids"] == list(range(49, 65))

    baseline = json.loads((output / "baseline" / "manifest.json").read_text())
    diloco = json.loads((output / "diloco" / "manifest.json").read_text())
    assert baseline["schema_version"] == diloco["schema_version"] == 3
    assert baseline["max_sequence_length"] == diloco["max_sequence_length"] == 131_072
    assert baseline["num_islands"] == 1
    assert diloco["num_islands"] == 2
    assert baseline["train_rollout_ids"] == list(range(48))
    assert baseline["validation_rollout_ids"] == list(range(48, 64))
    assert diloco["train_rollout_ids"] == list(range(24))
    assert diloco["validation_rollout_ids"] == list(range(24, 32))
    assert all(island["train"]["rollouts"] == 24 for island in diloco["islands"])
    assert all(island["validation"]["rollouts"] == 8 for island in diloco["islands"])

    assert _source_rollout_ids(output / "baseline", "train") == set(range(1, 49))
    assert _source_rollout_ids(output / "diloco", "train") == set(range(1, 49))
    assert _source_rollout_ids(output / "baseline", "validation") == set(range(49, 65))
    assert _source_rollout_ids(output / "diloco", "validation") == set(range(49, 65))
    assert (
        baseline["verification"]["train_union_sha256"]
        == diloco["verification"]["train_union_sha256"]
    )
    assert (
        baseline["verification"]["heldout_union_sha256"]
        == diloco["verification"]["heldout_union_sha256"]
    )

    # The sample and nested metadata are unchanged; only the payload rollout ID
    # is renumbered to match the destination file.
    first = torch.load(
        output / "baseline" / "island_0" / "data_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    for sample in first["samples"]:
        uid = sample["train_metadata"]["preserve_me"]
        source_rollout = int(uid.split("-", 1)[0])
        source = torch.load(
            parent / "island_1" / f"data_{source_rollout}.pt",
            map_location="cpu",
            weights_only=False,
        )
        source_sample = next(
            item
            for item in source["samples"]
            if item["train_metadata"]["preserve_me"] == uid
        )
        assert sample == source_sample

    for view in ("baseline", "diloco"):
        lines = (output / view / "ARTIFACTS.sha256").read_text().splitlines()
        assert lines
        assert all("  " in line for line in lines)


def test_corrupt_selected_parent_file_fails_without_publishing(tmp_path: Path) -> None:
    parent = _parent_bundle(tmp_path / "parent")
    (parent / "island_1" / "data_1.pt").write_bytes(b"corrupt")
    output = tmp_path / "ab-views"

    with pytest.raises(ValueError, match="source file hash mismatch"):
        builder.build_ab_views(parent, output)

    assert not output.exists()
