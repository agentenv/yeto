#!/usr/bin/env python3
"""Rebucket opaque Miles value samples into six non-empty step-aligned islands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch

from split_miles_value_islands import _sample_counts


NUM_ISLANDS = 6
TRAIN_IDS = tuple(range(364))
VALIDATION_IDS = tuple(range(364, 395))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(source: Path, rollout_ids: tuple[int, ...]):
    records = []
    hashes = {}
    for rollout_id in rollout_ids:
        path = source / f"data_{rollout_id}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if set(payload) != {"rollout_id", "samples"} or payload["rollout_id"] != rollout_id:
            raise ValueError(f"unexpected payload schema in {path}")
        if not isinstance(payload["samples"], list) or not payload["samples"]:
            raise ValueError(f"empty or invalid samples in {path}")
        hashes[path.name] = sha256(path)
        for position, sample in enumerate(payload["samples"]):
            _sample_counts(sample, rollout_id=rollout_id, position=position)
            records.append((rollout_id, position, sample))
    return records, hashes


def split_records(records, rollout_ids: tuple[int, ...]):
    per_island = [records[island_id::NUM_ISLANDS] for island_id in range(NUM_ISLANDS)]
    if sum(map(len, per_island)) != len(records):
        raise AssertionError("island sample counts do not reconstruct source")
    result = []
    for island_id, island_records in enumerate(per_island):
        if len(island_records) < len(rollout_ids):
            raise ValueError(
                f"island {island_id} has {len(island_records)} samples for "
                f"{len(rollout_ids)} required non-empty steps"
            )
        buckets = [island_records[index::len(rollout_ids)] for index in range(len(rollout_ids))]
        if any(not bucket for bucket in buckets):
            raise AssertionError(f"island {island_id} contains an empty bucket")
        result.append(buckets)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    train, train_hashes = load_split(source, TRAIN_IDS)
    validation, validation_hashes = load_split(source, VALIDATION_IDS)
    train_buckets = split_records(train, TRAIN_IDS)
    validation_buckets = split_records(validation, VALIDATION_IDS)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        mappings = []
        island_counts = []
        for island_id in range(NUM_ISLANDS):
            island_dir = staging / f"island_{island_id}"
            island_dir.mkdir()
            split_counts = {}
            for split_name, rollout_ids, buckets in (
                ("train", TRAIN_IDS, train_buckets[island_id]),
                ("validation", VALIDATION_IDS, validation_buckets[island_id]),
            ):
                split_counts[split_name] = sum(len(bucket) for bucket in buckets)
                for rollout_id, bucket in zip(rollout_ids, buckets, strict=True):
                    samples = [record[2] for record in bucket]
                    torch.save({"rollout_id": rollout_id, "samples": samples}, island_dir / f"data_{rollout_id}.pt")
                    mappings.append({
                        "island_id": island_id,
                        "rollout_id": rollout_id,
                        "split": split_name,
                        "sources": [[record[0], record[1]] for record in bucket],
                    })
            island_counts.append({
                "island_id": island_id,
                "train_samples": split_counts["train"],
                "validation_samples": split_counts["validation"],
                "nominal_global_batch_size": 5 if island_id == 0 else 4,
            })

        manifest = {
            "schema_version": 1,
            "strategy": "opaque_global_sample_stride_6_then_step_stride",
            "num_islands": NUM_ISLANDS,
            "train_rollout_ids": list(TRAIN_IDS),
            "validation_rollout_ids": list(VALIDATION_IDS),
            "source_train_samples": len(train),
            "source_validation_samples": len(validation),
            "source_hashes": {**train_hashes, **validation_hashes},
            "islands": island_counts,
            "mapping": mappings,
        }
        if sum(item["train_samples"] for item in island_counts) != len(train):
            raise AssertionError("train samples lost or duplicated")
        if sum(item["validation_samples"] for item in island_counts) != len(validation):
            raise AssertionError("validation samples lost or duplicated")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({"output": str(output), "islands": island_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
