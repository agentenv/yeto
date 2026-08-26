#!/usr/bin/env python3
"""Build deterministic, attention-balanced Miles value data for six islands.

The source files remain the immutable Miles offline replay buckets. Samples
are never decoded, concatenated, packed, padded, or otherwise rewritten: each
source sample is selected as one independent element of an output ``samples``
list. Only its island and output rollout bucket change.

Train and validation are balanced independently with fixed-cardinality LPT
(longest processing time first), using the true ``len(sample["tokens"]) ** 2``
attention proxy. Within an island, long samples are spread across rollout
buckets before shorter samples are paired with them. The manifest records
enough source coordinates and sufficient statistics to prove that every
source sample was assigned exactly once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from split_miles_value_islands import SplitValidationError, _sample_counts


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "qwen38_value_six_islands_v7.json"
SOURCE_FILE_TEMPLATE = "data_{rollout_id}.pt"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class Record:
    source_rollout_id: int
    source_position: int
    sample: Any
    token_length: int

    @property
    def source_key(self) -> tuple[int, int]:
        return (self.source_rollout_id, self.source_position)

    @property
    def attention_cost(self) -> int:
        return self.token_length * self.token_length


@dataclass(frozen=True)
class SplitStats:
    sample_count: int
    token_sum: int
    attention_cost_sum: int
    min_token_length: int
    max_token_length: int

    def as_dict(self) -> dict[str, int]:
        return {
            "sample_count": self.sample_count,
            "token_sum": self.token_sum,
            "attention_cost_sum": self.attention_cost_sum,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(config: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"config {key} must be an integer >= {minimum}, got {value!r}")
    return value


def _rollout_range(config: dict[str, Any], key: str) -> tuple[int, ...]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config {key} must be an object")
    start = value.get("start")
    end = value.get("end_exclusive")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise ValueError(f"config {key} must define 0 <= start < end_exclusive")
    return tuple(range(start, end))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    if config.get("dataset_version") != "qwen38-value-six-islands-fresh-v7":
        raise ValueError("config dataset_version must be qwen38-value-six-islands-fresh-v7")
    if config.get("balancing_cost") != "len(tokens)^2":
        raise ValueError("config balancing_cost must be len(tokens)^2")
    if config.get("sample_storage") != "independent-unpacked-records":
        raise ValueError("config sample_storage must be independent-unpacked-records")

    num_islands = _integer(config, "num_islands")
    max_sequence_length = _integer(config, "max_sequence_length")
    long_sample_threshold = _integer(config, "long_sample_threshold")
    if long_sample_threshold > max_sequence_length:
        raise ValueError("config long_sample_threshold exceeds max_sequence_length")

    train_ids = _rollout_range(config, "train_rollout_ids")
    validation_ids = _rollout_range(config, "validation_rollout_ids")
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise ValueError(f"train and validation rollout IDs overlap: {overlap}")
    if len(train_ids) != _integer(config, "expected_train_rollouts"):
        raise ValueError("expected_train_rollouts does not match train_rollout_ids")
    if len(validation_ids) != _integer(config, "expected_validation_rollouts"):
        raise ValueError("expected_validation_rollouts does not match validation_rollout_ids")
    expected_train_samples = _integer(config, "expected_train_samples")
    expected_validation_samples = _integer(config, "expected_validation_samples")
    if expected_train_samples % num_islands:
        raise ValueError("expected_train_samples must divide evenly across islands")
    if expected_train_samples // num_islands < len(train_ids):
        raise ValueError("each island needs at least one train sample per rollout")
    if expected_validation_samples // num_islands < len(validation_ids):
        raise ValueError("each island needs at least one validation sample per rollout")
    return config


def load_split(
    source: Path,
    rollout_ids: Sequence[int],
    *,
    max_sequence_length: int,
) -> tuple[list[Record], dict[str, str]]:
    records: list[Record] = []
    hashes: dict[str, str] = {}
    for rollout_id in rollout_ids:
        path = source / SOURCE_FILE_TEMPLATE.format(rollout_id=rollout_id)
        if not path.is_file():
            raise FileNotFoundError(f"missing source rollout {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise SplitValidationError(f"{path} must contain a dictionary payload")
        if payload.get("rollout_id") != rollout_id:
            raise SplitValidationError(
                f"{path} rollout_id={payload.get('rollout_id')!r}, expected {rollout_id}"
            )
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise SplitValidationError(f"empty or invalid samples in {path}")
        hashes[path.name] = sha256(path)
        for position, sample in enumerate(samples):
            counts = _sample_counts(sample, rollout_id=rollout_id, position=position)
            token_length = counts.total_tokens
            if token_length <= 0:
                raise SplitValidationError(
                    f"rollout {rollout_id} sample position {position} has no tokens"
                )
            if token_length > max_sequence_length:
                raise SplitValidationError(
                    f"rollout {rollout_id} sample position {position} has {token_length} "
                    f"tokens, exceeding {max_sequence_length}"
                )
            records.append(Record(rollout_id, position, sample, token_length))
    return records, hashes


def _capacities(total: int, bins: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, bins)
    return tuple(quotient + int(index < remainder) for index in range(bins))


def _record_order(record: Record) -> tuple[int, int, int, int]:
    return (
        -record.attention_cost,
        -record.token_length,
        record.source_rollout_id,
        record.source_position,
    )


def split_stats(records: Iterable[Record]) -> SplitStats:
    records = tuple(records)
    if not records:
        return SplitStats(0, 0, 0, 0, 0)
    lengths = [record.token_length for record in records]
    return SplitStats(
        sample_count=len(records),
        token_sum=sum(lengths),
        attention_cost_sum=sum(length * length for length in lengths),
        min_token_length=min(lengths),
        max_token_length=max(lengths),
    )


def balance_records(records: Sequence[Record], *, num_islands: int) -> list[list[Record]]:
    """Fixed-cardinality deterministic LPT by true attention cost."""
    if num_islands <= 0:
        raise ValueError("num_islands must be positive")
    capacities = _capacities(len(records), num_islands)
    per_island: list[list[Record]] = [[] for _ in range(num_islands)]
    cost_sums = [0] * num_islands
    token_sums = [0] * num_islands

    for record in sorted(records, key=_record_order):
        candidates = [
            island_id
            for island_id in range(num_islands)
            if len(per_island[island_id]) < capacities[island_id]
        ]
        if not candidates:
            raise AssertionError("no island capacity remains")
        island_id = min(
            candidates,
            key=lambda index: (
                cost_sums[index],
                token_sums[index],
                len(per_island[index]),
                index,
            ),
        )
        per_island[island_id].append(record)
        cost_sums[island_id] += record.attention_cost
        token_sums[island_id] += record.token_length

    if tuple(map(len, per_island)) != capacities:
        raise AssertionError("island capacities were not filled exactly")
    return per_island


def bucket_records(
    records: Sequence[Record],
    rollout_ids: Sequence[int],
    *,
    long_sample_threshold: int,
) -> list[list[Record]]:
    """Spread long records first, then pair remaining records by bucket cost."""
    if len(records) < len(rollout_ids):
        raise ValueError(
            f"{len(records)} samples cannot fill {len(rollout_ids)} non-empty rollout buckets"
        )
    max_bucket_size = (len(records) + len(rollout_ids) - 1) // len(rollout_ids)
    ordered = sorted(records, key=_record_order)
    buckets = [[record] for record in ordered[: len(rollout_ids)]]
    bucket_costs = [bucket[0].attention_cost for bucket in buckets]

    for record in ordered[len(rollout_ids) :]:
        candidates = [
            index for index, bucket in enumerate(buckets) if len(bucket) < max_bucket_size
        ]
        bucket_id = min(
            candidates,
            key=lambda index: (bucket_costs[index], len(buckets[index]), index),
        )
        buckets[bucket_id].append(record)
        bucket_costs[bucket_id] += record.attention_cost

    if any(not bucket for bucket in buckets):
        raise AssertionError("empty rollout bucket")
    if sum(map(len, buckets)) != len(records):
        raise AssertionError("rollout buckets do not reconstruct island records")
    multiple_long = [
        index
        for index, bucket in enumerate(buckets)
        if sum(record.token_length >= long_sample_threshold for record in bucket) > 1
    ]
    if multiple_long:
        raise ValueError(
            "cannot keep long samples in separate DP1/MBS1 rollout buckets; "
            f"multiple long samples in bucket indexes {multiple_long[:8]}"
        )
    return buckets


def _source_keys(records: Iterable[Record]) -> list[tuple[int, int]]:
    return [record.source_key for record in records]


def _verify_assignments(
    source_records: Sequence[Record],
    island_buckets: Sequence[Sequence[Sequence[Record]]],
    *,
    max_sequence_length: int,
) -> dict[str, Any]:
    source_keys = _source_keys(source_records)
    assigned_records = [
        record for island in island_buckets for bucket in island for record in bucket
    ]
    assigned_keys = _source_keys(assigned_records)
    source_set = set(source_keys)
    assigned_set = set(assigned_keys)
    duplicate_count = len(assigned_keys) - len(assigned_set)
    omissions = sorted(source_set - assigned_set)
    unexpected = sorted(assigned_set - source_set)
    if len(source_keys) != len(source_set):
        raise AssertionError("source coordinates are not unique")
    if duplicate_count or omissions or unexpected:
        raise AssertionError(
            f"assignment mismatch: duplicates={duplicate_count} "
            f"omissions={omissions[:8]} unexpected={unexpected[:8]}"
        )
    observed_max = max((record.token_length for record in assigned_records), default=0)
    if observed_max > max_sequence_length:
        raise AssertionError("assignment exceeds configured sequence length")
    return {
        "source_sample_count": len(source_keys),
        "assigned_sample_count": len(assigned_keys),
        "unique_assigned_sample_count": len(assigned_set),
        "duplicate_count": duplicate_count,
        "omission_count": len(omissions),
        "unexpected_count": len(unexpected),
        "max_token_length": observed_max,
        "max_sequence_length": max_sequence_length,
        "within_max_sequence_length": True,
        "samples_are_independent_unpacked_records": True,
    }


def _bucket_manifest(
    island_id: int,
    split: str,
    rollout_ids: Sequence[int],
    buckets: Sequence[Sequence[Record]],
    *,
    long_sample_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    bucket_costs: list[int] = []
    bucket_tokens: list[int] = []
    bucket_sizes: list[int] = []
    multiple_long = 0
    for rollout_id, bucket in zip(rollout_ids, buckets, strict=True):
        stats = split_stats(bucket)
        long_samples = sum(
            record.token_length >= long_sample_threshold for record in bucket
        )
        multiple_long += int(long_samples > 1)
        bucket_costs.append(stats.attention_cost_sum)
        bucket_tokens.append(stats.token_sum)
        bucket_sizes.append(stats.sample_count)
        mappings.append(
            {
                "island_id": island_id,
                "rollout_id": rollout_id,
                "split": split,
                "sample_count": stats.sample_count,
                "token_sum": stats.token_sum,
                "attention_cost_sum": stats.attention_cost_sum,
                "long_sample_count": long_samples,
                "sources": [
                    {
                        "rollout_id": record.source_rollout_id,
                        "position": record.source_position,
                        "token_length": record.token_length,
                        "attention_cost": record.attention_cost,
                    }
                    for record in bucket
                ],
            }
        )
    return mappings, {
        "rollout_count": len(buckets),
        "min_samples_per_rollout": min(bucket_sizes),
        "max_samples_per_rollout": max(bucket_sizes),
        "min_tokens_per_rollout": min(bucket_tokens),
        "max_tokens_per_rollout": max(bucket_tokens),
        "min_attention_cost_per_rollout": min(bucket_costs),
        "max_attention_cost_per_rollout": max(bucket_costs),
        "rollouts_with_multiple_long_samples": multiple_long,
    }


def _spread(values: Sequence[int]) -> dict[str, float | int]:
    minimum = min(values)
    maximum = max(values)
    mean = sum(values) / len(values)
    return {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "max_over_min": maximum / minimum if minimum else 0.0,
        "relative_range": (maximum - minimum) / mean if mean else 0.0,
    }


def plan_dataset(source: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    num_islands = config["num_islands"]
    max_sequence_length = config["max_sequence_length"]
    long_sample_threshold = config["long_sample_threshold"]
    train_ids = _rollout_range(config, "train_rollout_ids")
    validation_ids = _rollout_range(config, "validation_rollout_ids")

    train, train_hashes = load_split(
        source, train_ids, max_sequence_length=max_sequence_length
    )
    validation, validation_hashes = load_split(
        source, validation_ids, max_sequence_length=max_sequence_length
    )
    if len(train) != config["expected_train_samples"]:
        raise ValueError(
            f"train sample count {len(train)} != expected {config['expected_train_samples']}"
        )
    if len(validation) != config["expected_validation_samples"]:
        raise ValueError(
            "validation sample count "
            f"{len(validation)} != expected {config['expected_validation_samples']}"
        )

    train_islands = balance_records(train, num_islands=num_islands)
    validation_islands = balance_records(validation, num_islands=num_islands)
    train_buckets = [
        bucket_records(
            records,
            train_ids,
            long_sample_threshold=long_sample_threshold,
        )
        for records in train_islands
    ]
    validation_buckets = [
        bucket_records(
            records,
            validation_ids,
            long_sample_threshold=long_sample_threshold,
        )
        for records in validation_islands
    ]

    train_verification = _verify_assignments(
        train,
        train_buckets,
        max_sequence_length=max_sequence_length,
    )
    validation_verification = _verify_assignments(
        validation,
        validation_buckets,
        max_sequence_length=max_sequence_length,
    )
    mappings: list[dict[str, Any]] = []
    islands: list[dict[str, Any]] = []
    for island_id in range(num_islands):
        train_mapping, train_bucket_stats = _bucket_manifest(
            island_id,
            "train",
            train_ids,
            train_buckets[island_id],
            long_sample_threshold=long_sample_threshold,
        )
        validation_mapping, validation_bucket_stats = _bucket_manifest(
            island_id,
            "validation",
            validation_ids,
            validation_buckets[island_id],
            long_sample_threshold=long_sample_threshold,
        )
        mappings.extend(train_mapping)
        mappings.extend(validation_mapping)
        train_stats = split_stats(train_islands[island_id])
        validation_stats = split_stats(validation_islands[island_id])
        islands.append(
            {
                "island_id": island_id,
                "directory": f"island_{island_id}",
                "nominal_global_batch_size": 5 if island_id == 0 else 4,
                "train": train_stats.as_dict(),
                "validation": validation_stats.as_dict(),
                "train_rollout_buckets": train_bucket_stats,
                "validation_rollout_buckets": validation_bucket_stats,
            }
        )

    train_costs = [item["train"]["attention_cost_sum"] for item in islands]
    validation_costs = [item["validation"]["attention_cost_sum"] for item in islands]
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "dataset_version": config["dataset_version"],
        "strategy": "fixed-cardinality-lpt-by-true-token-length-squared",
        "balancing_cost": config["balancing_cost"],
        "sample_storage": config["sample_storage"],
        "num_islands": num_islands,
        "max_sequence_length": max_sequence_length,
        "long_sample_threshold": long_sample_threshold,
        "train_rollout_ids": list(train_ids),
        "validation_rollout_ids": list(validation_ids),
        "source_hashes": {**train_hashes, **validation_hashes},
        "source": {
            "root": str(source),
            "file_template": SOURCE_FILE_TEMPLATE,
            "hash_algorithm": "sha256",
            "train": split_stats(train).as_dict(),
            "validation": split_stats(validation).as_dict(),
        },
        "verification": {
            "train": train_verification,
            "validation": validation_verification,
        },
        "balance": {
            "train_attention_cost": _spread(train_costs),
            "validation_attention_cost": _spread(validation_costs),
        },
        "islands": islands,
        "mapping": mappings,
    }
    plan = {
        "train_buckets": train_buckets,
        "validation_buckets": validation_buckets,
        "train_rollout_ids": train_ids,
        "validation_rollout_ids": validation_ids,
    }
    return manifest, plan


def _write_dataset(output: Path, manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        for island_id in range(manifest["num_islands"]):
            island_dir = staging / f"island_{island_id}"
            island_dir.mkdir()
            for rollout_ids_key, buckets_key in (
                ("train_rollout_ids", "train_buckets"),
                ("validation_rollout_ids", "validation_buckets"),
            ):
                rollout_ids = plan[rollout_ids_key]
                buckets = plan[buckets_key][island_id]
                for rollout_id, bucket in zip(rollout_ids, buckets, strict=True):
                    # Selecting source objects into a list is intentional. Do
                    # not concatenate or pack token arrays here: DP1/MBS1 must
                    # see each source trajectory as one independent sample.
                    samples = [record.sample for record in bucket]
                    torch.save(
                        {"rollout_id": rollout_id, "samples": samples},
                        island_dir / SOURCE_FILE_TEMPLATE.format(rollout_id=rollout_id),
                    )
        with (staging / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_dataset(
    source: Path,
    output: Path,
    config: dict[str, Any],
    *,
    plan_only: bool = False,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source is not a directory: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output trees must be disjoint")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest, plan = plan_dataset(source, config)
    if not plan_only:
        _write_dataset(output, manifest, plan)
    return manifest


def _summary(manifest: dict[str, Any], *, output: Path, plan_only: bool) -> dict[str, Any]:
    return {
        "dataset_version": manifest["dataset_version"],
        "output": str(output),
        "plan_only": plan_only,
        "max_sequence_length": manifest["max_sequence_length"],
        "verification": manifest["verification"],
        "balance": manifest["balance"],
        "islands": [
            {
                "island_id": item["island_id"],
                "train": item["train"],
                "validation": item["validation"],
                "train_rollout_buckets": item["train_rollout_buckets"],
                "validation_rollout_buckets": item["validation_rollout_buckets"],
            }
            for item in manifest["islands"]
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate and print metrics without writing the output dataset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config.expanduser().resolve())
    manifest = build_dataset(
        args.source,
        args.output,
        config,
        plan_only=args.plan_only,
    )
    print(json.dumps(_summary(manifest, output=args.output, plan_only=args.plan_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
