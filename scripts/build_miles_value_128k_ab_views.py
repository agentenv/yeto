#!/usr/bin/env python3
"""Build topology-comparable 128K value-replay views.

The source is one island from the audited schema-3 contrastive replay pack.
This tool selects the first 64 source-train buckets whose contexts are all at
most 128K tokens, then preserves every selected bucket and sample verbatim:

* selected buckets 0..47 are training data;
* selected buckets 48..63 are held-out data;
* the baseline view exposes all 48 train buckets on one island; and
* the DiLoCo view exposes the identical train union as two 24-bucket islands.

Only the outer ``rollout_id`` and the destination coordinates in manifest rows
are changed.  Sample dictionaries, including their metadata and value weights,
are not edited.  The build is fail-closed and publishes both views together by
one atomic rename.
"""

from __future__ import annotations

import argparse
import array
import collections
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

MANIFEST_NAME = "manifest.json"
ROWS_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
VERIFICATION_NAME = "verification.json"
HASHES_NAME = "ARTIFACTS.sha256"
EXPECTED_PARENT_SCHEMA = 3
EXPECTED_PARENT_STRATEGY = "atomic-thread-reward-contrastive-window-balanced-v2"
OUTPUT_STRATEGY = "preserved-contrastive-buckets-128k-topology-ab-v1"
DEFAULT_MAX_SEQUENCE_LENGTH = 131_072
DEFAULT_SELECTED_BUCKETS = 64
DEFAULT_TRAIN_BUCKETS = 48
DEFAULT_SYNC_WINDOW = 12


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_hash(sample: dict[str, Any]) -> str:
    """Match the semantic hash used by the authoritative replay builders."""

    digest = hashlib.sha256()
    digest.update(array.array("I", sample["tokens"]).tobytes())
    digest.update(bytes(int(value) for value in sample["loss_mask"]))
    metadata = sample.get("metadata") or {}
    identity = {
        "thread_id": metadata.get("thread_id"),
        "compaction_epoch": metadata.get("compaction_epoch", 0),
        "trace_context_index": metadata.get("trace_context_index"),
        "trace_context_count": metadata.get("trace_context_count"),
        "reward": sample.get("reward"),
        "response_length": sample.get("response_length"),
        "label": sample.get("label"),
    }
    digest.update(
        json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()


def _stable_key(seed: int, *parts: object) -> str:
    return hashlib.sha256(
        "\0".join([str(seed), *(str(part) for part in parts)]).encode()
    ).hexdigest()


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _binary_reward(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} reward must be numeric 0/1, got {value!r}")
    reward = float(value)
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError(f"{context} reward must be finite 0/1, got {value!r}")
    return int(reward)


@dataclass(frozen=True)
class SourceBucket:
    rollout_id: int
    relative_path: str
    file_sha256: str
    rows: tuple[dict[str, Any], ...]

    @property
    def rewards(self) -> frozenset[int]:
        return frozenset(int(row["reward"]) for row in self.rows)

    @property
    def positive_rate(self) -> float:
        return sum(int(row["reward"]) for row in self.rows) / len(self.rows)

    @property
    def attention_cost(self) -> int:
        return sum(int(row["attention_cost"]) for row in self.rows)

    @property
    def identity_sha256(self) -> str:
        identities = [
            {
                "source_uid": row["source_uid"],
                "semantic_sha256": row["destination_sample_semantic_sha256"],
            }
            for row in self.rows
        ]
        payload = {
            "source_rollout_id": self.rollout_id,
            "samples": identities,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class SourceBundle:
    root: Path
    manifest: dict[str, Any]
    buckets: tuple[SourceBucket, ...]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_source(parent: Path, *, source_island: int) -> SourceBundle:
    manifest = _read_json(parent / MANIFEST_NAME)
    if manifest.get("schema_version") != EXPECTED_PARENT_SCHEMA:
        raise ValueError(
            f"parent schema must be {EXPECTED_PARENT_SCHEMA}, got "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("strategy") != EXPECTED_PARENT_STRATEGY:
        raise ValueError(f"unexpected parent strategy {manifest.get('strategy')!r}")
    num_islands = _require_int(
        manifest.get("num_islands"), name="manifest.num_islands", minimum=1
    )
    if source_island >= num_islands:
        raise ValueError(f"source island {source_island} is outside [0, {num_islands})")

    train_ids = manifest.get("train_rollout_ids")
    if not isinstance(train_ids, list) or train_ids != list(range(len(train_ids))):
        raise ValueError("parent train_rollout_ids must be contiguous from zero")
    recipe = manifest.get("critic_recipe") or {}
    required_recipe = {
        "value_loss_type": "classification",
        "value_num_bins": 51,
        "value_reward_range": [0.0, 1.0],
        "value_target_type": "hl_gauss",
        "hl_gauss_sigma_ratio": 0.75,
        "sample_weighting": "atomic-group-equal-within-step-v1",
    }
    for key, expected in required_recipe.items():
        if recipe.get(key) != expected:
            raise ValueError(
                f"parent critic recipe {key}={recipe.get(key)!r}, expected {expected!r}"
            )

    by_rollout: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    group_rollouts: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    group_rewards: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    seen_uids: set[str] = set()
    with (parent / ROWS_NAME).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{ROWS_NAME} line {line_number} is not an object")
            if row.get("split") != "train" or row.get("island_id") != source_island:
                continue
            rollout_id = _require_int(
                row.get("rollout_id"), name=f"row {line_number}.rollout_id"
            )
            if rollout_id not in train_ids:
                raise ValueError(f"row {line_number} has an undeclared train rollout")
            uid = str(row.get("source_uid") or "")
            if not uid or uid in seen_uids:
                raise ValueError(f"missing or duplicate source_uid {uid!r}")
            seen_uids.add(uid)
            row["reward"] = float(
                _binary_reward(row.get("reward"), context=f"row {line_number}")
            )
            token_length = _require_int(
                row.get("token_length"),
                name=f"row {line_number}.token_length",
                minimum=1,
            )
            attention_cost = _require_int(
                row.get("attention_cost"),
                name=f"row {line_number}.attention_cost",
                minimum=1,
            )
            if attention_cost != token_length * token_length:
                raise ValueError(f"row {line_number} attention_cost != token_length^2")
            by_rollout[rollout_id].append(row)
            group_key = (
                str(row.get("source_corpus") or ""),
                str(row.get("thread_id") or ""),
            )
            if not all(group_key):
                raise ValueError(f"row {line_number} is missing its atomic group key")
            group_rollouts[group_key].add(rollout_id)
            group_rewards[group_key].add(int(row["reward"]))

    if set(by_rollout) != set(train_ids):
        missing = sorted(set(train_ids) - set(by_rollout))
        raise ValueError(
            f"parent island {source_island} is missing train buckets {missing}"
        )
    split_groups = [
        key for key, rollouts in group_rollouts.items() if len(rollouts) != 1
    ]
    mixed_reward_groups = [
        key for key, rewards in group_rewards.items() if len(rewards) != 1
    ]
    if split_groups:
        raise ValueError(
            f"parent splits atomic groups across buckets: {split_groups[:3]}"
        )
    if mixed_reward_groups:
        raise ValueError(f"parent atomic groups mix rewards: {mixed_reward_groups[:3]}")

    buckets: list[SourceBucket] = []
    for rollout_id in train_ids:
        rows = sorted(
            by_rollout[rollout_id], key=lambda row: int(row["output_position"])
        )
        positions = [int(row["output_position"]) for row in rows]
        if positions != list(range(len(rows))):
            raise ValueError(
                f"source rollout {rollout_id} positions are not contiguous"
            )
        relative_path = f"island_{source_island}/data_{rollout_id}.pt"
        if {str(row.get("output_path")) for row in rows} != {relative_path}:
            raise ValueError(f"source rollout {rollout_id} has inconsistent paths")
        hashes = {str(row.get("output_file_sha256") or "") for row in rows}
        if len(hashes) != 1 or not next(iter(hashes)):
            raise ValueError(f"source rollout {rollout_id} has inconsistent hashes")
        if {_binary_reward(row["reward"], context=relative_path) for row in rows} != {
            0,
            1,
        }:
            raise ValueError(f"source rollout {rollout_id} is not mixed-label")
        buckets.append(
            SourceBucket(
                rollout_id=rollout_id,
                relative_path=relative_path,
                file_sha256=next(iter(hashes)),
                rows=tuple(rows),
            )
        )
    return SourceBundle(parent, manifest, tuple(buckets))


def select_buckets(
    source: SourceBundle,
    *,
    selected_buckets: int,
    max_sequence_length: int,
) -> tuple[SourceBucket, ...]:
    eligible = tuple(
        bucket
        for bucket in source.buckets
        if max(int(row["token_length"]) for row in bucket.rows) <= max_sequence_length
    )
    if len(eligible) < selected_buckets:
        raise ValueError(
            f"need {selected_buckets} eligible mixed-label buckets, found {len(eligible)}"
        )
    return eligible[:selected_buckets]


def _balance_windows(
    buckets: Sequence[SourceBucket],
    *,
    window_count: int,
    seed: int,
) -> tuple[tuple[SourceBucket, ...], ...]:
    if not buckets or window_count < 1 or len(buckets) % window_count:
        raise ValueError("buckets must divide evenly into non-empty windows")
    capacity = len(buckets) // window_count
    target_rate = sum(bucket.positive_rate for bucket in buckets) / len(buckets)
    target_cost = sum(bucket.attention_cost for bucket in buckets) / window_count
    windows: list[list[SourceBucket]] = [[] for _ in range(window_count)]
    reward_sums = [0.0] * window_count
    cost_sums = [0] * window_count
    ordered = sorted(
        buckets,
        key=lambda bucket: (
            -abs(bucket.positive_rate - target_rate),
            -bucket.attention_cost,
            _stable_key(seed, "order", bucket.identity_sha256),
        ),
    )
    for bucket in ordered:
        candidates = [
            index for index in range(window_count) if len(windows[index]) < capacity
        ]

        def score(
            index: int, candidate_bucket: SourceBucket = bucket
        ) -> tuple[float, float, float, str]:
            count = len(windows[index]) + 1
            reward_deviation = abs(
                reward_sums[index]
                + candidate_bucket.positive_rate
                - target_rate * count
            )
            cost_deviation = abs(
                cost_sums[index] + candidate_bucket.attention_cost - target_cost
            )
            return (
                reward_deviation,
                cost_deviation,
                len(windows[index]) / capacity,
                _stable_key(seed, "place", candidate_bucket.identity_sha256, index),
            )

        selected = min(candidates, key=score)
        windows[selected].append(bucket)
        reward_sums[selected] += bucket.positive_rate
        cost_sums[selected] += bucket.attention_cost

    for index, window in enumerate(windows):
        window.sort(key=lambda bucket: bucket.rollout_id)
        if len(window) != capacity:
            raise AssertionError(f"window {index} has the wrong capacity")
    return tuple(tuple(window) for window in windows)


def plan_views(
    selected: Sequence[SourceBucket],
    *,
    train_buckets: int,
    sync_window: int,
    seed: int,
) -> dict[str, dict[tuple[str, int], tuple[SourceBucket, ...]]]:
    if train_buckets < 1 or train_buckets >= len(selected):
        raise ValueError("train bucket count must leave a non-empty held-out split")
    if train_buckets % (2 * sync_window):
        raise ValueError("train buckets must divide into two islands of full H windows")
    heldout_buckets = len(selected) - train_buckets
    if heldout_buckets % 2:
        raise ValueError("held-out buckets must divide evenly across two islands")

    train_windows = _balance_windows(
        selected[:train_buckets],
        window_count=train_buckets // sync_window,
        seed=seed,
    )
    heldout_windows = _balance_windows(
        selected[train_buckets:], window_count=2, seed=seed + 1_000_003
    )
    baseline_train = tuple(bucket for window in train_windows for bucket in window)
    baseline_validation = tuple(
        bucket for window in heldout_windows for bucket in window
    )

    # Assign whole H windows to DiLoCo islands.  This retains exactly the same
    # four train windows as the baseline while giving each learner two windows.
    diloco_train = {
        island: tuple(
            bucket
            for window_index, window in enumerate(train_windows)
            if window_index % 2 == island
            for bucket in window
        )
        for island in (0, 1)
    }
    plan = {
        "baseline": {
            ("train", 0): baseline_train,
            ("validation", 0): baseline_validation,
        },
        "diloco": {
            ("train", 0): diloco_train[0],
            ("validation", 0): heldout_windows[0],
            ("train", 1): diloco_train[1],
            ("validation", 1): heldout_windows[1],
        },
    }
    if (
        len(diloco_train[0]) != train_buckets // 2
        or len(diloco_train[1]) != train_buckets // 2
    ):
        raise AssertionError("DiLoCo train split is not even")
    return plan


def _union_sha256(buckets: Sequence[SourceBucket]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(bucket.identity_sha256 for bucket in buckets)).encode()
    ).hexdigest()


def _window_metrics(
    buckets: Sequence[SourceBucket], window_size: int
) -> dict[str, Any]:
    global_rate = sum(bucket.positive_rate for bucket in buckets) / len(buckets)
    windows: list[dict[str, Any]] = []
    for start in range(0, len(buckets), window_size):
        window = buckets[start : start + window_size]
        rate = sum(bucket.positive_rate for bucket in window) / len(window)
        windows.append(
            {
                "start_rollout_id": start,
                "end_rollout_id_exclusive": start + len(window),
                "steps": len(window),
                "step_weighted_positive_rate": rate,
                "attention_cost": sum(bucket.attention_cost for bucket in window),
            }
        )
    return {
        "window_size": window_size,
        "window_count": len(windows),
        "global_step_weighted_positive_rate": global_rate,
        "max_step_positive_rate_deviation": max(
            abs(window["step_weighted_positive_rate"] - global_rate)
            for window in windows
        ),
        "windows": windows,
    }


def _split_metrics(
    buckets: Sequence[SourceBucket], *, window_size: int
) -> dict[str, Any]:
    rows = [row for bucket in buckets for row in bucket.rows]
    positive = sum(int(row["reward"]) for row in rows)
    group_keys = {
        (str(row.get("source_corpus")), str(row.get("thread_id"))) for row in rows
    }
    metrics = {
        "rollouts": len(buckets),
        "contexts": len(rows),
        "threads": len(group_keys),
        "pass": positive,
        "fail": len(rows) - positive,
        "tokens": sum(int(row["token_length"]) for row in rows),
        "supervised_tokens": sum(int(row["supervised_tokens"]) for row in rows),
        "attention_cost": sum(int(row["attention_cost"]) for row in rows),
        "min_contexts_per_rollout": min(len(bucket.rows) for bucket in buckets),
        "max_contexts_per_rollout": max(len(bucket.rows) for bucket in buckets),
        "mixed_label_rollouts": sum(bucket.rewards == {0, 1} for bucket in buckets),
        "all_steps_contrastive": all(bucket.rewards == {0, 1} for bucket in buckets),
        "context_positive_rate": positive / len(rows),
        "step_weighted_positive_rate": sum(bucket.positive_rate for bucket in buckets)
        / len(buckets),
        "source_rollout_ids": [bucket.rollout_id for bucket in buckets],
    }
    metrics["windows"] = _window_metrics(buckets, min(window_size, len(buckets)))
    return metrics


def _plan_verification(
    mapping: dict[tuple[str, int], tuple[SourceBucket, ...]],
    *,
    max_sequence_length: int,
    sync_window: int,
) -> dict[str, Any]:
    all_buckets = [bucket for buckets in mapping.values() for bucket in buckets]
    all_rows = [row for bucket in all_buckets for row in bucket.rows]
    uids = [str(row["source_uid"]) for row in all_rows]
    destinations: dict[tuple[str, str], set[tuple[str, int, int]]] = (
        collections.defaultdict(set)
    )
    step_label_failures = 0
    window_label_failures = 0
    max_deviation = 0.0
    for (split, island), buckets in mapping.items():
        for rollout_id, bucket in enumerate(buckets):
            step_label_failures += int(bucket.rewards != {0, 1})
            for row in bucket.rows:
                destinations[
                    (str(row.get("source_corpus")), str(row.get("thread_id")))
                ].add((split, island, rollout_id))
        if split == "train":
            window_metrics = _window_metrics(buckets, sync_window)
            max_deviation = max(
                max_deviation,
                float(window_metrics["max_step_positive_rate_deviation"]),
            )
            for start in range(0, len(buckets), sync_window):
                rewards = {
                    reward
                    for bucket in buckets[start : start + sync_window]
                    for reward in bucket.rewards
                }
                window_label_failures += int(rewards != {0, 1})
    verification = {
        "schema_version": 1,
        "materialized_contexts": len(all_rows),
        "duplicate_contexts": len(uids) - len(set(uids)),
        "omitted_contexts": 0,
        "unexpected_contexts": 0,
        "atomic_group_failures": sum(
            len(value) != 1 for value in destinations.values()
        ),
        "empty_rollouts": sum(not bucket.rows for bucket in all_buckets),
        "oversized_rollouts": 0,
        "context_length_failures": sum(
            int(int(row["token_length"]) > max_sequence_length) for row in all_rows
        ),
        "step_label_failures": step_label_failures,
        "window_label_failures": window_label_failures,
        "max_step_positive_rate_deviation": max_deviation,
        "max_allowed_step_positive_rate_deviation": 0.10,
    }
    verification["launch_ready"] = (
        not any(
            verification[key]
            for key in (
                "duplicate_contexts",
                "atomic_group_failures",
                "empty_rollouts",
                "context_length_failures",
                "step_label_failures",
                "window_label_failures",
            )
        )
        and max_deviation <= 0.10
    )
    if not verification["launch_ready"]:
        raise ValueError(f"view failed launch gates: {verification}")
    return verification


class SourceStore:
    def __init__(self, source: SourceBundle):
        self.source = source
        self.cache: dict[int, dict[str, Any]] = {}
        self.verified_files: set[int] = set()
        self.verified_samples: set[str] = set()

    def payload(self, bucket: SourceBucket) -> dict[str, Any]:
        cached = self.cache.get(bucket.rollout_id)
        if cached is not None:
            return cached
        path = self.source.root / bucket.relative_path
        if sha256_file(path) != bucket.file_sha256:
            raise ValueError(f"source file hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"source payload is not an object: {path}")
        samples = payload.get("samples")
        if not isinstance(samples, list) or len(samples) != len(bucket.rows):
            raise ValueError(f"source payload/manifest sample count mismatch: {path}")
        if payload.get("rollout_id") != bucket.rollout_id:
            raise ValueError(f"source payload rollout_id mismatch: {path}")
        for position, (row, sample) in enumerate(
            zip(bucket.rows, samples, strict=True)
        ):
            if not isinstance(sample, dict):
                raise TypeError(f"source sample {path}:{position} is not an object")
            tokens = sample.get("tokens")
            if tokens is None or len(tokens) != int(row["token_length"]):
                raise ValueError(f"source token length mismatch: {path}:{position}")
            semantic_hash = stable_sample_hash(sample)
            if semantic_hash != row.get("destination_sample_semantic_sha256"):
                raise ValueError(f"source semantic hash mismatch: {path}:{position}")
            if _binary_reward(
                sample.get("reward"), context=f"{path}:{position}"
            ) != int(row["reward"]):
                raise ValueError(f"source reward mismatch: {path}:{position}")
            train_metadata = sample.get("train_metadata") or {}
            weight = train_metadata.get("value_sample_weight")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0.0
                or not math.isclose(
                    float(weight),
                    float(row.get("value_sample_weight", float("nan"))),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"source value sample weight mismatch: {path}:{position}"
                )
            self.verified_samples.add(str(row["source_uid"]))
        self.verified_files.add(bucket.rollout_id)
        self.cache[bucket.rollout_id] = payload
        return payload


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_hashes(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != HASHES_NAME
    )
    with (root / HASHES_NAME).open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _materialize_view(
    *,
    source: SourceBundle,
    store: SourceStore,
    destination: Path,
    dataset_version: str,
    view_name: str,
    mapping: dict[tuple[str, int], tuple[SourceBucket, ...]],
    selected: Sequence[SourceBucket],
    max_sequence_length: int,
    sync_window: int,
    seed: int,
) -> dict[str, Any]:
    num_islands = 1 if view_name == "baseline" else 2
    train_count = len(mapping[("train", 0)])
    validation_count = len(mapping[("validation", 0)])
    verification = _plan_verification(
        mapping,
        max_sequence_length=max_sequence_length,
        sync_window=sync_window,
    )
    train_union = tuple(
        bucket
        for (split, _), buckets in mapping.items()
        if split == "train"
        for bucket in buckets
    )
    validation_union = tuple(
        bucket
        for (split, _), buckets in mapping.items()
        if split == "validation"
        for bucket in buckets
    )
    all_buckets = train_union + validation_union
    verification.update(
        {
            "planned_contexts": sum(len(bucket.rows) for bucket in all_buckets),
            "expected_parent_files": len({bucket.rollout_id for bucket in all_buckets}),
            "verified_parent_files": len({bucket.rollout_id for bucket in all_buckets}),
            "materialized_output_files": len(all_buckets),
            "selected_source_buckets": len(selected),
            "train_source_buckets": len(train_union),
            "heldout_source_buckets": len(validation_union),
            "train_union_sha256": _union_sha256(train_union),
            "heldout_union_sha256": _union_sha256(validation_union),
        }
    )

    manifest_rows: list[dict[str, Any]] = []
    for island in range(num_islands):
        island_dir = destination / f"island_{island}"
        island_dir.mkdir(parents=True)
        for split, offset in (("train", 0), ("validation", train_count)):
            for local_id, bucket in enumerate(mapping[(split, island)]):
                rollout_id = offset + local_id
                source_payload = store.payload(bucket)
                output_payload = dict(source_payload)
                output_payload["rollout_id"] = rollout_id
                output_path = island_dir / f"data_{rollout_id}.pt"
                torch.save(output_payload, output_path)
                output_hash = sha256_file(output_path)
                for position, source_row in enumerate(bucket.rows):
                    row = dict(source_row)
                    row.update(
                        {
                            "split": split,
                            "island_id": island,
                            "rollout_id": rollout_id,
                            "output_path": str(output_path.relative_to(destination)),
                            "output_file_sha256": output_hash,
                            "output_position": position,
                            "parent_split": source_row["split"],
                            "parent_island_id": source_row["island_id"],
                            "parent_rollout_id": source_row["rollout_id"],
                            "parent_output_path": source_row["output_path"],
                            "parent_output_position": source_row["output_position"],
                        }
                    )
                    manifest_rows.append(row)

    islands = [
        {
            "island_id": island,
            "train": _split_metrics(
                mapping[("train", island)], window_size=sync_window
            ),
            "validation": _split_metrics(
                mapping[("validation", island)], window_size=sync_window
            ),
        }
        for island in range(num_islands)
    ]
    parent = {
        "root": str(source.root),
        "dataset_version": source.manifest["dataset_version"],
        "source_island_id": int(selected[0].rows[0]["island_id"]),
        "manifest_sha256": sha256_file(source.root / MANIFEST_NAME),
        "manifest_jsonl_sha256": sha256_file(source.root / ROWS_NAME),
        "selected_source_rollout_ids": [bucket.rollout_id for bucket in selected],
    }
    manifest = {
        "schema_version": 3,
        "dataset_version": dataset_version,
        "strategy": OUTPUT_STRATEGY,
        "view": view_name,
        "seed": seed,
        "sync_window_size": sync_window,
        "num_islands": num_islands,
        "max_sequence_length": max_sequence_length,
        "max_contexts_per_rollout_file": max(len(bucket.rows) for bucket in selected),
        "train_rollout_ids": list(range(train_count)),
        "validation_rollout_ids": list(
            range(train_count, train_count + validation_count)
        ),
        "parent": parent,
        "critic_recipe": dict(source.manifest["critic_recipe"]),
        "islands": islands,
        "verification": verification,
    }
    summary = {
        **manifest,
        "schema_version": 1,
        "train_rollouts_per_island": train_count,
        "validation_rollouts_per_island": validation_count,
        "validation_start_rollout": train_count,
    }
    with (destination / ROWS_NAME).open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _write_json(destination / MANIFEST_NAME, manifest)
    _write_json(destination / SUMMARY_NAME, summary)
    _write_json(destination / VERIFICATION_NAME, verification)
    tooling = destination / "tooling"
    tooling.mkdir()
    shutil.copy2(Path(__file__), tooling / Path(__file__).name)
    _write_hashes(destination)
    return manifest


def build_ab_views(
    parent: Path,
    output: Path,
    *,
    source_island: int = 1,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    selected_buckets: int = DEFAULT_SELECTED_BUCKETS,
    train_buckets: int = DEFAULT_TRAIN_BUCKETS,
    sync_window: int = DEFAULT_SYNC_WINDOW,
    seed: int = 20260831,
) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"parent bundle does not exist: {parent}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("parent and output trees must be disjoint")
    for name, value in (
        ("source_island", source_island),
        ("max_sequence_length", max_sequence_length),
        ("selected_buckets", selected_buckets),
        ("train_buckets", train_buckets),
        ("sync_window", sync_window),
    ):
        if value < (0 if name == "source_island" else 1):
            raise ValueError(f"{name} has an invalid value: {value}")

    source = _read_source(parent, source_island=source_island)
    selected = select_buckets(
        source,
        selected_buckets=selected_buckets,
        max_sequence_length=max_sequence_length,
    )
    plans = plan_views(
        selected,
        train_buckets=train_buckets,
        sync_window=sync_window,
        seed=seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    store = SourceStore(source)
    try:
        manifests = {}
        for view_name in ("baseline", "diloco"):
            destination = staging / view_name
            destination.mkdir()
            manifests[view_name] = _materialize_view(
                source=source,
                store=store,
                destination=destination,
                dataset_version=f"{output.name}-{view_name}",
                view_name=view_name,
                mapping=plans[view_name],
                selected=selected,
                max_sequence_length=max_sequence_length,
                sync_window=sync_window,
                seed=seed,
            )
        baseline_verification = manifests["baseline"]["verification"]
        diloco_verification = manifests["diloco"]["verification"]
        train_equal = (
            baseline_verification["train_union_sha256"]
            == diloco_verification["train_union_sha256"]
        )
        heldout_equal = (
            baseline_verification["heldout_union_sha256"]
            == diloco_verification["heldout_union_sha256"]
        )
        ab_manifest = {
            "schema_version": 1,
            "kind": "miles-value-128k-topology-ab-views",
            "dataset_version": output.name,
            "seed": seed,
            "max_sequence_length": max_sequence_length,
            "source_island_id": source_island,
            "selection": {
                "eligible_policy": "source-train-order; mixed-label; every-context-lte-max",
                "selected_source_rollout_ids": [
                    bucket.rollout_id for bucket in selected
                ],
                "train_source_rollout_ids": [
                    bucket.rollout_id for bucket in selected[:train_buckets]
                ],
                "heldout_source_rollout_ids": [
                    bucket.rollout_id for bucket in selected[train_buckets:]
                ],
            },
            "views": {
                name: {
                    "path": name,
                    "manifest_sha256": sha256_file(staging / name / MANIFEST_NAME),
                    "num_islands": manifests[name]["num_islands"],
                    "train_rollouts_per_island": len(plans[name][("train", 0)]),
                    "validation_rollouts_per_island": len(
                        plans[name][("validation", 0)]
                    ),
                }
                for name in ("baseline", "diloco")
            },
            "verification": {
                "train_union_identical": train_equal,
                "heldout_union_identical": heldout_equal,
                "train_union_sha256": baseline_verification["train_union_sha256"],
                "heldout_union_sha256": baseline_verification["heldout_union_sha256"],
                "selected_source_files_verified": len(store.verified_files),
                "selected_source_samples_verified": len(store.verified_samples),
                "launch_ready": train_equal and heldout_equal,
            },
        }
        if not ab_manifest["verification"]["launch_ready"]:
            raise ValueError("A/B views do not contain identical train/heldout unions")
        _write_json(staging / "ab_manifest.json", ab_manifest)
        tooling = staging / "tooling"
        tooling.mkdir()
        shutil.copy2(Path(__file__), tooling / Path(__file__).name)
        _write_hashes(staging)
        os.replace(staging, output)
        return ab_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-island", type=int, default=1)
    parser.add_argument(
        "--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH
    )
    parser.add_argument(
        "--selected-buckets", type=int, default=DEFAULT_SELECTED_BUCKETS
    )
    parser.add_argument("--train-buckets", type=int, default=DEFAULT_TRAIN_BUCKETS)
    parser.add_argument("--sync-window", type=int, default=DEFAULT_SYNC_WINDOW)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_ab_views(
        args.parent,
        args.output,
        source_island=args.source_island,
        max_sequence_length=args.max_sequence_length,
        selected_buckets=args.selected_buckets,
        train_buckets=args.train_buckets,
        sync_window=args.sync_window,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
