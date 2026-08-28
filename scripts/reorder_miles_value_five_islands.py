#!/usr/bin/env python3
"""Reorder the validated five-island Qwen value replay without changing samples.

The parent bundle is immutable.  This tool preserves every sample's island,
train/validation split, semantic hash, and atomic compaction-thread group.  It
only changes which rollout file contains a group and the chronological order
of those files.

The output is designed for sequential Miles debug replay:

* compaction groups remain atomic;
* every optimizer bucket contains both reward labels;
* contexts are weighted so each source thread contributes equally per step; and
* buckets are scheduled so every DiLoCo window sees a stable reward and
  attention-cost mixture.

The build is fail-closed and publishes by one atomic rename.  It refuses to
overwrite either the parent or an existing destination.
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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

MAX_CONTEXTS_PER_BUCKET = 5
EXPECTED_NUM_ISLANDS = 5
EXPECTED_PARENT_SCHEMA = 2
EXPECTED_PARENT_STRATEGY = (
    "atomic-thread-compaction-group-lpt-by-sum-true-token-length-squared"
)
OUTPUT_STRATEGY = "atomic-thread-reward-contrastive-window-balanced-v2"
MANIFEST_NAME = "manifest.json"
ROWS_NAME = "manifest.jsonl"
SUMMARY_NAME = "summary.json"
VERIFICATION_NAME = "verification.json"
HASHES_NAME = "ARTIFACTS.sha256"


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_sample_hash(sample: dict[str, Any]) -> str:
    """Match the semantic hash used by the authoritative parent builder."""

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


def _stable_random_key(seed: int, *parts: object) -> str:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode()).hexdigest()


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_binary_reward(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} reward must be numeric 0/1, got {value!r}")
    reward = float(value)
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError(f"{context} reward must be finite 0/1, got {value!r}")
    return int(reward)


@dataclass(frozen=True)
class Context:
    row: dict[str, Any]

    @property
    def uid(self) -> str:
        return str(self.row["source_uid"])

    @property
    def reward(self) -> int:
        return int(self.row["reward"])

    @property
    def token_length(self) -> int:
        return int(self.row["token_length"])

    @property
    def supervised_tokens(self) -> int:
        return int(self.row["supervised_tokens"])

    @property
    def attention_cost(self) -> int:
        return int(self.row["attention_cost"])

    @property
    def old_path(self) -> str:
        return str(self.row["output_path"])

    @property
    def old_position(self) -> int:
        return int(self.row["output_position"])


@dataclass(frozen=True)
class Group:
    key: tuple[str, int, str, str]
    contexts: tuple[Context, ...]

    @property
    def reward(self) -> int:
        return self.contexts[0].reward

    @property
    def size(self) -> int:
        return len(self.contexts)

    @property
    def attention_cost(self) -> int:
        return sum(context.attention_cost for context in self.contexts)

    @property
    def supervised_tokens(self) -> int:
        return sum(context.supervised_tokens for context in self.contexts)

    @property
    def positive_supervised_tokens(self) -> int:
        return self.reward * self.supervised_tokens

    @property
    def stable_id(self) -> str:
        return ":".join(map(str, self.key))


@dataclass
class Bucket:
    groups: list[Group]

    @property
    def contexts(self) -> tuple[Context, ...]:
        return tuple(context for group in self.groups for context in group.contexts)

    @property
    def size(self) -> int:
        return sum(group.size for group in self.groups)

    @property
    def positives(self) -> int:
        return sum(group.reward * group.size for group in self.groups)

    @property
    def reward_fraction(self) -> float:
        return self.positives / self.size

    @property
    def attention_cost(self) -> int:
        return sum(group.attention_cost for group in self.groups)

    @property
    def supervised_tokens(self) -> int:
        return sum(group.supervised_tokens for group in self.groups)

    @property
    def positive_supervised_tokens(self) -> int:
        return sum(group.positive_supervised_tokens for group in self.groups)

    @property
    def stable_id(self) -> str:
        return "+".join(sorted(group.stable_id for group in self.groups))

    def sample_weight(self, group: Group) -> float:
        """Scale contexts so every atomic thread has equal weight in this step.

        Miles divides the summed per-context means by the dynamic batch size.
        These weights sum to that same batch size while making the aggregate
        weight of each atomic group identical.
        """

        if group not in self.groups:
            raise ValueError("group is not present in bucket")
        return self.size / (len(self.groups) * group.size)


def _read_parent(parent: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = parent / MANIFEST_NAME
    rows_path = parent / ROWS_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("parent manifest root must be an object")
    if manifest.get("schema_version") != EXPECTED_PARENT_SCHEMA:
        raise ValueError(
            f"parent schema must be {EXPECTED_PARENT_SCHEMA}, got "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("strategy") != EXPECTED_PARENT_STRATEGY:
        raise ValueError(f"unexpected parent strategy {manifest.get('strategy')!r}")
    if manifest.get("num_islands") != EXPECTED_NUM_ISLANDS:
        raise ValueError("parent must contain exactly five islands")

    rows: list[dict[str, Any]] = []
    with rows_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad {ROWS_NAME} line {line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{ROWS_NAME} line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("parent manifest.jsonl is empty")
    return manifest, rows


def _validate_rows(
    manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[tuple[str, int], list[Context]]:
    train_ids = tuple(manifest.get("train_rollout_ids", ()))
    validation_ids = tuple(manifest.get("validation_rollout_ids", ()))
    if not train_ids or not validation_ids:
        raise ValueError("parent train/validation rollout IDs must be non-empty")
    if train_ids != tuple(range(len(train_ids))):
        raise ValueError("parent train rollout IDs must be contiguous from zero")
    if validation_ids != tuple(
        range(len(train_ids), len(train_ids) + len(validation_ids))
    ):
        raise ValueError("parent validation rollout IDs must follow train IDs")

    expected_ids = {"train": set(train_ids), "validation": set(validation_ids)}
    seen_uids: set[str] = set()
    seen_destinations: set[tuple[int, int, int]] = set()
    by_split_island: dict[tuple[str, int], list[Context]] = collections.defaultdict(
        list
    )
    for index, row in enumerate(rows):
        context = f"manifest row {index}"
        split = row.get("split")
        if split not in expected_ids:
            raise ValueError(f"{context} has invalid split {split!r}")
        island = _require_int(row.get("island_id"), name=f"{context}.island_id")
        if island >= EXPECTED_NUM_ISLANDS:
            raise ValueError(f"{context}.island_id is outside [0, 5)")
        rollout = _require_int(row.get("rollout_id"), name=f"{context}.rollout_id")
        if rollout not in expected_ids[split]:
            raise ValueError(f"{context} rollout is outside its declared split")
        position = _require_int(
            row.get("output_position"), name=f"{context}.output_position"
        )
        expected_path = f"island_{island}/data_{rollout}.pt"
        if row.get("output_path") != expected_path:
            raise ValueError(f"{context} output path does not match coordinates")
        destination = (island, rollout, position)
        if destination in seen_destinations:
            raise ValueError(f"duplicate parent destination {destination}")
        seen_destinations.add(destination)

        uid = str(row.get("source_uid") or "")
        if not uid or uid in seen_uids:
            raise ValueError(f"missing or duplicate source_uid in {context}: {uid!r}")
        seen_uids.add(uid)
        row["reward"] = float(
            _require_binary_reward(row.get("reward"), context=context)
        )
        token_length = _require_int(
            row.get("token_length"), name=f"{context}.token_length", minimum=1
        )
        supervised_tokens = _require_int(
            row.get("supervised_tokens"),
            name=f"{context}.supervised_tokens",
            minimum=1,
        )
        attention_cost = _require_int(
            row.get("attention_cost"), name=f"{context}.attention_cost", minimum=1
        )
        if attention_cost != token_length * token_length:
            raise ValueError(f"{context} attention cost is not token_length^2")
        if supervised_tokens > token_length:
            raise ValueError(f"{context} supervised tokens exceed token length")
        if not str(row.get("thread_id") or ""):
            raise ValueError(f"{context} is missing thread_id")
        if not str(row.get("source_corpus") or ""):
            raise ValueError(f"{context} is missing source_corpus")
        by_split_island[(split, island)].append(Context(row))

    expected_pairs = {
        (split, island)
        for split in ("train", "validation")
        for island in range(EXPECTED_NUM_ISLANDS)
    }
    if set(by_split_island) != expected_pairs:
        raise ValueError("parent rows do not cover every split/island pair")
    return by_split_island


def _groups(contexts: Iterable[Context]) -> list[Group]:
    grouped: dict[tuple[str, int, str, str], list[Context]] = collections.defaultdict(
        list
    )
    for context in contexts:
        row = context.row
        key = (
            str(row["split"]),
            int(row["island_id"]),
            str(row["source_corpus"]),
            str(row["thread_id"]),
        )
        grouped[key].append(context)

    result: list[Group] = []
    for key, items in grouped.items():
        rewards = {item.reward for item in items}
        if len(rewards) != 1:
            raise ValueError(f"atomic group {key} contains mixed rewards")
        items.sort(
            key=lambda item: (
                int(item.row.get("trace_context_index", 0)),
                int(item.row.get("compaction_epoch", 0)),
                item.uid,
            )
        )
        epochs = [int(item.row.get("compaction_epoch", 0)) for item in items]
        if len(epochs) != len(set(epochs)):
            raise ValueError(f"atomic group {key} repeats a compaction epoch")
        if len(items) > MAX_CONTEXTS_PER_BUCKET:
            raise ValueError(
                f"atomic group {key} has {len(items)} contexts, exceeding "
                f"the bucket limit {MAX_CONTEXTS_PER_BUCKET}"
            )
        result.append(Group(key, tuple(items)))
    return result


def _strict_mixed_buckets(
    groups: Sequence[Group], *, num_buckets: int, seed: int
) -> list[Bucket]:
    """Place all groups while guaranteeing pass/fail contrast in every bucket."""

    by_reward = {
        reward: sorted(
            (group for group in groups if group.reward == reward),
            key=lambda group: (
                -group.size,
                -group.attention_cost,
                _stable_random_key(seed, "seed", reward, group.stable_id),
            ),
        )
        for reward in (0, 1)
    }
    for reward, candidates in by_reward.items():
        if len(candidates) < num_buckets:
            raise ValueError(
                f"strict mixing needs {num_buckets} reward={reward} groups, "
                f"but only {len(candidates)} are available"
            )

    # Seed every bucket with one group of each label.  Larger compaction groups
    # are consumed first so they cannot become impossible to place late.
    buckets = [Bucket([group]) for group in by_reward[0][:num_buckets]]
    positive_seeds = by_reward[1][:num_buckets]
    for group in positive_seeds:
        candidates = [
            bucket
            for bucket in buckets
            if len(bucket.groups) == 1
            and bucket.size + group.size <= MAX_CONTEXTS_PER_BUCKET
        ]
        if not candidates:
            raise ValueError("cannot seed strict mixed buckets within size limit")
        selected = min(
            candidates,
            key=lambda bucket: (
                bucket.attention_cost,
                bucket.size,
                _stable_random_key(seed, "pair", group.stable_id, bucket.stable_id),
            ),
        )
        selected.groups.append(group)

    remaining = by_reward[0][num_buckets:] + by_reward[1][num_buckets:]
    remaining.sort(
        key=lambda group: (
            -group.size,
            -group.attention_cost,
            _stable_random_key(seed, "extra", group.stable_id),
        )
    )
    target = sum(group.reward * group.size for group in groups) / sum(
        group.size for group in groups
    )
    for group in remaining:
        candidates = [
            bucket
            for bucket in buckets
            if bucket.size + group.size <= MAX_CONTEXTS_PER_BUCKET
        ]
        if not candidates:
            raise ValueError("cannot place all groups within bucket size limit")

        def score(
            bucket: Bucket, candidate_group: Group = group
        ) -> tuple[int, float, int, str]:
            new_size = bucket.size + candidate_group.size
            new_fraction = (
                bucket.positives + candidate_group.reward * candidate_group.size
            ) / new_size
            return (
                len(bucket.groups),
                abs(new_fraction - target),
                bucket.attention_cost,
                _stable_random_key(
                    seed, "place", candidate_group.stable_id, bucket.stable_id
                ),
            )

        min(candidates, key=score).groups.append(group)

    if any(
        bucket.size > MAX_CONTEXTS_PER_BUCKET
        or {group.reward for group in bucket.groups} != {0, 1}
        for bucket in buckets
    ):
        raise AssertionError("strict mixed bucket construction violated its contract")
    return buckets


def _schedule_windows(
    buckets: Sequence[Bucket], *, window_size: int, seed: int
) -> list[Bucket]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    capacities = [window_size] * (len(buckets) // window_size)
    remainder = len(buckets) % window_size
    if remainder:
        capacities.append(remainder)

    windows: list[list[Bucket]] = [[] for _ in capacities]
    reward_sums = [0.0] * len(capacities)
    cost_sums = [0] * len(capacities)
    ordered = sorted(
        buckets,
        key=lambda bucket: (
            -bucket.reward_fraction,
            -bucket.attention_cost,
            _stable_random_key(seed, "schedule", bucket.stable_id),
        ),
    )
    for bucket in ordered:
        candidates = [
            index
            for index, capacity in enumerate(capacities)
            if len(windows[index]) < capacity
        ]
        if not candidates:
            raise AssertionError("window scheduler ran out of capacity")

        def score(
            index: int, candidate_bucket: Bucket = bucket
        ) -> tuple[float, float, float, str]:
            # Reward-LPT: positive-heavy buckets are placed first and always
            # go to the window with the lowest final-capacity-normalized reward
            # load.  Attention cost is the deterministic secondary load.
            return (
                reward_sums[index] / capacities[index],
                cost_sums[index] / capacities[index],
                len(windows[index]) / capacities[index],
                _stable_random_key(seed, "window", candidate_bucket.stable_id, index),
            )

        selected = min(candidates, key=score)
        windows[selected].append(bucket)
        reward_sums[selected] += bucket.reward_fraction
        cost_sums[selected] += bucket.attention_cost

    scheduled: list[Bucket] = []
    for window_index, window in enumerate(windows):
        # A second greedy pass makes prefixes inside H deterministic and mixed.
        remaining = list(window)
        prefix_reward = 0.0
        prefix_positive_tokens = 0
        prefix_tokens = 0
        ordered_window: list[Bucket] = []
        window_reward = sum(bucket.reward_fraction for bucket in window) / len(window)
        window_tokens = sum(bucket.supervised_tokens for bucket in window)
        window_token_reward = (
            sum(bucket.positive_supervised_tokens for bucket in window) / window_tokens
        )
        while remaining:
            position = len(ordered_window) + 1

            def prefix_score(
                bucket: Bucket,
                current_reward: float = prefix_reward,
                current_positive_tokens: int = prefix_positive_tokens,
                current_tokens: int = prefix_tokens,
                prefix_position: int = position,
                target_reward: float = window_reward,
                target_token_reward: float = window_token_reward,
                current_window: int = window_index,
            ) -> tuple[float, float, str]:
                reward_dev = abs(
                    current_reward
                    + bucket.reward_fraction
                    - prefix_position * target_reward
                )
                tokens = current_tokens + bucket.supervised_tokens
                token_dev = abs(
                    current_positive_tokens
                    + bucket.positive_supervised_tokens
                    - tokens * target_token_reward
                ) / max(tokens, 1)
                return (
                    reward_dev,
                    token_dev,
                    _stable_random_key(
                        seed, "prefix", current_window, bucket.stable_id
                    ),
                )

            selected_bucket = min(remaining, key=prefix_score)
            remaining.remove(selected_bucket)
            ordered_window.append(selected_bucket)
            prefix_reward += selected_bucket.reward_fraction
            prefix_positive_tokens += selected_bucket.positive_supervised_tokens
            prefix_tokens += selected_bucket.supervised_tokens
        scheduled.extend(ordered_window)
    return scheduled


def pack_groups(
    groups: Sequence[Group],
    *,
    num_buckets: int,
    window_size: int,
    seed: int,
) -> list[Bucket]:
    if len(groups) < num_buckets:
        raise ValueError(
            f"{len(groups)} atomic groups cannot fill {num_buckets} buckets"
        )
    buckets = _strict_mixed_buckets(groups, num_buckets=num_buckets, seed=seed)
    if len(buckets) != num_buckets:
        raise AssertionError("packer emitted the wrong number of buckets")
    if any(
        not bucket.groups
        or bucket.size > MAX_CONTEXTS_PER_BUCKET
        or {group.reward for group in bucket.groups} != {0, 1}
        for bucket in buckets
    ):
        raise AssertionError("packer emitted an empty or oversized bucket")
    return _schedule_windows(buckets, window_size=window_size, seed=seed)


def _window_metrics(buckets: Sequence[Bucket], window_size: int) -> dict[str, Any]:
    global_step_reward = sum(bucket.reward_fraction for bucket in buckets) / len(
        buckets
    )
    total_supervised = sum(bucket.supervised_tokens for bucket in buckets)
    global_token_reward = (
        sum(bucket.positive_supervised_tokens for bucket in buckets) / total_supervised
    )
    windows: list[dict[str, Any]] = []
    for start in range(0, len(buckets), window_size):
        window = buckets[start : start + window_size]
        supervised = sum(bucket.supervised_tokens for bucket in window)
        row = {
            "start_rollout_id": start,
            "end_rollout_id_exclusive": start + len(window),
            "steps": len(window),
            "contexts": sum(bucket.size for bucket in window),
            "step_weighted_positive_rate": sum(
                bucket.reward_fraction for bucket in window
            )
            / len(window),
            "supervised_token_positive_rate": sum(
                bucket.positive_supervised_tokens for bucket in window
            )
            / supervised,
            "attention_cost": sum(bucket.attention_cost for bucket in window),
        }
        windows.append(row)
    return {
        "window_size": window_size,
        "window_count": len(windows),
        "global_step_weighted_positive_rate": global_step_reward,
        "global_supervised_token_positive_rate": global_token_reward,
        "max_step_positive_rate_deviation": max(
            abs(row["step_weighted_positive_rate"] - global_step_reward)
            for row in windows
        ),
        "max_token_positive_rate_deviation": max(
            abs(row["supervised_token_positive_rate"] - global_token_reward)
            for row in windows
        ),
        "windows": windows,
    }


def _bucket_metrics(buckets: Sequence[Bucket], window_size: int) -> dict[str, Any]:
    contexts = [context for bucket in buckets for context in bucket.contexts]
    sizes = [bucket.size for bucket in buckets]
    mixed = sum(0 < bucket.positives < bucket.size for bucket in buckets)
    all_positive = sum(bucket.positives == bucket.size for bucket in buckets)
    all_negative = sum(bucket.positives == 0 for bucket in buckets)
    return {
        "rollouts": len(buckets),
        "contexts": len(contexts),
        "threads": sum(len(bucket.groups) for bucket in buckets),
        "pass": sum(context.reward for context in contexts),
        "fail": sum(1 - context.reward for context in contexts),
        "tokens": sum(context.token_length for context in contexts),
        "supervised_tokens": sum(context.supervised_tokens for context in contexts),
        "attention_cost": sum(context.attention_cost for context in contexts),
        "min_contexts_per_rollout": min(sizes),
        "max_contexts_per_rollout": max(sizes),
        "mixed_label_rollouts": mixed,
        "all_positive_rollouts": all_positive,
        "all_negative_rollouts": all_negative,
        "mixed_label_rollout_rate": mixed / len(buckets),
        "all_steps_contrastive": mixed == len(buckets),
        "context_positive_rate": sum(context.reward for context in contexts)
        / len(contexts),
        "step_weighted_positive_rate": sum(bucket.reward_fraction for bucket in buckets)
        / len(buckets),
        "supervised_token_positive_rate": sum(
            context.reward * context.supervised_tokens for context in contexts
        )
        / sum(context.supervised_tokens for context in contexts),
        "windows": _window_metrics(buckets, window_size),
    }


def plan_bundle(
    parent: Path,
    *,
    train_rollouts: int,
    validation_rollouts: int,
    window_size: int,
    seed: int,
) -> tuple[dict[str, Any], dict[tuple[str, int], list[Bucket]], list[dict[str, Any]]]:
    parent_manifest, rows = _read_parent(parent)
    by_split_island = _validate_rows(parent_manifest, rows)
    plan: dict[tuple[str, int], list[Bucket]] = {}
    islands: list[dict[str, Any]] = []
    for island in range(EXPECTED_NUM_ISLANDS):
        island_stats: dict[str, Any] = {"island_id": island}
        for split, count in (
            ("train", train_rollouts),
            ("validation", validation_rollouts),
        ):
            groups = _groups(by_split_island[(split, island)])
            buckets = pack_groups(
                groups,
                num_buckets=count,
                window_size=window_size,
                seed=seed + island * 1009 + (0 if split == "train" else 1_000_003),
            )
            plan[(split, island)] = buckets
            island_stats[split] = _bucket_metrics(buckets, window_size)
        islands.append(island_stats)

    verification = _verify_plan(rows, plan, window_size=window_size)
    summary = {
        "schema_version": 1,
        "dataset_version": "pending-output-name",
        "strategy": OUTPUT_STRATEGY,
        "seed": seed,
        "sync_window_size": window_size,
        "num_islands": EXPECTED_NUM_ISLANDS,
        "max_contexts_per_rollout_file": MAX_CONTEXTS_PER_BUCKET,
        "train_rollouts_per_island": train_rollouts,
        "validation_rollouts_per_island": validation_rollouts,
        "validation_start_rollout": train_rollouts,
        "parent": {
            "root": str(parent),
            "dataset_version": parent_manifest["dataset_version"],
            "manifest_sha256": sha256_file(parent / MANIFEST_NAME),
            "manifest_jsonl_sha256": sha256_file(parent / ROWS_NAME),
        },
        "islands": islands,
        "verification": verification,
        "critic_recipe": {
            "value_loss_type": "classification",
            "value_num_bins": 51,
            "value_reward_range": [0.0, 1.0],
            "value_target_type": "hl_gauss",
            "hl_gauss_sigma_ratio": 0.75,
            "sample_weighting": "atomic-group-equal-within-step-v1",
        },
    }
    return summary, plan, rows


def _verify_plan(
    parent_rows: Sequence[dict[str, Any]],
    plan: dict[tuple[str, int], list[Bucket]],
    *,
    window_size: int,
) -> dict[str, Any]:
    parent_uids = [str(row["source_uid"]) for row in parent_rows]
    planned_contexts = [
        context
        for buckets in plan.values()
        for bucket in buckets
        for context in bucket.contexts
    ]
    planned_uids = [context.uid for context in planned_contexts]
    duplicate_count = len(planned_uids) - len(set(planned_uids))
    omitted = set(parent_uids) - set(planned_uids)
    unexpected = set(planned_uids) - set(parent_uids)

    destinations_by_group: dict[tuple[str, int, str, str], set[int]] = (
        collections.defaultdict(set)
    )
    oversized = 0
    empty = 0
    step_label_failures = 0
    window_label_failures = 0
    max_step_deviation = 0.0
    for buckets in plan.values():
        for rollout_id, bucket in enumerate(buckets):
            empty += int(not bucket.groups)
            oversized += int(bucket.size > MAX_CONTEXTS_PER_BUCKET)
            step_label_failures += int(
                {group.reward for group in bucket.groups} != {0, 1}
            )
            for group in bucket.groups:
                destinations_by_group[group.key].add(rollout_id)
        metrics = _window_metrics(buckets, window_size)
        max_step_deviation = max(
            max_step_deviation, metrics["max_step_positive_rate_deviation"]
        )
        for row in metrics["windows"]:
            start = int(row["start_rollout_id"])
            end = int(row["end_rollout_id_exclusive"])
            contexts = [
                context for bucket in buckets[start:end] for context in bucket.contexts
            ]
            window_label_failures += int(
                len({context.reward for context in contexts}) != 2
            )

    atomic_failures = sum(
        len(destinations) != 1 for destinations in destinations_by_group.values()
    )
    launch_ready = (
        not any(
            (
                duplicate_count,
                len(omitted),
                len(unexpected),
                atomic_failures,
                oversized,
                empty,
                step_label_failures,
                window_label_failures,
            )
        )
        and max_step_deviation <= 0.10
    )
    verification = {
        "schema_version": 1,
        "parent_contexts": len(parent_uids),
        "planned_contexts": len(planned_uids),
        "duplicate_contexts": duplicate_count,
        "omitted_contexts": len(omitted),
        "unexpected_contexts": len(unexpected),
        "atomic_group_failures": atomic_failures,
        "empty_rollouts": empty,
        "oversized_rollouts": oversized,
        "step_label_failures": step_label_failures,
        "window_label_failures": window_label_failures,
        "max_step_positive_rate_deviation": max_step_deviation,
        "max_allowed_step_positive_rate_deviation": 0.10,
        "launch_ready": launch_ready,
    }
    if not launch_ready:
        raise ValueError(f"planned bundle failed launch gates: {verification}")
    return verification


class SourceCache:
    def __init__(self, parent: Path, max_files: int = 128):
        self.parent = parent
        self.max_files = max_files
        self.cache: collections.OrderedDict[str, dict[str, Any]] = (
            collections.OrderedDict()
        )
        self.verified_hashes: set[str] = set()

    def sample(self, context: Context) -> dict[str, Any]:
        relative = context.old_path
        payload = self.cache.pop(relative, None)
        if payload is None:
            path = self.parent / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_hash = str(context.row["output_file_sha256"])
            if relative not in self.verified_hashes:
                actual_hash = sha256_file(path)
                if actual_hash != expected_hash:
                    raise ValueError(
                        f"parent file hash mismatch for {relative}: "
                        f"{actual_hash} != {expected_hash}"
                    )
                self.verified_hashes.add(relative)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"parent payload is not an object: {relative}")
        self.cache[relative] = payload
        while len(self.cache) > self.max_files:
            self.cache.popitem(last=False)
        samples = payload.get("samples")
        if not isinstance(samples, list) or context.old_position >= len(samples):
            raise ValueError(
                f"bad parent sample position {relative}:{context.old_position}"
            )
        sample = samples[context.old_position]
        if not isinstance(sample, dict):
            raise TypeError(
                f"parent sample is not an object: {relative}:{context.old_position}"
            )
        if (
            stable_sample_hash(sample)
            != context.row["destination_sample_semantic_sha256"]
        ):
            raise ValueError(
                f"semantic hash mismatch: {relative}:{context.old_position}"
            )
        if (
            _require_binary_reward(sample.get("reward"), context=relative)
            != context.reward
        ):
            raise ValueError(
                f"sample/manifest reward mismatch: {relative}:{context.old_position}"
            )
        return sample


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_hashes(root: Path) -> None:
    targets = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != HASHES_NAME
    )
    with (root / HASHES_NAME).open("w", encoding="utf-8") as handle:
        for path in targets:
            handle.write(f"{sha256_file(path)}  {path.relative_to(root)}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _materialize(
    parent: Path,
    output: Path,
    summary: dict[str, Any],
    plan: dict[tuple[str, int], list[Bucket]],
    *,
    cache_files: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    cache = SourceCache(parent, max_files=cache_files)
    manifest_rows: list[dict[str, Any]] = []
    try:
        train_rollouts = int(summary["train_rollouts_per_island"])
        for island in range(EXPECTED_NUM_ISLANDS):
            island_dir = staging / f"island_{island}"
            island_dir.mkdir()
            for split, offset in (("train", 0), ("validation", train_rollouts)):
                for local_id, bucket in enumerate(plan[(split, island)]):
                    rollout_id = offset + local_id
                    contexts = bucket.contexts
                    samples = []
                    sample_weights: list[float] = []
                    for group in bucket.groups:
                        weight = bucket.sample_weight(group)
                        for context in group.contexts:
                            sample = dict(cache.sample(context))
                            train_metadata = dict(sample.get("train_metadata") or {})
                            if "value_sample_weight" in train_metadata:
                                raise ValueError(
                                    "source sample already defines value_sample_weight: "
                                    f"{context.uid}"
                                )
                            train_metadata.update(
                                {
                                    "value_sample_weight": weight,
                                    "value_atomic_group_size": group.size,
                                    "value_atomic_group_id": group.stable_id,
                                }
                            )
                            sample["train_metadata"] = train_metadata
                            samples.append(sample)
                            sample_weights.append(weight)
                    if not math.isclose(
                        math.fsum(sample_weights),
                        bucket.size,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise AssertionError(
                            "value sample weights do not preserve batch normalization"
                        )
                    output_path = island_dir / f"data_{rollout_id}.pt"
                    torch.save(
                        {"rollout_id": rollout_id, "samples": samples}, output_path
                    )
                    output_hash = sha256_file(output_path)
                    for position, context in enumerate(contexts):
                        row = dict(context.row)
                        row.update(
                            {
                                "schema_version": 2,
                                "island_id": island,
                                "rollout_id": rollout_id,
                                "output_path": str(output_path.relative_to(staging)),
                                "output_file_sha256": output_hash,
                                "output_position": position,
                                "parent_output_path": context.old_path,
                                "parent_output_position": context.old_position,
                                "value_sample_weight": sample_weights[position],
                            }
                        )
                        manifest_rows.append(row)

        summary = dict(summary)
        summary["dataset_version"] = output.name
        verification = dict(summary["verification"])
        expected_parent_files = {
            context.old_path
            for buckets in plan.values()
            for bucket in buckets
            for context in bucket.contexts
        }
        expected_output_files = EXPECTED_NUM_ISLANDS * (
            train_rollouts + int(summary["validation_rollouts_per_island"])
        )
        verification.update(
            {
                "materialized_contexts": len(manifest_rows),
                "verified_parent_files": len(cache.verified_hashes),
                "expected_parent_files": len(expected_parent_files),
                "materialized_output_files": expected_output_files,
            }
        )
        if cache.verified_hashes != expected_parent_files:
            raise ValueError("not every referenced parent file was hash-verified")
        if len(manifest_rows) != int(verification["parent_contexts"]):
            raise ValueError("materialized context count differs from the parent")
        summary["verification"] = verification
        manifest = {
            "schema_version": 3,
            "dataset_version": output.name,
            "strategy": OUTPUT_STRATEGY,
            "seed": summary["seed"],
            "sync_window_size": summary["sync_window_size"],
            "num_islands": EXPECTED_NUM_ISLANDS,
            "max_sequence_length": 262144,
            "max_contexts_per_rollout_file": MAX_CONTEXTS_PER_BUCKET,
            "train_rollout_ids": list(range(train_rollouts)),
            "validation_rollout_ids": list(
                range(
                    train_rollouts,
                    train_rollouts + int(summary["validation_rollouts_per_island"]),
                )
            ),
            "parent": summary["parent"],
            "critic_recipe": summary["critic_recipe"],
            "islands": summary["islands"],
            "verification": verification,
        }
        with (staging / ROWS_NAME).open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _write_json(staging / SUMMARY_NAME, summary)
        _write_json(staging / VERIFICATION_NAME, verification)
        _write_json(staging / MANIFEST_NAME, manifest)
        tooling = staging / "tooling"
        tooling.mkdir()
        shutil.copy2(Path(__file__), tooling / Path(__file__).name)
        _write_hashes(staging)
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_bundle(
    parent: Path,
    output: Path,
    *,
    train_rollouts: int,
    validation_rollouts: int,
    window_size: int,
    seed: int,
    cache_files: int,
    plan_only: bool,
) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    if not parent.is_dir():
        raise FileNotFoundError(f"parent bundle does not exist: {parent}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if parent == output or parent in output.parents or output in parent.parents:
        raise ValueError("parent and output trees must be disjoint")
    summary, plan, _ = plan_bundle(
        parent,
        train_rollouts=train_rollouts,
        validation_rollouts=validation_rollouts,
        window_size=window_size,
        seed=seed,
    )
    summary["dataset_version"] = output.name
    if plan_only:
        return summary
    return _materialize(
        parent,
        output,
        summary,
        plan,
        cache_files=cache_files,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rollouts", type=int, default=240)
    parser.add_argument("--validation-rollouts", type=int, default=24)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--cache-files", type=int, default=128)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("train_rollouts", "validation_rollouts", "window_size", "cache_files"):
        value = getattr(args, name)
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    result = build_bundle(
        args.parent,
        args.output,
        train_rollouts=args.train_rollouts,
        validation_rollouts=args.validation_rollouts,
        window_size=args.window_size,
        seed=args.seed,
        cache_files=args.cache_files,
        plan_only=args.plan_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
