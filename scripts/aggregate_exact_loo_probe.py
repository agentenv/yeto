#!/usr/bin/env python3
"""Aggregate completed exact leave-one-out probe replay shards."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "replay_exact_loo_probe", ROOT / "scripts" / "replay_exact_loo_probe.py"
)
REPLAY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


AGGREGATE_SCHEMA = "exact_loo_probe_aggregate_v2"
GATE_MAX_NEXT_STATE_STEP_RELATIVE_ERROR = 1e-4
GATE_MIN_MEAN_GAIN = 0.0005
GATE_MIN_NEGATIVE_DROP = 0.20
GATE_MIN_STRICT_NEGATIVE_DROP = 0.20
GATE_MIN_ACTION_RATE = 0.05
GATE_MAX_ACTION_RATE = 0.50


def read_jsonl(path: Path) -> list[dict]:
    """Preserve the old helper interface while using strict JSON parsing."""

    return REPLAY.read_jsonl(path)


def _artifact(path: Path) -> dict[str, Any]:
    try:
        records, completion = REPLAY.read_replay_artifact(path)
        if completion is None:
            raise ValueError("missing terminal completion record")
        REPLAY.validate_completion_artifact(records, completion)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{path}: invalid replay artifact: {exc}") from exc
    if completion.get("full_shard_complete") is not True:
        raise ValueError(
            f"{path}: replay completed only a max-groups subset, not its full shard"
        )
    return {
        "path": path,
        "sha256": REPLAY._sha256_file(path),
        "records": records,
        "completion": completion,
    }


def _one_value(values: Sequence[Any], *, context: str) -> Any:
    normalized = list(values)
    if not normalized or any(value != normalized[0] for value in normalized[1:]):
        raise ValueError(f"{context} are not compatible")
    return normalized[0]


def _validate_seed_coverage(
    seed: int,
    artifacts: Sequence[Mapping[str, Any]],
    *,
    expected_groups_per_seed: int | None,
) -> dict[str, Any]:
    completions = [artifact["completion"] for artifact in artifacts]
    capture_digest = _one_value(
        [completion["capture_config_sha256"] for completion in completions],
        context=f"seed {seed} capture configs",
    )
    capture_ids = list(
        _one_value(
            [completion["capture_group_ids"] for completion in completions],
            context=f"seed {seed} capture group coverage",
        )
    )
    coordinates = list(
        _one_value(
            [completion["capture_group_coordinates"] for completion in completions],
            context=f"seed {seed} capture coordinates",
        )
    )
    if len(capture_ids) != len(set(capture_ids)):
        raise ValueError(f"seed {seed} capture metadata contains duplicate group IDs")
    if len(coordinates) != len(capture_ids):
        raise ValueError(f"seed {seed} capture coordinates do not match group count")
    coordinate_keys = [
        (int(item["step"]), int(item["fragment"])) for item in coordinates
    ]
    if len(coordinate_keys) != len(set(coordinate_keys)):
        raise ValueError(f"seed {seed} capture metadata contains duplicate coordinates")
    if (
        expected_groups_per_seed is not None
        and len(capture_ids) != expected_groups_per_seed
    ):
        raise ValueError(
            f"seed {seed} has {len(capture_ids)} captured groups, expected "
            f"{expected_groups_per_seed}"
        )
    if expected_groups_per_seed is not None:
        steps = sorted(step for step, _ in coordinate_keys)
        expected_steps = list(range(1, expected_groups_per_seed + 1))
        if steps != expected_steps:
            missing = sorted(set(expected_steps) - set(steps))
            extra = sorted(set(steps) - set(expected_steps))
            raise ValueError(
                f"seed {seed} step coverage mismatch: missing={missing}, extra={extra}"
            )

    strides = {int(completion["group_shard"]["stride"]) for completion in completions}
    if len(strides) != 1:
        raise ValueError(f"seed {seed} shards use different strides: {sorted(strides)}")
    stride = next(iter(strides))
    starts = [int(completion["group_shard"]["start"]) for completion in completions]
    if len(starts) != len(set(starts)):
        raise ValueError(f"seed {seed} has duplicate shard starts {sorted(starts)}")
    expected_starts = list(range(stride))
    if sorted(starts) != expected_starts:
        missing = sorted(set(expected_starts) - set(starts))
        extra = sorted(set(starts) - set(expected_starts))
        raise ValueError(
            f"seed {seed} shard coverage mismatch: missing starts={missing}, extra={extra}"
        )

    owner: dict[str, Path] = {}
    records_by_id: dict[str, dict] = {}
    for artifact in artifacts:
        completion = artifact["completion"]
        start = int(completion["group_shard"]["start"])
        if completion["group_shard"].get("max_groups") is not None:
            raise ValueError(
                f"{artifact['path']}: completed artifact used --max-groups"
            )
        expected_for_start = capture_ids[start::stride]
        full_shard_ids = list(completion["full_shard_group_ids"])
        expected_ids = list(completion["expected_group_ids"])
        if full_shard_ids != expected_for_start or expected_ids != expected_for_start:
            raise ValueError(
                f"{artifact['path']}: shard group IDs do not match start={start}, stride={stride}"
            )
        for record in artifact["records"]:
            identifier = str(record["group_id"])
            if identifier in owner:
                raise ValueError(
                    f"duplicate replay coverage for seed {seed} group {identifier}: "
                    f"{owner[identifier]} and {artifact['path']}"
                )
            owner[identifier] = artifact["path"]
            records_by_id[identifier] = record

    covered_ids = set(owner)
    expected_id_set = set(capture_ids)
    if covered_ids != expected_id_set:
        missing = [
            identifier for identifier in capture_ids if identifier not in covered_ids
        ]
        extra = sorted(covered_ids - expected_id_set)
        raise ValueError(
            f"seed {seed} replay group coverage mismatch: missing={missing}, extra={extra}"
        )
    ordered_records = [records_by_id[identifier] for identifier in capture_ids]
    for record in ordered_records:
        available = bool(record.get("production_baseline_next_state_available"))
        error = record.get("production_baseline_next_state_step_relative_error")
        if available:
            try:
                error = float(error)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"seed {seed} has a malformed next-state parity error"
                ) from exc
            if not math.isfinite(error) or error < 0.0:
                raise ValueError(
                    f"seed {seed} has a non-finite next-state parity error"
                )
        elif error is not None:
            raise ValueError(
                f"seed {seed} marks next-state parity unavailable but records an error"
            )
    parity_records = sum(
        bool(record.get("production_baseline_next_state_available"))
        for record in ordered_records
    )
    expected_parity_records = max(len(capture_ids) - 1, 0)
    if parity_records != expected_parity_records:
        raise ValueError(
            f"seed {seed} has {parity_records} next-state parity checks, expected "
            f"{expected_parity_records} from complete capture coverage"
        )
    return {
        "seed": seed,
        "capture_config_sha256": capture_digest,
        "capture_group_count": len(capture_ids),
        "capture_group_ids_sha256": REPLAY._ordered_id_digest(capture_ids),
        "capture_group_coordinates": coordinates,
        "shard_stride": stride,
        "shard_starts": sorted(starts),
        "next_state_parity_records": parity_records,
        "records": ordered_records,
    }


def _offline_gate(summary: Mapping[str, Any], expected_seeds: Sequence[int]) -> dict:
    per_seed = summary["per_seed"]
    max_error = summary["next_state_validation"]["max_step_relative_error"]
    mean_gain = summary["mean_gain_vs_baseline"]
    negative_drop = summary["negative_rate_relative_drop"]
    strict_drop = summary["strict_negative_rate_relative_drop"]
    action_rate = summary["action_rate"]
    random_gain = summary["mean_random_loo_gain"]
    checks = {
        "production_baseline_next_state_max_step_relative_error_lt_1e-4": (
            max_error is not None
            and float(max_error) < GATE_MAX_NEXT_STATE_STEP_RELATIVE_ERROR
        ),
        "all_seeds_positive_held_out_gain": all(
            str(seed) in per_seed
            and float(per_seed[str(seed)]["mean_gain_vs_baseline"]) > 0.0
            for seed in expected_seeds
        ),
        "mean_held_out_gain_ge_0.0005": (
            mean_gain is not None and float(mean_gain) >= GATE_MIN_MEAN_GAIN
        ),
        "negative_merge_relative_drop_ge_0.20": (
            negative_drop is not None and float(negative_drop) >= GATE_MIN_NEGATIVE_DROP
        ),
        "strict_negative_relative_drop_ge_0.20": (
            strict_drop is not None
            and float(strict_drop) >= GATE_MIN_STRICT_NEGATIVE_DROP
        ),
        "action_rate_between_0.05_and_0.50": (
            action_rate is not None
            and GATE_MIN_ACTION_RATE <= float(action_rate) <= GATE_MAX_ACTION_RATE
        ),
        "beats_deterministic_random_valid_loo": (
            mean_gain is not None
            and random_gain is not None
            and float(mean_gain) > float(random_gain)
        ),
    }
    return {
        "predeclared": True,
        "thresholds": {
            "max_next_state_step_relative_error": (
                GATE_MAX_NEXT_STATE_STEP_RELATIVE_ERROR
            ),
            "min_mean_held_out_gain": GATE_MIN_MEAN_GAIN,
            "min_negative_merge_relative_drop": GATE_MIN_NEGATIVE_DROP,
            "min_strict_negative_relative_drop": GATE_MIN_STRICT_NEGATIVE_DROP,
            "min_action_rate": GATE_MIN_ACTION_RATE,
            "max_action_rate": GATE_MAX_ACTION_RATE,
            "random_control": "mean chosen gain strictly exceeds deterministic random valid-LOO gain",
        },
        "metrics": {
            "max_next_state_step_relative_error": max_error,
            "per_seed_mean_held_out_gain": {
                str(seed): per_seed[str(seed)]["mean_gain_vs_baseline"]
                for seed in expected_seeds
            },
            "mean_held_out_gain": mean_gain,
            "negative_merge_relative_drop": negative_drop,
            "strict_negative_relative_drop": strict_drop,
            "action_rate": action_rate,
            "mean_random_valid_loo_gain": random_gain,
            "chosen_minus_random_gain": (
                None
                if mean_gain is None or random_gain is None
                else float(mean_gain) - float(random_gain)
            ),
        },
        "checks": checks,
        "gate_pass": all(checks.values()),
    }


def aggregate_completed_artifacts(
    replay_paths: Sequence[Path],
    *,
    expected_seeds: Sequence[int],
    expected_groups_per_seed: int | None,
) -> dict[str, Any]:
    if not replay_paths:
        raise ValueError("at least one replay artifact is required")
    resolved_paths = [path.expanduser().resolve() for path in replay_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("the same replay artifact was passed more than once")
    expected = sorted(int(seed) for seed in expected_seeds)
    if len(expected) != len(set(expected)):
        raise ValueError("--expected-seeds contains duplicates")
    artifacts = [_artifact(path) for path in resolved_paths]
    found = sorted({int(artifact["completion"]["seed"]) for artifact in artifacts})
    if found != expected:
        missing = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        raise ValueError(
            f"seed mismatch: missing={missing}, extra={extra}, found={found}"
        )

    compatibility_digest = _one_value(
        [
            artifact["completion"]["compatibility_config_sha256"]
            for artifact in artifacts
        ],
        context="replay compatibility config digests",
    )
    compatibility_config = _one_value(
        [artifact["completion"]["compatibility_config"] for artifact in artifacts],
        context="replay compatibility configs",
    )
    if REPLAY._config_sha256(compatibility_config) != compatibility_digest:
        raise ValueError("shared compatibility config digest is invalid")

    by_seed: dict[int, list[dict]] = defaultdict(list)
    for artifact in artifacts:
        by_seed[int(artifact["completion"]["seed"])].append(artifact)
    coverage = {
        seed: _validate_seed_coverage(
            seed,
            by_seed[seed],
            expected_groups_per_seed=expected_groups_per_seed,
        )
        for seed in expected
    }
    coordinate_sets = [coverage[seed]["capture_group_coordinates"] for seed in expected]
    _one_value(coordinate_sets, context="cross-seed expected group coordinates")

    records = [record for seed in expected for record in coverage[seed]["records"]]
    seen_record_keys = set()
    for record in records:
        key = (int(record["seed"]), str(record["group_id"]))
        if key in seen_record_keys:
            raise ValueError(f"duplicate replay record {key}")
        seen_record_keys.add(key)

    result = REPLAY.summarize(records)
    result["schema"] = AGGREGATE_SCHEMA
    result["all_seeds_positive"] = all(
        result["per_seed"][str(seed)]["mean_gain_vs_baseline"] > 0.0
        for seed in expected
    )
    result["compatibility_config"] = compatibility_config
    result["compatibility_config_sha256"] = compatibility_digest
    result["coverage"] = {
        "expected_groups_per_seed": expected_groups_per_seed,
        "per_seed": {
            str(seed): {
                key: value for key, value in coverage[seed].items() if key != "records"
            }
            for seed in expected
        },
    }
    result["offline_gate"] = _offline_gate(result, expected)
    result["source_files"] = [str(path) for path in resolved_paths]
    result["source_artifacts"] = [
        {
            "path": str(artifact["path"]),
            "sha256": artifact["sha256"],
            "seed": int(artifact["completion"]["seed"]),
            "group_shard": artifact["completion"]["group_shard"],
            "records": len(artifact["records"]),
        }
        for artifact in artifacts
    ]
    return result


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            REPLAY.jsonable(dict(value)), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", nargs="+", required=True, type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--expected-groups-per-seed",
        type=int,
        default=REPLAY.DEFAULT_EXPECTED_GROUPS,
    )
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.expected_groups_per_seed < 1:
        parser.error("--expected-groups-per-seed must be positive")
    try:
        result = aggregate_completed_artifacts(
            args.replays,
            expected_seeds=args.expected_seeds,
            expected_groups_per_seed=args.expected_groups_per_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _atomic_write(args.out_json, result)
    print(
        json.dumps(REPLAY.jsonable(result), indent=2, sort_keys=True, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
