#!/usr/bin/env python3
"""Merge partitioned anchor-gradient replays and summarize them across seeds."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_replay():
    path = REPO_ROOT / "scripts" / "replay_anchor_gradient_syncer.py"
    spec = importlib.util.spec_from_file_location("replay_anchor_gradient_syncer", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


replay = _load_replay()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no records")
    return rows


def _same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)
    return left == right


def merge_records(paths: list[Path]) -> list[dict]:
    merged: dict[tuple[int, int, int], dict] = {}
    keys_by_path = []
    for path in paths:
        keys = set()
        for row in _read_jsonl(path):
            key = (int(row["seed"]), int(row["step"]), int(row["fragment"]))
            keys.add(key)
            target = merged.setdefault(key, {})
            for field, value in row.items():
                if field in target and not _same(target[field], value):
                    raise SystemExit(
                        f"{path}: conflicting {field} for seed/step/fragment={key}: "
                        f"{target[field]!r} != {value!r}"
                    )
                target[field] = value
        keys_by_path.append((path, keys))
    seeds_by_path = [{key[0] for key in keys} for _, keys in keys_by_path]
    for (path, keys), path_seeds in zip(keys_by_path, seeds_by_path):
        expected = {key for key in merged if key[0] in path_seeds}
        if keys != expected:
            missing = sorted(expected - keys)[:5]
            extra = sorted(keys - expected)[:5]
            raise SystemExit(f"{path}: replay key mismatch; missing={missing}, extra={extra}")
    return [merged[key] for key in sorted(merged)]


def _policies(records: list[dict]) -> tuple[str, ...]:
    found = set()
    for row in records:
        for field in row:
            match = re.fullmatch(r"(.+)_gain_vs_token", field)
            if match:
                found.add(match.group(1))
    found.discard("token_weighted")
    policies = ("token_weighted", *sorted(found))
    for policy in policies:
        missing = [row for row in records if f"{policy}_utility" not in row]
        if missing:
            raise SystemExit(f"policy {policy} is missing from {len(missing)} merged records")
    return policies


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def _quantile(values: list[float], p: float) -> float:
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return float("nan")
    position = p * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _bootstrap_gain(
    by_seed: dict[int, list[dict]],
    policy: str,
    *,
    block_size: int,
    replicates: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    blocks_by_seed = {}
    for experiment_seed, rows in by_seed.items():
        rows = sorted(rows, key=lambda row: (int(row["step"]), int(row["fragment"])))
        blocks_by_seed[experiment_seed] = [
            rows[offset : offset + block_size]
            for offset in range(0, len(rows), block_size)
        ]
    samples = []
    for _ in range(replicates):
        seed_means = []
        for blocks in blocks_by_seed.values():
            sampled = [rng.choice(blocks) for _ in range(len(blocks))]
            seed_means.append(
                _mean(
                    row[f"{policy}_gain_vs_token"]
                    for block in sampled
                    for row in block
                )
            )
        samples.append(_mean(seed_means))
    return {
        "low": _quantile(samples, 0.025),
        "high": _quantile(samples, 0.975),
        "positive_probability": _mean(value > 0.0 for value in samples),
    }


def aggregate(records: list[dict], *, block_size: int, replicates: int, bootstrap_seed: int) -> dict:
    policies = _policies(records)
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        by_seed[int(row["seed"])].append(row)
    per_seed = {
        seed: replay.summarize(rows, policies)
        for seed, rows in sorted(by_seed.items())
    }
    results = {}
    for policy in policies:
        seed_rows = [summary["policies"][policy] for summary in per_seed.values()]
        gains = [row["mean_gain_vs_token"] for row in seed_rows]
        ci = None
        if policy != "token_weighted":
            ci = _bootstrap_gain(
                by_seed,
                policy,
                block_size=block_size,
                replicates=replicates,
                seed=bootstrap_seed,
            )
        results[policy] = {
            "seed_balanced_mean_gain_vs_token": _mean(gains),
            "all_seeds_positive": all(gain > 0.0 for gain in gains),
            "per_seed_gain": {
                str(seed): per_seed[seed]["policies"][policy]["mean_gain_vs_token"]
                for seed in per_seed
            },
            "negative_rate_relative_drop": _mean(
                row["negative_rate_relative_drop"] or 0.0 for row in seed_rows
            ),
            "strict_negative_rate_relative_drop": _mean(
                row["strict_negative_rate_relative_drop"] or 0.0 for row in seed_rows
            ),
            "delta_norm_ratio": _mean(row["delta_norm_ratio"] for row in seed_rows),
            "delta_cosine_to_baseline": _mean(
                row["delta_cosine_to_baseline"] for row in seed_rows
            ),
            "conflict_tensor_fraction": _mean(
                row["conflict_tensor_fraction"] for row in seed_rows
            ),
            "paired_block_bootstrap_95": ci,
        }
    non_token = [policy for policy in policies if policy != "token_weighted"]
    best = max(non_token, key=lambda policy: results[policy]["seed_balanced_mean_gain_vs_token"])
    return {
        "schema": "anchor_gradient_syncer_aggregate_v1",
        "records": len(records),
        "seeds": sorted(per_seed),
        "records_per_seed": {str(seed): len(rows) for seed, rows in by_seed.items()},
        "policies": results,
        "best_non_token_policy": best,
        "gate": {
            "best_all_seeds_positive": results[best]["all_seeds_positive"],
            "best_negative_drop_nonnegative": results[best]["negative_rate_relative_drop"] >= 0.0,
            "best_strict_drop_nonnegative": (
                results[best]["strict_negative_rate_relative_drop"] >= 0.0
            ),
            "best_bootstrap_positive_probability_ge_0.95": (
                results[best]["paired_block_bootstrap_95"]["positive_probability"] >= 0.95
            ),
        },
        "per_seed": per_seed,
    }


def _fmt(value, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if value != 0.0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown(result: dict) -> str:
    lines = ["# Anchor-Gradient Syncer Replay Aggregate", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Best non-token policy: `{result['best_non_token_policy']}`")
    lines.append("")
    lines.append(
        "| Policy | Mean gain | Seed gains | Neg drop | Strict drop | Norm ratio | Cosine | Conflict tensors | Bootstrap 95% | P(gain>0) |"
    )
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for policy, row in result["policies"].items():
        ci = row["paired_block_bootstrap_95"]
        interval = "n/a" if ci is None else f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]"
        probability = "n/a" if ci is None else _fmt(ci["positive_probability"], 3)
        seed_gains = ", ".join(
            f"{seed}:{_fmt(gain)}" for seed, gain in row["per_seed_gain"].items()
        )
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                policy,
                _fmt(row["seed_balanced_mean_gain_vs_token"]),
                seed_gains,
                _fmt(row["negative_rate_relative_drop"], 3),
                _fmt(row["strict_negative_rate_relative_drop"], 3),
                _fmt(row["delta_norm_ratio"], 3),
                _fmt(row["delta_cosine_to_baseline"], 3),
                _fmt(row["conflict_tensor_fraction"], 3),
                interval,
                probability,
            )
        )
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    for key, value in result["gate"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260709)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.block_size < 1:
        parser.error("--block-size must be >= 1")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates must be >= 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    records = merge_records(args.inputs)
    result = aggregate(
        records,
        block_size=args.block_size,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
