#!/usr/bin/env python3
"""Aggregate partitioned production-faithful buffered replay records."""

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
    path = REPO_ROOT / "scripts" / "replay_buffered_nesterov_syncer.py"
    spec = importlib.util.spec_from_file_location("replay_buffered_nesterov_syncer", path)
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
    path_keys = []
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
        path_keys.append((path, keys))
    expected = set(merged)
    for path, keys in path_keys:
        if keys != {key for key in expected if key[0] in {item[0] for item in keys}}:
            missing = sorted(expected - keys)[:5]
            extra = sorted(keys - expected)[:5]
            raise SystemExit(f"{path}: replay key mismatch; missing={missing}, extra={extra}")
    return [merged[key] for key in sorted(merged)]


def _policies(records: list[dict]) -> tuple[str, ...]:
    found = set()
    for field in records[0]:
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


def _mean(values: list[float]) -> float:
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
    seed_blocks = {}
    for experiment_seed, rows in by_seed.items():
        rows = sorted(rows, key=lambda row: (int(row["step"]), int(row["fragment"])))
        seed_blocks[experiment_seed] = [
            rows[offset : offset + block_size]
            for offset in range(0, len(rows), block_size)
        ]
    samples = []
    for _ in range(replicates):
        seed_means = []
        for blocks in seed_blocks.values():
            sampled = [rng.choice(blocks) for _ in range(len(blocks))]
            gains = [
                float(row[f"{policy}_gain_vs_token"])
                for block in sampled
                for row in block
            ]
            seed_means.append(_mean(gains))
        samples.append(_mean(seed_means))
    return {
        "low": _quantile(samples, 0.025),
        "high": _quantile(samples, 0.975),
        "positive_probability": _mean([1.0 if value > 0.0 else 0.0 for value in samples]),
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
        ci = None if policy == "token_weighted" else _bootstrap_gain(
            by_seed,
            policy,
            block_size=block_size,
            replicates=replicates,
            seed=bootstrap_seed,
        )
        results[policy] = {
            "seed_balanced_mean_gain_vs_token": _mean(gains),
            "all_seeds_positive": all(gain > 0.0 for gain in gains),
            "seeds_positive": sum(1 for gain in gains if gain > 0.0),
            "seed_count": len(gains),
            "per_seed_gain": {
                str(seed): per_seed[seed]["policies"][policy]["mean_gain_vs_token"]
                for seed in per_seed
            },
            "negative_rate_relative_drop": _mean([
                row["negative_rate_relative_drop"] or 0.0 for row in seed_rows
            ]),
            "strict_negative_rate_relative_drop": _mean([
                row["strict_negative_rate_relative_drop"] or 0.0 for row in seed_rows
            ]),
            "selected_mass_mean": _mean([row["selected_mass_mean"] for row in seed_rows]),
            "fresh_effective_share": _mean([row["fresh_effective_share"] for row in seed_rows]),
            "history_effective_share": _mean([row["history_effective_share"] for row in seed_rows]),
            "paired_block_bootstrap_95": ci,
        }
    non_token = [policy for policy in policies if policy != "token_weighted"]
    best = max(non_token, key=lambda policy: results[policy]["seed_balanced_mean_gain_vs_token"])
    validation = [summary["next_state_validation"] for summary in per_seed.values()]
    return {
        "schema": "buffered_nesterov_aggregate_v1",
        "records": len(records),
        "seeds": sorted(per_seed),
        "records_per_seed": {str(seed): len(rows) for seed, rows in by_seed.items()},
        "policies": results,
        "best_non_token_policy": best,
        "next_state_validation": {
            "max_step_relative_error": max(
                float(row["max_step_relative_error"])
                for row in validation if row["max_step_relative_error"] is not None
            ),
            "all_seed_gates_pass": all(
                summary["gate"]["baseline_replay_matches_next_state"]
                for summary in per_seed.values()
            ),
        },
        "gate": {
            "best_all_seeds_positive": results[best]["all_seeds_positive"],
            "best_negative_drop_nonnegative": results[best]["negative_rate_relative_drop"] >= 0.0,
            "best_strict_drop_nonnegative": results[best]["strict_negative_rate_relative_drop"] >= 0.0,
            "best_selected_mass_ge_0.95": results[best]["selected_mass_mean"] >= 0.95,
            "best_bootstrap_positive_probability_ge_0.95": (
                results[best]["paired_block_bootstrap_95"]["positive_probability"] >= 0.95
            ),
        },
        "per_seed": per_seed,
    }


def _fmt(value, digits=6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not math.isfinite(value):
        return "n/a"
    if value != 0.0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown(result: dict) -> str:
    lines = ["# Buffered Nesterov Replay Aggregate", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Best non-token policy: `{result['best_non_token_policy']}`")
    lines.append(
        "- Baseline max next-state step-relative error: "
        f"`{_fmt(result['next_state_validation']['max_step_relative_error'])}`"
    )
    lines.append("")
    lines.append("| Policy | Mean gain | Seed gains | Neg drop | Strict drop | Mass | Fresh/history | Bootstrap 95% | P(gain>0) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for policy, row in result["policies"].items():
        ci = row["paired_block_bootstrap_95"]
        interval = "n/a" if ci is None else f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]"
        probability = "n/a" if ci is None else _fmt(ci["positive_probability"], 3)
        seed_gains = ", ".join(
            f"{seed}:{_fmt(gain)}" for seed, gain in row["per_seed_gain"].items()
        )
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {}/{} | {} | {} |".format(
                policy,
                _fmt(row["seed_balanced_mean_gain_vs_token"]),
                seed_gains,
                _fmt(row["negative_rate_relative_drop"], 3),
                _fmt(row["strict_negative_rate_relative_drop"], 3),
                _fmt(row["selected_mass_mean"], 3),
                _fmt(row["fresh_effective_share"], 2),
                _fmt(row["history_effective_share"], 2),
                interval,
                probability,
            )
        )
    lines.extend(["", "## Gates", ""])
    for name, passed in result["gate"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", nargs="+", required=True, type=Path)
    parser.add_argument("--expected-seeds", default=None)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260709)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = merge_records(args.replays)
    if args.expected_seeds:
        expected = sorted(int(seed) for seed in args.expected_seeds.split(","))
        actual = sorted({int(row["seed"]) for row in records})
        if actual != expected:
            raise SystemExit(f"expected seeds {expected}, got {actual}")
    result = aggregate(
        records,
        block_size=args.block_size,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(replay._jsonable(result), indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
