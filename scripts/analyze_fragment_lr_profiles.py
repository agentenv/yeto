#!/usr/bin/env python3
"""Evaluate fixed per-fragment outer-LR profiles from replayed LR actions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ACTIONS = (25, 35, 40, 50, 60, 75, 90)
BASELINE_ACTION = 50
PROFILES = {
    "fragment_extremes": (75, 50, 25, 50),
    "fragment_smooth": (75, 50, 25, 40),
    "fragment_monotone": (75, 50, 35, 35),
}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no replay records")
    return rows


def _same(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-8)
    return left == right


def merge_records(paths: list[Path]) -> list[dict]:
    merged: dict[tuple[int, int, int], dict] = {}
    for path in paths:
        for row in _read_jsonl(path):
            key = (int(row["seed"]), int(row["step"]), int(row["fragment"]))
            target = merged.setdefault(key, {})
            for field, value in row.items():
                if field in target and not _same(target[field], value):
                    raise SystemExit(
                        f"{path}: conflicting {field} for seed/step/fragment={key}"
                    )
                target[field] = value
    records = [merged[key] for key in sorted(merged)]
    required = [f"current_outer_lr{action}_utility" for action in ACTIONS]
    for field in required:
        missing = sum(field not in row for row in records)
        if missing:
            raise SystemExit(f"{field} missing from {missing} merged records")
    return records


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def _quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    position = p * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _action_field(action: int, suffix: str) -> str:
    return f"current_outer_lr{action}_{suffix}"


def evaluate_profile(rows: list[dict], profile: tuple[int, ...]) -> dict:
    utilities = []
    baseline_utilities = []
    token_utilities = []
    negative = []
    strict_negative = []
    gains = []
    for row in rows:
        fragment = int(row["fragment"])
        if fragment >= len(profile):
            raise SystemExit(
                f"profile has {len(profile)} rates but record uses fragment {fragment}"
            )
        action = profile[fragment]
        utility = float(row[_action_field(action, "utility")])
        baseline = float(row[_action_field(BASELINE_ACTION, "utility")])
        utilities.append(utility)
        baseline_utilities.append(baseline)
        token_utilities.append(float(row["token_weighted_utility"]))
        negative.append(bool(row[_action_field(action, "negative")]))
        strict = row.get(_action_field(action, "strict_negative"))
        if strict is not None:
            strict_negative.append(bool(strict))
        gains.append(utility - baseline)
    baseline_negative = _mean(
        row[_action_field(BASELINE_ACTION, "negative")] for row in rows
    )
    baseline_strict = _mean(
        row[_action_field(BASELINE_ACTION, "strict_negative")]
        for row in rows
        if row.get(_action_field(BASELINE_ACTION, "strict_negative")) is not None
    )
    negative_rate = _mean(negative)
    strict_rate = _mean(strict_negative)
    return {
        "records": len(rows),
        "profile": list(profile),
        "mean_utility": _mean(utilities),
        "mean_gain_vs_fixed_lr50": _mean(gains),
        "mean_gain_vs_global_lr035": _mean(
            utility - token for utility, token in zip(utilities, token_utilities)
        ),
        "gain_positive_rate_vs_fixed_lr50": _mean(gain > 0.0 for gain in gains),
        "negative_rate": negative_rate,
        "negative_rate_relative_drop_vs_fixed_lr50": (
            None
            if baseline_negative <= 0.0
            else (baseline_negative - negative_rate) / baseline_negative
        ),
        "strict_negative_rate": strict_rate,
        "strict_negative_rate_relative_drop_vs_fixed_lr50": (
            None
            if baseline_strict <= 0.0
            else (baseline_strict - strict_rate) / baseline_strict
        ),
    }


def _bootstrap(
    by_seed: dict[int, list[dict]],
    profile: tuple[int, ...],
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
        seed_gains = []
        for blocks in blocks_by_seed.values():
            sampled = [rng.choice(blocks) for _ in range(len(blocks))]
            rows = [row for block in sampled for row in block]
            seed_gains.append(evaluate_profile(rows, profile)["mean_gain_vs_fixed_lr50"])
        samples.append(_mean(seed_gains))
    return {
        "low": _quantile(samples, 0.025),
        "high": _quantile(samples, 0.975),
        "positive_probability": _mean(sample > 0.0 for sample in samples),
    }


def _fragment_table(by_seed: dict[int, list[dict]]) -> dict:
    table = {}
    fragments = sorted({int(row["fragment"]) for rows in by_seed.values() for row in rows})
    for fragment in fragments:
        table[str(fragment)] = {}
        for seed, rows in sorted(by_seed.items()):
            subset = [row for row in rows if int(row["fragment"]) == fragment]
            means = {
                str(action): _mean(row[_action_field(action, "utility")] for row in subset)
                for action in ACTIONS
            }
            best = max(means, key=means.get)
            table[str(fragment)][str(seed)] = {
                "records": len(subset),
                "mean_utility_by_action": means,
                "best_action": int(best),
            }
    return table


def analyze(
    records: list[dict],
    *,
    profiles: dict[str, tuple[int, ...]],
    block_size: int,
    replicates: int,
    bootstrap_seed: int,
) -> dict:
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        by_seed[int(row["seed"])].append(row)
    results = {}
    for name, profile in profiles.items():
        per_seed = {
            str(seed): evaluate_profile(rows, profile)
            for seed, rows in sorted(by_seed.items())
        }
        gains = [row["mean_gain_vs_fixed_lr50"] for row in per_seed.values()]
        results[name] = {
            "profile": list(profile),
            "seed_balanced_mean_gain_vs_fixed_lr50": _mean(gains),
            "all_seeds_positive": all(gain > 0.0 for gain in gains),
            "per_seed": per_seed,
            "paired_block_bootstrap_95": _bootstrap(
                by_seed,
                profile,
                block_size=block_size,
                replicates=replicates,
                seed=bootstrap_seed,
            ),
        }
    best = max(results, key=lambda name: results[name]["seed_balanced_mean_gain_vs_fixed_lr50"])
    return {
        "schema": "fragment_lr_profile_analysis_v1",
        "records": len(records),
        "seeds": sorted(by_seed),
        "baseline_action": BASELINE_ACTION,
        "profiles": results,
        "fragment_action_table": _fragment_table(by_seed),
        "best_profile": best,
        "gate": {
            "best_all_seeds_positive": results[best]["all_seeds_positive"],
            "best_bootstrap_positive_probability_ge_0.95": (
                results[best]["paired_block_bootstrap_95"]["positive_probability"] >= 0.95
            ),
        },
    }


def _fmt(value, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    value = float(value)
    if value != 0.0 and abs(value) < 0.001:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown(result: dict) -> str:
    lines = ["# Fragment-Specific Outer-LR Profiles", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Fixed baseline action: `lr{result['baseline_action']}`")
    lines.append(f"- Best profile: `{result['best_profile']}`")
    lines.append("")
    lines.append("| Profile | Fragment actions | Mean gain vs fixed | Seed gains | All seeds + | Bootstrap 95% | P(gain>0) |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for name, row in result["profiles"].items():
        ci = row["paired_block_bootstrap_95"]
        seed_gains = ", ".join(
            f"{seed}:{_fmt(seed_row['mean_gain_vs_fixed_lr50'])}"
            for seed, seed_row in row["per_seed"].items()
        )
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | [{}, {}] | {} |".format(
                name,
                ",".join(str(value) for value in row["profile"]),
                _fmt(row["seed_balanced_mean_gain_vs_fixed_lr50"]),
                seed_gains,
                row["all_seeds_positive"],
                _fmt(ci["low"]),
                _fmt(ci["high"]),
                _fmt(ci["positive_probability"], 3),
            )
        )
    lines.append("")
    lines.append("## Per-Fragment Best Actions")
    lines.append("")
    lines.append("| Fragment | " + " | ".join(f"Seed {seed}" for seed in result["seeds"]) + " |")
    lines.append("|---:|" + "---:|" * len(result["seeds"]))
    for fragment, seed_rows in result["fragment_action_table"].items():
        lines.append(
            "| {} | {} |".format(
                fragment,
                " | ".join(str(seed_rows[str(seed)]["best_action"]) for seed in result["seeds"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_profiles(specs: list[str]) -> dict[str, tuple[int, ...]]:
    profiles = dict(PROFILES)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid --profile {spec!r}; expected name=a,b,c,d")
        name, raw = spec.split("=", 1)
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
        if not name or not values or any(value not in ACTIONS for value in values):
            raise SystemExit(f"invalid --profile {spec!r}")
        profiles[name] = values
    return profiles


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", nargs="+", required=True, type=Path)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260709)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.block_size < 1 or args.bootstrap_replicates < 1:
        parser.error("block size and bootstrap replicates must be positive")
    args.profiles = parse_profiles(args.profile)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    result = analyze(
        merge_records(args.replays),
        profiles=args.profiles,
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
