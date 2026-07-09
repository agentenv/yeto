#!/usr/bin/env python3
"""Evaluate disjoint anchor-to-oracle selection over scalar outer-LR actions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_ACTIONS = (
    "current_outer_lr25",
    "current_outer_lr40",
    "current_outer_lr50",
    "current_outer_lr60",
    "current_outer_lr75",
)


def _read_merge(paths: list[Path]) -> dict[tuple[int, int, int], dict]:
    merged: dict[tuple[int, int, int], dict] = {}
    path_keys = []
    for path in paths:
        keys = set()
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (int(row["seed"]), int(row["step"]), int(row["fragment"]))
                keys.add(key)
                target = merged.setdefault(key, {})
                for field, value in row.items():
                    if field not in target:
                        target[field] = value
        path_keys.append((path, keys))
    expected_by_seed: dict[int, set] = defaultdict(set)
    for key in merged:
        expected_by_seed[key[0]].add(key)
    for path, keys in path_keys:
        seeds = {key[0] for key in keys}
        expected = set().union(*(expected_by_seed[seed] for seed in seeds))
        if keys != expected:
            raise SystemExit(f"{path}: key mismatch; got {len(keys)}, expected {len(expected)}")
    return merged


def _mean(values) -> float:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def _strict(utility: float, utility_se: float | None) -> bool | None:
    return None if utility_se is None else utility + utility_se < 0.0


def _rank_concordance(anchor: dict[str, float], oracle: dict[str, float], actions: tuple[str, ...]) -> float:
    agree = 0
    total = 0
    for idx, left in enumerate(actions):
        for right in actions[idx + 1 :]:
            anchor_delta = anchor[left] - anchor[right]
            oracle_delta = oracle[left] - oracle[right]
            if anchor_delta == 0.0 or oracle_delta == 0.0:
                agree += 0.5
            elif (anchor_delta > 0.0) == (oracle_delta > 0.0):
                agree += 1.0
            total += 1
    return agree / max(total, 1)


def _choice(
    anchor_utility: dict[str, float],
    anchor_se: dict[str, float | None],
    actions: tuple[str, ...],
    baseline: str,
    z_margin: float,
) -> str:
    best = max(actions, key=lambda action: anchor_utility[action])
    if best == baseline or z_margin <= 0.0:
        return best
    best_se = anchor_se[best] or 0.0
    baseline_se = anchor_se[baseline] or 0.0
    threshold = z_margin * math.sqrt(best_se * best_se + baseline_se * baseline_se)
    if anchor_utility[best] - anchor_utility[baseline] < threshold:
        return baseline
    return best


def analyze(
    anchor: dict[tuple[int, int, int], dict],
    oracle: dict[tuple[int, int, int], dict],
    actions: tuple[str, ...],
    baseline: str,
    margins: tuple[float, ...],
    random_replicates: int,
    random_seed: int,
) -> dict:
    if set(anchor) != set(oracle):
        raise SystemExit(
            f"anchor/oracle keys differ: anchor={len(anchor)}, oracle={len(oracle)}"
        )
    records = []
    for key in sorted(anchor):
        anchor_row = anchor[key]
        oracle_row = oracle[key]
        anchor_utility = {action: float(anchor_row[f"{action}_utility"]) for action in actions}
        oracle_utility = {action: float(oracle_row[f"{action}_utility"]) for action in actions}
        anchor_se = {action: anchor_row.get(f"{action}_utility_se") for action in actions}
        oracle_se = {action: oracle_row.get(f"{action}_utility_se") for action in actions}
        row = {
            "seed": key[0],
            "step": key[1],
            "fragment": key[2],
            "anchor_utility": anchor_utility,
            "oracle_utility": oracle_utility,
            "anchor_se": anchor_se,
            "oracle_se": oracle_se,
            "anchor_oracle_pairwise_concordance": _rank_concordance(
                anchor_utility, oracle_utility, actions
            ),
            "oracle_best_action": max(actions, key=lambda action: oracle_utility[action]),
        }
        for margin in margins:
            row[f"choice_z{margin:g}"] = _choice(
                anchor_utility, anchor_se, actions, baseline, margin
            )
        records.append(row)

    rng = random.Random(random_seed)
    policies = {}
    for margin in margins:
        name = f"anchor_lr_z{margin:g}"
        chosen = [row[f"choice_z{margin:g}"] for row in records]
        gains = [
            row["oracle_utility"][action] - row["oracle_utility"][baseline]
            for row, action in zip(records, chosen)
        ]
        selected_negative = [row["oracle_utility"][action] < 0.0 for row, action in zip(records, chosen)]
        baseline_negative = [row["oracle_utility"][baseline] < 0.0 for row in records]
        selected_strict = [
            _strict(row["oracle_utility"][action], row["oracle_se"][action])
            for row, action in zip(records, chosen)
        ]
        baseline_strict = [
            _strict(row["oracle_utility"][baseline], row["oracle_se"][baseline])
            for row in records
        ]
        oracle_headroom = [
            max(row["oracle_utility"].values()) - row["oracle_utility"][baseline]
            for row in records
        ]
        random_gains = []
        for _ in range(random_replicates):
            shuffled = chosen[:]
            rng.shuffle(shuffled)
            random_gains.append(
                _mean(
                    row["oracle_utility"][action] - row["oracle_utility"][baseline]
                    for row, action in zip(records, shuffled)
                )
            )
        base_neg = _mean(baseline_negative)
        selected_neg = _mean(selected_negative)
        base_strict = _mean(value for value in baseline_strict if value is not None)
        selected_strict_rate = _mean(value for value in selected_strict if value is not None)
        policy = {
            "mean_gain_vs_fixed_lr": _mean(gains),
            "gain_positive_rate": _mean(gain > 0.0 for gain in gains),
            "negative_rate": selected_neg,
            "negative_rate_relative_drop": (
                None if base_neg <= 0.0 else (base_neg - selected_neg) / base_neg
            ),
            "strict_negative_rate": selected_strict_rate,
            "strict_negative_rate_relative_drop": (
                None
                if base_strict <= 0.0
                else (base_strict - selected_strict_rate) / base_strict
            ),
            "oracle_headroom_captured": (
                None
                if _mean(oracle_headroom) <= 0.0
                else _mean(gains) / _mean(oracle_headroom)
            ),
            "chosen_action_distribution": dict(Counter(chosen)),
            "random_schedule_gain_mean": _mean(random_gains),
            "random_schedule_gain_p95": sorted(random_gains)[int(0.95 * (len(random_gains) - 1))],
        }
        per_seed = {}
        for seed in sorted({row["seed"] for row in records}):
            indices = [idx for idx, row in enumerate(records) if row["seed"] == seed]
            seed_gains = [gains[idx] for idx in indices]
            seed_base_neg = _mean(baseline_negative[idx] for idx in indices)
            seed_selected_neg = _mean(selected_negative[idx] for idx in indices)
            per_seed[str(seed)] = {
                "records": len(indices),
                "mean_gain_vs_fixed_lr": _mean(seed_gains),
                "negative_rate_relative_drop": (
                    None
                    if seed_base_neg <= 0.0
                    else (seed_base_neg - seed_selected_neg) / seed_base_neg
                ),
                "chosen_action_distribution": dict(Counter(chosen[idx] for idx in indices)),
            }
        policy["per_seed"] = per_seed
        policies[name] = policy

    return {
        "schema": "lr_action_probe_analysis_v1",
        "records": len(records),
        "seeds": sorted({row["seed"] for row in records}),
        "actions": list(actions),
        "fixed_lr_action": baseline,
        "mean_anchor_oracle_pairwise_concordance": _mean(
            row["anchor_oracle_pairwise_concordance"] for row in records
        ),
        "anchor_top1_oracle_top1_match": _mean(
            max(actions, key=lambda action: row["anchor_utility"][action])
            == row["oracle_best_action"]
            for row in records
        ),
        "policies": policies,
    }


def _fmt(value, digits=6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if abs(value) < 0.001 and value != 0.0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown(result: dict) -> str:
    lines = ["# Outer-LR Action Probe", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Actions: `{result['actions']}`")
    lines.append(
        "- Anchor/oracle pairwise concordance: "
        f"`{_fmt(result['mean_anchor_oracle_pairwise_concordance'], 3)}`"
    )
    lines.append(
        "- Anchor top-1 / oracle top-1 match: "
        f"`{_fmt(result['anchor_top1_oracle_top1_match'], 3)}`"
    )
    lines.extend(
        [
            "",
            "| Policy | Gain vs fixed LR | Gain-positive | Neg drop | Strict drop | Headroom | Random mean | Random p95 | Choices |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for policy, row in result["policies"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | `{}` |".format(
                policy,
                _fmt(row["mean_gain_vs_fixed_lr"]),
                _fmt(row["gain_positive_rate"], 3),
                _fmt(row["negative_rate_relative_drop"], 3),
                _fmt(row["strict_negative_rate_relative_drop"], 3),
                _fmt(row["oracle_headroom_captured"], 3),
                _fmt(row["random_schedule_gain_mean"]),
                _fmt(row["random_schedule_gain_p95"]),
                row["chosen_action_distribution"],
            )
        )
    lines.append("")
    for policy, row in result["policies"].items():
        lines.append(f"## {policy}")
        lines.append("")
        for seed, seed_row in row["per_seed"].items():
            lines.append(
                f"- Seed {seed}: gain `{_fmt(seed_row['mean_gain_vs_fixed_lr'])}`, "
                f"negative drop `{_fmt(seed_row['negative_rate_relative_drop'], 3)}`, "
                f"choices `{seed_row['chosen_action_distribution']}`"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-replays", nargs="+", required=True, type=Path)
    parser.add_argument("--oracle-replays", nargs="+", required=True, type=Path)
    parser.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    parser.add_argument("--baseline", default="current_outer_lr50")
    parser.add_argument("--z-margins", default="0,0.5,1")
    parser.add_argument("--random-replicates", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=20260709)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    actions = tuple(action.strip() for action in args.actions.split(",") if action.strip())
    margins = tuple(float(value) for value in args.z_margins.split(","))
    if args.baseline not in actions:
        raise SystemExit("--baseline must be included in --actions")
    result = analyze(
        _read_merge(args.anchor_replays),
        _read_merge(args.oracle_replays),
        actions,
        args.baseline,
        margins,
        args.random_replicates,
        args.random_seed,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
