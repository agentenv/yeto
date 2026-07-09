#!/usr/bin/env python3
"""Aggregate EXP2.9 action-probe replay summaries across seeds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


POLICY_ORDER = (
    "token_weighted",
    "best_fixed_deployable",
    "action_probe_top1",
    "action_probe_margin_gated",
    "action_probe_risk_aware",
    "best_deployable_oracle",
    "oracle_positive",
    "oracle_topk",
    "random_top1_action_count",
)


def jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def read_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "policies" not in data:
        raise SystemExit(f"{path}: missing policies")
    return data


def merge_action_distributions(rows: list[dict], policy: str) -> dict:
    counter = Counter()
    for row in rows:
        dist = row["policies"].get(policy, {}).get("chosen_action_distribution", {})
        counter.update({str(k): int(v) for k, v in dist.items()})
    return dict(sorted(counter.items()))


def aggregate_policy(rows: list[dict], policy: str) -> dict:
    per_seed = []
    for row in rows:
        item = row["policies"].get(policy)
        if item is None:
            continue
        seed = row["seeds"][0] if len(row.get("seeds", [])) == 1 else row.get("seeds")
        per_seed.append(
            {
                "seed": seed,
                "mean_gain_vs_token": item.get("mean_gain_vs_token"),
                "negative_rate_relative_drop": item.get("negative_rate_relative_drop"),
                "strict_negative_rate_relative_drop": item.get("strict_negative_rate_relative_drop"),
                "oracle_positive_headroom_captured": item.get("oracle_positive_headroom_captured"),
                "selected_mass_mean": item.get("selected_mass_mean"),
                "chosen_action_distribution": item.get("chosen_action_distribution", {}),
                "fixed_action": item.get("fixed_action"),
            }
        )
    return {
        "seeds": len(per_seed),
        "mean_gain_vs_token": mean([r["mean_gain_vs_token"] for r in per_seed]),
        "all_seed_gains_positive": all(
            r["mean_gain_vs_token"] is not None and r["mean_gain_vs_token"] > 0.0
            for r in per_seed
        ),
        "negative_rate_relative_drop": mean(
            [r["negative_rate_relative_drop"] for r in per_seed]
        ),
        "strict_negative_rate_relative_drop": mean(
            [r["strict_negative_rate_relative_drop"] for r in per_seed]
        ),
        "oracle_positive_headroom_captured": mean(
            [r["oracle_positive_headroom_captured"] for r in per_seed]
        ),
        "selected_mass_mean": mean([r["selected_mass_mean"] for r in per_seed]),
        "chosen_action_distribution": merge_action_distributions(rows, policy),
        "per_seed": per_seed,
    }


def aggregate(rows: list[dict], expected_seeds: list[int]) -> dict:
    seen = sorted({int(seed) for row in rows for seed in row.get("seeds", [])})
    missing = sorted(set(expected_seeds) - set(seen))
    extra = sorted(set(seen) - set(expected_seeds))
    if missing or extra:
        raise SystemExit(f"seed mismatch: missing={missing}, extra={extra}, seen={seen}")
    policies = {
        policy: aggregate_policy(rows, policy)
        for policy in POLICY_ORDER
        if any(policy in row["policies"] for row in rows)
    }
    candidate_policies = [
        policy
        for policy in ("action_probe_top1", "action_probe_margin_gated", "action_probe_risk_aware")
        if policy in policies
    ]
    main_policy = max(candidate_policies, key=lambda p: policies[p]["mean_gain_vs_token"])
    main = policies[main_policy]
    seed67 = None
    for item in main["per_seed"]:
        if item["seed"] == 67:
            seed67 = item
            break
    random_control = policies.get("random_top1_action_count", {})
    gates = {
        "main_policy": main_policy,
        "mean_gain_ge_0.0005": main["mean_gain_vs_token"] >= 0.0005,
        "all_seeds_positive": main["all_seed_gains_positive"],
        "negative_drop_ge_0.20": main["negative_rate_relative_drop"] >= 0.20,
        "strict_drop_ge_0.20": main["strict_negative_rate_relative_drop"] >= 0.20,
        "headroom_captured_ge_0.40": main["oracle_positive_headroom_captured"] >= 0.40,
        "seed67_positive_gain": bool(seed67 and seed67["mean_gain_vs_token"] > 0.0),
        "seed67_nonnegative_headroom": bool(
            seed67 and seed67["oracle_positive_headroom_captured"] >= 0.0
        ),
        "beats_random_action_count": bool(
            random_control
            and main["mean_gain_vs_token"] > random_control.get("mean_gain_vs_token", float("inf"))
        ),
    }
    gates["gate_pass"] = all(bool(v) for k, v in gates.items() if k != "main_policy")
    gates["strong_pass"] = bool(
        gates["gate_pass"]
        and main["mean_gain_vs_token"] >= 0.0007
        and main["negative_rate_relative_drop"] >= 0.30
        and main["oracle_positive_headroom_captured"] >= 0.50
    )
    return {
        "schema": "action_probe_aggregate_v1",
        "records": sum(int(row["records"]) for row in rows),
        "seeds": seen,
        "policies": policies,
        "gates": gates,
    }


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) < 0.001 and value != 0.0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def to_markdown(result: dict) -> str:
    lines = ["# Action-Probe Aggregate", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Main policy: `{result['gates']['main_policy']}`")
    lines.append(f"- Gate pass: `{result['gates']['gate_pass']}`")
    lines.append("")
    lines.append("| Policy | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for policy in POLICY_ORDER:
        row = result["policies"].get(policy)
        if not row:
            continue
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                policy,
                fmt(row.get("mean_gain_vs_token"), 6),
                fmt(row.get("negative_rate_relative_drop")),
                fmt(row.get("strict_negative_rate_relative_drop")),
                fmt(row.get("oracle_positive_headroom_captured")),
                fmt(row.get("selected_mass_mean")),
            )
        )
    lines.append("")
    main_policy = result["gates"]["main_policy"]
    lines.append(f"## Per-Seed `{main_policy}`")
    lines.append("")
    lines.append("| Seed | Gain | Negative drop | Strict drop | Headroom captured | Chosen action distribution |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in result["policies"][main_policy]["per_seed"]:
        lines.append(
            "| {} | {} | {} | {} | {} | `{}` |".format(
                row["seed"],
                fmt(row["mean_gain_vs_token"], 6),
                fmt(row["negative_rate_relative_drop"]),
                fmt(row["strict_negative_rate_relative_drop"]),
                fmt(row["oracle_positive_headroom_captured"]),
                row.get("chosen_action_distribution", {}),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries", nargs="+", required=True, type=Path)
    p.add_argument("--expected-seeds", nargs="+", type=int, default=[53, 67, 79])
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    summaries = [read_summary(path) for path in args.summaries]
    result = aggregate(summaries, args.expected_seeds)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(to_markdown(result))
    print(json.dumps(jsonable(result["gates"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
