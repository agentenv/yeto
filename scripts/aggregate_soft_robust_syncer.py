#!/usr/bin/env python3
"""Aggregate EXP2.17 soft robust syncer replay summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def aggregate(paths: list[Path]) -> dict:
    summaries = [json.loads(path.read_text()) for path in paths]
    policies = sorted(summaries[0]["policies"])
    out = {}
    for policy in policies:
        rows = [summary["policies"][policy] for summary in summaries]
        gains = [row["mean_gain_vs_token"] for row in rows]
        out[policy] = {
            "mean_gain_vs_token": _mean(gains),
            "seeds_positive": sum(1 for gain in gains if gain > 0.0),
            "seed_count": len(gains),
            "negative_rate_relative_drop": _mean([row["negative_rate_relative_drop"] or 0.0 for row in rows]),
            "strict_negative_rate_relative_drop": _mean(
                [row["strict_negative_rate_relative_drop"] or 0.0 for row in rows]
            ),
            "selected_mass_mean": _mean([row["selected_mass_mean"] for row in rows]),
            "gain_positive_rate": _mean([row["gain_positive_rate"] for row in rows]),
            "per_seed_gain": gains,
        }
    best = max((p for p in policies if p != "token_weighted"), key=lambda p: out[p]["mean_gain_vs_token"])
    high_mass = [p for p in policies if p != "token_weighted" and out[p]["selected_mass_mean"] >= 0.95]
    best_high_mass = max(high_mass, key=lambda p: out[p]["mean_gain_vs_token"]) if high_mass else None
    return {
        "schema": "soft_robust_syncer_aggregate_v1",
        "inputs": [str(path) for path in paths],
        "records": sum(int(summary["records"]) for summary in summaries),
        "seeds": [summary["seeds"][0] for summary in summaries],
        "policies": out,
        "best_non_token_policy": best,
        "best_high_mass_non_token_policy": best_high_mass,
        "gate": {
            "best_mean_gain_positive": out[best]["mean_gain_vs_token"] > 0.0,
            "best_all_seeds_positive": out[best]["seeds_positive"] == out[best]["seed_count"],
            "best_negative_drop_positive": out[best]["negative_rate_relative_drop"] > 0.0,
            "best_strict_drop_positive": out[best]["strict_negative_rate_relative_drop"] > 0.0,
            "best_selected_mass_ge_0.95": out[best]["selected_mass_mean"] >= 0.95,
            "best_high_mass_mean_gain_positive": (
                False if best_high_mass is None else out[best_high_mass]["mean_gain_vs_token"] > 0.0
            ),
        },
    }


def fmt(value, digits=6) -> str:
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


def markdown(result: dict) -> str:
    lines = ["# Soft Robust Syncer Aggregate", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Best non-token policy: `{result['best_non_token_policy']}`")
    lines.append(f"- Best high-mass non-token policy: `{result['best_high_mass_non_token_policy']}`")
    lines.append(f"- Gate: `{all(result['gate'].values())}`")
    lines.append("")
    lines.append("| Policy | Mean gain | Seeds positive | Neg drop | Strict drop | Mass | Gain-positive rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for policy, row in result["policies"].items():
        lines.append(
            "| `{}` | {} | {}/{} | {} | {} | {} | {} |".format(
                policy,
                fmt(row["mean_gain_vs_token"]),
                row["seeds_positive"],
                row["seed_count"],
                fmt(row["negative_rate_relative_drop"], 3),
                fmt(row["strict_negative_rate_relative_drop"], 3),
                fmt(row["selected_mass_mean"], 3),
                fmt(row["gain_positive_rate"], 3),
            )
        )
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for key, value in result["gate"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summaries", nargs="+", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = aggregate(args.summaries)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.out_md.write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
