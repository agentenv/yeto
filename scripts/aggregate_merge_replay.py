#!/usr/bin/env python3
"""Aggregate per-seed merge replay summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


POLICIES = (
    "token_weighted",
    "uniform",
    "freshness_weighted",
    "hand_score_weighted",
    "oracle_positive",
    "oracle_topk",
    "oracle_drop_strict_bad",
    "random_positive_count",
    "random_drop_strict_count",
)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _infer_seed(path: Path, summary: dict) -> int | None:
    seeds = summary.get("seeds")
    if isinstance(seeds, list) and len(seeds) == 1:
        return int(seeds[0])
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _metric(summary: dict, policy: str, name: str) -> float | None:
    value = summary.get("policies", {}).get(policy, {}).get(name)
    return float(value) if value is not None else None


def aggregate(paths: list[Path]) -> dict:
    seed_rows = []
    for path in paths:
        summary = json.loads(path.read_text())
        seed = _infer_seed(path, summary)
        row = {
            "seed": seed,
            "path": str(path),
            "records": int(summary["records"]),
            "candidate_count_mean": float(summary["candidate_count_mean"]),
            "bad_weight_mass_mean": float(summary["bad_weight_mass_mean"]),
            "strict_bad_weight_mass_mean": float(summary["strict_bad_weight_mass_mean"]),
        }
        for policy in POLICIES:
            row[f"{policy}_mean_utility"] = _metric(summary, policy, "mean_utility")
            row[f"{policy}_negative_merge_rate"] = _metric(
                summary, policy, "negative_merge_rate"
            )
            row[f"{policy}_strict_negative_merge_rate"] = _metric(
                summary, policy, "strict_negative_merge_rate"
            )
            row[f"{policy}_mean_selected_count"] = _metric(
                summary, policy, "mean_selected_count"
            )
        for name, values in summary.get("headroom", {}).items():
            row[f"{name}_mean"] = float(values["mean"])
            row[f"{name}_positive_rate"] = float(values["positive_rate"])
        for name, values in summary.get("random_headroom", {}).items():
            row[f"{name}_mean"] = float(values["mean"])
            row[f"{name}_positive_rate"] = float(values["positive_rate"])
        seed_rows.append(row)

    numeric_keys = sorted(
        {
            key
            for row in seed_rows
            for key, value in row.items()
            if key not in {"seed", "path"} and isinstance(value, (int, float)) and value is not None
        }
    )
    aggregate_rows = {
        key: {
            "mean": _mean([float(row[key]) for row in seed_rows if row.get(key) is not None]),
            "std": _std([float(row[key]) for row in seed_rows if row.get(key) is not None]),
        }
        for key in numeric_keys
    }
    gates = {
        "min_records_per_seed": min(row["records"] for row in seed_rows),
        "records_gate_500": min(row["records"] for row in seed_rows) >= 500,
        "token_negative_rate_mean": aggregate_rows[
            "token_weighted_negative_merge_rate"
        ]["mean"],
        "token_negative_rate_above_20pct": aggregate_rows[
            "token_weighted_negative_merge_rate"
        ]["mean"]
        > 0.20,
        "oracle_positive_headroom_mean": aggregate_rows[
            "oracle_positive_minus_token_mean"
        ]["mean"],
        "oracle_topk_headroom_mean": aggregate_rows["oracle_topk_minus_token_mean"][
            "mean"
        ],
        "oracle_positive_headroom_positive": aggregate_rows[
            "oracle_positive_minus_token_mean"
        ]["mean"]
        > 0.0,
        "oracle_topk_headroom_positive": aggregate_rows["oracle_topk_minus_token_mean"][
            "mean"
        ]
        > 0.0,
        "oracle_positive_beats_random": aggregate_rows[
            "oracle_positive_minus_token_mean"
        ]["mean"]
        > aggregate_rows["random_positive_count_minus_token_mean"]["mean"],
        "oracle_topk_beats_random": aggregate_rows["oracle_topk_minus_token_mean"][
            "mean"
        ]
        > aggregate_rows["random_positive_count_minus_token_mean"]["mean"],
        "all_seed_oracle_positive_headroom_positive": all(
            row.get("oracle_positive_minus_token_mean", 0.0) > 0.0 for row in seed_rows
        ),
        "all_seed_oracle_topk_headroom_positive": all(
            row.get("oracle_topk_minus_token_mean", 0.0) > 0.0 for row in seed_rows
        ),
    }
    gates["gate_a_pass"] = all(
        [
            gates["records_gate_500"],
            gates["token_negative_rate_above_20pct"],
            gates["oracle_positive_headroom_positive"]
            or gates["oracle_topk_headroom_positive"],
            gates["oracle_positive_beats_random"] or gates["oracle_topk_beats_random"],
            gates["all_seed_oracle_positive_headroom_positive"]
            or gates["all_seed_oracle_topk_headroom_positive"],
        ]
    )
    return {"seeds": seed_rows, "aggregate": aggregate_rows, "gates": gates}


def to_markdown(result: dict) -> str:
    lines = ["# Merge Replay Aggregate", ""]
    lines.append("## Per Seed")
    lines.append("")
    lines.append(
        "| Seed | Groups | Token utility | Token neg. | Token strict neg. | Oracle+ utility | Oracle+ headroom | Oracle top-k utility | Oracle top-k headroom |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["seeds"]:
        lines.append(
            "| {seed} | {records} | {tu:.6f} | {tn:.3f} | {ts:.3f} | "
            "{opu:.6f} | {oph:.6f} | {otu:.6f} | {oth:.6f} |".format(
                seed=row["seed"],
                records=row["records"],
                tu=row["token_weighted_mean_utility"],
                tn=row["token_weighted_negative_merge_rate"],
                ts=row["token_weighted_strict_negative_merge_rate"],
                opu=row["oracle_positive_mean_utility"],
                oph=row["oracle_positive_minus_token_mean"],
                otu=row["oracle_topk_mean_utility"],
                oth=row["oracle_topk_minus_token_mean"],
            )
        )
    agg = result["aggregate"]
    gates = result["gates"]
    lines.extend(["", "## Aggregate", ""])
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---:|---:|")
    for key in [
        "records",
        "candidate_count_mean",
        "bad_weight_mass_mean",
        "strict_bad_weight_mass_mean",
        "token_weighted_mean_utility",
        "token_weighted_negative_merge_rate",
        "token_weighted_strict_negative_merge_rate",
        "oracle_positive_mean_utility",
        "oracle_positive_minus_token_mean",
        "oracle_topk_mean_utility",
        "oracle_topk_minus_token_mean",
        "random_positive_count_minus_token_mean",
    ]:
        value = agg[key]
        lines.append(f"| `{key}` | {value['mean']:.6f} | {value['std']:.6f} |")
    lines.extend(["", "## Gate A", ""])
    lines.append("| Gate | Value |")
    lines.append("|---|---:|")
    for key, value in gates.items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    args = p.parse_args(argv)
    result = aggregate(args.summaries)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(to_markdown(result))
    print(json.dumps(_jsonable(result["gates"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
