#!/usr/bin/env python3
"""Aggregate ProbeCommit offline policy replay summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


POLICIES = (
    "token_weighted",
    "freshness_weighted",
    "metadata_calibrated",
    "anchor_reweight_sigmoid",
    "anchor_reweight_softplus",
    "anchor_top50",
    "anchor_drop_bottom25",
    "anchor_drop_bottom50",
    "anchor_positive_threshold",
    "anchor_shrink",
    "probecommit_v1",
    "oracle_positive",
    "oracle_topk",
    "random_probecommit_count",
    "random_oracle_positive_count",
)


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _infer_seed(path: Path, summary: dict) -> int | None:
    seeds = summary.get("seeds")
    if isinstance(seeds, list) and len(seeds) == 1:
        return int(seeds[0])
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def aggregate(paths: list[Path]) -> dict:
    seeds = []
    for path in paths:
        summary = json.loads(path.read_text())
        row = {
            "seed": _infer_seed(path, summary),
            "path": str(path),
            "records": int(summary["records"]),
            "candidate_count_mean": float(summary["candidate_count_mean"]),
            "oracle_positive_headroom_mean": float(summary["oracle_positive_headroom_mean"]),
            "oracle_topk_headroom_mean": float(summary["oracle_topk_headroom_mean"]),
        }
        for policy in POLICIES:
            pdata = summary["policies"][policy]
            for key, value in pdata.items():
                if value is None:
                    row[f"{policy}_{key}"] = None
                elif isinstance(value, (int, float)):
                    row[f"{policy}_{key}"] = float(value)
        seeds.append(row)

    numeric_keys = sorted(
        {
            key
            for row in seeds
            for key, value in row.items()
            if key not in {"seed", "path"} and isinstance(value, (int, float)) and value is not None
        }
    )
    aggregate_rows = {
        key: {
            "mean": _mean([float(row[key]) for row in seeds if row.get(key) is not None]),
            "std": _std([float(row[key]) for row in seeds if row.get(key) is not None]),
        }
        for key in numeric_keys
    }
    token_mean = aggregate_rows["token_weighted_mean_utility"]["mean"]
    probe_mean = aggregate_rows["probecommit_v1_mean_utility"]["mean"]
    random_mean = aggregate_rows["random_probecommit_count_mean_utility"]["mean"]
    seed_gain = [
        float(row["probecommit_v1_mean_utility"]) - float(row["token_weighted_mean_utility"])
        for row in seeds
    ]
    gates = {
        "all_seeds_positive_probecommit_gain": all(g > 0.0 for g in seed_gain),
        "probecommit_mean_utility_gain": probe_mean - token_mean,
        "probecommit_headroom_positive_ge_50pct": aggregate_rows[
            "probecommit_v1_oracle_positive_headroom_captured"
        ]["mean"]
        >= 0.50,
        "probecommit_negative_drop_ge_25pct": aggregate_rows[
            "probecommit_v1_negative_rate_relative_drop"
        ]["mean"]
        >= 0.25,
        "probecommit_strict_negative_decreases": aggregate_rows[
            "probecommit_v1_strict_negative_merge_rate"
        ]["mean"]
        < aggregate_rows["token_weighted_strict_negative_merge_rate"]["mean"],
        "probecommit_selected_mass_ge_40pct": aggregate_rows[
            "probecommit_v1_selected_mass_mean"
        ]["mean"]
        >= 0.40,
        "probecommit_beats_random_count": probe_mean > random_mean,
    }
    gates["gate_c_pass"] = all(bool(v) for k, v in gates.items() if k != "probecommit_mean_utility_gain")
    return {"seeds": seeds, "aggregate": aggregate_rows, "gates": gates}


def to_markdown(result: dict) -> str:
    lines = ["# ProbeCommit Policy Replay Aggregate", ""]
    lines.append("## Per Seed")
    lines.append("")
    lines.append(
        "| Seed | Groups | Token utility | Token neg. | ProbeCommit utility | ProbeCommit neg. | Headroom captured | Selected mass | Random utility |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["seeds"]:
        lines.append(
            "| {seed} | {records} | {tu:.6f} | {tn:.3f} | {pu:.6f} | {pn:.3f} | "
            "{hc:.3f} | {sm:.3f} | {ru:.6f} |".format(
                seed=row["seed"],
                records=row["records"],
                tu=row["token_weighted_mean_utility"],
                tn=row["token_weighted_negative_merge_rate"],
                pu=row["probecommit_v1_mean_utility"],
                pn=row["probecommit_v1_negative_merge_rate"],
                hc=row["probecommit_v1_oracle_positive_headroom_captured"],
                sm=row["probecommit_v1_selected_mass_mean"],
                ru=row["random_probecommit_count_mean_utility"],
            )
        )
    lines.extend(["", "## Aggregate", ""])
    lines.append("| Policy | Mean utility | Negative rate | Strict negative | Headroom captured | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    agg = result["aggregate"]
    for policy in [
        "token_weighted",
        "freshness_weighted",
        "metadata_calibrated",
        "anchor_reweight_sigmoid",
        "anchor_reweight_softplus",
        "anchor_top50",
        "anchor_drop_bottom50",
        "probecommit_v1",
        "oracle_positive",
        "oracle_topk",
        "random_probecommit_count",
    ]:
        lines.append(
            "| `{}` | {:.6f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                policy,
                agg[f"{policy}_mean_utility"]["mean"],
                agg[f"{policy}_negative_merge_rate"]["mean"],
                agg[f"{policy}_strict_negative_merge_rate"]["mean"],
                agg[f"{policy}_oracle_positive_headroom_captured"]["mean"],
                agg[f"{policy}_selected_mass_mean"]["mean"],
            )
        )
    lines.extend(["", "## Gate C", ""])
    lines.append("| Gate | Value |")
    lines.append("|---|---:|")
    for key, value in result["gates"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summaries", nargs="+", type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    args = p.parse_args(argv)
    result = aggregate(args.summaries)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.out_md.write_text(to_markdown(result))
    print(json.dumps(_jsonable(result["gates"]), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
