#!/usr/bin/env python3
"""Aggregate group-local feature and policy-grid summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else float("nan")
    mean = _mean(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "n/a"
    if abs(v) < 0.001 and v != 0.0:
        return f"{v:.2e}"
    return f"{v:.{digits}f}"


def aggregate(feature_summaries: list[Path], policy_grids: list[Path]) -> dict:
    features = [_read(path) for path in feature_summaries]
    grids = [_read(path) for path in policy_grids]
    result = {
        "schema": "group_local_results_aggregate_v1",
        "feature_summaries": [str(path) for path in feature_summaries],
        "policy_grids": [str(path) for path in policy_grids],
        "features": {
            "records_total": sum(int(item["records"]) for item in features),
            "records_min": min(int(item["records"]) for item in features) if features else 0,
            "token_weighted_negative_rate_mean": _mean(
                [item.get("token_weighted_negative_rate") for item in features]
            ),
            "oracle_positive_headroom_mean": _mean(
                [item.get("oracle_positive_headroom_mean") for item in features]
            ),
        },
        "policy_grid": {},
    }
    if grids:
        aggs = [item["heldout_seed"]["aggregate"] for item in grids]
        keys = sorted({key for agg in aggs for key, value in agg.items() if isinstance(value, (int, float, bool))})
        for key in keys:
            vals = [float(agg[key]) for agg in aggs if isinstance(agg.get(key), (int, float, bool))]
            result["policy_grid"][key] = {"mean": _mean(vals), "std": _std(vals)}
        result["policy_grid"]["all_gate_pass"] = all(bool(agg.get("gate_pass")) for agg in aggs)
    return result


def to_markdown(result: dict) -> str:
    lines = ["# Group-Local Results Aggregate", ""]
    lines.append("## Features")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in result["features"].items():
        lines.append(f"| `{key}` | {_fmt(value, 6)} |")
    lines.append("")
    lines.append("## Policy Grid")
    lines.append("")
    lines.append("| Metric | Mean | Std |")
    lines.append("|---|---:|---:|")
    for key, value in result["policy_grid"].items():
        if isinstance(value, dict):
            lines.append(f"| `{key}` | {_fmt(value['mean'], 6)} | {_fmt(value['std'], 6)} |")
        else:
            lines.append(f"| `{key}` | {value} |  |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-summary", nargs="+", required=True, type=Path)
    p.add_argument("--policy-grid", nargs="+", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = aggregate(args.feature_summary, args.policy_grid)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(to_markdown(result))
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
