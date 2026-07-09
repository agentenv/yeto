#!/usr/bin/env python3
"""Analyze anchor-vs-oracle deployable action ranking agreement."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path


DEFAULT_ACTIONS = (
    "token_weighted",
    "freshness_weighted",
    "anchor_drop_bottom25",
    "anchor_positive_threshold",
    "anchor_shrink",
    "probecommit_v1",
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


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


def rank(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    out = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg = (i + j - 1) / 2.0
        for k in range(i, j):
            out[ordered[k][0]] = avg
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman(left: dict[str, float], right: dict[str, float], actions: tuple[str, ...]) -> float | None:
    lr = rank({a: left[a] for a in actions})
    rr = rank({a: right[a] for a in actions})
    return pearson([lr[a] for a in actions], [rr[a] for a in actions])


def pairwise_concordance(
    left: dict[str, float], right: dict[str, float], actions: tuple[str, ...]
) -> float | None:
    total = 0
    wins = 0
    for a, b in combinations(actions, 2):
        dl = left[a] - left[b]
        dr = right[a] - right[b]
        if dl == 0.0 or dr == 0.0:
            continue
        total += 1
        if dl * dr > 0.0:
            wins += 1
    return wins / total if total else None


def bin_value(value: float, bins: list[float]) -> str:
    prev = -float("inf")
    for edge in bins:
        if value <= edge:
            return f"({prev:g},{edge:g}]"
        prev = edge
    return f"({prev:g},inf)"


def row_metrics(row: dict, actions: tuple[str, ...]) -> dict:
    anchor = {a: float(row[f"{a}_anchor_utility"]) for a in actions}
    oracle = {a: float(row[f"{a}_oracle_utility"]) for a in actions}
    anchor_order = sorted(actions, key=lambda a: anchor[a], reverse=True)
    oracle_order = sorted(actions, key=lambda a: oracle[a], reverse=True)
    top = anchor_order[0]
    second = anchor_order[1] if len(anchor_order) > 1 else top
    token = float(row["token_weighted_oracle_utility"])
    chosen = float(row[f"{top}_oracle_utility"])
    oracle_positive = float(row["oracle_positive_oracle_utility"])
    denom = oracle_positive - token
    return {
        "seed": row.get("seed"),
        "step": row.get("step"),
        "fragment": row.get("fragment"),
        "candidate_count": int(row.get("candidate_count", 0)),
        "anchor_top1": top,
        "oracle_top1": oracle_order[0],
        "top1_match": top == oracle_order[0],
        "spearman": spearman(anchor, oracle, actions),
        "pairwise_concordance": pairwise_concordance(anchor, oracle, actions),
        "anchor_margin": anchor[top] - anchor[second],
        "anchor_top1_utility": anchor[top],
        "oracle_top1_utility": oracle[oracle_order[0]],
        "anchor_top1_gain_vs_token": chosen - token,
        "anchor_top1_negative": bool(row[f"{top}_oracle_negative"]),
        "anchor_top1_strict_negative": row.get(f"{top}_oracle_strict_negative"),
        "anchor_top1_selected_mass": float(row.get(f"{top}_selected_mass", 1.0)),
        "headroom_captured": None if denom <= 0.0 else (chosen - token) / denom,
        "token_negative": bool(row["token_weighted_oracle_negative"]),
        "token_strict_negative": row.get("token_weighted_oracle_strict_negative"),
    }


def summarize(metrics: list[dict]) -> dict:
    if not metrics:
        return {"records": 0}
    return {
        "records": len(metrics),
        "mean_spearman": mean([m["spearman"] for m in metrics]),
        "mean_pairwise_concordance": mean([m["pairwise_concordance"] for m in metrics]),
        "top1_oracle_match": mean([1.0 if m["top1_match"] else 0.0 for m in metrics]),
        "anchor_top1_mean_gain_vs_token": mean([m["anchor_top1_gain_vs_token"] for m in metrics]),
        "anchor_top1_negative_rate": mean([1.0 if m["anchor_top1_negative"] else 0.0 for m in metrics]),
        "anchor_top1_strict_negative_rate": mean(
            [
                1.0 if m["anchor_top1_strict_negative"] else 0.0
                for m in metrics
                if m["anchor_top1_strict_negative"] is not None
            ]
        ),
        "headroom_captured": mean([m["headroom_captured"] for m in metrics]),
        "selected_mass_mean": mean([m["anchor_top1_selected_mass"] for m in metrics]),
        "anchor_margin_mean": mean([m["anchor_margin"] for m in metrics]),
    }


def grouped_summary(metrics: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for metric in metrics:
        groups[str(metric[key])].append(metric)
    return {k: summarize(v) for k, v in sorted(groups.items())}


def analyze(paths: list[Path], actions: tuple[str, ...]) -> dict:
    metrics = []
    for path in paths:
        for row in read_jsonl(path):
            missing = [
                action
                for action in actions
                if f"{action}_anchor_utility" not in row or f"{action}_oracle_utility" not in row
            ]
            if missing:
                raise SystemExit(f"{path}: missing actions {missing}")
            metrics.append(row_metrics(row, actions))
    margin_groups = defaultdict(list)
    for metric in metrics:
        margin_groups[bin_value(metric["anchor_margin"], [0.00025, 0.0005, 0.001, 0.002, 0.005])].append(metric)
    mass_groups = defaultdict(list)
    for metric in metrics:
        mass_groups[bin_value(metric["anchor_top1_selected_mass"], [0.5, 0.67, 0.8, 0.9, 1.0])].append(metric)
    return {
        "schema": "action_rank_agreement_v1",
        "paths": [str(p) for p in paths],
        "actions": list(actions),
        "overall": summarize(metrics),
        "by_seed": grouped_summary(metrics, "seed"),
        "by_candidate_count": grouped_summary(metrics, "candidate_count"),
        "by_anchor_margin": {k: summarize(v) for k, v in sorted(margin_groups.items())},
        "by_selected_mass": {k: summarize(v) for k, v in sorted(mass_groups.items())},
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


def summary_table(title: str, rows: dict) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| Group | Records | Spearman | Pairwise | Top1 match | Top1 gain | Headroom | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, row in rows.items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                key,
                row["records"],
                fmt(row.get("mean_spearman")),
                fmt(row.get("mean_pairwise_concordance")),
                fmt(row.get("top1_oracle_match")),
                fmt(row.get("anchor_top1_mean_gain_vs_token"), 6),
                fmt(row.get("headroom_captured")),
                fmt(row.get("selected_mass_mean")),
            )
        )
    lines.append("")
    return lines


def to_markdown(result: dict) -> str:
    lines = ["# Action Rank Agreement", ""]
    overall = result["overall"]
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in (
        "records",
        "mean_spearman",
        "mean_pairwise_concordance",
        "top1_oracle_match",
        "anchor_top1_mean_gain_vs_token",
        "anchor_top1_negative_rate",
        "anchor_top1_strict_negative_rate",
        "headroom_captured",
        "selected_mass_mean",
        "anchor_margin_mean",
    ):
        lines.append(f"| `{key}` | {fmt(overall.get(key), 6)} |")
    lines.append("")
    lines.extend(summary_table("By Seed", result["by_seed"]))
    lines.extend(summary_table("By Candidate Count", result["by_candidate_count"]))
    lines.extend(summary_table("By Anchor Margin", result["by_anchor_margin"]))
    lines.extend(summary_table("By Selected Mass", result["by_selected_mass"]))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="+", required=True, type=Path)
    p.add_argument("--actions", nargs="+", default=list(DEFAULT_ACTIONS))
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = analyze(args.replays, tuple(args.actions))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.out_md.write_text(to_markdown(result))
    print(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
