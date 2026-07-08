#!/usr/bin/env python3
"""Tune and evaluate group-local policy grids from built feature rows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_group_local():
    path = REPO_ROOT / "scripts" / "replay_group_local_probecommit.py"
    spec = importlib.util.spec_from_file_location("_group_local_probecommit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_group_local_probecommit"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


gl = _load_group_local()

DEPLOYABLE_ACTIONS = (
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(gl.jsonable(row), sort_keys=True, allow_nan=False) + "\n")


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def stat(row: dict, field: str, name: str) -> float:
    return float(row["scores"][field].get(name, 0.0))


def agreement(row: dict, left: str, right: str) -> bool:
    return bool(row.get("agreement", {}).get(f"top_agree:{left}:{right}", False))


def action_metrics(row: dict, action: str) -> dict:
    if action not in DEPLOYABLE_ACTIONS and not action.startswith("oracle_") and not action.startswith("random_"):
        raise ValueError(f"unexpected action {action!r}")
    data = row["actions"][action]
    return {
        "action": action,
        "utility": float(data["utility"]),
        "negative": bool(data["negative"]),
        "strict_negative": bool(data["strict_negative"]),
        "selected_mass": float(data["selected_mass"]),
        "selected_count": float(data["selected_count"]),
    }


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    field: str
    params: dict


def decide(rule: Rule, row: dict) -> str:
    if rule.family == "base":
        return rule.params["action"]
    if rule.family == "drop25_if_spread":
        return "anchor_drop_bottom25" if stat(row, rule.field, "score_spread") >= rule.params["spread"] else "token_weighted"
    if rule.family == "drop25_if_iqr":
        return "anchor_drop_bottom25" if stat(row, rule.field, "score_iqr") >= rule.params["iqr"] else "token_weighted"
    if rule.family == "drop25_if_gapz":
        return "anchor_drop_bottom25" if stat(row, rule.field, "top_gap_z") >= rule.params["gapz"] else "token_weighted"
    if rule.family == "drop25_if_spread_and_gapz":
        return (
            "anchor_drop_bottom25"
            if stat(row, rule.field, "score_spread") >= rule.params["spread"]
            and stat(row, rule.field, "top_gap_z") >= rule.params["gapz"]
            else "token_weighted"
        )
    if rule.family == "drop25_if_low_entropy":
        return "anchor_drop_bottom25" if stat(row, rule.field, "score_entropy") <= rule.params["entropy"] else "token_weighted"
    if rule.family == "positive_if_spread_and_gapz":
        return (
            "anchor_positive_threshold"
            if stat(row, rule.field, "score_spread") >= rule.params["spread"]
            and stat(row, rule.field, "top_gap_z") >= rule.params["gapz"]
            else "token_weighted"
        )
    if rule.family == "shrink_if_mean":
        return "anchor_shrink" if stat(row, rule.field, "score_mean") < rule.params["mean"] else "token_weighted"
    if rule.family == "drop_or_shrink":
        if stat(row, rule.field, "score_mean") < rule.params["mean"]:
            return "anchor_shrink"
        if stat(row, rule.field, "score_spread") >= rule.params["spread"]:
            return "anchor_drop_bottom25"
        return "token_weighted"
    if rule.family == "agree_drop25":
        left = rule.field
        right = rule.params["right"]
        left_ok = stat(row, left, "score_spread") >= rule.params["left_spread"]
        right_ok = stat(row, right, "score_spread") >= rule.params["right_spread"]
        return "anchor_drop_bottom25" if left_ok and right_ok and agreement(row, left, right) else "token_weighted"
    raise ValueError(f"unknown rule family {rule.family!r}")


def evaluate_rule(rule: Rule, rows: list[dict]) -> dict:
    metrics = []
    actions = Counter()
    for row in rows:
        action = decide(rule, row)
        actions[action] += 1
        m = action_metrics(row, action)
        token = action_metrics(row, "token_weighted")
        oracle = action_metrics(row, "oracle_positive")
        denom = oracle["utility"] - token["utility"]
        captured = None if denom <= 0.0 else (m["utility"] - token["utility"]) / denom
        metrics.append(
            {
                **m,
                "gain_vs_token": m["utility"] - token["utility"],
                "gain_positive": m["utility"] > token["utility"],
                "oracle_positive_headroom_captured": captured,
            }
        )
    token_neg = gl.mean([1.0 if action_metrics(row, "token_weighted")["negative"] else 0.0 for row in rows])
    token_strict = gl.mean(
        [1.0 if action_metrics(row, "token_weighted")["strict_negative"] else 0.0 for row in rows]
    )
    neg = gl.mean([1.0 if m["negative"] else 0.0 for m in metrics])
    strict = gl.mean([1.0 if m["strict_negative"] else 0.0 for m in metrics])
    captured = [
        float(m["oracle_positive_headroom_captured"])
        for m in metrics
        if m["oracle_positive_headroom_captured"] is not None
    ]
    return {
        "rule": rule.name,
        "family": rule.family,
        "field": rule.field,
        "params": rule.params,
        "groups": len(rows),
        "mean_utility": gl.mean([m["utility"] for m in metrics]),
        "mean_gain_vs_token": gl.mean([m["gain_vs_token"] for m in metrics]),
        "median_gain_vs_token": gl.quantile([m["gain_vs_token"] for m in metrics], 0.5),
        "gain_positive_rate": gl.mean([1.0 if m["gain_positive"] else 0.0 for m in metrics]),
        "negative_rate": neg,
        "negative_rate_relative_drop": None if token_neg <= 0.0 else (token_neg - neg) / token_neg,
        "strict_negative_rate": strict,
        "strict_negative_rate_relative_drop": (
            None if token_strict <= 0.0 else (token_strict - strict) / token_strict
        ),
        "selected_mass_mean": gl.mean([m["selected_mass"] for m in metrics]),
        "selected_count_mean": gl.mean([m["selected_count"] for m in metrics]),
        "oracle_positive_headroom_captured": gl.mean(captured),
        "headroom_excluded_fraction": 1.0 - len(captured) / len(metrics),
        "actions": dict(sorted(actions.items())),
        "act_rate": 1.0 - actions.get("token_weighted", 0) / len(rows),
    }


def metric_values(rows: list[dict], field: str, name: str) -> list[float]:
    return [stat(row, field, name) for row in rows if finite(stat(row, field, name))]


def thresholds(rows: list[dict], field: str, name: str) -> list[float]:
    values = metric_values(rows, field, name)
    qs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    out = sorted({gl.quantile(values, q) for q in qs if finite(gl.quantile(values, q))})
    return out or [0.0]


def candidate_rules(rows: list[dict], fields: list[str]) -> list[Rule]:
    rules = [Rule(action, "base", fields[0], {"action": action}) for action in DEPLOYABLE_ACTIONS]
    for field in fields:
        spreads = thresholds(rows, field, "score_spread")
        iqrs = thresholds(rows, field, "score_iqr")
        gapzs = thresholds(rows, field, "top_gap_z")
        entropies = thresholds(rows, field, "score_entropy")
        means = thresholds(rows, field, "score_mean")
        for spread in spreads:
            rules.append(Rule(f"{field}:drop25_spread:{spread:.6g}", "drop25_if_spread", field, {"spread": spread}))
        for iqr in iqrs:
            rules.append(Rule(f"{field}:drop25_iqr:{iqr:.6g}", "drop25_if_iqr", field, {"iqr": iqr}))
        for gapz in gapzs:
            rules.append(Rule(f"{field}:drop25_gapz:{gapz:.6g}", "drop25_if_gapz", field, {"gapz": gapz}))
        for entropy in entropies:
            rules.append(
                Rule(
                    f"{field}:drop25_entropy_le:{entropy:.6g}",
                    "drop25_if_low_entropy",
                    field,
                    {"entropy": entropy},
                )
            )
        for mean_threshold in means:
            rules.append(
                Rule(
                    f"{field}:shrink_mean_lt:{mean_threshold:.6g}",
                    "shrink_if_mean",
                    field,
                    {"mean": mean_threshold},
                )
            )
        for spread in spreads[:: max(1, len(spreads) // 4)]:
            for gapz in gapzs[:: max(1, len(gapzs) // 4)]:
                rules.append(
                    Rule(
                        f"{field}:drop25_spread_gapz:{spread:.6g}:{gapz:.6g}",
                        "drop25_if_spread_and_gapz",
                        field,
                        {"spread": spread, "gapz": gapz},
                    )
                )
                rules.append(
                    Rule(
                        f"{field}:positive_spread_gapz:{spread:.6g}:{gapz:.6g}",
                        "positive_if_spread_and_gapz",
                        field,
                        {"spread": spread, "gapz": gapz},
                    )
                )
        for spread in spreads[:: max(1, len(spreads) // 4)]:
            for mean_threshold in means[:: max(1, len(means) // 4)]:
                rules.append(
                    Rule(
                        f"{field}:drop_or_shrink:{spread:.6g}:{mean_threshold:.6g}",
                        "drop_or_shrink",
                        field,
                        {"spread": spread, "mean": mean_threshold},
                    )
                )
    for left in fields:
        for right in fields:
            if left >= right:
                continue
            for left_spread in thresholds(rows, left, "score_spread")[::3]:
                for right_spread in thresholds(rows, right, "score_spread")[::3]:
                    rules.append(
                        Rule(
                            f"{left}+{right}:agree_drop25:{left_spread:.6g}:{right_spread:.6g}",
                            "agree_drop25",
                            left,
                            {
                                "right": right,
                                "left_spread": left_spread,
                                "right_spread": right_spread,
                            },
                        )
                    )
    return rules


def objective(result: dict, args) -> float:
    return (
        float(result["mean_utility"])
        - args.negative_penalty * float(result["negative_rate"])
        - args.strict_penalty * float(result["strict_negative_rate"])
        - args.act_rate_penalty * float(result["act_rate"])
    )


def heldout_seed(rows: list[dict], fields: list[str], args) -> dict:
    seeds = sorted({int(r["seed"]) for r in rows})
    splits = []
    for test_seed in seeds:
        train = [r for r in rows if int(r["seed"]) != test_seed]
        test = [r for r in rows if int(r["seed"]) == test_seed]
        rules = candidate_rules(train, fields)
        train_results = [evaluate_rule(rule, train) for rule in rules]
        train_results.sort(
            key=lambda r: (
                objective(r, args),
                r["mean_utility"],
                r["oracle_positive_headroom_captured"],
                -r["negative_rate"],
            ),
            reverse=True,
        )
        best = train_results[0]
        best_rule = next(rule for rule in rules if rule.name == best["rule"])
        splits.append(
            {
                "test_seed": test_seed,
                "train_seeds": [s for s in seeds if s != test_seed],
                "train_groups": len(train),
                "test_groups": len(test),
                "selected_rule_train": best,
                "test_result": evaluate_rule(best_rule, test),
                "top_train_rules": train_results[: args.keep_top],
            }
        )
    agg = {
        "splits": len(splits),
        "mean_gain_vs_token": gl.mean([s["test_result"]["mean_gain_vs_token"] for s in splits]),
        "all_test_gains_positive": all(s["test_result"]["mean_gain_vs_token"] > 0.0 for s in splits),
        "negative_rate_relative_drop": gl.mean(
            [s["test_result"]["negative_rate_relative_drop"] for s in splits]
        ),
        "strict_negative_rate_relative_drop": gl.mean(
            [s["test_result"]["strict_negative_rate_relative_drop"] for s in splits]
        ),
        "oracle_positive_headroom_captured": gl.mean(
            [s["test_result"]["oracle_positive_headroom_captured"] for s in splits]
        ),
        "selected_mass_mean": gl.mean([s["test_result"]["selected_mass_mean"] for s in splits]),
        "act_rate": gl.mean([s["test_result"]["act_rate"] for s in splits]),
    }
    agg["gate_pass"] = bool(
        agg["all_test_gains_positive"]
        and agg["mean_gain_vs_token"] > 0.0
        and agg["oracle_positive_headroom_captured"] >= args.min_headroom_captured
        and agg["negative_rate_relative_drop"] >= args.min_negative_drop
        and agg["selected_mass_mean"] >= args.min_selected_mass
    )
    return {"splits": splits, "aggregate": agg}


def baseline_results(rows: list[dict]) -> dict:
    out = {}
    for action in DEPLOYABLE_ACTIONS + ("oracle_positive", "oracle_topk", "random_probecommit_count"):
        if action not in rows[0]["actions"]:
            continue
        rule = Rule(action, "base", "probe_grad_dot", {"action": action})
        out[action] = evaluate_rule(rule, rows)
    return out


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


def to_markdown(result: dict) -> str:
    lines = ["# Group-Local Policy Grid", ""]
    agg = result["heldout_seed"]["aggregate"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Groups: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Mean held-out gain: `{_fmt(agg['mean_gain_vs_token'], 6)}`")
    lines.append(f"- Headroom captured: `{_fmt(agg['oracle_positive_headroom_captured'])}`")
    lines.append(f"- Negative-rate drop: `{_fmt(agg['negative_rate_relative_drop'])}`")
    lines.append(f"- Act rate: `{_fmt(agg['act_rate'])}`")
    lines.append(f"- Gate pass: `{agg['gate_pass']}`")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Action | Mean utility | Gain vs token | Negative drop | Strict drop | Headroom captured | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for action, row in result["baselines"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                action,
                _fmt(row["mean_utility"], 6),
                _fmt(row["mean_gain_vs_token"], 6),
                _fmt(row["negative_rate_relative_drop"]),
                _fmt(row["strict_negative_rate_relative_drop"]),
                _fmt(row["oracle_positive_headroom_captured"]),
                _fmt(row["selected_mass_mean"]),
            )
        )
    lines.append("")
    lines.append("## Held-Out Seed")
    lines.append("")
    lines.append("| Test seed | Rule | Gain | Negative drop | Strict drop | Headroom captured | Selected mass | Act rate | Actions |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for split in result["heldout_seed"]["splits"]:
        row = split["test_result"]
        lines.append(
            "| {} | `{}` | {} | {} | {} | {} | {} | {} | `{}` |".format(
                split["test_seed"],
                row["rule"],
                _fmt(row["mean_gain_vs_token"], 6),
                _fmt(row["negative_rate_relative_drop"]),
                _fmt(row["strict_negative_rate_relative_drop"]),
                _fmt(row["oracle_positive_headroom_captured"]),
                _fmt(row["selected_mass_mean"]),
                _fmt(row["act_rate"]),
                row["actions"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument(
        "--score-fields",
        default="probe_grad_dot,probe_grad_cosine,calibrated_score,consensus_cosine,freshness,combined_score",
    )
    p.add_argument("--negative-penalty", type=float, default=0.0005)
    p.add_argument("--strict-penalty", type=float, default=0.0005)
    p.add_argument("--act-rate-penalty", type=float, default=0.0)
    p.add_argument("--min-headroom-captured", type=float, default=0.40)
    p.add_argument("--min-negative-drop", type=float, default=0.20)
    p.add_argument("--min-selected-mass", type=float, default=0.40)
    p.add_argument("--keep-top", type=int, default=20)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-splits", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.features)
    fields = [field.strip() for field in args.score_fields.split(",") if field.strip()]
    result = {
        "schema": "group_local_policy_grid_v1",
        "records": len(rows),
        "seeds": sorted({int(r["seed"]) for r in rows}),
        "score_fields": fields,
        "objective": {
            "negative_penalty": args.negative_penalty,
            "strict_penalty": args.strict_penalty,
            "act_rate_penalty": args.act_rate_penalty,
        },
        "baselines": baseline_results(rows),
        "heldout_seed": heldout_seed(rows, fields, args),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(gl.jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.out_md.write_text(to_markdown(result))
    if args.out_splits:
        split_rows = []
        for split in result["heldout_seed"]["splits"]:
            split_rows.append(
                {
                    "test_seed": split["test_seed"],
                    "train_seeds": split["train_seeds"],
                    "selected_rule_train": split["selected_rule_train"],
                    "test_result": split["test_result"],
                }
            )
        write_jsonl(args.out_splits, split_rows)
    print(json.dumps(gl.jsonable(result["heldout_seed"]["aggregate"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
