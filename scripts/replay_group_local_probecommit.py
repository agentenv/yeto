#!/usr/bin/env python3
"""Evaluate group-local ProbeCommit replay policies from offline artifacts.

This script does not apply model deltas or run model probes. It joins per-candidate
feature records with already computed per-group policy replay records, then asks
whether group-local confidence features can decide when to use an existing exact
action such as `anchor_drop_bottom25`, `anchor_shrink`, or the token-weighted
fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


BASE_POLICIES = (
    "token_weighted",
    "freshness_weighted",
    "anchor_drop_bottom25",
    "anchor_positive_threshold",
    "anchor_shrink",
    "probecommit_v1",
    "oracle_positive",
    "oracle_topk",
    "random_probecommit_count",
)

TUNABLE_BASE_POLICIES = (
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
            f.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")


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


def std(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0 if vals else float("nan")
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def quantile(values: list[float], q: float) -> float:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def safe_bool(value) -> bool:
    return bool(value) if value is not None else False


def infer_seed(path: Path, row: dict | None = None) -> int | None:
    if row is not None and row.get("seed") is not None:
        return int(row["seed"])
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def feature_key(row: dict, source_path: Path) -> tuple[int, int, int]:
    seed = infer_seed(source_path, row)
    if seed is None:
        raise ValueError(f"cannot infer seed for feature row from {source_path}")
    step = int(row.get("pull_step", row.get("syncer_global_step", 0)))
    return seed, step, int(row["fragment"])


def replay_key(row: dict, source_path: Path) -> tuple[int, int, int]:
    seed = infer_seed(source_path, row)
    if seed is None:
        raise ValueError(f"cannot infer seed for replay row from {source_path}")
    return seed, int(row["step"]), int(row["fragment"])


def score_value(row: dict, field: str) -> float:
    return float(row.get(field, 0.0))


def token_weight(row: dict) -> float:
    return max(float(row.get("weight", row.get("c_tokens", 1.0))), 0.0)


def selected_mass(rows: list[dict], all_rows: list[dict]) -> float:
    denom = sum(token_weight(r) for r in all_rows)
    if denom <= 0.0:
        return 0.0
    return sum(token_weight(r) for r in rows) / denom


def weighted_utility(rows: list[dict]) -> float:
    total = sum(token_weight(r) for r in rows)
    if total <= 0.0:
        return float("nan")
    return sum(token_weight(r) * float(r["utility"]) for r in rows) / total


def pairwise_concordance(rows: list[dict], field: str) -> tuple[int, float]:
    pairs = 0
    wins = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ds = score_value(rows[i], field) - score_value(rows[j], field)
            du = float(rows[i]["utility"]) - float(rows[j]["utility"])
            if ds == 0.0 or du == 0.0:
                continue
            pairs += 1
            if ds * du > 0.0:
                wins += 1
    return pairs, wins / pairs if pairs else float("nan")


def good_auc(rows: list[dict], field: str) -> float:
    pairs = 0
    wins = 0.0
    good = [r for r in rows if not safe_bool(r.get("bad"))]
    bad = [r for r in rows if safe_bool(r.get("bad"))]
    if not good or not bad:
        return float("nan")
    for g in good:
        for b in bad:
            pairs += 1
            gs = score_value(g, field)
            bs = score_value(b, field)
            if gs > bs:
                wins += 1.0
            elif gs == bs:
                wins += 0.5
    return wins / pairs if pairs else float("nan")


def group_stats(rows: list[dict], field: str) -> dict:
    ordered = sorted(rows, key=lambda r: score_value(r, field), reverse=True)
    scores = [score_value(r, field) for r in ordered]
    utilities = [float(r["utility"]) for r in ordered]
    top = ordered[0]
    bottom = ordered[-1]
    q25 = quantile(scores, 0.25)
    q75 = quantile(scores, 0.75)
    score_std = std(scores)
    pairs, concordance = pairwise_concordance(rows, field)
    top_half = ordered[: max(1, math.ceil(len(ordered) * 0.5))]
    drop25 = ordered[: max(1, math.ceil(len(ordered) * 0.75))]
    token = weighted_utility(rows)
    return {
        "score_mean": mean(scores),
        "score_std": score_std,
        "score_min": min(scores),
        "score_max": max(scores),
        "score_spread": max(scores) - min(scores),
        "score_iqr": q75 - q25,
        "top_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "top_gap_z": 0.0 if score_std <= 0.0 else (scores[0] - scores[1]) / score_std,
        "top1_utility": float(top["utility"]),
        "bottom1_utility": float(bottom["utility"]),
        "top1_bad": safe_bool(top.get("bad")),
        "bottom1_bad": safe_bool(bottom.get("bad")),
        "top1_strict_bad": safe_bool(top.get("bad_strict")),
        "bottom1_strict_bad": safe_bool(bottom.get("bad_strict")),
        "candidate_bad_rate": mean([1.0 if safe_bool(r.get("bad")) else 0.0 for r in rows]),
        "candidate_strict_bad_rate": mean(
            [1.0 if safe_bool(r.get("bad_strict")) else 0.0 for r in rows]
        ),
        "pairwise_pairs": pairs,
        "pairwise_concordance": concordance,
        "linear_token_utility": token,
        "linear_top50_utility": weighted_utility(top_half),
        "linear_drop25_utility": weighted_utility(drop25),
        "linear_top50_gain": weighted_utility(top_half) - token,
        "linear_drop25_gain": weighted_utility(drop25) - token,
        "linear_drop25_selected_mass": selected_mass(drop25, rows),
    }


def load_groups(feature_paths: list[Path], replay_paths: list[Path], score_fields: list[str]) -> list[dict]:
    feature_groups: dict[tuple[int, int, int], list[dict]] = {}
    replay_groups: dict[tuple[int, int, int], dict] = {}
    for path in feature_paths:
        for row in read_jsonl(path):
            key = feature_key(row, path)
            feature_groups.setdefault(key, []).append(row)
    for path in replay_paths:
        for row in read_jsonl(path):
            replay_groups[replay_key(row, path)] = row

    groups = []
    missing_replay = 0
    for key, rows in sorted(feature_groups.items()):
        replay = replay_groups.get(key)
        if replay is None:
            missing_replay += 1
            continue
        if len(rows) < 2:
            continue
        seed, step, fragment = key
        record = {
            "seed": seed,
            "step": step,
            "fragment": fragment,
            "candidate_count": len(rows),
            "replay": replay,
            "candidates": sorted(rows, key=lambda r: int(r.get("learner_id", 0))),
            "stats": {field: group_stats(rows, field) for field in score_fields},
        }
        groups.append(record)
    if not groups:
        raise SystemExit("no joined feature/replay groups")
    return groups


def action_metrics(row: dict, action: str) -> dict:
    replay = row["replay"]
    return {
        "action": action,
        "utility": float(replay[f"{action}_utility"]),
        "negative": safe_bool(replay.get(f"{action}_negative")),
        "strict_negative": safe_bool(replay.get(f"{action}_strict_negative")),
        "selected_mass": float(replay.get(f"{action}_selected_mass", 1.0)),
        "selected_count": float(replay.get(f"{action}_selected_count", row["candidate_count"])),
    }


@dataclass(frozen=True)
class Rule:
    name: str
    family: str
    field: str
    params: dict


def decide(rule: Rule, row: dict) -> str:
    s = row["stats"][rule.field]
    if rule.family == "base":
        return rule.params["action"]
    if rule.family == "drop25_if_spread_ge":
        return "anchor_drop_bottom25" if s["score_spread"] >= rule.params["spread"] else "token_weighted"
    if rule.family == "drop25_if_iqr_ge":
        return "anchor_drop_bottom25" if s["score_iqr"] >= rule.params["iqr"] else "token_weighted"
    if rule.family == "drop25_if_top_gap_ge":
        return "anchor_drop_bottom25" if s["top_gap"] >= rule.params["gap"] else "token_weighted"
    if rule.family == "positive_threshold_if_spread_ge":
        return (
            "anchor_positive_threshold"
            if s["score_spread"] >= rule.params["spread"]
            else "token_weighted"
        )
    if rule.family == "shrink_if_mean_lt":
        return "anchor_shrink" if s["score_mean"] < rule.params["mean"] else "token_weighted"
    if rule.family == "drop_or_shrink":
        if s["score_mean"] < rule.params["mean"]:
            return "anchor_shrink"
        if s["score_spread"] >= rule.params["spread"]:
            return "anchor_drop_bottom25"
        return "token_weighted"
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
                "token_utility": token["utility"],
                "oracle_positive_utility": oracle["utility"],
                "gain_vs_token": m["utility"] - token["utility"],
                "gain_positive": m["utility"] > token["utility"],
                "oracle_positive_headroom_captured": captured,
            }
        )
    captured_vals = [
        float(m["oracle_positive_headroom_captured"])
        for m in metrics
        if m["oracle_positive_headroom_captured"] is not None
    ]
    token_neg = mean([1.0 if action_metrics(row, "token_weighted")["negative"] else 0.0 for row in rows])
    token_strict = mean(
        [1.0 if action_metrics(row, "token_weighted")["strict_negative"] else 0.0 for row in rows]
    )
    neg = mean([1.0 if m["negative"] else 0.0 for m in metrics])
    strict = mean([1.0 if m["strict_negative"] else 0.0 for m in metrics])
    return {
        "rule": rule.name,
        "family": rule.family,
        "field": rule.field,
        "params": rule.params,
        "groups": len(rows),
        "mean_utility": mean([m["utility"] for m in metrics]),
        "mean_gain_vs_token": mean([m["gain_vs_token"] for m in metrics]),
        "median_gain_vs_token": quantile([m["gain_vs_token"] for m in metrics], 0.5),
        "gain_positive_rate": mean([1.0 if m["gain_positive"] else 0.0 for m in metrics]),
        "negative_rate": neg,
        "negative_rate_relative_drop": None if token_neg <= 0.0 else (token_neg - neg) / token_neg,
        "strict_negative_rate": strict,
        "strict_negative_rate_relative_drop": (
            None if token_strict <= 0.0 else (token_strict - strict) / token_strict
        ),
        "selected_mass_mean": mean([m["selected_mass"] for m in metrics]),
        "selected_count_mean": mean([m["selected_count"] for m in metrics]),
        "oracle_positive_headroom_captured": mean(captured_vals),
        "headroom_excluded_fraction": 1.0 - len(captured_vals) / len(metrics),
        "actions": dict(sorted(actions.items())),
    }


def objective(result: dict, negative_penalty: float, strict_penalty: float) -> float:
    return (
        float(result["mean_utility"])
        - negative_penalty * float(result["negative_rate"])
        - strict_penalty * float(result["strict_negative_rate"])
    )


def threshold_values(rows: list[dict], field: str, stat_name: str) -> list[float]:
    values = [float(row["stats"][field][stat_name]) for row in rows]
    qs = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    out = sorted({quantile(values, q) for q in qs if math.isfinite(quantile(values, q))})
    return out or [0.0]


def candidate_rules(train_rows: list[dict], fields: list[str]) -> list[Rule]:
    rules = []
    for action in TUNABLE_BASE_POLICIES:
        rules.append(Rule(name=action, family="base", field=fields[0], params={"action": action}))
    for field in fields:
        for spread in threshold_values(train_rows, field, "score_spread"):
            rules.append(
                Rule(
                    name=f"{field}:drop25_if_spread_ge:{spread:.6g}",
                    family="drop25_if_spread_ge",
                    field=field,
                    params={"spread": spread},
                )
            )
            rules.append(
                Rule(
                    name=f"{field}:positive_threshold_if_spread_ge:{spread:.6g}",
                    family="positive_threshold_if_spread_ge",
                    field=field,
                    params={"spread": spread},
                )
            )
        for iqr in threshold_values(train_rows, field, "score_iqr"):
            rules.append(
                Rule(
                    name=f"{field}:drop25_if_iqr_ge:{iqr:.6g}",
                    family="drop25_if_iqr_ge",
                    field=field,
                    params={"iqr": iqr},
                )
            )
        for gap in threshold_values(train_rows, field, "top_gap"):
            rules.append(
                Rule(
                    name=f"{field}:drop25_if_top_gap_ge:{gap:.6g}",
                    family="drop25_if_top_gap_ge",
                    field=field,
                    params={"gap": gap},
                )
            )
        for mean_threshold in threshold_values(train_rows, field, "score_mean"):
            rules.append(
                Rule(
                    name=f"{field}:shrink_if_mean_lt:{mean_threshold:.6g}",
                    family="shrink_if_mean_lt",
                    field=field,
                    params={"mean": mean_threshold},
                )
            )
        spreads = threshold_values(train_rows, field, "score_spread")
        means = threshold_values(train_rows, field, "score_mean")
        for spread in spreads[:: max(1, len(spreads) // 4)]:
            for mean_threshold in means[:: max(1, len(means) // 4)]:
                rules.append(
                    Rule(
                        name=f"{field}:drop_or_shrink:{spread:.6g}:{mean_threshold:.6g}",
                        family="drop_or_shrink",
                        field=field,
                        params={"spread": spread, "mean": mean_threshold},
                    )
                )
    return rules


def summarize_score_fields(groups: list[dict], fields: list[str]) -> dict:
    out = {}
    all_candidates = []
    for group in groups:
        all_candidates.extend(group["candidates"])
    for field in fields:
        pair_counts = []
        pair_weighted = []
        top_bad = []
        bottom_bad = []
        top_strict = []
        bottom_strict = []
        top_utility = []
        bottom_utility = []
        linear_drop_gain = []
        for group in groups:
            stats = group["stats"][field]
            if math.isfinite(stats["pairwise_concordance"]):
                pair_counts.append(stats["pairwise_pairs"])
                pair_weighted.append(stats["pairwise_pairs"] * stats["pairwise_concordance"])
            top_bad.append(1.0 if stats["top1_bad"] else 0.0)
            bottom_bad.append(1.0 if stats["bottom1_bad"] else 0.0)
            top_strict.append(1.0 if stats["top1_strict_bad"] else 0.0)
            bottom_strict.append(1.0 if stats["bottom1_strict_bad"] else 0.0)
            top_utility.append(stats["top1_utility"])
            bottom_utility.append(stats["bottom1_utility"])
            linear_drop_gain.append(stats["linear_drop25_gain"])
        total_pairs = sum(pair_counts)
        out[field] = {
            "candidate_good_auc": good_auc(all_candidates, field),
            "groups": len(groups),
            "pairwise_pairs": total_pairs,
            "pairwise_concordance": (
                sum(pair_weighted) / total_pairs if total_pairs else float("nan")
            ),
            "top1_bad_rate": mean(top_bad),
            "bottom1_bad_rate": mean(bottom_bad),
            "top1_strict_bad_rate": mean(top_strict),
            "bottom1_strict_bad_rate": mean(bottom_strict),
            "top1_utility_mean": mean(top_utility),
            "bottom1_utility_mean": mean(bottom_utility),
            "linear_drop25_gain_mean": mean(linear_drop_gain),
            "linear_drop25_gain_positive_rate": mean(
                [1.0 if gain > 0.0 else 0.0 for gain in linear_drop_gain]
            ),
        }
    return out


def heldout_seed_replay(
    groups: list[dict],
    fields: list[str],
    *,
    negative_penalty: float,
    strict_penalty: float,
) -> dict:
    seeds = sorted({int(g["seed"]) for g in groups})
    splits = []
    for test_seed in seeds:
        train = [g for g in groups if int(g["seed"]) != test_seed]
        test = [g for g in groups if int(g["seed"]) == test_seed]
        rules = candidate_rules(train, fields)
        train_results = [evaluate_rule(rule, train) for rule in rules]
        train_results.sort(
            key=lambda r: (
                objective(r, negative_penalty, strict_penalty),
                r["mean_utility"],
                -r["negative_rate"],
            ),
            reverse=True,
        )
        best_rule = next(rule for rule in rules if rule.name == train_results[0]["rule"])
        test_result = evaluate_rule(best_rule, test)
        split = {
            "test_seed": test_seed,
            "train_seeds": [s for s in seeds if s != test_seed],
            "train_groups": len(train),
            "test_groups": len(test),
            "selected_rule": train_results[0],
            "test_result": test_result,
            "top_train_rules": train_results[:10],
        }
        splits.append(split)
    aggregate = {
        "splits": len(splits),
        "mean_utility": mean([s["test_result"]["mean_utility"] for s in splits]),
        "mean_gain_vs_token": mean([s["test_result"]["mean_gain_vs_token"] for s in splits]),
        "all_test_gains_positive": all(
            s["test_result"]["mean_gain_vs_token"] > 0.0 for s in splits
        ),
        "negative_rate": mean([s["test_result"]["negative_rate"] for s in splits]),
        "negative_rate_relative_drop": mean(
            [s["test_result"]["negative_rate_relative_drop"] for s in splits]
        ),
        "strict_negative_rate": mean([s["test_result"]["strict_negative_rate"] for s in splits]),
        "strict_negative_rate_relative_drop": mean(
            [s["test_result"]["strict_negative_rate_relative_drop"] for s in splits]
        ),
        "selected_mass_mean": mean([s["test_result"]["selected_mass_mean"] for s in splits]),
        "oracle_positive_headroom_captured": mean(
            [s["test_result"]["oracle_positive_headroom_captured"] for s in splits]
        ),
    }
    aggregate["gate_pass"] = bool(
        aggregate["all_test_gains_positive"]
        and aggregate["mean_gain_vs_token"] > 0.0
        and aggregate["oracle_positive_headroom_captured"] >= 0.40
        and aggregate["negative_rate_relative_drop"] >= 0.20
        and aggregate["selected_mass_mean"] >= 0.40
    )
    return {"splits": splits, "aggregate": aggregate}


def group_records(groups: list[dict], fields: list[str]) -> list[dict]:
    rows = []
    for group in groups:
        replay = group["replay"]
        row = {
            "seed": group["seed"],
            "step": group["step"],
            "fragment": group["fragment"],
            "candidate_count": group["candidate_count"],
            "token_weighted_utility": float(replay["token_weighted_utility"]),
            "token_weighted_negative": safe_bool(replay.get("token_weighted_negative")),
            "oracle_positive_utility": float(replay["oracle_positive_utility"]),
            "anchor_drop_bottom25_utility": float(replay["anchor_drop_bottom25_utility"]),
            "probecommit_v1_utility": float(replay["probecommit_v1_utility"]),
        }
        for field in fields:
            for key, value in group["stats"][field].items():
                if key in {
                    "score_mean",
                    "score_std",
                    "score_spread",
                    "score_iqr",
                    "top_gap",
                    "top_gap_z",
                    "pairwise_concordance",
                    "linear_drop25_gain",
                    "linear_drop25_selected_mass",
                }:
                    row[f"{field}_{key}"] = value
        rows.append(row)
    return rows


def to_markdown(result: dict) -> str:
    def fmt(value, digits: int = 3) -> str:
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

    lines = ["# EXP2.6 Group-Local ProbeCommit Offline Replay", ""]
    lines.append("## Result Summary")
    lines.append("")
    lines.append(
        f"- Joined `{result['groups']}` complete groups from `{len(result['seeds'])}` seeds."
    )
    agg = result["heldout_seed"]["aggregate"]
    lines.append(
        f"- Best train-seed-tuned group-local rule mean test gain: `{agg['mean_gain_vs_token']:.6f}`."
    )
    lines.append(
        f"- Mean oracle-positive headroom captured: `{agg['oracle_positive_headroom_captured']:.3f}`."
    )
    lines.append(
        f"- Mean negative-rate relative drop: `{agg['negative_rate_relative_drop']:.3f}`."
    )
    lines.append(f"- Gate pass: `{agg['gate_pass']}`.")
    lines.append("")
    lines.append("Decision: do not start online ProbeCommit from these group-local rules.")
    lines.append("")

    lines.append("## Group-Local Score Diagnostics")
    lines.append("")
    lines.append(
        "| Score | Candidate good AUC | Pairwise concordance | Top1 bad | Bottom1 bad | Top1 strict bad | Bottom1 strict bad | Linear drop25 gain |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for field, row in result["score_diagnostics"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
                field,
                fmt(row["candidate_good_auc"]),
                fmt(row["pairwise_concordance"]),
                fmt(row["top1_bad_rate"]),
                fmt(row["bottom1_bad_rate"]),
                fmt(row["top1_strict_bad_rate"]),
                fmt(row["bottom1_strict_bad_rate"]),
                fmt(row["linear_drop25_gain_mean"], 6),
            )
        )
    lines.append("")

    lines.append("## Held-Out Seed Replay")
    lines.append("")
    lines.append(
        "| Test seed | Selected rule | Test groups | Utility gain | Negative drop | Strict drop | Headroom captured | Selected mass | Actions |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for split in result["heldout_seed"]["splits"]:
        tr = split["test_result"]
        lines.append(
            "| {} | `{}` | {} | {} | {} | {} | {} | {} | `{}` |".format(
                split["test_seed"],
                tr["rule"],
                tr["groups"],
                fmt(tr["mean_gain_vs_token"], 6),
                fmt(tr["negative_rate_relative_drop"]),
                fmt(tr["strict_negative_rate_relative_drop"]),
                fmt(tr["oracle_positive_headroom_captured"]),
                fmt(tr["selected_mass_mean"]),
                tr["actions"],
            )
        )
    lines.append("")

    lines.append("## Gate")
    lines.append("")
    lines.append("| Gate | Value |")
    lines.append("|---|---:|")
    for key, value in agg.items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The extra replay confirms the EXP2.5 diagnosis: group-local confidence rules are not yet strong enough. "
        "The tuned rules can reduce some strict-negative events, but they do not recover enough oracle headroom "
        "and do not pass all held-out seeds with a meaningful margin."
    )
    lines.append("")
    lines.append(
        "The next offline improvement should target stronger within-group evidence rather than another global "
        "candidate classifier: more complete groups, richer group-normalized features, and abstain/wait policies "
        "that only act when score separation is clearly reliable."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", nargs="+", required=True, type=Path)
    p.add_argument("--policy-replay", nargs="+", required=True, type=Path)
    p.add_argument(
        "--score-fields",
        default="probe_grad_dot,calibrated_score,freshness,combined_score",
        help="Comma-separated candidate score fields to analyze and tune.",
    )
    p.add_argument("--negative-penalty", type=float, default=0.0005)
    p.add_argument("--strict-penalty", type=float, default=0.0005)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-records", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fields = [field.strip() for field in args.score_fields.split(",") if field.strip()]
    groups = load_groups(args.features, args.policy_replay, fields)
    result = {
        "schema": "group_local_probecommit_replay_v1",
        "groups": len(groups),
        "seeds": sorted({int(g["seed"]) for g in groups}),
        "score_fields": fields,
        "score_diagnostics": summarize_score_fields(groups, fields),
        "heldout_seed": heldout_seed_replay(
            groups,
            fields,
            negative_penalty=args.negative_penalty,
            strict_penalty=args.strict_penalty,
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    args.out_md.write_text(to_markdown(result))
    if args.out_records is not None:
        write_jsonl(args.out_records, group_records(groups, fields))
    print(
        json.dumps(
            jsonable(result["heldout_seed"]["aggregate"]),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
