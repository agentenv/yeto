#!/usr/bin/env python3
"""Search repeated-anchor action policies on action-probe replay records.

The input JSONL must contain `{action}_anchor_batch_utilities`, which lets us
estimate action utility uncertainty from multiple anchor batches. Oracle fields
are used only for held-out evaluation, never for per-group action selection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from itertools import combinations, product
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


def std(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def se(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    return std(vals) / math.sqrt(len(vals))


def infer_seed(path: Path, rows: list[dict]) -> int:
    for row in rows:
        if row.get("seed") is not None:
            return int(row["seed"])
    match = re.search(r"seed(\d+)", str(path))
    if not match:
        raise SystemExit(f"could not infer seed from {path}")
    return int(match.group(1))


def batch_values(row: dict, action: str) -> list[float]:
    key = f"{action}_anchor_batch_utilities"
    if key not in row:
        raise KeyError(key)
    return [float(v) for v in row[key]]


def paired_gain_values(row: dict, action: str) -> list[float]:
    token = batch_values(row, "token_weighted")
    action_values = batch_values(row, action)
    return [a - t for a, t in zip(action_values, token)]


def metric(row: dict, action: str, config: dict) -> dict:
    cache = row.setdefault("_metric_cache", {})
    cache_key = (action, float(config["lcb_k"]))
    if cache_key in cache:
        return dict(cache[cache_key])
    vals = batch_values(row, action)
    gains = paired_gain_values(row, action)
    k = float(config["lcb_k"])
    out = {
        "mean": mean(vals),
        "se": se(vals),
        "lcb": mean(vals) - k * se(vals),
        "ucb": mean(vals) + k * se(vals),
        "gain_mean": mean(gains),
        "gain_se": se(gains),
        "gain_lcb": mean(gains) - k * se(gains),
        "win_rate": mean([1.0 if g > 0.0 else 0.0 for g in gains]),
        "selected_mass": float(row.get(f"{action}_selected_mass", 1.0)),
    }
    cache[cache_key] = dict(out)
    return out


def subset_top_fraction(row: dict, action: str, actions: tuple[str, ...], subset_size: int | None = None) -> float:
    cache = row.setdefault("_subset_top_cache", {})
    cache_key = (action, tuple(actions), subset_size)
    if cache_key in cache:
        return float(cache[cache_key])
    values_by_action = {
        a: batch_values(row, a)
        for a in actions
        if f"{a}_anchor_batch_utilities" in row
    }
    if action not in values_by_action:
        cache[cache_key] = 0.0
        return 0.0
    batches = len(values_by_action[action])
    if batches <= 1:
        cache[cache_key] = 1.0
        return 1.0
    if subset_size is None:
        subset_size = max(1, batches // 2)
    subset_size = max(1, min(subset_size, batches))
    wins = 0
    total = 0
    for subset in combinations(range(batches), subset_size):
        means = {
            a: mean([vals[i] for i in subset])
            for a, vals in values_by_action.items()
            if len(vals) == batches
        }
        if not means:
            continue
        top_value = max(means.values())
        top_actions = [a for a, v in means.items() if v == top_value]
        total += 1
        if action in top_actions:
            wins += 1 / len(top_actions)
    out = wins / total if total else 0.0
    cache[cache_key] = out
    return out


def choose(row: dict, actions: tuple[str, ...], config: dict) -> str:
    variant = config["variant"]
    margin = float(config["margin"])
    min_mass = float(config["min_mass"])
    min_win = float(config["min_win"])
    min_subset_top = float(config.get("min_subset_top", 0.0))
    fallback = str(config["fallback"])
    non_token = [a for a in actions if a != "token_weighted"]
    candidates = []
    for action in non_token:
        if f"{action}_anchor_batch_utilities" not in row:
            continue
        m = metric(row, action, config)
        if m["selected_mass"] < min_mass:
            continue
        m["subset_top_frac"] = subset_top_fraction(row, action, actions)
        candidates.append((action, m))
    if not candidates:
        return fallback

    token = metric(row, "token_weighted", config)

    if variant == "paired_lcb":
        best_action, best_metric = max(candidates, key=lambda item: item[1]["gain_lcb"])
        if (
            best_metric["gain_lcb"] > margin
            and best_metric["win_rate"] >= min_win
            and best_metric["subset_top_frac"] >= min_subset_top
        ):
            return best_action
        return fallback

    if variant == "lcb_vs_token_ucb":
        best_action, best_metric = max(candidates, key=lambda item: item[1]["lcb"])
        if (
            best_metric["lcb"] - token["ucb"] > margin
            and best_metric["win_rate"] >= min_win
            and best_metric["subset_top_frac"] >= min_subset_top
        ):
            return best_action
        return fallback

    if variant == "mean_gain_and_consensus":
        best_action, best_metric = max(candidates, key=lambda item: item[1]["gain_mean"])
        if (
            best_metric["gain_mean"] > margin
            and best_metric["win_rate"] >= min_win
            and best_metric["subset_top_frac"] >= min_subset_top
        ):
            return best_action
        return fallback

    if variant == "batch_consensus_lcb":
        passing = [
            (action, m)
            for action, m in candidates
            if m["gain_lcb"] >= margin
            and m["gain_mean"] >= float(config.get("min_mean_gain", margin))
            and m["win_rate"] >= min_win
            and m["subset_top_frac"] >= min_subset_top
        ]
        if passing:
            return max(passing, key=lambda item: item[1]["gain_lcb"])[0]
        return fallback

    if variant == "batch_consensus_mean":
        passing = [
            (action, m)
            for action, m in candidates
            if m["gain_mean"] >= margin
            and m["win_rate"] >= min_win
            and m["subset_top_frac"] >= min_subset_top
        ]
        if passing:
            return max(passing, key=lambda item: (item[1]["win_rate"], item[1]["gain_mean"]))[0]
        return fallback

    if variant == "safe_shrink":
        # Only shrink when token looks bad on most anchor batches and the
        # shrink action is at least not worse than token by paired LCB.
        token_vals = batch_values(row, "token_weighted")
        token_bad_rate = mean([1.0 if v < 0.0 else 0.0 for v in token_vals])
        if token_bad_rate >= min_win and "anchor_shrink" in actions:
            shrink = metric(row, "anchor_shrink", config)
            if shrink["selected_mass"] >= min_mass and shrink["gain_lcb"] > -abs(margin):
                return "anchor_shrink"
        return fallback

    raise ValueError(f"unknown variant {variant}")


def summarize(records: list[dict], choices: list[str]) -> dict:
    token_neg = mean([1.0 if r["token_weighted_oracle_negative"] else 0.0 for r in records])
    token_strict = mean(
        [
            1.0 if r["token_weighted_oracle_strict_negative"] else 0.0
            for r in records
            if r.get("token_weighted_oracle_strict_negative") is not None
        ]
    )
    gains = []
    negs = []
    stricts = []
    captured = []
    masses = []
    for row, action in zip(records, choices):
        utility = float(row[f"{action}_oracle_utility"])
        token = float(row["token_weighted_oracle_utility"])
        oracle = float(row["oracle_positive_oracle_utility"])
        gains.append(utility - token)
        negs.append(1.0 if row[f"{action}_oracle_negative"] else 0.0)
        strict_value = row.get(f"{action}_oracle_strict_negative")
        if strict_value is not None:
            stricts.append(1.0 if strict_value else 0.0)
        masses.append(float(row.get(f"{action}_selected_mass", 1.0)))
        denom = oracle - token
        if denom > 0.0:
            captured.append((utility - token) / denom)
    neg_rate = mean(negs)
    strict_rate = mean(stricts)
    return {
        "records": len(records),
        "mean_gain_vs_token": mean(gains),
        "gain_positive_rate": mean([1.0 if g > 0.0 else 0.0 for g in gains]),
        "negative_rate": neg_rate,
        "strict_negative_rate": strict_rate,
        "negative_rate_relative_drop": None if token_neg <= 0.0 else (token_neg - neg_rate) / token_neg,
        "strict_negative_rate_relative_drop": None
        if token_strict <= 0.0
        else (token_strict - strict_rate) / token_strict,
        "oracle_positive_headroom_captured": mean(captured),
        "selected_mass_mean": mean(masses),
        "chosen_action_distribution": dict(sorted(Counter(choices).items())),
        "acted_fraction": mean([1.0 if c != "token_weighted" else 0.0 for c in choices]),
        "token_negative_rate": token_neg,
        "token_strict_negative_rate": token_strict,
    }


def evaluate(records: list[dict], actions: tuple[str, ...], config: dict) -> dict:
    choices = [choose(row, actions, config) for row in records]
    out = summarize(records, choices)
    out["config"] = dict(config)
    return out


def grid() -> list[dict]:
    out = []
    for variant, lcb_k, margin, min_mass, min_win, fallback in product(
        (
            "paired_lcb",
            "mean_gain_and_consensus",
            "batch_consensus_lcb",
            "batch_consensus_mean",
            "safe_shrink",
        ),
        (0.0, 1.0, 2.0),
        (0.0, 0.00025, 0.0005, 0.001, 0.002),
        (0.67, 0.8, 0.9, 1.0),
        (0.5, 0.75, 1.0),
        ("token_weighted",),
    ):
        subset_thresholds = (0.0, 0.75, 1.0)
        mean_gains = (margin,) if variant != "batch_consensus_lcb" else (0.0, 0.00025, 0.0005, 0.001)
        for min_subset_top, min_mean_gain in product(subset_thresholds, mean_gains):
            out.append(
                {
                    "variant": variant,
                    "lcb_k": lcb_k,
                    "margin": margin,
                    "min_mass": min_mass,
                    "min_win": min_win,
                    "min_subset_top": min_subset_top,
                    "min_mean_gain": min_mean_gain,
                    "fallback": fallback,
                }
            )
    return out


def objective(summary: dict, min_mass_target: float) -> float:
    gain = float(summary["mean_gain_vs_token"])
    neg = summary["negative_rate_relative_drop"]
    strict = summary["strict_negative_rate_relative_drop"]
    head = float(summary["oracle_positive_headroom_captured"])
    mass = float(summary["selected_mass_mean"])
    neg = -1.0 if neg is None else float(neg)
    strict = -0.25 if strict is None else float(strict)
    return 120.0 * gain + 1.2 * neg + 0.8 * strict + 0.25 * head - 2.0 * max(0, min_mass_target - mass)


def heldout_search(by_seed: dict[int, list[dict]], actions: tuple[str, ...]) -> dict:
    seeds = sorted(by_seed)
    configs = grid()
    folds = []
    for test_seed in seeds:
        train = [r for s in seeds if s != test_seed for r in by_seed[s]]
        test = by_seed[test_seed]
        scored = []
        for config in configs:
            train_summary = evaluate(train, actions, config)
            scored.append((objective(train_summary, 0.85), train_summary, config))
        scored.sort(key=lambda item: item[0], reverse=True)
        _, train_summary, config = scored[0]
        test_summary = evaluate(test, actions, config)
        folds.append(
            {
                "test_seed": test_seed,
                "train_seeds": [s for s in seeds if s != test_seed],
                "config": config,
                "train_summary": train_summary,
                "test_summary": test_summary,
            }
        )
    aggregate = {
        "mean_gain_vs_token": mean([f["test_summary"]["mean_gain_vs_token"] for f in folds]),
        "all_seed_gains_positive": all(f["test_summary"]["mean_gain_vs_token"] > 0.0 for f in folds),
        "negative_rate_relative_drop": mean([f["test_summary"]["negative_rate_relative_drop"] for f in folds]),
        "strict_negative_rate_relative_drop": mean(
            [f["test_summary"]["strict_negative_rate_relative_drop"] for f in folds]
        ),
        "oracle_positive_headroom_captured": mean(
            [f["test_summary"]["oracle_positive_headroom_captured"] for f in folds]
        ),
        "selected_mass_mean": mean([f["test_summary"]["selected_mass_mean"] for f in folds]),
        "acted_fraction": mean([f["test_summary"]["acted_fraction"] for f in folds]),
        "chosen_action_distribution": dict(
            sorted(
                sum(
                    (Counter(f["test_summary"]["chosen_action_distribution"]) for f in folds),
                    Counter(),
                ).items()
            )
        ),
    }
    gates = {
        "mean_gain_ge_0.0005": aggregate["mean_gain_vs_token"] >= 0.0005,
        "all_seeds_positive": aggregate["all_seed_gains_positive"],
        "negative_drop_ge_0.20": aggregate["negative_rate_relative_drop"] >= 0.20,
        "headroom_captured_ge_0.40": aggregate["oracle_positive_headroom_captured"] >= 0.40,
        "selected_mass_ge_0.85": aggregate["selected_mass_mean"] >= 0.85,
    }
    gates["gate_pass"] = all(gates.values())
    return {
        "schema": "repeated_anchor_policy_search_v1",
        "seeds": seeds,
        "records": sum(len(v) for v in by_seed.values()),
        "actions": list(actions),
        "aggregate": aggregate,
        "gates": gates,
        "folds": folds,
    }


def fmt(value, digits=3) -> str:
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
    lines = ["# Repeated-Anchor Policy Search", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Gate pass: `{result['gates']['gate_pass']}`")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key, value in result["aggregate"].items():
        if key == "chosen_action_distribution":
            continue
        lines.append(f"| `{key}` | {fmt(value, 6)} |")
    lines.append("")
    lines.append("| Test seed | Gain | Neg drop | Strict drop | Headroom | Mass | Config |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for fold in result["folds"]:
        s = fold["test_summary"]
        c = fold["config"]
        compact = (
            f"{c['variant']}, k={c['lcb_k']}, margin={c['margin']}, "
            f"mass>={c['min_mass']}, win>={c['min_win']}, "
            f"top>={c.get('min_subset_top', 0)}, mean>={c.get('min_mean_gain', c['margin'])}, "
            f"fb={c['fallback']}"
        )
        lines.append(
            "| {} | {} | {} | {} | {} | {} | `{}` |".format(
                fold["test_seed"],
                fmt(s["mean_gain_vs_token"], 6),
                fmt(s["negative_rate_relative_drop"]),
                fmt(s["strict_negative_rate_relative_drop"]),
                fmt(s["oracle_positive_headroom_captured"]),
                fmt(s["selected_mass_mean"]),
                compact,
            )
        )
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    for key, value in result["gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
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
    by_seed = {}
    for path in args.replays:
        rows = read_jsonl(path)
        for action in args.actions:
            key = f"{action}_anchor_batch_utilities"
            if key not in rows[0]:
                raise SystemExit(f"{path}: missing {key}")
        by_seed[infer_seed(path, rows)] = rows
    result = heldout_search(by_seed, tuple(args.actions))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.out_md.write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
