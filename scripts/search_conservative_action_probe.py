#!/usr/bin/env python3
"""Search conservative action-probe selectors on replay JSONL artifacts.

This is intentionally offline-only. It never uses oracle utilities to make a
per-group decision; oracle fields are used only to score train/test splits.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import product
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


def infer_seed(path: Path, rows: list[dict]) -> int:
    for row in rows:
        if row.get("seed") is not None:
            return int(row["seed"])
    import re

    match = re.search(r"seed(\d+)", str(path))
    if not match:
        raise SystemExit(f"could not infer seed from {path}")
    return int(match.group(1))


def bool_field(row: dict, key: str) -> bool | None:
    value = row.get(key)
    if value is None:
        return None
    return bool(value)


def action_available(row: dict, action: str) -> bool:
    return f"{action}_anchor_utility" in row and f"{action}_oracle_utility" in row


def lcb(row: dict, action: str, k: float) -> float:
    utility = float(row[f"{action}_anchor_utility"])
    se = row.get(f"{action}_anchor_utility_se")
    if se is None:
        se = 0.0
    return utility - k * float(se)


def choose_action(row: dict, config: dict, actions: tuple[str, ...]) -> str:
    variant = config["variant"]
    k = float(config["lcb_k"])
    margin = float(config["margin"])
    min_mass = float(config["min_mass"])
    min_anchor = float(config["min_anchor"])
    require_non_strict = bool(config["require_non_strict"])
    fallback = str(config["fallback"])

    selectable = []
    for action in actions:
        if action == "token_weighted" or not action_available(row, action):
            continue
        if float(row.get(f"{action}_selected_mass", 1.0)) < min_mass:
            continue
        if require_non_strict and bool_field(row, f"{action}_anchor_strict_negative") is True:
            continue
        selectable.append(action)

    if not selectable:
        return fallback

    token_score = lcb(row, "token_weighted", k)
    best = max(selectable, key=lambda action: lcb(row, action, k))
    best_score = lcb(row, best, k)
    best_utility = float(row[f"{best}_anchor_utility"])
    token_utility = float(row["token_weighted_anchor_utility"])
    token_strict = bool_field(row, "token_weighted_anchor_strict_negative") is True
    token_negative = bool_field(row, "token_weighted_anchor_negative") is True

    if variant == "high_conf_gain":
        if best_utility >= min_anchor and best_score - token_score >= margin:
            return best
        return fallback

    if variant == "safety_veto":
        token_looks_bad = token_strict or token_negative or token_score < min_anchor
        if token_looks_bad and best_score >= min_anchor and best_score - token_score >= margin:
            return best
        return fallback

    if variant == "shrink_veto":
        if token_strict or token_score < min_anchor:
            return "anchor_shrink" if action_available(row, "anchor_shrink") else fallback
        return fallback

    if variant == "risk_lcb":
        if best_score >= min_anchor and best_score - token_score >= margin:
            return best
        if token_score < min_anchor and action_available(row, "anchor_shrink"):
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
        "headroom_excluded_fraction": 1.0 - (len(captured) / len(records) if records else 0.0),
        "selected_mass_mean": mean(masses),
        "chosen_action_distribution": dict(sorted(Counter(choices).items())),
        "token_negative_rate": token_neg,
        "token_strict_negative_rate": token_strict,
    }


def score_for_search(summary: dict, *, min_mass_target: float) -> float:
    gain = float(summary["mean_gain_vs_token"])
    neg_drop = summary["negative_rate_relative_drop"]
    strict_drop = summary["strict_negative_rate_relative_drop"]
    headroom = float(summary["oracle_positive_headroom_captured"])
    mass = float(summary["selected_mass_mean"])
    neg_drop = -1.0 if neg_drop is None else float(neg_drop)
    strict_drop = -1.0 if strict_drop is None else float(strict_drop)
    mass_penalty = max(0.0, min_mass_target - mass)
    # Safety-first: reward negative drops before mean utility. Keep a weak
    # gain/headroom term so degenerate shrink-only policies do not win.
    return (
        2.0 * neg_drop
        + 1.5 * strict_drop
        + 75.0 * gain
        + 0.25 * headroom
        - 2.0 * mass_penalty
    )


def evaluate_config(records: list[dict], config: dict, actions: tuple[str, ...]) -> tuple[dict, list[str]]:
    choices = [choose_action(row, config, actions) for row in records]
    return summarize(records, choices), choices


def config_grid() -> list[dict]:
    configs = []
    for variant, lcb_k, margin, min_mass, min_anchor, require_non_strict, fallback in product(
        ("high_conf_gain", "safety_veto", "shrink_veto", "risk_lcb"),
        (0.0, 0.5, 1.0, 2.0),
        (0.0, 0.00025, 0.0005, 0.001, 0.002),
        (0.5, 0.67, 0.8, 0.9, 1.0),
        (-0.002, -0.001, 0.0, 0.0005),
        (False, True),
        ("token_weighted", "anchor_shrink"),
    ):
        configs.append(
            {
                "variant": variant,
                "lcb_k": lcb_k,
                "margin": margin,
                "min_mass": min_mass,
                "min_anchor": min_anchor,
                "require_non_strict": require_non_strict,
                "fallback": fallback,
            }
        )
    return configs


def heldout_search(by_seed: dict[int, list[dict]], actions: tuple[str, ...]) -> dict:
    seeds = sorted(by_seed)
    configs = config_grid()
    folds = []
    for test_seed in seeds:
        train_records = [
            row
            for seed in seeds
            if seed != test_seed
            for row in by_seed[seed]
        ]
        test_records = by_seed[test_seed]
        scored = []
        for config in configs:
            train_summary, _ = evaluate_config(train_records, config, actions)
            scored.append(
                (
                    score_for_search(train_summary, min_mass_target=0.85),
                    train_summary,
                    config,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, train_summary, best_config = scored[0]
        test_summary, test_choices = evaluate_config(test_records, best_config, actions)
        folds.append(
            {
                "test_seed": test_seed,
                "train_seeds": [seed for seed in seeds if seed != test_seed],
                "search_score": best_score,
                "config": best_config,
                "train_summary": train_summary,
                "test_summary": test_summary,
                "test_choices": test_choices,
            }
        )
    aggregate = {
        "mean_gain_vs_token": mean([f["test_summary"]["mean_gain_vs_token"] for f in folds]),
        "all_seed_gains_positive": all(f["test_summary"]["mean_gain_vs_token"] > 0.0 for f in folds),
        "negative_rate_relative_drop": mean(
            [f["test_summary"]["negative_rate_relative_drop"] for f in folds]
        ),
        "strict_negative_rate_relative_drop": mean(
            [f["test_summary"]["strict_negative_rate_relative_drop"] for f in folds]
        ),
        "oracle_positive_headroom_captured": mean(
            [f["test_summary"]["oracle_positive_headroom_captured"] for f in folds]
        ),
        "selected_mass_mean": mean([f["test_summary"]["selected_mass_mean"] for f in folds]),
        "chosen_action_distribution": dict(
            sorted(
                sum((Counter(f["test_summary"]["chosen_action_distribution"]) for f in folds), Counter()).items()
            )
        ),
    }
    gates = {
        "mean_gain_ge_0.0005": aggregate["mean_gain_vs_token"] >= 0.0005,
        "all_seeds_positive": aggregate["all_seed_gains_positive"],
        "negative_drop_ge_0.20": aggregate["negative_rate_relative_drop"] >= 0.20,
        "strict_drop_ge_0.20": aggregate["strict_negative_rate_relative_drop"] >= 0.20,
        "headroom_captured_ge_0.40": aggregate["oracle_positive_headroom_captured"] >= 0.40,
        "selected_mass_ge_0.85": aggregate["selected_mass_mean"] >= 0.85,
    }
    gates["gate_pass"] = all(gates.values())
    return {
        "schema": "conservative_action_probe_search_v1",
        "seeds": seeds,
        "records": sum(len(rows) for rows in by_seed.values()),
        "actions": list(actions),
        "aggregate": aggregate,
        "gates": gates,
        "folds": folds,
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


def markdown(result: dict) -> str:
    lines = ["# Conservative Action-Probe Search", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Gate pass: `{result['gates']['gate_pass']}`")
    agg = result["aggregate"]
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for key in (
        "mean_gain_vs_token",
        "negative_rate_relative_drop",
        "strict_negative_rate_relative_drop",
        "oracle_positive_headroom_captured",
        "selected_mass_mean",
    ):
        lines.append(f"| `{key}` | {fmt(agg[key], 6)} |")
    lines.append("")
    lines.append("| Test seed | Gain | Negative drop | Strict drop | Headroom captured | Selected mass | Config |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for fold in result["folds"]:
        s = fold["test_summary"]
        c = fold["config"]
        compact = (
            f"{c['variant']}, k={c['lcb_k']}, margin={c['margin']}, "
            f"mass>={c['min_mass']}, min_anchor={c['min_anchor']}, fallback={c['fallback']}"
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
    actions = tuple(args.actions)
    by_seed = {}
    for path in args.replays:
        rows = read_jsonl(path)
        seed = infer_seed(path, rows)
        missing = [a for a in actions if not action_available(rows[0], a)]
        if missing:
            raise SystemExit(f"{path}: missing actions {missing}")
        by_seed[seed] = rows
    result = heldout_search(by_seed, actions)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False) + "\n")
    args.out_md.write_text(markdown(result))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
