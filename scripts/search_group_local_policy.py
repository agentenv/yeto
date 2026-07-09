#!/usr/bin/env python3
"""Hard-search group-local action policies from offline feature rows.

The search is intentionally offline-only. It uses train seeds to tune a
nearest-neighbor action selector, then evaluates the frozen selector on a
held-out seed. Oracle actions are reported only as upper bounds and are never
available to the deployable selector.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


DEPLOYABLE_ACTIONS = (
    "token_weighted",
    "freshness_weighted",
    "anchor_drop_bottom25",
    "anchor_positive_threshold",
    "anchor_shrink",
    "probecommit_v1",
)

_FEATURE_NAME_CACHE: dict[tuple[tuple[tuple[int, int, int], ...], int], list[str]] = {}


def row_signature(rows: list[dict]) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(row["seed"]), int(row["step"]), int(row["fragment"])) for row in rows)


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


def _num(value) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def flatten_features(row: dict) -> dict[str, float]:
    out = {
        "candidate_count": float(row["candidate_count"]),
        "step": float(row["step"]),
        f"fragment={int(row['fragment'])}": 1.0,
    }
    for field, stats in row.get("scores", {}).items():
        for key, value in stats.items():
            numeric = _num(value)
            if numeric is not None:
                out[f"score:{field}:{key}"] = numeric
    for key, value in row.get("agreement", {}).items():
        numeric = _num(value)
        if numeric is not None:
            out[f"agree:{key}"] = numeric
    return out


def action_gain(row: dict, action: str) -> float:
    return float(row["actions"][action]["utility"]) - float(row["actions"]["token_weighted"]["utility"])


def action_metrics(row: dict, action: str) -> dict:
    data = row["actions"][action]
    return {
        "utility": float(data["utility"]),
        "negative": bool(data["negative"]),
        "strict_negative": bool(data["strict_negative"]),
        "selected_mass": float(data["selected_mass"]),
        "selected_count": float(data["selected_count"]),
    }


def evaluate_choices(rows: list[dict], choices: list[str]) -> dict:
    assert len(rows) == len(choices)
    utilities = []
    gains = []
    negatives = []
    strict = []
    masses = []
    counts = []
    headroom = []
    token_neg = []
    token_strict = []
    for row, action in zip(rows, choices):
        metric = action_metrics(row, action)
        token = action_metrics(row, "token_weighted")
        oracle = action_metrics(row, "oracle_positive")
        utilities.append(metric["utility"])
        gains.append(metric["utility"] - token["utility"])
        negatives.append(1.0 if metric["negative"] else 0.0)
        strict.append(1.0 if metric["strict_negative"] else 0.0)
        masses.append(metric["selected_mass"])
        counts.append(metric["selected_count"])
        token_neg.append(1.0 if token["negative"] else 0.0)
        token_strict.append(1.0 if token["strict_negative"] else 0.0)
        denom = oracle["utility"] - token["utility"]
        if denom > 0.0:
            headroom.append((metric["utility"] - token["utility"]) / denom)
    token_neg_rate = mean(token_neg)
    token_strict_rate = mean(token_strict)
    neg_rate = mean(negatives)
    strict_rate = mean(strict)
    return {
        "groups": len(rows),
        "mean_utility": mean(utilities),
        "mean_gain_vs_token": mean(gains),
        "median_gain_vs_token": quantile(gains, 0.5),
        "gain_positive_rate": mean([1.0 if gain > 0.0 else 0.0 for gain in gains]),
        "negative_rate": neg_rate,
        "negative_rate_relative_drop": (
            0.0 if token_neg_rate <= 0.0 else (token_neg_rate - neg_rate) / token_neg_rate
        ),
        "strict_negative_rate": strict_rate,
        "strict_negative_rate_relative_drop": (
            0.0
            if token_strict_rate <= 0.0
            else (token_strict_rate - strict_rate) / token_strict_rate
        ),
        "selected_mass_mean": mean(masses),
        "selected_count_mean": mean(counts),
        "oracle_positive_headroom_captured": mean(headroom),
        "headroom_excluded_fraction": 1.0 - len(headroom) / len(rows),
        "actions": dict(sorted(Counter(choices).items())),
    }


def objective(result: dict, negative_penalty: float, strict_penalty: float, act_penalty: float) -> float:
    if "act_rate" in result:
        act_rate = float(result["act_rate"])
    else:
        act_rate = 1.0 - result["actions"].get("token_weighted", 0) / result["groups"]
    return (
        float(result["mean_utility"])
        - negative_penalty * float(result["negative_rate"])
        - strict_penalty * float(result["strict_negative_rate"])
        - act_penalty * act_rate
        + float(result.get("oracle_positive_headroom_captured", 0.0)) * float(
            result.get("_headroom_reward", 0.0)
        )
    )


def pearson_abs(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return abs(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy))


def select_feature_names(train: list[dict], max_features: int) -> list[str]:
    cache_key = (row_signature(train), int(max_features))
    if cache_key in _FEATURE_NAME_CACHE:
        return _FEATURE_NAME_CACHE[cache_key]
    flattened = [flatten_features(row) for row in train]
    names = sorted({name for item in flattened for name in item})
    targets = {
        action: [action_gain(row, action) for row in train]
        for action in DEPLOYABLE_ACTIONS
        if action != "token_weighted"
    }
    scored = []
    for name in names:
        xs = [item.get(name, 0.0) for item in flattened]
        score = max(pearson_abs(xs, ys) for ys in targets.values())
        if score > 0.0:
            scored.append((score, name))
    scored.sort(reverse=True)
    selected = [name for _, name in scored[:max_features]]
    _FEATURE_NAME_CACHE[cache_key] = selected
    return selected


def standardize(train: list[dict], test: list[dict], names: list[str]) -> tuple[list[list[float]], list[list[float]]]:
    train_flat = [flatten_features(row) for row in train]
    test_flat = [flatten_features(row) for row in test]
    means = []
    scales = []
    keep = []
    for name in names:
        vals = [item.get(name, 0.0) for item in train_flat]
        m = mean(vals)
        scale = math.sqrt(mean([(v - m) ** 2 for v in vals]))
        if scale > 1e-9:
            keep.append(name)
            means.append(m)
            scales.append(scale)

    def project(items: list[dict]) -> list[list[float]]:
        out = []
        for item in items:
            out.append([(item.get(name, 0.0) - means[i]) / scales[i] for i, name in enumerate(keep)])
        return out

    return project(train_flat), project(test_flat)


def squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def knn_choices(
    train: list[dict],
    test: list[dict],
    *,
    feature_count: int,
    k: int,
    threshold: float,
    weighted: bool,
) -> list[str]:
    names = select_feature_names(train, feature_count)
    x_train, x_test = standardize(train, test, names)
    choices = []
    for x in x_test:
        ordered = sorted(
            ((squared_distance(x, x_train[i]), i) for i in range(len(train))),
            key=lambda item: item[0],
        )[: min(k, len(train))]
        action_scores = {}
        for action in DEPLOYABLE_ACTIONS:
            if action == "token_weighted":
                action_scores[action] = 0.0
                continue
            if weighted:
                denom = 0.0
                numer = 0.0
                for dist, idx in ordered:
                    weight = 1.0 / (dist + 1e-6)
                    numer += weight * action_gain(train[idx], action)
                    denom += weight
                action_scores[action] = numer / denom if denom > 0.0 else 0.0
            else:
                action_scores[action] = mean([action_gain(train[idx], action) for _, idx in ordered])
        best = max(DEPLOYABLE_ACTIONS, key=lambda action: action_scores[action])
        choices.append(best if action_scores[best] > threshold else "token_weighted")
    return choices


def deployable_oracle_choices(rows: list[dict]) -> list[str]:
    return [
        max(DEPLOYABLE_ACTIONS, key=lambda action: float(row["actions"][action]["utility"]))
        for row in rows
    ]


def fixed_action_result(rows: list[dict], action: str) -> dict:
    result = evaluate_choices(rows, [action] * len(rows))
    result["policy"] = f"fixed:{action}"
    return result


def tune_knn(train: list[dict], args) -> tuple[dict, dict]:
    seeds = sorted({int(row["seed"]) for row in train})
    configs = []
    for feature_count in args.feature_counts:
        for k in args.k_values:
            for threshold in args.thresholds:
                for weighted in [False, True]:
                    configs.append(
                        {
                            "feature_count": int(feature_count),
                            "k": int(k),
                            "threshold": float(threshold),
                            "weighted": bool(weighted),
                        }
                    )
    best = None
    best_score = -float("inf")
    best_summary = None
    for config in configs:
        fold_results = []
        for val_seed in seeds:
            inner_train = [row for row in train if int(row["seed"]) != val_seed]
            inner_val = [row for row in train if int(row["seed"]) == val_seed]
            choices = knn_choices(inner_train, inner_val, **config)
            fold_results.append(evaluate_choices(inner_val, choices))
        aggregate = {
            "mean_gain_vs_token": mean([r["mean_gain_vs_token"] for r in fold_results]),
            "negative_rate_relative_drop": mean(
                [r["negative_rate_relative_drop"] for r in fold_results]
            ),
            "strict_negative_rate_relative_drop": mean(
                [r["strict_negative_rate_relative_drop"] for r in fold_results]
            ),
            "oracle_positive_headroom_captured": mean(
                [r["oracle_positive_headroom_captured"] for r in fold_results]
            ),
            "selected_mass_mean": mean([r["selected_mass_mean"] for r in fold_results]),
            "mean_utility": mean([r["mean_utility"] for r in fold_results]),
            "negative_rate": mean([r["negative_rate"] for r in fold_results]),
            "strict_negative_rate": mean([r["strict_negative_rate"] for r in fold_results]),
            "act_rate": mean(
                [
                    1.0 - r["actions"].get("token_weighted", 0) / r["groups"]
                    for r in fold_results
                ]
            ),
            "_headroom_reward": args.headroom_reward,
        }
        score = objective(
            aggregate,
            args.negative_penalty,
            args.strict_penalty,
            args.act_rate_penalty,
        )
        if score > best_score:
            best_score = score
            best = config
            best_summary = {"folds": fold_results, "aggregate": aggregate, "objective": score}
    assert best is not None and best_summary is not None
    return best, best_summary


def heldout_seed_search(rows: list[dict], args) -> dict:
    seeds = sorted({int(row["seed"]) for row in rows})
    splits = []
    for test_seed in seeds:
        train = [row for row in rows if int(row["seed"]) != test_seed]
        test = [row for row in rows if int(row["seed"]) == test_seed]
        config, train_cv = tune_knn(train, args)
        choices = knn_choices(train, test, **config)
        splits.append(
            {
                "test_seed": test_seed,
                "train_seeds": [seed for seed in seeds if seed != test_seed],
                "config": config,
                "train_cv": train_cv,
                "test_result": evaluate_choices(test, choices),
                "test_deployable_oracle": evaluate_choices(test, deployable_oracle_choices(test)),
            }
        )
    aggregate = {
        "splits": len(splits),
        "mean_gain_vs_token": mean([split["test_result"]["mean_gain_vs_token"] for split in splits]),
        "all_test_gains_positive": all(
            split["test_result"]["mean_gain_vs_token"] > 0.0 for split in splits
        ),
        "negative_rate_relative_drop": mean(
            [split["test_result"]["negative_rate_relative_drop"] for split in splits]
        ),
        "strict_negative_rate_relative_drop": mean(
            [split["test_result"]["strict_negative_rate_relative_drop"] for split in splits]
        ),
        "oracle_positive_headroom_captured": mean(
            [split["test_result"]["oracle_positive_headroom_captured"] for split in splits]
        ),
        "selected_mass_mean": mean([split["test_result"]["selected_mass_mean"] for split in splits]),
        "deployable_oracle_gain": mean(
            [split["test_deployable_oracle"]["mean_gain_vs_token"] for split in splits]
        ),
        "deployable_oracle_headroom_captured": mean(
            [split["test_deployable_oracle"]["oracle_positive_headroom_captured"] for split in splits]
        ),
    }
    aggregate["gate_pass"] = bool(
        aggregate["all_test_gains_positive"]
        and aggregate["mean_gain_vs_token"] > 0.0
        and aggregate["oracle_positive_headroom_captured"] >= args.min_headroom_captured
        and aggregate["negative_rate_relative_drop"] >= args.min_negative_drop
        and aggregate["selected_mass_mean"] >= args.min_selected_mass
    )
    return {"splits": splits, "aggregate": aggregate}


def baselines(rows: list[dict]) -> dict:
    out = {action: fixed_action_result(rows, action) for action in DEPLOYABLE_ACTIONS}
    out["deployable_oracle"] = evaluate_choices(rows, deployable_oracle_choices(rows))
    out["deployable_oracle"]["policy"] = "deployable_oracle"
    out["oracle_positive"] = evaluate_choices(rows, ["oracle_positive"] * len(rows))
    out["oracle_positive"]["policy"] = "oracle_positive"
    out["oracle_topk"] = evaluate_choices(rows, ["oracle_topk"] * len(rows))
    out["oracle_topk"]["policy"] = "oracle_topk"
    return out


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
    lines = ["# Group-Local Hard Policy Search", ""]
    agg = result["heldout_seed"]["aggregate"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Groups: `{result['records']}`")
    lines.append(f"- Mean held-out gain: `{fmt(agg['mean_gain_vs_token'], 6)}`")
    lines.append(f"- Headroom captured: `{fmt(agg['oracle_positive_headroom_captured'])}`")
    lines.append(f"- Negative-rate drop: `{fmt(agg['negative_rate_relative_drop'])}`")
    lines.append(f"- Deployable-action oracle gain: `{fmt(agg['deployable_oracle_gain'], 6)}`")
    lines.append(f"- Gate pass: `{agg['gate_pass']}`")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Policy | Gain | Negative drop | Strict drop | Headroom captured | Selected mass |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, row in result["baselines"].items():
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                name,
                fmt(row["mean_gain_vs_token"], 6),
                fmt(row["negative_rate_relative_drop"]),
                fmt(row["strict_negative_rate_relative_drop"]),
                fmt(row["oracle_positive_headroom_captured"]),
                fmt(row["selected_mass_mean"]),
            )
        )
    lines.append("")
    lines.append("## Held-Out Seed KNN Selector")
    lines.append("")
    lines.append("| Test seed | Config | Gain | Negative drop | Strict drop | Headroom captured | Selected mass | Actions |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    for split in result["heldout_seed"]["splits"]:
        row = split["test_result"]
        lines.append(
            "| {} | `{}` | {} | {} | {} | {} | {} | `{}` |".format(
                split["test_seed"],
                split["config"],
                fmt(row["mean_gain_vs_token"], 6),
                fmt(row["negative_rate_relative_drop"]),
                fmt(row["strict_negative_rate_relative_drop"]),
                fmt(row["oracle_positive_headroom_captured"]),
                fmt(row["selected_mass_mean"]),
                row["actions"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--feature-counts", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 9, 15, 25, 50])
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[-0.0001, 0.0, 0.00005, 0.0001, 0.0002],
    )
    p.add_argument("--negative-penalty", type=float, default=0.0005)
    p.add_argument("--strict-penalty", type=float, default=0.0005)
    p.add_argument("--act-rate-penalty", type=float, default=0.0)
    p.add_argument(
        "--headroom-reward",
        type=float,
        default=0.0,
        help="train-seed objective reward for oracle-positive headroom capture",
    )
    p.add_argument("--min-headroom-captured", type=float, default=0.40)
    p.add_argument("--min-negative-drop", type=float, default=0.20)
    p.add_argument("--min-selected-mass", type=float, default=0.40)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(args.features)
    result = {
        "schema": "group_local_hard_policy_search_v1",
        "records": len(rows),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "baselines": baselines(rows),
        "heldout_seed": heldout_seed_search(rows, args),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    args.out_md.write_text(to_markdown(result))
    print(json.dumps(jsonable(result["heldout_seed"]["aggregate"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
