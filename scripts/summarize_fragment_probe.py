#!/usr/bin/env python3
"""Summarize fragment utility probe JSONL logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    xd = [v - xm for v in x]
    yd = [v - ym for v in y]
    denom = math.sqrt(sum(v * v for v in xd) * sum(v * v for v in yd))
    if denom < 1e-12:
        return 0.0
    return sum(a * b for a, b in zip(xd, yd)) / denom


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def auroc(labels: list[bool], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    pos_rank_sum = sum(r for r, label in zip(ranks, labels) if label)
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def calibrated_probability(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.5] * len(values)
    return [sigmoid((v - mean) / std) for v in values]


def calibration_error(good: list[bool], prob_good: list[float], bins: int = 10) -> float:
    out = 0.0
    n = len(good)
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        idx = [
            i
            for i, p in enumerate(prob_good)
            if (lo <= p <= hi if b == bins - 1 else lo <= p < hi)
        ]
        if not idx:
            continue
        conf = sum(prob_good[i] for i in idx) / len(idx)
        acc = sum(1.0 for i in idx if good[i]) / len(idx)
        out += len(idx) / n * abs(conf - acc)
    return out


def quantile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[idx]


def coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean = sum(values) / len(values)
    if abs(mean) < 1e-12:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / abs(mean)


def token_cv_by_round(rows: list[dict]) -> list[float]:
    groups: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        key = (int(r.get("pull_step", 0)), int(r.get("fragment", -1)))
        groups.setdefault(key, []).append(float(r["c_tokens"]))
    return [
        cv
        for values in groups.values()
        if len(values) >= 2 and math.isfinite((cv := coefficient_of_variation(values)))
    ]


def summarize(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no probe records found")

    utility = [float(r["utility"]) for r in rows]
    good = [u > 0.0 for u in utility]
    bad = [not v for v in good]
    signals = {
        "token_count": [float(r["c_tokens"]) for r in rows],
        "freshness": [float(r["freshness"]) for r in rows],
        "alignment": [float(r["alignment"]) for r in rows],
        "norm_anomaly": [-float(r["norm_anomaly"]) for r in rows],
        "combined_score": [float(r["combined_score"]) for r in rows],
    }
    if all("calibrated_score" in r for r in rows):
        signals["calibrated_score"] = [float(r["calibrated_score"]) for r in rows]

    table = {}
    for name, values in signals.items():
        prob_good = (
            values
            if name in {"combined_score", "calibrated_score"}
            else calibrated_probability(values)
        )
        table[name] = {
            "pearson_utility": pearson(values, utility),
            "spearman_utility": spearman(values, utility),
            "bad_fragment_auroc": auroc(bad, [-v for v in values]),
            "calibration_error": calibration_error(good, prob_good),
        }

    sorted_utility = sorted(utility)

    def q(p: float) -> float:
        idx = min(len(sorted_utility) - 1, max(0, round(p * (len(sorted_utility) - 1))))
        return sorted_utility[idx]

    round_token_cvs = token_cv_by_round(rows)
    utility_ses = [
        float(r["utility_se"])
        for r in rows
        if r.get("utility_se") is not None and math.isfinite(float(r["utility_se"]))
    ]
    bad_strict_values = [r.get("bad_strict") for r in rows if r.get("bad_strict") is not None]
    token_auroc = table["token_count"]["bad_fragment_auroc"]
    hand_score_auroc = table["combined_score"]["bad_fragment_auroc"]
    calibrated_score_auroc = (
        table["calibrated_score"]["bad_fragment_auroc"]
        if "calibrated_score" in table
        else None
    )

    return {
        "records": len(rows),
        "negative_utility_rate": sum(bad) / len(rows),
        "bad_strict_rate": (
            sum(1 for v in bad_strict_values if v) / len(bad_strict_values)
            if bad_strict_values
            else None
        ),
        "round_token_cv_mean": (
            sum(round_token_cvs) / len(round_token_cvs) if round_token_cvs else None
        ),
        "round_token_cv_p95": quantile(round_token_cvs, 0.95) if round_token_cvs else None,
        "utility_noise_estimate": (
            sum(utility_ses) / len(utility_ses) if utility_ses else None
        ),
        "token_auroc": token_auroc,
        "freshness_auroc": table["freshness"]["bad_fragment_auroc"],
        "alignment_auroc": table["alignment"]["bad_fragment_auroc"],
        "hand_score_auroc": hand_score_auroc,
        "calibrated_score_auroc": calibrated_score_auroc,
        "score_minus_token_auroc": (
            hand_score_auroc - token_auroc
            if hand_score_auroc is not None and token_auroc is not None
            else None
        ),
        "utility_quantiles": {"p05": q(0.05), "p50": q(0.50), "p95": q(0.95)},
        "signals": table,
    }


def jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    summary = summarize(args.logs)
    text = json.dumps(jsonable(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
