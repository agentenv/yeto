#!/usr/bin/env python3
"""Analyze action-probe rank stability from per-anchor-batch replay records."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
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
    ranks = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


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


def choose_subsets(total: int, size: int, count: int, rng: random.Random) -> list[tuple[int, ...]]:
    if size > total:
        return []
    if size == total:
        return [tuple(range(total))]
    all_subsets = list(combinations(range(total), size))
    if len(all_subsets) <= count:
        return all_subsets
    return sorted(rng.sample(all_subsets, count))


def action_values(record: dict, actions: tuple[str, ...], scope: str) -> dict[str, float]:
    return {action: float(record[f"{action}_{scope}_utility"]) for action in actions}


def anchor_subset_values(record: dict, actions: tuple[str, ...], subset: tuple[int, ...]) -> dict[str, float]:
    out = {}
    for action in actions:
        values = record.get(f"{action}_anchor_batch_utilities")
        if values is None:
            raise SystemExit(
                f"missing {action}_anchor_batch_utilities; rerun action probe with "
                "--include-anchor-batch-utilities"
            )
        out[action] = mean([float(values[i]) for i in subset])
    return out


def strict_bool(value) -> bool | None:
    return None if value is None else bool(value)


def summarize_choice(records: list[dict], choices: list[str], actions: tuple[str, ...]) -> dict:
    token_neg = mean([1.0 if r["token_weighted_oracle_negative"] else 0.0 for r in records])
    token_strict = mean(
        [
            1.0 if r["token_weighted_oracle_strict_negative"] else 0.0
            for r in records
            if r["token_weighted_oracle_strict_negative"] is not None
        ]
    )
    gains = []
    negatives = []
    strict = []
    captured = []
    masses = []
    oracle_matches = []
    for record, action in zip(records, choices):
        utility = float(record[f"{action}_oracle_utility"])
        token = float(record["token_weighted_oracle_utility"])
        oracle = float(record["oracle_positive_oracle_utility"])
        gains.append(utility - token)
        negatives.append(1.0 if bool(record[f"{action}_oracle_negative"]) else 0.0)
        strict_value = strict_bool(record.get(f"{action}_oracle_strict_negative"))
        if strict_value is not None:
            strict.append(1.0 if strict_value else 0.0)
        masses.append(float(record.get(f"{action}_selected_mass", 1.0)))
        denom = oracle - token
        if denom > 0.0:
            captured.append((utility - token) / denom)
        oracle_best = max(actions, key=lambda a: float(record[f"{a}_oracle_utility"]))
        oracle_matches.append(1.0 if action == oracle_best else 0.0)
    neg_rate = mean(negatives)
    strict_rate = mean(strict)
    return {
        "mean_gain_vs_token": mean(gains),
        "negative_rate_relative_drop": None
        if token_neg <= 0.0
        else (token_neg - neg_rate) / token_neg,
        "strict_negative_rate_relative_drop": None
        if token_strict <= 0.0
        else (token_strict - strict_rate) / token_strict,
        "oracle_positive_headroom_captured": mean(captured),
        "selected_mass_mean": mean(masses),
        "oracle_top1_action_match": mean(oracle_matches),
        "chosen_action_distribution": dict(sorted(Counter(choices).items())),
    }


def evaluate_subset(
    records: list[dict],
    actions: tuple[str, ...],
    subset: tuple[int, ...],
    margin: float,
) -> dict:
    top1_choices = []
    margin_choices = []
    top1_matches = []
    spearmans = []
    concordances = []
    margins = []
    for record in records:
        anchor = anchor_subset_values(record, actions, subset)
        oracle = action_values(record, actions, "oracle")
        anchor_ranked = sorted(anchor.items(), key=lambda item: item[1], reverse=True)
        oracle_top = max(actions, key=lambda action: oracle[action])
        top_action = anchor_ranked[0][0]
        second = anchor_ranked[1][1] if len(anchor_ranked) > 1 else -float("inf")
        gap = anchor_ranked[0][1] - second
        top1_choices.append(top_action)
        margin_choices.append(top_action if gap >= margin else "token_weighted")
        top1_matches.append(1.0 if top_action == oracle_top else 0.0)
        if (rho := spearman(anchor, oracle, actions)) is not None:
            spearmans.append(rho)
        if (pc := pairwise_concordance(anchor, oracle, actions)) is not None:
            concordances.append(pc)
        margins.append(gap)
    top1 = summarize_choice(records, top1_choices, actions)
    gated = summarize_choice(records, margin_choices, actions)
    return {
        "subset": list(subset),
        "anchor_batches": len(subset),
        "top1_action_agreement": mean(top1_matches),
        "spearman": mean(spearmans),
        "pairwise_concordance": mean(concordances),
        "anchor_margin_mean": mean(margins),
        "top1_policy": top1,
        "margin_gated_policy": gated,
    }


def summarize_stability(
    records: list[dict],
    actions: tuple[str, ...],
    batch_sizes: list[int],
    subsets_per_size: int,
    margin: float,
    seed: int,
) -> dict:
    max_batches = min(len(records[0][f"{actions[0]}_anchor_batch_utilities"]), 10**9)
    rng = random.Random(seed)
    by_size = {}
    for size in batch_sizes:
        subset_results = [
            evaluate_subset(records, actions, subset, margin)
            for subset in choose_subsets(max_batches, size, subsets_per_size, rng)
        ]
        by_size[str(size)] = {
            "subsets": len(subset_results),
            "top1_action_agreement": mean([r["top1_action_agreement"] for r in subset_results]),
            "spearman": mean([r["spearman"] for r in subset_results]),
            "pairwise_concordance": mean([r["pairwise_concordance"] for r in subset_results]),
            "anchor_margin_mean": mean([r["anchor_margin_mean"] for r in subset_results]),
            "top1_gain_vs_token": mean(
                [r["top1_policy"]["mean_gain_vs_token"] for r in subset_results]
            ),
            "top1_negative_drop": mean(
                [r["top1_policy"]["negative_rate_relative_drop"] for r in subset_results]
            ),
            "top1_headroom_captured": mean(
                [r["top1_policy"]["oracle_positive_headroom_captured"] for r in subset_results]
            ),
            "margin_gated_gain_vs_token": mean(
                [r["margin_gated_policy"]["mean_gain_vs_token"] for r in subset_results]
            ),
            "margin_gated_negative_drop": mean(
                [r["margin_gated_policy"]["negative_rate_relative_drop"] for r in subset_results]
            ),
            "margin_gated_headroom_captured": mean(
                [r["margin_gated_policy"]["oracle_positive_headroom_captured"] for r in subset_results]
            ),
            "subset_results": subset_results,
        }
    return by_size


def aggregate_seed_summaries(seed_summaries: list[dict]) -> dict:
    sizes = sorted({size for item in seed_summaries for size in item["by_anchor_batches"]}, key=int)
    out = {}
    for size in sizes:
        rows = [item["by_anchor_batches"][size] for item in seed_summaries if size in item["by_anchor_batches"]]
        out[size] = {
            "seeds": len(rows),
            "top1_action_agreement": mean([r["top1_action_agreement"] for r in rows]),
            "spearman": mean([r["spearman"] for r in rows]),
            "pairwise_concordance": mean([r["pairwise_concordance"] for r in rows]),
            "top1_gain_vs_token": mean([r["top1_gain_vs_token"] for r in rows]),
            "top1_negative_drop": mean([r["top1_negative_drop"] for r in rows]),
            "top1_headroom_captured": mean([r["top1_headroom_captured"] for r in rows]),
            "margin_gated_gain_vs_token": mean([r["margin_gated_gain_vs_token"] for r in rows]),
            "margin_gated_negative_drop": mean([r["margin_gated_negative_drop"] for r in rows]),
            "margin_gated_headroom_captured": mean([r["margin_gated_headroom_captured"] for r in rows]),
        }
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
    lines = ["# Action-Probe Stability", ""]
    lines.append(f"- Records: `{result['records']}`")
    lines.append(f"- Seeds: `{result['seeds']}`")
    lines.append(f"- Actions: `{result['actions']}`")
    lines.append("")
    lines.append("| Anchor batches | Top1 agreement | Spearman | Pairwise concordance | Top1 gain | Top1 neg. drop | Top1 headroom | Gated gain | Gated neg. drop | Gated headroom |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for size, row in result["aggregate_by_anchor_batches"].items():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                size,
                fmt(row["top1_action_agreement"]),
                fmt(row["spearman"]),
                fmt(row["pairwise_concordance"]),
                fmt(row["top1_gain_vs_token"], 6),
                fmt(row["top1_negative_drop"]),
                fmt(row["top1_headroom_captured"]),
                fmt(row["margin_gated_gain_vs_token"], 6),
                fmt(row["margin_gated_negative_drop"]),
                fmt(row["margin_gated_headroom_captured"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replays", nargs="+", required=True, type=Path)
    p.add_argument("--actions", nargs="+", default=list(DEFAULT_ACTIONS))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--subsets-per-size", type=int, default=12)
    p.add_argument("--margin", type=float, default=0.0005)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    actions = tuple(args.actions)
    seed_summaries = []
    for replay in args.replays:
        records = read_jsonl(replay)
        seeds = sorted({int(r["seed"]) for r in records if r.get("seed") is not None})
        seed_summaries.append(
            {
                "replay": str(replay),
                "records": len(records),
                "seeds": seeds,
                "by_anchor_batches": summarize_stability(
                    records,
                    actions,
                    args.batch_sizes,
                    args.subsets_per_size,
                    args.margin,
                    args.sample_seed,
                ),
            }
        )
    result = {
        "schema": "action_probe_stability_v1",
        "records": sum(item["records"] for item in seed_summaries),
        "seeds": sorted({seed for item in seed_summaries for seed in item["seeds"]}),
        "actions": list(actions),
        "seed_summaries": seed_summaries,
        "aggregate_by_anchor_batches": aggregate_seed_summaries(seed_summaries),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(to_markdown(result))
    print(json.dumps(jsonable(result["aggregate_by_anchor_batches"]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
