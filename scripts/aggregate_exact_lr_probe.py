#!/usr/bin/env python3
"""Aggregate sharded exact scalar action-probe replays.

The input JSONL files already contain anchor-only decisions and disjoint-oracle
losses for every scalar action. This script validates complete seed coverage,
reports the recorded selector, and replays two explicitly labelled diagnostic
rules from the stored anchor panels without re-evaluating the model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "replay_exact_lr_probe", ROOT / "scripts" / "replay_exact_lr_probe.py"
)
LR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LR)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if record.get("schema") != LR.REPLAY_SCHEMA:
                raise ValueError(f"{path}:{line_number}: unexpected replay schema")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no replay records")
    return records


def _record_key(record: dict) -> tuple[int, int, int]:
    return int(record["seed"]), int(record["step"]), int(record["fragment"])


def load_and_validate(
    paths: list[Path], expected_seeds: list[int], expected_groups_per_seed: int
) -> list[dict]:
    records = []
    seen = set()
    panel_contract = None
    action_multipliers = None
    for path in paths:
        for record in _read_jsonl(path):
            key = _record_key(record)
            if key in seen:
                raise ValueError(f"duplicate replay group {key}")
            seen.add(key)
            metadata = record.get("panel_metadata", {})
            if not metadata.get("anchor_oracle_disjoint"):
                raise ValueError(
                    f"group {key} does not prove anchor/oracle disjointness"
                )
            contract = (
                metadata.get("anchor_panels_sha256"),
                metadata.get("oracle_panels_sha256"),
                metadata.get("anchor_rows_sha256"),
                metadata.get("oracle_rows_sha256"),
            )
            if panel_contract is None:
                panel_contract = contract
            elif contract != panel_contract:
                raise ValueError("replay shards do not share one panel contract")
            multipliers = tuple(float(value) for value in record["action_multipliers"])
            if action_multipliers is None:
                action_multipliers = multipliers
            elif multipliers != action_multipliers:
                raise ValueError("replay shards do not share one action grid")
            records.append(record)

    actual_seeds = sorted({int(record["seed"]) for record in records})
    if actual_seeds != sorted(expected_seeds):
        raise ValueError(
            f"expected seeds {sorted(expected_seeds)}, found {actual_seeds}"
        )
    by_seed = Counter(int(record["seed"]) for record in records)
    bad_counts = {
        seed: by_seed.get(seed, 0)
        for seed in expected_seeds
        if by_seed.get(seed, 0) != expected_groups_per_seed
    }
    if bad_counts:
        raise ValueError(
            f"incomplete replay coverage; expected {expected_groups_per_seed} groups per seed, "
            f"found {bad_counts}"
        )
    return sorted(records, key=_record_key)


def _paired_stats(base: list[float], trial: list[float], z: float) -> dict:
    decision = LR.paired_decision(
        base,
        trial,
        min_gain=0.0,
        lcb_z=z,
        min_win_rate=0.0,
    )
    return {
        "mean": float(decision["gain"]),
        "lcb": float(decision["lcb"]),
        "win_rate": float(decision["win_rate"]),
    }


def _action_map(record: dict) -> dict[float, dict]:
    return {float(action["multiplier"]): action for action in record["actions"]}


def replay_rule(
    records: list[dict],
    *,
    family: str,
    fallback: float,
    min_gain: float,
    z: float,
    min_win_rate: float,
) -> dict:
    chosen = []
    for record in records:
        actions = _action_map(record)
        if fallback not in actions:
            raise ValueError(f"fallback multiplier {fallback} is absent")
        if family == "positive_utility_largest":
            reference = list(record["anchor_current_panel_losses"])
            candidates = list(actions.values())
        elif family == "shrink_from_fallback":
            reference = list(actions[fallback]["anchor_panel_losses"])
            candidates = [
                action
                for multiplier, action in actions.items()
                if multiplier < fallback
            ]
        else:
            raise ValueError(f"unknown replay family {family!r}")

        eligible = []
        for action in candidates:
            stats = _paired_stats(reference, list(action["anchor_panel_losses"]), z)
            if (
                stats["mean"] >= min_gain
                and stats["lcb"] > 0.0
                and stats["win_rate"] >= min_win_rate
            ):
                eligible.append((float(action["multiplier"]), stats, action))
        if family == "positive_utility_largest":
            selected = (
                max(eligible, key=lambda item: item[0])[2]
                if eligible
                else actions[fallback]
            )
        else:
            selected = (
                max(
                    eligible,
                    key=lambda item: (item[1]["lcb"], item[1]["mean"], item[0]),
                )[2]
                if eligible
                else actions[fallback]
            )
        chosen.append((record, actions[fallback], selected))

    gains_vs_one = [float(action["oracle_gain_vs_baseline"]) for _, _, action in chosen]
    gains_vs_fallback = [
        float(base["oracle_loss"]) - float(action["oracle_loss"])
        for _, base, action in chosen
    ]
    baseline_negative = [
        float(record["baseline_oracle_utility"]) < 0.0 for record, _, _ in chosen
    ]
    chosen_negative = [float(action["oracle_utility"]) < 0.0 for _, _, action in chosen]
    fallback_negative = [float(base["oracle_utility"]) < 0.0 for _, base, _ in chosen]
    chosen_strict = [
        float(action["oracle_utility"]) + float(action["oracle_utility_se"]) < 0.0
        for _, _, action in chosen
    ]
    fallback_strict = [
        float(base["oracle_utility"]) + float(base["oracle_utility_se"]) < 0.0
        for _, base, _ in chosen
    ]

    def rate(values):
        return sum(bool(value) for value in values) / len(values)

    def relative_drop(before, after):
        left = rate(before)
        right = rate(after)
        return None if left == 0.0 else (left - right) / left

    per_seed = {}
    for seed in sorted({int(record["seed"]) for record, _, _ in chosen}):
        rows = [row for row in chosen if int(row[0]["seed"]) == seed]
        per_seed[str(seed)] = {
            "records": len(rows),
            "mean_gain_vs_multiplier_1": LR.mean(
                [float(action["oracle_gain_vs_baseline"]) for _, _, action in rows]
            ),
            "mean_gain_vs_fallback": LR.mean(
                [
                    float(base["oracle_loss"]) - float(action["oracle_loss"])
                    for _, base, action in rows
                ]
            ),
            "selection_rate": LR.mean(
                [float(action["multiplier"]) != fallback for _, _, action in rows]
            ),
        }

    return {
        "family": family,
        "fallback_multiplier": fallback,
        "min_gain": min_gain,
        "lcb_z": z,
        "min_win_rate": min_win_rate,
        "records": len(chosen),
        "mean_gain_vs_multiplier_1": LR.mean(gains_vs_one),
        "mean_gain_vs_fallback": LR.mean(gains_vs_fallback),
        "selection_rate": LR.mean(
            [float(action["multiplier"]) != fallback for _, _, action in chosen]
        ),
        "chosen_multiplier_distribution": dict(
            sorted(
                Counter(
                    LR.multiplier_key(float(action["multiplier"]))
                    for _, _, action in chosen
                ).items()
            )
        ),
        "baseline_negative_rate": rate(baseline_negative),
        "fallback_negative_rate": rate(fallback_negative),
        "chosen_negative_rate": rate(chosen_negative),
        "negative_drop_vs_fallback": relative_drop(fallback_negative, chosen_negative),
        "fallback_strict_negative_rate": rate(fallback_strict),
        "chosen_strict_negative_rate": rate(chosen_strict),
        "strict_negative_drop_vs_fallback": relative_drop(
            fallback_strict, chosen_strict
        ),
        "per_seed": per_seed,
    }


def render_markdown(result: dict) -> str:
    summary = result["recorded_selector"]
    lines = [
        "# Exact Scalar Action-Probe Aggregate",
        "",
        f"Processed **{summary['records']}** complete groups across seeds "
        f"{', '.join(str(seed) for seed in summary['seeds'])}.",
        "",
        "| Policy | Gain vs x1 | Gain vs fallback | Selection rate | Negative drop vs fallback |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = [
        (
            "recorded x1-fallback selector",
            summary["mean_chosen_oracle_gain_vs_baseline"],
            0.0,
            summary["selection_rate"],
            summary.get("negative_rate_relative_drop"),
        ),
    ]
    for name in ("risk_aware_largest", "shrink_from_max"):
        policy = result[name]
        rows.append(
            (
                name,
                policy["mean_gain_vs_multiplier_1"],
                policy["mean_gain_vs_fallback"],
                policy["selection_rate"],
                policy["negative_drop_vs_fallback"],
            )
        )
    for name, gain, fallback_gain, rate, drop in rows:
        lines.append(
            f"| `{name}` | {gain:.7f} | {fallback_gain:.7f} | {rate:.3f} | "
            f"{'n/a' if drop is None else f'{drop:.3f}'} |"
        )
    lines.extend(
        [
            "",
            f"Best fixed multiplier: **{summary['best_fixed_multiplier']}** "
            f"(mean gain {summary['best_fixed_mean_oracle_gain_vs_baseline']:.7f} vs x1).",
            "",
            "The two replayed policies are diagnostics computed from stored anchor panels; they are not untouched confirmations.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", nargs="+", required=True, type=Path)
    parser.add_argument("--expected-seeds", nargs="+", required=True, type=int)
    parser.add_argument("--expected-groups-per-seed", type=int, default=80)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = load_and_validate(
        args.replays, args.expected_seeds, args.expected_groups_per_seed
    )
    recorded = LR.summarize(records)
    maximum = max(float(value) for value in recorded["action_multipliers"])
    result = {
        "schema": "exact_lr_probe_aggregate_v1",
        "source_files": [str(path) for path in args.replays],
        "recorded_selector": recorded,
        "risk_aware_largest": replay_rule(
            records,
            family="positive_utility_largest",
            fallback=1.0,
            min_gain=0.005,
            z=1.0,
            min_win_rate=0.75,
        ),
        "shrink_from_max": replay_rule(
            records,
            family="shrink_from_fallback",
            fallback=maximum,
            min_gain=0.0001,
            z=0.0,
            min_win_rate=0.625,
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
