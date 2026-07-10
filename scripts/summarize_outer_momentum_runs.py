#!/usr/bin/env python3
"""Summarize paired outer-momentum training runs with strict tape validation."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Sequence


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
M4_CONFIG_RE = re.compile(r"(?m)^\s*m4\s+M=(\d+)\b")
M4_RESULT_RE = re.compile(
    rf"(?m)^\s*(?:\+\s*)?\[compare\]\s+m4\s+eval loss/token:\s*"
    rf"({NUMBER})\s+\(({NUMBER})s\)\s*$"
)
EVAL_LOSS_RE = re.compile(rf"(?m)^\s*EVAL_LOSS\s+({NUMBER})\s*$")

LOSS_METRICS = (
    "internal_loss",
    "holdout_loss",
    "holdout_indices7000_loss",
    "holdout_mean_loss",
)


class SummaryError(RuntimeError):
    """Raised when an input run is incomplete, malformed, or uncontrolled."""


@dataclass(frozen=True)
class RunResult:
    directory: str
    internal_loss: float
    holdout_loss: float
    holdout_indices7000_loss: float
    holdout_mean_loss: float
    wall_time_s: float
    outer_steps: int
    quorum: int
    responder_count_min: int
    responder_count_max: int


def _read_required(path: Path) -> str:
    if not path.is_file():
        raise SummaryError(f"missing required file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc


def _finite_float(value: str, *, field: str, path: Path) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise SummaryError(f"{path}: invalid {field}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise SummaryError(f"{path}: non-finite {field}: {value!r}")
    return parsed


def _single_match(pattern: re.Pattern[str], text: str, *, field: str, path: Path) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise SummaryError(
            f"{path}: expected exactly one {field}, found {len(matches)}"
        )
    match = matches[0]
    if isinstance(match, tuple):
        raise AssertionError("_single_match only supports one capture group")
    return match


def parse_run_log(path: Path) -> tuple[float, float, int]:
    text = _read_required(path)

    quorum_values = {int(value) for value in M4_CONFIG_RE.findall(text)}
    if len(quorum_values) != 1:
        raise SummaryError(
            f"{path}: expected one unambiguous m4 M=<quorum> configuration, "
            f"found {sorted(quorum_values)}"
        )
    quorum = quorum_values.pop()
    if quorum <= 0:
        raise SummaryError(f"{path}: invalid m4 quorum {quorum}")

    results = M4_RESULT_RE.findall(text)
    if len(results) != 1:
        raise SummaryError(
            f"{path}: expected exactly one m4 eval-loss/wall-time result, "
            f"found {len(results)}"
        )
    internal_text, wall_text = results[0]
    internal_loss = _finite_float(internal_text, field="m4 internal loss", path=path)
    wall_time_s = _finite_float(wall_text, field="m4 wall time", path=path)
    if internal_loss <= 0.0:
        raise SummaryError(f"{path}: m4 internal loss must be positive")
    if wall_time_s <= 0.0:
        raise SummaryError(f"{path}: m4 wall time must be positive")
    return internal_loss, wall_time_s, quorum


def parse_eval_loss(path: Path) -> float:
    text = _read_required(path)
    value = _single_match(EVAL_LOSS_RE, text, field="EVAL_LOSS", path=path)
    loss = _finite_float(value, field="EVAL_LOSS", path=path)
    if loss <= 0.0:
        raise SummaryError(f"{path}: EVAL_LOSS must be positive")
    return loss


def parse_tape(path: Path, *, expected_quorum: int) -> tuple[int, int, int]:
    text = _read_required(path)
    lines = [(line_number, line) for line_number, line in enumerate(text.splitlines(), 1) if line.strip()]
    if not lines:
        raise SummaryError(f"{path}: event tape has no records")

    steps: list[int] = []
    responder_counts: list[int] = []
    expected_ids = set(range(expected_quorum))
    for line_number, line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SummaryError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SummaryError(f"{path}:{line_number}: tape record must be an object")

        step = row.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise SummaryError(f"{path}:{line_number}: invalid step {step!r}")
        responders = row.get("responders")
        if not isinstance(responders, list):
            raise SummaryError(f"{path}:{line_number}: responders must be a list")

        ids = []
        for responder in responders:
            if not isinstance(responder, dict):
                raise SummaryError(
                    f"{path}:{line_number}: every responder must be an object"
                )
            responder_id = responder.get("id")
            if not isinstance(responder_id, int) or isinstance(responder_id, bool):
                raise SummaryError(
                    f"{path}:{line_number}: invalid responder id {responder_id!r}"
                )
            ids.append(responder_id)

        responder_count = len(ids)
        if responder_count != expected_quorum or set(ids) != expected_ids:
            raise SummaryError(
                f"{path}:{line_number}: non-full-quorum step {step}: "
                f"expected responder ids {sorted(expected_ids)}, got {ids}"
            )
        if len(ids) != len(set(ids)):
            raise SummaryError(
                f"{path}:{line_number}: duplicate responder id at step {step}: {ids}"
            )

        steps.append(step)
        responder_counts.append(responder_count)

    expected_steps = list(range(1, len(steps) + 1))
    if steps != expected_steps:
        raise SummaryError(
            f"{path}: outer steps must be contiguous and ordered from 1; "
            f"expected {expected_steps[:5]}...{expected_steps[-1]}, "
            f"got {steps[:5]}...{steps[-1]}"
        )
    return len(steps), min(responder_counts), max(responder_counts)


def parse_run(directory: Path) -> RunResult:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise SummaryError(f"run directory does not exist: {directory}")

    internal_loss, wall_time_s, quorum = parse_run_log(directory / "run.log")
    holdout_loss = parse_eval_loss(directory / "holdout_eval.log")
    holdout_indices7000_loss = parse_eval_loss(
        directory / "holdout_indices7000_eval.log"
    )
    outer_steps, responder_min, responder_max = parse_tape(
        directory / "work" / "m4" / "tape.jsonl",
        expected_quorum=quorum,
    )
    return RunResult(
        directory=str(directory),
        internal_loss=internal_loss,
        holdout_loss=holdout_loss,
        holdout_indices7000_loss=holdout_indices7000_loss,
        holdout_mean_loss=fmean((holdout_loss, holdout_indices7000_loss)),
        wall_time_s=wall_time_s,
        outer_steps=outer_steps,
        quorum=quorum,
        responder_count_min=responder_min,
        responder_count_max=responder_max,
    )


def summarize_pair(seed: int, baseline_dir: Path, treatment_dir: Path) -> dict:
    if seed < 0:
        raise SummaryError(f"seed must be non-negative, got {seed}")
    if baseline_dir.expanduser().resolve() == treatment_dir.expanduser().resolve():
        raise SummaryError(f"seed {seed}: baseline and treatment directories are identical")

    baseline = parse_run(baseline_dir)
    treatment = parse_run(treatment_dir)
    if baseline.outer_steps != treatment.outer_steps:
        raise SummaryError(
            f"seed {seed}: nonmatching outer-step counts: "
            f"baseline={baseline.outer_steps}, treatment={treatment.outer_steps}"
        )
    if baseline.quorum != treatment.quorum:
        raise SummaryError(
            f"seed {seed}: nonmatching configured quorums: "
            f"baseline={baseline.quorum}, treatment={treatment.quorum}"
        )

    baseline_dict = asdict(baseline)
    treatment_dict = asdict(treatment)
    treatment_minus_baseline = {
        metric: treatment_dict[metric] - baseline_dict[metric]
        for metric in (*LOSS_METRICS, "wall_time_s")
    }
    improvement = {
        metric: baseline_dict[metric] - treatment_dict[metric]
        for metric in LOSS_METRICS
    }
    wall_relative_change = (
        treatment.wall_time_s - baseline.wall_time_s
    ) / baseline.wall_time_s
    return {
        "seed": seed,
        "outer_steps": baseline.outer_steps,
        "quorum": baseline.quorum,
        "baseline": baseline_dict,
        "treatment": treatment_dict,
        "treatment_minus_baseline": treatment_minus_baseline,
        "loss_improvement": improvement,
        "wall_time_relative_change": wall_relative_change,
    }


def aggregate_pairs(pairs: Sequence[dict]) -> dict:
    if not pairs:
        raise SummaryError("at least one --pair is required")
    seeds = [int(pair["seed"]) for pair in pairs]
    if len(seeds) != len(set(seeds)):
        duplicates = sorted(seed for seed in set(seeds) if seeds.count(seed) > 1)
        raise SummaryError(f"duplicate seeds are not allowed: {duplicates}")

    pairs = sorted(pairs, key=lambda pair: int(pair["seed"]))
    baseline_mean = {
        metric: fmean(float(pair["baseline"][metric]) for pair in pairs)
        for metric in (*LOSS_METRICS, "wall_time_s")
    }
    treatment_mean = {
        metric: fmean(float(pair["treatment"][metric]) for pair in pairs)
        for metric in (*LOSS_METRICS, "wall_time_s")
    }
    mean_delta = {
        metric: fmean(float(pair["treatment_minus_baseline"][metric]) for pair in pairs)
        for metric in (*LOSS_METRICS, "wall_time_s")
    }
    mean_improvement = {
        metric: fmean(float(pair["loss_improvement"][metric]) for pair in pairs)
        for metric in LOSS_METRICS
    }
    better_seed_count = {
        metric: sum(float(pair["loss_improvement"][metric]) > 0.0 for pair in pairs)
        for metric in LOSS_METRICS
    }
    return {
        "schema": "outer_momentum_run_summary_v1",
        "pair_count": len(pairs),
        "seeds": [int(pair["seed"]) for pair in pairs],
        "validation": {
            "all_runs_full_quorum": True,
            "all_pairs_match_outer_steps": True,
            "all_pairs_match_quorum": True,
        },
        "aggregate": {
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "mean_treatment_minus_baseline": mean_delta,
            "mean_loss_improvement": mean_improvement,
            "treatment_better_seed_count": better_seed_count,
            "all_seeds_treatment_better": {
                metric: count == len(pairs)
                for metric, count in better_seed_count.items()
            },
            "mean_wall_time_relative_change": fmean(
                float(pair["wall_time_relative_change"]) for pair in pairs
            ),
        },
        "per_seed": pairs,
    }


def _fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def markdown(summary: dict) -> str:
    lines = [
        "# Outer-Momentum Run Summary",
        "",
        "Positive improvement means the treatment has lower loss than the baseline.",
        "",
        "| Seed | Steps | Quorum | Internal B | Internal T | Internal improvement | Holdout A B | Holdout A T | Holdout A improvement | Holdout B B | Holdout B T | Holdout B improvement | Mean holdout improvement | Wall B (s) | Wall T (s) | Wall change |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in summary["per_seed"]:
        baseline = pair["baseline"]
        treatment = pair["treatment"]
        gain = pair["loss_improvement"]
        lines.append(
            "| {seed} | {steps} | {quorum} | {ib} | {it} | {ig} | "
            "{ab} | {at} | {ag} | {bb} | {bt} | {bg} | {mg} | "
            "{wb:.0f} | {wt:.0f} | {wc:+.2%} |".format(
                seed=pair["seed"],
                steps=pair["outer_steps"],
                quorum=pair["quorum"],
                ib=_fmt(baseline["internal_loss"]),
                it=_fmt(treatment["internal_loss"]),
                ig=_fmt(gain["internal_loss"]),
                ab=_fmt(baseline["holdout_loss"]),
                at=_fmt(treatment["holdout_loss"]),
                ag=_fmt(gain["holdout_loss"]),
                bb=_fmt(baseline["holdout_indices7000_loss"]),
                bt=_fmt(treatment["holdout_indices7000_loss"]),
                bg=_fmt(gain["holdout_indices7000_loss"]),
                mg=_fmt(gain["holdout_mean_loss"]),
                wb=baseline["wall_time_s"],
                wt=treatment["wall_time_s"],
                wc=pair["wall_time_relative_change"],
            )
        )

    aggregate = summary["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Pairs: `{summary['pair_count']}`",
            f"- Seeds: `{summary['seeds']}`",
            "- Mean internal-loss improvement: "
            f"`{_fmt(aggregate['mean_loss_improvement']['internal_loss'])}`",
            "- Mean holdout-A improvement: "
            f"`{_fmt(aggregate['mean_loss_improvement']['holdout_loss'])}`",
            "- Mean holdout-B improvement: "
            f"`{_fmt(aggregate['mean_loss_improvement']['holdout_indices7000_loss'])}`",
            "- Mean two-holdout improvement: "
            f"`{_fmt(aggregate['mean_loss_improvement']['holdout_mean_loss'])}`",
            "- Mean wall-time change: "
            f"`{aggregate['mean_wall_time_relative_change']:+.2%}`",
            "",
            "## Validation",
            "",
            "- Every event-tape step had the configured full quorum: `True`",
            "- Baseline and treatment outer-step counts matched within every pair: `True`",
            "- Baseline and treatment configured quorums matched within every pair: `True`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        required=True,
        metavar=("SEED", "BASELINE_DIR", "TREATMENT_DIR"),
        help="paired seed and run directories; repeat for each seed",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pairs = []
        for seed_text, baseline_text, treatment_text in args.pair:
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise SummaryError(f"invalid seed {seed_text!r}") from exc
            pairs.append(summarize_pair(seed, Path(baseline_text), Path(treatment_text)))
        summary = aggregate_pairs(pairs)
    except SummaryError as exc:
        raise SystemExit(f"error: {exc}") from exc

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    rendered = markdown(summary)
    args.out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out_md.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
