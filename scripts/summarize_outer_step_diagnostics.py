#!/usr/bin/env python3
"""Summarize mechanism diagnostics from strict-run syncer event tapes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Sequence


REQUIRED_FIELDS = {
    "step",
    "fragment",
    "gnorm",
    "outer_step_norm",
    "outer_direction_cosine",
    "outer_history_current_ratio",
    "outer_restarted",
}


class SummaryError(RuntimeError):
    """Raised when a tape is missing, malformed, or diagnostically incomplete."""


def _required_number(
    row: dict,
    field: str,
    *,
    path: Path,
    line_number: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(
            f"{path}:{line_number}: {field} must be a finite number, got {value!r}"
        )
    parsed = float(value)
    if not math.isfinite(parsed):
        raise SummaryError(
            f"{path}:{line_number}: {field} must be finite, got {value!r}"
        )
    if minimum is not None and parsed < minimum:
        raise SummaryError(
            f"{path}:{line_number}: {field} must be >= {minimum}, got {parsed}"
        )
    if maximum is not None and parsed > maximum:
        raise SummaryError(
            f"{path}:{line_number}: {field} must be <= {maximum}, got {parsed}"
        )
    return parsed


def _optional_number(
    row: dict,
    field: str,
    *,
    path: Path,
    line_number: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if row[field] is None:
        return None
    return _required_number(
        row,
        field,
        path=path,
        line_number=line_number,
        minimum=minimum,
        maximum=maximum,
    )


def parse_tape(path: Path) -> list[dict]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SummaryError(f"missing tape file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc

    records: list[dict] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SummaryError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SummaryError(f"{path}:{line_number}: tape record must be an object")

        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            raise SummaryError(
                f"{path}:{line_number}: inconsistent step schema; missing required "
                f"diagnostics {missing}"
            )

        step = row["step"]
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise SummaryError(f"{path}:{line_number}: invalid step {step!r}")
        fragment = row["fragment"]
        if (
            not isinstance(fragment, int)
            or isinstance(fragment, bool)
            or fragment < 0
        ):
            raise SummaryError(f"{path}:{line_number}: invalid fragment {fragment!r}")
        restarted = row["outer_restarted"]
        if not isinstance(restarted, bool):
            raise SummaryError(
                f"{path}:{line_number}: outer_restarted must be a boolean, "
                f"got {restarted!r}"
            )

        records.append(
            {
                "step": step,
                "fragment": fragment,
                "gnorm": _required_number(
                    row,
                    "gnorm",
                    path=path,
                    line_number=line_number,
                    minimum=0.0,
                ),
                "outer_step_norm": _required_number(
                    row,
                    "outer_step_norm",
                    path=path,
                    line_number=line_number,
                    minimum=0.0,
                ),
                "outer_direction_cosine": _optional_number(
                    row,
                    "outer_direction_cosine",
                    path=path,
                    line_number=line_number,
                    minimum=-1.0,
                    maximum=1.0,
                ),
                "outer_history_current_ratio": _optional_number(
                    row,
                    "outer_history_current_ratio",
                    path=path,
                    line_number=line_number,
                    minimum=0.0,
                ),
                "outer_restarted": restarted,
            }
        )

    if not records:
        raise SummaryError(f"{path}: event tape has no records")

    steps = sorted(record["step"] for record in records)
    expected_steps = list(range(1, len(records) + 1))
    if steps != expected_steps:
        raise SummaryError(
            f"{path}: steps must be unique and contiguous from 1; "
            f"expected {expected_steps}, got {steps}"
        )
    return sorted(records, key=lambda record: record["step"])


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _defined_stats(
    values: Sequence[float | None], *, percentiles: Sequence[tuple[str, float]]
) -> dict:
    defined = [float(value) for value in values if value is not None]
    total = len(values)
    result = {
        "defined_count": len(defined),
        "mean": fmean(defined) if defined else None,
        "undefined_fraction": None if total == 0 else (total - len(defined)) / total,
    }
    for name, quantile in percentiles:
        result[name] = _percentile(defined, quantile)
    return result


def summarize_records(records: Sequence[dict]) -> dict:
    step_norms = [float(record["outer_step_norm"]) for record in records]
    gnorms = [float(record["gnorm"]) for record in records]
    cosines = [record["outer_direction_cosine"] for record in records]
    history_ratios = [record["outer_history_current_ratio"] for record in records]
    restart_count = sum(1 for record in records if record["outer_restarted"])
    total_step_norm = sum(step_norms)
    return {
        "record_count": len(records),
        "step_start": records[0]["step"] if records else None,
        "step_end": records[-1]["step"] if records else None,
        "outer_step_norm": {
            "mean": fmean(step_norms) if step_norms else None,
            "p50": _percentile(step_norms, 0.50),
            "p95": _percentile(step_norms, 0.95),
        },
        "outer_direction_cosine": _defined_stats(
            cosines, percentiles=(("p10", 0.10),)
        ),
        "outer_history_current_ratio": _defined_stats(
            history_ratios, percentiles=(("p50", 0.50), ("p95", 0.95))
        ),
        "restart_count": restart_count,
        "restart_rate": None if not records else restart_count / len(records),
        "gnorm_to_outer_step_norm_ratio": (
            None if total_step_norm == 0.0 else sum(gnorms) / total_step_norm
        ),
    }


def summarize_run(label: str, path: Path) -> dict:
    records = parse_tape(path)
    midpoint = len(records) // 2
    fragments = sorted({record["fragment"] for record in records})
    return {
        "label": label,
        "tape": str(path.expanduser().resolve()),
        "overall": summarize_records(records),
        "first_half": summarize_records(records[:midpoint]),
        "second_half": summarize_records(records[midpoint:]),
        "per_fragment": [
            {
                "fragment": fragment,
                **summarize_records(
                    [record for record in records if record["fragment"] == fragment]
                ),
            }
            for fragment in fragments
        ],
    }


def summarize_runs(run_specs: Sequence[tuple[str, Path]]) -> dict:
    if not run_specs:
        raise SummaryError("at least one --run is required")
    seen: set[str] = set()
    runs = []
    for raw_label, path in run_specs:
        label = raw_label.strip()
        if not label:
            raise SummaryError("run labels must not be empty")
        if label in seen:
            raise SummaryError(f"duplicate run label: {label!r}")
        seen.add(label)
        runs.append(summarize_run(label, path))
    return {
        "schema": "outer_step_diagnostics_summary_v1",
        "run_count": len(runs),
        "ratio_definition": (
            "sum(gnorm) / sum(outer_step_norm) within each reported scope"
        ),
        "half_definition": (
            "records sorted by step; first_half uses floor(N/2), second_half "
            "contains the remainder"
        ),
        "runs": runs,
    }


def _fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _summary_row(scope: str, summary: dict) -> str:
    step_norm = summary["outer_step_norm"]
    cosine = summary["outer_direction_cosine"]
    history = summary["outer_history_current_ratio"]
    return (
        "| {scope} | {records} | {step_mean} | {step_p50} | {step_p95} | "
        "{cos_mean} | {cos_p10} | {cos_undef} | {hist_mean} | {hist_p50} | "
        "{hist_p95} | {hist_undef} | {restarts} | {restart_rate} | {ratio} |"
    ).format(
        scope=scope,
        records=summary["record_count"],
        step_mean=_fmt(step_norm["mean"]),
        step_p50=_fmt(step_norm["p50"]),
        step_p95=_fmt(step_norm["p95"]),
        cos_mean=_fmt(cosine["mean"]),
        cos_p10=_fmt(cosine["p10"]),
        cos_undef=_fmt(cosine["undefined_fraction"]),
        hist_mean=_fmt(history["mean"]),
        hist_p50=_fmt(history["p50"]),
        hist_p95=_fmt(history["p95"]),
        hist_undef=_fmt(history["undefined_fraction"]),
        restarts=summary["restart_count"],
        restart_rate=_fmt(summary["restart_rate"]),
        ratio=_fmt(summary["gnorm_to_outer_step_norm_ratio"]),
    )


def markdown(summary: dict) -> str:
    lines = [
        "# Outer-Step Diagnostics",
        "",
        "No policy ranking is performed. Values describe realized outer-step mechanics.",
        "",
    ]
    header = (
        "| Scope | Records | Step norm mean | Step p50 | Step p95 | Cosine mean | "
        "Cosine p10 | Cosine undefined | History/current mean | History p50 | "
        "History p95 | History undefined | Restarts | Restart rate | gnorm/step norm |"
    )
    separator = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for run in summary["runs"]:
        lines.extend(
            [
                f"## {run['label']}",
                "",
                f"Tape: `{run['tape']}`",
                "",
                header,
                separator,
                _summary_row("overall", run["overall"]),
                _summary_row("first-half", run["first_half"]),
                _summary_row("second-half", run["second_half"]),
            ]
        )
        for fragment in run["per_fragment"]:
            lines.append(_summary_row(f"fragment {fragment['fragment']}", fragment))
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        required=True,
        metavar=("LABEL", "TAPE_JSONL"),
        help="run label and event-tape JSONL path; repeat for every run",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_runs(
            [(label, Path(tape)) for label, tape in args.run]
        )
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
