#!/usr/bin/env python3
"""Summarize comparable strict-quorum outer-policy training runs."""

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
ARM_CONFIG_RE = re.compile(r"(?m)^\s*(\S+)\s+M=(\d+)\b")
ARM_RESULT_RE = re.compile(
    rf"(?m)^\s*(?:\+\s*)?\[compare\]\s+(\S+)\s+eval loss/token:\s*"
    rf"({NUMBER})\s+\(({NUMBER})s\)\s*$"
)
EVAL_LOSS_RE = re.compile(rf"(?m)^\s*EVAL_LOSS\s+({NUMBER})\s*$")

LOSS_FIELDS = (
    "internal_loss",
    "holdout_loss",
    "holdout_indices7000_loss",
    "holdout_mean_loss",
)


class SummaryError(RuntimeError):
    """Raised when a run is missing, malformed, or not strictly comparable."""


@dataclass(frozen=True)
class RunResult:
    directory: str
    run_name: str
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


def parse_run_log(path: Path) -> tuple[str, float, float, int]:
    text = _read_required(path)
    configs = ARM_CONFIG_RE.findall(text)
    if len(configs) != 1:
        raise SummaryError(
            f"{path}: expected exactly one async arm M=<quorum> configuration, "
            f"found {len(configs)}"
        )
    run_name, quorum_text = configs[0]

    results = [
        result
        for result in ARM_RESULT_RE.findall(text)
        if result[0] == run_name
    ]
    if len(results) != 1:
        raise SummaryError(
            f"{path}: expected exactly one internal eval-loss/wall-time result "
            f"for configured arm {run_name!r}, "
            f"found {len(results)}"
        )
    result_name, internal_text, wall_text = results[0]
    assert result_name == run_name

    quorum = int(quorum_text)
    if quorum <= 0:
        raise SummaryError(f"{path}: invalid quorum {quorum}")
    internal_loss = _finite_float(
        internal_text, field="internal loss", path=path
    )
    wall_time_s = _finite_float(wall_text, field="wall time", path=path)
    if internal_loss <= 0.0:
        raise SummaryError(f"{path}: internal loss must be positive")
    if wall_time_s <= 0.0:
        raise SummaryError(f"{path}: wall time must be positive")
    return run_name, internal_loss, wall_time_s, quorum


def parse_eval_loss(path: Path) -> float:
    text = _read_required(path)
    matches = EVAL_LOSS_RE.findall(text)
    if len(matches) != 1:
        raise SummaryError(
            f"{path}: expected exactly one EVAL_LOSS, found {len(matches)}"
        )
    loss = _finite_float(matches[0], field="EVAL_LOSS", path=path)
    if loss <= 0.0:
        raise SummaryError(f"{path}: EVAL_LOSS must be positive")
    return loss


def parse_tape(path: Path, *, expected_quorum: int) -> tuple[int, int, int]:
    text = _read_required(path)
    lines = [
        (line_number, line)
        for line_number, line in enumerate(text.splitlines(), 1)
        if line.strip()
    ]
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

        ids: list[int] = []
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

        if len(ids) != expected_quorum or set(ids) != expected_ids:
            raise SummaryError(
                f"{path}:{line_number}: non-full-quorum step {step}: expected "
                f"responder ids {sorted(expected_ids)}, got {ids}"
            )
        if len(ids) != len(set(ids)):
            raise SummaryError(
                f"{path}:{line_number}: duplicate responder id at step {step}: {ids}"
            )

        steps.append(step)
        responder_counts.append(len(ids))

    expected_steps = list(range(1, len(steps) + 1))
    if steps != expected_steps:
        raise SummaryError(
            f"{path}: outer steps must be contiguous and ordered from 1; "
            f"expected {expected_steps}, got {steps}"
        )
    return len(steps), min(responder_counts), max(responder_counts)


def parse_run(directory: Path) -> RunResult:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise SummaryError(f"run directory does not exist: {directory}")

    run_name, internal_loss, wall_time_s, quorum = parse_run_log(
        directory / "run.log"
    )
    holdout_loss = parse_eval_loss(directory / "holdout_eval.log")
    holdout_indices7000_loss = parse_eval_loss(
        directory / "holdout_indices7000_eval.log"
    )
    outer_steps, responder_min, responder_max = parse_tape(
        directory / "work" / run_name / "tape.jsonl",
        expected_quorum=quorum,
    )
    return RunResult(
        directory=str(directory),
        run_name=run_name,
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


def summarize_policy_matrix(
    arm_specs: Sequence[tuple[str, Path]], reference: str | None = None
) -> dict:
    if not arm_specs:
        raise SummaryError("at least one --arm is required")

    parsed: dict[str, RunResult] = {}
    for raw_label, directory in arm_specs:
        label = raw_label.strip()
        if not label:
            raise SummaryError("arm labels must not be empty")
        if label in parsed:
            raise SummaryError(f"duplicate arm label: {label!r}")
        parsed[label] = parse_run(directory)

    reference_label = reference.strip() if reference is not None else None
    if reference_label == "":
        raise SummaryError("reference label must not be empty")
    if reference_label is not None and reference_label not in parsed:
        raise SummaryError(
            f"reference label {reference_label!r} is not one of "
            f"{sorted(parsed)}"
        )

    steps_by_label = {label: run.outer_steps for label, run in parsed.items()}
    if len(set(steps_by_label.values())) != 1:
        raise SummaryError(f"nonmatching outer-step counts: {steps_by_label}")
    quorum_by_label = {label: run.quorum for label, run in parsed.items()}
    if len(set(quorum_by_label.values())) != 1:
        raise SummaryError(f"nonmatching configured quorums: {quorum_by_label}")

    ranked = sorted(
        parsed.items(), key=lambda item: (item[1].holdout_mean_loss, item[0])
    )
    reference_run = parsed.get(reference_label) if reference_label is not None else None
    arms = []
    for rank, (label, run) in enumerate(ranked, 1):
        row = {"label": label, "rank": rank, **asdict(run)}
        row["gain_vs_reference"] = (
            None
            if reference_run is None
            else {
                field: getattr(reference_run, field) - getattr(run, field)
                for field in LOSS_FIELDS
            }
        )
        arms.append(row)

    common_steps = next(iter(steps_by_label.values()))
    common_quorum = next(iter(quorum_by_label.values()))
    return {
        "schema": "outer_policy_matrix_summary_v1",
        "arm_count": len(arms),
        "reference_label": reference_label,
        "gain_definition": "reference loss minus arm loss; positive favors the arm",
        "outer_steps": common_steps,
        "quorum": common_quorum,
        "ranking": [row["label"] for row in arms],
        "validation": {
            "all_runs_full_quorum": True,
            "all_runs_match_outer_steps": True,
            "all_runs_match_quorum": True,
        },
        "arms": arms,
    }


def _fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def markdown(summary: dict) -> str:
    reference = summary["reference_label"]
    lines = [
        "# Outer Policy Matrix",
        "",
        f"Reference: `{reference}`" if reference is not None else "Reference: `none`",
        "",
        "Positive gain means the arm has lower loss than the reference.",
        "",
        "| Rank | Arm | Internal | Holdout A | Holdout B | Mean holdout | Internal gain | A gain | B gain | Mean gain | Steps | Quorum | Wall (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        gain = arm["gain_vs_reference"]
        lines.append(
            "| {rank} | {label} | {internal} | {holdout_a} | {holdout_b} | "
            "{holdout_mean} | {internal_gain} | {a_gain} | {b_gain} | "
            "{mean_gain} | {steps} | {quorum} | {wall:.1f} |".format(
                rank=arm["rank"],
                label=arm["label"],
                internal=_fmt(arm["internal_loss"]),
                holdout_a=_fmt(arm["holdout_loss"]),
                holdout_b=_fmt(arm["holdout_indices7000_loss"]),
                holdout_mean=_fmt(arm["holdout_mean_loss"]),
                internal_gain=_fmt(
                    None if gain is None else gain["internal_loss"]
                ),
                a_gain=_fmt(None if gain is None else gain["holdout_loss"]),
                b_gain=_fmt(
                    None if gain is None else gain["holdout_indices7000_loss"]
                ),
                mean_gain=_fmt(
                    None if gain is None else gain["holdout_mean_loss"]
                ),
                steps=arm["outer_steps"],
                quorum=arm["quorum"],
                wall=arm["wall_time_s"],
            )
        )

    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Every tape step used the complete configured responder set: `True`",
            "- All arms have matching outer-step counts: `True`",
            "- All arms have matching configured quorums: `True`",
            "",
            "## Runs",
            "",
        ]
    )
    for arm in summary["arms"]:
        lines.append(f"- `{arm['label']}`: `{arm['directory']}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        nargs=2,
        required=True,
        metavar=("LABEL", "RUN_DIR"),
        help="policy label and strict-run directory; repeat for every arm",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="optional arm label used to compute positive-is-better loss gains",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_policy_matrix(
            [(label, Path(directory)) for label, directory in args.arm],
            reference=args.reference,
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
