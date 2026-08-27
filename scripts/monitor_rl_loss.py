#!/usr/bin/env python3
"""Extract and optionally follow privacy-bounded RL train/eval telemetry."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto.rl.loss_curve import (
    MetricRow,
    iter_metric_stream,
    persist_summaries,
    scan_log,
    validate_eval_dataset,
    validate_source_label,
    write_png,
)


@dataclass
class SourceState:
    label: str
    path: Path
    offset: int = 0
    identity: tuple[int, int] | None = None
    anchor_start: int = 0
    anchor: bytes = b""


_ANCHOR_BYTES = 256


def parse_source(spec: str) -> SourceState:
    try:
        label, raw_path = spec.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source must be LABEL=PATH") from error
    try:
        validate_source_label(label)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not raw_path:
        raise argparse.ArgumentTypeError("source path must not be empty")
    return SourceState(label, Path(raw_path).expanduser())


def parse_source_label(label: str) -> str:
    try:
        return validate_source_label(label)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only allowlisted scalar train/eval metrics from Miles/Yeto logs. "
            "Unknown fields and payloads are never written."
        )
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--source",
        action="append",
        type=parse_source,
        metavar="LABEL=PATH",
        help="repeat for each island or learner log; labels must be opaque",
    )
    inputs.add_argument(
        "--stdin-source",
        type=parse_source_label,
        metavar="LABEL",
        help=(
            "read a seekless live log stream from stdin; run one process per "
            "stream and share the same locked outputs"
        ),
    )
    parser.add_argument(
        "--eval-dataset",
        type=validate_eval_dataset,
        help="bind eval extraction to this exact Miles dataset label",
    )
    parser.add_argument(
        "--csv", required=True, type=Path, help="append-safe CSV output"
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="atomic JSON summary (default: CSV path with .json suffix)",
    )
    parser.add_argument(
        "--png",
        type=Path,
        help="optional plot; skipped if matplotlib is unavailable",
    )
    parser.add_argument(
        "--follow", action="store_true", help="follow growing/rotated logs"
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=30.0,
        help="follow polling interval (default: 30)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.interval_seconds) or args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be finite and positive")
    if args.stdin_source is not None and args.follow:
        raise ValueError("stdin is already streaming; do not combine it with --follow")
    output_list = [args.csv.resolve(), args.json.resolve()]
    if args.png is not None:
        output_list.append(args.png.resolve())
    if len(output_list) != len(set(output_list)):
        raise ValueError("CSV, JSON, and PNG output paths must be distinct")
    outputs = set(output_list)
    sources = args.source or []
    if len({source.label for source in sources}) != len(sources):
        raise ValueError("source labels must be unique")
    for source in sources:
        if source.path.resolve() in outputs:
            raise ValueError("an output path cannot also be a log source")


def _anchor_matches(state: SourceState) -> bool:
    if not state.anchor:
        return True
    with state.path.open("rb") as handle:
        handle.seek(state.anchor_start)
        return handle.read(len(state.anchor)) == state.anchor


def _record_anchor(state: SourceState) -> None:
    if state.offset <= 0:
        state.anchor_start = 0
        state.anchor = b""
        return
    state.anchor_start = max(0, state.offset - _ANCHOR_BYTES)
    with state.path.open("rb") as handle:
        handle.seek(state.anchor_start)
        state.anchor = handle.read(state.offset - state.anchor_start)


def _scan_state(
    state: SourceState, *, follow: bool, eval_dataset: str | None = None
) -> list[MetricRow]:
    try:
        stat = state.path.stat()
    except FileNotFoundError:
        if follow:
            return []
        raise
    identity = stat.st_dev, stat.st_ino
    if (
        state.identity != identity
        or stat.st_size < state.offset
        or not _anchor_matches(state)
    ):
        state.offset = 0
        state.identity = identity
        state.anchor_start = 0
        state.anchor = b""
    rows, state.offset = scan_log(
        state.path,
        state.label,
        offset=state.offset,
        complete_lines_only=follow,
        eval_dataset=eval_dataset,
    )
    _record_anchor(state)
    return rows


def _refresh(args: argparse.Namespace, *, first: bool) -> list[MetricRow]:
    fresh: list[MetricRow] = []
    for source in args.source or []:
        fresh.extend(
            _scan_state(
                source,
                follow=args.follow,
                eval_dataset=args.eval_dataset,
            )
        )
    if not fresh and not first and args.csv.exists():
        return []
    merged = persist_summaries(fresh, args.csv, args.json)
    if args.png is not None and not write_png(merged, args.png):
        print(
            "PNG skipped: matplotlib unavailable or no scalar points", file=sys.stderr
        )
    latest = ",".join(
        f"{label}:{max(row.step for row in merged if row.source == label)}"
        for label in sorted({row.source for row in merged})
    )
    print(f"scalar_points={len(merged)} latest_steps={latest or 'none'}")
    return merged


def _consume_stdin(args: argparse.Namespace) -> None:
    for row in iter_metric_stream(
        sys.stdin.buffer,
        args.stdin_source,
        eval_dataset=args.eval_dataset,
    ):
        merged = persist_summaries([row], args.csv, args.json)
        if args.png is not None and not write_png(merged, args.png):
            print(
                "PNG skipped: matplotlib unavailable or no scalar points",
                file=sys.stderr,
            )
        latest = max(
            point.step for point in merged if point.source == args.stdin_source
        )
        print(f"scalar_points={len(merged)} latest_steps={args.stdin_source}:{latest}")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json is None:
        args.json = args.csv.with_suffix(".json")
    try:
        _validate_args(args)
        if args.stdin_source is not None:
            _consume_stdin(args)
            return 0
        _refresh(args, first=True)
        if not args.follow:
            return 0
        while True:
            time.sleep(args.interval_seconds)
            _refresh(args, first=False)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
