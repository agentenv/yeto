#!/usr/bin/env python3
"""Run one online stock/PTI comparison and its fail-closed validator.

All arguments after ``--`` are passed unchanged to ``compare_diloco.py``.
The compare argv must explicitly declare ``--work-dir``, ``--report-dir``, and
``--settings STOCK,PTI`` so the validator cannot inspect a different run than
the one this wrapper launched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARE = REPO_ROOT / "scripts" / "compare_diloco.py"
VALIDATOR = REPO_ROOT / "scripts" / "validate_pti_online_pair.py"


class RunnerError(RuntimeError):
    pass


def _declared_option(argv: list[str], name: str) -> str:
    values: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == name:
            if index + 1 >= len(argv):
                raise RunnerError(f"compare argv has no value after {name}")
            values.append(argv[index + 1])
            index += 2
            continue
        prefix = name + "="
        if argument.startswith(prefix):
            values.append(argument[len(prefix) :])
        index += 1
    if len(values) != 1 or not values[0]:
        raise RunnerError(
            f"compare argv must declare {name} exactly once; found {len(values)}"
        )
    return values[0]


def run_compare_then_validate(
    *,
    compare_argv: list[str],
    stock_arm: str,
    pti_arm: str,
    output: Path,
    python: str,
) -> None:
    if not compare_argv:
        raise RunnerError("missing compare argv after --")
    if stock_arm == pti_arm:
        raise RunnerError("stock and PTI arm names must differ")
    work_dir = Path(_declared_option(compare_argv, "--work-dir")).resolve()
    report_dir = Path(_declared_option(compare_argv, "--report-dir")).resolve()
    settings = _declared_option(compare_argv, "--settings")
    expected_settings = f"{stock_arm},{pti_arm}"
    if settings != expected_settings:
        raise RunnerError(
            f"compare --settings must be exactly {expected_settings!r}, got {settings!r}"
        )

    subprocess.run(
        [python, str(COMPARE), *compare_argv],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            python,
            str(VALIDATOR),
            "--results",
            str(report_dir / "results.jsonl"),
            "--stock-arm-dir",
            str(work_dir / stock_arm),
            "--pti-arm-dir",
            str(work_dir / pti_arm),
            "--stock-arm",
            stock_arm,
            "--pti-arm",
            pti_arm,
            "--output",
            str(output.resolve()),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock-arm", required=True)
    parser.add_argument("--pti-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("compare_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.compare_argv[:1] == ["--"]:
        args.compare_argv = args.compare_argv[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_compare_then_validate(
            compare_argv=args.compare_argv,
            stock_arm=args.stock_arm,
            pti_arm=args.pti_arm,
            output=args.output,
            python=args.python,
        )
    except RunnerError as exc:
        print(f"PTI online runner configuration error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
