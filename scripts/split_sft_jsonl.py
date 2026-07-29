#!/usr/bin/env python3
"""Create deterministic train/evaluation splits from an SFT JSONL file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def split_rows(rows: list[dict], eval_rows: int, seed: int) -> tuple[list[dict], list[dict]]:
    if eval_rows <= 0:
        raise ValueError("--eval-rows must be greater than zero")
    if eval_rows >= len(rows):
        raise ValueError("--eval-rows must leave at least one training row")

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    eval_ids = set(order[:eval_rows])
    train = [row for idx, row in enumerate(rows) if idx not in eval_ids]
    evaluation = [row for idx, row in enumerate(rows) if idx in eval_ids]
    return train, evaluation


def _nested_value(row: dict, dotted_key: str):
    value = row
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"row is missing group key {dotted_key!r}")
        value = value[key]
    return value


def split_grouped_rows(
    rows: list[dict], group_key: str, eval_groups: int, seed: int
) -> tuple[list[dict], list[dict]]:
    groups = list(dict.fromkeys(_nested_value(row, group_key) for row in rows))
    if eval_groups <= 0:
        raise ValueError("--eval-groups must be greater than zero")
    if eval_groups >= len(groups):
        raise ValueError("--eval-groups must leave at least one training group")
    random.Random(seed).shuffle(groups)
    evaluation_groups = set(groups[:eval_groups])
    train = [row for row in rows if _nested_value(row, group_key) not in evaluation_groups]
    evaluation = [row for row in rows if _nested_value(row, group_key) in evaluation_groups]
    return train, evaluation


def split_grouped_rows_near_target(
    rows: list[dict], group_key: str, eval_rows: int, seed: int
) -> tuple[list[dict], list[dict]]:
    if eval_rows <= 0:
        raise ValueError("--eval-rows must be greater than zero")
    if eval_rows >= len(rows):
        raise ValueError("--eval-rows must leave at least one training row")

    grouped: dict[object, int] = {}
    for row in rows:
        group = _nested_value(row, group_key)
        grouped[group] = grouped.get(group, 0) + 1
    if len(grouped) < 2:
        raise ValueError("grouped splitting requires at least two groups")

    groups = list(grouped)
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: grouped[group], reverse=True)
    selected: list[object] = []
    selected_rows = 0
    for group in groups:
        size = grouped[group]
        if selected_rows + size <= eval_rows:
            selected.append(group)
            selected_rows += size
    if selected_rows < eval_rows:
        remaining = [group for group in groups if group not in selected]
        selected.append(
            min(remaining, key=lambda group: abs(selected_rows + grouped[group] - eval_rows))
        )

    evaluation_groups = set(selected)
    if len(evaluation_groups) == len(groups):
        raise ValueError("grouped split must leave at least one training group")
    train = [row for row in rows if _nested_value(row, group_key) not in evaluation_groups]
    evaluation = [row for row in rows if _nested_value(row, group_key) in evaluation_groups]
    return train, evaluation


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument("--eval-rows", type=int, default=40)
    parser.add_argument("--group-key", help="Dotted row key used to prevent group leakage")
    parser.add_argument(
        "--eval-groups",
        type=int,
        help="Exact number of held-out groups; otherwise --eval-rows is the grouped target",
    )
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if args.group_key:
        if args.eval_groups is not None:
            train, evaluation = split_grouped_rows(
                rows, args.group_key, args.eval_groups, args.seed
            )
        else:
            train, evaluation = split_grouped_rows_near_target(
                rows, args.group_key, args.eval_rows, args.seed
            )
    else:
        train, evaluation = split_rows(rows, args.eval_rows, args.seed)
    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.eval_output), evaluation)
    print(
        f"split {len(rows)} rows into {len(train)} train and "
        f"{len(evaluation)} evaluation rows (seed={args.seed}"
        f"{', grouped by ' + args.group_key if args.group_key else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
