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
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    train, evaluation = split_rows(rows, args.eval_rows, args.seed)
    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.eval_output), evaluation)
    print(
        f"split {len(rows)} rows into {len(train)} train and "
        f"{len(evaluation)} evaluation rows (seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
