#!/usr/bin/env python3
"""Build a deterministic CyberGym curriculum from baseline outcomes.

The selector keeps the historical task split reproducible while changing the
order seen by RL: high-signal boundary tasks lead each band, hard zero-success
tasks are interleaved as anchors, and a few known-success tasks are distributed
throughout the stream.  A held-out list is never reused for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            metadata = row.get("metadata") if isinstance(row, dict) else None
            task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"prompt row {line_number} has no metadata.task_id")
            if task_id in rows:
                raise ValueError(f"duplicate task_id in prompt data: {task_id}")
            rows[task_id] = row
    if not rows:
        raise ValueError(f"prompt data is empty: {path}")
    return rows


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("selection manifest must be a JSON object")
    return value


def _ids(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(
            f"selection manifest field {name!r} must be a list of task IDs"
        )
    if len(value) != len(set(value)):
        raise ValueError(f"selection manifest field {name!r} contains duplicates")
    return value


def _baseline_successes(manifest: dict[str, Any]) -> dict[str, int]:
    values = manifest.get("boundary_baseline_successes")
    if values is None:
        values = manifest.get("baseline_successes", {})
    if not isinstance(values, dict):
        raise TypeError("baseline success counts must be a JSON object")
    result = {}
    for task_id, count in values.items():
        if (
            not isinstance(task_id, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
        ):
            raise TypeError("baseline success counts must map task IDs to integers")
        result[task_id] = count
    return result


def _annotated(
    row: dict[str, Any], *, bucket: str, successes: int | None, attempts: int
) -> dict[str, Any]:
    result = dict(row)
    metadata = dict(result.get("metadata") or {})
    metadata["training_bucket"] = bucket
    metadata["baseline_attempts"] = attempts
    if successes is not None:
        metadata["baseline_successes"] = successes
    result["metadata"] = metadata
    return result


def _interleave(
    boundary: list[dict[str, Any]],
    hard: list[dict[str, Any]],
    easy: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return an easy-to-hard stream with hard anchors and easy anchors."""

    output: list[dict[str, Any]] = []
    hard_index = 0
    easy_index = 0
    for index in range(0, len(boundary), 3):
        output.extend(boundary[index : index + 3])
        if hard_index < len(hard):
            output.append(hard[hard_index])
            hard_index += 1
        # Spread known-success anchors over the stream, rather than putting
        # them in one contiguous batch that would hide collapse late in a run.
        if easy and easy_index < len(easy):
            target = round((easy_index + 1) * len(boundary) / (len(easy) + 1))
            if index + 3 >= target:
                output.append(easy[easy_index])
                easy_index += 1
    output.extend(hard[hard_index:])
    output.extend(easy[easy_index:])
    return output


def select(
    prompts: Path,
    selection_manifest: Path,
    *,
    train_count: int,
    eval_count: int,
    boundary_count: int | None = None,
    hard_count: int | None = None,
    easy_count: int | None = None,
    attempts: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(prompts)
    manifest = _read_manifest(selection_manifest)
    successes = _baseline_successes(manifest)
    attempts = int(attempts or manifest.get("baseline_attempts_per_task", 16))
    if attempts <= 0:
        raise ValueError("baseline attempts must be positive")

    boundary_ids = _ids(manifest.get("boundary_task_ids", []), "boundary_task_ids")
    hard_ids = _ids(manifest.get("hard_task_ids", []), "hard_task_ids")
    easy_ids = _ids(manifest.get("easy_success_task_ids", []), "easy_success_task_ids")
    heldout_ids = _ids(manifest.get("heldout_task_ids", []), "heldout_task_ids")
    if not boundary_ids and not hard_ids and not easy_ids:
        for task_id, count in successes.items():
            if count == 0:
                hard_ids.append(task_id)
            elif count >= attempts:
                easy_ids.append(task_id)
            else:
                boundary_ids.append(task_id)
    selected_ids = set(boundary_ids + hard_ids + easy_ids + heldout_ids)
    missing = sorted(task_id for task_id in selected_ids if task_id not in rows)
    if missing:
        raise ValueError(
            f"prompt data is missing selected task IDs: {', '.join(missing[:5])}"
        )
    heldout_set = set(heldout_ids)
    if heldout_set & set(boundary_ids + hard_ids + easy_ids):
        raise ValueError("held-out tasks overlap the training selection")

    def limit(values: list[str], requested: int | None, name: str) -> list[str]:
        count = len(values) if requested is None else requested
        if count < 0 or count > len(values):
            raise ValueError(f"{name} exceeds the available selection")
        return values[:count]

    boundary_ids = limit(boundary_ids, boundary_count, "boundary_count")
    hard_ids = limit(hard_ids, hard_count, "hard_count")
    easy_ids = limit(easy_ids, easy_count, "easy_count")
    if len(boundary_ids) + len(hard_ids) + len(easy_ids) != train_count:
        raise ValueError(
            "train_count must equal boundary_count + hard_count + easy_count"
        )
    if len(heldout_ids) < eval_count:
        raise ValueError("eval_count exceeds the available held-out selection")
    heldout_ids = heldout_ids[:eval_count]

    def annotated(task_id: str, bucket: str) -> dict[str, Any]:
        successes_for_task = successes.get(task_id)
        if successes_for_task is None and bucket == "hard":
            successes_for_task = 0
        elif successes_for_task is None and bucket == "easy_success":
            successes_for_task = attempts
        return _annotated(
            rows[task_id],
            bucket=bucket,
            successes=successes_for_task,
            attempts=attempts,
        )

    boundary = [annotated(task_id, "boundary") for task_id in boundary_ids]
    boundary.sort(
        key=lambda row: (
            -int(row["metadata"].get("baseline_successes", -1)),
            row["metadata"]["task_id"],
        )
    )
    hard = [annotated(task_id, "hard") for task_id in sorted(hard_ids)]
    easy = [annotated(task_id, "easy_success") for task_id in easy_ids]
    train = _interleave(boundary, hard, easy)
    evaluation = [annotated(task_id, "heldout") for task_id in heldout_ids]
    if len(train) != train_count or len(
        {row["metadata"]["task_id"] for row in train}
    ) != train_count:
        raise ValueError("curriculum selector produced an invalid training set")
    return train, evaluation, {
        "selection": "baseline-outcome curriculum with variance anchors",
        "baseline_attempts_per_task": attempts,
        "train_tasks": [row["metadata"]["task_id"] for row in train],
        "eval_tasks": [row["metadata"]["task_id"] for row in evaluation],
        "counts": {
            "boundary": len(boundary),
            "hard": len(hard),
            "easy_success": len(easy),
        },
        "source_prompts_sha256": _sha256(prompts),
        "source_selection_manifest_sha256": _sha256(selection_manifest),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=110)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--boundary-count", type=int, default=None)
    parser.add_argument("--hard-count", type=int, default=None)
    parser.add_argument("--easy-count", type=int, default=None)
    parser.add_argument("--baseline-attempts", type=int, default=None)
    args = parser.parse_args(argv)
    train, evaluation, manifest = select(
        args.prompts,
        args.selection_manifest,
        train_count=args.train_count,
        eval_count=args.eval_count,
        boundary_count=args.boundary_count,
        hard_count=args.hard_count,
        easy_count=args.easy_count,
        attempts=args.baseline_attempts,
    )
    _write_jsonl(args.train_output, train)
    _write_jsonl(args.eval_output, evaluation)
    manifest.update(
        {
            "train_output_sha256": _sha256(args.train_output),
            "eval_output_sha256": _sha256(args.eval_output),
        }
    )
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
