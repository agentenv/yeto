"""Build an attested, deterministic subset of a SecrlEnv JSONL dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class DatasetSubsetError(ValueError):
    """Raised when a requested dataset subset is ambiguous or invalid."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _validate_paths(
    parent_jsonl: Path,
    ordered_task_ids: Path | None,
    subset_output: Path,
    manifest_output: Path,
) -> None:
    inputs = {_resolved(parent_jsonl)}
    if ordered_task_ids is not None:
        inputs.add(_resolved(ordered_task_ids))
    outputs = {_resolved(subset_output), _resolved(manifest_output)}
    if len(outputs) != 2:
        raise DatasetSubsetError("subset and manifest outputs must be different files")
    if inputs & outputs:
        raise DatasetSubsetError("outputs must not overwrite an input file")


def _read_parent(
    path: Path,
) -> tuple[bytes, list[str], dict[str, bytes]]:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise DatasetSubsetError(f"could not read parent JSONL: {exc}") from exc
    if not value:
        raise DatasetSubsetError("parent JSONL is empty")

    ordered_ids: list[str] = []
    rows: dict[str, bytes] = {}
    for line_number, raw_row in enumerate(value.splitlines(keepends=True), 1):
        if not raw_row.endswith(b"\n"):
            raise DatasetSubsetError(
                "parent JSONL must terminate every row with a newline"
            )
        if not raw_row.strip():
            raise DatasetSubsetError(
                f"parent JSONL contains a blank row at line {line_number}"
            )
        try:
            decoded = raw_row.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatasetSubsetError(
                f"parent JSONL row {line_number} is not UTF-8"
            ) from exc
        try:
            row = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise DatasetSubsetError(
                f"parent JSONL row {line_number} is invalid JSON"
            ) from exc
        metadata = row.get("metadata") if isinstance(row, dict) else None
        task_id = metadata.get("task_id") if isinstance(metadata, dict) else None
        if (
            not isinstance(task_id, str)
            or not task_id.strip()
            or task_id != task_id.strip()
            or any(
                ord(character) < 32 or ord(character) == 127 for character in task_id
            )
        ):
            raise DatasetSubsetError(
                f"parent JSONL row {line_number} has no valid metadata.task_id"
            )
        if task_id in rows:
            raise DatasetSubsetError("parent JSONL contains duplicate task IDs")
        ordered_ids.append(task_id)
        rows[task_id] = raw_row
    return value, ordered_ids, rows


def _read_ordered_ids(path: Path) -> tuple[bytes, list[str]]:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise DatasetSubsetError(f"could not read ordered task-ID file: {exc}") from exc
    try:
        decoded = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetSubsetError("ordered task-ID file is not UTF-8") from exc
    task_ids = decoded.splitlines()
    if not task_ids:
        raise DatasetSubsetError("ordered task-ID file is empty")
    if any(not task_id.strip() or task_id != task_id.strip() for task_id in task_ids):
        raise DatasetSubsetError(
            "ordered task-ID file contains a blank or whitespace-padded ID"
        )
    if len(task_ids) != len(set(task_ids)):
        raise DatasetSubsetError("ordered task-ID file contains duplicate IDs")
    return value, task_ids


def _stage(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _write_exclusive_artifacts(
    subset_output: Path,
    subset: bytes,
    manifest_output: Path,
    manifest: bytes,
) -> None:
    if os.path.lexists(subset_output) or os.path.lexists(manifest_output):
        raise DatasetSubsetError("refusing to overwrite an existing output")

    staged_subset: Path | None = None
    staged_manifest: Path | None = None
    created: list[Path] = []
    try:
        staged_subset = _stage(subset_output, subset)
        staged_manifest = _stage(manifest_output, manifest)
        os.link(staged_subset, subset_output)
        created.append(subset_output)
        os.link(staged_manifest, manifest_output)
        created.append(manifest_output)
    except FileExistsError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise DatasetSubsetError("refusing to overwrite an existing output") from exc
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        if staged_subset is not None:
            staged_subset.unlink(missing_ok=True)
        if staged_manifest is not None:
            staged_manifest.unlink(missing_ok=True)


def build_subset(
    parent_jsonl: Path,
    subset_output: Path,
    manifest_output: Path,
    *,
    expected_parent_sha256: str,
    expected_parent_count: int,
    expected_count: int,
    task_pack_sha256: str,
    selection_rule: str,
    selection_seed: int | None = None,
    ordered_task_ids: Path | None = None,
    first_n: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed subset and return its deterministic manifest.

    Explicit task IDs are read as one UTF-8 ID per line. ``first_n`` is the
    only mode that derives selection from parent-row order, and callers must
    opt into it explicitly.
    """

    parent_jsonl = Path(parent_jsonl)
    subset_output = Path(subset_output)
    manifest_output = Path(manifest_output)
    ordered_task_ids = None if ordered_task_ids is None else Path(ordered_task_ids)
    if (ordered_task_ids is None) == (not first_n):
        raise DatasetSubsetError(
            "choose exactly one of an ordered task-ID file or explicit first-N mode"
        )
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise DatasetSubsetError("expected count must be a positive integer")
    if not _SHA256.fullmatch(expected_parent_sha256):
        raise DatasetSubsetError(
            "expected parent SHA-256 must be 64 lowercase hex digits"
        )
    if (
        isinstance(expected_parent_count, bool)
        or not isinstance(expected_parent_count, int)
        or expected_parent_count <= 0
    ):
        raise DatasetSubsetError("expected parent count must be a positive integer")
    if not _SHA256.fullmatch(task_pack_sha256):
        raise DatasetSubsetError("task-pack SHA-256 must be 64 lowercase hex digits")
    if not isinstance(selection_rule, str) or not selection_rule.strip():
        raise DatasetSubsetError("selection rule metadata must be non-empty")
    if isinstance(selection_seed, bool) or (
        selection_seed is not None and not isinstance(selection_seed, int)
    ):
        raise DatasetSubsetError("selection seed metadata must be an integer or null")
    if first_n and selection_seed is not None:
        raise DatasetSubsetError("parent-order first-N mode does not use a seed")

    _validate_paths(
        parent_jsonl,
        ordered_task_ids,
        subset_output,
        manifest_output,
    )
    parent_bytes, parent_ids, rows = _read_parent(parent_jsonl)
    actual_parent_sha256 = _sha256(parent_bytes)
    if actual_parent_sha256 != expected_parent_sha256:
        raise DatasetSubsetError("parent JSONL SHA-256 does not match the attestation")
    if len(parent_ids) != expected_parent_count:
        raise DatasetSubsetError(
            "parent JSONL row count does not match the attestation"
        )

    if ordered_task_ids is not None:
        ordered_id_bytes, selected_ids = _read_ordered_ids(ordered_task_ids)
        mode = "explicit_ordered_task_ids"
        ordered_id_sha_basis = "ordered_task_id_file_bytes"
    else:
        if expected_count > len(parent_ids):
            raise DatasetSubsetError(
                "first-N count exceeds the number of parent JSONL rows"
            )
        selected_ids = parent_ids[:expected_count]
        ordered_id_bytes = "".join(f"{task_id}\n" for task_id in selected_ids).encode(
            "utf-8"
        )
        mode = "parent_order_first_n"
        ordered_id_sha_basis = "generated_utf8_lf_bytes"

    if len(selected_ids) != expected_count:
        raise DatasetSubsetError(
            "ordered task-ID count does not match the exact expected count"
        )
    missing_count = sum(task_id not in rows for task_id in selected_ids)
    if missing_count:
        raise DatasetSubsetError(
            f"ordered task-ID file contains {missing_count} IDs absent from parent"
        )

    subset_bytes = b"".join(rows[task_id] for task_id in selected_ids)
    emitted_ids = []
    for raw_row in subset_bytes.splitlines():
        row = json.loads(raw_row.decode("utf-8"))
        emitted_ids.append(row["metadata"]["task_id"])
    if emitted_ids != selected_ids:
        raise DatasetSubsetError("emitted subset has missing, extra, or reordered IDs")

    manifest: dict[str, Any] = {
        "schema": "secrlenv-dataset-subset/v1",
        "id_field": "metadata.task_id",
        "parent": {
            "row_count": len(parent_ids),
            "sha256": actual_parent_sha256,
        },
        "subset": {
            "row_count": len(selected_ids),
            "sha256": _sha256(subset_bytes),
        },
        "ordered_task_ids_sha256": _sha256(ordered_id_bytes),
        "ordered_task_ids_sha256_basis": ordered_id_sha_basis,
        "task_pack_sha256": task_pack_sha256,
        "selection": {
            "mode": mode,
            "rule": selection_rule,
            "seed": selection_seed,
            "expected_count": expected_count,
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode("utf-8")
    _write_exclusive_artifacts(
        subset_output,
        subset_bytes,
        manifest_output,
        manifest_bytes,
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-jsonl", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ordered-task-ids", type=Path)
    selection.add_argument("--first-n", action="store_true")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--expected-parent-count", type=int, required=True)
    parser.add_argument("--task-pack-sha256", required=True)
    parser.add_argument("--selection-rule", required=True)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_subset(
        args.parent_jsonl,
        args.output,
        args.manifest_output,
        expected_parent_sha256=args.expected_parent_sha256,
        expected_parent_count=args.expected_parent_count,
        expected_count=args.count,
        task_pack_sha256=args.task_pack_sha256,
        selection_rule=args.selection_rule,
        selection_seed=args.selection_seed,
        ordered_task_ids=args.ordered_task_ids,
        first_n=args.first_n,
    )
    print(f"secrlenv_dataset_subset=ready count={manifest['subset']['row_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetSubsetError as exc:
        print(f"secrlenv dataset subset failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
