#!/usr/bin/env python3
"""Build a deterministic held-out JSONL split disjoint from existing examples.

The source is scanned in its native order beginning at ``--search-start``.
Rows are canonicalized to the conversation fields consumed by Yeto, then
excluded by exact canonical content. No model output or random ordering is
used during selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from yeto.data import load_rows  # noqa: E402


CANONICALIZATION = "yeto-messages-tools-v1"


class HoldoutError(RuntimeError):
    """Raised when a deterministic, disjoint holdout cannot be built."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_json(value, *, context: str):
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise HoldoutError(f"{context}: value is not finite JSON data: {exc}") from exc


def canonical_example(row, *, context: str = "row") -> dict:
    """Return the model-consumed, JSON-stable representation of one row.

    Training materialization keeps ``messages`` and optional ``tools`` while
    discarding source metadata. Missing, null, and empty tools are equivalent
    in Yeto's conversation rendering, so the canonical form omits them.
    """

    if not isinstance(row, Mapping):
        try:
            row = dict(row)
        except (TypeError, ValueError) as exc:
            raise HoldoutError(f"{context}: expected an object row") from exc
    plain = _plain_json(dict(row), context=context)
    messages = plain.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HoldoutError(f"{context}: expected a non-empty messages list")
    canonical = {"messages": messages}
    tools = plain.get("tools")
    if tools:
        canonical["tools"] = tools
    return canonical


def _canonical_payload(row: dict) -> bytes:
    return json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(row: dict) -> str:
    return _sha256_bytes(_canonical_payload(row))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise HoldoutError(f"missing JSONL file: {path}")
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HoldoutError(
                        f"{path}:{line_number}: malformed JSON: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise HoldoutError(
                        f"{path}:{line_number}: expected a JSON object"
                    )
                rows.append(row)
    except OSError as exc:
        raise HoldoutError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise HoldoutError(f"{path}: no JSONL rows")
    return rows


def _load_source(data: str):
    """Use Yeto's dataset loader, with a dependency-light JSONL fast path."""

    local = Path(os.path.expanduser(data))
    if local.is_file() and local.suffix.lower() == ".jsonl":
        return _read_jsonl(local)
    try:
        return load_rows(data)
    except ModuleNotFoundError as exc:
        raise HoldoutError(
            f"loading {data!r} requires the repository dataset dependencies"
        ) from exc
    except Exception as exc:
        raise HoldoutError(f"cannot load source {data!r}: {exc}") from exc


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise HoldoutError(f"cannot write {path}: {exc}") from exc


def build_holdout(
    *,
    data: str,
    exclude_jsonl: Sequence[Path],
    search_start: int,
    rows: int,
    out_jsonl: Path,
    manifest_out: Path,
) -> dict:
    if search_start < 0:
        raise HoldoutError(f"search_start must be >= 0, got {search_start}")
    if rows <= 0:
        raise HoldoutError(f"rows must be > 0, got {rows}")

    out_jsonl = _resolved(out_jsonl)
    manifest_out = _resolved(manifest_out)
    if out_jsonl == manifest_out:
        raise HoldoutError("out_jsonl and manifest_out must be different files")

    exclusion_paths = [_resolved(Path(path)) for path in exclude_jsonl]
    protected_paths = set(exclusion_paths)
    local_source = _resolved(Path(data)) if Path(os.path.expanduser(data)).exists() else None
    if local_source is not None:
        protected_paths.add(local_source)
    if out_jsonl in protected_paths or manifest_out in protected_paths:
        raise HoldoutError("output paths must not overwrite the source or an exclusion JSONL")

    exclusion_hashes: set[str] = set()
    exclusion_records: list[dict] = []
    for path in exclusion_paths:
        file_rows = _read_jsonl(path)
        file_hashes: list[str] = []
        canonical_payloads: list[bytes] = []
        for row_number, row in enumerate(file_rows, 1):
            canonical = canonical_example(row, context=f"{path}:{row_number}")
            canonical_payloads.append(_canonical_payload(canonical))
            file_hashes.append(canonical_hash(canonical))
        unique_file_hashes = set(file_hashes)
        exclusion_hashes.update(unique_file_hashes)
        exclusion_records.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "canonical_sha256": _sha256_bytes(
                    b"".join(payload + b"\n" for payload in canonical_payloads)
                ),
                "row_count": len(file_rows),
                "unique_canonical_count": len(unique_file_hashes),
                "duplicate_canonical_count": len(file_rows) - len(unique_file_hashes),
            }
        )

    source = _load_source(data)
    try:
        source_rows = len(source)
    except TypeError as exc:
        raise HoldoutError(f"source {data!r} does not provide a finite row count") from exc
    if search_start >= source_rows:
        raise HoldoutError(
            f"search_start {search_start} is outside source with {source_rows} rows"
        )

    selected_rows: list[dict] = []
    selected_indices: list[int] = []
    selected_hashes: set[str] = set()
    selected_hashes_in_order: list[str] = []
    excluded_count = 0
    duplicate_count = 0
    last_scanned = search_start - 1

    for source_index in range(search_start, source_rows):
        last_scanned = source_index
        canonical = canonical_example(
            source[source_index], context=f"{data}[{source_index}]"
        )
        row_hash = canonical_hash(canonical)
        if row_hash in exclusion_hashes:
            excluded_count += 1
            continue
        if row_hash in selected_hashes:
            duplicate_count += 1
            continue
        selected_rows.append(canonical)
        selected_indices.append(source_index)
        selected_hashes.add(row_hash)
        selected_hashes_in_order.append(row_hash)
        if len(selected_rows) == rows:
            break

    if len(selected_rows) != rows:
        raise HoldoutError(
            f"source exhausted after index {last_scanned}: selected "
            f"{len(selected_rows)} of {rows} requested rows "
            f"({excluded_count} excluded, {duplicate_count} duplicate)"
        )

    overlap = selected_hashes & exclusion_hashes
    if overlap:
        raise HoldoutError(
            f"internal verification failed: {len(overlap)} selected examples overlap exclusions"
        )
    if len(selected_hashes) != len(selected_rows):
        raise HoldoutError("internal verification failed: selected examples are not unique")

    output_payload = b"".join(_canonical_payload(row) + b"\n" for row in selected_rows)
    output_hash = _sha256_bytes(output_payload)
    _write_atomic(out_jsonl, output_payload)
    written_rows = _read_jsonl(out_jsonl)
    written_hashes = {
        canonical_hash(canonical_example(row, context=f"{out_jsonl}:{index}"))
        for index, row in enumerate(written_rows, 1)
    }
    written_overlap = written_hashes & exclusion_hashes
    if len(written_rows) != rows or len(written_hashes) != rows or written_overlap:
        raise HoldoutError(
            "written-output verification failed: "
            f"rows={len(written_rows)}, unique={len(written_hashes)}, "
            f"overlap={len(written_overlap)}"
        )
    if _sha256_file(out_jsonl) != output_hash:
        raise HoldoutError("written-output verification failed: output hash changed")

    source_fingerprint = getattr(source, "_fingerprint", None)
    manifest = {
        "schema": "disjoint_hf_holdout_v1",
        "canonicalization": CANONICALIZATION,
        "source": data,
        "source_resolved_path": str(local_source) if local_source is not None else None,
        "source_fingerprint": str(source_fingerprint) if source_fingerprint else None,
        "source_rows": source_rows,
        "search_start": search_start,
        "search_stop_exclusive": last_scanned + 1,
        "scanned_count": last_scanned - search_start + 1,
        "requested_rows": rows,
        "selected_count": len(selected_rows),
        "selected_source_indices": selected_indices,
        "selected_canonical_hashes": selected_hashes_in_order,
        "excluded_count": excluded_count,
        "duplicate_count": duplicate_count,
        "unique_excluded_canonical_count": len(exclusion_hashes),
        "exclusion_paths": [record["path"] for record in exclusion_records],
        "exclusion_hashes": [record["sha256"] for record in exclusion_records],
        "exclusions": exclusion_records,
        "output_path": str(out_jsonl),
        "output_sha256": output_hash,
        "overlap_count": 0,
        "verified_zero_overlap": True,
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_atomic(manifest_out, manifest_payload)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="HF dataset id or local JSONL")
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        default=[],
        type=Path,
        help="JSONL whose canonical examples must be excluded; repeat as needed",
    )
    parser.add_argument("--search-start", required=True, type=int)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--out-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_holdout(
            data=args.data,
            exclude_jsonl=args.exclude_jsonl,
            search_start=args.search_start,
            rows=args.rows,
            out_jsonl=args.out_jsonl,
            manifest_out=args.manifest_out,
        )
    except HoldoutError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        "selected {selected_count} rows from {source!r} at indices "
        "{selected_source_indices}; excluded={excluded_count}, "
        "duplicates={duplicate_count}, output_sha256={output_sha256}".format(**manifest)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
