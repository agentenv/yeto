#!/usr/bin/env python3
"""Build the frozen, no-wrap training bundles for outer-muP v8.

The source partition and fixed development/audit sets are byte-identical to
v3.  Only the prospectively registered v8 training shuffle seeds differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path


V8_SEEDS = (801, 809, 811, 821, 823)
SOURCE_SHA256 = "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409"
EVAL_SHA256 = "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc"
AUDIT_SHA256 = "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b"
SOURCE_ROWS = 15_806
LEGACY_POPULATION_ROWS = 7_048
LEGACY_TRAIN_ROWS = 5_000
EVAL_ROWS = 1_024
AUDIT_ROWS = 1_024
EVAL_SPLIT_SEED = 331


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def dump_rows(path: Path, dataset, indices: list[int]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for index in indices:
            row = dataset[index]
            payload = {key: row[key] for key in ("messages", "tools") if key in row}
            destination.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/yeto"))
    parser.add_argument(
        "--source", type=Path, default=Path("/root/yeto-data/train.parquet")
    )
    parser.add_argument(
        "--reference-eval",
        type=Path,
        default=Path("/root/yeto-data/splits/seed-337/eval.jsonl"),
    )
    parser.add_argument(
        "--reference-audit",
        type=Path,
        default=Path("/root/yeto-data/splits/seed-337/confirmation-audit.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("/root/yeto-data/outer-mup-v8")
    )
    args = parser.parse_args()

    source = args.source.resolve()
    reference_eval = args.reference_eval.resolve()
    reference_audit = args.reference_audit.resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing existing output: {out}")
    if sha256_file(source) != SOURCE_SHA256:
        raise SystemExit("source parquet differs from the frozen Day-1 source")
    if sha256_file(reference_eval) != EVAL_SHA256:
        raise SystemExit("frozen development input hash mismatch")
    if sha256_file(reference_audit) != AUDIT_SHA256:
        raise SystemExit("frozen audit input hash mismatch")

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    from yeto.data import load_rows

    dataset = load_rows(str(source))
    if len(dataset) != SOURCE_ROWS:
        raise SystemExit(f"source row count changed: {len(dataset)} != {SOURCE_ROWS}")

    legacy_order = list(range(LEGACY_POPULATION_ROWS))
    random.Random(EVAL_SPLIT_SEED).shuffle(legacy_order)
    legacy_train_pool = legacy_order[:LEGACY_TRAIN_ROWS]
    eval_indices = legacy_order[
        LEGACY_TRAIN_ROWS : LEGACY_TRAIN_ROWS + EVAL_ROWS
    ]
    audit_indices = legacy_order[LEGACY_TRAIN_ROWS + EVAL_ROWS :]
    train_pool = legacy_train_pool + list(range(LEGACY_POPULATION_ROWS, SOURCE_ROWS))
    if len(train_pool) != SOURCE_ROWS - EVAL_ROWS - AUDIT_ROWS:
        raise SystemExit("expanded training-pool length mismatch")
    if set(train_pool) & (set(eval_indices) | set(audit_indices)):
        raise SystemExit("training pool overlaps frozen evaluation rows")
    if set(train_pool) | set(eval_indices) | set(audit_indices) != set(range(SOURCE_ROWS)):
        raise SystemExit("source partition is not exhaustive")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".outer-mup-v8-", dir=out.parent))
    try:
        seed_records: dict[str, object] = {}
        for seed in V8_SEEDS:
            destination = temporary_root / f"seed-{seed}"
            destination.mkdir()
            train_indices = list(train_pool)
            random.Random(seed).shuffle(train_indices)
            train_path = destination / "train.jsonl"
            dump_rows(train_path, dataset, train_indices)
            shutil.copy2(reference_eval, destination / "eval.jsonl")
            shutil.copy2(reference_audit, destination / "confirmation-audit.jsonl")
            provenance = {
                "schema": "yeto_split_provenance_v8",
                "source_row_count": SOURCE_ROWS,
                "source_population_count": SOURCE_ROWS,
                "train_row_count": len(train_indices),
                "eval_row_count": EVAL_ROWS,
                "confirmation_audit_row_count": AUDIT_ROWS,
                "eval_split_seed": EVAL_SPLIT_SEED,
                "train_shuffle_seed": seed,
                "expanded_train_pool_rule": (
                    "day1_train_pool_then_canonical_source_tail_7048_15806"
                ),
                "train_pool_source_indices": train_pool,
                "train_source_indices": train_indices,
                "eval_source_indices": eval_indices,
                "audit_eval_source_indices": audit_indices,
            }
            write_json_atomic(destination / "split_provenance.json", provenance)
            if count_nonempty_lines(train_path) != len(train_pool):
                raise SystemExit(f"seed {seed}: train row count mismatch")

            final_root = out / f"seed-{seed}"
            final_paths = {
                "train": final_root / "train.jsonl",
                "eval": final_root / "eval.jsonl",
                "audit": final_root / "confirmation-audit.jsonl",
                "split_provenance": final_root / "split_provenance.json",
            }
            temp_paths = {
                "train": destination / "train.jsonl",
                "eval": destination / "eval.jsonl",
                "audit": destination / "confirmation-audit.jsonl",
                "split_provenance": destination / "split_provenance.json",
            }
            record = {
                "schema": "yeto_outer_mup_seed_input_v8",
                "seed": seed,
                "training_seed": int(f"{seed}{seed}"),
                "split_rule": {
                    "source_rows": SOURCE_ROWS,
                    "eval_split_seed": EVAL_SPLIT_SEED,
                    "train_rows": len(train_pool),
                    "development_rows": EVAL_ROWS,
                    "reserved_audit_rows": AUDIT_ROWS,
                    "train_shuffle_seed": seed,
                },
                "source_parquet_sha256": SOURCE_SHA256,
                "files": {
                    label: {
                        "path": str(final_paths[label]),
                        "sha256": sha256_file(temp_paths[label]),
                        "bytes": temp_paths[label].stat().st_size,
                    }
                    for label in final_paths
                },
                "source_index_hashes": {
                    "train_pool": canonical_sha256(train_pool),
                    "train_order": canonical_sha256(train_indices),
                    "development": canonical_sha256(eval_indices),
                    "audit": canonical_sha256(audit_indices),
                },
            }
            write_json_atomic(destination / "input-manifest.json", record)
            seed_records[str(seed)] = record

        combined = {
            "schema": "yeto_outer_mup_v8_inputs_v1",
            "source": {
                "path": str(source),
                "sha256": SOURCE_SHA256,
                "bytes": source.stat().st_size,
                "rows": SOURCE_ROWS,
                "dataset": "trl-lib/Capybara",
                "revision": "e235e846458bff3398a88aed812347f7f0756520",
            },
            "reference_development": {
                "path": str(reference_eval),
                "sha256": EVAL_SHA256,
                "rows": EVAL_ROWS,
            },
            "reference_audit": {
                "path": str(reference_audit),
                "sha256": AUDIT_SHA256,
                "rows": AUDIT_ROWS,
            },
            "training_pool": {
                "rows": len(train_pool),
                "source_index_sha256": canonical_sha256(train_pool),
                "rule": "frozen_day1_train_pool_plus_all_unused_source_rows",
            },
            "seeds": seed_records,
        }
        write_json_atomic(temporary_root / "input-manifest.json", combined)
        temporary_root.replace(out)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)

    print(
        json.dumps(
            {
                "manifest": str(out / "input-manifest.json"),
                "sha256": sha256_file(out / "input-manifest.json"),
                "seeds": list(V8_SEEDS),
                "train_rows_per_seed": len(train_pool),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
