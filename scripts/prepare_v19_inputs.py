#!/usr/bin/env python3
"""Materialize the frozen V19 no-wrap Capybara inputs for three fresh seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path


V19_SEEDS = (1201, 1213, 1217)
SOURCE_SHA256 = "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409"
EVAL_SHA256 = "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc"
AUDIT_SHA256 = "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SOURCE_ROWS = 15_806
LEGACY_POPULATION_ROWS = 7_048
LEGACY_TRAIN_ROWS = 5_000
EVAL_ROWS = 1_024
AUDIT_ROWS = 1_024
TRAIN_ROWS = 13_758
EVAL_SPLIT_SEED = 331


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
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


def model_manifest(model_dir: Path) -> dict[str, object]:
    checksum_path = model_dir / "model-files.sha256"
    revision_path = model_dir / "model-revision.txt"
    model_id_path = model_dir / "model-id.txt"
    if not checksum_path.is_file() or not revision_path.is_file():
        raise SystemExit(f"model provenance is incomplete: {model_dir}")
    if revision_path.read_text().strip() != MODEL_REVISION:
        raise SystemExit("SmolLM2-135M revision differs from the V19 registration")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.strip()] = digest
    observed: dict[str, object] = {}
    for name, digest in sorted(expected.items()):
        path = model_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise SystemExit(f"model file hash mismatch: {path}")
        observed[name] = {"bytes": path.stat().st_size, "sha256": digest}
    return {
        "path": str(model_dir),
        "model_id": (
            model_id_path.read_text().strip()
            if model_id_path.is_file()
            else "HuggingFaceTB/SmolLM2-135M"
        ),
        "revision": MODEL_REVISION,
        "model_files_manifest_sha256": sha256_file(checksum_path),
        "files": observed,
    }


def validate_existing(destination: Path, seed: int) -> dict[str, object]:
    manifest_path = destination / "input-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"refusing incomplete existing directory: {destination}")
    record = json.loads(manifest_path.read_text())
    if record.get("seed") != seed or record.get("training_seed") != int(f"{seed}{seed}"):
        raise SystemExit(f"existing V19 seed identity mismatch: {destination}")
    for label in ("train", "eval", "audit", "split_provenance"):
        item = record["files"][label]
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise SystemExit(f"existing V19 input hash mismatch: {path}")
    if count_nonempty_lines(Path(record["files"]["train"]["path"])) != TRAIN_ROWS:
        raise SystemExit(f"existing V19 train row count mismatch: {destination}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/yeto"))
    parser.add_argument("--source", type=Path, default=Path("/root/yeto-data/train.parquet"))
    parser.add_argument("--model", type=Path, default=Path("/root/yeto-data/model"))
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
    parser.add_argument("--out", type=Path, default=Path("/root/yeto-data/outer-mup-v19"))
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in V19_SEEDS),
        help="must be the complete registered V19 seed set",
    )
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if seeds != V19_SEEDS:
        raise SystemExit(f"V19 seed set/order must be exactly {V19_SEEDS}")
    source = args.source.resolve()
    reference_eval = args.reference_eval.resolve()
    reference_audit = args.reference_audit.resolve()
    model_dir = args.model.resolve()
    output = args.out.resolve()
    if sha256_file(source) != SOURCE_SHA256:
        raise SystemExit("source parquet differs from the V19 registration")
    if sha256_file(reference_eval) != EVAL_SHA256:
        raise SystemExit("frozen development input hash mismatch")
    if sha256_file(reference_audit) != AUDIT_SHA256:
        raise SystemExit("frozen confirmation-audit input hash mismatch")

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    from yeto.data import load_rows  # noqa: PLC0415

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
    if len(train_pool) != TRAIN_ROWS:
        raise SystemExit("expanded V19 training-pool length mismatch")
    if set(train_pool) & (set(eval_indices) | set(audit_indices)):
        raise SystemExit("V19 training pool overlaps a held-out stream")
    if set(train_pool) | set(eval_indices) | set(audit_indices) != set(range(SOURCE_ROWS)):
        raise SystemExit("V19 source partition is not exhaustive")

    output.mkdir(parents=True, exist_ok=True)
    seed_records: dict[str, object] = {}
    for seed in seeds:
        destination = output / f"seed-{seed}"
        if destination.exists():
            seed_records[str(seed)] = validate_existing(destination, seed)
            continue
        temporary = Path(tempfile.mkdtemp(prefix=f".seed-{seed}-", dir=output))
        try:
            train_indices = list(train_pool)
            random.Random(seed).shuffle(train_indices)
            dump_rows(temporary / "train.jsonl", dataset, train_indices)
            shutil.copy2(reference_eval, temporary / "eval.jsonl")
            shutil.copy2(reference_audit, temporary / "confirmation-audit.jsonl")
            provenance = {
                "schema": "yeto_split_provenance_v19",
                "source_row_count": SOURCE_ROWS,
                "train_row_count": TRAIN_ROWS,
                "eval_row_count": EVAL_ROWS,
                "confirmation_audit_row_count": AUDIT_ROWS,
                "eval_split_seed": EVAL_SPLIT_SEED,
                "train_shuffle_seed": seed,
                "expanded_train_pool_rule": "day1_train_pool_then_canonical_source_tail_7048_15806",
                "train_pool_source_indices": train_pool,
                "train_source_indices": train_indices,
                "eval_source_indices": eval_indices,
                "audit_eval_source_indices": audit_indices,
            }
            write_json_atomic(temporary / "split_provenance.json", provenance)
            if count_nonempty_lines(temporary / "train.jsonl") != TRAIN_ROWS:
                raise SystemExit(f"seed {seed}: train row count mismatch")
            if sha256_file(temporary / "eval.jsonl") != EVAL_SHA256:
                raise SystemExit(f"seed {seed}: development hash changed")
            if sha256_file(temporary / "confirmation-audit.jsonl") != AUDIT_SHA256:
                raise SystemExit(f"seed {seed}: audit hash changed")
            final_paths = {
                "train": destination / "train.jsonl",
                "eval": destination / "eval.jsonl",
                "audit": destination / "confirmation-audit.jsonl",
                "split_provenance": destination / "split_provenance.json",
            }
            temporary_paths = {
                "train": temporary / "train.jsonl",
                "eval": temporary / "eval.jsonl",
                "audit": temporary / "confirmation-audit.jsonl",
                "split_provenance": temporary / "split_provenance.json",
            }
            record = {
                "schema": "yeto_outer_mup_seed_input_v19",
                "seed": seed,
                "training_seed": int(f"{seed}{seed}"),
                "split_rule": {
                    "source_rows": SOURCE_ROWS,
                    "eval_split_seed": EVAL_SPLIT_SEED,
                    "train_rows": TRAIN_ROWS,
                    "development_rows": EVAL_ROWS,
                    "reserved_audit_rows": AUDIT_ROWS,
                    "train_shuffle_seed": seed,
                },
                "source_parquet_sha256": SOURCE_SHA256,
                "files": {
                    label: {
                        "path": str(final_paths[label]),
                        "sha256": sha256_file(temporary_paths[label]),
                        "bytes": temporary_paths[label].stat().st_size,
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
            write_json_atomic(temporary / "input-manifest.json", record)
            temporary.replace(destination)
            seed_records[str(seed)] = record
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    combined = {
        "schema": "yeto_outer_mup_v19_inputs_v1",
        "source": {
            "path": str(source),
            "sha256": SOURCE_SHA256,
            "bytes": source.stat().st_size,
            "rows": SOURCE_ROWS,
            "dataset": "trl-lib/Capybara",
            "revision": "e235e846458bff3398a88aed812347f7f0756520",
        },
        "reference_development": {"path": str(reference_eval), "sha256": EVAL_SHA256, "rows": EVAL_ROWS},
        "reference_audit": {"path": str(reference_audit), "sha256": AUDIT_SHA256, "rows": AUDIT_ROWS},
        "training_pool": {
            "rows": TRAIN_ROWS,
            "source_index_sha256": canonical_sha256(train_pool),
            "rule": "frozen_day1_train_pool_plus_all_unused_source_rows",
        },
        "model": model_manifest(model_dir),
        "seeds": seed_records,
    }
    manifest_path = output / "input-manifest.json"
    write_json_atomic(manifest_path, combined)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "seeds": list(seeds),
                "train_rows_per_seed": TRAIN_ROWS,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
