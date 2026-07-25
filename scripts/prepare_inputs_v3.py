#!/usr/bin/env python3
"""Materialize the preregistered outer-muP v3 (finite-horizon) train order for every fresh v3 seed.

Identical to prepare_inputs_v2.py except: fresh v3 seeds {301,311,313,317,331}
(307 stays reserved) and a v3 combined manifest filename so no v1/v2 input
record is overwritten."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


DEFAULT_SEEDS = (301, 311, 313, 317, 331)


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


def count_nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def model_manifest(model_dir: Path) -> dict[str, object]:
    checksum_path = model_dir / "model-files.sha256"
    expected: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    observed = {}
    for name, digest in sorted(expected.items()):
        path = model_dir / name
        actual = sha256_file(path)
        if actual != digest:
            raise SystemExit(f"model file hash mismatch: {path}: {actual} != {digest}")
        observed[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return {
        "path": str(model_dir),
        "model_id": (model_dir / "model-id.txt").read_text().strip(),
        "revision": (model_dir / "model-revision.txt").read_text().strip(),
        "model_files_manifest_sha256": sha256_file(checksum_path),
        "files": observed,
    }


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
    parser.add_argument(
        "--out", type=Path, default=Path("/root/yeto-data/outer-mup-day1")
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated scientific shuffle seeds",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from scripts.compare_diloco import split_data  # noqa: PLC0415

    source = args.source.resolve()
    model_dir = args.model.resolve()
    reference_eval = args.reference_eval.resolve()
    reference_audit = args.reference_audit.resolve()
    out = args.out.resolve()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if tuple(sorted(set(seeds))) != tuple(sorted(seeds)):
        raise SystemExit("seeds must be unique")
    if not source.is_file() or not reference_eval.is_file() or not reference_audit.is_file():
        raise SystemExit("source/reference input is missing")

    expected_source_sha = "970f88b3f2fa6758f3b5f94052f4e91b872541a2ba530223b44a779168c51409"
    source_sha = sha256_file(source)
    if source_sha != expected_source_sha:
        raise SystemExit(f"source parquet mismatch: {source_sha}")
    reference_eval_sha = sha256_file(reference_eval)
    reference_audit_sha = sha256_file(reference_audit)
    if reference_eval_sha != "533838a0564b13519956a044d23ed8db6705ddc7ae5f0ddb96538f49460bcebc":
        raise SystemExit("reference development file differs from frozen setup")
    if reference_audit_sha != "d71b90040a57731f25c78a2d191017ce90a12c1bb79f55a1cd2f3d085a706d7b":
        raise SystemExit("reference audit file differs from frozen setup")

    out.mkdir(parents=True, exist_ok=True)
    seed_records: dict[str, object] = {}
    for seed in seeds:
        destination = out / f"seed-{seed}"
        existing = destination / "input-manifest.json"
        if destination.exists():
            if not existing.is_file():
                raise SystemExit(f"refusing incomplete existing directory: {destination}")
            record = json.loads(existing.read_text())
            for label in ("train", "eval", "audit", "split_provenance"):
                item = record["files"][label]
                path = Path(item["path"])
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise SystemExit(f"existing seed materialization fails hash check: {path}")
            seed_records[str(seed)] = record
            continue

        temporary_root = Path(
            tempfile.mkdtemp(prefix=f".seed-{seed}-", dir=str(out))
        )
        try:
            train, eval_path, train_count = split_data(
                str(source),
                temporary_root,
                eval_rows=1024,
                max_rows=5000,
                shuffle_seed=seed,
                eval_split_seed=331,
                confirmation_audit_rows=1024,
            )
            audit_path = temporary_root / "confirmation-audit.jsonl"
            provenance_path = temporary_root / "split_provenance.json"
            provenance = json.loads(provenance_path.read_text())
            if train_count != 5000 or count_nonempty_lines(train) != 5000:
                raise SystemExit(f"seed {seed}: train row count mismatch")
            if count_nonempty_lines(eval_path) != 1024 or count_nonempty_lines(audit_path) != 1024:
                raise SystemExit(f"seed {seed}: development/audit row count mismatch")
            if sha256_file(eval_path) != reference_eval_sha:
                raise SystemExit(f"seed {seed}: development split is not invariant")
            if sha256_file(audit_path) != reference_audit_sha:
                raise SystemExit(f"seed {seed}: audit split is not invariant")
            if provenance.get("train_shuffle_seed") != seed:
                raise SystemExit(f"seed {seed}: shuffle seed not recorded")
            if provenance.get("eval_split_seed") != 331:
                raise SystemExit(f"seed {seed}: eval seed not recorded")
            if len(provenance.get("train_pool_source_indices", [])) != 5000:
                raise SystemExit(f"seed {seed}: train pool length mismatch")

            temporary_destination = out / f".publish-seed-{seed}-{os.getpid()}"
            temporary_root.replace(temporary_destination)
            destination_record = {
                "schema": "yeto_outer_mup_seed_input_v1",
                "seed": seed,
                "training_seed": int(f"{seed}{seed}"),
                "split_rule": {
                    "source_prefix_rows": 7048,
                    "eval_split_seed": 331,
                    "train_rows": 5000,
                    "development_rows": 1024,
                    "reserved_audit_rows": 1024,
                    "train_shuffle_seed": seed,
                },
                "source_parquet_sha256": source_sha,
                "files": {},
                "source_index_hashes": {
                    "train_pool": canonical_sha256(provenance["train_pool_source_indices"]),
                    "train_order": canonical_sha256(provenance["train_source_indices"]),
                    "development": canonical_sha256(provenance["eval_source_indices"]),
                    "audit": canonical_sha256(provenance["audit_eval_source_indices"]),
                },
            }
            path_map = {
                "train": temporary_destination / "train.jsonl",
                "eval": temporary_destination / "eval.jsonl",
                "audit": temporary_destination / "confirmation-audit.jsonl",
                "split_provenance": temporary_destination / "split_provenance.json",
            }
            for label, path in path_map.items():
                destination_record["files"][label] = {
                    "path": str(destination / path.name),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            write_json_atomic(
                temporary_destination / "input-manifest.json", destination_record
            )
            temporary_destination.replace(destination)
            seed_records[str(seed)] = destination_record
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)

    combined = {
        "schema": "yeto_outer_mup_day1_inputs_v1",
        "source": {
            "path": str(source),
            "sha256": source_sha,
            "bytes": source.stat().st_size,
            "dataset": "trl-lib/Capybara",
            "revision": "e235e846458bff3398a88aed812347f7f0756520",
        },
        "reference_development": {
            "path": str(reference_eval),
            "sha256": reference_eval_sha,
            "rows": 1024,
        },
        "reference_audit": {
            "path": str(reference_audit),
            "sha256": reference_audit_sha,
            "rows": 1024,
        },
        "model": model_manifest(model_dir),
        "seeds": seed_records,
    }
    write_json_atomic(out / "input-manifest-v3.json", combined)
    print(json.dumps({
        "manifest": str(out / "input-manifest-v3.json"),
        "sha256": sha256_file(out / "input-manifest-v3.json"),
        "seeds": list(seeds),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
