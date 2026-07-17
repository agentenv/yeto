#!/usr/bin/env python3
"""Build current-stage audit train bundles without opening the audit surface.

The input is the sealed, sanitized seed-347 development bundle from P1.  It
contains the frozen 5,000-row train pool, the locked development evaluation
surface, and no audit-evaluation object.  For exactly the seeds registered for
one requested audit sub-stage, this script reconstructs ``source_index -> row``,
applies ``random.Random(seed).shuffle`` to the frozen train-pool indices, and
emits a deterministic multi-seed tar plus the registry consumed by
``audit_135m_contract.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import tarfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from scripts import audit_135m_contract as audit


BASE_SEED = 347
EXPECTED_BASE_TAR_SHA256 = (
    "c60c594665c4ded60e266ed2e51afee542423a7559e43d88e2adc334561b8a25"
)
EXPECTED_BASE_BUILD_SHA256 = (
    "2ba54f4c6f949e9f81ed5b753750dfb3121f87230f8effe638cc05581a86c9d8"
)

REQUIRED_MEMBERS = (
    "materialized/train.jsonl",
    "materialized/eval.jsonl",
    "materialized/split_provenance.json",
    "eval-freeze.json",
    "parallel-eval-freeze.json",
    "provenance/eval_packed.jsonl",
    "provenance/eval_provenance.json",
    "provenance/eval_rows.jsonl",
    "provenance/eval_sequences.jsonl",
)


class SeedBundleError(RuntimeError):
    """The source bundle or requested seed suffix violates the frozen contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _read_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SeedBundleError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SeedBundleError(f"{label} must be a JSON object")
    return value


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not name.startswith("./")
        and not name.endswith("/")
    )


def _load_source_members(tar_path: Path) -> dict[str, bytes]:
    prefix = f"seed-{BASE_SEED}/"
    wanted = {prefix + suffix for suffix in REQUIRED_MEMBERS}
    values: dict[str, bytes] = {}
    try:
        with tarfile.open(tar_path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isdir():
                    continue
                if not member.isfile() or not _safe_member(member.name):
                    raise SeedBundleError(
                        f"source tar contains an unsafe non-regular member: {member.name}"
                    )
                if "audit" in PurePosixPath(member.name).name.casefold():
                    raise SeedBundleError(
                        f"source tar unexpectedly names an audit object: {member.name}"
                    )
                if member.name in wanted:
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise SeedBundleError(f"cannot read source member {member.name}")
                    values[member.name.removeprefix(prefix)] = handle.read()
    except (OSError, tarfile.TarError) as exc:
        raise SeedBundleError(f"cannot read sanitized source tar: {exc}") from exc
    missing = set(REQUIRED_MEMBERS) - set(values)
    if missing:
        raise SeedBundleError(f"sanitized source tar lacks members: {sorted(missing)}")
    return values


def _validate_source(
    tar_path: Path, build_path: Path, members: Mapping[str, bytes]
) -> dict[str, Any]:
    if sha256_file(tar_path) != EXPECTED_BASE_TAR_SHA256:
        raise SeedBundleError("sanitized seed-347 tar SHA-256 differs")
    if sha256_file(build_path) != EXPECTED_BASE_BUILD_SHA256:
        raise SeedBundleError("sanitized seed-347 build SHA-256 differs")
    build = _read_object(build_path.read_bytes(), "sanitized build manifest")
    if (
        build.get("schema") != "yeto_p1r0_sanitized_development_bundle_v1"
        or build.get("seed") != BASE_SEED
        or build.get("audit_objects_included") is not False
        or build.get("audit_model_evaluation_accesses") != []
        or build.get("tar_sha256") != EXPECTED_BASE_TAR_SHA256
    ):
        raise SeedBundleError("sanitized build manifest audit/source contract differs")
    split = _read_object(
        members["materialized/split_provenance.json"], "source split provenance"
    )
    if (
        split.get("schema") != "yeto_split_provenance_v1"
        or split.get("train_shuffle_seed") != BASE_SEED
        or split.get("train_row_count") != audit.TRAIN_ROWS
        or split.get("eval_row_count") != audit.DEVELOPMENT_EVAL_ROWS
    ):
        raise SeedBundleError("sanitized split provenance differs")
    train_lines = members["materialized/train.jsonl"].splitlines(keepends=True)
    source_indices = split.get("train_source_indices")
    pool_indices = split.get("train_pool_source_indices")
    if (
        not isinstance(source_indices, list)
        or not isinstance(pool_indices, list)
        or len(source_indices) != audit.TRAIN_ROWS
        or len(pool_indices) != audit.TRAIN_ROWS
        or len(train_lines) != audit.TRAIN_ROWS
        or len(set(source_indices)) != audit.TRAIN_ROWS
        or set(source_indices) != set(pool_indices)
        or any(not line.endswith(b"\n") for line in train_lines)
    ):
        raise SeedBundleError("sanitized train rows/source indices are not one exact pool")
    replay = list(pool_indices)
    random.Random(BASE_SEED).shuffle(replay)
    if replay != source_indices:
        raise SeedBundleError("seed-347 shuffle does not reproduce the sealed train order")
    if sha256_bytes(members["materialized/train.jsonl"]) != build.get(
        "train_rows_sha256"
    ):
        raise SeedBundleError("seed-347 train bytes differ from their sealed hash")
    if sha256_bytes(canonical_json(source_indices)) != build.get(
        "train_source_indices_sha256"
    ):
        raise SeedBundleError("seed-347 train-index order differs from its sealed hash")
    return build


def _seed_files(
    *, seed: int, source_members: Mapping[str, bytes]
) -> tuple[dict[str, bytes], dict[str, Any]]:
    split = _read_object(
        source_members["materialized/split_provenance.json"],
        "source split provenance",
    )
    source_indices = list(split["train_source_indices"])
    train_lines = source_members["materialized/train.jsonl"].splitlines(keepends=True)
    rows_by_source = dict(zip(source_indices, train_lines, strict=True))
    shuffled = list(split["train_pool_source_indices"])
    random.Random(seed).shuffle(shuffled)
    if len(set(shuffled)) != audit.TRAIN_ROWS or set(shuffled) != set(rows_by_source):
        raise SeedBundleError(f"seed {seed} shuffle changed the frozen train pool")
    train_bytes = b"".join(rows_by_source[index] for index in shuffled)
    train_rows_hash = sha256_bytes(train_bytes)
    train_indices_hash = sha256_bytes(canonical_json(shuffled))

    new_split = deepcopy(split)
    new_split["train_shuffle_seed"] = seed
    new_split["train_source_indices"] = shuffled
    split_bytes = pretty_json(new_split)

    eval_freeze = _read_object(source_members["eval-freeze.json"], "eval freeze")
    eval_freeze.update(
        {
            "seed": seed,
            "train_shuffle_seed": seed,
            "train_file_sha256": train_rows_hash,
            "train_source_indices_hash": train_indices_hash,
            "split_provenance_sha256": sha256_bytes(canonical_json(new_split)),
            "audit_model_evaluation_accesses": [],
            "audit_outcome_fields_emitted": False,
        }
    )

    parallel_eval = _read_object(
        source_members["parallel-eval-freeze.json"], "parallel eval freeze"
    )
    parallel_eval["seed"] = seed
    parallel_eval_bytes = pretty_json(parallel_eval)

    eval_provenance = _read_object(
        source_members["provenance/eval_provenance.json"], "eval provenance"
    )
    eval_provenance.update(
        {
            "train_shuffle_seed": seed,
            "train_source_indices_hash": train_indices_hash,
            "split_provenance_sha256": sha256_bytes(canonical_json(new_split)),
        }
    )

    files = {
        "materialized/train.jsonl": train_bytes,
        "materialized/eval.jsonl": source_members["materialized/eval.jsonl"],
        "materialized/split_provenance.json": split_bytes,
        "eval-freeze.json": pretty_json(eval_freeze),
        "parallel-eval-freeze.json": parallel_eval_bytes,
        "provenance/eval_packed.jsonl": source_members[
            "provenance/eval_packed.jsonl"
        ],
        "provenance/eval_provenance.json": pretty_json(eval_provenance),
        "provenance/eval_rows.jsonl": source_members["provenance/eval_rows.jsonl"],
        "provenance/eval_sequences.jsonl": source_members[
            "provenance/eval_sequences.jsonl"
        ],
    }
    registry_entry = {
        "train_rows_sha256": train_rows_hash,
        "train_source_indices_sha256": train_indices_hash,
        "train_pool_source_indices_sha256": sha256_bytes(
            canonical_json(new_split["train_pool_source_indices"])
        ),
        "development_eval_rows_sha256": sha256_bytes(
            files["materialized/eval.jsonl"]
        ),
        "parallel_eval_freeze_sha256": sha256_bytes(parallel_eval_bytes),
        "split_provenance_sha256": sha256_bytes(split_bytes),
    }
    return files, registry_entry


def _write_deterministic_tar(path: Path, files: Mapping[str, bytes]) -> None:
    if path.exists():
        raise SeedBundleError(f"refusing to overwrite output tar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                directories: set[str] = set()
                for name in files:
                    parent = PurePosixPath(name).parent
                    while str(parent) not in ("", "."):
                        directories.add(str(parent) + "/")
                        parent = parent.parent
                for directory in sorted(directories, key=lambda value: value.encode("utf-8")):
                    info = tarfile.TarInfo(directory)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    archive.addfile(info)
                for name in sorted(files, key=lambda value: value.encode("utf-8")):
                    payload = files[name]
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(payload))


def build(
    *,
    stage_code: str,
    base_tar: Path,
    base_build: Path,
    output_dir: Path,
) -> dict[str, Any]:
    audit.load_authority()
    if stage_code not in audit.STAGE_CODES:
        raise SeedBundleError(f"unsupported audit stage code: {stage_code}")
    if output_dir.exists():
        raise SeedBundleError(f"refusing to reuse output directory: {output_dir}")
    members = _load_source_members(base_tar)
    _validate_source(base_tar, base_build, members)
    seeds = tuple(seed for seed, _training in audit.stage_seed_pairs(stage_code))
    if not seeds or len(set(seeds)) != len(seeds):
        raise SeedBundleError("registered stage seed suffix is empty or duplicated")

    all_files: dict[str, bytes] = {}
    registry_entries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_files, entry = _seed_files(seed=seed, source_members=members)
        prefix = f"seed-{seed}/"
        all_files.update({prefix + name: payload for name, payload in seed_files.items()})
        registry_entries[str(seed)] = entry

    output_dir.mkdir(parents=True)
    tar_path = output_dir / f"audit-135m-{stage_code}-seeds.tar.gz"
    _write_deterministic_tar(tar_path, all_files)
    registry = {
        "schema": "audit_135m_seed_bundle_registry_v1",
        "stage_code": stage_code,
        "authority_prereg_sha256": audit.PREREG_JSON_SHA256,
        "source_sanitized_tar_sha256": EXPECTED_BASE_TAR_SHA256,
        "source_sanitized_build_sha256": EXPECTED_BASE_BUILD_SHA256,
        "audit_objects_included": False,
        "audit_model_evaluation_accesses": [],
        "bundle_path": tar_path.name,
        "bundle_sha256": sha256_file(tar_path),
        "seeds": registry_entries,
    }
    registry_path = output_dir / "seed-bundle-registry.json"
    registry_path.write_bytes(pretty_json(registry))
    summary = {
        "schema": "audit_135m_seed_bundle_build_v1",
        "status": "SEALED_DEVELOPMENT_ONLY",
        "stage_code": stage_code,
        "seeds": list(seeds),
        "bundle_path": str(tar_path),
        "bundle_sha256": registry["bundle_sha256"],
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "audit_objects_included": False,
        "audit_model_evaluation_accesses": [],
    }
    (output_dir / "build-summary.json").write_bytes(pretty_json(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-code", choices=sorted(audit.STAGE_CODES), required=True)
    parser.add_argument("--base-tar", type=Path, required=True)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build(
            stage_code=args.stage_code,
            base_tar=args.base_tar,
            base_build=args.base_build,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, SeedBundleError, audit.AuditContractError) as exc:
        print(f"audit seed-bundle error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
