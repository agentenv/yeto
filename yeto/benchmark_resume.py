"""Immutable configuration and data manifests for benchmark resume."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


_CHUNK_SIZE = 1024 * 1024


def jsonable_arguments(args, *, exclude: Iterable[str] = ()) -> dict:
    excluded = set(exclude)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in excluded
    }


def _feed(hasher, value: str | bytes) -> None:
    data = value.encode("utf-8") if isinstance(value, str) else value
    hasher.update(len(data).to_bytes(8, "little"))
    hasher.update(data)


def _feed_file(hasher, path: Path) -> int:
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            hasher.update(chunk)
            size += len(chunk)
    return size


def _snapshot_digest(path: Path) -> tuple[str, int, int]:
    hasher = hashlib.sha256()
    total_bytes = 0
    files = 0
    if path.is_file():
        _feed(hasher, "file")
        _feed(hasher, ".")
        total_bytes = _feed_file(hasher, path)
        return hasher.hexdigest(), 1, total_bytes
    if not path.is_dir():
        raise ValueError(f"benchmark input does not exist: {path}")
    _feed(hasher, "directory")
    for child in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            raise ValueError(f"benchmark manifests do not support symlinks: {child}")
        if child.is_dir():
            _feed(hasher, "directory")
            _feed(hasher, relative)
            continue
        if not child.is_file():
            raise ValueError(f"unsupported benchmark input entry: {child}")
        _feed(hasher, "file")
        _feed(hasher, relative)
        total_bytes += _feed_file(hasher, child)
        files += 1
    return hasher.hexdigest(), files, total_bytes


def _location(work_dir: Path, path: Path, *, require_within_work: bool) -> dict:
    work = work_dir.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(work)
    except ValueError:
        if require_within_work:
            raise ValueError(f"benchmark split must live under {work}: {resolved}") from None
        return {"scope": "absolute", "path": str(resolved)}
    return {"scope": "work", "path": relative.as_posix()}


def snapshot_path(
    work_dir: Path,
    path: Path,
    *,
    require_within_work: bool = False,
) -> dict:
    resolved = path.expanduser().resolve()
    digest, files, total_bytes = _snapshot_digest(resolved)
    return {
        **_location(work_dir, resolved, require_within_work=require_within_work),
        "kind": "directory" if resolved.is_dir() else "file",
        "sha256": digest,
        "files": files,
        "bytes": total_bytes,
    }


def build_data_manifest(
    work_dir: Path,
    train_path: Path,
    eval_path: Path,
    *,
    train_rows: int,
    eval_rows: int,
    source: str | Path | None = None,
) -> dict:
    manifest = {
        "format_version": 1,
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "train": snapshot_path(work_dir, train_path, require_within_work=True),
        "eval": snapshot_path(work_dir, eval_path, require_within_work=True),
    }
    if source is not None:
        source_path = Path(os.path.expanduser(str(source))).resolve()
        if source_path.exists():
            work = work_dir.expanduser().resolve()
            try:
                work.relative_to(source_path)
            except ValueError:
                manifest["source"] = snapshot_path(work_dir, source_path)
    return manifest


def _manifest_path(work_dir: Path, snapshot: dict) -> Path:
    scope = snapshot.get("scope")
    value = snapshot.get("path")
    if not isinstance(value, str):
        raise ValueError("benchmark data manifest has no valid path")
    if scope == "work":
        work = work_dir.expanduser().resolve()
        path = (work / value).resolve()
        try:
            path.relative_to(work)
        except ValueError:
            raise ValueError(f"benchmark data path escapes work directory: {value}") from None
        return path
    if scope == "absolute":
        return Path(value).expanduser().resolve()
    raise ValueError(f"benchmark data manifest has unknown scope: {scope!r}")


def _validate_snapshot(work_dir: Path, snapshot: dict, label: str) -> Path:
    path = _manifest_path(work_dir, snapshot)
    expected_kind = snapshot.get("kind")
    actual_kind = "directory" if path.is_dir() else "file" if path.is_file() else None
    if actual_kind != expected_kind:
        raise ValueError(
            f"{label} benchmark data changed: expected {expected_kind}, found {actual_kind}"
        )
    digest, files, total_bytes = _snapshot_digest(path)
    if (
        digest != snapshot.get("sha256")
        or files != snapshot.get("files")
        or total_bytes != snapshot.get("bytes")
    ):
        raise ValueError(f"{label} benchmark data changed since the run started: {path}")
    return path


def validate_data_manifest(work_dir: Path, manifest: dict) -> tuple[Path, Path, int]:
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported benchmark data manifest; restart with --overwrite")
    train = _validate_snapshot(work_dir, manifest.get("train") or {}, "train")
    evaluation = _validate_snapshot(work_dir, manifest.get("eval") or {}, "eval")
    if "source" in manifest:
        _validate_snapshot(work_dir, manifest["source"], "source")
    train_rows = manifest.get("train_rows")
    if not isinstance(train_rows, int) or train_rows < 1:
        raise ValueError("benchmark data manifest has an invalid train row count")
    eval_rows = manifest.get("eval_rows")
    if not isinstance(eval_rows, int) or eval_rows < 1:
        raise ValueError("benchmark data manifest has an invalid eval row count")
    return train, evaluation, train_rows


def implementation_fingerprint(repo_root: Path, paths: Iterable[Path]) -> str:
    root = repo_root.expanduser().resolve()
    files: set[Path] = set()
    for value in paths:
        path = value.expanduser().resolve()
        if path.is_dir():
            files.update(
                child
                for child in path.rglob("*")
                if child.is_file()
                and "__pycache__" not in child.parts
                and child.suffix != ".pyc"
            )
        elif path.is_file():
            files.add(path)
        else:
            raise ValueError(f"implementation input does not exist: {path}")
    hasher = hashlib.sha256()
    for path in sorted(files, key=lambda value: str(value)):
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        _feed(hasher, label)
        _feed_file(hasher, path)
    return hasher.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _identity_differences(expected, actual, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                differences.append(name)
            else:
                differences.extend(_identity_differences(expected[key], actual[key], name))
        return differences
    if expected != actual:
        return [prefix or "root"]
    return []


def load_resume_config(config_path: Path, expected_identity: dict) -> dict:
    if not config_path.is_file():
        raise ValueError(f"resume config is missing: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read resume config {config_path}: {exc}") from exc
    actual_identity = config.get("resume_identity")
    if actual_identity is None:
        raise ValueError(
            "this benchmark predates immutable resume manifests; restart with --overwrite"
        )
    differences = _identity_differences(expected_identity, actual_identity)
    if differences:
        detail = ", ".join(differences[:8])
        if len(differences) > 8:
            detail += f", and {len(differences) - 8} more"
        raise ValueError(f"resume configuration does not match the original run: {detail}")
    manifest = config.get("data_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("resume config has no data manifest; restart with --overwrite")
    return manifest


def validate_record_keys(records: list[dict], expected: set[tuple]) -> None:
    seen = set()
    for record in records:
        key = (record.get("kind"), record.get("arm"), record.get("seed"))
        if key not in expected:
            raise ValueError(f"resume results contain an unexpected record: {key}")
        if key in seen:
            raise ValueError(f"resume results contain a duplicate record: {key}")
        seen.add(key)
    if seen and ("base", "base", None) not in seen:
        raise ValueError("resume results are missing the base evaluation record")
    for record in records:
        if record.get("kind") != "diloco":
            continue
        baseline = (
            "baseline",
            f"baseline-m{record.get('learners')}",
            record.get("seed"),
        )
        if baseline not in seen:
            raise ValueError(
                f"resume results are missing {baseline[1]} for seed={baseline[2]}"
            )
