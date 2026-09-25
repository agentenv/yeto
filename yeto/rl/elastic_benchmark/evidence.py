"""Evidence index, per-attempt results and resume validation.

Every attempt directory carries ``evidence-index.json`` (content digests of the
raw files it produced) and ``result.json`` (the four result layers). Resume
only reuses attempts whose study hash, matrix key and every digest still
match; failed attempts are kept as evidence and never overwritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.benchmark_resume import write_json_atomic

EVIDENCE_INDEX = "evidence-index.json"
RESULT_FILE = "result.json"
EXECUTION_STATES = ("completed", "failed", "unsupported", "blocked_dependency", "pending")
_CHUNK = 1024 * 1024


class EvidenceError(ValueError):
    """Evidence is missing, altered or belongs to a different study."""


@dataclass(frozen=True, order=True)
class MatrixKey:
    arm: str
    scenario: str
    seed: int
    config: str | None = None

    def relative_dir(self, attempt: int) -> Path:
        arm = self.arm if self.config is None else f"{self.arm}@{self.config}"
        return Path("runs") / arm / self.scenario / f"seed-{self.seed}" / f"attempt-{attempt}"

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm, "scenario": self.scenario, "seed": self.seed, "config": self.config}

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "MatrixKey":
        try:
            return MatrixKey(
                str(payload["arm"]), str(payload["scenario"]), int(payload["seed"]), payload.get("config")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceError(f"malformed matrix key: {payload}") from exc


def file_digest(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def build_evidence_index(
    attempt_dir: Path, *, study_hash: str, key: MatrixKey, attempt: int
) -> dict[str, Any]:
    """Digest every raw file under the attempt directory except the index and result."""
    files = {}
    for path in sorted(attempt_dir.rglob("*")):
        if path.is_symlink():
            raise EvidenceError(f"evidence directories do not support symlinks: {path}")
        if not path.is_file() or path.name in (EVIDENCE_INDEX, RESULT_FILE):
            continue
        digest, size = file_digest(path)
        files[path.relative_to(attempt_dir).as_posix()] = {"sha256": digest, "bytes": size}
    index = {
        "format_version": 1,
        "study_hash": study_hash,
        "key": key.as_dict(),
        "attempt": attempt,
        "files": files,
    }
    write_json_atomic(attempt_dir / EVIDENCE_INDEX, index)
    return index


def verify_evidence_index(attempt_dir: Path, *, study_hash: str, key: MatrixKey) -> dict[str, Any]:
    index = _read_json(attempt_dir / EVIDENCE_INDEX, "evidence index")
    if index.get("study_hash") != study_hash:
        raise EvidenceError(f"{attempt_dir}: evidence belongs to a different study")
    if MatrixKey.from_dict(index.get("key") or {}) != key:
        raise EvidenceError(f"{attempt_dir}: evidence belongs to a different matrix item")
    files = index.get("files")
    if not isinstance(files, dict):
        raise EvidenceError(f"{attempt_dir}: evidence index has no file table")
    for relative, expected in files.items():
        path = attempt_dir / relative
        if not path.is_file():
            raise EvidenceError(f"{attempt_dir}: evidence file is missing: {relative}")
        digest, size = file_digest(path)
        if digest != expected.get("sha256") or size != expected.get("bytes"):
            raise EvidenceError(f"{attempt_dir}: evidence file was modified: {relative}")
    extra = sorted(
        p.relative_to(attempt_dir).as_posix()
        for p in attempt_dir.rglob("*")
        if p.is_file() and p.name not in (EVIDENCE_INDEX, RESULT_FILE)
    )
    unexpected = sorted(set(extra) - set(files))
    if unexpected:
        raise EvidenceError(f"{attempt_dir}: unindexed evidence files: {unexpected[:4]}")
    return index


def write_result(attempt_dir: Path, result: dict[str, Any]) -> None:
    if result.get("execution") not in EXECUTION_STATES:
        raise EvidenceError(f"result.execution must be one of {EXECUTION_STATES}")
    for layer in ("correctness", "quality", "benefit"):
        if layer not in result:
            raise EvidenceError(f"result is missing the {layer!r} layer")
    target = attempt_dir / RESULT_FILE
    if target.exists():
        raise EvidenceError(f"refusing to overwrite an existing attempt result: {target}")
    write_json_atomic(target, result)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvidenceError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"{label} must be a JSON object: {path}")
    return payload


def list_attempts(study_dir: Path, key: MatrixKey) -> list[tuple[int, Path]]:
    base = study_dir / key.relative_dir(0).parent
    if not base.is_dir():
        return []
    attempts = []
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("attempt-"):
            try:
                attempts.append((int(child.name.split("-", 1)[1]), child))
            except ValueError:
                raise EvidenceError(f"malformed attempt directory: {child}") from None
    return sorted(attempts)


def next_attempt(study_dir: Path, key: MatrixKey) -> int:
    attempts = list_attempts(study_dir, key)
    return attempts[-1][0] + 1 if attempts else 1


def load_attempt_result(attempt_dir: Path) -> dict[str, Any] | None:
    path = attempt_dir / RESULT_FILE
    return _read_json(path, "attempt result") if path.is_file() else None


def reusable_completion(study_dir: Path, *, study_hash: str, key: MatrixKey) -> dict[str, Any] | None:
    """The verified completed result for a key, or None when it must be (re)run.

    A completed attempt whose evidence fails verification is an error, not a
    silent rerun: a benchmark must not launder tampered evidence by repeating.
    """
    for _attempt, attempt_dir in list_attempts(study_dir, key):
        result = load_attempt_result(attempt_dir)
        if result is None or result.get("execution") != "completed":
            continue
        verify_evidence_index(attempt_dir, study_hash=study_hash, key=key)
        return result
    return None


def collect_results(study_dir: Path, keys: list[MatrixKey]) -> dict[MatrixKey, list[dict[str, Any]]]:
    """Every attempt result per key, failed ones included, oldest first."""
    collected = {}
    for key in keys:
        results = []
        for attempt, attempt_dir in list_attempts(study_dir, key):
            result = load_attempt_result(attempt_dir)
            if result is not None:
                results.append({**result, "attempt": attempt})
        collected[key] = results
    return collected
