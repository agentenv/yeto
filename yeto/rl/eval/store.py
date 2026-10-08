"""Durable eval store (rl-eval-difficulty-buckets D11.2/D11.3/D11.7).

A plain directory tree; on the first version it is a Modal Volume mounted on
both the training island (or uploaded through the Modal API when training is
not on Modal) and the eval island. Layout::

    <root>/versions/v000010/manifest.json      written LAST (presence = complete)
    <root>/versions/v000010/files/<name>        policy files (adapter, config)
    <root>/queue/v000010.json                   registered pending version
    <root>/results/v000010.jsonl                append-only unit log
    <root>/done/v000010.json                    the emitted ``rl_eval`` payload

The unit log holds two record kinds per ``(policy_version, task_id, trial)``:
``start`` (written before the attempt) and ``result`` (written after). A start
without a result means the attempt was lost (eval island preempted); it is
counted in ``eval/preemptions`` and the unit is run again. Duplicate results
of one unit keep the first.

``commit`` / ``reload`` hooks let a Modal Volume flush and refresh (Volume
writes are visible to other containers only after ``commit()``); a local
directory needs neither.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

MANIFEST_SCHEMA = "yeto-eval-policy/1"
RESULT_SCHEMA = "yeto-eval-unit/1"
FINISHED_MARKER = "training-finished.json"  # written by the training side after its final version


class EvalIntegrityError(RuntimeError):
    """A stored policy does not match its manifest (fail closed, never evaluate)."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _vname(version: int) -> str:
    if int(version) < 0:
        raise ValueError(f"policy version must be >= 0, got {version}")
    return f"v{int(version):06d}"


def unit_key(record: Mapping[str, Any]) -> tuple[int, str, int]:
    return int(record["policy_version"]), str(record["task_id"]), int(record["trial"])


@dataclass
class UnitLog:
    """Parsed unit log of one version."""

    results: dict[tuple[int, str, int], dict[str, Any]]
    orphan_starts: int          # starts never followed by a result (lost attempts)
    duplicate_results: int


class EvalStore:
    def __init__(self, root: str | os.PathLike[str], *, commit: Callable[[], Any] | None = None,
                 reload: Callable[[], Any] | None = None) -> None:
        self.root = Path(root)
        self._commit = commit
        self._reload = reload

    # -- volume hooks ---------------------------------------------------
    def commit(self) -> None:
        if self._commit is not None:
            self._commit()

    def reload(self) -> None:
        if self._reload is not None:
            self._reload()

    # -- paths ------------------------------------------------------------
    def version_dir(self, version: int) -> Path:
        return self.root / "versions" / _vname(version)

    def files_dir(self, version: int) -> Path:
        return self.version_dir(version) / "files"

    def _manifest_path(self, version: int) -> Path:
        return self.version_dir(version) / "manifest.json"

    def _queue_path(self, version: int) -> Path:
        return self.root / "queue" / f"{_vname(version)}.json"

    def _results_path(self, version: int) -> Path:
        return self.root / "results" / f"{_vname(version)}.jsonl"

    def _done_path(self, version: int) -> Path:
        return self.root / "done" / f"{_vname(version)}.json"

    # -- training side ----------------------------------------------------
    def put_version(self, version: int, *, files: Mapping[str, bytes | str | os.PathLike[str]],
                    policy_tensor_hash: str, policy_token: str, sampling: Mapping[str, Any],
                    extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write the policy files, then the manifest (with every file's sha256),
        then register the version in the queue. A reader never sees a manifest
        whose files are not all written."""
        if not files:
            raise ValueError("an eval version needs at least one policy file")
        out = self.files_dir(version)
        out.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, dict[str, Any]] = {}
        for name, src in sorted(files.items()):
            if "/" in name or name in ("", ".", ".."):
                raise ValueError(f"policy file name must be a plain name: {name!r}")
            target = out / name
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            _atomic_write(target, data)
            hashes[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        manifest = {
            "schema": MANIFEST_SCHEMA, "policy_version": int(version),
            "policy_tensor_hash": str(policy_tensor_hash), "rl/policy_token": str(policy_token),
            "sampling": dict(sampling), "files": hashes, **dict(extra or {}),
        }
        _atomic_write(self._manifest_path(version), json.dumps(manifest, sort_keys=True, indent=1).encode())
        _atomic_write(self._queue_path(version), json.dumps(
            {"policy_version": int(version), "rl/policy_token": str(policy_token)}, sort_keys=True).encode())
        self.commit()
        return manifest

    def mark_training_finished(self, payload: Mapping[str, Any] | None = None) -> None:
        """Training wrote its last eval version (5.10): the eval island stops once the queue is empty."""
        _atomic_write(self.root / FINISHED_MARKER, json.dumps(dict(payload or {}), sort_keys=True).encode())
        self.commit()

    def training_finished(self) -> bool:
        return (self.root / FINISHED_MARKER).is_file()

    # -- eval side ----------------------------------------------------------
    def queued_versions(self) -> list[int]:
        qdir = self.root / "queue"
        if not qdir.is_dir():
            return []
        return sorted(int(p.stem[1:]) for p in qdir.glob("v*.json"))

    def done_versions(self) -> list[int]:
        ddir = self.root / "done"
        if not ddir.is_dir():
            return []
        return sorted(int(p.stem[1:]) for p in ddir.glob("v*.json"))

    def pending_versions(self) -> list[int]:
        done = set(self.done_versions())
        return [v for v in self.queued_versions() if v not in done]

    def load_manifest(self, version: int, *, verify: bool = True) -> dict[str, Any]:
        path = self._manifest_path(version)
        if not path.is_file():
            raise EvalIntegrityError(f"{_vname(version)}: no manifest (version not completely written)")
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != MANIFEST_SCHEMA or int(manifest.get("policy_version", -1)) != int(version):
            raise EvalIntegrityError(f"{_vname(version)}: manifest schema/version mismatch")
        if verify:
            self.verify_files(version, manifest)
        return manifest

    def verify_files(self, version: int, manifest: Mapping[str, Any]) -> None:
        for name, meta in sorted(manifest["files"].items()):
            path = self.files_dir(version) / name
            if not path.is_file():
                raise EvalIntegrityError(f"{_vname(version)}: missing policy file {name}")
            got = sha256_file(path)
            if got != meta["sha256"]:
                raise EvalIntegrityError(
                    f"{_vname(version)}: sha256 mismatch for {name}: manifest {meta['sha256'][:12]}, "
                    f"stored {got[:12]}")

    def append_unit(self, record: Mapping[str, Any]) -> None:
        kind = record.get("kind")
        if kind not in ("start", "result"):
            raise ValueError(f"unit record kind must be start/result, got {kind!r}")
        unit_key(record)  # validates the key fields
        path = self._results_path(int(record["policy_version"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"schema": RESULT_SCHEMA, **record}, sort_keys=True, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self.commit()

    def read_units(self, version: int) -> UnitLog:
        path = self._results_path(version)
        results: dict[tuple[int, str, int], dict[str, Any]] = {}
        starts: dict[tuple[int, str, int], int] = {}
        dup = 0
        if path.is_file():
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue  # a torn last line from a preempted writer
                key = unit_key(rec)
                if rec.get("kind") == "start":
                    starts[key] = starts.get(key, 0) + 1
                elif rec.get("kind") == "result":
                    if key in results:
                        dup += 1
                    else:
                        results[key] = rec
        orphans = sum(n - (1 if k in results else 0) for k, n in starts.items())
        return UnitLog(results=results, orphan_starts=max(orphans, 0), duplicate_results=dup)

    def results_sha256(self, version: int) -> str | None:
        """sha256 of the deduplicated results, canonical order (resume == one-shot)."""
        log = self.read_units(version)
        if not log.results:
            return None
        rows = [_canonical_result(log.results[k]) for k in sorted(log.results)]
        return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()

    def mark_done(self, version: int, payload: Mapping[str, Any]) -> None:
        _atomic_write(self._done_path(version), json.dumps(dict(payload), sort_keys=True, indent=1).encode())
        self.commit()

    def done_payload(self, version: int) -> dict[str, Any] | None:
        path = self._done_path(version)
        return json.loads(path.read_text()) if path.is_file() else None


# fields that depend on wall clock / attempt identity, not on the outcome
_VOLATILE = ("t", "seconds", "attempt", "island", "trajectory_id", "timing")


def _canonical_result(rec: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in _VOLATILE}


def iter_results(log: UnitLog) -> Iterable[dict[str, Any]]:
    return (log.results[k] for k in sorted(log.results))
