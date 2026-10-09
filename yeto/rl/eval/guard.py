"""Start-time hold-out checks (rl-eval-difficulty-buckets D6.c, tasks 2.1/2.2).

Run by the training driver before any GPU action, on CPU. Refuses to start when

1. a training row's ``task_id`` is in any hold-out list;
2. (SWE-like rows) training and eval rows share ``(repo, base_commit)`` or the
   sha256 of the normalized ``problem_statement`` (different datasets may give
   one issue different ids);
3. a hold-out file's sha256 differs from the pinned value, the eval data has a
   task outside its list, or a listed task is missing from the eval data.

The returned report goes into ``rl_driver_start`` (key ``eval_guard``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from yeto.rl.harness.reward_env.benchmark import holdout_ids, holdout_sha256, task_id_of

HOLDOUT_ENV = "YETO_RL_EVAL_HOLDOUT"   # "path[@sha256][,path[@sha256]...]"
EVAL_DATA_ENV = "YETO_RL_EVAL_DATA"    # "path[,path...]" eval jsonl (rows in D6.a shape)


class EvalGuardError(RuntimeError):
    """Training and evaluation data are not cleanly separated (fail closed)."""


@dataclass(frozen=True)
class HoldoutRef:
    path: str
    sha256: str | None = None


def parse_holdout_env(value: str | None) -> list[HoldoutRef]:
    refs = []
    for part in (value or "").split(","):
        part = part.strip()
        if not part:
            continue
        path, _, sha = part.partition("@")
        refs.append(HoldoutRef(path, sha or None))
    return refs


def _norm_statement(text: str) -> str:
    return hashlib.sha256("".join(str(text).split()).lower().encode()).hexdigest()


def _meta(row: Mapping[str, Any]) -> Mapping[str, Any]:
    meta = row.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def _field(row: Mapping[str, Any], name: str) -> Any:
    return _meta(row).get(name, row.get(name))


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvalGuardError(f"{path}:{n}: not JSON ({exc})") from exc
    return rows


def _fingerprints(rows: Iterable[Mapping[str, Any]]) -> tuple[set[tuple[str, str]], set[str]]:
    repo_commit: set[tuple[str, str]] = set()
    statements: set[str] = set()
    for row in rows:
        repo, commit = _field(row, "repo"), _field(row, "base_commit")
        if repo and commit:
            repo_commit.add((str(repo), str(commit)))
        stmt = _field(row, "problem_statement")
        if stmt:
            statements.add(_norm_statement(stmt))
    return repo_commit, statements


def _refuse(what: str, overlap: Iterable[Any]) -> None:
    items = sorted(overlap, key=str)
    if items:
        raise EvalGuardError(f"training and evaluation sets overlap by {what}: {len(items)}, e.g. {items[:5]}")


def check(train_rows: list[Mapping[str, Any]], holdouts: list[HoldoutRef],
          eval_rows: list[Mapping[str, Any]] | None = None, *,
          train_label: str | None = None) -> dict[str, Any]:
    """Return the ``eval_guard`` report or raise :class:`EvalGuardError`."""
    if not holdouts:
        raise EvalGuardError("no hold-out list given")
    held: dict[str, str] = {}
    lists = []
    for ref in holdouts:
        try:
            doc = json.loads(Path(ref.path).read_text(encoding="utf-8"))
            ids = holdout_ids(doc)
        except (OSError, ValueError) as exc:
            raise EvalGuardError(f"hold-out list {ref.path}: {exc}") from exc
        sha = holdout_sha256(doc)
        if ref.sha256 and ref.sha256 != sha:
            raise EvalGuardError(f"hold-out list {ref.path}: sha256 {sha[:12]} != pinned {ref.sha256[:12]}")
        for tid in ids:
            held[tid] = ref.path
        lists.append({"path": os.path.basename(ref.path), "benchmark": doc.get("benchmark"),
                      "tasks": len(ids), "sha256": sha})
    train_ids = []
    for n, row in enumerate(train_rows):
        tid = task_id_of(dict(row))
        if tid is None:
            raise EvalGuardError(f"training row {n} has no metadata.task_id")
        train_ids.append(tid)
    _refuse("task_id", set(train_ids) & set(held))
    report: dict[str, Any] = {"train_rows": len(train_rows), "train_tasks": len(set(train_ids)),
                              "holdouts": lists, "overlap": 0}
    if train_label:
        report["train_data"] = train_label
    if eval_rows is not None:
        eval_ids = []
        for n, row in enumerate(eval_rows):
            tid = task_id_of(dict(row))
            if tid is None:
                raise EvalGuardError(f"eval row {n} has no metadata.task_id")
            eval_ids.append(tid)
        outside = sorted(set(eval_ids) - set(held))
        if outside:
            raise EvalGuardError(f"eval data has tasks outside the hold-out lists: {outside[:5]}")
        missing = sorted(set(held) - set(eval_ids))
        if missing:
            raise EvalGuardError(f"hold-out tasks missing from the eval data: {len(missing)}, e.g. {missing[:5]}")
        t_rc, t_st = _fingerprints(train_rows)
        e_rc, e_st = _fingerprints(eval_rows)
        _refuse("(repo, base_commit)", t_rc & e_rc)
        _refuse("problem_statement sha256", t_st & e_st)
        report["eval_rows"] = len(eval_rows)
    return report


def check_from_env(train_paths: Iterable[str], env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Launch-time entry: ``None`` when no hold-out is configured (old runs unchanged)."""
    env = os.environ if env is None else env
    refs = parse_holdout_env(env.get(HOLDOUT_ENV))
    if not refs:
        return None
    paths = [p for p in train_paths if p]
    if not paths:
        raise EvalGuardError(f"{HOLDOUT_ENV} is set but the run has no training data path to check")
    train_rows: list[Mapping[str, Any]] = []
    for path in paths:
        train_rows += read_jsonl(path)
    eval_rows = None
    if env.get(EVAL_DATA_ENV):
        eval_rows = []
        for path in env[EVAL_DATA_ENV].split(","):
            if path.strip():
                eval_rows += read_jsonl(path.strip())
    return check(train_rows, refs, eval_rows, train_label=",".join(os.path.basename(p) for p in paths))
