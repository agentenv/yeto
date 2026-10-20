"""The verdict every capability judge writes (``<run dir>/verdict.json``).

Status values:

* ``pass`` -- every criterion passed;
* ``fail`` -- a criterion failed on a backend that supports the capability;
* ``expected_fail`` -- a criterion failed and the backend does not support the
  capability (:attr:`cells.Cell.supported` is False): the run is kept as the
  regression baseline of the gap;
* ``unexpected_pass`` -- an unsupported cell passed (update the matrix);
* ``error`` -- the run produced no judgeable data (harness or launch failure).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PASS, FAIL, EXPECTED_FAIL, UNEXPECTED_PASS, ERROR = "pass", "fail", "expected_fail", "unexpected_pass", "error"
VERDICT_FILE = "verdict.json"


@dataclass
class Criterion:
    id: str
    text: str
    passed: bool
    evidence: Any = None


@dataclass
class Verdict:
    capability: str
    backend: str
    run_id: str
    supported: bool
    criteria: list[Criterion] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)  # wall seconds, card seconds, cost estimate, code sha
    error: str | None = None

    def check(self, cid: str, text: str, passed: bool, evidence: Any = None) -> bool:
        self.criteria.append(Criterion(cid, text, bool(passed), evidence))
        return bool(passed)

    @property
    def status(self) -> str:
        if self.error is not None or not self.criteria:
            return ERROR
        ok = all(c.passed for c in self.criteria)
        if self.supported:
            return PASS if ok else FAIL
        return UNEXPECTED_PASS if ok else EXPECTED_FAIL

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "status": self.status}

    def write(self, run_dir: Path) -> Path:
        path = Path(run_dir) / VERDICT_FILE
        path.write_text(json.dumps(self.to_dict(), indent=1, default=str) + "\n")
        return path


def load(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / VERDICT_FILE).read_text())


# ---------------------------------------------------------------- raw data readers
def jsonl(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of a JSONL file (missing file: []; bad lines skipped)."""
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def find(root: Path, name: str) -> Path:
    """The first file called ``name`` under ``root`` (sorted), else ``root/name``."""
    hits = sorted(Path(root).rglob(name)) if Path(root).is_dir() else []
    return hits[0] if hits else Path(root) / name


def find_all(root: Path, pattern: str) -> list[Path]:
    return sorted(Path(root).rglob(pattern)) if Path(root).is_dir() else []


def events(tape: Iterable[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [e for e in tape if e.get("event") == name]


def objects(directory: Path) -> list[dict[str, Any]]:
    """One JSON object per ``*.json`` file of a directory (object-store layouts)."""
    directory = Path(directory)
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))] if directory.is_dir() else []
