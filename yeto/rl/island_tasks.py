"""Per-island training task lists (rl-per-island-tasks).

``yeto launch --rl-island-tasks`` gives each training island its own subset of
the tasks in ``--data``. The launcher filters the ``--data`` JSONL file on the
launching machine (P3-Q1), writes one prompt file per island and sends each
island its own file with that file's SHA256; the island checks the SHA256
before it trains and records its task list in ``rl_driver_start``.

The spec has two forms:

* inline: ``l0=web1-10,pwn1-10;l1=web11-20,pwn11-20`` -- island keys are
  ``l{id}`` or ``{id}`` (the ``--learner-image`` key rule);
* a JSON file: ``{"0": ["web1", ...], "1": [...]}``.

A token ``<prefix><a>-<b>`` names the tasks ``<prefix><a>`` ... ``<prefix><b>``
unless the token itself is a task id. A row's task id is
``metadata.task_id`` (what ``yeto.rl.rlf.catalog.prompt_rows`` writes) or a
top-level ``task_id``.

This module is backend-neutral: it knows JSONL rows and task ids only.
Callers pass the flag name (``flag=``) that error messages name.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ISLAND_KEY = re.compile(r"^l?(\d+)$")
_RANGE = re.compile(r"^(?P<prefix>.*?)(?P<start>\d+)-(?P<end>\d+)$")
_SEPARATORS = (",", ";", "=")


@dataclass(frozen=True)
class IslandTaskData:
    """One island's prompt file content and the tasks it holds."""

    learner_id: int
    task_ids: tuple[str, ...]
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    def record(self) -> dict[str, Any]:
        """What the run manifest and ``rl_driver_start`` carry for this island."""

        return {"task_ids": list(self.task_ids), "data_sha256": self.sha256}


def parse_island_tasks(value: str, *, flag: str) -> dict[int, tuple[str, ...]]:
    """Parse ``--rl-island-tasks`` (inline spec or JSON file) into island -> tokens.

    Tokens are not expanded here: range expansion needs the task ids in the
    data (see :func:`plan_island_tasks`).
    """

    text = (value or "").strip()
    if not text:
        raise ValueError(f"{flag} is empty")
    path = Path(text).expanduser()
    if "=" not in text and path.is_file():
        return _parse_json_spec(path, flag)
    if "=" not in text:
        raise ValueError(f"{flag} {value!r} is neither 'l0=...;l1=...' "
                         "nor an existing JSON file")
    spec: dict[int, tuple[str, ...]] = {}
    for entry in filter(None, (part.strip() for part in text.split(";"))):
        key, _, tokens = entry.partition("=")
        _add_island(spec, key.strip(), [t.strip() for t in tokens.split(",") if t.strip()], flag)
    return spec


def _parse_json_spec(path: Path, flag: str) -> dict[int, tuple[str, ...]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{flag} {path}: not a readable JSON file ({exc})") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{flag} {path}: expected an object of island -> task list")
    spec: dict[int, tuple[str, ...]] = {}
    for key, tokens in raw.items():
        if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
            raise ValueError(f"{flag} {path}: island {key!r} needs a list of strings")
        for token in tokens:
            if any(sep in token for sep in _SEPARATORS):
                raise ValueError(f"{flag} {path}: task id {token!r} contains one "
                                 f"of {_SEPARATORS}")
        _add_island(spec, str(key).strip(), [t.strip() for t in tokens], flag)
    return spec


def _add_island(spec: dict[int, tuple[str, ...]], key: str, tokens: list[str],
                flag: str) -> None:
    match = _ISLAND_KEY.match(key)
    if match is None:
        raise ValueError(f"{flag}: bad island key {key!r} (use l<id> or <id>)")
    island = int(match.group(1))
    if island in spec:
        raise ValueError(f"{flag}: island {island} is given twice")
    if not tokens or not all(tokens):
        raise ValueError(f"{flag}: island {island} has an empty task list")
    spec[island] = tuple(tokens)


def format_island_tasks(spec: Mapping[int, Sequence[str]]) -> str:
    """The inline form of ``spec`` (round-trips through :func:`parse_island_tasks`)."""

    return ";".join(f"l{island}={','.join(spec[island])}" for island in sorted(spec))


def row_task_id(row: Mapping[str, Any]) -> str | None:
    """A data row's task id: ``metadata.task_id``, else top-level ``task_id``."""

    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("task_id") is not None:
        return str(metadata["task_id"])
    return None if row.get("task_id") is None else str(row["task_id"])


def _read_rows(path: Path, flag: str) -> list[tuple[str, bytes]]:
    """``(task id, original line bytes)`` for every non-empty line of a JSONL file."""

    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: not a JSON line ({exc})") from exc
        task_id = row_task_id(row) if isinstance(row, dict) else None
        if task_id is None:
            raise ValueError(f"{path}:{number}: row has no metadata.task_id or task_id; "
                             f"{flag} needs a task id on every row")
        rows.append((task_id, line if line.endswith(b"\n") else line + b"\n"))
    if not rows:
        raise ValueError(f"{path}: no data rows")
    return rows


def _expand(token: str, known: set[str], flag: str) -> list[str]:
    if token in known:
        return [token]
    match = _RANGE.match(token)
    if match is None:
        return [token]
    start, end = match.group("start"), match.group("end")
    if int(start) > int(end):
        raise ValueError(f"{flag}: range {token!r} runs backwards")
    width = len(start) if start.startswith("0") and len(start) > 1 else 0
    return [f"{match.group('prefix')}{n:0{width}d}" for n in range(int(start), int(end) + 1)]


def _unique_in_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def plan_island_tasks(
    source: str | Path,
    spec: Mapping[int, Sequence[str]],
    islands: Sequence[int],
    *,
    flag: str,
) -> dict[int, IslandTaskData]:
    """Check ``spec`` against the data and build each island's prompt file content.

    Raises when an island is out of range, an island has no list, a task id is
    not in the data or one island names a task twice. Rows keep the source
    order and bytes, so the same source and spec always give the same SHA256.
    """

    unknown_islands = sorted(set(spec) - set(islands))
    if unknown_islands:
        raise ValueError(f"{flag} names islands {unknown_islands}, but the run has "
                         f"training islands {list(islands)}")
    missing = [island for island in islands if island not in spec]
    if missing:
        raise ValueError(f"{flag} has no task list for islands {missing}; "
                         "give every training island a list")
    rows = _read_rows(Path(source).expanduser(), flag)
    known = {task_id for task_id, _line in rows}
    plan = {}
    for island in islands:
        task_ids = [task for token in spec[island] for task in _expand(token, known, flag)]
        repeated = sorted(t for t, n in Counter(task_ids).items() if n > 1)
        if repeated:
            raise ValueError(f"{flag}: island {island} names {repeated} twice")
        absent = [t for t in task_ids if t not in known]
        if absent:
            raise ValueError(f"{flag}: island {island} names tasks not in "
                             f"{source}: {absent}")
        chosen = set(task_ids)
        lines = [line for task_id, line in rows if task_id in chosen]
        plan[island] = IslandTaskData(
            learner_id=island,
            task_ids=tuple(_unique_in_order(t for t, _line in rows if t in chosen)),
            content=b"".join(lines),
        )
    return plan


def task_count_warning(plan: Mapping[int, IslandTaskData], *, flag: str) -> str | None:
    """The launch warning for unequal per-island task counts (P3-Q2), or None."""

    counts = {island: len(data.task_ids) for island, data in sorted(plan.items())}
    if len(set(counts.values())) <= 1:
        return None
    listed = ", ".join(f"island {island}: {count}" for island, count in counts.items())
    return (f"{flag}: islands have different task counts ({listed}); smaller "
            "islands repeat their tasks more often, and task difficulty differences "
            "change each island's share of the merged update")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_island_data(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Island side: refuse a prompt file whose SHA256 is not the launcher's.

    Returns the ``rl_driver_start`` record (task ids in file order and SHA256).
    """

    expected = expected_sha256.strip().lower()
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"island task data {path} SHA256 mismatch: launcher sent "
                         f"{expected}, file has {actual}")
    rows = _read_rows(Path(path).expanduser(), "--data")
    task_ids = _unique_in_order(task_id for task_id, _line in rows)
    return {"task_ids": task_ids, "data_sha256": actual}
