"""Cloud capability table (rl-spot-cost-saving D2, task 1.1 / 1.8).

One static table, ``capabilities.json`` next to this file, one row per cloud.
Every field carries ``status``: ``checked`` (with ``source`` link and
``checked_on`` date) or ``unchecked`` (``value`` is None; never an estimate).

This module is the single reader. The future ``CloudProvider`` capability
declaration (yeto-framework-decoupling 8.1 / 8.5) MUST call
:func:`notice_seconds` / :func:`capability` instead of keeping its own copy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

TABLE_PATH = Path(__file__).with_name("capabilities.json")
FIELDS = ("notice_s", "notice_method", "durable_store", "spot", "price_source")
CHECKED, UNCHECKED = "checked", "unchecked"


@dataclass(frozen=True)
class Capability:
    cloud: str
    field: str
    value: Any
    status: str
    source: str | None = None
    checked_on: str | None = None
    kind: str | None = None
    note: str | None = None

    @property
    def checked(self) -> bool:
        return self.status == CHECKED


class CapabilityTableError(ValueError):
    pass


def validate(table: dict[str, Any]) -> None:
    """Raise if any field breaks the D2 rules (unchecked has a value; checked lacks a source)."""
    for cloud, row in table["clouds"].items():
        for name in FIELDS:
            cell = row.get(name)
            if cell is None:
                raise CapabilityTableError(f"{cloud}.{name}: missing")
            status = cell.get("status")
            if status not in (CHECKED, UNCHECKED):
                raise CapabilityTableError(f"{cloud}.{name}: bad status {status!r}")
            if status == CHECKED and not (cell.get("source") and cell.get("checked_on")):
                raise CapabilityTableError(f"{cloud}.{name}: checked without source and date")
            if status == UNCHECKED and name in ("notice_s",) and cell.get("value") is not None:
                raise CapabilityTableError(f"{cloud}.{name}: unchecked must have value null")


@lru_cache(maxsize=4)
def load_table(path: str | None = None) -> dict[str, Any]:
    table = json.loads(Path(path or TABLE_PATH).read_text())
    validate(table)
    return table


def clouds(path: str | None = None) -> list[str]:
    return sorted(load_table(path)["clouds"])


def capability(cloud: str, field: str, *, path: str | None = None) -> Capability:
    row = load_table(path)["clouds"].get(cloud.lower())
    if row is None:
        return Capability(cloud, field, None, UNCHECKED)
    cell = row[field]
    return Capability(cloud.lower(), field, cell.get("value"), cell["status"], cell.get("source"),
                      cell.get("checked_on"), cell.get("kind"), cell.get("note"))


def notice_seconds(cloud: str, *, path: str | None = None) -> float | None:
    """Seconds between the reclaim signal and the kill; None = treat as no notice.

    Only a ``checked`` value is returned. Unknown cloud or unchecked -> None."""
    cap = capability(cloud, "notice_s", path=path)
    return float(cap.value) if cap.checked and cap.value is not None else None


def durable_store(cloud: str, *, path: str | None = None) -> str | None:
    cap = capability(cloud, "durable_store", path=path)
    return cap.value  # value may be present but unchecked (e.g. AWS S3 not yet wired)
