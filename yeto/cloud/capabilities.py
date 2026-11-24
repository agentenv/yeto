"""Cloud capability table (rl-spot-cost-saving).

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
    """Raise if any field breaks the table rules (unchecked has a value; checked lacks a source)."""
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


PROVIDER_FIELDS = ("lease_bounded", "lease_s", "renewable", "notice_s", "cold_start_s", "hard_end", "currency",
                   "balance")
HARD_ENDS = (None, "lease", "function_timeout")


def validate_providers(table: dict[str, Any]) -> None:
    """Raise if a provider kind's row misses a field, a field misses its source / date, or a
    ``from_cloud`` notice carries a value of its own."""
    for kind, row in (table.get("providers") or {}).items():
        if kind.startswith("_"):
            continue
        for name in PROVIDER_FIELDS:
            cell = row.get(name)
            if not isinstance(cell, dict) or "value" not in cell:
                raise CapabilityTableError(f"providers.{kind}.{name}: missing")
            if not (cell.get("source") and cell.get("as_of")):
                raise CapabilityTableError(f"providers.{kind}.{name}: no source and date")
        if row["notice_s"].get("from_cloud") and row["notice_s"]["value"] is not None:
            raise CapabilityTableError(f"providers.{kind}.notice_s: from_cloud must have value null")
        if row["hard_end"]["value"] not in HARD_ENDS:
            raise CapabilityTableError(f"providers.{kind}.hard_end: one of {list(HARD_ENDS)}")
        if row["lease_bounded"]["value"] and not row["lease_s"]["value"]:
            raise CapabilityTableError(f"providers.{kind}: lease_bounded without lease_s")


@lru_cache(maxsize=4)
def load_table(path: str | None = None) -> dict[str, Any]:
    table = json.loads(Path(path or TABLE_PATH).read_text())
    validate(table)
    validate_providers(table)
    return table


def provider_row(kind: str, *, path: str | None = None) -> dict[str, Any] | None:
    """``{field: value}`` of a provider kind (``nc``, ``sky``, ``modal``, ...), plus
    ``notice_from_cloud``; None for a kind the table does not declare."""
    row = (load_table(path).get("providers") or {}).get(kind)
    if row is None or kind.startswith("_"):
        return None
    out = {name: row[name]["value"] for name in PROVIDER_FIELDS}
    out["notice_from_cloud"] = bool(row["notice_s"].get("from_cloud"))
    return out


def provider_notice_seconds(kind: str, cloud: str | None = None, *, path: str | None = None) -> float:
    """Seconds of warning before a machine of ``kind`` is reclaimed: the kind's own notice,
    or (``from_cloud``: sky) the checked notice of the cloud the machine runs on; 0 when
    neither is known (treat as no notice)."""
    row = provider_row(kind, path=path)
    if row is None:
        return 0.0
    if row["notice_from_cloud"]:
        return (notice_seconds(cloud, path=path) or 0.0) if cloud else 0.0
    return float(row["notice_s"] or 0.0)


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
