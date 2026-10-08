"""Read the elastic syncer's status snapshot (rl-inter-island-scheduling 0.17/0.18).

The Rust syncer in elastic mode writes ``status.json`` next to ``--event-tape``
(schema ``yeto.syncer.elastic-status/v1``, see syncer/src/elastic.rs
``status_json``). This module only reads it: a missing file, unparsable JSON or
a different schema gives ``None`` -- callers then leave scheduling fields
``None`` (never estimated). Legacy mode never calls into here.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

STATUS_SCHEMA = "yeto.syncer.elastic-status/v1"
STATUS_FILE = "status.json"

# Per-island fields copied from status["islands"][str(island_id)].
ISLAND_FIELDS = ("capacity", "round_wall_ema_s", "lease_remaining_s", "arrival_history",
                 "pending", "carried_over_lag")
# Global fields copied from the top level.
GLOBAL_FIELDS = ("syncer_epoch", "outer_version", "policy_hash")

log = logging.getLogger(__name__)


def status_path(location: str | Path) -> Path:
    """Accept the status file, its directory, or the ``--event-tape`` path."""
    p = Path(location).expanduser()
    if p.is_dir():
        return p / STATUS_FILE
    return p if p.name == STATUS_FILE else p.with_name(STATUS_FILE)


def read_syncer_status(location: str | Path) -> dict[str, Any] | None:
    path = status_path(location)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:  # half-written/garbled: treat as absent
        log.warning("syncer status %s unreadable: %s", path, exc)
        return None
    if not isinstance(raw, dict) or raw.get("schema") != STATUS_SCHEMA:
        log.warning("syncer status %s has schema %r, expected %s", path,
                    raw.get("schema") if isinstance(raw, dict) else None, STATUS_SCHEMA)
        return None
    return raw


def scheduling_fields(status: Mapping[str, Any] | None, island_id: int | str) -> dict[str, Any]:
    """The IslandStatus scheduling fields of one island; absent ones are omitted."""
    if not status:
        return {}
    out = {k: status[k] for k in GLOBAL_FIELDS if status.get(k) is not None}
    island = (status.get("islands") or {}).get(str(island_id))
    if isinstance(island, Mapping):
        for k in ISLAND_FIELDS:
            v = island.get(k)
            if v is not None:
                out[k] = tuple(v) if k == "arrival_history" else v
    return out


def scheduling_probe_from_status(location: str | Path, *, island_id: int | str) -> Callable[[], dict[str, Any]]:
    """A ``IslandController(scheduling_probe=...)`` reading the snapshot on each call."""
    def probe() -> dict[str, Any]:
        return scheduling_fields(read_syncer_status(location), island_id)
    return probe
