"""Where a quarantined node is recorded for every run of an inventory.

A spare-pool island (:mod:`.island_spares`) quarantines a node it lost; with an inventory
``lease_store`` the record is also ``nodes/<name>/quarantine.json`` there, next to
:mod:`.node_lease`'s ``nodes/<name>/lease/`` tokens and ``nodes/<name>/events/`` (never
inside them). Every run's trainer pick leaves such a node out until ``yeto cluster
release-node``. A leaf module: the trainer pick reads it without importing the spares.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

QUARANTINE_NAME = "quarantine.json"
SHARED_NODES_DIR = "nodes"  # under the inventory's lease_store
RELEASE_NODE_HINT = "run `yeto cluster release-node <name>`"


def shared_quarantined_names(shared: Any) -> set[str]:
    """The nodes quarantined in a lease store (every run's), for a pick's ``excluded``;
    empty when there is no store or it does not list."""
    if shared is None:
        return set()
    try:
        keys = shared.list(f"{SHARED_NODES_DIR}/")
    except Exception:  # noqa: BLE001 - unknown: none known
        return set()
    out = set()
    for key in keys:  # nodes/<name>/quarantine.json only
        parts = str(key).split("/")
        if len(parts) == 3 and parts[0] == SHARED_NODES_DIR and parts[2] == QUARANTINE_NAME:
            out.add(parts[1])
    return out


def shared_quarantine_note(shared: Any, names: Iterable[str]) -> str | None:
    """One line for a pick that leaves ``names`` out as quarantined: by which run and island,
    why, and how to give each node back (None: no names)."""
    parts = []
    for name in sorted(set(names)):
        try:
            raw = shared.read(f"{SHARED_NODES_DIR}/{name}/{QUARANTINE_NAME}")
            doc = json.loads(raw) if raw else {}
        except Exception:  # noqa: BLE001 - the record unread: the node is named all the same
            doc = {}
        doc = doc if isinstance(doc, dict) else {}
        who = f"run {doc.get('run') or '?'} island {doc.get('island') or '?'}"
        why = str(doc.get("reason") or "").strip()[:200]
        parts.append(f"{name} by {who}" + (f" ({why})" if why else ""))
    if not parts:
        return None
    return ("quarantined, left out: " + "; ".join(parts) + ". Once a node is fixed, " + RELEASE_NODE_HINT
            + " with this inventory and the run store of the run that quarantined it")
