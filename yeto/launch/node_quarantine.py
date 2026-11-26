"""Where a quarantined node is recorded for every run of an inventory.

A spare-pool island (:mod:`.island_spares`) quarantines a node it lost; with an inventory
``lease_store`` the record is also ``nodes/<name>/quarantine.json`` there, next to
:mod:`.node_lease`'s ``nodes/<name>/lease/`` tokens and ``nodes/<name>/events/`` (never
inside them). Every run's trainer pick leaves such a node out until ``yeto cluster
release-node``. A leaf module: the trainer pick reads it without importing the spares.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any

SPARE_ROLE = "trainer-spare"  # an inventory node that may take a lost trainer node's place
QUARANTINE_NAME = "quarantine.json"
SHARED_NODES_DIR = "nodes"  # under the inventory's lease_store
RELEASE_NODE_HINT = "run `yeto cluster release-node <name>`"


def shared_key(name: str) -> str:
    return f"{SHARED_NODES_DIR}/{name}/{QUARANTINE_NAME}"


def _listed(shared: Any) -> set[str]:
    """The nodes with a quarantine record in the lease store (raises when it does not list)."""
    out = set()
    for key in shared.list(f"{SHARED_NODES_DIR}/"):  # nodes/<name>/quarantine.json only
        parts = str(key).split("/")
        if len(parts) == 3 and parts[0] == SHARED_NODES_DIR and parts[2] == QUARANTINE_NAME:
            out.add(parts[1])
    return out


def shared_quarantined_names(shared: Any) -> set[str]:
    """The nodes quarantined in a lease store (every run's); empty when there is no store
    or it does not list (:func:`shared_quarantines` reads each node's record then)."""
    if shared is None:
        return set()
    try:
        return _listed(shared)
    except Exception:  # noqa: BLE001 - see shared_quarantines
        return set()


def read_record(shared: Any, name: str) -> dict[str, Any] | None:
    """Node ``name``'s quarantine record in the lease store (None: none). A record that
    cannot be read (the store failed: ``read_error``) or parsed counts as a quarantine:
    ``{"unreadable": True, "reason": "quarantine record unreadable: ..."}``."""
    try:
        raw = shared.read(shared_key(name))
    except Exception as error:  # noqa: BLE001 - unknown is not free
        return {"unreadable": True, "read_error": True,
                "reason": f"quarantine record unreadable: {type(error).__name__}: {error}"[:300]}
    if raw is None:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        return {"unreadable": True, "reason": "quarantine record unreadable: not a JSON object"}
    return doc


def shared_quarantines(shared: Any, names: Iterable[str], *,
                       log: Callable[[str], None] | None = None) -> dict[str, dict[str, Any]]:
    """The quarantine records of ``names`` in the lease store (any run's): name -> record.
    A record that cannot be read counts as one (:func:`read_record`); when the store does
    not list, each node's record is read (a line says so)."""
    names = set(names)
    if shared is None or not names:
        return {}
    try:
        listed = _listed(shared) & names
    except Exception as error:  # noqa: BLE001 - each node read below
        (log or (lambda m: None))(f"[launcher] WARN: the lease store's quarantine list failed "
                                  f"({type(error).__name__}: {error}); each node's record is read instead")
        listed = names
    out = {}
    for name in sorted(listed):
        doc = read_record(shared, name)
        if doc is not None:
            out[name] = doc
    return out


def describe(name: str, doc: Mapping[str, Any]) -> str:
    """``<name> by run R island I (reason)`` of a quarantine record."""
    if doc.get("unreadable"):
        return f"{name} ({doc.get('reason') or 'quarantine record unreadable'})"
    who = f"run {doc.get('run') or '?'} island {doc.get('island') or '?'}"
    why = str(doc.get("reason") or "").strip()[:200]
    return f"{name} by {who}" + (f" ({why})" if why else "")


RELEASE_HOW = " with this inventory and the run store of the run that quarantined it"


def release_how(name: str) -> str:
    return RELEASE_NODE_HINT.replace("<name>", name) + RELEASE_HOW


def quarantine_note(docs: Mapping[str, Mapping[str, Any]]) -> str | None:
    """One line for a pick that leaves the nodes of ``docs`` out as quarantined: by which
    run and island, why, and how to give each back (None: none)."""
    if not docs:
        return None
    return ("quarantined, left out: " + "; ".join(describe(n, d) for n, d in sorted(docs.items()))
            + ". Once a node is fixed, " + RELEASE_NODE_HINT + RELEASE_HOW)
