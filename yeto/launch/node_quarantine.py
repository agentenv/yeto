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
from pathlib import Path
from typing import Any

SPARE_ROLE = "trainer-spare"  # an inventory node that may take a lost trainer node's place
JOURNAL_SUFFIX = "-spares.json"  # next to a resident island's jobs journal: the nodes of each incarnation
QUARANTINE_NAME = "quarantine.json"
SHARED_NODES_DIR = "nodes"  # under the inventory's lease_store
RUN_NODES_DIR = "fleet/nodes"  # under a run store: fleet/nodes/<name>/quarantine.json (island_spares)
RELEASE_NODE_HINT = "run `yeto cluster release-node <name>`"


def latest_spare_nodes(path: Any) -> list[str] | None:
    """The nodes the newest incarnation of a resident island with spare nodes runs on, by
    its spares journal (``<island>-spares.json``, written before any start of that
    incarnation); None without one."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = dict(doc.get("nodes") or {})
        return [str(n) for n in nodes[max(nodes, key=int)]] if nodes else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def shared_key(name: str) -> str:
    return f"{SHARED_NODES_DIR}/{name}/{QUARANTINE_NAME}"


def run_key(name: str) -> str:
    return f"{RUN_NODES_DIR}/{name}/{QUARANTINE_NAME}"


def _listed(shared: Any, base: str = SHARED_NODES_DIR) -> set[str]:
    """The nodes with a quarantine record under ``base`` (``nodes`` in a lease store,
    ``fleet/nodes`` in a run store); raises when the store does not list."""
    out = set()
    depth = base.count("/") + 1
    for key in shared.list(f"{base}/"):  # <base>/<name>/quarantine.json only
        parts = str(key).split("/")
        if len(parts) == depth + 2 and "/".join(parts[:depth]) == base and parts[-1] == QUARANTINE_NAME:
            out.add(parts[depth])
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


def unreadable(error: BaseException | str) -> dict[str, Any]:
    """The record of a node whose quarantine could not be read (the store failed): it counts
    as quarantined."""
    what = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    return {"unreadable": True, "read_error": True, "reason": f"quarantine record unreadable: {what}"[:300]}


def read_record(shared: Any, name: str, *, key: str | None = None) -> dict[str, Any] | None:
    """Node ``name``'s quarantine record in the lease store (``key``: another place, e.g.
    :func:`run_key` in a run store; None: none). A record that cannot be read (the store
    failed: ``read_error``) or parsed counts as a quarantine:
    ``{"unreadable": True, "reason": "quarantine record unreadable: ..."}``."""
    try:
        raw = shared.read(key or shared_key(name))
    except Exception as error:  # noqa: BLE001 - unknown is not free
        return unreadable(error)
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


def run_store_quarantines(objects: Any, names: Iterable[str], *,
                          log: Callable[[str], None] | None = None) -> dict[str, dict[str, Any]]:
    """The quarantine records of ``names`` in a run store (``fleet/nodes/<name>/quarantine.json``,
    this run's islands): name -> record, as :func:`shared_quarantines` reads a lease store.
    A record that cannot be read counts as one (:func:`read_record`); when the store does
    not list, each node's record is read (a line says so)."""
    names = set(names)
    if objects is None or not names:
        return {}
    try:
        listed = _listed(objects, RUN_NODES_DIR) & names
    except Exception as error:  # noqa: BLE001 - each node read below
        (log or (lambda m: None))(f"[launcher] WARN: the run store's quarantine list failed "
                                  f"({type(error).__name__}: {error}); each node's record is read instead")
        listed = names
    out = {}
    for name in sorted(listed):
        doc = read_record(objects, name, key=run_key(name))
        if doc is not None:
            out[name] = doc
    return out


def trainer_pick_excluded(names: Iterable[str], *, lease_objects: Any = None, run_objects: Any = None,
                          assigned: Iterable[str] = (), log: Callable[[str], None] | None = None,
                          ) -> tuple[set[str], dict[str, dict[str, Any]], str | None]:
    """What a trainer pick among ``names`` leaves out: ``(excluded, quarantines, note)``.

    ``excluded``: ``assigned`` (nodes this process gave other islands already) and every
    node quarantined in the lease store (``lease_objects``: any run's, :func:`shared_quarantines`)
    or in the run store (``run_objects``: this run's, :func:`run_store_quarantines`); an
    unreadable record counts. ``quarantines``: name -> record; ``note``:
    :func:`quarantine_note` (None: none). The launch's trainer pick
    (``ssh_trainer_island.lease_trainer_nodes``) calls it with the run's ``NodeLeases``
    (``lease_objects=leases.objects``, ``assigned=leases.assigned``, no ``run_objects``: with a
    lease store every quarantine is there too); a prediction of that pick calls it with the
    same arguments."""
    names = set(names)
    docs = dict(shared_quarantines(lease_objects, names, log=log))
    for name, doc in run_store_quarantines(run_objects, names, log=log).items():
        docs.setdefault(name, doc)
    return set(assigned) | set(docs), docs, quarantine_note(docs)


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
