"""Resource-ledger rows of SSH trainer incarnations (``yeto.cost.ledger`` kind
:data:`KIND`).

A Modal trainer is a FunctionCall and a sky trainer a cluster: both are in the
run's ledger the moment they exist, so ``yeto cleanup`` and the run's reaper
stop them when the launcher dies without a teardown, and ``yeto cost`` prices
them. An SSH trainer incarnation (one process group, and on docker nodes one
container, per node: ``ssh_trainer_island``) is recorded the same way:

* ``created`` right after its start was recorded in the jobs file (before any
  node runs it), with what a cleanup needs to stop it without the launcher: the
  run id, the island name, the incarnation, each node's SSH target and jump host,
  whether the nodes run docker, the kill grace;
* ``released`` once every node confirmed the stop (``SshTrainerIslands.release``
  / ``release_incarnation``); an unconfirmed stop stays open for the cleanup. A lost node
  that is not waited for (an island with spare nodes, ``island_spares``) gets its own row
  (:func:`record_owed`: that node and its rank only) once the other nodes confirmed;
  the incarnation's row closes. A lost node whose own stop was confirmed owes nothing; a
  confirmed owed stop later (:func:`release_owed`) closes that row;
* :func:`cleanup_release` (``yeto cleanup``, the reaper) runs the same kill
  script on every node, then ends the run's node leases there (inventory
  ``lease_store``: the row records the lease store and holder);
* the cost is the inventory's declared ``price_per_card_hour`` times the cards
  (:func:`usd_per_hour`); a node without a price leaves the row unpriced (the
  customer's own machines).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .node_inventory import Node

KIND = "ssh_trainer"


def resource_id(run_id: str, island: str, incarnation: int) -> str:
    return f"{run_id}/{island}-i{int(incarnation)}"


def usd_per_hour(nodes: Sequence[Node]) -> float | None:
    """The declared price of the nodes per hour; None when a node declares none."""
    if any(n.price_per_card_hour is None for n in nodes):
        return None
    return round(sum(float(n.price_per_card_hour) * n.cards_per_node for n in nodes), 6)


def created_attrs(*, run_id: str, island: str, incarnation: int, nodes: Sequence[Node], docker: bool,
                  kill_grace_s: int) -> dict[str, Any]:
    return {"run_id": str(run_id), "island": str(island), "incarnation": int(incarnation),
            "nodes": [{"name": n.name, "host": n.host, "proxy_jump": n.proxy_jump} for n in nodes],
            "docker": bool(docker), "kill_grace_s": int(kill_grace_s),
            "gpus": sum(n.cards_per_node for n in nodes), "usd_per_hour": usd_per_hour(nodes)}


def record(op: str, *, run_id: str, island: str, incarnation: int, attrs: dict[str, Any] | None = None) -> None:
    """``created`` / ``released`` on the run's ledger (``$YETO_RESOURCE_LEDGER``); a
    ledger error is printed, never raised (``record_safely``)."""
    from yeto.cost.ledger import record_safely

    rid = resource_id(run_id, island, incarnation)
    record_safely(op, KIND, rid, f"{island}-i{int(incarnation)}", **dict(attrs or {}))


def record_owed(*, run_id: str, island: str, incarnation: int, node: Node, rank: int, docker: bool,
                kill_grace_s: int) -> None:
    """The row of the stop a lost node still owes for an incarnation (``cleanup_release``
    stops it there, at its own rank); priced as that node only."""
    from yeto.cost.ledger import record_safely

    rid = owed_id(run_id, island, incarnation, node.name)
    attrs = created_attrs(run_id=run_id, island=island, incarnation=incarnation, nodes=[node], docker=docker,
                          kill_grace_s=kill_grace_s)
    record_safely("created", KIND, rid, f"{island}-i{int(incarnation)}-owed-{node.name}", ranks=[int(rank)],
                  **attrs)


def owed_id(run_id: str, island: str, incarnation: int, node: str) -> str:
    return f"{resource_id(run_id, island, incarnation)}-owed-{node}"


OWED_RELEASED = "released"  # release_owed: the row closed
OWED_ABSENT = "absent"  # not open in this ledger (never recorded, closed, or the ledger is elsewhere)
OWED_NO_LEDGER = "no-ledger"


def _run_ledger(run_id: str) -> Any:
    from yeto.cost.ledger import ResourceLedger, current

    return current() or ResourceLedger.for_run(str(run_id))


def release_owed(*, run_id: str, island: str, incarnation: int, node: str) -> tuple[str, str]:
    """The owed stop of ``node`` for that incarnation is confirmed (its own stop in the
    background, or ``yeto cluster release-node``): its row closes. The run's ledger is
    ``$YETO_RESOURCE_LEDGER``, else the run's own (a ``release-node`` outside the run).
    ``(status, ledger path)``: :data:`OWED_RELEASED`, :data:`OWED_ABSENT` (a ``release-node``
    on a machine without the run dir finds no row: the caller says where it is) or
    :data:`OWED_NO_LEDGER`."""
    from yeto.cost.ledger import record_safely

    try:
        ledger = _run_ledger(run_id)
    except Exception as error:  # noqa: BLE001 - printed, the row stays for yeto cleanup
        print(f"[cost] WARNING: no resource ledger for run {run_id} ({error}); the owed row of {node} stays "
              "open for `yeto cleanup`", flush=True)
        return OWED_NO_LEDGER, ""
    rid = owed_id(run_id, island, incarnation, node)
    if not any(r.id == rid for r in ledger.open_resources()):
        return OWED_ABSENT, str(ledger.path)  # never recorded (the stop came first), closed, or elsewhere
    record_safely("released", KIND, rid, f"{island}-i{int(incarnation)}-owed-{node}", ledger=ledger)
    return OWED_RELEASED, str(ledger.path)


def release_owed_elsewhere(*, run_id: str, node: str, stop: Any = None) -> list[str]:
    """``node`` is picked again and has no quarantine record: its owed stops were run
    elsewhere (``yeto cluster release-node`` on a machine without this run's ledger), so
    the run's open owed rows of that node close (``reason: released elsewhere``) and it is
    not billed twice. ``stop(row)``: run the row's stop first, the row closes only when it
    returns True. Returns the rows closed."""
    from yeto.cost.ledger import record_safely

    try:
        ledger = _run_ledger(run_id)
        rows = [r for r in ledger.open_resources()
                if r.kind == KIND and str(r.id).startswith(f"{run_id}/") and str(r.id).endswith(f"-owed-{node}")]
    except Exception:  # noqa: BLE001 - no ledger: nothing billed here
        return []
    closed = []
    for row in rows:
        if stop is not None and not stop(row):
            continue
        record_safely("released", KIND, row.id, getattr(row, "name", None), ledger=ledger, reason="released elsewhere")
        closed.append(str(row.id))
    return closed


class _LedgerNode:
    """The SSH reach of a node as the ledger row keeps it (``SshOps`` needs only this)."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.name = str(row.get("name") or row.get("host"))
        self.host = str(row["host"])
        self.proxy_jump = row.get("proxy_jump") or None

    def ssh_options(self) -> list[str]:
        return ["-o", f"ProxyJump={self.proxy_jump}"] if self.proxy_jump else []


def cleanup_release(resource: Any, ops: Any = None, dry_run: bool = False,
                    ssh: Any = None) -> tuple[bool, str, dict[str, Any]]:
    """Stop the incarnation of a ledger row on every node (``yeto cleanup``):
    ``(ok, action, result)``; ok only when every node confirmed the stop.
    ``ssh``: the ``SshOps`` (default: ``ops.ssh_ops`` when the cleanup ops carry one)."""
    from .ssh_trainer_island import (
        KILL_GRACE_S,
        SshOps,
        container_name,
        kill_script,
        remote_paths,
    )

    attrs = dict(getattr(resource, "attrs", None) or {})
    try:
        run_id, island, incarnation = str(attrs["run_id"]), str(attrs["island"]), int(attrs["incarnation"])
        nodes = [_LedgerNode(dict(row)) for row in attrs["nodes"]]
    except (KeyError, TypeError, ValueError):
        return False, "ledger row lacks the run id, island, incarnation or nodes", {}
    where = f"{island}-i{incarnation} on {len(nodes)} SSH node(s)"
    if dry_run:
        return True, f"would stop trainer incarnation {where}", {}
    grace = int(attrs.get("kill_grace_s") or KILL_GRACE_S)
    paths = remote_paths(run_id, island, incarnation)
    ssh = ssh or getattr(ops, "ssh_ops", None) or SshOps()
    failed: list[str] = []
    ranks = [int(r) for r in attrs.get("ranks") or range(len(nodes))]  # an owed row: the lost node's rank
    for rank, node in zip(ranks, nodes):
        container = container_name(run_id, island, incarnation, rank) if attrs.get("docker") else None
        try:
            done = ssh.run(node, kill_script(paths, grace, container=container), timeout=grace + 60.0)
            if done.returncode != 0:
                failed.append(f"{node.name}: rc {done.returncode}")
        except Exception as error:  # noqa: BLE001 - reported, the row stays open
            failed.append(f"{node.name}: {type(error).__name__}")
    if failed:
        return False, f"stop of trainer incarnation {where} not confirmed ({'; '.join(failed)})", {}
    released = release_node_leases(attrs, nodes)
    return True, f"stopped trainer incarnation {where}" + released, {"nodes": len(nodes)}


def release_node_leases(attrs: dict[str, Any], nodes: Sequence[Any]) -> str:
    """After a confirmed stop: end the node leases the row's run holds (inventory
    ``lease_store``: a launcher that ended with ``--keep`` kept them for good). Only a
    lease of the recorded holder that is kept or ran out is ended (a live one runs out by
    itself). Returns a note for the cleanup's action; never raises (an unreachable node
    keeps its lease kept: docs/OWN_CLUSTERS.md, Node leases)."""
    store, holder = attrs.get("lease_store"), attrs.get("lease_holder")
    if not store or not holder:
        return ""
    from .node_lease import NodeLeases  # noqa: PLC0415

    try:
        leases = NodeLeases(str(store), "cleanup")
        ended = [n.name for n in nodes if leases.force_release(n.name, holder=str(holder))]
    except Exception as error:  # noqa: BLE001 - the stop is confirmed; the lease is secondary
        return f"; node leases not released ({type(error).__name__})"
    return f"; node leases released on {ended}" if ended else ""
