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
  / ``release_incarnation``); an unconfirmed stop stays open for the cleanup;
* :func:`cleanup_release` (``yeto cleanup``, the reaper) runs the same kill
  script on every node;
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
    for rank, node in enumerate(nodes):
        container = container_name(run_id, island, incarnation, rank) if attrs.get("docker") else None
        try:
            done = ssh.run(node, kill_script(paths, grace, container=container), timeout=grace + 60.0)
            if done.returncode != 0:
                failed.append(f"{node.name}: rc {done.returncode}")
        except Exception as error:  # noqa: BLE001 - reported, the row stays open
            failed.append(f"{node.name}: {type(error).__name__}")
    if failed:
        return False, f"stop of trainer incarnation {where} not confirmed ({'; '.join(failed)})", {}
    return True, f"stopped trainer incarnation {where}", {"nodes": len(nodes)}
