"""Checks of a run's own (SSH inventory) nodes across its pools and its trainer islands.

``yeto run up`` (and its dry run) refuses, before anything starts:

* two ``provider.kind: ssh`` pools of the run on one node whose inventory keeps node
  leases (``lease_store``): a node lease is held per node, so the second pool would wait
  for that node for as long as the first runs. Two pools may share a node's cards
  (``provider.devices``) through an inventory without ``lease_store``;
* an ssh pool on a node of the run's own ``ssh:`` trainer islands (their nodes and
  their ``trainer-spare`` nodes): the trainer takes every card of its nodes.

One machine is one SSH target (``host``), one cluster address (``address``, a loopback
one excepted: it is each container's own), or one node name of one inventory file.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .node_inventory import Node


@dataclass(frozen=True)
class TrainerNode:
    """A node of the run's trainer side: its island entry (``ssh:...``), its inventory file,
    and whether it is a spare (``trainer-spare``)."""

    node: Node
    entry: str
    inventory: str | None
    spare: bool = False


def _path(value: Any) -> str | None:
    if not value:
        return None
    try:
        return str(Path(os.path.expanduser(str(value))).resolve())
    except OSError:
        return str(value)


def _address(node: Node) -> str | None:
    address = getattr(node, "address", None)
    if not address:
        return None
    try:
        return None if ipaddress.ip_address(address).is_loopback else str(address)
    except ValueError:
        return str(address)


def same_machine(a: Node, a_inventory: str | None, b: Node, b_inventory: str | None) -> bool:
    """One machine (module doc): one SSH target, one address, or one name of one inventory."""
    if a.host == b.host:
        return True
    if _address(a) is not None and _address(a) == _address(b):
        return True
    return a.name == b.name and a_inventory is not None and a_inventory == b_inventory


def _store(value: Any) -> str | None:
    return str(value).rstrip("/") if value else None


def pool_problems(pools: Sequence[tuple[str, Mapping[str, Any]]],
                  trainers: Sequence[TrainerNode] = ()) -> list[str]:
    """The refusals of the module doc for the run's ``(pool name, pool config)`` (pools of
    another kind are skipped; a pool config its own checks refuse is skipped too: its
    pool dry run says why) against its trainer side ``trainers``."""
    from .ssh_sampler_pool import parse_config  # noqa: PLC0415

    parsed = []
    for name, cfg in pools:
        provider = cfg.get("provider") if isinstance(cfg.get("provider"), Mapping) else {}
        if provider.get("kind") != "ssh":
            continue
        try:
            conf = parse_config(dict(cfg))
        except (ValueError, TypeError, KeyError, OSError):
            continue
        parsed.append((name, conf, _path(provider.get("inventory"))))
    out = []
    for i, (name_a, conf_a, inv_a) in enumerate(parsed):
        for name_b, conf_b, inv_b in parsed[i + 1:]:
            store = _store(conf_a.lease_store)
            if store is None or store != _store(conf_b.lease_store):
                continue  # a node lease is taken per node in one lease store: no shared store, no wait
            for node in conf_a.nodes:
                if any(n.name == node.name for n in conf_b.nodes):
                    out.append(
                        f"pools {name_a} and {name_b} both start samplers on node {node.name}, whose inventory "
                        f"keeps node leases (lease_store {store}): the lease is held per node, so the second pool "
                        f"would wait for {node.name} as long as the first runs. Either give the pools an "
                        "inventory without lease_store and split the node's cards with provider.devices, or put "
                        "the pools on different nodes")
    for name, conf, inv in parsed:
        for node in conf.nodes:
            for t in trainers:
                if same_machine(node, inv, t.node, t.inventory):
                    what = "a trainer-spare node" if t.spare else "a node"
                    out.append(
                        f"pool {name} starts samplers on node {node.name}, which is {what} of this run's trainer "
                        f"island {t.entry} ({t.node.name}): the trainer takes every card of its nodes. Put the pool "
                        "on nodes with the sampler role only (inventory roles), or name them in provider.nodes")
                    break
    return out


def trainer_nodes(entries: Sequence[str], inventory: str | None, *, every_candidate: bool = False) -> list[TrainerNode]:
    """The trainer side of the run's ``ssh:`` island entries (``--gpu``): the nodes each
    entry takes (``every_candidate``: every node of its inventory a trainer may take, the
    nodes of a fleet's islands being picked at launch) and the inventory's
    ``trainer-spare`` nodes. An entry whose inventory does not load is skipped (the
    trainer launch's own dry run says why)."""
    from .node_quarantine import SPARE_ROLE  # noqa: PLC0415
    from .node_inventory import InventoryError, island_nodes  # noqa: PLC0415
    from .preflight.cluster import resolve_inventory  # noqa: PLC0415

    out: list[TrainerNode] = []
    for entry in dict.fromkeys(entries):
        try:
            nodes, every = resolve_inventory(entry, inventory)
        except (InventoryError, OSError, ValueError):
            continue
        rest = entry[4:]
        path = _path(rest if rest.endswith((".yaml", ".yml", ".json")) or Path(rest).expanduser().is_file()
                     else inventory)
        taken = island_nodes(list(every), "trainer") if every_candidate else list(nodes)
        out += [TrainerNode(n, entry, path) for n in taken]
        out += [TrainerNode(n, entry, path, spare=True) for n in every
                if SPARE_ROLE in n.roles and n.name not in {t.name for t in taken}]
    return out
