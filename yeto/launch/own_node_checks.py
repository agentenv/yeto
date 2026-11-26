"""Checks of a run's own (SSH inventory) nodes across its pools and its trainer islands.

``yeto run up`` (and its dry run) checks, before anything starts:

* two ``provider.kind: ssh`` pools of the run that may take one node of one lease store
  (``lease_store``: a node lease is held per node name). A pool skips a node another
  pool holds and takes another free one of its slots, so this is refused only when
  there are not enough nodes: both pools name the node (``provider.nodes``), or their
  targets together exceed the nodes the two pools can take. Otherwise a note says it.
  Two pools may share a node's cards (``provider.devices``) through an inventory
  without ``lease_store``;
* an ssh pool on a node of the run's own ``ssh:`` trainer islands (their nodes and their
  ``trainer-spare`` nodes). Without node leases common to both (one lease store, one
  node name) the trainer takes every card of its nodes: refused. With them, the
  trainer picks free nodes and skips one a pool holds: refused only when the trainer
  nodes left after the pools' (quarantined nodes not counted) are fewer than the
  islands need; otherwise a warning.

One machine is one SSH host (``host`` after any ``user@``), one cluster address
(``address``, a loopback one excepted: it is each container's own), or one node name of
one inventory file.
"""

from __future__ import annotations

import ipaddress
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .node_inventory import Node

ERROR, WARN, NOTE = "error", "warn", "note"


@dataclass(frozen=True)
class Finding:
    level: str  # ERROR refuses the run; WARN / NOTE are preflight lines
    text: str


@dataclass(frozen=True)
class OwnPool:
    """An ssh pool of the run: the nodes its slots may take, its target, and whether it
    names them (``provider.nodes``)."""

    name: str
    nodes: tuple[Node, ...]
    target: int
    named: bool
    store: str | None
    inventory: str | None


@dataclass
class TrainerSide:
    """The ``ssh:`` trainer islands of one inventory file: their entries (one per island),
    the nodes they may take (``nodes``; every trainer node when they are picked at launch),
    the inventory's spares, its lease store and the nodes the islands need."""

    inventory: str | None
    store: str | None
    entries: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    spares: list[Node] = field(default_factory=list)
    need: int = 0
    quarantined: set[str] = field(default_factory=set)


def _path(value: Any) -> str | None:
    if not value:
        return None
    try:
        return str(Path(os.path.expanduser(str(value))).resolve())
    except OSError:
        return str(value)


def store_key(value: Any) -> str | None:
    """A lease store as one key (:func:`~.node_inventory.parse_lease_store`; ``file://``
    and the bare directory are one store)."""
    from .node_inventory import InventoryError, parse_lease_store  # noqa: PLC0415

    try:
        uri = parse_lease_store(value)
    except InventoryError:
        return None
    if uri is not None and uri.startswith("file://"):
        uri = uri[len("file://"):]
    return uri


def _host(node: Node) -> str:
    return str(node.host).rsplit("@", 1)[-1].strip().lower()


def _address(node: Node) -> str | None:
    address = getattr(node, "address", None)
    if not address:
        return None
    try:
        return None if ipaddress.ip_address(address).is_loopback else str(address)
    except ValueError:
        return str(address)


def same_machine(a: Node, a_inventory: str | None, b: Node, b_inventory: str | None) -> bool:
    """One machine (module doc): one SSH host (any user), one address, or one name of one inventory."""
    if _host(a) == _host(b):
        return True
    if _address(a) is not None and _address(a) == _address(b):
        return True
    return a.name == b.name and a_inventory is not None and a_inventory == b_inventory


def own_pools(pools: Sequence[tuple[str, Mapping[str, Any]]]) -> list[OwnPool]:
    """The run's ssh pools (``(name, pool config)``; another kind, or a config its own
    checks refuse, is skipped: its pool dry run says why)."""
    from .nc.cli import slot_list  # noqa: PLC0415
    from .ssh_sampler_pool import parse_config  # noqa: PLC0415

    out = []
    for name, cfg in pools:
        provider = cfg.get("provider") if isinstance(cfg.get("provider"), Mapping) else {}
        if provider.get("kind") != "ssh":
            continue
        try:
            conf = parse_config(dict(cfg))
            slots = slot_list(dict(cfg)) if ("slots" in cfg or "accounts" in cfg) else list(range(len(conf.nodes)))
            target = int(cfg.get("target", 1))
        except (ValueError, TypeError, KeyError, OSError):
            continue
        nodes = tuple(conf.nodes[s] for s in dict.fromkeys(slots) if 0 <= int(s) < len(conf.nodes))
        out.append(OwnPool(name, nodes, target, provider.get("nodes") is not None, store_key(conf.lease_store),
                           _path(provider.get("inventory"))))
    return out


def _pool_pair(a: OwnPool, b: OwnPool) -> Finding | None:
    if a.store is None or a.store != b.store:
        return None  # a node lease is taken per node in one lease store: no common store, no wait
    shared = [n.name for n in a.nodes if any(m.name == n.name for m in b.nodes)]
    if not shared:
        return None
    union = {n.name for n in a.nodes} | {n.name for n in b.nodes}
    ways = ("Either give the pools an inventory without lease_store and split the nodes' cards with "
            "provider.devices, or put the pools on different nodes")
    if a.named and b.named:
        return Finding(ERROR, f"pools {a.name} and {b.name} both name node(s) {', '.join(shared)} in provider.nodes "
                              f"on an inventory with node leases (lease_store {a.store}): a node lease is held per "
                              f"node, so the pool that does not hold one waits for it while the other runs. {ways}")
    if a.target + b.target > len(union):
        return Finding(ERROR, f"pools {a.name} and {b.name} share node(s) {', '.join(shared)} of an inventory with "
                              f"node leases (lease_store {a.store}): their targets ({a.target} + {b.target}) are "
                              f"more than the {len(union)} node(s) the two can take, so a pool waits for a node "
                              f"the other holds. {ways}")
    return Finding(NOTE, f"pools {a.name} and {b.name} may both take node(s) {', '.join(shared)} (lease_store "
                         f"{a.store}): a pool skips a node the other holds; their targets ({a.target} + {b.target}) "
                         f"fit the {len(union)} node(s) they can take")


def _protected(pool: OwnPool, side: TrainerSide, pool_node: Node, trainer_node: Node) -> bool:
    """The pool's and the trainer's leases see each other on this node: one lease store, one name."""
    return side.store is not None and pool.store == side.store and pool_node.name == trainer_node.name


def _trainer_findings(pools: Sequence[OwnPool], side: TrainerSide,
                      quarantined: Callable[[TrainerSide], set[str]]) -> list[Finding]:
    out: list[Finding] = []
    entries = ", ".join(dict.fromkeys(side.entries))
    overlap: dict[str, list[str]] = {}  # pool -> its nodes on the trainer's (leased: may share)
    leased_nodes: set[str] = set()
    for pool in pools:
        for node in pool.nodes:
            for t, spare in [(t, False) for t in side.nodes] + [(t, True) for t in side.spares]:
                if not same_machine(node, pool.inventory, t, side.inventory):
                    continue
                what = "a trainer-spare node" if spare else "a node"
                if not _protected(pool, side, node, t):
                    out.append(Finding(ERROR, f"pool {pool.name} starts samplers on node {node.name}, which is {what} "
                                              f"of this run's trainer island {entries} ({t.name}): the trainer takes "
                                              "every card of its nodes. Put the pool on nodes with the sampler role "
                                              "only (inventory roles), or name them in provider.nodes"))
                elif spare:
                    out.append(Finding(WARN, f"pool {pool.name} may take node {node.name}, a trainer-spare node of "
                                             f"this run's trainer island {entries}: while the pool holds its lease "
                                             f"(lease_store {side.store}) it cannot replace a lost trainer node"))
                else:
                    overlap.setdefault(pool.name, []).append(node.name)
                    leased_nodes.add(node.name)
                break
    if not overlap:
        return out
    gone = quarantined(side)
    candidates = [t.name for t in side.nodes if t.name not in gone]
    by_pool = {p.name: p for p in pools}
    held = min(len(leased_nodes & set(candidates)),
               sum(min(by_pool[name].target, len(set(nodes) & set(candidates))) for name, nodes in overlap.items()))
    left = len(candidates) - held
    shown = "; ".join(f"pool {name} on {', '.join(nodes)}" for name, nodes in overlap.items())
    counted = (f"{len(candidates)} trainer node(s)" + (f" ({len(gone)} quarantined not counted)" if gone else "")
               + f", the pools may hold {held} of them, {left} left for the {side.need} node(s) the islands need")
    if left < side.need:
        out.append(Finding(ERROR, f"{shown}: trainer nodes of this run's trainer island(s) {entries} (lease_store "
                                  f"{side.store}): {counted}; the trainer would wait for a node a pool holds. Put "
                                  "the pools on nodes with the sampler role only, or lower their targets"))
    else:
        out.append(Finding(WARN, f"{shown}: trainer nodes of this run's trainer island(s) {entries}; the trainer "
                                 f"skips a node a pool holds (lease_store {side.store}): {counted}"))
    return out


def findings(pools: Sequence[tuple[str, Mapping[str, Any]]], sides: Sequence[TrainerSide] = (), *,
             quarantined: Callable[[TrainerSide], set[str]] | None = None) -> list[Finding]:
    """The checks of the module doc for the run's ``(pool name, pool config)`` against
    its trainer side ``sides`` (:func:`trainer_sides`). ``quarantined``: the quarantined
    nodes of a leased side (default :func:`lease_quarantines`; read only when a pool
    may share its nodes)."""
    own = own_pools(pools)
    out = []
    for i, a in enumerate(own):
        for b in own[i + 1:]:
            found = _pool_pair(a, b)
            if found is not None:
                out.append(found)
    for side in sides:
        out += _trainer_findings(own, side, quarantined or lease_quarantines)
    return out


def lease_quarantines(side: TrainerSide) -> set[str]:
    """The side's trainer nodes quarantined in its lease store (``node_quarantine``: any
    run's record); none when the store cannot be opened (the launch reads it again)."""
    if side.store is None:
        return set()
    from .node_lease import open_objects  # noqa: PLC0415
    from .node_quarantine import shared_quarantines  # noqa: PLC0415

    try:
        return set(shared_quarantines(open_objects(side.store), [n.name for n in side.nodes]))
    except Exception:  # noqa: BLE001 - not read here: every node counts (the launch reads it)
        return set()


def _need(entry: str, every: Sequence[Node]) -> int:
    from ..gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from .node_inventory import island_nodes  # noqa: PLC0415

    try:
        (spec,) = parse_gpu_spec(entry)
        return int(spec.num_nodes)
    except (ValueError, TypeError):  # ssh:<inventory file>: every trainer node of it
        return len(island_nodes(list(every), "trainer"))


def trainer_sides(entries: Sequence[str], inventory: str | None) -> list[TrainerSide]:
    """The trainer side of the run's ``ssh:`` island entries (``--gpu``, one per island),
    by inventory file: the nodes the islands take and the inventory's ``trainer-spare``
    nodes. Every trainer node of the inventory counts when the islands' nodes are picked
    at launch (a leased inventory, or several islands of one inventory) or the entry is
    refused there. An entry whose inventory does not load is skipped (the trainer
    launch's own dry run says why)."""
    from .node_inventory import InventoryError, island_nodes, lease_store_of, load_inventory  # noqa: PLC0415
    from .node_quarantine import SPARE_ROLE  # noqa: PLC0415
    from .preflight.cluster import resolve_inventory  # noqa: PLC0415

    sides: dict[str | None, TrainerSide] = {}
    every_of: dict[str | None, Any] = {}
    picked: dict[str | None, list[list[Node]] | None] = {}
    for entry in entries:
        rest = entry[4:]
        own_file = rest.endswith((".yaml", ".yml", ".json")) or Path(rest).expanduser().is_file()
        source = rest if own_file else inventory
        path = _path(source)
        if path not in every_of:
            try:
                every_of[path] = load_inventory(Path(str(source)).expanduser())
            except (InventoryError, OSError, ValueError, TypeError):
                every_of[path] = None
        every = every_of[path]
        if every is None:
            continue
        side = sides.setdefault(path, TrainerSide(path, store_key(lease_store_of(every))))
        side.entries.append(entry)
        side.need += _need(entry, every)
        try:
            nodes, _every = resolve_inventory(entry, inventory)
            if picked.get(path, []) is not None:
                picked.setdefault(path, []).append(list(nodes))
        except (InventoryError, OSError, ValueError):  # refused at launch, or picked there
            picked[path] = None
    for path, side in sides.items():
        every = every_of[path]
        rows = picked.get(path)
        if side.store is not None or rows is None or len(rows) > 1:  # picked at launch: any trainer node
            side.nodes = island_nodes(list(every), "trainer")
        else:
            side.nodes = list(rows[0])
        side.spares = [n for n in every if SPARE_ROLE in n.roles and n.name not in {t.name for t in side.nodes}]
    return list(sides.values())


def island_entries(argv_gpu: str | None, island_gpus: Sequence[str] = ()) -> list[str]:
    """The ``ssh:`` entries of the run's trainer islands, one per island: the trainer's
    ``--gpu`` (a fleet's compile lists every island's cluster there) and the fleet
    islands' own (each counted once, however many lists name it)."""
    from .preflight.gate import ssh_entries  # noqa: PLC0415

    counts = Counter(ssh_entries(argv_gpu))
    for entry, n in Counter(e for gpu in island_gpus for e in ssh_entries(gpu)).items():
        counts[entry] = max(counts[entry], n)
    return [e for e, n in counts.items() for _ in range(n)]
