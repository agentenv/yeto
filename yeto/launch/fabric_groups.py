"""Fabric groups of an SSH node inventory and the placement rule of multi-node islands.

A fabric group is a set of nodes in which every two nodes reach each other at
RDMA rate (a clique). ``yeto cluster probe`` (:mod:`yeto.cluster_cli`) measures
it with the cross-node steps of ``yeto preflight`` (:mod:`.preflight.cluster`):
``ib_write_bw`` rail by rail between every pair, and the NCCL collectives on node
pairs with ``NCCL_DEBUG=INFO``. Two nodes share an edge when

- their ``ib_write_bw`` check passes against a known link rate (every rail at
  least ``Requirements.ib_pass_fraction`` of its link; an unknown link rate is a
  WARN and no edge),
- their NCCL run goes over ``NET/IB``,
- and neither node failed an HCA check of the preflight (``ib_ports``, ``ib_hca``,
  ``gid``, ``dev_infiniband``) or has fewer active rails than HCAs it declares.

Each connected component is cut into cliques (:func:`cliques`): the node with the
most missing edges in it leaves (ties: the last by name) until the rest is a
clique; the nodes that left are grouped again among themselves. A node in no
clique of two is "fabric unavailable", with the reasons.

Group names are stable (:func:`assign_ids`): old names go to new groups by the
most shared nodes first; a group without an old name is named after a hash of its
fabric (the IB subnet prefixes of its HCA ports, or the IP networks of its RoCE
ports), or of its sorted node names when the fabric gives no distinct key.

Placement (:func:`pick_placement`, :func:`check_placement`): once any node of the
inventory has the ``fabric_group`` key, a multi-node island takes healthy nodes of
one group, never ``unavailable`` ones; an inventory without the key places as
before and only gets a hint.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .node_inventory import FABRIC_UNAVAILABLE, FABRIC_UNGROUPED, InventoryError, Node, parse_inventory

HINT = ("the node inventory declares no fabric_group: a multi-node island may land on nodes that do not "
        "reach each other over IB / RoCE; `yeto cluster probe` groups them (docs/OWN_CLUSTERS.md)")
UNAVAILABLE = "fabric unavailable"
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------- placement
class PlacementError(ValueError):
    """No group of the candidates holds the island's healthy nodes."""


@dataclass(frozen=True)
class Placement:
    """The nodes an island takes: ``group`` is their fabric group (None: an inventory
    without groups, or a single-node island), ``nodes`` in placement order."""

    group: str | None
    nodes: tuple[Node, ...]


def grouped(nodes: Iterable[Node]) -> bool:
    """The inventory has the ``fabric_group`` key on some node (``null`` included)."""
    return any(n.fabric_group is not None for n in nodes)


def placeable(node: Node) -> bool:
    """The node is in a group a multi-node island may take."""
    return node.fabric_group not in (None, FABRIC_UNGROUPED, FABRIC_UNAVAILABLE)


def _reasons(excluded: Iterable[str] | Mapping[str, str]) -> dict[str, str]:
    if isinstance(excluded, Mapping):
        return {str(k): str(v) for k, v in excluded.items()}
    return {str(n): "excluded" for n in excluded}


def _names(nodes: Sequence[Node], limit: int = 6) -> str:
    return ", ".join(n.name for n in nodes[:limit]) + (", ..." if len(nodes) > limit else "")


def pick_placement(candidates: Sequence[Node], need: int, *,
                   excluded: Iterable[str] | Mapping[str, str] = (),
                   inventory: Sequence[Node] | None = None) -> Placement:
    """The ``need`` nodes of ``candidates`` an island takes; raises :class:`PlacementError`.

    1. ``excluded`` nodes leave (names, or name -> reason: not healthy, quarantined,
       leased by another run);
    2. an inventory without groups (``inventory``, default ``candidates``) or a
       single-node island is one bucket in the candidates' order (as before);
    3. otherwise the free nodes of each group form a bucket (``unavailable`` and
       ungrouped nodes in none), and the bucket with the most free nodes that holds
       ``need`` wins (ties: the group name); its nodes go by name.

    A caller whose node could not be taken (a lease lost) adds it to ``excluded``
    and asks again."""
    why = _reasons(excluded)
    free = [n for n in candidates if n.name not in why]
    if need <= 1 or not grouped(candidates if inventory is None else inventory):
        if len(free) < need:
            raise PlacementError(f"the island needs {need} node(s); {len(free)} of {len(candidates)} candidate(s) "
                                 "are free" + _excluded_note(candidates, why))
        return Placement(None, tuple(free[:need]))
    buckets: dict[str, list[Node]] = {}
    for n in candidates:
        if placeable(n):
            buckets.setdefault(str(n.fabric_group), [])
            if n.name not in why:
                buckets[str(n.fabric_group)].append(n)
    fits = sorted(((g, m) for g, m in buckets.items() if len(m) >= need), key=lambda gm: (-len(gm[1]), gm[0]))
    if fits:
        group, members = fits[0]
        return Placement(group, tuple(sorted(members, key=lambda n: n.name)[:need]))
    counts = ", ".join(f"group {g} has only {len(m)}" for g, m in sorted(buckets.items())) or "no group has a node"
    parts = [f"the island needs {need} healthy nodes of one fabric group; {counts}"]
    ungrouped = [n for n in candidates if n.fabric_group in (None, FABRIC_UNGROUPED)]
    if ungrouped:
        parts.append(f"{len(ungrouped)} node(s) without fabric_group ({_names(ungrouped)})")
    unavailable = [n for n in candidates if n.fabric_group == FABRIC_UNAVAILABLE]
    if unavailable:
        parts.append(f"{len(unavailable)} node(s) fabric unavailable ({_names(unavailable)})")
    message = "; ".join(parts) + _excluded_note([n for n in candidates if placeable(n)], why)
    raise PlacementError(message + " (yeto cluster probe groups the nodes)")


def _excluded_note(nodes: Sequence[Node], why: Mapping[str, str]) -> str:
    out = [n.name for n in nodes if n.name in why]
    if not out:
        return ""
    return f"; {len(out)} not healthy or taken ({', '.join(f'{n}: {why[n]}' for n in out[:4])})"


def check_placement(nodes: Sequence[Node], need: int, *, unhealthy: Mapping[str, str] | None = None,
                    inventory: Sequence[Node] | None = None) -> tuple[bool, str, bool]:
    """Can a ``need``-node island be placed on ``nodes`` (its candidates)?

    Returns ``(ok, message, hint_only)`` (:func:`pick_placement` decides). A
    single-node island has no fabric constraint. Without any ``fabric_group`` key in
    the inventory (``inventory``, default ``nodes``) the placement is as before:
    ``(True, HINT, True)``. ``unhealthy`` (node name -> reason) removes nodes."""
    if need <= 1:
        return True, "", False
    if not grouped(nodes if inventory is None else inventory):
        return True, HINT, True
    try:
        placement = pick_placement(nodes, need, excluded=unhealthy or {}, inventory=inventory)
    except PlacementError as exc:
        return False, str(exc), False
    return True, f"fabric group {placement.group}: the island takes {_names(placement.nodes)}", False


_hinted: set[tuple[str, bool]] = set()


def placement_hint(message: str, *, log_only: bool, say: Callable[[str], None] | None = None) -> None:
    """The no-groups hint, once per process and way: one line into the launcher log.
    With ``log_only`` (a dry run, the preflight) the output stays as it was (the
    customer's is compared byte for byte): the hint goes to this module's ``logging``
    logger only, and does not use up the launcher log's line."""
    if (message, log_only) in _hinted:
        return
    _hinted.add((message, log_only))
    if log_only:
        _log.info("%s", message)
        return
    (say or (lambda m: print(m, file=sys.stderr, flush=True)))(f"[launcher] note: {message}")


# ---------------------------------------------------------------- the probe
@dataclass
class ProbeResult:
    """The groups of one probe: ``groups`` name -> sorted node names; ``unavailable``
    node -> reasons; ``edges`` the node pairs at RDMA rate; ``bad_edges`` the missing
    edges inside a connected component (what cut it into cliques)."""

    groups: dict[str, list[str]] = field(default_factory=dict)
    unavailable: dict[str, list[str]] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    pair_problems: dict[str, str] = field(default_factory=dict)
    bad_edges: list[tuple[str, str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.unavailable and not self.bad_edges

    def assignment(self) -> dict[str, str]:
        """Node name -> group name (``unavailable``: no RDMA to the others)."""
        out = {n: FABRIC_UNAVAILABLE for n in self.unavailable}
        for g, members in self.groups.items():
            out.update({n: g for n in members})
        return dict(sorted(out.items()))

    def lines(self) -> list[str]:
        out = [f"fabric group {g}: {', '.join(m)} ({len(m)} node(s))" for g, m in self.groups.items()]
        out += [f"missing edge {a}<->{b}: {why}" for a, b, why in self.bad_edges]
        for n, why in self.unavailable.items():
            out.append(f"{UNAVAILABLE}: {n}: {'; '.join(why) or 'no RDMA edge to another node'}")
        return out

    def record(self) -> dict[str, Any]:
        return {"groups": self.groups, "unavailable": self.unavailable,
                "edges": [list(e) for e in self.edges], "pair_problems": self.pair_problems,
                "bad_edges": [list(e) for e in self.bad_edges], "warnings": self.warnings}


# node checks of the preflight that explain why a node has no edge
_NODE_REASONS = ("ssh", "host_probe", "image_pull", "container", "address", "ib_hca", "ib_ports", "gid",
                 "dev_infiniband", "dev_infiniband_container", "memlock_container", "tools")
# a FAIL of any of these: the node's HCAs cannot carry the island (no edge at all)
_HCA_CHECKS = ("ib_ports", "ib_hca", "gid", "dev_infiniband")


def _over_ib(check: Any) -> bool:
    """A preflight ``nccl_transport`` check saying NCCL ran over ``NET/IB`` (a WARN without
    GPUDirect RDMA is still the IB transport; a WARN on a non-RDMA pair is NET/Socket)."""
    return check.status in ("PASS", "WARN") and check.detail.startswith("NET/IB")


def _pair_key(a: str, b: str) -> str:
    return f"{a}<->{b}"


def node_blocks(names: Sequence[str], checks: Sequence[Any], states: Sequence[Any] = ()) -> dict[str, str]:
    """The nodes that may have no edge, with the reason: a FAIL of an HCA check of the
    preflight, or fewer active rails than the HCAs the node declares (``NCCL_IB_HCA``,
    else the ports of its fabric's link layer)."""
    from .preflight.cluster import compute_ports  # noqa: PLC0415 - preflight imports the launch layer

    out: dict[str, str] = {}
    wanted = set(names)
    for c in checks:
        if c.scope in wanted and c.status == "FAIL" and c.name in _HCA_CHECKS and c.scope not in out:
            out[c.scope] = f"{c.name} FAIL: {c.detail}"
    for s in states:
        name = s.node.name
        if name not in wanted or name in out or not getattr(s, "ports", None):
            continue
        selected, missing, _ = compute_ports(s.node, s.ports)
        declared = len(selected) + len(missing)
        active = sum(1 for p in selected if p.active)
        if declared and active < declared:
            out[name] = f"{active} of {declared} declared HCA rails active"
    return out


def _link_unknown(check: Any) -> bool:
    link = (getattr(check, "data", None) or {}).get("link_gbps")
    return check.detail.endswith("(link rate unknown)") or (link is not None and any(not lk for lk in link))


def evaluate_pairs(names: Sequence[str], checks: Sequence[Any], *, nccl_pairs: str = "all",
                   blocked: Mapping[str, str] | None = None
                   ) -> tuple[list[tuple[str, str]], dict[str, str], list[str]]:
    """The edges among ``names`` from the preflight ``checks``: ``ib_write_bw`` PASS against
    a known link rate and the pair's NCCL transport ``NET/IB``, neither node ``blocked``.
    ``nccl_pairs`` ``disjoint``: a pair without its own NCCL run counts when both nodes ran
    NCCL over ``NET/IB`` with another node. Returns ``(edges, problem of every other pair,
    warnings)``."""
    blocked = blocked or {}

    def find(name: str, scopes: set[str]) -> list[Any]:
        return [c for c in checks if c.name == name and c.scope in scopes]

    nccl: dict[tuple[str, str], tuple[bool, str] | None] = {}
    ib_ok: set[str] = set()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            scopes = {f"{a}+{b}", f"{b}+{a}"}
            transport, run = find("nccl_transport", scopes), [c for c in find("nccl", scopes) if c.status == "FAIL"]
            if not transport and not run:
                nccl[(a, b)] = None
            elif run:
                nccl[(a, b)] = (False, f"NCCL: {run[0].detail}")
            elif not _over_ib(transport[0]):
                nccl[(a, b)] = (False, f"NCCL: {transport[0].detail}")
            else:
                nccl[(a, b)] = (True, "")
                ib_ok |= {a, b}
    edges, problems, warns = [], {}, []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            key = _pair_key(a, b)
            if a in blocked or b in blocked:
                n = a if a in blocked else b
                problems[key] = f"{n}: {blocked[n]}"
                continue
            ib = find("ib_write_bw", {key, _pair_key(b, a)})
            if not ib:
                problems[key] = "no ib_write_bw result"
                continue
            if ib[0].status != "PASS":
                problems[key] = f"ib_write_bw {ib[0].status}: {ib[0].detail}"
                continue
            if _link_unknown(ib[0]):
                problems[key] = "ib_write_bw: the link rate is unknown, the rails cannot be judged"
                warns.append(f"[WARN] {key} ib_write_bw: link rate unknown (no edge): {ib[0].detail}")
                continue
            verdict = nccl[(a, b)]
            if verdict is None:
                if nccl_pairs == "disjoint" and a in ib_ok and b in ib_ok:
                    edges.append((a, b))
                else:
                    problems[key] = "no NCCL result"
                continue
            if verdict[0]:
                edges.append((a, b))
            else:
                problems[key] = verdict[1]
    return edges, problems, warns


def components(names: Sequence[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """Connected components (sorted node names), largest first, then by first name."""
    parent = {n: n for n in names}

    def root(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in edges:
        if a not in parent or b not in parent:
            continue
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(root(n), []).append(n)
    return sorted((sorted(c) for c in out.values()), key=lambda c: (-len(c), c[0]))


def cliques(nodes: Sequence[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    """``nodes`` (one connected component) cut into cliques: the node with the most
    missing edges leaves (ties: the last by name) until the rest is a clique; the
    nodes that left are cut the same way among themselves. Largest first."""
    edge = {frozenset(e) for e in edges}
    rest, left = sorted(nodes), []
    while True:
        bad = {n: sum(1 for m in rest if m != n and frozenset((n, m)) not in edge) for n in rest}
        if not any(bad.values()):
            break
        worst = max(bad.values())
        drop = max(n for n in rest if bad[n] == worst)
        rest.remove(drop)
        left.append(drop)
    out = [rest] if rest else []
    if left:
        sub = [tuple(e) for e in edge if e <= set(left)]
        for comp in components(sorted(left), sub):
            out += cliques(comp, sub)
    return sorted(out, key=lambda c: (-len(c), c[0]))


def _digest(text: str) -> str:
    return "fg-" + hashlib.sha256(text.encode()).hexdigest()[:8]


def _old_name(value: str | None) -> bool:
    return value not in (None, FABRIC_UNGROUPED, FABRIC_UNAVAILABLE)


def assign_ids(comps: Sequence[Sequence[str]], *, keys: Mapping[str, Iterable[str]] | None = None,
               previous: Mapping[str, str | None] | None = None,
               reserved: Iterable[str] = ()) -> list[str]:
    """A stable name for every group (``comps``, in order).

    1. old names (``previous``: the inventory's ``fabric_group``) go to groups by the
       number of nodes they share, most first over every (old name, group) pair (ties:
       the old name, then the group's order), each old name and each group once;
    2. ``fg-`` + a hash of its fabric keys (``keys``: node -> IB subnet prefixes or RoCE
       networks), when no other group has the same keys;
    3. ``fg-`` + a hash of its sorted node names.

    ``reserved``: names held by inventory nodes outside the probe; only a group whose
    nodes carry one may keep it."""
    previous, keys = previous or {}, keys or {}
    overlap: dict[tuple[str, int], int] = {}
    for i, comp in enumerate(comps):
        for n in comp:
            if _old_name(previous.get(n)):
                k = (str(previous[n]), i)
                overlap[k] = overlap.get(k, 0) + 1
    taken: set[str] = set()
    out: list[str | None] = [None] * len(comps)
    for (name, i), _ in sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])):
        if out[i] is None and name not in taken:
            out[i] = name
            taken.add(name)
    blocked = {r for r in reserved if _old_name(r)} | taken | {FABRIC_UNAVAILABLE}
    fabric = {i: "|".join(sorted({k for n in comp for k in keys.get(n, ())})) for i, comp in enumerate(comps)}
    shared: dict[str, int] = {}
    for v in fabric.values():
        if v:
            shared[v] = shared.get(v, 0) + 1
    for i, comp in enumerate(comps):
        if out[i] is not None:
            continue
        name = _digest(fabric[i]) if fabric[i] and shared[fabric[i]] == 1 else None
        if name is None or name in blocked:
            name = _digest("nodes:" + ",".join(sorted(comp)))
        base, k = name, 2
        while name in blocked:
            name, k = f"{base}-{k}", k + 1
        out[i] = name
        blocked.add(name)
    return [str(n) for n in out]


_INET = re.compile(r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+/\d+)")


def fabric_keys(state: Any) -> set[str]:
    """What names the fabric of a probed node (a preflight ``NodeState``): the IB
    subnet prefixes (GID 0) of its compute ports, or the IP networks of its RoCE
    ports. Empty when the probe did not read them."""
    cas = {p.ca for p in getattr(state, "compute", ()) or ()}
    gids = [g for g in getattr(state, "gids", ()) or () if g.dev in cas]
    if getattr(state.node, "fabric", "") == "ib":
        return {"ib:" + ":".join(g.gid.split(":")[:4]) for g in gids if g.index == 0 and len(g.gid.split(":")) == 8}
    addr = (getattr(state, "host", {}) or {}).get("addr")
    nets: dict[str, str] = {}
    for line in (addr.text if addr is not None else "").splitlines():
        m = _INET.match(line.strip())
        if m:
            nets[m.group(1)] = str(ipaddress.ip_interface(m.group(2)).network)
    return {"roce:" + nets[g.ndev] for g in gids if g.ipv4 and g.ndev in nets}


def group_nodes(names: Sequence[str], checks: Sequence[Any], *, states: Sequence[Any] = (),
                previous: Mapping[str, str | None] | None = None, reserved: Iterable[str] = (),
                nccl_pairs: str = "all") -> ProbeResult:
    """The fabric groups of ``names`` from a preflight's ``checks`` (and its node
    ``states``: the declared rails, the fabric keys)."""
    blocked = node_blocks(names, checks, states)
    edges, problems, warns = evaluate_pairs(names, checks, nccl_pairs=nccl_pairs, blocked=blocked)
    edge = {frozenset(e) for e in edges}
    parts: list[list[str]] = []
    bad: list[tuple[str, str, str]] = []
    for comp in components(names, edges):
        parts += cliques(comp, edges)
        bad += [(a, b, problems.get(_pair_key(a, b), "no edge")) for i, a in enumerate(comp) for b in comp[i + 1:]
                if frozenset((a, b)) not in edge]
    parts = sorted(parts, key=lambda c: (-len(c), c[0]))
    multi = [c for c in parts if len(c) > 1]
    keys = {s.node.name: fabric_keys(s) for s in states}
    ids = assign_ids(multi, keys=keys, previous=previous, reserved=reserved)
    result = ProbeResult(groups=dict(sorted(zip(ids, multi))), edges=edges, pair_problems=problems,
                         bad_edges=bad, warnings=warns)
    for part in parts:
        if len(part) > 1:
            continue
        (n,) = part
        why = [f"{c.name}: {c.detail}" for c in checks
               if c.scope == n and c.status == "FAIL" and c.name in _NODE_REASONS]
        if n in blocked and not any(w.startswith(blocked[n].split(" FAIL")[0]) for w in why):
            why.append(blocked[n])
        for pair, problem in problems.items():
            a, _, b = pair.partition("<->")
            if n in (a, b) and not problem.startswith(f"{n}: "):
                why.append(f"to {b if n == a else a}: {problem}")
        result.unavailable[n] = list(dict.fromkeys(why))[:4]
    result.unavailable = dict(sorted(result.unavailable.items()))
    return result


# ---------------------------------------------------------- writing it back
def _entries(document: Any) -> list[Any]:
    return document if isinstance(document, list) else (document or {}).get("nodes") or []


def _effective(document: Any) -> dict[str, Any]:
    return {(e.get("name") or f"node{i}"): e.get("fabric_group") for i, e in enumerate(_entries(document))
            if isinstance(e, dict)}


def _stripped(document: Any) -> Any:
    doc = copy.deepcopy(document)
    for e in _entries(doc):
        if isinstance(e, dict):
            e.pop("fabric_group", None)
    return doc


def _scalar(value: str | None) -> str:
    import yaml  # noqa: PLC0415

    if value is None:
        return "null"
    return value if yaml.safe_load(value) == value else json.dumps(value)


_MERGE = "tag:yaml.org,2002:merge"


def _yaml_edits(text: str, todo: Mapping[str, str | None]) -> str:
    """Set ``fabric_group`` of the entries named in ``todo`` in the YAML ``text``, in
    place: a value is replaced, a missing key is added under the entry's ``name``
    (a block mapping) or before its closing brace (a flow mapping).

    An entry other entries merge (``<<: *anchor``) is edited alone first; every entry
    that merges it and is not in ``todo`` gets its current value written on itself, so
    it keeps it. The caller reads the file again and edits the entries whose merged
    value is still not what it wants."""
    import yaml  # noqa: PLC0415

    root = yaml.compose(text, Loader=yaml.SafeLoader)
    seq = root
    if isinstance(root, yaml.MappingNode):
        seq = next((v for k, v in root.value if getattr(k, "value", None) == "nodes"), None)
    if not isinstance(seq, yaml.SequenceNode):
        raise InventoryError("the inventory has no nodes: list to write fabric_group into")
    current = _effective(yaml.safe_load(text))
    items: list[tuple[str, Any]] = []
    for index, item in enumerate(seq.value):
        if isinstance(item, yaml.MappingNode):
            own = {getattr(k, "value", None): v for k, v in item.value}
            name = own["name"].value if isinstance(own.get("name"), yaml.ScalarNode) else f"node{index}"
            items.append((name, item))
    mergers: dict[int, list[tuple[str, Any]]] = {}
    for name, item in items:
        for k, v in item.value:
            if getattr(k, "tag", "") == _MERGE:
                for src in (v.value if isinstance(v, yaml.SequenceNode) else [v]):
                    mergers.setdefault(id(src), []).append((name, item))

    def edit(name: str, item: Any, value: str | None) -> tuple[int, int, str]:
        own = {getattr(k, "value", None): (k, v) for k, v in item.value}
        text_value = _scalar(value)
        if "fabric_group" in own:
            v = own["fabric_group"][1]
            return v.start_mark.index, v.end_mark.index, text_value
        if item.flow_style:
            end = item.end_mark.index - 1
            if text[end] != "}":
                raise InventoryError(f"node {name}: cannot find the end of its flow mapping")
            at = end
            while at > 0 and text[at - 1] in " \t\n":
                at -= 1
            return at, at, f"{', ' if item.value else ''}fabric_group: {text_value}"
        k, v = own.get("name") or item.value[0]
        nl = text.find("\n", v.end_mark.index)
        at = len(text) if nl < 0 else nl + 1
        return at, at, ("" if nl >= 0 else "\n") + " " * k.start_mark.column + f"fabric_group: {text_value}\n"

    sources = [(n, it) for n, it in items if n in todo and id(it) in mergers]
    if sources:
        edits = []
        pinned: set[str] = set()
        for name, item in sources:
            edits.append(edit(name, item, todo[name]))
            for other, merger in mergers[id(item)]:
                own = {getattr(k, "value", None) for k, _ in merger.value}
                if other not in todo and other not in pinned and "fabric_group" not in own \
                        and current.get(other) != todo[name]:
                    edits.append(edit(other, merger, current.get(other)))
                    pinned.add(other)
    else:
        edits = [edit(n, it, todo[n]) for n, it in items if n in todo]
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def updated_text(text: str, groups: Mapping[str, str | None], *, json_file: bool = False) -> str:
    """The inventory ``text`` with ``fabric_group`` of ``groups`` (node -> name), every
    other byte kept (YAML; a JSON file is rewritten). Verified: the result parses to
    the same inventory except those fields, and every other node keeps its value."""
    import yaml  # noqa: PLC0415

    original = yaml.safe_load(text)
    unknown = sorted(set(groups) - set(_effective(original)))
    if unknown:
        raise InventoryError(f"nodes {unknown} are not in the inventory")
    if json_file:
        doc = json.loads(text)
        for i, e in enumerate(_entries(doc)):
            name = e.get("name") or f"node{i}"
            if name in groups and (groups[name] is not None or "fabric_group" in e):
                e["fabric_group"] = groups[name]
        new = json.dumps(doc, indent=2) + "\n"
    else:
        new = text
        for _ in range(4):  # an edit of a merged anchor reaches the entries that merge it
            current = _effective(yaml.safe_load(new))
            todo = {n: v for n, v in groups.items() if current.get(n) != v}
            if not todo:
                break
            new = _yaml_edits(new, todo)
    document = yaml.safe_load(new)
    want = {**_effective(original), **groups}
    if _effective(document) != want or _stripped(document) != _stripped(original):
        raise InventoryError("could not write fabric_group in place (the edited file reads differently); set it by "
                             f"hand: {json.dumps(dict(groups))}")
    parse_inventory(document)
    return new


def write_groups(path: str | Path, groups: Mapping[str, str | None]) -> bool:
    """Write ``groups`` into the inventory file; False when the file already says so.

    Atomic: a temporary file next to the target (a symlink's target, the link kept)
    is written, synced to disk and renamed over it, with the file's mode and, where
    allowed, its owner; the directory is synced after the rename."""
    file = Path(os.path.realpath(path))
    text = file.read_text()
    new = updated_text(text, groups, json_file=file.suffix == ".json")
    if new == text:
        return False
    st = file.stat()
    fd, tmp = tempfile.mkstemp(dir=file.parent, prefix=f".{file.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, st.st_mode & 0o7777)
        try:
            os.chown(tmp, st.st_uid, st.st_gid)
        except OSError:
            pass  # another owner needs privileges: the file is the writer's now
        os.replace(tmp, file)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    try:
        dfd = os.open(file.parent, os.O_RDONLY)
    except OSError:
        return True
    try:
        os.fsync(dfd)
    except OSError:
        pass
    finally:
        os.close(dfd)
    return True
