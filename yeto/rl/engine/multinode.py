"""Node-aware island topology (rl-multinode-island D2-D5, D7, D8). Pure Python.

A logical bundle ``p`` of a ``gpus_per_node = G`` island lives on node
``p // G`` as local GPU ``p % G`` (design D3; Miles' PACK placement group
sorted by (node, gpu) yields exactly this blocking, asserted at startup by
``StartupBundles``). Everything here is a plain function so the launcher, the
resources-manifest parser and the placement port share one rule set.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_SLOT_RE = re.compile(r"^n(?P<node>\d+):(?P<gpu>\d+)$")


class TopologyError(ValueError):
    """Illegal node topology or a placement entry that cannot be resolved."""


@dataclass(frozen=True)
class Topology:
    nodes: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        for name in ("nodes", "gpus_per_node"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise TopologyError(f"topology {name} must be a positive integer")

    @property
    def total(self) -> int:
        return self.nodes * self.gpus_per_node

    def node_of(self, bundle: int) -> int:
        if isinstance(bundle, bool) or not isinstance(bundle, int) or not 0 <= bundle < self.total:
            raise TopologyError(f"logical bundle {bundle!r} outside 0..{self.total - 1}")
        return bundle // self.gpus_per_node

    def slot_of(self, bundle: int) -> tuple[int, int]:
        return self.node_of(bundle), bundle % self.gpus_per_node

    def bundle_of(self, node: int, gpu: int) -> int:
        if not 0 <= node < self.nodes or not 0 <= gpu < self.gpus_per_node:
            raise TopologyError(f"slot n{node}:{gpu} outside a {self.nodes}x{self.gpus_per_node} island")
        return node * self.gpus_per_node + gpu


def topology_of(resources: Mapping[str, Any]) -> Topology | None:
    """``Topology`` from a resources manifest's ``nodes``/``gpus_per_node``, or
    None when the manifest declares no node layout (legacy single-node cfg:
    behaviour unchanged)."""
    if "nodes" not in resources and "gpus_per_node" not in resources:
        return None
    nodes = resources.get("nodes", 1)
    per = resources.get("gpus_per_node")
    if per is None:
        gpus = resources.get("gpus") or []
        if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes < 1 or not gpus \
                or len(gpus) % nodes:
            raise TopologyError("resources.gpus_per_node is required when the GPU pool "
                                "is not an exact multiple of resources.nodes")
        per = len(gpus) // nodes
    try:
        return Topology(nodes, per)
    except TopologyError as exc:
        raise TopologyError(f"resources.{exc}") from None


def check_pool_topology(resources: Mapping[str, Any], topology: Topology) -> None:
    """A resolved ``resources.gpus`` pool must cover exactly nodes x gpus_per_node
    with each node's ``index`` running 0..G-1 (design D2)."""
    gpus = resources.get("gpus") or []
    if not gpus:
        return
    if len(gpus) != topology.total:
        raise TopologyError(f"resources.gpus lists {len(gpus)} GPUs but nodes x gpus_per_node "
                            f"= {topology.nodes} x {topology.gpus_per_node} = {topology.total}")
    by_node: dict[Any, list[int]] = {}
    for gpu in gpus:
        if "node" not in gpu or "index" not in gpu:
            raise TopologyError(f"GPU {gpu.get('uuid')!r} needs node and index on a multi-node pool")
        by_node.setdefault(gpu["node"], []).append(gpu["index"])
    if len(by_node) != topology.nodes:
        raise TopologyError(f"resources.gpus names {len(by_node)} nodes, topology has {topology.nodes}")
    want = list(range(topology.gpus_per_node))
    for node, idx in by_node.items():
        if sorted(idx) != want:
            raise TopologyError(f"node {node!r} GPU indices {sorted(idx)} are not 0..{topology.gpus_per_node - 1}")


def resolve_slot(entry: Any, topology: Topology, pool: Mapping[str, Mapping[str, Any]] | None = None,
                 ) -> tuple[int, int]:
    """One placement entry -> ``(node, local_gpu)``. Three spellings: ``"n{k}:{g}"``,
    a logical bundle int, or a pool GPU uuid (resolved through ``pool[uuid]['node'/'index']``,
    nodes taken in sorted order of their names)."""
    if isinstance(entry, bool):
        raise TopologyError(f"placement entry {entry!r} is not a slot")
    if isinstance(entry, int):
        return topology.slot_of(entry)
    if not isinstance(entry, str):
        raise TopologyError(f"placement entry {entry!r} is not a slot")
    m = _SLOT_RE.match(entry)
    if m:
        node, gpu = int(m.group("node")), int(m.group("gpu"))
        topology.bundle_of(node, gpu)
        return node, gpu
    if not pool or entry not in pool:
        raise TopologyError(f"placement entry {entry!r} is neither n<k>:<g> nor a pool GPU uuid")
    names = sorted({str(g.get("node")) for g in pool.values()})
    gpu = pool[entry]
    if "node" not in gpu or "index" not in gpu:
        raise TopologyError(f"pool GPU {entry!r} has no node/index")
    node = names.index(str(gpu["node"]))
    topology.bundle_of(node, int(gpu["index"]))
    return node, int(gpu["index"])


def _spelling(entry: Any) -> str:
    if isinstance(entry, int) and not isinstance(entry, bool):
        return "bundle"
    if isinstance(entry, str) and _SLOT_RE.match(entry):
        return "slot"
    return "uuid"


def normalize_placement(placement: Mapping[str, Any], topology: Topology,
                        pool: Mapping[str, Mapping[str, Any]] | None = None,
                        ) -> dict[str, Any]:
    """``{"trainer": [slot], "rollout": [[slot]], "standby": [slot]}`` with every
    entry as ``(node, local_gpu)``. Mixed spellings in one placement are refused."""
    flat: list[Any] = list(placement.get("trainer", ())) + list(placement.get("standby", ()))
    for engine in placement.get("rollout", ()):
        flat += list(engine)
    spellings = {_spelling(e) for e in flat}
    if len(spellings) > 1:
        raise TopologyError(f"placement mixes spellings {sorted(spellings)}; use one of "
                            "n<k>:<g>, logical bundle ints or pool uuids")
    out = {
        "trainer": [resolve_slot(e, topology, pool) for e in placement.get("trainer", ())],
        "rollout": [[resolve_slot(e, topology, pool) for e in engine]
                    for engine in placement.get("rollout", ())],
        "standby": [resolve_slot(e, topology, pool) for e in placement.get("standby", ())],
    }
    return out


def spans_nodes(slots: Iterable[tuple[int, int]]) -> bool:
    return len({node for node, _ in slots}) > 1


def rectangular_trainer(slots: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """``(nodes, gpus_per_node)`` the trainer occupies, or TopologyError when the
    per-node counts differ (Miles needs actor_num_nodes x actor_num_gpus_per_node)."""
    if not slots:
        raise TopologyError("trainer has no GPUs")
    counts: dict[int, int] = {}
    for node, _ in slots:
        counts[node] = counts.get(node, 0) + 1
    sizes = set(counts.values())
    if len(sizes) != 1:
        raise TopologyError("trainer GPUs per node must be equal on every node it uses, got "
                            + ", ".join(f"n{n}:{c}" for n, c in sorted(counts.items())))
    return len(counts), sizes.pop()


def node_placement_rejection(slots: Mapping[str, Any], *, model_parallel: int = 1,
                             expert_parallel: int = 1, gpus_per_engine: int | None = None,
                             ) -> str | None:
    """Design D4 rules on a normalized placement; None when legal.

    1. each rollout engine on one node; 2. each trainer model-parallel group
    (tp*pp*cp consecutive trainer slots) on one node; 3. EP groups whole-node
    aligned (Q1: EP does not span nodes -> an EP group of model-parallel groups
    must fit inside one node); 5. the trainer occupies a rectangle."""
    for engine in slots.get("rollout", ()):
        if gpus_per_engine is not None and len(engine) != gpus_per_engine:
            return f"rollout engine {engine} does not have {gpus_per_engine} GPUs"
        if spans_nodes(engine):
            return f"rollout engine {engine} spans nodes"
    trainer = list(slots.get("trainer", ()))
    mp = max(1, int(model_parallel))
    if len(trainer) % mp:
        return f"trainer GPUs {len(trainer)} not divisible by model parallel {mp}"
    for start in range(0, len(trainer), mp):
        group = trainer[start:start + mp]
        if spans_nodes(group):
            return f"trainer model-parallel group {group} spans nodes"
    try:
        _nodes, per_node = rectangular_trainer(trainer) if trainer else (0, 0)
    except TopologyError as exc:
        return str(exc)
    ep = max(1, int(expert_parallel))
    if ep > 1 and trainer:
        groups_per_node = per_node // mp
        if groups_per_node == 0 or groups_per_node % ep:
            return (f"expert parallel {ep} must divide the {groups_per_node} model-parallel "
                    "groups on one node (EP does not span nodes)")
    return None


def chunk_by_node(bundles: Sequence[int], topology: Topology | None, per: int,
                  ) -> tuple[list[list[int]], list[int]]:
    """Split a role's logical bundles into engine runs of ``per`` that never
    cross a node (design D7). Returns ``(runs, leftover)``; without a topology
    this is the legacy consecutive split."""
    bundles = list(bundles)
    if topology is None:
        runs = [bundles[i:i + per] for i in range(0, len(bundles) - per + 1, per)]
        rest = bundles[len(runs) * per:]
        return runs, rest
    runs: list[list[int]] = []
    rest: list[int] = []
    block: list[int] = []
    current: int | None = None
    for b in bundles:
        node = topology.node_of(b)
        if node != current and block:
            rest += block
            block = []
        current = node
        block.append(b)
        if len(block) == per:
            runs.append(block)
            block = []
    rest += block
    return runs, rest


def min_nodes(*, trainer_min_gpus: int, rollout_min_gpus: int, standby_gpus: int,
              gpus_per_node: int) -> int:
    """Design D8: the smallest island (whole nodes) that fits one trainer model
    replica, one rollout engine and the standby reservation, with the trainer
    and the engine each on whole-node-aligned slots."""
    for name, value in (("trainer_min_gpus", trainer_min_gpus), ("rollout_min_gpus", rollout_min_gpus),
                        ("gpus_per_node", gpus_per_node)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TopologyError(f"{name} must be a positive integer")
    if standby_gpus < 0:
        raise TopologyError("standby_gpus must be non-negative")
    if trainer_min_gpus > gpus_per_node and trainer_min_gpus % gpus_per_node:
        raise TopologyError(f"trainer replica {trainer_min_gpus} GPUs is not a whole number "
                            f"of {gpus_per_node}-GPU nodes")
    if rollout_min_gpus > gpus_per_node:
        raise TopologyError(f"rollout engine {rollout_min_gpus} GPUs does not fit a "
                            f"{gpus_per_node}-GPU node")
    return max(1, math.ceil((trainer_min_gpus + rollout_min_gpus + standby_gpus) / gpus_per_node))


def assert_node_blocks(node_ids: Sequence[Any], gpus_per_node: int) -> tuple[Any, ...]:
    """Design D3 startup assertion: the placement group's bundle -> node sequence
    (logical order) must be whole blocks of ``gpus_per_node`` with one distinct
    node per block. Returns the node id per block."""
    ids = list(node_ids)
    if not ids or len(ids) % gpus_per_node:
        raise TopologyError(f"{len(ids)} bundles are not whole {gpus_per_node}-GPU nodes")
    blocks = []
    for start in range(0, len(ids), gpus_per_node):
        block = ids[start:start + gpus_per_node]
        if len(set(block)) != 1:
            raise TopologyError(f"bundles {start}..{start + gpus_per_node - 1} span nodes {sorted(set(map(str, block)))}")
        blocks.append(block[0])
    if len(set(map(str, blocks))) != len(blocks):
        raise TopologyError(f"a node repeats across blocks: {blocks}")
    return tuple(blocks)
