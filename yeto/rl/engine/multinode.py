"""Node-aware island topology (rl-multinode-island D2-D5, D7, D8). Pure Python.

A logical bundle ``p`` of a ``gpus_per_node = G`` island lives on node
``p // G`` as local GPU ``p % G`` (design D3; Miles' PACK placement group
sorted by (node, gpu) yields exactly this blocking, asserted at startup by
``StartupBundles``). Node 0 is the Ray head (D3 head pin, 2026-10-03 ruling):
block 0 of the placement group requests a sliver of the head's node resource
(:func:`head_pinned_bundles`), the fork's (node, gpu) sort puts the head first
(:func:`head_first_sort_key`), and :func:`assert_head_block` rejects any other
outcome at startup (fail closed; Miles' plain IP sort put the trainer on the
worker in the 2x1 L40S run, run.log bundle lines of s1-mn-20261003g). Everything here is a plain function so the launcher, the
resources-manifest parser and the placement port share one rule set.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_SLOT_RE = re.compile(r"^n(?P<node>\d+):(?P<gpu>\d+)$")

# Q1/Q3 user ruling 2026-10-04: one learner spans several nodes with several
# GPUs each; the trainer's EP and PP groups MAY cross nodes, TP (x CP) stays
# inside a node by default. The in-node group is therefore ``tp*cp`` (Megatron
# rank order tp-cp-ep-dp-pp puts TP/CP innermost, so consecutive trainer ranks
# form the TP*CP group), and only that group is checked against node borders.
CROSS_NODE_DIMS: frozenset[str] = frozenset({"ep", "pp"})
IN_NODE_DIMS: tuple[str, ...] = ("tp", "cp")


def node_parallel_of(dims: Mapping[str, int]) -> int:
    """Size of the in-node trainer group (``tp*cp``) from a parallel dims map."""
    out = 1
    for dim in IN_NODE_DIMS:
        out *= max(1, int(dims.get(dim, 1) or 1))
    return out


def trainer_replica_gpus(dims: Mapping[str, int]) -> int:
    """GPUs of the smallest trainer model replica: ``tp*cp*ep*pp`` (D8 min_nodes)."""
    out = 1
    for dim in ("tp", "cp", "ep", "pp"):
        out *= max(1, int(dims.get(dim, 1) or 1))
    return out


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


def node_placement_rejection(slots: Mapping[str, Any], *, node_parallel: int | None = None,
                             model_parallel: int = 1, expert_parallel: int = 1,
                             gpus_per_engine: int | None = None) -> str | None:
    """Design D4 rules on a normalized placement; None when legal.

    1. each rollout engine on one node; 2. each trainer in-node group
    (``node_parallel`` = tp*cp consecutive trainer slots; ``model_parallel`` is
    the pre-2026-10-04 spelling, used when ``node_parallel`` is None) on one
    node and dividing the trainer GPUs on every node; 3. EP and PP groups MAY
    span nodes (Q1/Q3 ruling) -- EP only needs ``tp*cp*ep`` to divide the
    trainer GPUs; 5. the trainer occupies a rectangle (Miles derives local_rank
    from RANK % gpus_per_node)."""
    for engine in slots.get("rollout", ()):
        if gpus_per_engine is not None and len(engine) != gpus_per_engine:
            return f"rollout engine {engine} does not have {gpus_per_engine} GPUs"
        if spans_nodes(engine):
            return f"rollout engine {engine} spans nodes"
    trainer = list(slots.get("trainer", ()))
    np_ = max(1, int(model_parallel if node_parallel is None else node_parallel))
    if len(trainer) % np_:
        return f"trainer GPUs {len(trainer)} not divisible by in-node parallel tp*cp {np_}"
    for start in range(0, len(trainer), np_):
        group = trainer[start:start + np_]
        if spans_nodes(group):
            return f"trainer in-node (tp*cp) group {group} spans nodes"
    try:
        _nodes, per_node = rectangular_trainer(trainer) if trainer else (0, 0)
    except TopologyError as exc:
        return str(exc)
    if trainer and per_node % np_:
        return (f"in-node parallel tp*cp {np_} does not divide the {per_node} trainer GPUs "
                "on one node")
    ep = max(1, int(expert_parallel))
    if ep > 1 and trainer and len(trainer) % (np_ * ep):
        return (f"expert parallel {ep} needs trainer GPUs divisible by tp*cp*ep = {np_ * ep}, "
                f"got {len(trainer)} (EP groups may span nodes)")
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
              gpus_per_node: int, node_parallel: int | None = None) -> int:
    """Design D8: the smallest island (whole nodes) that fits one trainer model
    replica (``trainer_min_gpus`` = tp*cp*ep*pp, which may span nodes when it
    exceeds a node: Q1/Q3 ruling), one rollout engine and the standby
    reservation. A replica larger than a node must be a whole number of nodes;
    the in-node group ``node_parallel`` (tp*cp) must fit and divide a node."""
    for name, value in (("trainer_min_gpus", trainer_min_gpus), ("rollout_min_gpus", rollout_min_gpus),
                        ("gpus_per_node", gpus_per_node)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TopologyError(f"{name} must be a positive integer")
    if standby_gpus < 0:
        raise TopologyError("standby_gpus must be non-negative")
    if node_parallel is not None:
        if isinstance(node_parallel, bool) or not isinstance(node_parallel, int) or node_parallel < 1:
            raise TopologyError("node_parallel must be a positive integer")
        if node_parallel > gpus_per_node or gpus_per_node % node_parallel:
            raise TopologyError(f"in-node parallel tp*cp {node_parallel} must fit and divide a "
                                f"{gpus_per_node}-GPU node (TP stays inside a node)")
        if trainer_min_gpus % node_parallel:
            raise TopologyError(f"trainer replica {trainer_min_gpus} GPUs is not a multiple of "
                                f"tp*cp {node_parallel}")
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


HEAD_RESOURCE = "node:__internal_head__"  # Ray's built-in resource label present only on the head node
HEAD_SHARE = 0.001  # per bundle; gpus_per_node * HEAD_SHARE must stay <= the label's quantity (1.0)


def head_pinned_bundles(num_gpus: int, gpus_per_node: int, *, head_resource: str = HEAD_RESOURCE,
                        share: float = HEAD_SHARE) -> list[dict[str, float]]:
    """Startup placement-group bundles with logical block 0 pinned to the Ray
    head: the first ``gpus_per_node`` bundles also request ``share`` of
    ``head_resource`` so Ray cannot place them anywhere else (D3 head pin)."""
    if gpus_per_node <= 0 or num_gpus <= 0 or num_gpus % gpus_per_node:
        raise TopologyError(f"{num_gpus} GPUs are not whole {gpus_per_node}-GPU nodes")
    if gpus_per_node * share > 1.0:
        raise TopologyError(f"{gpus_per_node} bundles x {share} exceed the head label quantity 1.0")
    bundles: list[dict[str, float]] = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    for bundle in bundles[:gpus_per_node]:
        bundle[head_resource] = share
    return bundles


def head_first_sort_key(head_ip: Any, base_key: Any) -> Any:
    """Wrap the fork's ``sort_key`` ((index, node_ip, gpu) -> key) so the head
    node's bundles sort first, i.e. become logical node 0."""
    head = str(head_ip)

    def key(entry: Any) -> tuple[int, Any]:
        _index, node, _gpu = entry
        return (0 if str(node) == head else 1, base_key(entry))

    return key


def assert_head_block(node_blocks: Sequence[Any], head_node: Any) -> None:
    """D3 head pin: logical node 0 (block 0 of :func:`assert_node_blocks`) must be
    the Ray head node; anything else is a startup error (fail closed)."""
    blocks = list(node_blocks)
    if not blocks:
        raise TopologyError("no node blocks")
    if str(blocks[0]) != str(head_node):
        raise TopologyError(f"logical node 0 is {blocks[0]!r}, not the Ray head {head_node!r} "
                            "(D3: head = node 0 = trainer)")


# ------------------------------------------------------------ GPU UUID reconciliation (Q6)
@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of :func:`reconcile_gpu_pool` (rl-multinode-island Q6, ruling 2026-10-04).

    ``observed`` is the runtime pool per logical node (uuid tuples, local index order)
    and becomes the journal's binding baseline; ``rebind`` says the declared uuids were
    replaced (``mapping`` old -> new, only with ``accept_rebind``); ``diffs`` lists every
    slot whose uuid differs; ``error`` is why the pool is refused (None when accepted)."""

    ok: bool
    observed: tuple[tuple[str, ...], ...]
    rebind: bool = False
    mapping: dict[str, str] = field(default_factory=dict)
    diffs: tuple[str, ...] = ()
    error: str | None = None

    @property
    def flat(self) -> tuple[str, ...]:
        """Observed uuids in logical bundle order (node-major)."""
        return tuple(u for node in self.observed for u in node)


def declared_gpu_pool(resources: Mapping[str, Any], topology: Topology) -> tuple[tuple[str | None, ...], ...]:
    """cfg ``resources.gpus`` -> per-node uuid tuples (None for a slot the cfg does not
    name, i.e. ``n{k}:{g}``/integer spellings or an empty pool). Shape follows the topology;
    :func:`check_pool_topology` has already rejected pools of another shape."""
    pool: list[list[str | None]] = [[None] * topology.gpus_per_node for _ in range(topology.nodes)]
    for gpu in resources.get("gpus") or []:
        uuid = gpu.get("uuid")
        node, index = gpu.get("node"), gpu.get("index")
        if not isinstance(uuid, str) or not uuid or node is None or index is None:
            continue
        try:
            pool[int(node)][int(index)] = uuid
        except (IndexError, ValueError, TypeError):
            raise TopologyError(f"GPU {uuid!r} at n{node}:{index} outside the "
                                f"{topology.nodes}x{topology.gpus_per_node} island") from None
    return tuple(tuple(n) for n in pool)


def observed_gpu_pool(probe: Sequence[Sequence[Any]], topology: Topology) -> tuple[tuple[str, ...], ...]:
    """``nvidia-smi --query-gpu=index,uuid`` rows per logical node (``(index, uuid)`` pairs,
    any order) -> per-node uuid tuples in local index order. Fail closed: missing node,
    wrong card count, indices not 0..G-1, empty/duplicate uuids all raise."""
    nodes = list(probe)
    if len(nodes) != topology.nodes:
        raise TopologyError(f"GPU probe covers {len(nodes)} nodes, topology has {topology.nodes}")
    out: list[tuple[str, ...]] = []
    seen: set[str] = set()
    want = list(range(topology.gpus_per_node))
    for k, rows in enumerate(nodes):
        rows = list(rows or ())
        if len(rows) != topology.gpus_per_node:
            raise TopologyError(f"node {k} exposes {len(rows)} GPUs, topology says "
                                f"{topology.gpus_per_node}")
        by_index: dict[int, str] = {}
        for row in rows:
            index, uuid = row[0], row[1]
            uuid = str(uuid).strip()
            if not uuid or uuid in seen:
                raise TopologyError(f"node {k} GPU {index!r}: empty or repeated uuid {uuid!r}")
            seen.add(uuid)
            by_index[int(index)] = uuid
        if sorted(by_index) != want:
            raise TopologyError(f"node {k} GPU indices {sorted(by_index)} are not 0..{topology.gpus_per_node - 1}")
        out.append(tuple(by_index[i] for i in want))
    return tuple(out)


def reconcile_gpu_pool(declared: Sequence[Sequence[str | None]] | None,
                       observed: Sequence[Sequence[str]], *, accept_rebind: bool) -> ReconcileResult:
    """Q6 rule (pure): the runtime pool must have the declared shape (nodes x cards per
    node); where ``declared`` names a uuid (cfg uuid spelling, or the journal's binding
    baseline from an earlier incarnation) it must match slot by slot. A mismatch is
    refused unless ``accept_rebind`` (``--rl-elastic-accept-rebind``), which accepts the
    new pool and returns the old->new ``mapping``; with no declared uuid at all the
    observed pool simply becomes the baseline (no rebind)."""
    obs = tuple(tuple(str(u) for u in node) for node in observed)
    if declared is None:
        return ReconcileResult(ok=True, observed=obs)
    dec = tuple(tuple(node) for node in declared)
    if len(dec) != len(obs) or any(len(d) != len(o) for d, o in zip(dec, obs)):
        return ReconcileResult(ok=False, observed=obs, error=(
            f"gpu pool shape {[len(o) for o in obs]} differs from the declared "
            f"{[len(d) for d in dec]} (nodes x gpus_per_node)"))
    diffs: list[str] = []
    mapping: dict[str, str] = {}
    for k, (d_node, o_node) in enumerate(zip(dec, obs)):
        for g, (d, o) in enumerate(zip(d_node, o_node)):
            if d is not None and d != o:
                diffs.append(f"n{k}:{g} declared {d} observed {o}")
                mapping[d] = o
    if not diffs:
        return ReconcileResult(ok=True, observed=obs)
    if not accept_rebind:
        return ReconcileResult(ok=False, observed=obs, diffs=tuple(diffs), error=(
            f"{len(diffs)} GPU uuid(s) differ from the binding ({'; '.join(diffs)}); restart "
            "with --rl-elastic-accept-rebind to bind the new GPUs"))
    return ReconcileResult(ok=True, observed=obs, rebind=True, mapping=mapping, diffs=tuple(diffs))


def merge_declared_pool(cfg: Sequence[Sequence[str | None]] | None,
                        baseline: Sequence[Sequence[str]] | None) -> tuple[tuple[str | None, ...], ...] | None:
    """What this incarnation reconciles against: the cfg's uuid where it names one, else
    the journal baseline from the previous incarnation; None when neither exists (first
    run of a cfg without uuids: the observed pool is journaled as the baseline)."""
    if baseline is None:
        if cfg is None or not any(u is not None for node in cfg for u in node):
            return None
        return tuple(tuple(node) for node in cfg)
    base = tuple(tuple(node) for node in baseline)
    if cfg is None:
        return base
    if len(cfg) != len(base) or any(len(c) != len(b) for c, b in zip(cfg, base)):
        return tuple(tuple(node) for node in cfg)  # shape changed: the cfg rules, shape check fails
    return tuple(tuple(c if c is not None else b for c, b in zip(cn, bn)) for cn, bn in zip(cfg, base))
