"""Spare nodes of a resident ``ssh:`` trainer island (own-cluster-fleet-stability D3).

An inventory may hold more machines than the trainer island needs: a node with the role
``trainer-spare`` (:data:`SPARE_ROLE`) is no island node (``island_nodes(..., "trainer")``
does not match it), so the island's node count and its dry run are what they were. It is a
machine the island may take when one of its own nodes is lost. An inventory without spare
nodes (e.g. the 6 x 8 B300 K3.1 layout, whose 48 cards take every node) runs exactly as
before: a lost node fails the island and the fleet relaunches it on the same nodes.

With spare nodes, :class:`~.ssh_resident.SshIslandOps` replaces a lost node:

1. **lost**: a node is lost once its log stream over SSH (the heartbeat: ServerAlive
   keepalives) ended and it then failed ``unreachable_polls`` SSH checks in a row, or once
   its health turned bad twice in a row (``yeto.hw.node_health.HealthTracker`` returned
   ``lost``). An SSH check that fails while the node's stream still flows is no loss. The
   island writes ``island_node_lost`` and stops every other node (its island STOP: the
   incarnation is released on every node but the lost one, which is not waited for);
2. **pick**: a spare of the same ``fabric_group`` (``fabric_groups.check_placement`` on
   the new node set), whose health check passes now, that no other island of this run
   holds (its claim ``fleet/nodes/<name>/claim.json`` in the run store, an exclusive
   create), that is not quarantined, and whose cross-run lease (``node_lease.NodeLeases``,
   when the inventory names a ``lease_store``) is free. Candidates go by node name, so the
   pick is reproducible;
3. **resume**: the island is rendered again with the spare in the lost node's place
   (same learner id, same layout; the rendezvous address is node 0's, so a lost rank 0
   is replaced by the spare as the new node 0) and its next incarnation starts; the
   trainer resumes from the newest checkpoint of its store (:func:`latest_checkpoint_step`
   names it in the event). To the syncer this is an absence of the island and its re-JOIN
   under the same learner id: it catches up to the newest outer weights, then merges;
4. ``island_node_replaced`` names the old node, the new one and the resume step. The lost
   node is **quarantined** (``fleet/nodes/<name>/quarantine.json`` in the run store, with
   the stop it still owes; with a ``lease_store`` also ``nodes/<name>/quarantine.json``
   there, which every run's pick skips): no island picks it until ``yeto cluster
   release-node <name>`` (:func:`release_node`) checked it, ran that stop and removed the
   record. Nothing lifts a quarantine on its own.

A node whose stop is not confirmed when the island fails (rank 0 often exits first on a
power loss; the lost node's stream only ends minutes later) keeps its stream and is checked
with ``ssh true`` at every relaunch poll: ``unreachable_polls`` checks in a row without an
answer make it lost too. A lost node that still answers (a health loss) is stopped at once
in the background; when that stop is confirmed it owes nothing.

No spare: the island waits (``waiting_spare``) and checks again after 60 s, 120 s, ... at
most 900 s, with an ``island_waiting_spare`` event (``elapsed_s``) every hour; past
``--spare-max-wait-s`` (default 24 h) the relaunch is refused and the island is given up.
The fleet controller does not give it up on ``--recover-timeout`` meanwhile; a waiting
island counts as absent for ``fleet.min_islands``. At most
:data:`MAX_REPLACEMENTS` replacements per island within :data:`REPLACEMENT_WINDOW_S`: the next
loss stops the island (its relaunch is refused) and names every node it replaced.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .node_inventory import Node
from .node_quarantine import (  # noqa: F401 - the shared names here too
    QUARANTINE_NAME,
    RELEASE_NODE_HINT,
    SHARED_NODES_DIR,
    describe,
    release_how,
    SPARE_ROLE,
    shared_quarantined_names,
)

NODES_DIR = "fleet/nodes"  # under the run store
CLAIM_NAME = "claim.json"
CLAIM_SCHEMA = "yeto.fleet.node-claim/v1"
QUARANTINE_SCHEMA = "yeto.fleet.node-quarantine/v1"
WAIT_FIRST_S = 60.0
WAIT_MAX_S = 900.0
MAX_REPLACEMENTS = 3
REPLACEMENT_WINDOW_S = 24 * 3600.0
MAX_WAIT_S = 24 * 3600.0  # --spare-max-wait-s: waiting for a spare longer gives the island up
WAIT_REPORT_S = 3600.0  # an island_waiting_spare event this often while it waits
SSH_FAILED = 255  # the exit code of ssh itself (no connection, no session)
JOURNAL_SUFFIX = "-spares.json"  # next to the island's jobs journal
# the inventory fields a spare must share with the island's nodes (the island's layout,
# image and device flags take it as one of them)
SAME_AS_ISLAND = ("device_family", "cpu_arch", "card", "cards_per_node", "runtime")

# Event kinds (fleet.jsonl and the island's record)
NODE_LOST = "island_node_lost"
NODE_REPLACED = "island_node_replaced"
WAITING_SPARE = "island_waiting_spare"
REPLACEMENT_LIMIT = "island_replacement_limit"


class SpareError(RuntimeError):
    """The spare nodes of an inventory cannot serve the island."""


def spare_nodes(nodes: Iterable[Node]) -> list[Node]:
    """The inventory's spare nodes of ``ssh:`` trainer islands (role :data:`SPARE_ROLE`)."""
    return [n for n in nodes if SPARE_ROLE in tuple(getattr(n, "roles", ()) or ())]


def check_spares(island: Sequence[Node], spares: Sequence[Node]) -> None:
    """Every spare must be able to take any island node's place: the same device family, CPU
    architecture, card, card count and runtime (raises :class:`SpareError`)."""
    if not island:
        return
    ref = island[0]
    names = {n.name for n in island}
    for node in spares:
        if node.name in names:
            raise SpareError(f"node {node.name} is both an island node and a {SPARE_ROLE}")
        if "trainer" in node.roles:
            raise SpareError(f"node {node.name}: roles trainer and {SPARE_ROLE} exclude each other")
        wrong = [f"{key} {getattr(node, key)!r} (island: {getattr(ref, key)!r})" for key in SAME_AS_ISLAND
                 if getattr(node, key) != getattr(ref, key)]
        if bool(node.store_env_file) != bool(ref.store_env_file):
            wrong.append("store_env_file " + ("set" if node.store_env_file else "unset")
                         + f" (island: {'set' if ref.store_env_file else 'unset'})")
        if wrong:
            raise SpareError(f"{SPARE_ROLE} node {node.name} cannot take an island node's place: "
                             + "; ".join(wrong))


def spares_from_args(args: Any, spec: Any, *, objects: Any = None, leases: Any = None,
                     log: Callable[[str], None] | None = None, island_index: int | None = None,
                     num_islands: int | None = None, shared: Any = None) -> IslandSpares | None:
    """The :class:`IslandSpares` of the resident ``ssh:`` island ``spec`` (an
    ``SshTrainerSpec``) from ``--rl-node-inventory``; None when the inventory has no spare
    node, or the island's device family has no node health check (a WARN says the spares
    do not apply): nothing changes then, as on main. Every spare must fit the island (:func:`check_spares`)
    and the island's spec must take it in any node's place (its rendezvous address, docker
    flags); raises :class:`SpareError`. ``objects``: the run store's object API (claims,
    quarantine, the resume point; None: :func:`store_objects_of` ``--rl-checkpoint-store``).
    ``island_index`` / ``num_islands``: the island's index among the launch's islands (its
    own checkpoint keys, :func:`resume_keys`, which finds its learner id). ``shared``: the inventory's ``lease_store``
    object API (None: :func:`lease_store_objects_of`), where a quarantine is shared."""
    import dataclasses  # noqa: PLC0415

    spares = spare_nodes(getattr(args, "rl_node_inventory_nodes", None) or ())
    if not spares:
        return None
    check_spares(spec.nodes, spares)
    for node in spares:
        try:
            dataclasses.replace(spec, nodes=(node, *spec.nodes[1:]))
        except Exception as error:  # noqa: BLE001 - SshIslandError: the spare as the island sees it
            raise SpareError(f"{SPARE_ROLE} node {node.name}: {error}") from None
    note = no_checker_note(spec)
    if note is not None:  # no spare without a health check: the island runs as on main
        (log or (lambda m: print(m, file=sys.stderr, flush=True)))(note)
        return None
    if objects is None:
        objects = store_objects_of(args)
    if shared is None:
        shared = lease_store_objects_of(args)
    keys = resume_keys(args, island_index, num_islands)
    max_wait = getattr(args, "spare_max_wait_s", None)
    return IslandSpares(pool=(*spec.nodes, *spares),
                        records=NodeRecords(objects, run_id=str(getattr(args, "cluster_prefix", "") or spec.run_id),
                                            log=log, shared=shared),
                        leases=leases, resume_probe=lambda: latest_checkpoint_step(objects, keys),
                        max_wait_s=float(MAX_WAIT_S if max_wait is None else max_wait))


def no_checker_note(spec: Any) -> str | None:
    """The one line said when an island has spare nodes but its device family has no node
    health check here (``yeto.hw.node_health``): no spare is picked without one, so the
    spares do not apply at all (:func:`spares_from_args` returns None)."""
    family = spec.nodes[0].device_family if getattr(spec, "nodes", None) else None
    if family is None or _checker_for(family) is not None:
        return None
    return (f"[launcher] WARN: the inventory declares {SPARE_ROLE} nodes, but device family {family} has no node "
            "health check in this version (yeto.hw.node_health): the spare nodes do not apply; a failed island "
            "is relaunched on its own nodes, as without spares")


def spare_view(args: Any, spec: Any) -> dict[str, Any] | None:
    """The dry run's ``spares`` of a resident ``ssh:`` island (None: no spare node, the
    record is unchanged): the spare nodes, their fabric groups and the replacement rules.
    Without a health check for the island's device family it says so (:func:`no_checker_note`)."""
    spares = spare_nodes(getattr(args, "rl_node_inventory_nodes", None) or ())
    if not spares:
        return None
    note = no_checker_note(spec)
    if note is not None:
        print(note, file=sys.stderr, flush=True)
    return {"nodes": [n.name for n in spares], **({"note": note} if note is not None else {}),
            "fabric_groups": {n.name: fabric_group(n) for n in (*spec.nodes, *spares)},
            "lost": "stream ended and no SSH answer unreachable_polls times in a row, or health bad twice",
            "pick": "same fabric group, health check passed now, not held by another island of the run "
                    f"({NODES_DIR}/<name>/{CLAIM_NAME}), not quarantined, cross-run lease free; by name",
            "quarantine": f"{NODES_DIR}/<name>/{QUARANTINE_NAME} until released by hand: {RELEASE_NODE_HINT}",
            "waiting_spare_s": [WAIT_FIRST_S, WAIT_MAX_S],
            "max_wait_s": float(MAX_WAIT_S if getattr(args, "spare_max_wait_s", None) is None
                                else args.spare_max_wait_s),
            "max_replacements": {"count": MAX_REPLACEMENTS, "window_h": REPLACEMENT_WINDOW_S / 3600}}


def store_objects_of(args: Any) -> Any:
    """The object API of the run's ``--rl-checkpoint-store`` (None: none here)."""
    from .island_lifecycle import store_objects  # noqa: PLC0415

    try:
        from .storage import store_client_environ  # noqa: PLC0415

        return store_objects(getattr(args, "rl_checkpoint_store", None), store_client_environ(args))
    except Exception:  # noqa: BLE001 - no object API: the records stay with this launcher
        return None


def lease_store_objects_of(args: Any) -> Any:
    """The object API of the inventory's ``lease_store`` (``node_inventory.lease_store_of``;
    None: not declared, or no object API here)."""
    uri = getattr(getattr(args, "rl_node_inventory_nodes", None), "lease_store", None)
    if not uri:
        return None
    try:
        from .node_lease import open_objects  # noqa: PLC0415

        return open_objects(str(uri))
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - no object API: the run store's record still holds
        return None
    from .island_lifecycle import store_objects  # noqa: PLC0415

    try:
        from .storage import store_client_environ  # noqa: PLC0415

        return store_objects(str(uri), store_client_environ(args))
    except Exception:  # noqa: BLE001 - as above
        return None


def fabric_group(node: Node) -> str | None:
    return getattr(node, "fabric_group", None)


# -- the dependencies of other groups (node health, fabric groups) ------------------------
def _checker_for(family: str, **options: Any) -> Any:
    """The device family's node health checker (``yeto.hw.node_health.checker_for``); None
    for a family without one (its ``NullChecker``: nothing run, nothing judged)."""
    from yeto.hw import node_health  # noqa: PLC0415

    checker = node_health.checker_for(family, **options)
    return checker if getattr(checker, "enabled", True) else None


UNAVAILABLE_GROUP = "unavailable"  # fabric_groups' reserved group: never a node of a multi-node island


def _unavailable(node: Node) -> bool:
    return str(fabric_group(node) or "").lower() == UNAVAILABLE_GROUP


def _pick_placement(candidates: Sequence[Node], need: int, *, excluded: Mapping[str, str] | None = None,
                    required: Sequence[Node] = (), inventory: Sequence[Node] | None = None) -> tuple[Node | None, str]:
    """``(node, why)``: the node ``fabric_groups.pick_placement`` adds to the ``required``
    ones (the island's nodes that stay: the island is never moved to other nodes);
    ``excluded``: node name -> why it is out. No node: ``why`` is the
    :class:`~.fabric_groups.PlacementError`'s text (it goes into the wait reason); any other
    error (a ``TypeError``) is raised. A placement that drops a required node is refused."""
    from .fabric_groups import PlacementError, pick_placement  # noqa: PLC0415

    required = list(required)
    names = {r.name for r in required}
    pool = [*required, *(n for n in candidates if n.name not in names)]
    try:
        placement = pick_placement(pool, need, excluded=dict(excluded or {}), required=required,
                                   inventory=list(candidates) if inventory is None else list(inventory))
    except PlacementError as error:
        return None, str(error)
    nodes = list(placement.nodes)
    if any(r.name not in {n.name for n in nodes} for r in required):  # the nodes that stay never move
        return None, f"the placement {[n.name for n in nodes]} drops a node that stays ({sorted(names)})"
    added = [n for n in nodes if n.name not in names]
    return (added[0], "") if added else (None, f"the placement {[n.name for n in nodes]} adds no node")


def hcas_of(node: Node, env: Mapping[str, str] | None = None) -> list[str]:
    """The job ports of the node as the island's health watch reads them: ``NCCL_IB_HCA``
    of the node's inventory env with the island's job ``env`` over it (the job's own
    order), ``node_health.hcas_of``."""
    from yeto.hw import node_health  # noqa: PLC0415

    return node_health.hcas_of({**dict(node.env or {}), **dict(env or {})}.get("NCCL_IB_HCA"))


def node_check(node: Node, ops: Any, *, multi: bool, timeout: float = 120.0,
               env: Mapping[str, str] | None = None) -> tuple[bool, str]:
    """Whether ``node`` may take an island node's place now: its health check (one SSH
    call, ``yeto.hw.node_health``) ran (``checked``) and passes for a multi-node (``multi``)
    or single-node island. An answer over SSH alone is not enough: a family without a
    checker, or a check that did not run, is refused. ``env``: the island's job env (its
    ``NCCL_IB_HCA`` names the ports checked, as the island's own health watch does)."""
    from .ssh_trainer_island import RDMA_FABRICS  # noqa: PLC0415

    checker = _checker_for(node.device_family, expect_gpus=node.cards_per_node,
                           fabric=node.fabric if node.fabric in RDMA_FABRICS else None)
    if checker is None:
        return False, f"no health check for device family {node.device_family} here (yeto.hw.node_health)"
    hcas = hcas_of(node, env) if multi else []
    script = checker.script(hcas)
    if not script:
        return False, f"the health check of device family {node.device_family} runs nothing"
    try:
        done = ops.run(node, script, timeout=timeout)
    except Exception as error:  # noqa: BLE001 - a timeout is an unreachable node
        return False, f"unreachable ({type(error).__name__})"
    if done.returncode == SSH_FAILED:
        return False, f"unreachable (rc {done.returncode})"  # 255: ssh itself failed, no output to judge
    result = checker.parse(getattr(done, "stdout", "") or "", hcas, None)
    if not getattr(result, "checked", False):
        return False, f"health not checked (rc {done.returncode})"
    ok = result.ok_multi if multi else result.ok_single
    if ok:
        return True, "healthy"
    return False, "unhealthy: " + ("; ".join(result.reasons) or "check failed")


# -- the run store's node records ---------------------------------------------------------
class NodeRecords:
    """Claims and quarantine records of nodes in the run store (``fleet/nodes/<name>/``).

    A claim says which island of the run holds the node (an exclusive create, so two islands
    of one run never take one spare; a store whose create is check-then-write, e.g. the OSS
    relay, is refused); a quarantine record keeps a lost node out of every pick until
    :func:`release_node`. ``objects``: the run store's object API; None keeps the records in
    this process only (said once). ``shared``: the object API of the inventory's
    ``lease_store``, where a quarantine is written too (``nodes/<name>/quarantine.json``),
    so another run's pick skips the node."""

    def __init__(self, objects: Any, *, run_id: str, log: Callable[[str], None] | None = None,
                 wall: Callable[[], float] = time.time, shared: Any = None) -> None:
        self.objects = objects
        self.shared = shared
        self.run_id = str(run_id)
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.wall = wall
        self._memory: dict[str, bytes] = {}
        if objects is None:
            self.log("[launcher] spare nodes: the run store has no object API here; node claims and "
                     "quarantine records are kept by this launcher only")

    @staticmethod
    def key(name: str, what: str) -> str:
        return f"{NODES_DIR}/{name}/{what}"

    @staticmethod
    def shared_key(name: str) -> str:
        return f"{SHARED_NODES_DIR}/{name}/{QUARANTINE_NAME}"

    # -- raw access
    @staticmethod
    def _parse(raw: bytes | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        return doc if isinstance(doc, dict) else {"unreadable": True,
                                                  "reason": "quarantine record unreadable: not a JSON object"}

    def _read(self, key: str) -> dict[str, Any] | None:
        return self._parse(self._memory.get(key) if self.objects is None else self.objects.read(key))

    def _write(self, key: str, doc: Mapping[str, Any]) -> None:
        data = json.dumps(dict(doc), sort_keys=True).encode()
        if self.objects is None:
            self._memory[key] = data
        else:
            self.objects.write(key, data)

    def _create(self, key: str, doc: Mapping[str, Any]) -> bool:
        data = json.dumps(dict(doc), sort_keys=True).encode()
        if self.objects is None:
            if key in self._memory:
                return False
            self._memory[key] = data
            return True
        from yeto.rl.engine.object_store import atomic_create  # noqa: PLC0415

        refused = SpareError("the run store has no atomic exclusive create (a check-then-write store such as "
                             "the OSS relay, or an endpoint that refuses If-None-Match): two islands could "
                             "claim one spare node there; use an s3:// / r2:// store or a local directory")
        if not atomic_create(self.objects):
            raise refused
        created = bool(self.objects.create(key, data))
        if created and not atomic_create(self.objects):  # the store found out on this create
            self.objects.delete(key)
            raise refused
        return created

    def _delete(self, key: str) -> None:
        if self.objects is None:
            self._memory.pop(key, None)
        else:
            self.objects.delete(key)

    # -- claims
    def holder(self, name: str) -> str | None:
        """The island of this run that holds ``name`` (None: free)."""
        doc = self._read(self.key(name, CLAIM_NAME))
        if doc is None:
            return None
        return f"{doc.get('run')}/{doc.get('island')}" if doc.get("run") != self.run_id else str(doc.get("island"))

    def claim(self, name: str, island: str) -> str | None:
        """Take ``name`` for ``island``; None when it is the island's now, else its holder.
        Raises :class:`SpareError` on a store without an atomic create."""
        doc = {"schema": CLAIM_SCHEMA, "node": name, "run": self.run_id, "island": island, "at": self.wall()}
        if self._create(self.key(name, CLAIM_NAME), doc):
            return None
        held = self.holder(name)
        return None if held == island else (held or "unknown")

    def unclaim(self, name: str, island: str) -> None:
        if self.holder(name) == island:
            self._delete(self.key(name, CLAIM_NAME))

    # -- quarantine
    def shared_quarantined(self, name: str) -> dict[str, Any] | None:
        """The node's quarantine in the lease store (any run's; None: none, or no lease store)."""
        if self.shared is None:
            return None
        from .node_quarantine import read_record  # noqa: PLC0415

        return read_record(self.shared, name)  # unreadable: a quarantine (with that reason)

    def own_quarantine(self, name: str) -> dict[str, Any] | None:
        """The node's quarantine record in this run store (None: not quarantined here)."""
        return self._read(self.key(name, QUARANTINE_NAME))

    def quarantined(self, name: str) -> dict[str, Any] | None:
        """The node's quarantine record: this run's, else another run's in the lease store."""
        return self.own_quarantine(name) or self.shared_quarantined(name)

    def quarantine(self, name: str, *, island: str, reason: str, owed: Sequence[Mapping[str, Any]] = (),
                   lease_holder: str | None = None) -> None:
        """Keep ``name`` out of every pick; ``owed``: the stops it never got (run by release);
        ``lease_holder``: who holds its cross-run lease (kept until release-node ends it)."""
        earlier = self.own_quarantine(name) or {}
        doc = {"schema": QUARANTINE_SCHEMA, "node": name, "run": self.run_id, "island": island,
               "reason": str(reason)[:500], "at": self.wall(),
               "owed": [*list(earlier.get("owed") or ()), *map(dict, owed)]}
        holder = lease_holder or earlier.get("lease_holder")
        if holder:
            doc["lease_holder"] = str(holder)
        self._write(self.key(name, QUARANTINE_NAME), doc)
        self._share(name, doc)

    def _share(self, name: str, doc: Mapping[str, Any]) -> None:
        if self.shared is None:
            return
        try:
            self.shared.write(self.shared_key(name), json.dumps(dict(doc), sort_keys=True).encode())
        except Exception as error:  # noqa: BLE001 - the run store's record still holds it in this run
            self.log(f"[launcher] WARN: quarantine of {name} not written to the lease store ({error}): "
                     "another run may pick it")

    def settle(self, name: str, *, run_id: str, island: str, incarnation: int) -> None:
        """The stop ``name`` owed for that incarnation is confirmed: it owes it no more (the
        node stays quarantined)."""
        doc = self.own_quarantine(name)
        if doc is None:
            return
        key = (str(run_id), str(island), int(incarnation))
        owed = [o for o in doc.get("owed") or ()
                if (str(o.get("run_id")), str(o.get("island")), int(o.get("incarnation", -1))) != key]
        doc = {**doc, "owed": owed}
        self._write(self.key(name, QUARANTINE_NAME), doc)
        self._share(name, doc)

    def release(self, name: str) -> None:
        self._delete(self.key(name, QUARANTINE_NAME))
        if self.shared is not None:
            self.shared.delete(self.shared_key(name))


def lease_holder_id(leases: Any) -> str | None:
    """Who this launcher's leases are held as (``NodeLeases.holder_id``, ``<run>/<part>@<identity>``;
    an older lease API: its ``run_id``)."""
    return getattr(leases, "holder_id", None) or getattr(leases, "run_id", None)


def release_node(records: NodeRecords, node: Node, ssh: Any, *, kill_grace_s: int | None = None,
                 leases: Any = None, probe: Callable[[Any, str], str | None] | None = None) -> tuple[bool, str]:
    """``yeto cluster release-node``: run every stop the quarantined ``node`` still owes
    (the incarnations it ran when it was lost; each confirmed stop closes its owed ledger
    row), give up the cross-run lease the run kept on it while it was quarantined
    (``leases.force_release`` as the holder its record names, else this launcher's: a lease
    another run holds is not touched; a kept or run-out lease ends at once, a live one only
    after ``probe(node, holder)`` (``node_lease.ssh_probe``) found nothing of that run on the
    node, then with ``force``), then remove its quarantine record so the islands may pick it
    again.
    ``(ok, message)``; the record stays while a stop or the lease release fails."""
    from . import ssh_trainer_ledger  # noqa: PLC0415
    from .ssh_trainer_island import KILL_GRACE_S, kill_script, remote_paths  # noqa: PLC0415

    doc = records.quarantined(node.name)
    if doc is None:
        return True, f"node {node.name} is not quarantined"
    grace = int(KILL_GRACE_S if kill_grace_s is None else kill_grace_s)
    ledger_notes: list[str] = []
    for owed in doc.get("owed") or ():
        run_id = str(owed.get("run_id") or records.run_id)
        paths = remote_paths(run_id, str(owed["island"]), int(owed["incarnation"]))
        script = kill_script(paths, grace, container=owed.get("container") or None)
        try:
            done = ssh.run(node, script, timeout=grace + 60.0)
        except Exception as error:  # noqa: BLE001 - reported, the record stays
            return False, f"node {node.name}: the stop of {owed['island']} incarnation {owed['incarnation']} failed: {error}"
        if done.returncode != 0:
            err = (getattr(done, "stderr", "") or "").strip()[-300:]
            return False, (f"node {node.name}: the stop of {owed['island']} incarnation {owed['incarnation']} "
                           f"ended rc {done.returncode} {err}").rstrip()
        status, ledger_path = ssh_trainer_ledger.release_owed(run_id=run_id, island=str(owed["island"]),
                                                              incarnation=int(owed["incarnation"]), node=node.name)
        if status != ssh_trainer_ledger.OWED_RELEASED:
            ledger_notes.append(_owed_row_note(run_id, str(owed["island"]), int(owed["incarnation"]), node.name,
                                               ledger_path))
    lease_note = ""
    if leases is not None:
        holder = doc.get("lease_holder") or lease_holder_id(leases)
        try:
            released = leases.force_release(node, holder=holder)  # kept or run out
            current = None if released else leases.holder(node)
            if not released and current is not None and current == holder:  # a live lease of that run
                why = "no probe of the node" if probe is None else probe(node, holder)
                if why is not None:
                    return False, (f"node {node.name}: its lease of {holder} still runs and the node is not "
                                   f"confirmed free ({why}); it stays quarantined")
                released = leases.force_release(node, holder=holder, force=True)
            elif not released and current is not None:  # another run holds it now: left alone
                lease_note = f"; its lease is held by {current}, not released"
        except Exception as error:  # noqa: BLE001 - reported, the record stays
            return False, (f"node {node.name}: its lease could not be released ({type(error).__name__}: "
                           f"{error}); it stays quarantined")
    records.release(node.name)
    return True, (f"node {node.name} released ({len(doc.get('owed') or ())} owed stop(s) run){lease_note}"
                  + "".join(f"; {n}" for n in ledger_notes))


def _lease_plan(records: NodeRecords, node: Node, doc: Mapping[str, Any], leases: Any) -> str:
    """What release-node would do with the node's cross-run lease (its dry run)."""
    if leases is None:
        return "no lease store: no lease to end"
    holder = doc.get("lease_holder") or lease_holder_id(leases)
    try:
        current = leases.holder(node)
    except Exception as error:  # noqa: BLE001 - the dry run reads only
        return f"the lease of {node.name} could not be read ({type(error).__name__}: {error})"
    if current is None:
        return f"no lease held on {node.name}: none to end"
    if current != holder:
        return f"the lease of {node.name} is held by {current}, not {holder}: it is left alone"
    return (f"would end the lease of {holder} on {node.name} (a kept one at once; a live one only after a "
            "probe over SSH finds nothing of that run on the node)")


def _owed_row_note(run_id: str, island: str, incarnation: int, node: str, path: str) -> str:
    """Where the owed ledger row of a stop release-node ran lives when this machine has none open."""
    from yeto.cost.ledger import LEDGER_FILE  # noqa: PLC0415

    from . import ssh_trainer_ledger  # noqa: PLC0415

    rid = ssh_trainer_ledger.owed_id(run_id, island, incarnation, node)
    return (f"WARN: its owed ledger row {rid} is not open in {path or 'any ledger here'}: the row is in the run "
            f"dir of the machine that ran the launcher of run {run_id} (runs/{run_id}/{LEDGER_FILE}); the "
            "launcher closes it (released elsewhere) when it picks the node again, or close it there with "
            f"`yeto cleanup --run {run_id}` once the run has ended")


RELEASE_NODE_RUN = "yeto-cluster-release-node"  # who ends a lease (its released_by)


def release_leases_of(nodes: Any, say: Callable[[str], None]) -> Any:
    """The :class:`~.node_lease.NodeLeases` of the inventory's ``lease_store`` for release-node
    (None: none declared, or it does not open)."""
    from .node_inventory import lease_store_of  # noqa: PLC0415

    store = lease_store_of(nodes)
    if not store:
        return None
    from .node_lease import NodeLeases  # noqa: PLC0415

    try:
        return NodeLeases(str(store), RELEASE_NODE_RUN)
    except Exception as error:  # noqa: BLE001 - reported: the lease is left to its holder
        say(f"[yeto] WARN: the lease store {store} does not open ({error}); the node's lease is not released")
        return None


def release_node_main(*, inventory: str, name: str, store: str, ssh: Any = None, dry_run: bool = False,
                      force: bool = False, kill_grace_s: int | None = None, leases: Any = None,
                      objects: Any = None, shared: Any = None, say: Callable[[str], None] | None = None) -> int:
    """``yeto cluster release-node <name>`` (``inventory``: the inventory file, ``store``: the run
    store): show the
    node's quarantine record, check the node (its health check ran and passes; an answer
    over SSH alone is not enough, and a family without a checker is refused), run the stops
    it still owes and remove the record, so the islands may pick it again; its kept
    cross-run lease is given up (``leases``). ``force``: release without the health check
    (a WARN says so). ``dry_run``: the record and the stops that would run, no SSH, nothing
    removed. Exit 0 released (or not quarantined), 1 the check, a stop or the lease release
    failed (the record stays), 2 a usage error."""
    from .node_inventory import InventoryError, island_nodes, load_inventory  # noqa: PLC0415

    say = say or (lambda line: print(line, flush=True))
    try:
        nodes = load_inventory(inventory)
    except (InventoryError, OSError) as error:
        say(f"[yeto] cluster release-node: {error}")
        return 2
    node = next((n for n in nodes if n.name == name), None)
    if node is None:
        say(f"[yeto] cluster release-node: node {name!r} is not in {inventory}")
        return 2
    if objects is None:
        objects = store_objects_of(type("Args", (), {"rl_checkpoint_store": store})())
    if objects is None:
        say(f"[yeto] cluster release-node: the run store {store!r} has no object API here (s3://, r2://, oss:// "
            "or a local directory)")
        return 2
    if leases is None:
        leases = release_leases_of(nodes, say)
    if shared is None and leases is not None:
        shared = getattr(leases, "objects", None)
    records = NodeRecords(objects, run_id="", log=say, shared=shared)
    doc = records.quarantined(name)
    if doc is None:
        say(f"[yeto] node {name} is not quarantined in {store}")
        return 0
    if doc.get("unreadable"):  # a broken record: which stops the node owes is not known
        say(f"[yeto] node {name}: its quarantine record is unreadable ({doc.get('reason')}): the stops it still owes "
            "cannot be known")
        if not force:
            say(f"[yeto] node {name} stays quarantined: check it by hand for containers or processes of the run "
                "(yeto-<run>-*), then release it with --force")
            return 1
        say(f"[yeto] WARN: --force: node {name} is released without running any owed stop")
        doc = {**doc, "owed": []}
    owed = list(doc.get("owed") or ())
    if not doc.get("unreadable"):
        say(f"[yeto] node {name}: quarantined by island {doc.get('island')} of run {doc.get('run')}: "
            f"{doc.get('reason')}")
    for row in owed:
        say(f"[yeto]   owes the stop of {row.get('island')} incarnation {row.get('incarnation')} (rank "
            f"{row.get('rank')}" + (f", container {row['container']}" if row.get("container") else "") + ")")
    if dry_run:
        say(f"[yeto] dry run: would {'skip the health check of' if force else 'check'} {name}, run {len(owed)} "
            f"stop(s) and remove {NODES_DIR}/{name}/{QUARANTINE_NAME}")
        say(f"[yeto]   would remove {records.key(name, QUARANTINE_NAME)} in the run store {store}"
            + (f" and {records.shared_key(name)} in the lease store {getattr(leases, 'store_uri', '')}".rstrip()
               if records.shared is not None else ""))
        say(f"[yeto]   {_lease_plan(records, node, doc, leases)}")
        return 0
    if ssh is None:
        from .ssh_trainer_island import SshOps  # noqa: PLC0415

        ssh = SshOps()
    if force:
        say(f"[yeto] WARN: --force: node {name} is released without its health check; an island may pick a "
            "broken node")
    else:
        healthy, why = node_check(node, ssh, multi=len(island_nodes(nodes, "trainer")) > 1)
        if not healthy:
            say(f"[yeto] node {name} stays quarantined: {why} (--force releases it without the check)")
            return 1
        say(f"[yeto] node {name}: {why}")
    probe = None
    if leases is not None:
        from .node_lease import ssh_probe  # noqa: PLC0415

        probe = ssh_probe(lambda n, script: ssh.run(n, script, timeout=120.0))
    ok, message = release_node(records, node, ssh, kill_grace_s=kill_grace_s, leases=leases, probe=probe)
    say(f"[yeto] {message}" if ok else f"[yeto] node {name} stays quarantined: {message}")
    return 0 if ok else 1


# -- the resume point -------------------------------------------------------------------
def resume_keys(args: Any = None, island: int | None = None, num_islands: int | None = None) -> tuple[str, str]:
    """Where island ``island`` (its index among the launch's islands) keeps its trainer's
    newest checkpoint in the run store: ``(round cut LATEST, release LATEST.json)``. A
    synced island with round cuts writes them under ``island-cuts/island<learner id>/``
    (``--rl-island-cut-store``; the learner id is ``sample_roles.island_learner``'s, e.g.
    ``--rl-fleet-learner-id`` of a per-island launch, not the index); a trainer whose
    samplers follow it releases to its ``trainer<k>/policy-releases/``
    (``sample_roles.island_substore``). ``island`` None: the store's top (one trainer)."""
    from yeto.rl.engine.resume import LATEST  # noqa: PLC0415
    from yeto.rl.quant.schemes import RELEASE_DIR  # noqa: PLC0415

    cut, release = LATEST, f"{RELEASE_DIR}/LATEST.json"
    if args is None or island is None:
        return cut, release
    from yeto.rl.engine import sample_roles  # noqa: PLC0415
    from yeto.rl.engine.cut_interval import island_cut_store_for  # noqa: PLC0415

    from .rl_ports import island_cuts_requested  # noqa: PLC0415

    islands = int(num_islands or int(island) + 1)
    if island_cuts_requested(args):
        try:
            learners = sample_roles.syncer_learner_count(args, islands)
            learner = sample_roles.island_learner(args, int(island), learners, islands)[0]
        except Exception:  # noqa: BLE001 - a launch without a learner layout: its index
            learner = int(island)
        cut = f"{island_cut_store_for('island-cuts', int(learner))}/{LATEST}"
    try:
        substore = sample_roles.island_substore(args, int(island), islands)
    except Exception:  # noqa: BLE001 - a launch without that layout: the store's top
        substore = None
    if substore:
        release = f"{substore}/{release}"
    return cut, release


def latest_checkpoint_step(objects: Any, keys: tuple[str, str] | None = None) -> dict[str, Any] | None:
    """The newest checkpoint an island's trainer resumes from, as its store holds it at
    ``keys`` (:func:`resume_keys`): its round cut (``LATEST`` of :mod:`yeto.rl.engine.resume`:
    the cut's rollout) or, without round cuts, its newest BF16 policy release
    (``LATEST.json``: its version). None: no checkpoint, or the store does not read."""
    from yeto.rl.engine.resume import _cut_rollout  # noqa: PLC0415

    if objects is None:
        return None
    cut_key, release_key = keys or resume_keys()
    try:
        raw = objects.read(cut_key)
        if raw is not None:
            doc = json.loads(raw)
            step = _cut_rollout(str(doc.get("cut_id") or ""))
            if step is None and doc.get("next_rollout_id") is not None:
                step = int(doc["next_rollout_id"]) - 1
            if step is not None:
                return {"source": "round_cut", "step": int(step), "ref": doc.get("cut_id"), "key": cut_key}
        raw = objects.read(release_key)
        if raw is not None:
            doc = json.loads(raw)
            if doc.get("version") is not None:
                return {"source": "release", "step": int(doc["version"]), "ref": doc.get("path"), "key": release_key}
    except Exception:  # noqa: BLE001 - unknown, not a reason to stop the replacement
        return None
    return None


# -- one island's state ---------------------------------------------------------------------
@dataclass
class IslandSpares:
    """What a resident ``ssh:`` island with spare nodes keeps (``SshIslandOps.add(...,
    spares=...)``). ``pool``: the inventory's island and spare nodes (the candidates are the
    ones not in the island); ``leases``: ``node_lease.NodeLeases`` of the inventory's
    ``lease_store`` (None: no cross-run lease); ``resume_probe``: the island's newest
    checkpoint (:func:`latest_checkpoint_step` of its store)."""

    pool: tuple[Node, ...]
    records: NodeRecords
    job_env: dict[str, str] = field(default_factory=dict)  # the island's job env (SshIslandOps.add sets it)
    leases: Any = None
    resume_probe: Callable[[], dict[str, Any] | None] | None = None
    wait_first_s: float = WAIT_FIRST_S
    wait_max_s: float = WAIT_MAX_S
    max_replacements: int = MAX_REPLACEMENTS
    window_s: float = REPLACEMENT_WINDOW_S
    max_wait_s: float = MAX_WAIT_S  # --spare-max-wait-s (wall clock: a launcher restart keeps it)
    report_every_s: float = WAIT_REPORT_S
    # runtime
    journal: Path | None = None
    lost: dict[int, str] = field(default_factory=dict)  # rank -> why it is lost
    stream_ended: set[int] = field(default_factory=set)  # ranks whose stream ended on its own
    unconfirmed: set[int] = field(default_factory=set)  # ranks whose stop the release did not confirm
    silent: dict[int, int] = field(default_factory=dict)  # rank -> failed SSH checks in a row
    # (lost node, incarnation) whose own stop was confirmed: that stop is owed no more
    stopped: set[tuple[str, int]] = field(default_factory=set)
    stop_threads: list[Any] = field(default_factory=list, repr=False)  # background stops of lost nodes
    waiting: bool = False
    limited: bool = False  # the replacement limit was reported
    wait_s: float = 0.0
    next_check: float = 0.0
    wait_reason: str | None = None
    wait_since: float | None = None  # wall clock of the wait's start (kept in the journal)
    reported_at: float = 0.0  # wall clock of the last island_waiting_spare
    replacements: list[dict[str, Any]] = field(default_factory=list)
    nodes: dict[str, list[str]] = field(default_factory=dict)  # incarnation -> node names
    events: list[dict[str, Any]] = field(default_factory=list)
    # the events' own lock: the fleet controller drains them each poll while a relaunch (its
    # spare's preflight, a docker pull) may hold SshIslandOps' lock for minutes
    events_lock: Any = field(default_factory=threading.Lock, repr=False)
    # the owed stops' lock: the background stop of a lost node and the release that records
    # what it still owes agree on it (a confirmed stop owes nothing, never a row left open)
    owed_lock: Any = field(default_factory=threading.Lock, repr=False)

    def by_name(self) -> dict[str, Node]:
        return {n.name: n for n in self.pool}

    # -- the journal (next to the island's jobs journal)
    def load(self, path: Path) -> None:
        self.journal = Path(path)
        try:
            doc = json.loads(self.journal.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        self.nodes = {str(k): [str(n) for n in v] for k, v in dict(doc.get("nodes") or {}).items()}
        self.replacements = [dict(r) for r in doc.get("replacements") or () if isinstance(r, dict)]
        since = doc.get("waiting_since")
        self.wait_since = None if since is None else float(since)

    def save(self) -> None:
        if self.journal is None:
            return
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.journal.with_suffix(".tmp")
        doc = {"nodes": self.nodes, "replacements": self.replacements,
               **({"waiting_since": self.wait_since} if self.wait_since is not None else {})}
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.journal)

    def nodes_of(self, incarnation: int | None) -> list[str] | None:
        """The node names incarnation ``incarnation`` runs on (None: the newest recorded)."""
        if not self.nodes:
            return None
        if incarnation is None:
            return self.nodes[max(self.nodes, key=int)]
        return self.nodes.get(str(int(incarnation)))

    def record_nodes(self, incarnation: int, names: Sequence[str]) -> None:
        self.nodes[str(int(incarnation))] = list(names)
        self.save()

    # -- limits and waiting
    def recent(self, now_wall: float) -> list[dict[str, Any]]:
        return [r for r in self.replacements if now_wall - float(r.get("at") or 0.0) < self.window_s]

    def limit_reached(self, now_wall: float) -> bool:
        return len(self.recent(now_wall)) >= int(self.max_replacements)

    def wait(self, now: float, reason: str, *, wall: float | None = None) -> bool:
        """Wait for a spare (the next check after the backoff); True when the wait is new or
        its reason changed (worth an event). ``wall``: the wall clock (the wait's start is
        kept in the journal, so a restarted launcher keeps its ``max_wait_s``)."""
        self.wait_s = self.wait_first_s if not self.waiting else min(self.wait_max_s, max(self.wait_s, 1.0) * 2)
        self.next_check = now + self.wait_s
        fresh = not self.waiting or reason != self.wait_reason
        self.waiting, self.wait_reason = True, reason
        if wall is not None and self.wait_since is None:
            self.wait_since = float(wall)
            self.save()
        if fresh and wall is not None:
            self.reported_at = float(wall)
        return fresh

    def waited_s(self, wall: float) -> float:
        return 0.0 if self.wait_since is None else max(0.0, float(wall) - self.wait_since)

    def report_due(self, wall: float) -> bool:
        """An hourly ``island_waiting_spare`` is due (True once per ``report_every_s``)."""
        if not self.waiting or wall - self.reported_at < self.report_every_s:
            return False
        self.reported_at = float(wall)
        return True

    def done_waiting(self) -> None:
        self.waiting, self.wait_s, self.next_check, self.wait_reason = False, 0.0, 0.0, None
        if self.wait_since is not None:
            self.wait_since = None
            self.save()

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"kind": kind, **fields}
        with self.events_lock:
            self.events.append(event)
        return event

    def take_events(self) -> list[dict[str, Any]]:
        with self.events_lock:
            out, self.events = self.events, []
        return out


@dataclass(frozen=True)
class Pick:
    node: Node | None
    group: str | None
    skipped: dict[str, str]
    unchecked: tuple[str, ...] = ()  # candidates left that were not health checked (check them, pick again)
    placement: str = ""  # why fabric_groups.pick_placement placed no spare (its PlacementError)

    def why_none(self) -> str:
        where = f"fabric group {self.group}" if self.group else "the inventory"
        detail = "; ".join(f"{name}: {why}" for name, why in sorted(self.skipped.items()))
        return (f"{where} has no free healthy spare node" + (f" ({detail})" if detail else "")
                + (f"; placement: {self.placement}" if self.placement else ""))


def spare_candidates(spares: IslandSpares, island: str, current: Sequence[Node], lost: Iterable[int], *,
                     chosen: Sequence[Node] = ()) -> tuple[list[Node], str | None, dict[str, str], bool, list[Node]]:
    """``(candidates, group, skipped, multi, keep)`` of a pick for the ``lost`` ranks of ``current``:
    the pool's nodes that are neither the island's other nodes nor ``chosen`` (spares picked
    for other lost ranks), in the island's fabric group (a multi-node island never takes a
    node of the reserved group ``unavailable``). ``skipped``: node name -> why it is out:
    another fabric group (not a candidate), or quarantined (in this run or, with a
    ``lease_store``, any run), held by another island of the run, leased by another run (a
    candidate, in ``pick_placement``'s ``excluded`` with that reason). ``keep``: the
    island's nodes that stay (``required``). No SSH."""
    lost = sorted(set(lost))
    keep = [n for r, n in enumerate(current) if r not in lost] + list(chosen)
    group = (next((fabric_group(n) for n in keep if fabric_group(n)), None)
             or next((fabric_group(current[r]) for r in lost if fabric_group(current[r])), None))
    grouped = any(fabric_group(n) for n in (*current, *spares.pool))
    multi = len(current) > 1
    taken = {n.name for n in keep}
    skipped: dict[str, str] = {}
    candidates: list[Node] = []
    for node in sorted(spares.pool, key=lambda n: n.name):
        if node.name in taken:
            continue
        if multi and _unavailable(node):
            skipped[node.name] = f"fabric group {UNAVAILABLE_GROUP} (never in a multi-node island)"
        elif multi and grouped and fabric_group(node) != group:
            skipped[node.name] = f"fabric group {fabric_group(node) or 'none'} (the island's is {group or 'none'})"
        else:
            candidates.append(node)
            if (record := spares.records.own_quarantine(node.name)) is not None:
                skipped[node.name] = f"quarantined ({str(record.get('reason') or '')[:120]})"
            elif (record := spares.records.shared_quarantined(node.name)) is not None:  # another run's
                skipped[node.name] = (f"quarantined: {describe(node.name, record)}; once fixed, "
                                      f"{release_how(node.name)}")
            elif (held := spares.records.holder(node.name)) is not None and held != island:
                skipped[node.name] = f"held by island {held}"
            elif spares.leases is not None and (holder := spares.leases.holder(node)) is not None \
                    and holder != lease_holder_id(spares.leases):
                skipped[node.name] = f"leased by run {holder}"
    return candidates, group, skipped, multi, keep


def _place(spares: IslandSpares, candidates: Sequence[Node], keep: Sequence[Node],
           excluded: Mapping[str, str]) -> tuple[Node | None, str]:
    """The spare ``fabric_groups.pick_placement`` adds to the island's nodes that stay
    (``required``: the island is never moved to other nodes); ``excluded``: name -> why
    (quarantined, held, leased, unhealthy, checked before)."""
    names = {n.name for n in candidates}
    return _pick_placement(candidates, len(keep) + 1, required=keep, inventory=spares.pool,
                           excluded={k: v for k, v in excluded.items() if k in names})


def check_candidates(spares: IslandSpares, island: str, current: Sequence[Node], lost: Iterable[int], *,
                     ops: Any, need: int = 1, known: Iterable[str] = ()) -> dict[str, tuple[bool, str]]:
    """The health checks of a pick, run before it (outside ``SshIslandOps``' lock: each is
    an SSH call): the candidates in ``fabric_groups.pick_placement`` order, but ``known``
    (checked before), until ``need`` pass. Node name -> ``(ok, why)`` (:func:`node_check`)."""
    candidates, _group, skipped, multi, keep = spare_candidates(spares, island, current, lost)
    checked: dict[str, tuple[bool, str]] = {}
    out = {**skipped, **{k: "health checked before" for k in known}}
    passed = 0
    while passed < need:
        node, _why = _place(spares, candidates, keep, {**out, **{k: why for k, (_ok, why) in checked.items()}})
        if node is None:
            break
        checked[node.name] = node_check(node, ops, multi=multi, env=spares.job_env)
        passed += bool(checked[node.name][0])
    return checked


def pick_spare(spares: IslandSpares, island: str, current: Sequence[Node], lost: Iterable[int], *,
               ops: Any = None, chosen: Sequence[Node] = (),
               health: Mapping[str, tuple[bool, str]] | None = None) -> Pick:
    """A spare for one of the ``lost`` ranks of ``current``: :func:`spare_candidates` (read
    again now: a node may have been claimed, quarantined or leased since the checks), then
    ``fabric_groups.pick_placement`` (need 1) names the spare, by node name. One whose
    health check (``health``: :func:`check_candidates`' results; a node not checked there is
    passed over this time; None: checked here over ``ops``), claim or lease fails is
    excluded and the pick runs again. A lost node itself is a candidate again once released
    from its quarantine. The pick is claimed (and its lease taken) before it is returned;
    :func:`unpick` gives it back."""
    candidates, group, skipped, multi, keep = spare_candidates(spares, island, current, lost, chosen=chosen)
    unchecked: list[str] = []
    while True:
        node, why = _place(spares, candidates, keep, skipped)  # every node tried below is in skipped, with why
        if node is None:
            return Pick(None, group, skipped, tuple(unchecked), placement=why)
        skipped[node.name] = "tried"
        verdict = node_check(node, ops, multi=multi, env=spares.job_env) if health is None else health.get(node.name)
        if verdict is None:
            unchecked.append(node.name)
            skipped[node.name] = "health not checked yet"
            continue
        healthy, why = verdict
        if not healthy:
            skipped[node.name] = why
            continue
        try:
            held = spares.records.claim(node.name, island)
        except SpareError as error:
            skipped[node.name] = f"not claimed: {error}"
            continue
        if held is not None:
            skipped[node.name] = f"held by island {held}"
            continue
        if spares.leases is not None:
            lease = spares.leases.acquire(node)
            if not getattr(lease, "ok", False) or getattr(lease, "blocked", False):
                spares.records.unclaim(node.name, island)
                skipped[node.name] = (f"lease of run {getattr(lease, 'previous', None) or 'unknown'} not taken over "
                                      "(its run may still run there)" if getattr(lease, "blocked", False)
                                      else f"leased by run {getattr(lease, 'holder', None) or 'unknown'}")
                continue
        skipped.pop(node.name, None)
        with spares.owed_lock:  # picked again: a loss of its new incarnation owes its stop again
            spares.stopped -= {k for k in spares.stopped if k[0] == node.name}
        _close_owed_elsewhere(spares, node)
        return Pick(node, group, skipped)


def _close_owed_elsewhere(spares: IslandSpares, node: Node) -> None:
    """A picked node has no quarantine record: a stop it owed this run was run by a
    release-node elsewhere; its owed ledger rows here close (``released elsewhere``)."""
    from . import ssh_trainer_ledger  # noqa: PLC0415

    closed = ssh_trainer_ledger.release_owed_elsewhere(run_id=spares.records.run_id, node=node.name)
    if closed:
        spares.records.log(f"[launcher] node {node.name} is picked again with no quarantine record: its owed "
                           f"ledger row(s) {closed} are closed (released elsewhere)")


def unpick(spares: IslandSpares, island: str, node: Node) -> None:
    """Give a pick back (its start failed): its claim and lease."""
    spares.records.unclaim(node.name, island)
    if spares.leases is not None:
        try:
            spares.leases.release(node)
        except Exception:  # noqa: BLE001 - the lease runs out on its own
            pass


def owed_stop(run_id: str, island: str, incarnation: int, rank: int, docker: bool) -> dict[str, Any]:
    """The stop a lost node still owes for incarnation ``incarnation`` (run by release_node)."""
    from .ssh_trainer_island import container_name  # noqa: PLC0415

    return {"run_id": run_id, "island": island, "incarnation": int(incarnation), "rank": int(rank),
            "container": container_name(run_id, island, incarnation, rank) if docker else None}

