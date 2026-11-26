"""The checks a new ``ssh:`` trainer island's nodes pass before ``yeto fleet island add``
writes it into a running fleet, or ``yeto fleet island move`` moves an island onto them
(own-cluster-fleet-stability 9.3).

An island's nodes are the trainer nodes of its inventory (the command's inventory option:
its launch's ``--rl-node-inventory``). Before anything is written (and in the dry run):

* count and this fleet: an inventory without ``lease_store`` lists exactly as many
  trainer nodes as ``--gpu`` says, none of them a trainer node of another ``ssh:`` island
  of the fleet (same name, or same machine: the SSH target without its ``user@`` or the
  ``address``, in any case). An inventory with ``lease_store`` may list more (the launch's
  rule): the island takes ``need`` of them, as its launch picks them (below), and a node
  is another island's when that island's launch holds its lease;
* health: the node health checker of the device family (``yeto.hw.node_health``, built as
  the island's launch builds it), one SSH call per node, judged with absolute rules (a
  first check has no previous one to compare with): a hardware XID in this boot, a
  volatile uncorrectable ECC count above 0, an NVSwitch node whose fabric manager is not
  ``active``, a GPU missing; a multi-node island also needs its ``NCCL_IB_HCA`` ports
  active, at ``min_ib_gbps`` of the node when the inventory sets it, else at no less than
  half the fastest job port of the island's nodes. A family without a checker, or a node
  with ``node_health: off``, is noted. A check that does not read (no answer, a timeout, a
  check still stuck on the node, an output cut off) refuses, unless ``--force`` (a WARNING);
* fabric group (``yeto.launch.fabric_groups``): an inventory that declares groups places
  a multi-node island in one group; one without any group only notes it;
* leases (``yeto.launch.node_lease.NodeLeases``): a node another run holds, or another
  island of this fleet, is left out (without ``lease_store``: a note). A lease of the
  island's own launch (``<prefix>`` or ``<prefix>-m<n>``, part ``trainer``) is its own. A
  lease that does not read refuses.

With ``lease_store`` the island's nodes are the ones its launch will take: the free nodes
(not held by another run) in the inventory's order, those this launch holds first, placed
by ``fabric_groups.pick_placement`` (``ssh_trainer_island.lease_trainer_nodes``). The
check leaves out the nodes held by others and the unhealthy ones, then places ``need``
nodes; none fit: refused, naming every node left out and why. A placement of the launch's
own rule that would take a node the check found unhealthy is refused too (the launch
picks by leases only).

:func:`check` returns what was refused, noted and warned, and the nodes the island takes;
``yeto fleet island add`` / ``move`` print the notes and refuse the change (exit 2,
nothing written) on any refusal, naming each.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

INVENTORY_FLAG = "--rl-node-inventory"
ISLAND_ENV_FLAG = "--island-env"
HEALTH_TIMEOUT_S = 120.0  # the SSH call of one check (the check bounds itself at node_health.HEALTH_LIMIT_S)
# the previous check of a node that has none: an XID cursor of no boot, so every hardware
# XID of this boot is new (``CudaChecker`` counts the XIDs after a reboot the same way)
JOIN_PREV: Mapping[str, Any] = {"xid": {"boot": None}}
RATE_FRACTION = 0.5  # a job port below this fraction of the island's fastest one is refused
XID_UNREAD = ("the kernel log does not read (dmesg restricted, no journalctl -k): add the SSH user to the adm or "
              "systemd-journal group, or let it run dmesg with sudo")
XID_HINT = (" (a hardware Xid in the kernel log of this boot: fix the GPU and reboot the node, a new boot starts a "
            "new log, then add the island again)")
_HARDWARE_XID = re.compile(r"\bXID (\d+)\b")


@dataclass
class NodeHealth:
    """One node's health for the join: ``state`` ``ok`` / ``bad`` / ``unread`` /
    ``unchecked``; ``text``: what to say after ``node <name>``; ``rates``: its job ports'
    rates (Gb/s; a multi-node island)."""

    state: str
    text: str = ""
    rates: dict[str, float] = field(default_factory=dict)


@dataclass
class JoinCheck:
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)  # the nodes the island takes (passed)
    health: dict[str, NodeHealth] = field(default_factory=dict)  # by node name (a recheck reuses it)

    @property
    def ok(self) -> bool:
        return not self.refusals


def inventory_of(argv: Sequence[str]) -> str | None:
    """The last ``--rl-node-inventory`` of a launch argv (the island's own wins over the fleet's)."""
    found = None
    for i, arg in enumerate(argv):
        if arg == INVENTORY_FLAG and i + 1 < len(argv):
            found = str(argv[i + 1])
        elif str(arg).startswith(INVENTORY_FLAG + "="):
            found = str(arg).split("=", 1)[1]
    return found


def without_inventory(argv: Sequence[str]) -> list[str]:
    """``argv`` without its ``--rl-node-inventory`` flags."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == INVENTORY_FLAG:
            skip = True
            continue
        if str(arg).startswith(INVENTORY_FLAG + "="):
            continue
        out.append(arg)
    return out


def island_env_of(argv: Sequence[str]) -> dict[str, str]:
    """The ``--island-env KEY=VALUE`` items of a launch argv (the last of a key wins): the
    island's own environment, which an ``ssh:`` island exports after the node's ``env``."""
    out: dict[str, str] = {}
    for i, arg in enumerate(argv):
        item = None
        if arg == ISLAND_ENV_FLAG and i + 1 < len(argv):
            item = str(argv[i + 1])
        elif str(arg).startswith(ISLAND_ENV_FLAG + "="):
            item = str(arg).split("=", 1)[1]
        if item and "=" in item:
            key, value = item.split("=", 1)
            out[key.strip()] = value
    return out


def island_nodes(argv: Sequence[str], gpu: str) -> tuple[Any, int, str | None]:
    """``(trainer nodes, nodes the island needs, inventory path)`` of an ``ssh:`` island's
    launch argv; raises ValueError naming what is wrong."""
    from ..gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from .node_inventory import island_nodes as role_nodes  # noqa: PLC0415
    from .node_inventory import load_inventory  # noqa: PLC0415

    (spec, *rest) = parse_gpu_spec(gpu)
    if rest or spec.cloud != "ssh":
        raise ValueError(f"--gpu {gpu}: one ssh: cluster expected")
    path = inventory_of(argv)
    if path is None:
        raise ValueError(f"an ssh: island needs its node inventory ({INVENTORY_FLAG}; the inventory option "
                         "of yeto fleet island add / move)")
    return role_nodes(load_inventory(path), "trainer"), int(spec.num_nodes), path


def node_keys(node: Any) -> set[str]:
    """What names the node's machine: its name, and its SSH target (without ``user@``) and
    ``address`` in lower case."""
    keys = {f"name:{node.name}"}
    for value in (str(getattr(node, "host", "") or "").rpartition("@")[2], getattr(node, "address", None)):
        value = str(value or "").strip().lower()
        if value:
            keys.add(f"host:{value}")
    return keys


def own_lease(holder: str | None, run_id: str) -> bool:
    """``holder`` is the island's own launch (``<run_id>`` or ``<run_id>-m<n>`` after a
    move), as a trainer."""
    if not holder:
        return False
    from .node_lease import run_of  # noqa: PLC0415

    run = run_of(holder) or ""
    part = str(holder).split("@", 1)[0].rpartition("/")[2]
    return part == "trainer" and re.fullmatch(rf"{re.escape(str(run_id))}(-m\d+)?", run) is not None


def island_of_lease(holder: str | None, islands: Mapping[str, str]) -> str | None:
    """The fleet island (``islands``: launch prefix -> island) whose launch holds the lease."""
    return next((name for prefix, name in islands.items() if own_lease(holder, prefix)), None)


# -- the modules of the checks (yeto.hw.node_health, fabric_groups, node_lease) ---------------------

def _checker_for(node: Any) -> Any:
    """The node's health checker, built as the island's own launch builds it (ssh_resident)."""
    from yeto.hw.node_health import checker_for  # noqa: PLC0415

    from .ssh_resident import RDMA_FABRICS  # noqa: PLC0415

    fabric = getattr(node, "fabric", None)
    return checker_for(node.device_family, expect_gpus=getattr(node, "cards_per_node", None),
                       min_ib_gbps=getattr(node, "min_ib_gbps", None),
                       fabric=fabric if fabric in RDMA_FABRICS else None)


def _check_placement(nodes: Sequence[Any], need: int, inventory: Sequence[Any] | None) -> tuple[bool, str, bool]:
    from .fabric_groups import check_placement  # noqa: PLC0415

    return check_placement(nodes, need, inventory=inventory)


def _pick_placement(nodes: Sequence[Any], need: int, inventory: Sequence[Any] | None) -> list[Any]:
    """``need`` of ``nodes`` as the launch's lease pick places them; raises PlacementError."""
    from .fabric_groups import pick_placement  # noqa: PLC0415

    return list(pick_placement(nodes, need, inventory=inventory).nodes)


def _grouped(inventory: Sequence[Any]) -> bool:
    from .fabric_groups import grouped  # noqa: PLC0415

    return grouped(inventory)


def _node_leases(store_uri: str, run_id: str) -> Any:
    from .node_lease import NodeLeases  # noqa: PLC0415

    return NodeLeases(store_uri, f"{run_id}/trainer")  # read only: who holds each node


def _hcas(node: Any, env: Mapping[str, str] | None = None) -> list[str]:
    """The job's ``NCCL_IB_HCA`` on the node: the island's environment over the node's (as
    the launch exports them, ``ssh_resident``)."""
    from yeto.hw.node_health import hcas_of  # noqa: PLC0415

    merged = {**dict(getattr(node, "env", None) or {}), **dict(env or {})}
    return list(hcas_of(merged.get("NCCL_IB_HCA")))


# -- health ---------------------------------------------------------------------------------------

def _absolute(result: Any, *, fm_read: bool = True) -> list[str]:
    """What a first check cannot see as a change but a node joining must not have: a
    volatile uncorrectable ECC count above 0, an NVSwitch node whose fabric manager is not
    active (the checker's facts)."""
    facts = result.facts or {}
    bus = dict(facts.get("bus") or {})
    out = [f"GPU {bus.get(key, key)} uncorrectable ECC {count} (volatile)"
           for key, count in sorted((facts.get("ecc") or {}).items(), key=lambda kv: str(bus.get(kv[0], kv[0])))
           if count is not None and int(count) > 0]
    if fm_read and facts.get("fabric") not in (None, "none") and facts.get("fm") != "active":  # NVSwitch: the FM runs
        out.append(f"nvidia-fabricmanager {facts.get('fm') or 'not running'} on an NVSwitch node")
    return out


def _unread_parts(text: str, result: Any) -> list[str]:
    """The parts of a whole check that the join's absolute rules need and that did not read:
    the kernel log (no XID source: ``dmesg`` restricted and no ``journalctl -k``), the NVLink
    fabric or the fabric manager timed out."""
    from yeto.hw.host_probe import parse_sections  # noqa: PLC0415

    out = []
    if (result.facts or {}).get("xid_source") == "none":
        out.append(XID_UNREAD)
    sections = parse_sections(text)
    for name in ("fabric", "fabricmanager"):
        sec = sections.get(name)
        if sec is not None and sec.rc in (124, 137):
            tool = "nvidia-smi -q (NVLink fabric)" if name == "fabric" else "systemctl is-active nvidia-fabricmanager"
            out.append(f"{tool} timed out")
    return out


def _job_rates(text: str, hcas: Sequence[str], fabric: str | None) -> dict[str, float]:
    """The rates of the job's active ports (``NCCL_IB_HCA``, else the fabric's link layer),
    read by the preflight's parsers (``yeto.hw.host_probe``)."""
    from yeto.hw.host_probe import (  # noqa: PLC0415
        hca_filter,
        link_layer_ports,
        natural_order,
        parse_sections,
        ports_of,
        section_ok,
    )

    sections = parse_sections(text)
    if not (section_ok(sections.get("ibstat")) or section_ok(sections.get("ibv_devinfo"))):
        return {}
    ports = natural_order(ports_of(sections))
    if hcas:
        picked, _ = hca_filter(",".join(hcas), ports)
    elif fabric:
        picked = link_layer_ports(ports, fabric)
    else:
        picked = []
    return {(p.ca if p.port == 1 else f"{p.ca}:{p.port}"): float(p.rate_gbps)
            for p in natural_order(picked) if p.active and p.rate_gbps}


def read_health(node: Any, *, multi: bool, ssh: Any, env: Mapping[str, str] | None = None) -> NodeHealth:
    """One node's health check (module doc), as :class:`NodeHealth`."""
    if getattr(node, "node_health", None) == "off":
        return NodeHealth("unchecked", "node_health off in the inventory: its health is not checked")
    try:
        checker = _checker_for(node)
        if not getattr(checker, "enabled", True):
            return NodeHealth("unchecked", f"no health check for device family {node.device_family}")
        hcas = _hcas(node, env) if multi else []
        done = ssh.run(node, checker.script(hcas), timeout=HEALTH_TIMEOUT_S)
    except Exception as error:  # noqa: BLE001 - named in the refusal
        return NodeHealth("unread", f"its health check could not run ({type(error).__name__}: {error})"[:300])
    if done.returncode != 0:
        last = [line for line in str(done.stderr or "").splitlines() if line.strip()][-1:]
        return NodeHealth("unread", f"its health check did not answer (ssh exit {done.returncode}"
                          + (f": {last[0].strip()}" if last else "") + ")")
    text = str(done.stdout or "")
    result = checker.parse(text, hcas, dict(JOIN_PREV))
    if not result.checked:
        return NodeHealth("unread", "its health check did not read ("
                          + ("; ".join(result.reasons) or "no reason given") + ")")
    ok = result.ok(multi)
    unread = _unread_parts(text, result)
    reasons = (list(result.job_reasons(multi)) if ok is False else []) + _absolute(
        result, fm_read=not any("nvidia-fabricmanager" in part for part in unread))
    if reasons:
        hint = XID_HINT if any(_HARDWARE_XID.search(r) for r in reasons) else ""
        return NodeHealth("bad", "is not healthy: " + "; ".join(dict.fromkeys(reasons)) + hint)
    if ok is None:  # the GPUs pass, the job's ports did not read (multi-node)
        return NodeHealth("unread", "its health check did not read (" + ("; ".join(result.reasons) or "unknown")
                          + ")")
    if unread:  # a part the absolute rules need did not read
        return NodeHealth("unread", f"its health check did not read ({'; '.join(unread)})")
    rates = _job_rates(text, hcas, getattr(checker, "fabric", None)) if multi else {}
    return NodeHealth("ok", rates=rates)


def judge_rates(nodes: Sequence[Any], health: dict[str, NodeHealth], *, inventory: Sequence[Any] | None = None,
                candidates: bool = False) -> None:
    """A multi-node island's job port rates (in place): on a node without ``min_ib_gbps``
    (its checker holds the ports to that one), a port below half the fastest job port of
    the nodes of its fabric group (one group: an inventory without ``fabric_group``) makes
    the node bad. ``candidates``: ``nodes`` are the island's candidates (a shared inventory)."""
    by_group: dict[str, list[Any]] = {}
    grouped = _grouped(list(nodes if inventory is None else inventory))
    for node in nodes:
        by_group.setdefault(str(node.fabric_group or "") if grouped else "", []).append(node)
    whose = "candidate nodes" if candidates else "nodes"
    for group, members in by_group.items():
        fastest = max((r for n in members for r in (health.get(n.name) or NodeHealth("")).rates.values()),
                      default=None)
        if fastest is None:
            continue
        where = f" in fabric group {group}" if grouped and group else " without a fabric group" if grouped else ""
        base = float(fastest) * RATE_FRACTION
        for node in members:
            h = health.get(node.name)
            # a node with min_ib_gbps: its checker held its ports to it already (CudaChecker)
            if h is None or h.state != "ok" or not h.rates or getattr(node, "min_ib_gbps", None) is not None:
                continue
            slow = [f"{label} rate {rate:g} < {base:g} Gb/s (half the fastest job port of the island's {whose}{where}, "
                    f"{fastest:g} Gb/s)" for label, rate in h.rates.items() if rate < base * 0.999]
            if slow:
                health[node.name] = NodeHealth("bad", "is not healthy: " + "; ".join(slow), h.rates)


def check_health(nodes: Sequence[Any], *, multi: bool, ssh: Any, out: JoinCheck,
                 env: Mapping[str, str] | None = None, cache: Mapping[str, NodeHealth] | None = None,
                 inventory: Sequence[Any] | None = None, candidates: bool = False) -> None:
    """The health of ``nodes`` into ``out.health`` (a node in ``cache`` is not checked again)."""
    for node in nodes:
        if node.name in out.health:
            continue
        cached = (cache or {}).get(node.name)
        out.health[node.name] = cached if cached is not None else read_health(node, multi=multi, ssh=ssh, env=env)
    if multi:
        judge_rates(nodes, out.health, inventory=inventory, candidates=candidates)


def _health_problem(name: str, h: NodeHealth, *, force: bool, out: JoinCheck) -> str | None:
    """The problem a node's health makes (None: none); notes and warnings into ``out``."""
    if h.state == "unchecked":
        out.notes.append(f"node {name}: {h.text}")
        return None
    if h.state == "unread" and force:
        out.warnings.append(f"node {name}: {h.text}; --force: it joins unchecked")
        return None
    return None if h.state == "ok" else f"node {name} {h.text}" if h.state == "bad" else f"node {name}: {h.text}"


# -- fabric and leases ----------------------------------------------------------------------------

def check_fabric(nodes: Sequence[Any], need: int, out: JoinCheck, *, inventory: Sequence[Any] | None = None) -> None:
    """``inventory``: the whole inventory (whether it declares fabric groups at all)."""
    ok, message, hint_only = _check_placement(nodes, need, inventory)
    if not ok and not hint_only:
        out.refusals.append(f"fabric: {message}")
    elif message and (hint_only or not ok):
        out.notes.append(f"fabric: {message}")


def lease_holders(nodes: Sequence[Any], *, lease_store: str, run_id: str,
                  out: JoinCheck) -> dict[str, str | None] | None:
    """Who holds each node's lease (None: the store did not read: refused into ``out``)."""
    try:
        leases = _node_leases(lease_store, run_id)
    except Exception as error:  # noqa: BLE001
        out.refusals.append(f"lease store {lease_store}: {type(error).__name__}: {error}"[:300])
        return None
    holders: dict[str, str | None] = {}
    for node in nodes:
        try:
            holders[node.name] = leases.holder(node)
        except Exception as error:  # noqa: BLE001 - unknown is not free
            out.refusals.append(f"node {node.name}: its lease does not read ({type(error).__name__}: {error})"[:300])
    return None if len(holders) < len(nodes) else holders


def lease_problem(holder: str | None, *, run_id: str, islands: Mapping[str, str]) -> str | None:
    """Why a node held by ``holder`` is not the island's (None: free, or its own)."""
    if holder is None or own_lease(holder, run_id):
        return None
    island = island_of_lease(holder, islands)
    if island is not None:
        return f"is island {island}'s (its launch holds the lease)"
    from .node_lease import run_of  # noqa: PLC0415

    return f"is leased to run {run_of(holder) or holder}"


def check_leases(nodes: Sequence[Any], *, lease_store: str | None, run_id: str, out: JoinCheck,
                 islands: Mapping[str, str] | None = None) -> None:
    """Every one of ``nodes`` free for the island (or its own) in ``lease_store``."""
    if not lease_store:
        out.notes.append("no lease_store in the inventory: whether another run uses these nodes is not checked")
        return
    holders = lease_holders(nodes, lease_store=lease_store, run_id=run_id, out=out)
    for node in nodes if holders is not None else ():
        why = lease_problem(holders[node.name], run_id=run_id, islands=islands or {})
        if why:
            out.refusals.append(f"node {node.name} {why}")


# -- the check ------------------------------------------------------------------------------------

def check(nodes: Sequence[Any], *, need: int, run_id: str, lease_store: str | None = None,
          taken: Mapping[str, str] | None = None, islands: Mapping[str, str] | None = None, ssh: Any = None,
          inventory: Sequence[Any] | None = None, env: Mapping[str, str] | None = None, force: bool = False,
          health: Mapping[str, NodeHealth] | None = None) -> JoinCheck:
    """The checks of a new island's trainer ``nodes`` (module doc). ``taken``: a node key
    (:func:`node_keys`) -> the fleet island whose inventory lists it (an inventory without
    ``lease_store``); ``islands``: the fleet's other islands' launch prefix -> island (whose
    leases are theirs); ``inventory``: the whole inventory (default ``nodes``); ``env``: the
    island's own environment; ``force``: a check that does not read is a warning;
    ``health``: node health already read (a recheck). With ``lease_store``, ``taken`` holds the
    nodes of the islands whose inventory has no ``lease_store`` or another one (their leases
    are not in this store)."""
    out = JoinCheck()
    if ssh is None:
        from .ssh_trainer_island import SshOps  # noqa: PLC0415

        ssh = SshOps()
    every = list(nodes if inventory is None else inventory)
    if not lease_store:
        _check_listed(nodes, need=need, taken=taken or {}, ssh=ssh, env=env, force=force, health=health, out=out,
                      inventory=every)
    else:
        _check_leased(nodes, need=need, run_id=run_id, lease_store=lease_store, islands=islands or {}, ssh=ssh,
                      env=env, force=force, health=health, out=out, inventory=every, taken=taken or {})
    if out.refusals:
        out.nodes = []
    return out


def _check_listed(nodes: Sequence[Any], *, need: int, taken: Mapping[str, str], ssh: Any, env: Any, force: bool,
                  health: Any, out: JoinCheck, inventory: Sequence[Any]) -> None:
    """An inventory without ``lease_store``: its trainer nodes are the island's."""
    if len(nodes) != need:
        out.refusals.append(f"the island needs {need} trainer node(s); its inventory lists {len(nodes)} "
                            f"{[n.name for n in nodes]}")
        return
    for node in nodes:
        owner = next((taken[k] for k in sorted(node_keys(node)) if k in taken), None)
        if owner:
            out.refusals.append(f"node {node.name} is island {owner}'s")
    check_health(nodes, multi=need > 1, ssh=ssh, out=out, env=env, cache=health, inventory=inventory)
    for node in nodes:
        problem = _health_problem(node.name, out.health[node.name], force=force, out=out)
        if problem:
            out.refusals.append(problem)
    try:
        check_fabric(nodes, need, out, inventory=inventory)
    except Exception as error:  # noqa: BLE001 - the check that cannot run refuses
        out.refusals.append(f"fabric: the placement check could not run ({type(error).__name__}: {error})"[:300])
    check_leases(nodes, lease_store=None, run_id="", out=out)
    out.nodes = [n.name for n in nodes]


def _owner(node: Any, taken: Mapping[str, str]) -> str | None:
    return next((taken[k] for k in sorted(node_keys(node)) if k in taken), None)


def _check_leased(nodes: Sequence[Any], *, need: int, run_id: str, lease_store: str, islands: Mapping[str, str],
                  ssh: Any, env: Any, force: bool, health: Any, out: JoinCheck, inventory: Sequence[Any],
                  taken: Mapping[str, str]) -> None:
    """An inventory with ``lease_store``: ``need`` of its trainer nodes, as the launch picks them.

    The launch (``lease_trainer_nodes`` / ``pick_nodes``) runs under a new holder (a new
    run dir: a new identity, ``<prefix>-m<n>`` after a move), so no node is its own yet: it
    places the island on the free nodes, in the inventory's order; with too few it waits,
    and once the island's earlier incarnation has let its nodes go it places on the free
    nodes and those. Each of these picks that fits must take healthy nodes only."""
    if len(nodes) < need:
        out.refusals.append(f"the island needs {need} trainer node(s); its inventory lists {len(nodes)} "
                            f"{[n.name for n in nodes]}")
        return
    holders = lease_holders(nodes, lease_store=lease_store, run_id=run_id, out=out)
    if holders is None:
        return
    held: dict[str, str] = {}
    for node in nodes:
        owner = _owner(node, taken)
        why = f"is island {owner}'s" if owner else lease_problem(holders[node.name], run_id=run_id, islands=islands)
        if why:
            held[node.name] = why
    usable = [n for n in nodes if n.name not in held]  # free, or held by the island's earlier incarnation
    free = [n for n in usable if holders[n.name] is None]
    check_health(usable, multi=need > 1, ssh=ssh, out=out, env=env, cache=health, inventory=inventory,
                 candidates=True)
    sick = {}
    for node in usable:
        problem = _health_problem(node.name, out.health[node.name], force=force, out=out)
        if problem:
            sick[node.name] = problem
    left_out = [f"node {n.name} {held[n.name]}" if n.name in held else sick[n.name]
                for n in nodes if n.name in held or n.name in sick]
    grouped = need > 1 and _grouped(inventory)
    if need > 1 and not grouped:
        from .fabric_groups import HINT  # noqa: PLC0415

        out.notes.append(f"fabric: {HINT}")
    try:
        _pick_placement([n for n in usable if n.name not in sick], need, inventory)
    except Exception as error:  # noqa: BLE001 - no fit (PlacementError), or the check that cannot run
        fit = type(error).__name__ == "PlacementError"
        out.refusals += left_out + [(f"fabric: {error}" if grouped else
                                     f"{len(usable) - len(sick)} of {len(nodes)} trainer node(s) are free and "
                                     f"healthy, {need} needed") if fit else
                                    f"fabric: the placement check could not run ({type(error).__name__}: {error})"]
        return
    picks: list[list[Any]] = []  # the launch's picks: the free nodes; then with its earlier incarnation's
    for offered in ([free, usable] if len(free) < len(usable) else [free]):
        try:
            picks.append(_pick_placement(offered, need, inventory))
        except Exception:  # noqa: BLE001 - too few of them: the launch waits for the next ones
            continue
    bad = list(dict.fromkeys(n.name for pick in picks for n in pick if n.name in sick))
    if bad:
        out.refusals += [sick[name] for name in bad] + [
            f"the island's launch picks its nodes by their leases and would take {', '.join(bad)}; take "
            "such a node out of the inventory's trainer nodes (roles) or clear its fault"]
        return
    out.notes += [f"left out: {why}" for why in left_out]
    out.nodes = [n.name for n in picks[0]]
    if len(picks) > 1 and [n.name for n in picks[1]] != out.nodes:
        out.notes.append(f"the island takes {names(out.nodes)} (free now); were they taken meanwhile, its launch "
                         f"waits for its earlier incarnation's nodes and takes {names(n.name for n in picks[1])} "
                         "(checked too)")


def taken_nodes(doc: Mapping[str, Any], *, other_than: str, fleet_argv: Sequence[str],
                lease_store: str | None = None) -> dict[str, str]:
    """Node key (:func:`node_keys`) -> island, for the ``ssh:`` trainer islands of the desired
    state other than ``other_than`` (their inventories' trainer nodes). ``lease_store``: the
    new island's; an island whose inventory has the same one is left out (its leases say
    which nodes it holds)."""
    from yeto.launch.pool import desired as dz  # noqa: PLC0415

    from .node_inventory import island_nodes as role_nodes  # noqa: PLC0415
    from .node_inventory import load_inventory  # noqa: PLC0415

    out: dict[str, str] = {}
    for name, member in dz.trainers(dict(doc)).items():
        if name == other_than or not str(member.get("gpu") or "").startswith("ssh:"):
            continue
        path = inventory_of([*fleet_argv, *(member.get("argv") or ())])
        if path is None:
            continue
        try:
            listed = load_inventory(path)
        except Exception as error:  # noqa: BLE001 - its nodes unknown: said, not taken as free
            raise ValueError(f"island {name}'s inventory {path} does not read ({error})") from None
        if lease_store is not None and getattr(listed, "lease_store", None) == lease_store:
            continue
        for node in role_nodes(listed, "trainer"):
            for key in node_keys(node):
                out.setdefault(key, name)
    return out


def fleet_prefixes(doc: Mapping[str, Any], *, other_than: str) -> dict[str, str]:
    """Launch prefix -> island, for the trainer islands of the desired state other than ``other_than``."""
    from yeto.launch.pool import desired as dz  # noqa: PLC0415

    return {str(m["prefix"]): name for name, m in dz.trainers(dict(doc)).items()
            if name != other_than and m.get("prefix")}


def check_member(member: Mapping[str, Any], doc: Mapping[str, Any], *, fleet_argv: Sequence[str], run_id: str,
                 ssh: Any = None, log: Callable[[str], None] | None = None, force: bool = False,
                 health: Mapping[str, NodeHealth] | None = None, seen: Iterable[str] = ()) -> JoinCheck:
    """:func:`check` of an island's trainer member (added, or moved) against the desired
    state ``doc`` (its other islands). ``seen``: notes and warnings printed already (a
    recheck prints only the new ones)."""
    say = log or (lambda m: print(m, file=sys.stderr))
    argv = [*fleet_argv, *(member.get("argv") or ())]
    from .node_inventory import lease_store_of, load_inventory  # noqa: PLC0415

    nodes, need, path = island_nodes(argv, str(member["gpu"]))
    inventory = load_inventory(path)
    store = lease_store_of(inventory)
    name = str(member["name"])
    result = check(nodes, need=need, run_id=run_id, lease_store=store,
                   taken=taken_nodes(doc, other_than=name, fleet_argv=fleet_argv, lease_store=store),
                   islands=fleet_prefixes(doc, other_than=name), ssh=ssh, inventory=inventory,
                   env=island_env_of(argv), force=force, health=health)
    shown = set(seen)
    for note in result.notes:
        if note not in shown:
            say(f"fleet island: note: island {name}: {note}")
    for warning in result.warnings:
        if warning not in shown:
            say(f"fleet island: WARNING: island {name}: {warning}")
    return result


def names(nodes: Iterable[str]) -> str:
    return ", ".join(nodes) or "none"
