"""The checks a new ``ssh:`` trainer island's nodes pass before ``yeto fleet island add``
writes it into a running fleet (own-cluster-fleet-stability 9.3).

An island added at run time takes the trainer nodes of its own inventory (the inventory
option of ``yeto fleet island add``: its launch's ``--rl-node-inventory``). Before anything is written
(and in the dry run), those nodes must:

* be as many as the island's ``--gpu`` says, with no node another ``ssh:`` island of the
  fleet already trains on (same name or SSH target);
* be healthy: the node health checker of their device family (``yeto.hw.node_health``,
  built as the island's launch builds it), one SSH call per node; a multi-node island also
  needs its ``NCCL_IB_HCA`` ports up at rate, a single-node one only its GPUs. A device
  family without a checker, or a node with ``node_health: off``, is noted, not refused; a
  node that does not answer is refused;
* fit one fabric group (``yeto.launch.fabric_groups.check_placement``): an inventory that
  declares groups refuses a multi-node island across groups; one without any group only
  notes it;
* be free in the inventory's ``lease_store`` (``yeto.launch.node_lease.NodeLeases``): a node
  another run holds (another island of the fleet included) is refused, naming that run; a
  lease of the island's own launch (its member prefix, ``-m<n>`` after a move) is its own.
  Without a ``lease_store`` nothing is checked (noted once).

:func:`check` returns what was refused and noted; ``yeto fleet island add`` prints the notes
and refuses the change (exit 2, nothing written) on any refusal, naming each.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

INVENTORY_FLAG = "--rl-node-inventory"
HEALTH_TIMEOUT_S = 120.0


@dataclass
class JoinCheck:
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

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
                         "of yeto fleet island add)")
    return role_nodes(load_inventory(path), "trainer"), int(spec.num_nodes), path


# -- the three checks (their modules: yeto.hw.node_health, fabric_groups, node_lease) ------------

def _checker_for(node: Any) -> Any:
    """The node's health checker, built as the island's own launch builds it (ssh_resident)."""
    from yeto.hw.node_health import checker_for  # noqa: PLC0415

    from .ssh_resident import RDMA_FABRICS  # noqa: PLC0415

    fabric = getattr(node, "fabric", None)
    return checker_for(node.device_family, expect_gpus=getattr(node, "cards_per_node", None),
                       fabric=fabric if fabric in RDMA_FABRICS else None)


def _check_placement(nodes: Sequence[Any], need: int) -> tuple[bool, str, bool]:
    from .fabric_groups import check_placement  # noqa: PLC0415

    return check_placement(nodes, need)


def _node_leases(store_uri: str, run_id: str) -> Any:
    from .node_lease import NodeLeases  # noqa: PLC0415

    return NodeLeases(store_uri, f"{run_id}/trainer")  # read only: who holds each node


def _hcas(node: Any) -> list[str]:
    from yeto.hw.node_health import hcas_of  # noqa: PLC0415

    return list(hcas_of((getattr(node, "env", None) or {}).get("NCCL_IB_HCA")))


def _lease_run(holder: str) -> str:
    from .node_lease import run_of  # noqa: PLC0415

    return run_of(holder) or str(holder)


def check_health(nodes: Sequence[Any], *, multi: bool, ssh: Any, out: JoinCheck) -> None:
    for node in nodes:
        if getattr(node, "node_health", None) == "off":
            out.notes.append(f"node {node.name}: node_health off in the inventory: its health is not checked")
            continue
        try:
            checker = _checker_for(node)
            if not getattr(checker, "enabled", True):
                out.notes.append(f"node {node.name}: no health check for device family {node.device_family}")
                continue
            hcas = _hcas(node) if multi else []
            done = ssh.run(node, checker.script(hcas), timeout=HEALTH_TIMEOUT_S)
        except Exception as error:  # noqa: BLE001 - named in the refusal
            out.refusals.append(f"node {node.name}: its health check could not run ({type(error).__name__}: "
                                f"{error})"[:300])
            continue
        if done.returncode != 0:
            last = [line for line in str(done.stderr or "").splitlines() if line.strip()][-1:]
            out.refusals.append(f"node {node.name}: its health check did not answer (ssh exit {done.returncode}"
                                + (f": {last[0].strip()}" if last else "") + ")")
            continue
        result = checker.parse(str(done.stdout or ""), hcas, None)
        if not result.checked:
            out.notes.append(f"node {node.name}: no health check for device family {node.device_family}")
            continue
        ok, reasons = result.ok(multi), result.job_reasons(multi)  # a single-node island: its GPU reasons
        if ok is False:
            out.refusals.append(f"node {node.name} is not healthy: {'; '.join(reasons) or 'no reason given'}")
        elif ok is None:
            out.notes.append(f"node {node.name}: its health could not be judged ({'; '.join(result.reasons)})")


def check_fabric(nodes: Sequence[Any], need: int, out: JoinCheck) -> None:
    ok, message, hint_only = _check_placement(nodes, need)
    if not ok and not hint_only:
        out.refusals.append(f"fabric: {message}")
    elif message and (hint_only or not ok):
        out.notes.append(f"fabric: {message}")


def check_leases(nodes: Sequence[Any], *, lease_store: str | None, run_id: str, out: JoinCheck) -> None:
    if not lease_store:
        out.notes.append("no lease_store in the inventory: whether another run uses these nodes is not checked")
        return
    try:
        leases = _node_leases(lease_store, run_id)
    except Exception as error:  # noqa: BLE001
        out.refusals.append(f"lease store {lease_store}: {type(error).__name__}: {error}"[:300])
        return
    for node in nodes:
        try:
            holder = leases.holder(node)
        except Exception as error:  # noqa: BLE001 - unknown is not free
            out.refusals.append(f"node {node.name}: its lease does not read ({type(error).__name__}: {error})"[:300])
            continue
        run = None if holder is None else _lease_run(holder)
        # the island's own launch leases as <prefix>[-m<n>]/trainer: a lease of it is its own
        if run is not None and run != run_id and not run.startswith(f"{run_id}-m"):
            out.refusals.append(f"node {node.name} is leased to run {run}")


def check(nodes: Sequence[Any], *, need: int, run_id: str, lease_store: str | None = None,
          taken: Mapping[str, str] | None = None, ssh: Any = None) -> JoinCheck:
    """The checks of a new island's trainer ``nodes`` (module doc). ``taken``: node name or SSH
    target -> the fleet island that trains on it."""
    out = JoinCheck()
    if len(nodes) != need:
        out.refusals.append(f"the island needs {need} trainer node(s); its inventory lists {len(nodes)} "
                            f"{[n.name for n in nodes]}")
        return out
    for node in nodes:
        owner = (taken or {}).get(node.name) or (taken or {}).get(getattr(node, "host", None) or "")
        if owner:
            out.refusals.append(f"node {node.name} is island {owner}'s")
    if ssh is None:
        from .ssh_trainer_island import SshOps  # noqa: PLC0415

        ssh = SshOps()
    check_health(nodes, multi=need > 1, ssh=ssh, out=out)
    try:
        check_fabric(nodes, need, out)
    except Exception as error:  # noqa: BLE001 - the check that cannot run refuses
        out.refusals.append(f"fabric: the placement check could not run ({type(error).__name__}: {error})"[:300])
    check_leases(nodes, lease_store=lease_store, run_id=run_id, out=out)
    return out


def taken_nodes(doc: Mapping[str, Any], *, other_than: str, fleet_argv: Sequence[str]) -> dict[str, str]:
    """Node name and SSH target -> island, for the ``ssh:`` trainer islands of the desired
    state other than ``other_than`` (their inventories' trainer nodes)."""
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
            nodes = role_nodes(load_inventory(path), "trainer")
        except Exception as error:  # noqa: BLE001 - its nodes unknown: said, not taken as free
            raise ValueError(f"island {name}'s inventory {path} does not read ({error})") from None
        for node in nodes:
            out.setdefault(node.name, name)
            out.setdefault(node.host, name)
    return out


def check_member(member: Mapping[str, Any], doc: Mapping[str, Any], *, fleet_argv: Sequence[str], run_id: str,
                 ssh: Any = None, log: Callable[[str], None] | None = None) -> JoinCheck:
    """:func:`check` of an added island's trainer member (``yeto fleet island add``)."""
    say = log or (lambda m: print(m, file=sys.stderr))
    argv = [*fleet_argv, *(member.get("argv") or ())]
    from .node_inventory import lease_store_of, load_inventory  # noqa: PLC0415

    nodes, need, path = island_nodes(argv, str(member["gpu"]))
    result = check(nodes, need=need, run_id=run_id, lease_store=lease_store_of(load_inventory(path)),
                   taken=taken_nodes(doc, other_than=str(member["name"]), fleet_argv=fleet_argv), ssh=ssh)
    for note in result.notes:
        say(f"fleet island: note: island {member['name']}: {note}")
    return result
