"""``yeto cluster``: the machines of an SSH node inventory.

``yeto cluster probe --inventory <file>`` groups the nodes by fabric
(:mod:`yeto.launch.fabric_groups`). It runs the cross-node steps of ``yeto
preflight`` over SSH, inside the trainer image (``--image``, or the
inventory's): ``ib_write_bw`` rail by rail between every pair of nodes and the
NCCL collectives on every pair (``--nccl-pairs disjoint``: only disjoint pairs,
a pair without its own run counts when both nodes ran ``NET/IB`` elsewhere). It
runs on idle nodes only: a node another run's lease holds, a node whose idleness
cannot be confirmed (no answer, GPU processes or containers not listable), a node
with a ``yeto-`` container, and a node with a GPU process each refuse the probe
(exit 3). ``--force`` probes the unconfirmed nodes and those with containers,
``--force-busy-gpu`` those with GPU processes; a lease is never overridden. It
prints each group, the missing
edges inside a connected component, and every node that is "fabric unavailable"
with the reasons. Only ``--write`` writes the groups into the inventory
(``fabric_group`` of every probed node; ``unavailable`` for a node without RDMA
to the others); without it no file is written. ``--dry-run`` prints what the
probe would run and how long it may take, without SSH.

``yeto cluster check --inventory <file> [--need N]`` reads the groups the
inventory declares and, with ``--need``, where an N-node island would go.

Exit 0 when every probed node is in a group and no edge is missing, 1 when a
node is fabric unavailable, an edge is missing or the deadline cut the probe
short, 2 on a usage or inventory error, 3 when a node is busy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .launch import fabric_groups as fg
from .launch.node_inventory import (FABRIC_UNAVAILABLE, FABRIC_UNGROUPED, InventoryError, lease_store_of,
                                    load_inventory)

NO_IMAGE = "the docker nodes have no image: pass --image (the trainer image: ib_write_bw, torch with NCCL)"
GPU_APPS = "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"
EXIT_BUSY = 3


def containers_command(engine: str) -> str:
    """The ``yeto-`` containers of ``engine`` (``docker`` / ``podman``), listed as the node
    lease probe lists them (:func:`yeto.launch.node_lease.still_runs_script`): as the user
    and, when ``sudo -n`` works, through it. docker has one daemon: either view listing is
    enough, and a daemon that is not running (not "permission denied") has no containers;
    podman's rootless containers are the user's own: that view must list, the rootful
    one only adds. Exit 0 when the listing is complete; an engine not installed lists
    none."""
    fmt = "--format '{{.Names}} {{.Image}} {{.Status}}'"
    need = '[ $rc = 0 ] || [ $src = 0 ]' if engine == "docker" else '[ $rc = 0 ]'
    return (
        f"command -v {engine} >/dev/null 2>&1 || exit 0\n"
        "err=$(mktemp 2>/dev/null || echo \"/tmp/yeto-idle.$$\")\n"
        f"out=$({engine} ps --filter name=yeto- {fmt} 2>\"$err\"); rc=$?\n"
        "if [ $rc != 0 ] && ! grep -qi 'permission denied' \"$err\" "
        "&& grep -q 'Cannot connect to the Docker daemon' \"$err\"; then rc=0; out=; fi\n"
        "sout=; src=1\n"
        "if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then\n"
        f"  sout=$(sudo -n {engine} ps --filter name=yeto- {fmt} 2>/dev/null); src=$?\n"
        "fi\n"
        "rm -f \"$err\"\n"
        "printf '%s\\n%s\\n' \"$out\" \"$sout\" | grep -v '^$' | sort -u\n"
        f"{need}"
    )


def idle_sections(node: Any) -> list[tuple[str, str]]:
    """The idle check of ``node``: its GPU processes, and the ``yeto-`` containers of its
    container engine (the node profile's ``container``: docker, podman; none for a
    process node)."""
    from .launch.node_inventory import profile_of

    engine = profile_of(node).container
    sections = [("compute_apps", GPU_APPS)]
    if engine in ("docker", "podman"):
        sections.append(("yeto_containers", containers_command(engine)))
    return sections


def add_parser(sub: Any) -> argparse.ArgumentParser:
    doc = (__doc__ or "").split("\n\n")
    p = sub.add_parser("cluster", help="the machines of an SSH node inventory: fabric groups (probe, check)",
                       description=doc[0])
    csub = p.add_subparsers(dest="cluster_command", metavar="{probe,check}")
    csub.required = True
    probe = csub.add_parser(
        "probe", help="group the inventory's nodes by fabric (ib_write_bw + NCCL transport between nodes)",
        description=doc[1])
    probe.add_argument("--inventory", required=True, help="the SSH node inventory (YAML / JSON)")
    probe.add_argument("--nodes", default=None, help="probe only these nodes (comma-separated names)")
    probe.add_argument("--image", default=None, help="the trainer image (default: the inventory's)")
    probe.add_argument("--no-pull", action="store_true", help="do not docker pull the image")
    probe.add_argument("--nccl-pairs", choices=("all", "disjoint"), default="all",
                       help="NCCL on every node pair (default) or on disjoint pairs only (faster, less strict)")
    probe.add_argument("--ib-seconds", type=int, default=3, help="ib_write_bw duration per pair")
    probe.add_argument("--nccl-sizes", default="16M,128M", help="message sizes of the collectives")
    probe.add_argument("--nccl-iters", type=int, default=10)
    probe.add_argument("--max-parallel", type=int, default=16,
                       help="nodes / node pairs probed at once (at least 2)")
    probe.add_argument("--deadline", type=float, default=3600.0,
                       help="seconds for the whole probe; a cut-short probe writes nothing")
    probe.add_argument("--force", action="store_true",
                       help="probe nodes whose idleness is not confirmed or that run yeto- containers")
    probe.add_argument("--force-busy-gpu", action="store_true",
                       help="probe nodes that run GPU compute processes too (with a WARN)")
    probe.add_argument("--write", action="store_true", help="write fabric_group into the inventory file")
    probe.add_argument("--dry-run", action="store_true", help="print the plan and its time; no SSH, no file")
    probe.add_argument("--json", action="store_true", help="print the result as JSON")
    check = csub.add_parser("check", help="the fabric groups the inventory declares; where an N-node island goes",
                            description=doc[2])
    check.add_argument("--inventory", required=True, help="the SSH node inventory (YAML / JSON)")
    check.add_argument("--nodes", default=None, help="the candidate nodes (comma-separated names; default all)")
    check.add_argument("--need", type=int, default=None, help="place an island of this many nodes")
    return p


def _say(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(args: argparse.Namespace, *, ops: Any = None, log: Any = None, clock: Any = time.monotonic,
         leases: Any = None) -> int:
    command = getattr(args, "cluster_command", None)
    if command == "probe":
        return probe_main(args, ops=ops, log=log, clock=clock, leases=leases)
    if command == "check":
        return check_main(args)
    _say(f"[yeto] cluster: unknown command {command!r}")
    return 2


def _select(inventory: list, spec: str | None) -> list:
    names = [n.strip() for n in (spec or "").split(",") if n.strip()]
    by_name = {n.name: n for n in inventory}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise InventoryError(f"--nodes {missing} are not in the inventory")
    return [by_name[n] for n in names] if names else list(inventory)


# ---------------------------------------------------------------- check
def check_main(args: argparse.Namespace) -> int:
    try:
        inventory = load_inventory(args.inventory)
        nodes = _select(inventory, args.nodes)
    except InventoryError as exc:
        _say(f"[yeto] cluster check: {exc}")
        return 2
    groups: dict[str, list[str]] = {}
    for n in nodes:
        if fg.placeable(n):
            groups.setdefault(str(n.fabric_group), []).append(n.name)
    for g, members in sorted(groups.items()):
        print(f"fabric group {g}: {', '.join(sorted(members))} ({len(members)} node(s))")
    unavailable = [n.name for n in nodes if n.fabric_group == FABRIC_UNAVAILABLE]
    if unavailable:
        print(f"{fg.UNAVAILABLE}: {', '.join(unavailable)} (single-node islands only)")
    if not fg.grouped(inventory):
        _say(f"[yeto] cluster check: note: {fg.HINT}")
    else:
        ungrouped = [n.name for n in nodes if n.fabric_group in (None, FABRIC_UNGROUPED)]
        if ungrouped:
            print(f"no fabric_group: {', '.join(ungrouped)} (single-node islands only)")
    if args.need is None:
        return 0
    try:
        placement = fg.pick_placement(nodes, args.need, inventory=inventory)
    except fg.PlacementError as exc:
        print(f"placement of {args.need} node(s): refused: {exc}")
        return 1
    where = f"group {placement.group}" if placement.group else "no fabric group"
    print(f"placement of {args.need} node(s): {where}: {', '.join(n.name for n in placement.nodes)}")
    return 0


# ---------------------------------------------------------------- probe
class _DeadlineOps:
    """``ops`` whose every SSH call ends by the probe's deadline; ``hit``: the deadline
    passed during the probe (a call refused, cut or returned after it)."""

    def __init__(self, ops: Any, deadline_at: float, clock: Any) -> None:
        self._ops, self._at, self._clock = ops, deadline_at, clock
        self.hit = False

    def run(self, node: Any, script: str, *, timeout: float = 300.0) -> Any:
        left = self._at - self._clock()
        if left <= 0:
            self.hit = True
            raise TimeoutError("the probe's --deadline passed")
        try:
            return self._ops.run(node, script, timeout=min(timeout, left))
        finally:
            if self._at - self._clock() <= 0:
                self.hit = True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ops, name)


# what keeps a node from the probe, and which flag lets it through (None: nothing does)
OVERRIDE = {"lease": None, "unconfirmed": "--force", "container": "--force", "gpu": "--force-busy-gpu"}


def busy_nodes(nodes: list, ops: Any, *, max_parallel: int = 16,
               leases: Any = None) -> dict[str, list[tuple[str, str]]]:
    """Node -> what keeps it from the probe, as ``(kind, what)``: ``lease`` (another run
    holds it), ``unconfirmed`` (no answer, or its GPU processes or containers could not
    be listed), ``container`` (a ``yeto-`` container), ``gpu`` (a GPU compute process)."""
    from .launch.preflight.cluster import parse_sections, probe_script

    def one(node: Any) -> tuple[Any, list[tuple[str, str]]]:
        found: list[tuple[str, str]] = []
        if leases is not None:
            try:
                holder = leases.holder(node)
            except Exception as exc:  # noqa: BLE001 - the store did not answer: not confirmed
                found.append(("unconfirmed", f"cannot read its lease ({type(exc).__name__}: {exc})"))
            else:
                if holder:
                    found.append(("lease", f"leased by {holder}"))
        sections = idle_sections(node)
        try:
            done = ops.run(node, probe_script(sections), timeout=60.0)
            text = (done.stdout or "") + (("\n" + done.stderr) if getattr(done, "stderr", "") else "")
            got = parse_sections(text)
        except Exception as exc:  # noqa: BLE001 - no answer: idleness not confirmed
            return node, found + [("unconfirmed", f"cannot confirm it is idle: no answer ({type(exc).__name__}: {exc})")]
        if not got:
            tail = " ".join(text.split())[-160:]
            return node, found + [("unconfirmed", f"cannot confirm it is idle: no answer (rc {done.returncode}: {tail})")]
        apps = got.get("compute_apps")
        if apps is None or apps.rc != 0:
            found.append(("unconfirmed", "cannot confirm it is idle: GPU processes not listed "
                                         "(nvidia-smi --query-compute-apps failed)"))
        else:
            found += [("gpu", f"GPU process {line.strip()}") for line in apps.text.splitlines() if line.strip()]
        if len(sections) > 1:
            boxes = got.get("yeto_containers")
            if boxes is None or boxes.rc != 0:
                found.append(("unconfirmed", "cannot confirm it is idle: containers not listed (ps failed as the "
                                             "user and through sudo)"))
            else:
                found += [("container", f"container {line.strip()}") for line in boxes.text.splitlines()
                          if line.strip()]
        return node, found

    with ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(nodes)))) as pool:
        results = list(pool.map(one, nodes))
    return {node.name: found for node, found in results if found}


def idle_gate(busy: dict[str, list[tuple[str, str]]], *, force: bool, force_busy_gpu: bool) -> bool:
    """Print what keeps nodes from the probe; True when the probe may go on."""
    if not busy:
        return True
    allowed = {"--force": force, "--force-busy-gpu": force_busy_gpu}
    blocking = {n: [(k, w) for k, w in found if not (OVERRIDE[k] and allowed[OVERRIDE[k]])]
                for n, found in busy.items()}
    blocking = {n: f for n, f in blocking.items() if f}
    if blocking:
        _say(f"[yeto] cluster probe: {len(blocking)} node(s) not idle or not confirmed idle: refused (the probe "
             "runs on idle nodes; --nodes picks the idle ones)")
        for name, found in blocking.items():
            flags = sorted({OVERRIDE[k] or "nothing overrides a lease" for k, _ in found})
            _say(f"  {name}: {'; '.join(w for _, w in found[:6])}{'; ...' if len(found) > 6 else ''} "
                 f"({', '.join(flags)})")
        return False
    for name, found in busy.items():
        for kind, what in found:
            _say(f"[yeto] cluster probe: [WARN] {name}: {what}: probing anyway ({OVERRIDE[kind]})")
    return True


def estimate_s(n: int, *, nccl_pairs: str, ib_seconds: int, max_parallel: int) -> float:
    """A rough wall time of a probe of ``n`` nodes (not counting the image pull)."""
    from .launch.preflight.cluster import disjoint_pairs, round_robin

    width = max(2, max_parallel)

    def rounds_time(rounds: list, per_pair: float) -> float:
        return sum(math.ceil(len(r) / width) * per_pair for r in rounds)

    nccl_rounds = round_robin(n) if nccl_pairs == "all" else disjoint_pairs(n)
    waves = math.ceil(n / width)
    return (waves * (5 + 30 + 60 + 45)  # ssh + clock, host probes, container probes, TCP ports
            + rounds_time(round_robin(n), ib_seconds + 40)  # container start, ib_write_bw
            + rounds_time(nccl_rounds, 120))  # container start, torch + NCCL init, collectives


def probe_main(args: argparse.Namespace, *, ops: Any = None, log: Any = None, clock: Any = time.monotonic,
               leases: Any = None) -> int:
    from .launch.preflight.cluster import (
        ClusterPreflight, Steps, disjoint_pairs, requirements_from_env, round_robin, trainer_image)
    from .launch.ssh_trainer_island import RDMA_FABRICS, SshIslandError, SshOps

    started = clock()
    try:
        inventory = load_inventory(args.inventory)
        nodes = _select(inventory, args.nodes)
        if len(nodes) < 2:
            raise InventoryError("the probe needs two nodes or more")
        other = [n.name for n in nodes if n.device_family != "cuda"]
        if other:
            raise InventoryError(f"nodes {other} are not cuda nodes: the probe runs NCCL (pass --nodes)")
        if args.max_parallel < 2:
            raise InventoryError("--max-parallel must be 2 or more (both ends of a pair run together)")
        if args.deadline <= 0:
            raise InventoryError("--deadline must be positive (seconds)")
        no_rdma = [n.name for n in nodes if n.fabric not in RDMA_FABRICS]
        image = trainer_image(args.image, nodes)
        no_image = image is None and any(n.runtime == "docker" and not n.image for n in nodes)
        if no_image and not args.dry_run:
            raise InventoryError(NO_IMAGE)
        req = requirements_from_env()
    except (InventoryError, SshIslandError, ValueError) as exc:
        _say(f"[yeto] cluster probe: {exc}")
        return 2
    n = len(nodes)
    pairs = n * (n - 1) // 2
    nccl_rounds = len(round_robin(n)) if args.nccl_pairs == "all" else len(disjoint_pairs(n))
    est = estimate_s(n, nccl_pairs=args.nccl_pairs, ib_seconds=args.ib_seconds, max_parallel=args.max_parallel)
    _say(f"[yeto] cluster probe: {n} node(s) ({', '.join(x.name for x in nodes)}); ib_write_bw on {pairs} "
         f"pair(s) in {len(round_robin(n))} round(s); NCCL transport on {args.nccl_pairs} pairs in "
         f"{nccl_rounds} round(s); image {image or 'of the inventory'}; up to {args.max_parallel} at once; "
         f"about {math.ceil(est / 60)} min (not counting the image pull), deadline {args.deadline:g} s")
    if not fg.grouped(inventory):
        _say(f"[yeto] cluster probe: note: {fg.HINT}")
    if no_rdma:
        _say(f"[yeto] cluster probe: nodes {no_rdma} declare no RDMA fabric (ib / roce): they get no group")
    if args.dry_run:
        if no_image:
            _say(f"[yeto] cluster probe: {NO_IMAGE}")
        _say(f"[yeto] cluster probe: dry run: nothing run; the probe needs idle nodes (no GPU process, no yeto- "
             f"container); --write would set fabric_group in {args.inventory}")
        return 0
    timed = _DeadlineOps(ops or SshOps(), started + args.deadline, clock)
    if leases is None and lease_store_of(inventory) is not None:
        from .launch.node_lease import NodeLeases

        leases = NodeLeases(str(lease_store_of(inventory)), "yeto-cluster-probe")  # read only: holder()
    from .launch.node_inventory import load_profiles

    profiles = {pr.name: pr for pr in load_profiles(args.inventory)}  # the container engine of each node
    busy = busy_nodes([profiles.get(n.name, n) for n in nodes], timed, max_parallel=args.max_parallel,
                      leases=leases)
    if not idle_gate(busy, force=args.force, force_busy_gpu=args.force_busy_gpu):
        return EXIT_BUSY
    steps = Steps(pull=not args.no_pull, container=True, ports=True, ib=True, nccl=True, ib_pairs="all",
                  ib_seconds=args.ib_seconds, nccl_sizes=args.nccl_sizes, nccl_iters=args.nccl_iters,
                  nccl_all=False, nccl_pairs=args.nccl_pairs)
    pf = ClusterPreflight(nodes, image=image, ops=timed, req=req, steps=steps, out_dir=None, log=log or _say,
                          max_parallel=args.max_parallel)
    report = pf.run()
    for c in report.checks:  # busy or blocked ports: what the preflight warns about before a launch
        if c.name == "tcp_ports" and c.status != "PASS":
            _say(f"[yeto] cluster probe: [{c.status}] {c.scope} tcp_ports: {c.detail}")
    probed = {x.name for x in nodes}
    result = fg.group_nodes([x.name for x in nodes], report.checks, states=pf.states,
                            previous={x.name: x.fabric_group for x in nodes},
                            reserved={x.fabric_group for x in inventory if x.name not in probed and x.fabric_group},
                            nccl_pairs=args.nccl_pairs)
    for line in result.warnings:
        _say(f"[yeto] cluster probe: {line}")
    if args.json:
        print(json.dumps(result.record(), indent=2, sort_keys=True))
    else:
        for line in result.lines():
            print(line)
    if timed.hit:
        _say(f"[yeto] cluster probe: the --deadline of {args.deadline:g} s cut the probe short: the result is "
             "incomplete, nothing written")
        return 1
    if args.write:
        try:
            changed = fg.write_groups(args.inventory, result.assignment())
        except (InventoryError, OSError) as exc:
            _say(f"[yeto] cluster probe: {exc}")
            return 2
        _say(f"[yeto] cluster probe: {'wrote' if changed else 'unchanged:'} fabric_group of {n} node(s) in "
             f"{args.inventory}")
    else:
        _say(f"[yeto] cluster probe: nothing written (--write sets fabric_group in {args.inventory})")
    return 0 if result.clean else 1
