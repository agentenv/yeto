"""Resident ``ssh:`` trainer island: one the fleet keeps up for the whole run.

Without the trainer scheduler (``--rl-trainer-scheduler off``, e.g. the K3.1
trainer-only launch ``--rl-sample-role trainer --rl-single-island-no-sync``), an
``ssh:`` island of ``--gpu`` whose nodes come from ``--rl-node-inventory`` is
neither a sky cluster nor a held-back scheduler island: :class:`SshIslandOps` is the
fleet controller's ops for it (``job_status`` / ``cluster_up`` / ``relaunch`` /
``down`` / ``rl_strict_failure``, as ``modal_runner.ModalIslandOps`` for a Modal
island), over the same :class:`~.ssh_trainer_island.SshTrainerIslands` the
scheduler uses (``scheduled=False``: no scheduler control environment).

Lifecycle of one island:

* **start**: incarnation 0 (preflight: CPU arch, cards, scratch, ``docker pull``),
  recorded in the island's journal (``<run dir>/ssh-islands/<island>-jobs.json``)
  before any node starts, and in the run's resource ledger as every SSH trainer
  incarnation is (``ssh_trainer_ledger``: the cost guard counts it, ``yeto cleanup
  --run`` / ``yeto down`` stop it on its recorded nodes without the launcher);
* **reattach**: a launcher restarted on the same run reads the journal and asks
  every node about the newest incarnation. Running on every node, or ended with
  rank 0's exit code 0: the launcher follows it (the log replays from its start,
  so the echo tape is rebuilt). Partly running, or ended otherwise: it is released
  (verified) and the next incarnation starts. A node that does not answer refuses
  the start: a second copy must never run on the same cards. A second launcher of
  the same island is refused by a lock file;
* **watch** (each fleet poll): a node's stream ending is a node event, read from
  the node's own state (``status_script``): rank 0 exited 0 is the island's
  success; any rank exiting non-zero, a process gone without an exit code (killed,
  OOM, reboot), a node unreachable for ``unreachable_polls`` polls, or a worker
  ending cleanly while rank 0 still runs for ``worker_exit_grace_s`` fails the
  island at once and releases every node (one rank's failure is the island's). A
  stream that dropped while its node still runs is followed again. Every
  ``health_every_s`` the nodes are asked even while their streams flow; on a device
  family with a health checker (cuda, :mod:`yeto.hw.node_health`) the same SSH call
  also checks the node's GPUs, NVLink fabric and the ports of its ``NCCL_IB_HCA``
  (a multi-node island). The check is bounded on the node (a hung tool is a read
  failure, never an unreachable node). Two bad checks in a row of one node are a
  node lost (:meth:`SshIslandOps._node_health_lost`, called once when it turns lost,
  outside the ops' lock); each check is in the island's record
  (:meth:`SshIslandOps.island_health`) and its events go to the fleet's
  (:meth:`SshIslandOps.take_health_events`). An inventory ``node_health: off`` (the
  file's or a node's) turns the check off;
* **relaunch** (the fleet's recovery): release the failed incarnation (verified;
  unverified: no new start this poll) and start the next one; the trainer
  continues from the store (``--rl-init-from-release`` / its resume store);
* **release** (the run ends, ``yeto stop-run``, abandon, teardown): TERM to the
  container's processes (Modal's stop signal), ``docker rm -f`` after the grace,
  the process group TERM -> KILL; each node must then show neither process nor
  container, else the teardown is unverified (exit 5, the head stays up). Once it is
  verified the island's node leases (inventory ``lease_store``, :mod:`.node_lease`)
  are given up; a launcher that ends without a release (``--keep``) leaves them to
  run out, and a restarted launcher of the run takes them back.

* **spare nodes** (an inventory with ``roles: [trainer-spare]`` nodes, :mod:`.island_spares`):
  a lost node (its stream ended and it does not answer over SSH ``unreachable_polls``
  times in a row, or its health turned bad twice) is not waited for: the island is stopped
  on its other nodes, a spare of the same fabric group takes the lost node's place and the
  next incarnation resumes from the newest checkpoint (``island_node_lost`` /
  ``island_node_replaced``); no spare: ``waiting_spare`` with a backoff. Without spare
  nodes nothing of the above changes.

Each watch also renews the island's node leases that are due; a node whose lease
another run took fails the island at once (that run may start there).

The echo tape comes from node 0's stream (``on_line``); when it lacks the
finalized record, :meth:`SshIslandOps.recover_tape` re-reads node 0's log.
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config_errors import CONFIG_ERROR_EXIT, CONFIG_ERROR_MARKER, ISLAND_STOP_EXIT, ISLAND_STOP_MARKER
from ..hw import node_health
from . import island_lifecycle
from .ssh_trainer_island import (
    KILL_GRACE_S,
    RDMA_FABRICS,
    NodeState,
    SshIslandError,
    SshOps,
    SshTrainerIslands,
    SshTrainerSpec,
    container_name,
    docker_job,
    node_env,
    remote_paths,
)
from .trainer_island import read_jobs

if TYPE_CHECKING:
    from .island_spares import IslandSpares

RESIDENT_DIR = "ssh-islands"  # under the run dir
UNREACHABLE_POLLS = 3
MAX_REFOLLOWS = 5
WORKER_EXIT_GRACE_S = 120.0
HEALTH_EVERY_S = 60.0
LOG_TAIL_LINES = 400
# the unified table's strict-failure markers, and a configuration error's (never relaunched)
STRICT_MARKERS = (*island_lifecycle.STRICT_MARKERS, CONFIG_ERROR_MARKER, ISLAND_STOP_MARKER)


class _Status:
    """A job status for the fleet controller (sky's ``JobStatus`` duck type)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def is_terminal(self) -> bool:
        return self.text in ("SUCCEEDED", "FAILED")

    def __str__(self) -> str:
        return self.text


def is_resident(args: Any, spec: Any, island: int, num_islands: int) -> bool:
    """Whether ``--gpu`` island ``island`` (``spec``) is a resident ``ssh:`` trainer
    island: cloud ``ssh``, an RL run with ``--rl-node-inventory``, not held back by
    the trainer scheduler, and not a sampler island (those stay on the SSH node pool)."""
    if getattr(spec, "cloud", None) != "ssh" or getattr(args, "training_mode", "sft") != "rl":
        return False
    if not getattr(args, "rl_node_inventory_nodes", None):
        return False
    from . import trainer_loop

    if trainer_loop.held_island(args) == island:
        return False
    from yeto.rl.engine import sample_roles

    role = getattr(sample_roles.island_args(args, island, num_islands), "rl_sample_role", None)
    return role != sample_roles.SAMPLER


def check(args: Any, specs: Any) -> None:
    """Refuse, before any cloud or SSH work, a resident ``ssh:`` island the launch could
    not start: the inventory's trainer nodes must match it (count, card, cards), its
    code must reach the nodes and its image must be the run's (SystemExit). With an
    inventory ``lease_store`` the nodes checked are the ones the island's leases pick."""
    from .sky_ops import learner_cluster_names
    from .ssh_trainer_island import code_source, inventory_trainer_nodes, run_image, trainer_leases

    leases = trainer_leases(args)  # None without lease_store (and in a dry run): the check as before
    names = learner_cluster_names(args.cluster_prefix, list(specs)) if leases is not None else None
    for m, spec in enumerate(specs):
        if not is_resident(args, spec, m, len(specs)):
            continue
        try:
            nodes = inventory_trainer_nodes(args, spec, leases=leases, name=names[m] if names else None)
            run_image(args, nodes)
        except SshIslandError as error:
            raise SystemExit(f"--gpu island {m} ({spec}): {error}") from None
        from .island_spares import SpareError, check_spares, spare_nodes  # noqa: PLC0415

        try:
            check_spares(nodes, spare_nodes(getattr(args, "rl_node_inventory_nodes", None) or ()))
        except SpareError as error:
            raise SystemExit(f"--gpu island {m} ({spec}): {error}") from None
        if code_source(args) is None:
            raise SystemExit(f"--gpu island {m} ({spec}): the run's code cannot reach its nodes: use an s3:// "
                             "--rl-checkpoint-store with --rl-code-pack auto or git-patch (every node downloads "
                             "the run code pack), --rl-code-pack release, or --rl-ssh-trainer-workdir")


@dataclass
class _Island:
    name: str
    spec: SshTrainerSpec
    islands: SshTrainerIslands
    lock: Any = None
    job: Any = None
    incarnation: int | None = None
    verdict: str | None = None  # None (running), SUCCEEDED or FAILED
    reason: str = ""
    ended: dict[int, NodeState] = field(default_factory=dict)
    unreachable: dict[int, int] = field(default_factory=dict)
    refollows: dict[int, int] = field(default_factory=dict)
    workers_done_at: float | None = None
    health_at: float = 0.0
    lines: deque = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))
    released: set[int] = field(default_factory=set)  # incarnations confirmed stopped
    # node health (yeto.hw.node_health): one watch per rank (empty: the family has no checker)
    health: dict[int, Any] = field(default_factory=dict)
    health_lost: dict[int, list[str]] = field(default_factory=dict)  # rank -> reasons (this incarnation)
    health_events: list[dict[str, Any]] = field(default_factory=list)  # for the fleet's event log
    health_pending: list[tuple[int, list[str], Any]] = field(default_factory=list)  # lost nodes, hook not run yet
    spares: Any = None  # island_spares.IslandSpares: an inventory with spare nodes (None: none)
    rank_gen: dict[int, int] = field(default_factory=dict)  # rank -> how often a spare took its place
    # id(reading) -> (node, rank_gen) when a health reading turned lost: the hook ignores a stale one
    health_stamps: dict[int, tuple[str, int]] = field(default_factory=dict)
    owed_retries: dict[str, Any] = field(default_factory=dict)  # node -> its owed-stop retry thread
    on_line: Any = None  # node 0's line sink of every incarnation


_RECHECK = object()  # _replace: a candidate is not health checked yet (check it outside the lock)


class SshIslandOps:
    """The fleet controller's ops for resident ``ssh:`` islands (module doc).

    ``sink_for(name)``: where node 0's lines of island ``name`` go (the launcher
    prints them and feeds the echo tape collector); ``ops``: the SSH calls (tests
    pass a fake). The resource ledger rows are ``SshTrainerIslands``' own."""

    def __init__(self, *, run_id: str, journal_dir: Path, events_dir: Path | None = None,
                 ops: SshOps | None = None,
                 sink_for: Callable[[str], Callable[[str], None]] | None = None,
                 kill_grace_s: int = KILL_GRACE_S, clock: Callable[[], float] = time.monotonic,
                 health_every_s: float = HEALTH_EVERY_S, worker_exit_grace_s: float = WORKER_EXIT_GRACE_S,
                 unreachable_polls: int = UNREACHABLE_POLLS, max_refollows: int = MAX_REFOLLOWS,
                 log: Callable[[str], None] | None = None, wall: Callable[[], float] = time.time) -> None:
        self.run_id = str(run_id)
        self.journal_dir = Path(journal_dir)
        self.events_dir = None if events_dir is None else Path(events_dir)
        self.ops = ops or SshOps()
        self.sink_for = sink_for
        self.kill_grace_s = int(kill_grace_s)
        self.clock = clock
        self.health_every_s = float(health_every_s)
        self.worker_exit_grace_s = float(worker_exit_grace_s)
        self.unreachable_polls = int(unreachable_polls)
        self.max_refollows = int(max_refollows)
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.wall = wall  # wall clock of the replacement limit (a launcher restart keeps the count)
        self.islands: dict[str, _Island] = {}
        # reentrant: the node health watch may call _node_health_lost from inside job_status's
        # watch or from outside it; either way the hook takes this lock itself
        self._lock = threading.RLock()
        self.reattached: set[str] = set()  # islands whose start followed a running incarnation

    # -- registry -------------------------------------------------------------------
    def owns(self, name: str) -> bool:
        return name in self.islands

    def jobs_path(self, name: str) -> Path:
        return self.journal_dir / f"{name}-jobs.json"

    def add(self, name: str, spec: SshTrainerSpec, *, spares: IslandSpares | None = None) -> None:
        """Register island ``name`` (no SSH yet). ``spares``: the inventory's spare nodes
        (:mod:`.island_spares`); the island's journal of them names the nodes its newest
        incarnation runs on (a restarted launcher follows it there)."""
        if name in self.islands:
            raise SshIslandError(f"ssh: island {name} is registered twice")
        sink = self.sink_for(name) if self.sink_for is not None else None
        if spares is not None:
            from .island_spares import JOURNAL_SUFFIX  # noqa: PLC0415

            spares.load(self.journal_dir / f"{name}{JOURNAL_SUFFIX}")
            spec = self._spec_on(spec, spares, spares.nodes_of(None))
            spares.job_env = dict(getattr(spec, "env", None) or {})  # the spares' checks read its NCCL_IB_HCA
            if spares.leases is None:  # the island's own node leases (inventory lease_store)
                spares.leases = getattr(spec, "leases", None)
            if spares.records.shared is None and spares.leases is not None:
                spares.records.shared = getattr(spares.leases, "objects", None)  # the shared quarantine
        rec = _Island(name, spec, None, spares=spares)  # type: ignore[arg-type]

        def on_line(line: str) -> None:
            rec.lines.append(line)
            if sink is not None:
                sink(line)

        rec.on_line = on_line
        rec.islands = self._islands_for(rec, spec)
        rec.health = self._health_watches(spec)
        self.islands[name] = rec

    def _islands_for(self, rec: _Island, spec: SshTrainerSpec) -> SshTrainerIslands:
        # node_wait_s=0: a start finds an unreachable node at once and names it; the
        # fleet controller retries the relaunch every poll (a wait here would hold it)
        return SshTrainerIslands(spec, name=rec.name, jobs_path=self.jobs_path(rec.name), ops=self.ops,
                                 events_dir=self.events_dir, kill_grace_s=self.kill_grace_s,
                                 scheduled=False, tape=False, on_line=rec.on_line, node_wait_s=0.0)

    @staticmethod
    def _spec_on(spec: SshTrainerSpec, spares: Any, names: list[str] | None) -> SshTrainerSpec:
        """``spec`` on the nodes ``names`` of the spares' pool (None: unchanged)."""
        if not names or names == [n.name for n in spec.nodes]:
            return spec
        pool = spares.by_name()
        missing = [n for n in names if n not in pool]
        if missing or len(names) != len(spec.nodes):
            raise SshIslandError(f"the island's spare journal names nodes {names}; the inventory has no "
                                 f"{missing or 'such set of'} node(s): restore those inventory entries")
        import dataclasses  # noqa: PLC0415

        return dataclasses.replace(spec, nodes=tuple(pool[n] for n in names))

    def _health_watches(self, spec: SshTrainerSpec) -> dict[int, Any]:
        """One :class:`~yeto.hw.node_health.HealthWatch` per rank (a family without a checker,
        or ``health_every_s`` 0: none). A multi-node island's job ports are its ``NCCL_IB_HCA``."""
        if self.health_every_s <= 0 or not node_health.checker_for(spec.device_family).enabled:
            return {}
        out = {}
        for rank, node in enumerate(spec.nodes):
            if getattr(node, "node_health", None) == "off":
                continue
            env = {**dict(node.env), **dict(spec.env)}  # the job's own order (node_env): the island's wins
            out[rank] = node_health.HealthWatch(node.device_family, hcas=node_health.hcas_of(env.get("NCCL_IB_HCA")),
                                                multi=len(spec.nodes) > 1, expect_gpus=node.cards_per_node,
                                                min_ib_gbps=getattr(node, "min_ib_gbps", None),
                                                clock=time.time, every_s=self.health_every_s,
                                                fabric=node.fabric if node.fabric in RDMA_FABRICS else None)
        return out

    def threads(self, name: str) -> list[threading.Thread]:
        """The stream threads of the island's current incarnation (the launcher's tape drain)."""
        rec = self.islands[name]
        return list(rec.job.threads) if rec.job is not None else []

    # -- start / reattach -------------------------------------------------------------
    def start(self, name: str) -> int:
        """Start (or reattach to) island ``name``; returns the job id (its incarnation)."""
        rec = self.islands[name]
        self._take_lock(rec)
        lost = self._quarantined_ranks(rec)  # spare nodes: a node lost earlier (empty without spares)
        self._claim_nodes(rec, skip=lost)
        self._keep_quarantined_leases(rec)
        self._release_stale_leases(rec)
        jobs = read_jobs(self.jobs_path(name))
        if not jobs and not lost:
            return self._launch(rec, 0)
        if lost:
            return self._start_without(rec, lost, max(int(k) for k in jobs) if jobs else None)
        last = max(int(k) for k in jobs)
        states = [rec.islands.node_state(last, rank) for rank in range(len(rec.spec.nodes))]
        silent = [rec.spec.nodes[r].name for r, st in enumerate(states) if st is None]
        if silent:
            raise SshIslandError(
                f"ssh: island {name}: incarnation {last} is recorded in {self.jobs_path(name)}, and nodes "
                f"{silent} do not answer; refusing to start another copy on the same cards (check the nodes, "
                f"or release it with `yeto cleanup --run {self.run_id}`)")
        running = [r for r, st in enumerate(states) if st.running]
        if len(running) == len(states) or (states[0].state == "exited" and states[0].rc == 0):
            self.log(f"[launcher] {name}: incarnation {last} "
                     f"{'still runs on every node' if running else 'ended with rank 0 exit 0'}; "
                     "reattaching (no second start)")
            return self._attach(rec, last)
        detail = ", ".join(f"{rec.spec.nodes[r].name}={st.state}" + (f"(rc {st.rc})" if st.rc is not None else "")
                           for r, st in enumerate(states))
        self.log(f"[launcher] {name}: incarnation {last} is not whole ({detail}); releasing it before "
                 f"incarnation {last + 1}")
        if not self._release(rec, last):
            raise SshIslandError(f"ssh: island {name}: incarnation {last} could not be confirmed stopped; "
                                 "refusing to start another copy on the same cards")
        return self._launch(rec, last + 1)

    def _take_lock(self, rec: _Island) -> None:
        if rec.lock is not None:
            return
        path = self.journal_dir / f"{rec.name}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            holder = fh.read().strip()
            fh.close()
            raise SshIslandError(f"ssh: island {rec.name} is supervised by another launcher ({holder or path}); "
                                 "refusing a second one") from None
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"run": self.run_id, "pid": os.getpid(), "since": time.time()}))
        fh.flush()
        rec.lock = fh

    def _reset(self, rec: _Island, job: Any, incarnation: int) -> int:
        rec.job, rec.incarnation = job, int(incarnation)
        rec.verdict, rec.reason = None, ""
        rec.ended, rec.unreachable, rec.refollows = {}, {}, {}
        rec.health_lost, rec.health_pending = {}, []
        for watch in rec.health.values():  # a new job on the same nodes: a fresh streak, the same baselines
            watch.reset()
        rec.workers_done_at = None
        rec.health_at = self.clock()
        return int(incarnation)

    def _launch(self, rec: _Island, incarnation: int) -> int:
        if rec.spares is not None:  # the nodes of the incarnation, before any start
            rec.spares.record_nodes(incarnation, [n.name for n in rec.spec.nodes])
        job = rec.islands.launch(incarnation, None)  # journal and ledger row before any node starts
        return self._reset(rec, job, incarnation)

    def _claim_nodes(self, rec: _Island, *, skip: Any = ()) -> None:
        """An island with spare nodes holds its nodes in the run store (``island_spares``
        claims): no other island of the run picks them as its spare. ``skip``: ranks not
        claimed (quarantined nodes)."""
        if rec.spares is None:
            return
        from .island_spares import SpareError  # noqa: PLC0415

        for rank, node in enumerate(rec.spec.nodes):
            if rank in skip:
                continue
            try:
                held = rec.spares.records.claim(node.name, rec.name)
            except SpareError as error:
                raise SshIslandError(f"ssh: island {rec.name}: {error}") from None
            if held is not None:
                raise SshIslandError(f"ssh: island {rec.name}: node {node.name} is held by island {held} of this "
                                     "run (fleet/nodes/<name>/claim.json); two islands cannot share a node")

    def _keep_quarantined_leases(self, rec: _Island) -> None:
        """A start of an island with spare nodes and node leases: every quarantined node of
        the inventory whose lease this run holds (a restarted launcher took the island's
        recorded nodes back) keeps it out of the renewals until release-node
        (:meth:`_mark_lost`). The island's own start takes its current nodes' leases
        (``SshTrainerIslands._hold_leases``)."""
        sp = rec.spares
        leases = None if sp is None else sp.leases
        if leases is None or not hasattr(leases, "keep_all"):
            return
        mine = set(leases.held())
        quarantined = [n for n in sp.pool if n.name in mine and sp.records.own_quarantine(n.name) is not None]
        if quarantined:
            leases.keep_all(nodes=quarantined)

    def _hold_current_leases(self, rec: _Island) -> None:
        """A reattach with node leases: the island's current nodes (a spare among them) are
        this run's (``_hold_leases``: another run's lease refuses it)."""
        if rec.spec.leases is not None:
            rec.islands._hold_leases()  # noqa: SLF001 - the island's own lease step

    def _release_stale_leases(self, rec: _Island) -> None:
        """Every start (a first launch, a start without a quarantined node, a reattach): a
        node the leases picked for the island but that it no longer runs on (a lost node,
        released, named by an older lease record) is given back, unless it is quarantined
        (its lease stays kept until release-node) or another island of the run holds it
        (its claim, or its own pick); the record then names the island's nodes."""
        leases = rec.spec.leases
        if leases is None:
            return
        sp = rec.spares
        picks = getattr(leases, "picks", None)
        if sp is None or not isinstance(picks, dict):
            return
        current = [n.name for n in rec.spec.nodes]
        others = {n for k, v in picks.items() if k != rec.name for n in v}
        mine = set(leases.held())
        by_name = {n.name: n for n in sp.pool}
        for name in sorted(set(picks.get(rec.name, ())) - set(current) - others):
            held = sp.records.holder(name)
            if name not in mine or name not in by_name or sp.records.quarantined(name) is not None \
                    or held not in (None, rec.name):
                continue
            leases.release(by_name[name])
            self.log(f"[launcher] {rec.name}: node {name} is no longer one of its nodes (a spare took its place); "
                     "its lease is given back")
        self._record_nodes(rec)

    def _record_nodes(self, rec: _Island) -> None:
        """The island's current nodes into the run's lease record and the leases' picks (a
        restarted launcher leases these, not a node a spare replaced)."""
        leases = rec.spec.leases
        if leases is None:
            return
        names = [n.name for n in rec.spec.nodes]
        picks = getattr(leases, "picks", None)
        if isinstance(picks, dict):
            old = set(picks.get(rec.name, ()))
            others = {n for k, v in picks.items() if k != rec.name for n in v}
            picks[rec.name] = names
            assigned = getattr(leases, "assigned", None)
            if isinstance(assigned, set):  # this island's old nodes only; another island's stay assigned
                assigned -= old - set(names) - others
                assigned |= set(names)
        from .ssh_trainer_island import record_island_nodes  # noqa: PLC0415

        try:
            record_island_nodes(rec.spec.run_id, rec.name, rec.spec.nodes)
        except OSError as error:
            self.log(f"[launcher] WARN: {rec.name}: its nodes {names} are not in the lease record ({error})")

    def _quarantined_ranks(self, rec: _Island) -> dict[int, str]:
        """Ranks whose node is quarantined (lost earlier: by this run, or by another run in
        the inventory's ``lease_store``): rank -> why. Empty without spare nodes."""
        if rec.spares is None:
            return {}
        out = {}
        for rank, node in enumerate(rec.spec.nodes):
            doc = rec.spares.records.quarantined(node.name)
            if doc is not None and doc.get("unreadable") and doc.get("read_error"):  # the store did not answer
                self.log(f"[launcher] WARN: {rec.name}: the quarantine record of {node.name} could not be read "
                         f"({doc.get('reason')}); it is not taken as lost")
                continue
            if doc is not None:
                out[rank] = f"rank {rank} ({node.name}) is quarantined: {str(doc.get('reason') or '')[:300]}"
        return out

    def _start_without(self, rec: _Island, lost: dict[int, str], last: int | None) -> int:
        """A start (a restarted launcher, or a node another run quarantined) while nodes of
        the island are quarantined: they are lost; the newest incarnation is released on
        the other nodes and the next one starts with spares in their places. No spare yet:
        the island reads as failed (its waiting continues in the fleet's recovery)."""
        sp = rec.spares
        names = [rec.spec.nodes[r].name for r in sorted(lost)]
        self.log(f"[launcher] {rec.name}: node(s) {names} are quarantined; they are not started on "
                 f"({'; '.join(lost[r] for r in sorted(lost))})")
        if last is not None:
            live = [r for r in range(len(rec.spec.nodes)) if r not in lost]
            silent = [rec.spec.nodes[r].name for r in live if rec.islands.node_state(last, r) is None]
            if silent:
                raise SshIslandError(
                    f"ssh: island {rec.name}: incarnation {last} is recorded in {self.jobs_path(rec.name)}, and "
                    f"nodes {silent} do not answer; refusing to start another copy on the same cards")
            if not self._release(rec, last, skip=frozenset(names)):
                raise SshIslandError(f"ssh: island {rec.name}: incarnation {last} could not be confirmed stopped; "
                                     "refusing to start another copy on the same cards")
        for rank, why in lost.items():
            sp.lost.setdefault(rank, why)
        rec.incarnation = last
        job_id = self._replace_checked(rec)
        if job_id is not None:
            return job_id
        rec.verdict, rec.reason = "FAILED", (f"node(s) {names} quarantined; waiting for a spare node "
                                             f"({sp.wait_reason})")
        return -1 if last is None else last

    def _attach(self, rec: _Island, incarnation: int) -> int:
        self._hold_current_leases(rec)
        job = rec.islands.attach(incarnation)  # its ledger row is open since its start
        self.reattached.add(rec.name)
        return self._reset(rec, job, incarnation)

    # -- the fleet controller's ops -----------------------------------------------------
    def job_status(self, name: str, job_id: Any) -> _Status:
        rec = self.islands[name]
        with self._lock:
            if rec.incarnation is None or int(job_id) != rec.incarnation:
                return _Status("FAILED")  # an incarnation this launcher no longer runs
            self._watch(rec)
            pending, rec.health_pending = rec.health_pending, []
        for rank, reasons, reading in pending:  # outside the lock: the hook may stop and replace the island
            self._node_health_lost(rec, rank, reasons, reading)
        return _Status(rec.verdict or "RUNNING")

    def cluster_up(self, name: str) -> bool:
        rec = self.islands[name]
        return rec.job is not None and rec.verdict != "FAILED"

    def relaunch(self, task: Any, name: str) -> int | None:
        """Release the failed incarnation (verified) and start the next one; None when
        the release is not confirmed or the start failed (the controller retries). With
        spare nodes a lost node is replaced (:meth:`_relaunch_with_spares`)."""
        rec = self.islands[name]
        if rec.spares is not None:
            return self._relaunch_with_spares(rec)  # takes the lock itself (not over the health checks)
        with self._lock:
            if rec.incarnation is not None and not self._release(rec, rec.incarnation):
                self.log(f"[launcher] {name}: incarnation {rec.incarnation} not confirmed stopped; "
                         "no new start this poll")
                return None
            nxt = 0 if rec.incarnation is None else rec.incarnation + 1
            try:
                return self._launch(rec, nxt)
            except Exception as error:  # noqa: BLE001 - the controller retries next poll
                self.log(f"[launcher] {name}: start of incarnation {nxt} failed: {error}")
                self._release(rec, nxt)
                return None

    def down(self, name: str) -> bool:
        """Release the island's current incarnation, and any earlier one not confirmed stopped
        on a node; True once every node confirmed them (else its leases, claims and ledger
        rows are kept)."""
        rec = self.islands.get(name)
        if rec is None:
            return True
        if rec.incarnation is None:  # never started: nothing runs on its nodes
            rec.islands.release_leases()
            return True
        self._join_owed_retries(rec)
        with self._lock:
            self._flush_pending(rec)
            lost = self._lost_names(rec)
            # an earlier incarnation not confirmed stopped on a node (a failed start on spares):
            # stopped first; not confirmed, the island stays held (leases, claims, its ledger row)
            earlier_ok = True
            for k in sorted(set(rec.islands.unreleased) - {int(rec.incarnation)}):
                earlier_ok &= rec.islands.release_incarnation(k, skip=lost, owed=frozenset())
            # a lost node is not waited for: it owes its stop (its own ledger row, release-node)
            ok = self._release(rec, rec.incarnation, skip=lost) and earlier_ok
            if ok and rec.spares is not None:  # its nodes are free for another island of the run
                for node in rec.spec.nodes:
                    rec.spares.records.unclaim(node.name, rec.name)
        if ok:
            rec.islands.release_leases()
        if ok and rec.lock is not None:
            rec.lock.close()
            rec.lock = None
        return ok

    def close(self) -> None:
        """The launcher stops supervising (its end, ``--keep`` included): end its
        streams and give up the locks; the nodes are not touched."""
        for rec in self.islands.values():
            self._join_owed_retries(rec)
            if rec.job is not None:
                for proc in rec.job.procs:
                    if proc.poll() is None:
                        proc.terminate()
            rec.islands.keep_leases()  # not released (kept, or not confirmed): held until cleanup
            if rec.lock is not None:
                rec.lock.close()
                rec.lock = None

    def island_decision(self, name: str, job_id: Any) -> island_lifecycle.Decision | None:
        """Why the island's incarnation ended, by the unified decision table
        (``yeto.launch.island_lifecycle.classify``) from what this source observed: a rank's
        exit code (``CONFIG_ERROR_EXIT`` is a configuration error), the strict / config
        marker lines of node 0's log, a node that stopped answering (lost) or a trainer
        process gone without an exit code (killed, OOM, reboot: infrastructure). None while
        the incarnation runs, or for an incarnation this launcher no longer runs."""
        rec = self.islands.get(name)
        if rec is None or rec.verdict is None or rec.incarnation is None or str(job_id) != str(rec.incarnation):
            return None
        codes = [st.rc for st in rec.ended.values() if st.state == "exited" and st.rc not in (None, 0)]
        exit_code = next((c for c in (CONFIG_ERROR_EXIT, ISLAND_STOP_EXIT) if c in codes), codes[0] if codes else None)
        return island_lifecycle.classify(
            exit_code=0 if rec.verdict == "SUCCEEDED" else exit_code,
            log_text="\n".join(str(line) for line in rec.lines), succeeded=rec.verdict == "SUCCEEDED",
            lost="does not answer over SSH" in rec.reason or bool(rec.health_lost), detail=rec.reason)

    def replaces_machine(self, name: str) -> bool:
        """A relaunch runs on the inventory's same nodes (no other machine to order)."""
        return False

    def rl_strict_failure(self, name: str, job_id: Any) -> str | None:
        for line in reversed(list(self.islands[name].lines)):
            text = str(line).strip()
            if any(m in text for m in STRICT_MARKERS):
                return text
        return None

    # -- the watch --------------------------------------------------------------------
    def _watch(self, rec: _Island) -> None:
        if rec.verdict is not None:
            return
        lost = rec.islands.lease_lost()
        if lost is not None:
            self._fail(rec, lost)
            return
        procs = rec.job.procs
        for rank, proc in enumerate(procs):
            if rank in rec.ended or proc.poll() is None:
                continue
            self._node_event(rec, rank, rec.islands.node_state(rec.incarnation, rank), stream_ended=True)
            if rec.verdict is not None:
                return
        now = self.clock()
        if self.health_every_s > 0 and now - rec.health_at >= self.health_every_s:
            rec.health_at = now
            for rank in range(len(procs)):
                if rank in rec.ended:
                    continue
                watch = rec.health.get(rank)
                if watch is None:
                    self._node_event(rec, rank, rec.islands.node_state(rec.incarnation, rank), stream_ended=False)
                else:  # the node's health check rides on the same SSH call
                    state, text = rec.islands.node_check(rec.incarnation, rank, watch.script())
                    self._node_event(rec, rank, state, stream_ended=False)
                    if rec.verdict is None and state is not None:
                        self._health_reading(rec, rank, watch.observe(text))
                if rec.verdict is not None:
                    return
        workers = [r for r in rec.ended if r > 0]
        if workers and 0 not in rec.ended:
            rec.workers_done_at = rec.workers_done_at if rec.workers_done_at is not None else now
            if now - rec.workers_done_at >= self.worker_exit_grace_s:
                self._fail(rec, f"rank(s) {workers} ended (exit 0) while rank 0 still runs, for "
                                f"{now - rec.workers_done_at:.0f}s")

    def _node_event(self, rec: _Island, rank: int, state: NodeState | None, *, stream_ended: bool) -> None:
        node = rec.spec.nodes[rank].name
        if state is None:
            rec.unreachable[rank] = rec.unreachable.get(rank, 0) + 1
            if rec.unreachable[rank] >= self.unreachable_polls:
                self._fail(rec, f"rank {rank} ({node}) does not answer over SSH "
                                f"({rec.unreachable[rank]} checks in a row)", silent_rank=rank)
            return
        rec.unreachable[rank] = 0
        if state.running:
            if not stream_ended:
                return
            rec.refollows[rank] = rec.refollows.get(rank, 0) + 1
            if rec.refollows[rank] > self.max_refollows:
                self._fail(rec, f"rank {rank} ({node}): its log stream dropped {rec.refollows[rank]} times")
                return
            self.log(f"[launcher] {rec.name}: rank {rank} ({node}) still runs, its stream dropped; following again")
            rec.islands.follow(rec.job, rank)
            return
        rec.ended[rank] = state
        if state.state != "exited":
            self._fail(rec, f"rank {rank} ({node}): trainer process gone without an exit code ({state.state}: "
                            "killed, OOM-killed or the node rebooted)")
        elif rank == 0 and state.rc == 0:
            rec.verdict, rec.reason = "SUCCEEDED", "rank 0 exited 0"
            self.log(f"[launcher] {rec.name}: rank 0 exited 0; the island finished")
        elif state.rc != 0:
            self._fail(rec, f"rank {rank} ({node}) exited with code {state.rc}")

    # -- node health ------------------------------------------------------------------
    def _health_reading(self, rec: _Island, rank: int, reading: Any) -> None:
        """One node's health check: its event (``node_health`` when it turns lost,
        ``node_health_read_failed`` when reads start failing, ``node_health_fm_restarted`` at
        each check of a fabric manager restart streak), and a lost node's handling."""
        node = rec.spec.nodes[rank].name
        base = {"node": node, "rank": rank, "incarnation": rec.incarnation}
        if reading.degraded:  # once per node: what its check cannot judge
            rec.health_events.append({"event": node_health.EVENT_DEGRADED, **base, "reasons": list(reading.degraded)})
            self.log(f"[launcher] {rec.name}: {node_health.EVENT_DEGRADED}: rank {rank} ({node}): "
                     f"{'; '.join(reading.degraded)}")
        if reading.fm_restarted:  # every check of a fabric manager restart streak
            fields = reading.fm_restarted_fields()
            rec.health_events.append({"event": node_health.EVENT_FM_RESTARTED, **base, **fields})
            self.log(f"[launcher] {rec.name}: {node_health.EVENT_FM_RESTARTED}: rank {rank} ({node}) "
                     f"streak {fields.get('streak')}: {'; '.join(fields.get('reasons') or [])}")
        if reading.event is not None:
            rec.health_events.append({"event": reading.event, **base, **node_health.event_fields(reading)})
            self.log(f"[launcher] {rec.name}: {reading.event}: rank {rank} ({node}) {reading.verdict}: "
                     f"{'; '.join(reading.result.reasons) or '-'}")
        if reading.event == node_health.EVENT_LOST:  # it just turned lost: the hook runs once, after the lock
            rec.health_stamps[id(reading)] = (node, rec.rank_gen.get(rank, 0))
            rec.health_pending.append((rank, reading.result.job_reasons(len(rec.spec.nodes) > 1), reading))

    def island_health(self, name: str) -> dict[str, Any] | None:
        """The newest health check of every node of island ``name`` (its record's ``health``);
        None when its family has no checker or nothing was checked yet."""
        rec = self.islands.get(name)
        if rec is None or not rec.health:
            return None
        nodes = {rec.spec.nodes[rank].name: watch.record() for rank, watch in sorted(rec.health.items())
                 if watch.record() is not None}
        if not nodes:
            return None
        worst = max((doc["verdict"] for doc in nodes.values()), key=node_health.VERDICTS.index)
        return {"verdict": worst, "nodes": nodes}

    def take_health_events(self, name: str) -> list[dict[str, Any]]:
        """The node health events of island ``name`` since the last call."""
        rec = self.islands.get(name)
        if rec is None:
            return []
        with self._lock:
            out, rec.health_events = rec.health_events, []
        return out

    def _fail(self, rec: _Island, reason: str, *, silent_rank: int | None = None) -> None:
        rec.verdict, rec.reason = "FAILED", reason
        self.log(f"[launcher] {rec.name}: incarnation {rec.incarnation} FAILED: {reason}; stopping every node")
        tail = self._node_log_tail(rec, reason)
        if tail:
            self.log(tail)
        if rec.spares is None:
            self._release(rec, rec.incarnation)
            return
        # Spare nodes: which ranks' streams (their heartbeat) ended on their own, before the
        # release ends the others; a silent rank whose stream ended is lost now. A node whose
        # stop the release does not confirm keeps its stream and is checked at the relaunch.
        sp = rec.spares
        sp.stream_ended = {r for r, p in enumerate(rec.job.procs) if p.poll() is not None} if rec.job else set()
        # the watch's checks without an answer count on only for a rank whose stream ended;
        # a node with an open stream whose stop is not confirmed counts from the relaunch on
        sp.silent = {r: c for r, c in rec.unreachable.items() if c and r in sp.stream_ended}
        if silent_rank is not None and silent_rank in sp.stream_ended:
            self._mark_lost(rec, silent_rank, f"{reason}; its stream ended")
        self._release(rec, rec.incarnation, skip=self._lost_names(rec))

    # -- spare nodes (island_spares) ---------------------------------------------------
    def _node_health_lost(self, rec: _Island, rank: int, reasons: list[str], reading: Any) -> None:
        """Rank ``rank``'s node turned lost (``node_health``: bad twice in a row; called once,
        by ``job_status`` after it let go of the ops lock). ``reasons``: the check's reasons
        for the island's job; ``reading``: the ``node_health.Reading`` that judged it (a
        reading taken of another node than the rank's current one, or before a spare took the
        rank's place, is ignored: :meth:`_health_reading` stamps it). The island fails; with spare nodes the node is lost (replaced at the
        relaunch), else the relaunch runs on the same nodes as for any failure. It takes the
        ops lock itself (reentrant: safe from inside the watch too) and adds nothing when the
        node is already lost or the island already failed."""
        with self._lock:
            if not 0 <= int(rank) < len(rec.spec.nodes) or (rec.spares is not None and rank in rec.spares.lost):
                return
            if rec.spares is None and rec.verdict is not None:  # it ended meanwhile (as without spares on main)
                return
            node = rec.spec.nodes[rank].name
            stamp = rec.health_stamps.get(id(reading))
            if stamp is not None and (stamp[0] != node or stamp[1] != rec.rank_gen.get(rank, 0)):
                self.log(f"[launcher] {rec.name}: a health loss of {stamp[0]} for rank {rank} ignored (a spare "
                         f"took rank {rank}'s place since; it is {node} now)")
                return  # stale (again, if asked again)
            rec.health_stamps.pop(id(reading), None)
            reason = (f"rank {rank} ({node}) node health bad in {node_health.BAD_CHECKS} checks in a row: "
                      f"{'; '.join(reasons) or 'lost'}")
            notes = self._health_notes(rec, rank)
            if notes:
                reason = f"{reason}; notes: {'; '.join(notes)}"[:500]
            rec.health_lost[rank] = list(reasons)  # the unified table reads it as a lost node
            if rec.spares is not None:
                self._mark_lost(rec, rank, reason, answers=True)
            if rec.verdict is None:
                self._fail(rec, reason)

    @staticmethod
    def _health_notes(rec: _Island, rank: int) -> list[str]:
        """The notes of the node's health record (``rec.health[rank].record()``), if any."""
        try:
            record = rec.health[rank].record()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - no health record: no notes
            return []
        return [str(n) for n in (record or {}).get("notes") or ()]

    def _lost_names(self, rec: _Island) -> frozenset[str]:
        sp = rec.spares
        return frozenset(rec.spec.nodes[r].name for r in sp.lost) if sp is not None else frozenset()

    def _mark_lost(self, rec: _Island, rank: int, reason: str, *, answers: bool = False) -> None:
        """Rank ``rank``'s node is lost: quarantined (with the stop it owes), its claim given
        up, its stream ended, ``island_node_lost`` written. ``answers``: it still answers over
        SSH (a health loss), so it is stopped now in the background, not waited for; once that
        stop is confirmed it owes nothing (:meth:`_stop_lost`)."""
        from .island_spares import NODE_LOST, RELEASE_NODE_HINT, owed_stop  # noqa: PLC0415

        sp = rec.spares
        if rank in sp.lost:
            return
        sp.lost[rank] = reason
        node = rec.spec.nodes[rank]
        inc = rec.incarnation
        owed = [] if inc is None else [owed_stop(self.run_id, rec.name, inc, rank, rec.spec.docker)]
        from .island_spares import lease_holder_id  # noqa: PLC0415

        holder = lease_holder_id(sp.leases) if sp.leases is not None else None
        if self._quarantine(rec, node, reason=reason, owed=owed, lease_holder=holder):
            sp.records.unclaim(node.name, rec.name)  # not written: the claim keeps the run's islands off it
        elif inc is not None:  # no record to owe it in: its stop is in a ledger row of its own now
            from . import ssh_trainer_ledger  # noqa: PLC0415

            ssh_trainer_ledger.record_owed(run_id=rec.spec.run_id, island=rec.name, incarnation=inc, node=node,
                                           rank=rank, docker=rec.spec.docker, kill_grace_s=self.kill_grace_s)
        keep = getattr(sp.leases, "keep_all", None)
        if keep is not None:  # its cross-run lease is held (out of the renewals) until release-node
            try:
                keep(nodes=[node])
            except Exception as error:  # noqa: BLE001 - the quarantine still keeps this run's islands off it
                self.log(f"[launcher] WARN: {rec.name}: the lease of lost node {node.name} not kept ({error})")
        sp.emit(NODE_LOST, node=node.name, rank=int(rank), incarnation=inc, reason=reason[:500])
        self.log(f"[launcher] CRITICAL: {rec.name}: node {node.name} (rank {rank}) is lost: {reason}; it is "
                 "quarantined and the island is stopped on its other nodes (the lost node is not waited for); "
                 f"to give it back once fixed, {RELEASE_NODE_HINT.replace('<name>', node.name)}")
        if rec.job is not None and inc == rec.job.incarnation and rank < len(rec.job.procs):
            proc = rec.job.procs[rank]
            if proc.poll() is None:
                proc.terminate()
        if answers and inc is not None:
            thread = threading.Thread(target=self._stop_lost, args=(rec, rank, node, int(inc)), daemon=True,
                                      name=f"{rec.name}-i{inc}-stop-{node.name}")
            sp.stop_threads.append(thread)
            thread.start()

    def _stop_lost(self, rec: _Island, rank: int, node: Any, inc: int) -> None:
        """The background stop of a lost node that still answers (no ops lock): confirmed,
        it owes nothing (its quarantine record and ledger row say so); else it stays owed."""
        container = container_name(rec.spec.run_id, rec.name, inc, rank) if rec.spec.docker else None
        self._run_owed_stop(rec, node, inc, container, what="lost node")

    def _still_owed(self, rec: _Island, node: Any, inc: int) -> bool:
        """The node's quarantine record (read now) still lists its stop of ``inc`` (a
        release-node may have run it, and removed the record, since)."""
        if self._pending_owes(rec, node.name, inc):
            return True
        doc = rec.spares.records.own_quarantine(node.name)
        if doc is not None and doc.get("unreadable"):  # not known: owed (a stop is idempotent)
            return True
        return any(str(o.get("island")) == rec.name and int(o.get("incarnation", -1)) == int(inc)
                   for o in (doc or {}).get("owed") or ())

    @staticmethod
    def _pending_owes(rec: _Island, name: str, inc: int) -> bool:
        """A quarantine the store refused (``spares.pending``) lists that stop of ``name``."""
        held = rec.spares.pending.get(name)
        return held is not None and any(str(o.get("island")) == rec.name and int(o.get("incarnation", -1)) == int(inc)
                                        for o in held.get("owed") or ())

    def _run_owed_stop(self, rec: _Island, node: Any, inc: int, container: str | None, *, what: str) -> bool:
        """One owed stop of ``node`` (no ops lock); True once it went through. The record is
        read again before the stop (nothing to do once it is gone); a confirmed stop is
        settled (:meth:`~.island_spares.NodeRecords.settle`, a marker, never the record
        itself) and its owed ledger row closed, even when the marker cannot be written."""
        from . import island_spares as isp  # noqa: PLC0415
        from . import ssh_trainer_ledger  # noqa: PLC0415

        sp = rec.spares
        try:
            if not self._still_owed(rec, node, inc):
                return False
        except Exception as error:  # noqa: BLE001 - not read: tried at the next check
            self.log(f"[launcher] {rec.name}: the quarantine record of {node.name} not read ({error})")
            return False
        script = isp.owed_kill_script(rec.spec.run_id, rec.name, inc, self.kill_grace_s, container)
        try:
            done = self.ops.run(node, script, timeout=self.kill_grace_s + 60.0)
        except Exception as error:  # noqa: BLE001 - owed in its quarantine record
            self.log(f"[launcher] {rec.name}: stop of {what} {node.name} failed ({error}); it stays owed")
            return False
        if done.returncode != 0:
            self.log(f"[launcher] {rec.name}: stop of {what} {node.name} ended rc {done.returncode}; it stays owed")
            return False
        with sp.owed_lock:  # the release that records owed stops reads this under the same lock
            sp.stopped.add((node.name, int(inc)))  # the release owes it no row, even if the marker below fails
            held = sp.pending.get(node.name)
            if held is not None:  # a quarantine not written yet owes it no more either
                held["owed"] = [o for o in held.get("owed") or ()
                                if not (str(o.get("island")) == rec.name and int(o.get("incarnation", -1)) == int(inc))]
            try:
                sp.records.settle(node.name, run_id=self.run_id, island=rec.name, incarnation=inc)
            except Exception as error:  # noqa: BLE001 - its record still lists the stop (release-node runs it again)
                self.log(f"[launcher] WARN: {rec.name}: the stop of {node.name} is not marked settled ({error})")
            finally:
                ssh_trainer_ledger.release_owed(run_id=rec.spec.run_id, island=rec.name, incarnation=inc,
                                                node=node.name)
        self.log(f"[launcher] {rec.name}: {what} {node.name} confirmed its stop of incarnation {inc}; it owes "
                 "nothing for it (it stays quarantined)")
        return True

    def _next_incarnation(self, rec: _Island) -> int:
        """One past the newest incarnation the island ran or tried (its jobs journal): a start
        that failed after it was recorded keeps its number, its state dirs and its ledger row."""
        used = [int(k) for k in read_jobs(self.jobs_path(rec.name))]
        if rec.incarnation is not None:
            used.append(int(rec.incarnation))
        return max(used) + 1 if used else 0

    def _roll_back(self, rec: _Island, picks: dict[int, Any], nxt: int, error: Any, old_spec: Any,
                   old_islands: Any, old_health: Any) -> None:
        """A start on spares failed: incarnation ``nxt`` is stopped where it began; a spare
        whose stop is not confirmed is lost (:meth:`_quarantine_unstopped`), the others go back
        to the pool (a spare named in the error is quarantined); the island is back on its old
        nodes, which remember an incarnation not stopped on every node."""
        from . import island_spares as isp  # noqa: PLC0415
        from . import ssh_trainer_ledger  # noqa: PLC0415

        sp = rec.spares
        new_islands = rec.islands if rec.islands is not old_islands else None
        unstopped: frozenset[str] = frozenset()
        if new_islands is not None and str(nxt) in read_jobs(self.jobs_path(rec.name)) \
                and not new_islands.release_incarnation(nxt):
            unstopped = new_islands.unconfirmed.get(int(nxt), frozenset())
        lost = set()
        for rank, node in sorted(picks.items()):
            if node.name in unstopped:  # it may still run incarnation nxt: lost, owing that stop
                self._quarantine_unstopped(rec, node, rank, nxt, error)
                lost.add(node.name)
                continue
            isp.unpick(sp, rec.name, node)
            if node.name in str(error):
                self._quarantine(rec, node, reason=f"its start as a spare failed: {error}",
                                 lease_holder=self._lease_holder(sp))
        if new_islands is not None:
            if unstopped and set(unstopped) <= lost:  # the rest confirmed: each lost spare owes its own row
                ssh_trainer_ledger.record("released", run_id=rec.spec.run_id, island=rec.name, incarnation=nxt)
                new_islands.unreleased.discard(int(nxt))
            old_islands.unreleased |= set(new_islands.unreleased)
        rec.spec, rec.islands, rec.health = old_spec, old_islands, old_health
        try:
            self._record_nodes(rec)  # back to the nodes it runs on
        except Exception as warn:  # noqa: BLE001 - the next pick records them again
            self.log(f"[launcher] WARN: {rec.name}: its lease record not put back ({warn})")

    def _quarantine(self, rec: _Island, node: Any, **kw: Any) -> bool:
        """``NodeRecords.quarantine``; False when the store refused it (e.g. a record it could
        not read, whose owed stops would be written over): the quarantine is kept in memory
        (``spares.pending``: out of this island's picks, its stops owed), the caller keeps the
        node's claim (the run's other islands), and each check writes it again
        (:meth:`_flush_pending`)."""
        sp = rec.spares
        try:
            sp.records.quarantine(node.name, island=rec.name, **kw)
            return True
        except Exception as error:  # noqa: BLE001 - kept in memory, written again at each check
            held = sp.pending.get(node.name) or {}
            sp.pending[node.name] = {**held, **kw, "owed": [*(held.get("owed") or ()), *map(dict, kw.get("owed") or ())]}
            self.log(f"[launcher] CRITICAL: {rec.name}: node {node.name} is not quarantined in the store ({error}); "
                     "this launcher keeps it out of its picks, holds its claim and owes its stops, and writes "
                     "the record again at each check")
            return False

    def _flush_pending(self, rec: _Island) -> None:
        """Write the quarantines the store refused before (:meth:`_quarantine`); once one is
        written, the node's claim is given up (its record keeps every island off it)."""
        sp = rec.spares
        if sp is None or not sp.pending:
            return
        for name, kw in list(sp.pending.items()):
            try:
                sp.records.quarantine(name, island=rec.name, **kw)
            except Exception as error:  # noqa: BLE001 - tried again at the next check
                self.log(f"[launcher] {rec.name}: the quarantine of {name} still not written ({error})")
                continue
            sp.pending.pop(name, None)
            sp.records.unclaim(name, rec.name)  # a lost node or a lost spare: its record holds it now
            self.log(f"[launcher] {rec.name}: the quarantine of {name} is written to the store now")

    @staticmethod
    def _lease_holder(sp: Any) -> str | None:
        """Who holds the spares' cross-run leases (a quarantine names it, so release-node ends it)."""
        from .island_spares import lease_holder_id  # noqa: PLC0415

        return lease_holder_id(sp.leases) if sp.leases is not None else None

    def _quarantine_unstopped(self, rec: _Island, node: Any, rank: int, incarnation: int, error: Any) -> None:
        """A spare whose start failed and whose stop of that incarnation is not confirmed: it
        is lost, as a lost node is: quarantined with the stop it owes (the record keeps every
        island of every run off it), that stop in a ledger row of its own (priced as that node,
        open until the stop is confirmed), its claim given up and its lease kept."""
        from . import ssh_trainer_ledger  # noqa: PLC0415
        from .island_spares import owed_stop  # noqa: PLC0415

        sp = rec.spares
        if self._quarantine(rec, node,
                            reason=f"its start as a spare failed ({error}) and its stop is not confirmed",
                            owed=[owed_stop(self.run_id, rec.name, incarnation, rank, rec.spec.docker)],
                            lease_holder=self._lease_holder(sp)):
            sp.records.unclaim(node.name, rec.name)
        ssh_trainer_ledger.record_owed(run_id=rec.spec.run_id, island=rec.name, incarnation=incarnation, node=node,
                                       rank=rank, docker=rec.spec.docker, kill_grace_s=self.kill_grace_s)
        keep = getattr(sp.leases, "keep_all", None)
        if keep is not None:
            try:
                keep(nodes=[node])
            except Exception as warn:  # noqa: BLE001 - the quarantine still keeps the islands off it
                self.log(f"[launcher] WARN: {rec.name}: the lease of {node.name} not kept ({warn})")
        self.log(f"[launcher] CRITICAL: {rec.name}: spare {node.name} may still run incarnation {incarnation} (its stop "
                 "is not confirmed): it is quarantined and owes that stop")

    def _retry_owed_stops(self, rec: _Island) -> None:
        """At each check of a waiting island: every node this island quarantined that still
        owes a stop of this run gets it tried again, in the background (no ops lock); a stop
        that goes through settles it (it stays quarantined)."""
        sp = rec.spares
        for node in sp.pool:
            doc = sp.records.own_quarantine(node.name)
            rows = [dict(o) for o in (doc or {}).get("owed") or ()
                    if str(o.get("island")) == rec.name and str(o.get("run_id") or self.run_id) == self.run_id]
            running = rec.owed_retries.get(node.name)
            if not rows or (running is not None and running.is_alive()):
                continue
            thread = threading.Thread(target=self._stop_owed, args=(rec, node, rows), daemon=True,
                                      name=f"{rec.name}-owed-{node.name}")
            rec.owed_retries[node.name] = thread
            sp.stop_threads.append(thread)
            thread.start()

    def _stop_owed(self, rec: _Island, node: Any, rows: list[dict[str, Any]]) -> None:
        """The owed stops of ``rows`` on ``node`` (:meth:`_retry_owed_stops`), each only while
        the node's record still lists it."""
        for row in rows:
            if not self._run_owed_stop(rec, node, int(row["incarnation"]), row.get("container") or None,
                                       what="quarantined node"):
                return

    def _join_owed_retries(self, rec: _Island, timeout: float = 5.0) -> None:
        """The island goes down or the launcher closes: its owed-stop retries get a short
        while to finish; finished threads are dropped."""
        sp = rec.spares
        if sp is None:
            return
        end = time.monotonic() + timeout
        for name, thread in list(rec.owed_retries.items()):
            thread.join(max(0.0, end - time.monotonic()))
            if not thread.is_alive():
                rec.owed_retries.pop(name, None)
        sp.stop_threads[:] = [t for t in sp.stop_threads if t.is_alive()]

    def _probe_silent(self, rec: _Island) -> list[int]:
        """Ranks whose stop the release did not confirm, or whose stream ended when the island
        failed, and that do not answer over SSH now (not lost yet); one answer more or less
        each relaunch poll, lost at ``unreachable_polls`` in a row."""
        sp = rec.spares
        silent = []
        for rank in sorted((sp.stream_ended | sp.unconfirmed) - set(sp.lost)):
            node = rec.spec.nodes[rank]
            try:
                answered = self.ops.run(node, "true", timeout=60.0).returncode == 0
            except Exception:  # noqa: BLE001 - a timeout is no answer
                answered = False
            if answered:
                sp.silent[rank] = 0
                continue
            sp.silent[rank] = sp.silent.get(rank, 0) + 1
            ended = rank in sp.stream_ended or self._stream_ended(rec, rank)
            # a node whose stop is not confirmed and whose old stream still runs (sshd refusing new
            # sessions while the old connection lives) is lost only after twice as many checks
            limit = self.unreachable_polls if ended else 2 * self.unreachable_polls
            if sp.silent[rank] >= limit:
                why = "its stream ended" if ended else "its stop was not confirmed, its stream still open,"
                self._mark_lost(rec, rank, f"rank {rank} ({node.name}): {why} and it does not answer "
                                           f"over SSH ({sp.silent[rank]} checks in a row)")
            else:
                silent.append(rank)
        return silent

    @staticmethod
    def _stream_ended(rec: _Island, rank: int) -> bool:
        """Whether rank ``rank``'s stream of the current incarnation ended (its heartbeat)."""
        job = rec.job
        return (job is not None and job.incarnation == rec.incarnation and rank < len(job.procs)
                and job.procs[rank].poll() is not None)

    def _relaunch_with_spares(self, rec: _Island) -> int | None:
        """The relaunch of an island with spare nodes: release the failed incarnation on every
        node but the lost ones, then start the next incarnation on the same nodes (nothing
        lost) or with spares in the lost nodes' places (:meth:`_replace`). The spares' health
        checks run outside the ops lock (:meth:`_precheck`)."""
        with self._lock:
            inc = rec.incarnation
            if inc is not None and inc not in rec.released:
                silent = self._probe_silent(rec)
                if silent:
                    names = [rec.spec.nodes[r].name for r in silent]
                    self.log(f"[launcher] {rec.name}: node(s) {names} do not answer over SSH; no new start "
                             "this poll")
                    return None
                if not self._release(rec, inc, skip=self._lost_names(rec)):
                    self.log(f"[launcher] {rec.name}: incarnation {inc} not confirmed stopped; no new start "
                             "this poll")
                    return None
            if not rec.spares.lost:
                nxt = 0 if inc is None else inc + 1
                try:
                    return self._launch(rec, nxt)
                except Exception as error:  # noqa: BLE001 - the controller retries next poll
                    self.log(f"[launcher] {rec.name}: start of incarnation {nxt} failed: {error}")
                    self._release(rec, nxt)
                    return None
            if not self._replace_due(rec):
                return None
        return self._replace_checked(rec)

    def _replace_checked(self, rec: _Island) -> int | None:
        """:meth:`_replace` with the spares' health checks outside the ops lock: check, pick
        under the lock, and check the next candidates when a checked one was taken meanwhile."""
        with self._lock:
            self._flush_pending(rec)  # quarantines the store refused before
        if rec.spares.waiting:  # a backoff check: a lost node back (a network partition) gets its owed stop
            self._retry_owed_stops(rec)
        health: dict[str, tuple[bool, str]] = {}
        for _ in range(len(rec.spares.pool) + 1):  # each round checks at least one more candidate
            health.update(self._precheck(rec, known=health))
            with self._lock:
                job_id = self._replace(rec, health)
            if job_id is not _RECHECK:
                return job_id
        return None

    def _precheck(self, rec: _Island, known: Any = ()) -> dict[str, tuple[bool, str]]:
        """The health checks of the spares a replacement would pick, but ``known`` (SSH; no
        ops lock)."""
        from . import island_spares as isp  # noqa: PLC0415

        sp = rec.spares
        return isp.check_candidates(sp, rec.name, list(rec.spec.nodes), dict(sp.lost), ops=self.ops,
                                    need=len(sp.lost), known=known)

    def _replace_due(self, rec: _Island) -> bool:
        """Whether a replacement is tried this poll: past the replacement limit or the longest
        wait (``max_wait_s``) the relaunch is refused (RelaunchBlocked: the island stops);
        while it waits, an ``island_waiting_spare`` every ``report_every_s`` and nothing
        before the backoff's next check."""
        from . import island_spares as isp  # noqa: PLC0415
        from .fleet import RelaunchBlocked  # noqa: PLC0415

        sp = rec.spares
        now, wall = self.clock(), self.wall()
        lost = sorted(self._lost_names(rec))
        if sp.limit_reached(wall):
            done = sp.recent(wall)
            why = (f"{len(done)} node replacement(s) within {sp.window_s / 3600:.0f} h (max {sp.max_replacements}): "
                   + ", ".join(f"{r.get('lost')} -> {r.get('spare')}" for r in done)
                   + f"; node(s) {lost} lost again: the island stops")
            if not sp.limited:
                sp.limited = True
                sp.emit(isp.REPLACEMENT_LIMIT, replaced=[dict(r) for r in done], lost=lost, reason=why[:500])
                self.log(f"[launcher] CRITICAL: {rec.name}: {why}")
            raise RelaunchBlocked(why)
        waited = sp.waited_s(wall)
        if sp.wait_since is not None and waited >= sp.max_wait_s:
            why = (f"waited {waited:.0f} s for a spare node for {lost} (max {sp.max_wait_s:.0f} s, "
                   f"--spare-max-wait-s): {sp.wait_reason or 'no spare'}; the island stops")
            if not sp.limited:
                sp.limited = True
                sp.emit(isp.WAITING_SPARE, max_wait_s=float(sp.max_wait_s), lost=lost, reason=why[:500],
                        elapsed_s=round(waited), given_up=True)
                self.log(f"[launcher] CRITICAL: {rec.name}: {why}")
            raise RelaunchBlocked(why)
        if sp.report_due(wall):
            sp.emit(isp.WAITING_SPARE, max_wait_s=float(sp.max_wait_s), lost=lost,
                    reason=str(sp.wait_reason or "")[:500], elapsed_s=round(waited),
                    next_check_s=round(max(0.0, sp.next_check - now)))
        return not (sp.waiting and now < sp.next_check)

    def _replace(self, rec: _Island, health: dict[str, tuple[bool, str]] | None = None) -> int | None:
        """Spares in the lost nodes' places and the next incarnation on them; None while no
        spare is there (``waiting_spare``, checked again after the backoff) or the start on
        it failed. ``health``: the candidates' checks (:meth:`_precheck`); the claims, leases
        and records are read again here, and :data:`_RECHECK` when a candidate left is not
        checked yet (the caller checks it outside the lock and asks again). Past the
        replacement limit or the longest wait the relaunch is refused (the island stops)."""
        import dataclasses  # noqa: PLC0415

        from . import island_spares as isp  # noqa: PLC0415

        if not self._replace_due(rec):
            return None
        sp = rec.spares
        now, wall = self.clock(), self.wall()
        current = list(rec.spec.nodes)
        picks: dict[int, Any] = {}
        for rank in sorted(sp.lost):
            try:
                pick = isp.pick_spare(sp, rec.name, current, sp.lost, ops=self.ops, chosen=list(picks.values()),
                                      health=health)
            except BaseException:  # the picks so far are not taken: their claims and leases go back
                for node in picks.values():
                    isp.unpick(sp, rec.name, node)
                raise
            if pick.node is None:
                for node in picks.values():
                    isp.unpick(sp, rec.name, node)
                if pick.unchecked and health is not None:
                    return _RECHECK  # type: ignore[return-value]
                reason = pick.why_none()
                if sp.wait(now, reason, wall=wall):
                    sp.emit(isp.WAITING_SPARE, max_wait_s=float(sp.max_wait_s), lost=sorted(self._lost_names(rec)),
                            reason=reason[:500],
                            group=pick.group, next_check_s=sp.wait_s, elapsed_s=round(sp.waited_s(wall)))
                self.log(f"[launcher] {rec.name}: waiting for a spare node ({reason}); next check in "
                         f"{sp.wait_s:.0f} s")
                return None
            picks[rank] = pick.node
        nxt = self._next_incarnation(rec)  # never one a failed start used (its dirs, its ledger row)
        old_spec, old_islands = rec.spec, rec.islands
        try:
            new_spec = dataclasses.replace(old_spec, nodes=tuple(picks.get(r, n) for r, n in enumerate(current)))
        except SshIslandError as error:  # e.g. a spare without a rendezvous address
            for node in picks.values():
                isp.unpick(sp, rec.name, node)
                self._quarantine(rec, node, reason=f"cannot take a node's place: {error}",
                                 lease_holder=self._lease_holder(sp))
            sp.wait(now, f"spare refused: {error}", wall=wall)
            self.log(f"[launcher] {rec.name}: spare(s) {[n.name for n in picks.values()]} refused: {error}")
            return None
        old_health = rec.health
        for rank in picks:  # a health reading of the node it had is stale from now on
            rec.rank_gen[rank] = rec.rank_gen.get(rank, 0) + 1
        try:
            new_islands = self._islands_for(rec, new_spec)
            # an earlier incarnation not confirmed stopped on a node (a failed start): stopped
            # there first (SshTrainerIslands._stop_unreleased), never started over
            new_islands.unreleased |= set(old_islands.unreleased)
            rec.spec, rec.islands = new_spec, new_islands
            rec.health = self._health_watches(new_spec)  # one watch per rank of the new node set
            self._record_nodes(rec)  # before any start: a launcher gone meanwhile leases the spare, not the lost node
            job_id = self._launch(rec, nxt)  # every node's preflight first: the spare's image, cards, scratch
        except Exception as error:  # noqa: BLE001 - back to the old nodes; checked again after the backoff
            self.log(f"[launcher] {rec.name}: start of incarnation {nxt} on spare(s) "
                     f"{[n.name for n in picks.values()]} failed: {error}")
            self._roll_back(rec, picks, nxt, error, old_spec, old_islands, old_health)
            sp.nodes.pop(str(nxt), None)
            sp.save()
            sp.wait(now, f"the start on the spare node(s) failed: {error}", wall=wall)
            return None
        resume = sp.resume_probe() if sp.resume_probe is not None else None
        step = None if resume is None else resume.get("step")
        for rank, node in sorted(picks.items()):
            row = {"at": wall, "island": rec.name, "rank": int(rank), "lost": current[rank].name, "spare": node.name,
                   "incarnation": int(nxt), "reason": sp.lost[rank][:300], "resume_step": step}
            sp.replacements.append(row)
            sp.emit(isp.NODE_REPLACED, **{k: v for k, v in row.items() if k != "at"},
                    resume=resume, group=isp.fabric_group(node))
            self.log(f"[launcher] {rec.name}: node {current[rank].name} (rank {rank}) replaced by spare {node.name}; "
                     f"incarnation {nxt} resumes from the newest checkpoint"
                     + (f" (step {step}, {resume.get('source')})" if resume else " (none in the store yet)"))
        sp.save()
        sp.lost.clear()
        sp.stream_ended.clear()
        sp.unconfirmed.clear()
        sp.silent.clear()
        sp.done_waiting()
        return job_id

    # -- what the fleet controller reads of the spares ------------------------------------
    def waiting_spare(self, name: str) -> bool:
        """The island waits for a spare node (the controller does not give it up on its
        recover timeout meanwhile)."""
        rec = self.islands.get(name)
        return bool(rec is not None and rec.spares is not None and rec.spares.waiting)

    def island_events(self, name: str) -> list[dict[str, Any]]:
        """The island's spare-node events since the last call (``island_node_lost``, ...)."""
        rec = self.islands.get(name)
        if rec is None or rec.spares is None:
            return []
        return rec.spares.take_events()  # its own lock: never waits for a relaunch in flight

    def spares_view(self, name: str) -> dict[str, Any] | None:
        """The island record's ``spares`` (None: an island without spare nodes)."""
        rec = self.islands.get(name)
        sp = None if rec is None else rec.spares
        if sp is None:
            return None
        current = [n.name for n in rec.spec.nodes]
        last = sp.replacements[-1] if sp.replacements else None
        return {"nodes": current, "spare_nodes": sorted(n.name for n in sp.pool if n.name not in current),
                "lost": sorted(self._lost_names(rec)), "waiting_spare": bool(sp.waiting),
                "wait_reason": sp.wait_reason, "waiting_since": sp.wait_since if sp.waiting else None,
                "replacements": len(sp.replacements),
                "replacements_window": len(sp.recent(self.wall())), "max_replacements": int(sp.max_replacements),
                "last_replaced_at": None if last is None else last.get("at"), "max_wait_s": float(sp.max_wait_s)}

    def _node_log_tail(self, rec: _Island, reason: str, lines: int = 20) -> str:
        """The last lines of the failed rank's local log copy (rank > 0)."""
        if self.events_dir is None:
            return ""
        for rank in range(1, len(rec.spec.nodes)):
            if f"rank {rank} " not in reason:
                continue
            path = self.events_dir / f"{rec.name}-i{rec.incarnation}-node{rank}.log"
            try:
                text = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
            except OSError:
                return ""
            return "\n".join(f"[{rec.name} n{rank}] {line}" for line in text)
        return ""

    # -- release ----------------------------------------------------------------------
    def _release(self, rec: _Island, incarnation: int, *, skip: frozenset[str] = frozenset()) -> bool:
        if incarnation in rec.released:
            return True
        sp = rec.spares
        if sp is None:
            if rec.job is not None and rec.job.incarnation == incarnation:
                ok = rec.islands.release(rec.job)
            else:
                ok = rec.islands.release_incarnation(incarnation)
        else:
            # spare nodes: a lost node is not waited for and owes its stop (its own ledger row)
            # unless that stop was confirmed (owed_lock: never a row left open after it); a node
            # whose stop is not confirmed keeps its stream (the relaunch checks it)
            with sp.owed_lock:
                owed = frozenset(n for n in skip
                                 if (n, int(incarnation)) not in sp.stopped and self._owes(rec, n, incarnation))
                if rec.job is not None and rec.job.incarnation == incarnation:
                    ok = rec.islands.release(rec.job, skip=skip, owed=owed, keep_unconfirmed=True)
                else:
                    ok = rec.islands.release_incarnation(incarnation, skip=skip, owed=owed)
            failed = rec.islands.unconfirmed.get(int(incarnation), frozenset())
            sp.unconfirmed = {r for r, n in enumerate(rec.spec.nodes) if n.name in failed}
        if ok:
            rec.released.add(incarnation)  # SshTrainerIslands closed its ledger row
        else:
            self.log(f"[launcher] {rec.name}: release of incarnation {incarnation} NOT confirmed on every node")
        return ok

    def _owes(self, rec: _Island, name: str, incarnation: int) -> bool:
        """Whether lost node ``name`` still owes its stop of ``incarnation`` (its quarantine
        record in this run lists it; a node another run quarantined never ran it)."""
        if self._pending_owes(rec, name, incarnation):
            return True
        doc = rec.spares.records.own_quarantine(name)
        if doc is None:
            return False
        if doc.get("unreadable"):  # not known: owed (its own row; a stop is idempotent)
            return True
        return any(str(o.get("island")) == rec.name and int(o.get("incarnation", -1)) == int(incarnation)
                   for o in doc.get("owed") or ())

    # -- tape recovery ----------------------------------------------------------------
    def recover_tape(self, name: str, collector: Any) -> bool:
        """Complete ``collector`` from node 0's own log (the echo's source) of the current
        incarnation; True when the tape is then finalized."""
        from yeto.rl import event_echo

        rec = self.islands[name]
        if rec.incarnation is None:
            return False
        paths = remote_paths(rec.spec.run_id, name, rec.incarnation)
        script = f"cat \"{paths['log']}\""  # the login HOME holds the state dir
        try:
            done = self.ops.run(rec.spec.nodes[0], script, timeout=300.0)
        except Exception as error:  # noqa: BLE001 - stay incomplete
            self.log(f"[launcher] {name}: tape recovery read failed ({error})")
            return False
        if done.returncode != 0:
            self.log(f"[launcher] {name}: tape recovery read failed (rc {done.returncode})")
            return False
        raws = [r for r in (event_echo.parse_line(line) for line in (done.stdout or "").splitlines())
                if r is not None and r is not event_echo.INVALID]
        with tempfile.TemporaryDirectory(prefix="yeto-ssh-tape-") as tmp:
            path = Path(tmp) / "node0.jsonl"
            path.write_text("".join(r + "\n" for r in raws), encoding="utf-8")
            before = collector.count
            finalized = collector.recover_from_file(path)
        self.log(f"[launcher] {name}: tape recovered from node 0's log: +{collector.count - before} record(s), "
                 f"finalized={finalized}")
        return finalized


# ---------------------------------------------------------------------------- dry run
def _miles_sort_key(address: str) -> tuple[int, list[int]]:
    """Miles' placement order of a node (``miles.ray.placement_group.sort_key``): its
    IPv4 address numerically, else the characters of the name."""
    try:
        return 0, [int(p) for p in str(address).split(".")]
    except ValueError:
        return 1, [ord(c) for c in str(address)]


def rank_layout(spec: SshTrainerSpec, *, tensor_parallel: int, pipeline_parallel: int,
                context_parallel: int = 1) -> dict[str, Any]:
    """Which training ranks each node holds. Miles orders the trainer's GPUs by node IP
    (then GPU id), and Megatron puts PP outermost (``tp-cp-ep-dp-pp``), so PP stage
    ``s`` holds global ranks ``[s * W/PP, (s + 1) * W/PP)``."""
    addresses = spec.addresses()
    gpus = spec.nodes[0].cards_per_node
    world = gpus * len(spec.nodes)
    pp = max(1, int(pipeline_parallel))
    per_stage = world // pp if world % pp == 0 else None
    order = sorted(range(len(spec.nodes)), key=lambda i: (_miles_sort_key(addresses[i]), i))
    position = {node: pos for pos, node in enumerate(order)}
    nodes, stages = [], {}
    for rank, node in enumerate(spec.nodes):
        lo = position[rank] * gpus
        ranks = list(range(lo, lo + gpus))
        pp_stages = sorted({r // per_stage for r in ranks}) if per_stage else []
        for s in pp_stages:
            stages.setdefault(str(s), []).append(rank)
        nodes.append({"node_rank": rank, "name": node.name, "address": addresses[rank],
                      "trainer_position": position[rank], "global_ranks": f"{lo}-{lo + gpus - 1}",
                      "pp_stages": pp_stages})
    warnings = []
    if per_stage is None:
        warnings.append(f"{world} GPUs do not split into {pp} pipeline stages")
    elif per_stage % gpus and gpus % per_stage:
        warnings.append(f"a pipeline stage of {per_stage} GPUs does not align with {gpus}-GPU nodes")
    if order != list(range(len(spec.nodes))):
        warnings.append("the trainer orders nodes by IP, not by inventory order: node_rank "
                        f"{order} hold trainer positions 0..{len(order) - 1}; list the inventory in IP "
                        "order to make PP stage s the nodes listed at 2s and 2s+1")
    return {"world": world, "tensor_parallel": int(tensor_parallel), "context_parallel": int(context_parallel),
            "pipeline_parallel": pp, "rank_order": "node IP, then GPU id (Miles placement group)",
            "nodes": nodes, "pp_stage_nodes": {s: [spec.nodes[r].name for r in v] for s, v in stages.items()},
            "warnings": warnings}


def node_health_summary(spec: SshTrainerSpec) -> str | None:
    """``on (cuda)`` / ``off`` (``off on [nodes]`` for some) when the inventory declares
    ``node_health``; None when it does not (the dry run prints nothing then, a DEBUG log
    says it)."""
    import logging

    family = spec.device_family
    checker = node_health.checker_for(family).enabled
    off = [n.name for n in spec.nodes if getattr(n, "node_health", None) == "off"]
    state = "off" if not checker or len(off) == len(spec.nodes) else (
        f"on ({family})" + (f", off on {off}" if off else ""))
    if all(getattr(n, "node_health", None) is None for n in spec.nodes):
        logging.getLogger(__name__).debug("node health: %s (not declared in the inventory)", state)
        return None
    return state


def dry_run_record(args: Any, spec: SshTrainerSpec, name: str, *, mode: str = "resident") -> dict[str, Any]:
    """What a launch would run on every node of ``ssh:`` island ``name``: the
    ``docker run`` command line of incarnation 0 (environment by name; the island
    script by size), the environment names, the shipped files and the rank layout."""
    import hashlib

    from yeto import runs

    paths = remote_paths(spec.run_id, name, 0)  # $HOME: the login HOME of the node
    script = spec.run_script or shlex.join(spec.command)
    word = f"<island script: {len(script)} bytes, sha256 {hashlib.sha256(script.encode()).hexdigest()[:16]}>"
    nodes = []
    for rank, node in enumerate(spec.nodes):
        env = node_env(spec, rank, {})
        entry = {"node_rank": rank, "name": node.name, "host": node.host, "runtime": node.runtime,
                 "env": sorted(env), "state_dir": paths["dir"]}
        if spec.docker:
            container = container_name(spec.run_id, name, 0, rank)
            entry["container"] = container
            entry["docker_run"] = docker_job(spec, rank, paths, container, env, word)
        else:
            entry["command"] = f"cd {shlex.quote(spec.workdir)} && bash -c {word}"
        nodes.append(entry)
    warnings = [f"file mount {t}: {local} is not a readable file" for t, local in sorted(spec.files.items())
                if not Path(str(local)).expanduser().is_file()]
    from .ssh_trainer_island import activation

    scripts, required = activation(spec)
    if len(spec.nodes) > 1 and spec.device_family == "npu":
        bare = [n.name for n in spec.nodes if not n.interface]
        if bare:
            warnings.append(f"nodes {bare} declare no interface: HCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME are not "
                            "set, so HCCL picks the interface itself; set `interface` in the inventory")
    from .ssh_trainer_island import code_source

    source = code_source(args)
    code = {"source": source, "workdir": spec.workdir or None}
    if spec.code_pack is not None:
        code["pack"] = {"uri": spec.code_pack[0], "digest": spec.code_pack[1],
                        "node_tree": f"~/.cache/yeto/codepacks/{spec.code_pack[1]} (linked as ~/sky_workdir)"}
    if spec.docker and spec.workdir and spec.code_pack is None and source != "release":
        for node in spec.nodes:
            inside = [dst for dst in node.mounts.values()
                      if spec.workdir == dst or spec.workdir.startswith(str(dst).rstrip("/") + "/")]
            if not inside:
                warnings.append(f"node {node.name}: workdir {spec.workdir} is under no inventory mount, so the "
                                "image itself must hold the yeto checkout there")
                break
    health = node_health_summary(spec)
    from .island_spares import spare_view  # noqa: PLC0415

    spares = spare_view(args, spec)
    return {
        "mode": mode,
        **({"node_health": health} if health is not None else {}),
        "warnings": warnings,
        **({"spares": spares} if spares is not None else {}),
        "code": code,
        "journal": str(runs.run_dir(spec.run_id) / RESIDENT_DIR / f"{name}-jobs.json") if mode == "resident" else None,
        "env_order": "fabric defaults < inventory node env < --island-env and the island task env"
                     + ("; that environment is kept across the activation scripts (saved before, restored "
                        "after: a script never overrides it)" if scripts else ""),
        "image": spec.image or spec.nodes[0].image,
        **({"activate": {"scripts": list(scripts), "required": required}} if scripts else {}),
        "files": sorted(spec.files),
        "nodes": nodes,
        "rank_layout": rank_layout(spec, tensor_parallel=int(getattr(args, "tensor_parallel", 1) or 1),
                                   pipeline_parallel=int(getattr(args, "pipeline_parallel", 1) or 1),
                                   context_parallel=int(getattr(args, "context_parallel", 1) or 1)),
    }

