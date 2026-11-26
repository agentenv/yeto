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
from typing import Any

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
                 log: Callable[[str], None] | None = None) -> None:
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
        self.islands: dict[str, _Island] = {}
        self._lock = threading.Lock()
        self.reattached: set[str] = set()  # islands whose start followed a running incarnation

    # -- registry -------------------------------------------------------------------
    def owns(self, name: str) -> bool:
        return name in self.islands

    def jobs_path(self, name: str) -> Path:
        return self.journal_dir / f"{name}-jobs.json"

    def add(self, name: str, spec: SshTrainerSpec) -> None:
        """Register island ``name`` (no SSH yet)."""
        if name in self.islands:
            raise SshIslandError(f"ssh: island {name} is registered twice")
        sink = self.sink_for(name) if self.sink_for is not None else None
        rec = _Island(name, spec, None)  # type: ignore[arg-type]

        def on_line(line: str) -> None:
            rec.lines.append(line)
            if sink is not None:
                sink(line)

        # node_wait_s=0: a start finds an unreachable node at once and names it; the
        # fleet controller retries the relaunch every poll (a wait here would hold it)
        rec.islands = SshTrainerIslands(spec, name=name, jobs_path=self.jobs_path(name), ops=self.ops,
                                        events_dir=self.events_dir, kill_grace_s=self.kill_grace_s,
                                        scheduled=False, tape=False, on_line=on_line, node_wait_s=0.0)
        rec.health = self._health_watches(spec)
        self.islands[name] = rec

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
        jobs = read_jobs(self.jobs_path(name))
        if not jobs:
            return self._launch(rec, 0)
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
        job = rec.islands.launch(incarnation, None)  # journal and ledger row before any node starts
        return self._reset(rec, job, incarnation)

    def _attach(self, rec: _Island, incarnation: int) -> int:
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
        the release is not confirmed or the start failed (the controller retries)."""
        rec = self.islands[name]
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
        """Release the island's current incarnation; True once every node confirmed it."""
        rec = self.islands.get(name)
        if rec is None:
            return True
        if rec.incarnation is None:  # never started: nothing runs on its nodes
            rec.islands.release_leases()
            return True
        with self._lock:
            ok = self._release(rec, rec.incarnation)
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
                                f"({rec.unreachable[rank]} checks in a row)")
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
        ``node_health_read_failed`` when reads start failing), and a lost node's handling."""
        node = rec.spec.nodes[rank].name
        base = {"node": node, "rank": rank, "incarnation": rec.incarnation}
        if reading.degraded:  # once per node: what its check cannot judge
            rec.health_events.append({"event": node_health.EVENT_DEGRADED, **base, "reasons": list(reading.degraded)})
            self.log(f"[launcher] {rec.name}: {node_health.EVENT_DEGRADED}: rank {rank} ({node}): "
                     f"{'; '.join(reading.degraded)}")
        if reading.event is not None:
            rec.health_events.append({"event": reading.event, **base, **node_health.event_fields(reading)})
            self.log(f"[launcher] {rec.name}: {reading.event}: rank {rank} ({node}) {reading.verdict}: "
                     f"{'; '.join(reading.result.reasons) or '-'}")
        if reading.event == node_health.EVENT_LOST:  # it just turned lost: the hook runs once, after the lock
            rec.health_pending.append((rank, reading.result.job_reasons(len(rec.spec.nodes) > 1), reading))

    def _node_health_lost(self, rec: _Island, rank: int, reasons: list[str], reading: Any) -> None:
        """Node ``rank``'s health turned lost (two bad checks in a row): called once per turn, by
        :meth:`job_status` after it released the ops' lock. ``reading``: the
        :class:`~yeto.hw.node_health.Reading` (its result's facts, the node's record with its
        notes: ``rec.health[rank].record()``). The default: as for a node that stops
        answering, the island fails at once and every node is released; the fleet's recovery
        relaunches it (the unified table reads it as a lost node)."""
        node = rec.spec.nodes[rank].name
        with self._lock:
            if rec.verdict is not None:  # it ended meanwhile
                return
            rec.health_lost[rank] = list(reasons)
            self._fail(rec, f"rank {rank} ({node}) node health bad in {node_health.BAD_CHECKS} checks in a row: "
                            f"{'; '.join(reasons) or 'lost'}")

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

    def _fail(self, rec: _Island, reason: str) -> None:
        rec.verdict, rec.reason = "FAILED", reason
        self.log(f"[launcher] {rec.name}: incarnation {rec.incarnation} FAILED: {reason}; stopping every node")
        tail = self._node_log_tail(rec, reason)
        if tail:
            self.log(tail)
        self._release(rec, rec.incarnation)

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
    def _release(self, rec: _Island, incarnation: int) -> bool:
        if incarnation in rec.released:
            return True
        if rec.job is not None and rec.job.incarnation == incarnation:
            ok = rec.islands.release(rec.job)
        else:
            ok = rec.islands.release_incarnation(incarnation)
        if ok:
            rec.released.add(incarnation)  # SshTrainerIslands closed its ledger row
        else:
            self.log(f"[launcher] {rec.name}: release of incarnation {incarnation} NOT confirmed on every node")
        return ok

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
    return {
        "mode": mode,
        **({"node_health": health} if health is not None else {}),
        "warnings": warnings,
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

