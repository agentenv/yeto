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
  ``health_every_s`` the nodes are asked even while their streams flow;
* **relaunch** (the fleet's recovery): release the failed incarnation (verified;
  unverified: no new start this poll) and start the next one; the trainer
  continues from the store (``--rl-init-from-release`` / its resume store);
* **release** (the run ends, ``yeto stop-run``, abandon, teardown): TERM to the
  container's processes (Modal's stop signal), ``docker rm -f`` after the grace,
  the process group TERM -> KILL; each node must then show neither process nor
  container, else the teardown is unverified (exit 5, the head stays up).

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

from .ssh_trainer_island import (
    KILL_GRACE_S,
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
STRICT_MARKERS = ("[yeto-rl-strict-failure]", "StrictRlInvariantError:",
                  "backend identity mismatch, JOIN refused", "session mismatch (HELLO refused")


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
    code must reach the nodes and its image must be the run's (SystemExit)."""
    from .ssh_trainer_island import code_source, inventory_trainer_nodes, run_image

    for m, spec in enumerate(specs):
        if not is_resident(args, spec, m, len(specs)):
            continue
        try:
            nodes = inventory_trainer_nodes(args, spec)
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
        self.islands[name] = rec

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
        rec.workers_done_at = None
        rec.health_at = self.clock()
        return int(incarnation)

    def _launch(self, rec: _Island, incarnation: int) -> int:
        job = rec.islands.launch(incarnation, None)  # journal and ledger row before any node starts
        return self._reset(rec, job, incarnation)

    def _attach(self, rec: _Island, incarnation: int) -> int:
        job = rec.islands.attach(incarnation)  # its ledger row is open since its start
        return self._reset(rec, job, incarnation)

    # -- the fleet controller's ops -----------------------------------------------------
    def job_status(self, name: str, job_id: Any) -> _Status:
        rec = self.islands[name]
        with self._lock:
            if rec.incarnation is None or int(job_id) != rec.incarnation:
                return _Status("FAILED")  # an incarnation this launcher no longer runs
            self._watch(rec)
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
        if rec is None or rec.incarnation is None:
            return True
        with self._lock:
            ok = self._release(rec, rec.incarnation)
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
            if rec.lock is not None:
                rec.lock.close()
                rec.lock = None

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
                self._node_event(rec, rank, rec.islands.node_state(rec.incarnation, rank), stream_ended=False)
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
    return {
        "mode": mode,
        "warnings": warnings,
        "code": code,
        "journal": str(runs.run_dir(spec.run_id) / RESIDENT_DIR / f"{name}-jobs.json") if mode == "resident" else None,
        "env_order": "fabric defaults < inventory node env < --island-env and the island task env",
        "image": spec.image or spec.nodes[0].image,
        "files": sorted(spec.files),
        "nodes": nodes,
        "rank_layout": rank_layout(spec, tensor_parallel=int(getattr(args, "tensor_parallel", 1) or 1),
                                   pipeline_parallel=int(getattr(args, "pipeline_parallel", 1) or 1),
                                   context_parallel=int(getattr(args, "context_parallel", 1) or 1)),
    }

