"""Trainer island on SSH nodes for the trainer scheduler (S25 N7.3, design D4).

The trainer scheduler (``trainer_loop``) starts and stops the trainer island
through the ``TrainerIslands`` port (``launch`` / ``events`` / ``alive`` /
``release`` / ``release_incarnation`` / ``start_events``). Modal and sky
implement it in :mod:`.trainer_island`; :class:`SshTrainerIslands` implements
it over nodes of the SSH node inventory (:mod:`.node_inventory`), e.g. PPU or
Ascend machines that yeto reaches by SSH and does not provision.

One incarnation = one ``torchrun`` per node, started over SSH:

1. before the first start, every node's ``uname -m`` must equal its
   ``cpu_arch`` (``Node.check_cpu_arch``) and its device family's card tool
   must match the inventory (``Node.check_cards``); the memory estimate of the
   plan (:mod:`yeto.rl.fullparam.memory`) is checked when the spec is built
   (:func:`ssh_trainer_spec`), so a plan that does not fit never reaches a node;
2. each node runs, in a new session (one process group), the trainer command
   under ``torchrun`` with the incarnation environment
   (``trainer_remote.island_env``), the collective-communication variables of
   its family and fabric (``topology.island_comm_env``) and the run's
   environment. Node 0 also runs the W4A8 re-quantizer side process
   (``yeto.rl.quant.release --loop``) when the spec has one: each BF16 release
   becomes a W4A8 release in the store, which W4A8 samplers follow;
3. the SSH session streams node 0's log to the head (``IslandEvents`` reads the
   trainer events from it); the other nodes' logs go to local files;
4. release kills each node's process group (TERM, then KILL after a grace
   period) by the PID file the start wrote; a restarted head does the same for
   an orphan incarnation (the PID file path is derived from the run id, island
   name and incarnation, recorded in the journal before any start).

The head reads the cuts, the commands and the sample buffer through the store's
object API (``s3://`` or ``oss://``); with a sharded buffer it also counts the
records the trainer has not merged yet (``UnmergedShards``, N3.6).

Not done here: the launcher does not yet hold back an SSH island from ``--gpu``
(the SSH node pool launch path, N1.3); ``runtime: docker`` nodes are refused
(the per-family device flags of the trainer image are N0.2b). Nothing here was
run against a real node.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.rl.engine.start_phases import SUBMITTED_ENV
from yeto.rl.engine.trainer_lifecycle import CutRef
from yeto.rl.engine.trainer_remote import island_env

from .node_inventory import InventoryError, Node
from .topology import island_comm_env
from .trainer_island import (
    JOBS_FILE,
    CloudTrainerBinding,
    IslandEvents,
    TrainerIsland,
    head_store_objects,
    incarnation_tape,
    object_store_binding,
    read_jobs,
    save_job,
)

KILL_GRACE_S = 30
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30")
STATE_DIR = ".yeto/trainer-islands"  # under the remote $HOME


class SshIslandError(RuntimeError):
    """The SSH trainer island cannot start or stop as asked."""


@dataclass(frozen=True)
class SshTrainerSpec:
    """What every incarnation runs.

    ``command``: the trainer entry, as ``torchrun`` arguments after ``--``
    (e.g. ``["-m", "glm_rl.verl_ext.train", ...]``); ``workdir``: the remote
    directory it runs in; ``requant``: argv of the W4A8 re-quantizer side
    process on node 0 (None: BF16 samplers only)."""

    nodes: tuple[Node, ...]
    run_id: str
    command: tuple[str, ...]
    workdir: str
    env: Mapping[str, str] = field(default_factory=dict)
    requant: tuple[str, ...] | None = None
    requant_poll_s: float = 10.0
    master_port: int = 29500
    torchrun: str = "torchrun"
    memory: Mapping[str, Any] | None = None  # the estimate summary the spec was checked with

    def __post_init__(self) -> None:
        if not self.nodes:
            raise SshIslandError("an SSH trainer island needs at least one node")
        if not self.command:
            raise SshIslandError("an SSH trainer island needs a trainer command")
        families = {n.device_family for n in self.nodes}
        if len(families) != 1:
            raise SshIslandError(f"one trainer island spans one device family, got {sorted(families)}")
        if docker := [n.name for n in self.nodes if n.runtime != "rootfs"]:
            raise SshIslandError(f"nodes {docker}: runtime docker is not wired for trainer islands yet "
                                 "(the trainer image's device flags are task N0.2b); use runtime: rootfs")
        if len({n.cards_per_node for n in self.nodes}) != 1:
            raise SshIslandError("trainer nodes must have the same number of cards")

    @property
    def device_family(self) -> str:
        return self.nodes[0].device_family


def ssh_trainer_spec(nodes: Sequence[Node], *, shape: Any, plan_overrides: Mapping[str, Any],
                     **spec: Any) -> SshTrainerSpec:
    """An :class:`SshTrainerSpec` whose memory plan fits the nodes (raises
    ``MemoryPlanError`` before any SSH when it does not)."""
    from yeto.rl.fullparam.memory import check, plan_for_nodes

    estimate = check(shape, plan_for_nodes(list(nodes), **dict(plan_overrides)))
    return SshTrainerSpec(nodes=tuple(nodes), memory=estimate.summary(), **spec)


def master_address(node: Node) -> str:
    """The rendezvous address the other nodes dial: the SSH target without the
    user (an SSH alias must also resolve on the nodes)."""
    return node.host.rpartition("@")[2]


def remote_paths(run_id: str, island: str, incarnation: int) -> dict[str, str]:
    base = f"$HOME/{STATE_DIR}/{run_id}/{island}-i{int(incarnation)}"
    return {"dir": base, "pid": f"{base}/pgid", "log": f"{base}/trainer.log", "rc": f"{base}/rc",
            "requant_log": f"{base}/requant.log"}


def _exports(env: Mapping[str, str]) -> str:
    return "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(env.items()))


def start_script(spec: SshTrainerSpec, rank: int, env: Mapping[str, str], paths: Mapping[str, str]) -> str:
    """The bash script one node runs: start the island in its own session, then
    stream its log until it exits (node 0's stream carries the trainer events)."""
    node = spec.nodes[rank]
    torchrun = [spec.torchrun, f"--nnodes={len(spec.nodes)}", f"--node-rank={rank}",
                f"--nproc-per-node={node.cards_per_node}", f"--master-addr={master_address(spec.nodes[0])}",
                f"--master-port={spec.master_port}", *spec.command]
    if rank == 0 and spec.requant:
        # The loop converts releases while the trainer runs; after the trainer
        # ends, one more pass converts the last BF16 release.
        log = f"\"{paths['requant_log']}\""
        body = (f"{shlex.join(spec.requant)} --loop --poll-seconds {spec.requant_poll_s:g} > {log} 2>&1 &\n"
                "req=$!\n"
                f"{shlex.join(torchrun)} && rc=0 || rc=$?\n"
                "kill $req 2>/dev/null; wait $req 2>/dev/null\n"
                f"{shlex.join(spec.requant)} >> {log} 2>&1 || {{ [ $rc -ne 0 ] || rc=1; }}\n"
                f"echo $rc > \"{paths['rc']}\"\n")
    else:
        body = f"{shlex.join(torchrun)} && rc=0 || rc=$?\necho $rc > \"{paths['rc']}\"\n"
    comm = island_comm_env(node.device_family, node.fabric, node.interface, num_nodes=len(spec.nodes))
    return (
        "set -eu\n"
        f"mkdir -p \"{paths['dir']}\"\n"
        f"cd {shlex.quote(spec.workdir)}\n"
        + _exports({**comm, **spec.env, **env})
        + f"rm -f \"{paths['rc']}\"\n"
        # Job control is off in a non-interactive shell, so the background setsid
        # is not a group leader: it does not fork, and $! is the new group's id.
        # The subshell exits at once, so init reaps the island when it ends (a
        # zombie would keep both ``tail --pid`` and the kill check waiting).
        + f"( setsid bash -c {shlex.quote(body)} > \"{paths['log']}\" 2>&1 < /dev/null &\n"
        + f"  echo $! > \"{paths['pid']}\" )\n"
        + f"exec tail -n +1 -F --pid=\"$(cat \"{paths['pid']}\")\" \"{paths['log']}\" 2>/dev/null\n"
    )


def kill_script(paths: Mapping[str, str], grace_s: int = KILL_GRACE_S) -> str:
    """TERM the island's process group, KILL it after ``grace_s``; exit 0 when it is gone."""
    return (
        f"pid=$(cat \"{paths['pid']}\" 2>/dev/null) || exit 0\n"
        "kill -TERM -- -$pid 2>/dev/null || exit 0\n"
        f"for _ in $(seq {int(grace_s)}); do kill -0 -- -$pid 2>/dev/null || exit 0; sleep 1; done\n"
        "kill -KILL -- -$pid 2>/dev/null || true\n"
        "sleep 1; ! kill -0 -- -$pid 2>/dev/null\n"
    )


class SshOps:
    """The SSH calls (replaced in tests). ``run`` waits; ``start`` returns the
    running ``ssh`` process with its stdout as a text pipe."""

    def command(self, node: Node, script: str) -> list[str]:
        return ["ssh", *SSH_OPTIONS, *node.ssh_options(), node.host, "bash", "-lc", shlex.quote(script)]

    def run(self, node: Node, script: str, *, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.command(node, script), text=True, capture_output=True, timeout=timeout)

    def start(self, node: Node, script: str) -> Any:
        return subprocess.Popen(self.command(node, script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, bufsize=1)


@dataclass
class _SshJob:
    incarnation: int
    paths: dict[str, str]
    procs: list[Any]
    events: IslandEvents
    threads: list[threading.Thread] = field(default_factory=list)


class SshTrainerIslands:
    """``TrainerIslands`` over SSH nodes (module doc)."""

    def __init__(self, spec: SshTrainerSpec, *, name: str, jobs_path: Path, ops: SshOps | None = None,
                 poll_s: float | None = None, events_dir: Path | None = None,
                 clock: Callable[[], float] = time.time, kill_grace_s: int = KILL_GRACE_S) -> None:
        self.spec = spec
        self.name = name
        self.jobs_path = Path(jobs_path)
        self.ops = ops or SshOps()
        self.poll_s = poll_s
        self.events_dir = events_dir
        self.clock = clock
        self.kill_grace_s = int(kill_grace_s)
        self.jobs: list[_SshJob] = []
        self.checked = False

    # -- preflight ----------------------------------------------------------------
    def preflight(self) -> None:
        """Every node's CPU architecture and cards must match the inventory
        (raises before any start)."""
        for node in self.spec.nodes:
            done = self.ops.run(node, "uname -m", timeout=60.0)
            if done.returncode != 0:
                raise SshIslandError(f"node {node.name}: uname -m failed (rc {done.returncode})")
            try:
                node.check_cpu_arch(done.stdout)
            except InventoryError as exc:
                raise SshIslandError(str(exc)) from None
            for argv in node.preflight_commands():
                done = self.ops.run(node, shlex.join(argv), timeout=120.0)
                if done.returncode == 0:
                    node.check_cards(done.stdout)
                    break
            else:
                raise SshIslandError(f"node {node.name}: no card tool of family {node.device_family} "
                                     f"answered ({[shlex.join(a) for a in node.preflight_commands()]})")
        self.checked = True

    # -- TrainerIslands -------------------------------------------------------------
    def launch(self, incarnation: int, resume_from: CutRef | None) -> _SshJob:
        if not self.checked:
            self.preflight()
        paths = remote_paths(self.spec.run_id, self.name, incarnation)
        env = {**island_env(incarnation, resume_from, poll_s=self.poll_s), SUBMITTED_ENV: f"{self.clock():.3f}"}
        save_job(self.jobs_path, incarnation, f"ssh:{self.name}-i{int(incarnation)}")  # before any start
        job = _SshJob(int(incarnation), paths, [],
                      IslandEvents(incarnation_tape(self.events_dir, self.name, incarnation)))
        try:
            for rank, node in enumerate(self.spec.nodes):
                job.procs.append(self.ops.start(node, start_script(self.spec, rank, env, paths)))
        except Exception:
            self.release(job)
            raise
        for rank, proc in enumerate(job.procs):
            sink = job.events.feed if rank == 0 else self._node_log(incarnation, rank)
            thread = threading.Thread(target=_pump, args=(proc, sink), daemon=True,
                                      name=f"{self.name}-i{incarnation}-n{rank}")
            thread.start()
            job.threads.append(thread)
        self.jobs.append(job)
        print(f"[launcher] trainer incarnation {incarnation} on {len(self.spec.nodes)} SSH node(s) "
              f"(resume {None if resume_from is None else resume_from.cut_id})", flush=True)
        return job

    def events(self, job: _SshJob) -> list[Mapping[str, Any]]:
        return job.events.take()

    def alive(self, job: _SshJob) -> bool:
        """Every node's stream is open: one node ending ends the island."""
        return bool(job.procs) and all(p.poll() is None for p in job.procs)

    def release(self, job: _SshJob) -> bool:
        ok = self._kill(job.incarnation)
        for proc in job.procs:
            if proc.poll() is None:
                proc.terminate()
        return ok

    def release_incarnation(self, incarnation: int) -> bool:
        if str(int(incarnation)) not in read_jobs(self.jobs_path):
            print(f"[launcher] trainer incarnation {incarnation}: no SSH start recorded; nothing to release",
                  flush=True)
            return False
        return self._kill(incarnation)

    def start_events(self) -> list[Mapping[str, Any]]:
        return [e for job in self.jobs for e in job.events.all]

    # -- internals ------------------------------------------------------------------
    def _kill(self, incarnation: int) -> bool:
        script = kill_script(remote_paths(self.spec.run_id, self.name, incarnation), self.kill_grace_s)
        ok = True
        for node in self.spec.nodes:
            try:
                done = self.ops.run(node, script, timeout=self.kill_grace_s + 60.0)
                ok &= done.returncode == 0
            except Exception as error:  # noqa: BLE001 - not confirmed: the caller reports it
                print(f"[launcher] stop of trainer incarnation {incarnation} on {node.name} failed: {error}",
                      flush=True)
                ok = False
        return ok

    def _node_log(self, incarnation: int, rank: int) -> Callable[[str], None]:
        if self.events_dir is None:
            return lambda _line: None
        path = Path(self.events_dir) / f"{self.name}-i{int(incarnation)}-node{rank}.log"
        path.parent.mkdir(parents=True, exist_ok=True)

        def write(line: str) -> None:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)

        return write


def _pump(proc: Any, sink: Callable[[str], None]) -> None:
    for line in proc.stdout:
        sink(line)


def ssh_binding(args: Any, island: TrainerIsland, *, journal_dir: Path, files: Any = None,
                islands: Any = None) -> CloudTrainerBinding:
    """The :class:`CloudTrainerBinding` of an SSH trainer island (``island.ssh``:
    its :class:`SshTrainerSpec`; ``files`` / ``islands`` replace SSH and the store
    in tests). The run's checkpoint store must be one the head reads by object key."""
    if files is None:
        store = str(getattr(args, "rl_checkpoint_store", None) or "")
        if not store.startswith(("s3://", "oss://")):
            raise SshIslandError(f"an SSH trainer island needs an s3:// or oss:// --rl-checkpoint-store the "
                                 f"head reads (got {store or 'none'!r})")
        files = head_store_objects(store)
    if islands is None:
        spec = getattr(island, "ssh", None)
        if not isinstance(spec, SshTrainerSpec):
            raise SshIslandError(f"trainer island {island.name}: no SSH spec")
        islands = SshTrainerIslands(spec, name=island.name, jobs_path=Path(journal_dir) / JOBS_FILE,
                                    poll_s=getattr(args, "rl_trainer_command_poll_seconds", None),
                                    events_dir=island.events_dir)
    return object_store_binding(args, files, islands, journal_dir=journal_dir)


def requant_argv(store: str, template: str, *, recipe: str | None = None, awq_scales: str | None = None,
                 python: str = "python") -> tuple[str, ...]:
    """The one-pass argv of :mod:`yeto.rl.quant.release` (R2 with ``recipe`` and
    ``awq_scales``; R1 without). The start script adds ``--loop`` for the side process."""
    argv = [python, "-m", "yeto.rl.quant.release", "--store", store, "--template", template]
    if recipe or awq_scales:
        if not (recipe and awq_scales):
            raise SshIslandError("give both recipe and awq_scales (R2), or neither (R1)")
        argv += ["--recipe", recipe, "--awq-scales", awq_scales]
    else:
        argv.append("--no-awq")
    return tuple(argv)
