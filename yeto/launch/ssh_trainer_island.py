"""Trainer island on SSH nodes for the trainer scheduler.

The trainer scheduler (``trainer_loop``) starts and stops the trainer island
through the ``TrainerIslands`` port (``launch`` / ``events`` / ``alive`` /
``release`` / ``release_incarnation`` / ``start_events``). Modal and sky
implement it in :mod:`.trainer_island`; :class:`SshTrainerIslands` implements
it over nodes of the SSH node inventory (:mod:`.node_inventory`), e.g. PPU or
Ascend machines that yeto reaches by SSH and does not provision.

One incarnation = one trainer process per node, started over SSH:

1. before the first start, every node's ``uname -m`` must equal its
   ``cpu_arch`` (``Node.check_cpu_arch``) and its device family's card tool
   must match the inventory (``Node.check_cards``); the memory estimate of the
   plan (:mod:`yeto.rl.fullparam.memory`) is checked when the spec is built
   (:func:`ssh_trainer_spec`), so a plan that does not fit never reaches a node;
2. each node runs, in a new session (one process group), the trainer with the
   incarnation environment (``trainer_remote.island_env``), the
   collective-communication variables of its family and fabric
   (``topology.island_comm_env``), the run's environment and the inventory's
   device family (``YETO_DEVICE_FAMILY``: the telemetry's card collector and the
   learner's device follow it). The trainer is
   either ``command`` under ``torchrun`` or ``run_script``, a bash script that
   reads the sky task variables (``SKYPILOT_NODE_RANK``, ``SKYPILOT_NODE_IPS``,
   ``SKYPILOT_NUM_NODES``, ``SKYPILOT_NUM_GPUS_PER_NODE``), so the run script
   of a sky island (e.g. Ray head on rank 0, Ray workers on the others) runs
   unchanged. Node 0 also runs the re-quantizer side process
   (``yeto.rl.quant.release --loop``) when the spec has one: each BF16 release
   becomes a release of the template's scheme (W4A8 or MXFP4) in that scheme's
   directory of the store, which samplers of that scheme follow;
   on ``runtime: docker`` nodes the trainer runs in a container
   (``docker run --network host``) of the spec's image with the node's mounts
   and the device flags of its family (``docker_args``; a docker island
   without them is refused: the flags are not known for every family yet; a
   cuda node must pass its GPUs and, on a multi-node ib/roce island, the verbs
   devices and an unlimited memlock: :func:`cuda_docker_problems`). The
   preflight pulls the image on every docker node;
3. the SSH session streams node 0's log to the head (``IslandEvents`` reads the
   trainer events from it); the other nodes' logs go to local files;
4. release kills each node's process group (TERM, then KILL after a grace
   period) by the PID file the start wrote; a restarted head does the same for
   an orphan incarnation (the PID file path is derived from the run id, island
   name and incarnation, recorded in the journal before any start);
5. a later start first waits (up to ``node_wait_s``) until every node answers
   over SSH again: a node that lost power or its network is not given a new
   incarnation that can only fail (measured on the simulated cluster,
   tests/integration/bm_sim: such a start ran 78 s and ended "without a stopped
   event", naming no node). It then stops again every incarnation whose release
   was not confirmed on every node (a node that was unreachable then may have
   kept it running), and refuses to start over one it still cannot stop.

Rendezvous: the other nodes dial node 0 at its ``address`` from the inventory
(the IP on the cluster network); an SSH alias resolves on the head only. A
node without an ``address`` is accepted only when its SSH target is an IP.

The head reads the cuts, the commands and the sample buffer through the store's
object API (``s3://`` or ``oss://``); with a sharded buffer it also counts the
records the trainer has not merged yet (``UnmergedShards``).

The launcher holds back an ``ssh:`` island of ``--gpu`` as the trainer island
and builds its spec from ``--rl-node-inventory`` (:func:`island_spec_from_inventory`).
Nothing here was run against a real node.
"""

from __future__ import annotations

import ipaddress
import re
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

from ..gpu_spec import DEVICE_FAMILY_ENV
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
NODE_WAIT_S = 600.0  # a later start waits this long for an unreachable node (a reboot)
NODE_POLL_S = 15.0
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30")
# The remote command: the script comes on stdin. stdin moves to fd 3 and the login
# shell starts on /dev/null, so a profile (/etc/profile.d, ~/.bash_profile) that reads
# stdin cannot eat the script; the script then runs without fd 3.
READ_SCRIPT = "exec 3<&0 </dev/null; exec bash -lc 'eval \"$(cat <&3)\" 3<&-'"
STATE_DIR = ".yeto/trainer-islands"  # under the remote $HOME
CONTAINER_STATE_DIR = "/yeto-island-state"  # the incarnation's state dir inside a docker trainer
PULL_TIMEOUT_S = 3600.0  # a first pull of a trainer image (~24 GB compressed for yeto-miles-k31)
RDMA_FABRICS = ("ib", "roce")
# `docker run` flags (as NAME=VALUE) that give a cuda container its GPUs: the
# NVIDIA runtime by --gpus, --runtime nvidia, or a CDI device.
CUDA_GPU_FLAGS = ("--gpus=", "--runtime=nvidia", "--device=nvidia.com/gpu")
# What a cuda island on an RDMA fabric needs on top (multi-node): the verbs devices,
# without which NCCL finds no IB device and silently runs its collectives over TCP
# sockets; and an unlimited locked-memory limit for NCCL's ibv_reg_mr (a container
# inherits dockerd's limit, often 64 KiB, which --privileged does not lift).
RDMA_DEVICE_FLAGS = ("--privileged", "--device=/dev/infiniband", "--volume=/dev/infiniband")
MEMLOCK_FLAGS = ("--ulimit=memlock=-1", "--ulimit=memlock=unlimited")
CUDA_RDMA_DOCKER_ARGS = ("--gpus", "all", "--device", "/dev/infiniband", "--ulimit", "memlock=-1:-1",
                         "--ulimit", "stack=67108864", "--cap-add", "IPC_LOCK")


class SshIslandError(RuntimeError):
    """The SSH trainer island cannot start or stop as asked."""


@dataclass(frozen=True)
class SshTrainerSpec:
    """What every incarnation runs.

    ``command``: the trainer entry, as ``torchrun`` arguments after ``--``
    (e.g. ``["-m", "glm_rl.verl_ext.train", ...]``); or ``run_script``: a bash
    script run on every node with the sky task variables (module doc); exactly
    one of the two. ``workdir``: the directory it runs in (inside the container
    on docker nodes); ``requant``: argv of the re-quantizer side process on
    node 0 (None: BF16 samplers only). ``image`` / ``docker_args`` replace the
    nodes' inventory values on docker nodes (e.g. a trainer image)."""

    nodes: tuple[Node, ...]
    run_id: str
    workdir: str
    command: tuple[str, ...] = ()
    run_script: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    requant: tuple[str, ...] | None = None
    requant_poll_s: float = 10.0
    master_port: int = 29500
    torchrun: str = "torchrun"
    image: str | None = None
    docker_args: tuple[str, ...] | None = None
    memory: Mapping[str, Any] | None = None  # the estimate summary the spec was checked with

    def __post_init__(self) -> None:
        if not self.nodes:
            raise SshIslandError("an SSH trainer island needs at least one node")
        if bool(self.command) == bool(self.run_script):
            raise SshIslandError("an SSH trainer island needs one trainer command or one run script")
        families = {n.device_family for n in self.nodes}
        if len(families) != 1:
            raise SshIslandError(f"one trainer island spans one device family, got {sorted(families)}")
        if len({n.cards_per_node for n in self.nodes}) != 1:
            raise SshIslandError("trainer nodes must have the same number of cards")
        if len({n.runtime for n in self.nodes}) != 1:
            raise SshIslandError("trainer nodes must have the same runtime (docker or rootfs)")
        for node in self.nodes:
            if node.runtime != "docker":
                continue
            if not (self.image or node.image):
                raise SshIslandError(f"node {node.name}: runtime docker needs an image")
            if not self.docker_args_of(node):
                raise SshIslandError(
                    f"node {node.name}: runtime docker needs docker_args, the device flags of `docker run` "
                    f"for device family {node.device_family} (inventory docker_args)")
            if node.device_family == "cuda":
                missing = cuda_docker_problems(self.docker_args_of(node),
                                               rdma=len(self.nodes) > 1 and node.fabric in RDMA_FABRICS)
                if missing:
                    raise SshIslandError(
                        f"node {node.name}: docker_args lack {'; '.join(missing)} (fabric {node.fabric}); "
                        f"e.g. docker_args: {list(CUDA_RDMA_DOCKER_ARGS)}")
        if len(self.nodes) > 1:
            for node in self.nodes:
                rendezvous_address(node)  # raises on a node the others cannot dial
        if self.env.get(DEVICE_FAMILY_ENV, self.device_family) != self.device_family:
            raise SshIslandError(f"env sets {DEVICE_FAMILY_ENV}={self.env[DEVICE_FAMILY_ENV]}, but the "
                                 f"inventory nodes are {self.device_family}; the inventory decides")

    @property
    def device_family(self) -> str:
        return self.nodes[0].device_family

    @property
    def docker(self) -> bool:
        return self.nodes[0].runtime == "docker"

    def docker_args_of(self, node: Node) -> tuple[str, ...]:
        return tuple(self.docker_args if self.docker_args is not None else node.docker_args)

    def addresses(self) -> list[str]:
        """Rendezvous address of every node (one node: loopback when it has none)."""
        if len(self.nodes) == 1 and self.nodes[0].address is None and not _ip_host(self.nodes[0]):
            return ["127.0.0.1"]
        return [rendezvous_address(n) for n in self.nodes]


def _flag_forms(argv: Sequence[str]) -> list[str]:
    """``docker run`` options as ``NAME=VALUE`` (``-v`` as the long volume option), so the
    spellings ``--gpus all`` and ``--gpus=all`` compare alike."""
    short = {"-v": "--volume"}
    out: list[str] = []
    i = 0
    while i < len(argv):
        flag = short.get(argv[i], argv[i])
        if flag.startswith("--") and "=" not in flag and i + 1 < len(argv) and not argv[i + 1].startswith("-") \
                and flag not in ("--privileged", "--init", "--rm"):
            out.append(f"{flag}={argv[i + 1]}")
            i += 2
            continue
        out.append(flag)
        i += 1
    return out


def cuda_docker_problems(docker_args: Sequence[str], *, rdma: bool) -> list[str]:
    """What the ``docker run`` flags of a cuda trainer node miss: the GPUs, and on a
    multi-node RDMA island the verbs devices and the locked-memory limit."""
    flags = _flag_forms(docker_args)
    has = lambda prefixes: any(f.startswith(prefixes) for f in flags)
    problems = []
    if not has(CUDA_GPU_FLAGS):
        problems.append("the GPUs (--gpus all, --runtime nvidia or a CDI --device nvidia.com/gpu=...)")
    if rdma and not has(RDMA_DEVICE_FLAGS):
        problems.append("the RDMA devices (--device /dev/infiniband; without them NCCL falls back to TCP sockets)")
    if rdma and not has(MEMLOCK_FLAGS):
        problems.append("an unlimited locked-memory limit (--ulimit memlock=-1:-1)")
    return problems


def ssh_trainer_spec(nodes: Sequence[Node], *, shape: Any, plan_overrides: Mapping[str, Any],
                     **spec: Any) -> SshTrainerSpec:
    """An :class:`SshTrainerSpec` whose memory plan fits the nodes (raises
    ``MemoryPlanError`` before any SSH when it does not)."""
    from yeto.rl.fullparam.memory import check, plan_for_nodes

    estimate = check(shape, plan_for_nodes(list(nodes), **dict(plan_overrides)))
    return SshTrainerSpec(nodes=tuple(nodes), memory=estimate.summary(), **spec)


def _ip_host(node: Node) -> str | None:
    host = node.host.rpartition("@")[2]
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return None


def rendezvous_address(node: Node) -> str:
    """The address the other nodes dial: the inventory ``address``, else the
    SSH target when it is an IP. An SSH alias resolves on the head only."""
    address = node.address or _ip_host(node)
    if address is None:
        raise SshIslandError(f"node {node.name}: no rendezvous address; set `address` (its IP on the "
                             f"cluster network) in the inventory (SSH target {node.host!r} is not an IP)")
    return address


def remote_paths(run_id: str, island: str, incarnation: int) -> dict[str, str]:
    base = f"$HOME/{STATE_DIR}/{run_id}/{island}-i{int(incarnation)}"
    return {"dir": base, "pid": f"{base}/pgid", "log": f"{base}/trainer.log", "rc": f"{base}/rc",
            "requant_log": f"{base}/requant.log"}


def container_name(run_id: str, island: str, incarnation: int, rank: int) -> str:
    """The docker container of one node of one incarnation (the kill script removes it)."""
    raw = f"yeto-{run_id}-{island}-i{int(incarnation)}-n{int(rank)}"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", raw)


# The login HOME, pinned before the island's exports: ``paths`` hold ``$HOME``,
# and a run whose environment sets ``HOME`` must not move the state dir away from
# where the kill script (a fresh ssh shell, no exports) looks for the pid file.
STATE_HOME_VAR = "__yeto_state_home"


def _pin_state_home(paths: Mapping[str, str]) -> dict[str, str]:
    return {k: str(v).replace("$HOME", "$" + STATE_HOME_VAR) for k, v in paths.items()}


def _exports(env: Mapping[str, str]) -> str:
    return "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(env.items()))


def skypilot_env(spec: SshTrainerSpec, rank: int) -> dict[str, str]:
    """The sky task variables a run script reads (as ``modal_runner.skypilot_env``)."""
    addresses = spec.addresses()
    return {"SKYPILOT_NODE_IPS": "\n".join(addresses), "SKYPILOT_NUM_NODES": str(len(addresses)),
            "SKYPILOT_NODE_RANK": str(int(rank)),
            "SKYPILOT_NUM_GPUS_PER_NODE": str(spec.nodes[rank].cards_per_node)}


def _trainer_body(spec: SshTrainerSpec, rank: int, requant_log: str) -> str:
    """The node's job as bash; it exits with the trainer's exit code."""
    node = spec.nodes[rank]
    if spec.run_script:
        # The island run script (rl_island_task) enters ~/sky_workdir, where sky syncs
        # the code; an SSH node has no sync, its checkout is the spec's workdir.
        link = f'[ -e "$HOME/sky_workdir" ] || ln -s {shlex.quote(spec.workdir)} "$HOME/sky_workdir"\n'
        trainer = f"{link}(\n{spec.run_script.rstrip()}\n)"  # a subshell: the script's exit and exec stay inside
    else:
        trainer = shlex.join([spec.torchrun, f"--nnodes={len(spec.nodes)}", f"--node-rank={rank}",
                              f"--nproc-per-node={node.cards_per_node}",
                              f"--master-addr={spec.addresses()[0]}",
                              f"--master-port={spec.master_port}", *spec.command])
    if rank == 0 and spec.requant:
        # The loop converts releases while the trainer runs; after the trainer
        # ends, one more pass converts the last BF16 release.
        return (f"{shlex.join(spec.requant)} --loop --poll-seconds {spec.requant_poll_s:g} > {requant_log} 2>&1 &\n"
                "req=$!\n"
                f"{trainer} && rc=0 || rc=$?\n"
                "kill $req 2>/dev/null; wait $req 2>/dev/null\n"
                f"{shlex.join(spec.requant)} >> {requant_log} 2>&1 || {{ [ $rc -ne 0 ] || rc=1; }}\n"
                "exit $rc\n")
    return f"{trainer}\n"


def docker_argv(spec: SshTrainerSpec, rank: int, name: str, env_names: Sequence[str]) -> list[str]:
    """``docker run`` of one node, the image last; the environment goes by name
    (``-e KEY``), so no value appears in the process list."""
    node = spec.nodes[rank]
    argv = ["docker", "run", "--rm", "--name", name, "--network", "host", "--ipc", "host",
            *spec.docker_args_of(node)]
    for host_path, container_path in sorted(node.mounts.items()):
        argv += ["-v", f"{host_path}:{container_path}"]
    for key in sorted(env_names):
        argv += ["-e", key]
    return [*argv, "-w", spec.workdir, str(spec.image or node.image)]


def start_script(spec: SshTrainerSpec, rank: int, env: Mapping[str, str], paths: Mapping[str, str],
                 *, container: str | None = None) -> str:
    """The bash script one node runs: start the island in its own session, then
    stream its log until it exits (node 0's stream carries the trainer events).
    ``container``: the docker container name (docker nodes)."""
    node = spec.nodes[rank]
    paths = _pin_state_home(paths)
    comm = island_comm_env(node.device_family, node.fabric, node.interface, num_nodes=len(spec.nodes))
    # The learner reads the family to pick its card type and collective backend (accel).
    # The inventory's fabric settings (e.g. NCCL_IB_HCA) refine the family defaults.
    full_env = {DEVICE_FAMILY_ENV: node.device_family, **comm, **node.env, **spec.env, **env,
                **(skypilot_env(spec, rank) if spec.run_script else {})}
    if spec.docker:
        if container is None:
            raise SshIslandError("a docker node needs its container name")
        # The state dir is mounted so the re-quantizer log lands next to the trainer log.
        inner = _trainer_body(spec, rank, f"{CONTAINER_STATE_DIR}/requant.log")
        *options, image = docker_argv(spec, rank, container, list(full_env))
        mount = f"-v \"{paths['dir']}:{CONTAINER_STATE_DIR}\""  # $HOME expands on the node
        job = f"{shlex.join(options)} {mount} {shlex.quote(image)} bash -c {shlex.quote(inner)}"
        enter = ""
    else:
        requant_log = f"\"{paths['requant_log']}\""
        job = f"bash -c {shlex.quote(_trainer_body(spec, rank, requant_log))}"
        enter = f"cd {shlex.quote(spec.workdir)}\n"
    body = f"{job} && rc=0 || rc=$?\necho $rc > \"{paths['rc']}\"\n"
    return (
        "set -eu\n"
        # The state dir is pinned to the login HOME before the exports: ``paths``
        # are written with ``$HOME``, and a run whose environment sets ``HOME`` (a
        # shared island home on an on-prem node) would otherwise split the state
        # dir, the log and the pid file across two homes (measured 2026-11-14 on a
        # two-node Ascend island: every node died at once with "trainer-i1/pgid:
        # No such file or directory"), and the kill script, which runs without the
        # exports, would not find the pid file. The job still gets the run's HOME.
        # Exported: the rc and requant-log paths expand inside the job's own bash.
        + f"export {STATE_HOME_VAR}=\"$HOME\"\n"
        + _exports(full_env)
        + f"mkdir -p \"{paths['dir']}\"\n"
        + enter
        + f"rm -f \"{paths['rc']}\"\n"
        # Job control is off in a non-interactive shell, so the background setsid
        # is not a group leader: it does not fork, and $! is the new group's id.
        # The subshell exits at once, so init reaps the island when it ends (a
        # zombie would keep both ``tail --pid`` and the kill check waiting).
        + f"( setsid bash -c {shlex.quote(body)} > \"{paths['log']}\" 2>&1 < /dev/null &\n"
        + f"  echo $! > \"{paths['pid']}\" )\n"
        + f"exec tail -n +1 -F --pid=\"$(cat \"{paths['pid']}\")\" \"{paths['log']}\" 2>/dev/null\n"
    )


def kill_script(paths: Mapping[str, str], grace_s: int = KILL_GRACE_S, *, container: str | None = None) -> str:
    """TERM the island's process group, KILL it after ``grace_s``; exit 0 when it
    is gone. ``container``: remove that docker container first (a killed
    ``docker run`` client does not stop its container).

    The pid file is read from the login HOME, where :func:`start_script` pins the
    state dir. No pid file: a warning, and exit 0 only when the state dir does not
    exist either (this incarnation never started on the node); a state dir with no
    pid file exits 3, so a release cannot report ok while the island may still run."""
    remove = f"docker rm -f {shlex.quote(container)} >/dev/null 2>&1 || true\n" if container else ""
    paths = _pin_state_home(paths)
    return (
        remove
        + f"{STATE_HOME_VAR}=\"$HOME\"\n"
        + f"if ! pid=$(cat \"{paths['pid']}\" 2>/dev/null) || [ -z \"$pid\" ]; then\n"
        + f"  if [ -d \"{paths['dir']}\" ]; then\n"
        + f"    echo \"[yeto] kill: no pid in {paths['pid']} but the state dir exists; the island may "
          "still run\" >&2; exit 3\n"
        + "  fi\n"
        # never started on this node (a first incarnation the run stopped early):
        # nothing to stop, and not worth a line in the launcher log
        + "  exit 0\n"
        + "fi\n"
        + "kill -TERM -- -$pid 2>/dev/null || exit 0\n"
        + f"for _ in $(seq {int(grace_s)}); do kill -0 -- -$pid 2>/dev/null || exit 0; sleep 1; done\n"
        + "kill -KILL -- -$pid 2>/dev/null || true\n"
        + "sleep 1; ! kill -0 -- -$pid 2>/dev/null\n"
    )


class SshOps:
    """The SSH calls (replaced in tests). ``run`` waits; ``start`` returns the
    running ``ssh`` process with its stdout as a text pipe."""

    def command(self, node: Node) -> list[str]:
        """``ssh`` to ``node`` running the script it reads from stdin: a start script
        exports the task's secrets, which must not be in an argv (``ps`` on either
        end, a ``TimeoutExpired`` message). The remote bash reads the whole script
        before it runs, so nothing in it (or in a login profile) can consume the rest
        of it."""
        return ["ssh", *SSH_OPTIONS, *node.ssh_options(), node.host, "bash", "-c", shlex.quote(READ_SCRIPT)]

    def run(self, node: Node, script: str, *, timeout: float = 300.0) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.command(node), input=script, text=True, capture_output=True, timeout=timeout)

    def start(self, node: Node, script: str) -> Any:
        proc = subprocess.Popen(self.command(node), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.PIPE, text=True, bufsize=1)
        try:
            proc.stdin.write(script)
            proc.stdin.close()
        except BrokenPipeError:  # ssh already ended: its stream and exit code tell why
            pass
        return proc


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
                 clock: Callable[[], float] = time.time, kill_grace_s: int = KILL_GRACE_S,
                 node_wait_s: float = NODE_WAIT_S, node_poll_s: float = NODE_POLL_S,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.spec = spec
        self.name = name
        self.jobs_path = Path(jobs_path)
        self.ops = ops or SshOps()
        self.poll_s = poll_s
        self.events_dir = events_dir
        self.clock = clock
        self.kill_grace_s = int(kill_grace_s)
        self.node_wait_s = float(node_wait_s)
        self.node_poll_s = float(node_poll_s)
        self.sleep = sleep
        self.jobs: list[_SshJob] = []
        self.checked = False
        self.unreleased: set[int] = set()  # incarnations not confirmed stopped on every node

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
            if self.spec.docker:
                self._pull(node)
        self.checked = True

    def _pull(self, node: Node) -> None:
        """Pull the trainer image before the first start: a first ``docker run`` of a
        24 GB image would spend the start timeout, and a layer the node's docker
        cannot unpack shows here, not as a dead incarnation."""
        image = str(self.spec.image or node.image)
        done = self.ops.run(node, f"docker pull -q {shlex.quote(image)}", timeout=PULL_TIMEOUT_S)
        if done.returncode != 0:
            err = ((getattr(done, "stderr", "") or "") + (getattr(done, "stdout", "") or "")).strip()[-600:]
            hint = (" (a layer that whites out a path and writes it again: rebuild it with "
                    "docker/miles-ports/make_layer.py of this repository)" if "not a directory" in err else "")
            raise SshIslandError(f"node {node.name}: docker pull {image} failed (rc {done.returncode}): {err}{hint}")

    # -- TrainerIslands -------------------------------------------------------------
    def launch(self, incarnation: int, resume_from: CutRef | None) -> _SshJob:
        if not self.checked:
            self.preflight()
        else:
            self._wait_reachable(incarnation)
        self._stop_unreleased(incarnation)
        paths = remote_paths(self.spec.run_id, self.name, incarnation)
        env = {**island_env(incarnation, resume_from, poll_s=self.poll_s), SUBMITTED_ENV: f"{self.clock():.3f}"}
        save_job(self.jobs_path, incarnation, f"ssh:{self.name}-i{int(incarnation)}")  # before any start
        job = _SshJob(int(incarnation), paths, [],
                      IslandEvents(incarnation_tape(self.events_dir, self.name, incarnation)))
        try:
            for rank, node in enumerate(self.spec.nodes):
                job.procs.append(self.ops.start(node, start_script(
                    self.spec, rank, env, paths, container=self._container(incarnation, rank))))
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
    def _container(self, incarnation: int, rank: int) -> str | None:
        return container_name(self.spec.run_id, self.name, incarnation, rank) if self.spec.docker else None

    def _unreachable(self) -> dict[str, str]:
        """Node name -> why it does not answer over SSH (nodes that answer are left out)."""
        down: dict[str, str] = {}
        for node in self.spec.nodes:
            try:
                done = self.ops.run(node, "true", timeout=60.0)
            except Exception as error:  # noqa: BLE001 - a timeout is an unreachable node
                down[node.name] = repr(error)[:300]
                continue
            if done.returncode != 0:
                err = (getattr(done, "stderr", "") or "").strip()
                down[node.name] = f"rc {done.returncode}" + (f": {err[-300:]}" if err else "")
        return down

    def _wait_reachable(self, incarnation: int) -> None:
        """Before a later start: every node answers over SSH, else wait (up to
        ``node_wait_s``) and raise naming the nodes that still do not."""
        deadline = self.clock() + self.node_wait_s
        while True:
            down = self._unreachable()
            if not down:
                return
            detail = "; ".join(f"{name}: {why}" for name, why in sorted(down.items()))
            if self.clock() >= deadline:
                raise SshIslandError(f"trainer incarnation {incarnation}: node(s) {sorted(down)} unreachable over "
                                     f"SSH for {self.node_wait_s:.0f} s ({detail})")
            print(f"[launcher] trainer incarnation {incarnation}: waiting for node(s) {sorted(down)}, "
                  f"unreachable over SSH ({detail})", flush=True)
            self.sleep(self.node_poll_s)

    def _stop_unreleased(self, incarnation: int) -> None:
        """Stop again the incarnations whose release a node did not confirm."""
        for earlier in sorted(self.unreleased):
            if not self._kill(earlier):
                raise SshIslandError(f"trainer incarnation {incarnation}: incarnation {earlier} is not confirmed "
                                     "stopped on every node; not starting another over it")
            print(f"[launcher] trainer incarnation {earlier}: stopped on every node now", flush=True)

    def _kill(self, incarnation: int) -> bool:
        paths = remote_paths(self.spec.run_id, self.name, incarnation)
        ok = True
        for rank, node in enumerate(self.spec.nodes):
            script = kill_script(paths, self.kill_grace_s, container=self._container(incarnation, rank))
            try:
                done = self.ops.run(node, script, timeout=self.kill_grace_s + 60.0)
                ok &= done.returncode == 0
                err = (getattr(done, "stderr", "") or "").strip()
                if done.returncode != 0:  # rc 0 + stderr is ssh banner noise
                    print(f"[launcher] stop of trainer incarnation {incarnation} on {node.name}: "
                          f"rc={done.returncode} {err}", flush=True)
            except Exception as error:  # noqa: BLE001 - not confirmed: the caller reports it
                print(f"[launcher] stop of trainer incarnation {incarnation} on {node.name} failed: {error}",
                      flush=True)
                ok = False
        if ok:
            self.unreleased.discard(int(incarnation))
        else:
            self.unreleased.add(int(incarnation))
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
                 python: str = "python", frozen_base: str | None = None,
                 frozen_pattern: str | None = None) -> tuple[str, ...]:
    """The one-pass argv of :mod:`yeto.rl.quant.release` (R2 with ``recipe`` and
    ``awq_scales``; R1 without; ``frozen_base``/``frozen_pattern``: untrained
    tensors the release omits). The start script adds ``--loop`` for the side process."""
    argv = [python, "-m", "yeto.rl.quant.release", "--store", store, "--template", template]
    if bool(frozen_base) != bool(frozen_pattern):
        raise SshIslandError("give both frozen_base and frozen_pattern, or neither")
    if frozen_base:
        argv += ["--frozen-base", frozen_base, "--frozen-pattern", frozen_pattern]
    if recipe or awq_scales:
        if not (recipe and awq_scales):
            raise SshIslandError("give both recipe and awq_scales (R2), or neither (R1)")
        argv += ["--recipe", recipe, "--awq-scales", awq_scales]
    else:
        argv.append("--no-awq")
    return tuple(argv)


def inventory_trainer_nodes(args: Any, island: Any) -> tuple[Node, ...]:
    """The inventory nodes of an ``ssh:`` trainer island (``island``: its
    ``gpu_spec.ClusterSpec``): the nodes with role ``trainer`` (or no roles),
    as many as the island has, with its card and card count. Raises before any SSH."""
    nodes = getattr(args, "rl_node_inventory_nodes", None)
    if not nodes:
        raise SshIslandError("an ssh: trainer island needs --rl-node-inventory (the nodes to start it on)")
    from .node_inventory import island_nodes

    trainers = island_nodes(list(nodes), "trainer")
    if len(trainers) != island.num_nodes:
        raise SshIslandError(f"the ssh: trainer island has {island.num_nodes} node(s); the inventory lists "
                             f"{len(trainers)} trainer node(s) {[n.name for n in trainers]}")
    wrong = [n.name for n in trainers
             if n.cards_per_node != island.gpus_per_node or n.card.lower() != str(island.gpu).lower()]
    if wrong:
        raise SshIslandError(f"inventory trainer nodes {wrong} do not have {island.gpus_per_node} x {island.gpu} "
                             "as the ssh: island says")
    return tuple(trainers)


def island_spec_from_inventory(args: Any, island: Any, task: Any) -> SshTrainerSpec:
    """The :class:`SshTrainerSpec` of the held-back ``ssh:`` trainer island: the
    inventory's trainer nodes run the island's sky task (setup, then run) as
    ``run_script`` with the task's environment, in ``--rl-ssh-trainer-workdir``
    (the code pack must be there already: SSH nodes get no sky workdir sync)."""
    workdir = getattr(args, "rl_ssh_trainer_workdir", None)
    if not workdir:
        raise SshIslandError("an ssh: trainer island needs --rl-ssh-trainer-workdir (the yeto checkout on "
                             "the nodes; SSH nodes get no sky workdir sync)")
    run = getattr(task, "run", None)
    if not isinstance(run, str) or not run.strip():
        raise SshIslandError("the ssh: trainer island's task has no run script")
    setup = getattr(task, "setup", None) or ""
    env = {**(getattr(task, "envs", None) or {}), **(getattr(task, "secrets", None) or {})}
    return SshTrainerSpec(nodes=inventory_trainer_nodes(args, island), run_id=str(args.cluster_prefix),
                          workdir=str(workdir),
                          # sky runs setup and run as two scripts; a failed setup ends the job
                          run_script=f"bash -c {shlex.quote(setup)} || exit $?\n{run}" if setup.strip() else run,
                          env={k: str(v) for k, v in env.items() if v is not None})
